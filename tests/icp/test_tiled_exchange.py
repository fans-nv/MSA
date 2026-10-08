"""K5T (tiled symmetric push + merge) against the all_to_all + K2 path.

Oracles (the tests may use ATen; the kernel path may not):

* CPU torch: ``pack_send_carrier`` -> emulated ``all_to_all_single`` (the
  source-major stack of every rank's slab for this destination) ->
  the torch reference merge body (``merge_reference._torch_merge``).
* GPU K2: the same carrier -> ``merge_candidates`` (the CUDA K2 kernel), i.e.
  the route K5T replaces.

K5T runs either as a single-GPU emulation of ``W`` ranks (plain device buffers
as the "peer" windows, one stream per rank, launched concurrently so the ranks
really wait on each other's tags) or, under torchrun, on real CUDA-backend
symmetric memory against the real NCCL ``all_to_all``.

Negative controls must fire: a wrong ``head_offset`` is refused; a misrouted
oracle differs; an int32-poisoned slot (never NaN: ``-1`` as fp32 bits is a
NaN pattern) surfaces when the publish is removed; a stale tag times out into
the transport status bit and kills the workspace.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fmha_sm100.icp import carrier, forced_rows  # noqa: E402
from fmha_sm100.icp.merge_reference import _torch_merge  # noqa: E402

CAND_K = 16
POISON_SCORE = 3.0e38  # finite, must win
POISON_ID = 0x00BADBAD
SHORT_SPIN = 2_000_000
LONG_SPIN = 2_000_000_000


def _bits(value: float) -> int:
    return int(torch.tensor([value], dtype=torch.float32).view(torch.int32))


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def make_rank_candidates(
    T: int, Hg: int, rank: int, *, seed: int, domain: int = 64
) -> torch.Tensor:
    """Query-major int32 ``[T, Hg, 16, 2]`` records with the hard cases.

    A small shared id domain forces cross-rank duplicates; rows also carry
    exact score ties, -0.0/+0.0, +/-inf, invalid (-inf, -1) tails and ids
    above 2**24 (which a float conversion would corrupt).
    """
    g = torch.Generator().manual_seed(seed * 7919 + rank)
    scores = torch.randn((T, Hg, CAND_K), generator=g)
    scores = torch.round(scores * 4) / 4  # plenty of exact ties
    pool = torch.rand((T, Hg, domain), generator=g).argsort(dim=-1)
    ids = pool[..., :CAND_K].to(torch.int32)
    big = torch.rand((T, Hg, CAND_K), generator=g) < 0.05
    ids = torch.where(big, ids + (1 << 30), ids)
    special = torch.rand((T, Hg, CAND_K), generator=g)
    scores = torch.where(special < 0.03, torch.tensor(float("inf")), scores)
    scores = torch.where(
        (special >= 0.03) & (special < 0.06), torch.tensor(float("-inf")), scores
    )
    scores = torch.where(
        (special >= 0.06) & (special < 0.09), torch.tensor(-0.0), scores
    )
    tail = torch.rand((T, Hg, 1), generator=g) * CAND_K
    invalid = torch.arange(CAND_K).view(1, 1, CAND_K) >= tail.ceil()
    scores = torch.where(invalid, torch.tensor(float("-inf")), scores)
    ids = torch.where(invalid, torch.tensor(-1, dtype=torch.int32), ids)
    cand = torch.empty((T, Hg, CAND_K, 2), dtype=torch.int32)
    cand[..., 0] = scores.contiguous().view(torch.int32)
    cand[..., 1] = ids
    return cand


def make_planes(T: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """C3 planes with inactive, f=0, small-f and saturated rows."""
    g = torch.Generator().manual_seed(seed)
    positions = torch.randint(0, 1 << 20, (T,), generator=g, dtype=torch.int64)
    positions[: min(T, 3)] = torch.tensor([0, 130, 2000])[: min(T, 3)]
    active = torch.rand((T,), generator=g) > 0.1
    return forced_rows(positions, active)


# --------------------------------------------------------------------------
# oracles
# --------------------------------------------------------------------------


def recv_carrier(cands: list[torch.Tensor], dest: int) -> torch.Tensor:
    """What ``all_to_all_single`` delivers to ``dest``: source-major slabs."""
    world = len(cands)
    return torch.stack(
        [carrier.pack_send_carrier(c, world)[dest] for c in cands]
    ).contiguous()


def oracle_k2(cands, dest, forced, n_ord):
    from fmha_sm100.icp import merge_candidates

    world = len(cands)
    recv = recv_carrier(cands, dest)
    status = torch.zeros(1, dtype=torch.int32, device=recv.device)
    out = merge_candidates(
        recv, world=world, rank=dest, forced=forced, n_ordinary=n_ord, status=status
    )
    return out, int(status.item())


def oracle_cpu(cands, dest, forced, n_ord):
    """Pure CPU: the torch reference merge body, no CUDA anywhere."""
    world = len(cands)
    recv = recv_carrier([c.cpu() for c in cands], dest)
    T, hl = recv.shape[1], recv.shape[2]
    out = torch.empty((T, hl, CAND_K), dtype=torch.int32)
    status = torch.zeros(1, dtype=torch.int32)
    assert recv.shape[0] == world
    _torch_merge(
        recv,
        out,
        None if forced is None else forced.cpu(),
        None if n_ord is None else n_ord.cpu(),
        status,
    )
    return out, int(status.item())


# --------------------------------------------------------------------------
# single-GPU W-rank emulation of the symmetric windows
# --------------------------------------------------------------------------


class Emulation:
    """``world`` K5T workspaces on one device, one stream per rank."""

    def __init__(
        self, mod, *, world: int, t_cap: int, h_local: int, slots: int = 3
    ) -> None:
        self.mod, self.world, self.t_cap, self.hl = mod, world, t_cap, h_local
        self.hg, self.slots = world * h_local, slots
        self.slot_words = int(mod.k5t_slot_words(t_cap, self.hg))
        ntiles_cap = int(mod.k5t_plan(t_cap, t_cap, h_local, world, 1)[2])
        dev = torch.device("cuda")
        self.windows = [
            torch.zeros(slots * self.slot_words, dtype=torch.int32, device=dev)
            for _ in range(world)
        ]
        self.gens = [
            torch.zeros((slots, ntiles_cap), dtype=torch.int32, device=dev)
            for _ in range(world)
        ]
        self.status = [
            torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(world)
        ]
        self.streams = [torch.cuda.Stream() for _ in range(world)]
        self.ptrs = [w.data_ptr() for w in self.windows]

    def launch(
        self,
        cands,
        outs,
        forced,
        n_ord,
        *,
        layer_idx: int,
        max_ctas: int = 16,
        flags: int = 0,
        spin: int = LONG_SPIN,
        head_offsets=None,
    ) -> None:
        for r in range(self.world):
            ho = r * self.hl if head_offsets is None else head_offsets[r]
            with torch.cuda.stream(self.streams[r]):
                self.mod.k5t_exchange(
                    cands[r],
                    outs[r],
                    self.ptrs,
                    self.gens[r],
                    self.status[r],
                    forced,
                    n_ord,
                    r,
                    self.world,
                    ho,
                    layer_idx % self.slots,
                    self.slots,
                    self.t_cap,
                    self.slot_words,
                    max_ctas,
                    spin,
                    flags,
                )

    def run(self, cands, forced, n_ord, *, layer_idx: int, **kw):
        T = cands[0].shape[0]
        outs = [
            torch.full(
                (T, self.hl, CAND_K), 0x5A5A5A5A, dtype=torch.int32, device="cuda"
            )
            for _ in range(self.world)
        ]
        torch.cuda.synchronize()
        self.launch(cands, outs, forced, n_ord, layer_idx=layer_idx, **kw)
        torch.cuda.synchronize()
        return outs

    def poison(self, slot: int, tag: int) -> None:
        """Every record of ``slot`` := (3e38 bits, POISON_ID), tagged ``tag``."""
        for w in self.windows:
            region = w[slot * self.slot_words : (slot + 1) * self.slot_words]
            rec = region.view(-1, 4)
            rec[:, 0] = _bits(POISON_SCORE)
            rec[:, 1] = tag
            rec[:, 2] = POISON_ID
            rec[:, 3] = tag


@pytest.fixture(scope="module")
def k5t():
    from fmha_sm100.icp import _build

    return _build._k5t()


def _inputs(world, hl, T, seed, *, planes=True):
    cands = [
        make_rank_candidates(T, world * hl, r, seed=seed).cuda() for r in range(world)
    ]
    forced = n_ord = None
    if planes:
        forced, n_ord = (p.cuda() for p in make_planes(T, seed=seed))
    return cands, forced, n_ord


# --------------------------------------------------------------------------
# host-only: the grid rule
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "T,grid", [(1, 1), (4, 1), (16, 4), (1024, 256), (3392, 848), (16384, 848)]
)
def test_grid_follows_the_occupancy_rule_at_rubin_residency(k5t, T, grid):
    # Rubin: 212 SMs x 4 resident 256-thread CTAs (1024 threads/SM).
    tt, ntiles, _, g, cap = k5t.k5t_plan(T, 16384, 2, 2, 212 * 4)
    assert tt == 4 and cap == 848 and ntiles == -(-T // 4)
    assert g == grid == min(-(-T * 2 // 8), 848)


@pytest.mark.gpu
def test_auto_grid_is_bounded_by_live_residency(k5t):
    props = torch.cuda.get_device_properties(0)
    _, ntiles, _, g, cap = k5t.k5t_plan(16384, 16384, 2, 2, 0)
    assert cap % props.multi_processor_count == 0
    assert 1 <= cap // props.multi_processor_count <= 16
    assert g == min(ntiles, cap)


# --------------------------------------------------------------------------
# the CPU oracle agrees with the K2 path it replaces
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("world,hl,T", [(2, 2, 37), (4, 1, 20), (2, 2, 1)])
def test_cpu_torch_oracle_equals_k2(world, hl, T):
    cands, forced, n_ord = _inputs(world, hl, T, seed=3)
    for dest in range(world):
        k2, s_k2 = oracle_k2(cands, dest, forced, n_ord)
        cpu, s_cpu = oracle_cpu(cands, dest, forced, n_ord)
        assert torch.equal(k2.cpu(), cpu) and s_k2 == s_cpu == 0


# --------------------------------------------------------------------------
# bit identity, every extent, one window
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "world,hl,t_cap,extents",
    [
        (2, 2, 16384, [1, 3, 4, 5, 64, 257, 1024, 4096, 16384]),
        (4, 1, 1024, [1, 7, 1024]),
        (8, 1, 256, [1, 9, 256]),
        (1, 2, 64, [1, 64]),
    ],
)
@pytest.mark.parametrize("max_ctas", [16, 1])
def test_k5t_is_bit_identical_to_all_to_all_plus_k2(
    k5t, world, hl, t_cap, extents, max_ctas
):
    emu = Emulation(k5t, world=world, t_cap=t_cap, h_local=hl)
    for step, T in enumerate(extents):
        cands, forced, n_ord = _inputs(world, hl, T, seed=100 + step)
        outs = emu.run(cands, forced, n_ord, layer_idx=step, max_ctas=max_ctas)
        for r in range(world):
            ref, s_ref = oracle_k2(cands, r, forced, n_ord)
            assert torch.equal(outs[r], ref), (world, T, r)
            assert s_ref == 0 and int(emu.status[r].item()) == 0
            if T <= 1024:
                cpu, _ = oracle_cpu(cands, r, forced, n_ord)
                assert torch.equal(outs[r].cpu(), cpu)


@pytest.mark.gpu
def test_plain_merge_without_planes_matches_k2(k5t):
    emu = Emulation(k5t, world=2, t_cap=512, h_local=2)
    cands, _, _ = _inputs(2, 2, 300, seed=9, planes=False)
    outs = emu.run(cands, None, None, layer_idx=0)
    for r in range(2):
        assert torch.equal(outs[r], oracle_k2(cands, r, None, None)[0])


@pytest.mark.gpu
@pytest.mark.parametrize("world,hl", [(2, 2), (1, 2), (2, 1)])
def test_network_merge_equals_classic_and_k2(k5t, world, hl):
    # The default one-candidate-per-lane merge is the shuffle network; the
    # classic flag must give the same bits, and both must equal K2.
    classic = int(k5t.k5t_flag_classic_merge)
    emu = Emulation(k5t, world=world, t_cap=1024, h_local=hl)
    for layer, T in enumerate([1, 37, 256, 1024]):
        cands, forced, n_ord = _inputs(world, hl, T, seed=900 + layer)
        net = emu.run(cands, forced, n_ord, layer_idx=2 * layer)
        cla = emu.run(cands, forced, n_ord, layer_idx=2 * layer + 1, flags=classic)
        for r in range(world):
            ref, _ = oracle_k2(cands, r, forced, n_ord)
            assert torch.equal(net[r], cla[r]) and torch.equal(net[r], ref)


@pytest.mark.gpu
def test_nan_and_row_meta_failures_render_and_report_like_k2(k5t):
    emu = Emulation(k5t, world=2, t_cap=64, h_local=2)
    cands, forced, n_ord = _inputs(2, 2, 40, seed=11)
    nan_bits = _bits(float("nan"))
    cands[1][5, 0, 0] = torch.tensor([nan_bits, 7], dtype=torch.int32)
    cands[0][9, 3, 2] = torch.tensor([nan_bits, 8], dtype=torch.int32)
    n_ord = n_ord.clone()
    n_ord[17] = (n_ord[17] + 1) % 16  # disagree with C3
    outs = emu.run(cands, forced, n_ord, layer_idx=0)
    for r in range(2):
        ref, s_ref = oracle_k2(cands, r, forced, n_ord)
        assert torch.equal(outs[r], ref)
        assert int(emu.status[r].item()) == s_ref == 3  # NaN | RowMeta
        assert bool((outs[r][17] == -1).all())


# --------------------------------------------------------------------------
# slot reuse over the model sweep, changing extent AND grid per launch
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_57_layer_sweeps_with_3_slots_and_changing_grids(k5t):
    emu = Emulation(k5t, world=2, t_cap=4096, h_local=2, slots=3)
    extents = [16, 4096, 1000, 4, 4096]
    grids = [1, 3, 16, 64, 7]
    for step, T in enumerate(extents):
        for layer in range(57):
            cands, forced, n_ord = _inputs(2, 2, T, seed=1000 * step + layer)
            outs = emu.run(
                cands,
                forced,
                n_ord,
                layer_idx=layer,
                max_ctas=grids[(step + layer) % len(grids)],
            )
            for r in range(2):
                ref, _ = oracle_k2(cands, r, forced, n_ord)
                assert torch.equal(outs[r], ref), (step, layer, r)
    assert all(int(s.item()) == 0 for s in emu.status)
    # Both ranks' tile counters advanced in lockstep.
    assert torch.equal(emu.gens[0], emu.gens[1])


@pytest.mark.gpu
def test_cuda_graph_replay_advances_generations_on_device(k5t):
    emu = Emulation(k5t, world=2, t_cap=256, h_local=2)
    T = 200
    static = [
        torch.zeros((T, 4, CAND_K, 2), dtype=torch.int32, device="cuda")
        for _ in range(2)
    ]
    forced = torch.zeros(T, dtype=torch.int32, device="cuda")
    n_ord = torch.zeros(T, dtype=torch.int32, device="cuda")
    outs = [
        torch.zeros((T, 2, CAND_K), dtype=torch.int32, device="cuda") for _ in range(2)
    ]
    graphs = [torch.cuda.CUDAGraph() for _ in range(2)]
    torch.cuda.synchronize()
    for r in range(2):
        with torch.cuda.graph(graphs[r], stream=emu.streams[r]):
            emu.mod.k5t_exchange(
                static[r],
                outs[r],
                emu.ptrs,
                emu.gens[r],
                emu.status[r],
                forced,
                n_ord,
                r,
                2,
                r * 2,
                1,
                3,
                256,
                emu.slot_words,
                8,
                LONG_SPIN,
                0,
            )
    for rep in range(6):
        cands, f, q = _inputs(2, 2, T, seed=500 + rep)
        for r in range(2):
            static[r].copy_(cands[r])
        forced.copy_(f)
        n_ord.copy_(q)
        torch.cuda.synchronize()
        for r in range(2):
            with torch.cuda.stream(emu.streams[r]):
                graphs[r].replay()
        torch.cuda.synchronize()
        for r in range(2):
            assert torch.equal(outs[r], oracle_k2(cands, r, f, q)[0]), rep
    assert int(emu.gens[0][1, 0].item()) == 6


# --------------------------------------------------------------------------
# negative controls -- each must fire
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_negative_wrong_head_offset_is_refused(k5t):
    emu = Emulation(k5t, world=2, t_cap=64, h_local=2)
    cands, forced, n_ord = _inputs(2, 2, 8, seed=1)
    outs = [
        torch.empty((8, 2, CAND_K), dtype=torch.int32, device="cuda") for _ in range(2)
    ]
    with pytest.raises(RuntimeError, match="head_offset"):
        emu.launch(cands, outs, forced, n_ord, layer_idx=0, head_offsets=[1, 2])


@pytest.mark.gpu
def test_negative_misrouted_oracle_differs(k5t):
    emu = Emulation(k5t, world=2, t_cap=64, h_local=2)
    cands, forced, n_ord = _inputs(2, 2, 48, seed=5)
    outs = emu.run(cands, forced, n_ord, layer_idx=0)
    for r in range(2):
        wrong, _ = oracle_k2(cands, 1 - r, forced, n_ord)
        assert not torch.equal(outs[r], wrong)
        assert int((outs[r] != wrong).sum()) > 48


@pytest.mark.gpu
def test_negative_poisoned_slot_surfaces_only_without_the_publish(k5t):
    emu = Emulation(k5t, world=2, t_cap=128, h_local=2)
    T = 96
    cands, _, _ = _inputs(2, 2, T, seed=21, planes=False)
    # Fresh counters: the next generation on slot 0 is 1. A poison tagged 1
    # is indistinguishable from a real publish -- only the publish removes it.
    emu.poison(slot=0, tag=1)
    outs = emu.run(cands, None, None, layer_idx=0, flags=int(k5t.k5t_flag_no_publish))
    for r in range(2):
        ref, _ = oracle_k2(cands, r, None, None)
        assert not torch.equal(outs[r], ref)
        # Exactly one poison id per row: the duplicate max-reduce collapses
        # 32 identical records into one winner.
        assert torch.equal(
            (outs[r] == POISON_ID).sum(-1),
            torch.ones((T, 2), dtype=torch.int64, device="cuda"),
        )
    # Production flags over a poison carrying the PREVIOUS generation (the
    # counter is 1, the launch expects 2): the reachable stale-slot state.
    # It must never be accepted, so the result is exact.
    emu.poison(slot=0, tag=1)
    outs = emu.run(cands, None, None, layer_idx=0)
    for r in range(2):
        assert torch.equal(outs[r], oracle_k2(cands, r, None, None)[0])


@pytest.mark.gpu
def test_negative_stale_tag_times_out_and_kills_the_workspace(k5t):
    emu = Emulation(k5t, world=2, t_cap=64, h_local=2)
    cands, forced, n_ord = _inputs(2, 2, 32, seed=31)
    transport = int(k5t.k5t_status_transport)
    # The zeroed window carries tag 0; with the publish removed nothing ever
    # carries generation 1.
    outs = emu.run(
        cands,
        forced,
        n_ord,
        layer_idx=0,
        spin=SHORT_SPIN,
        flags=int(k5t.k5t_flag_no_publish),
    )
    for r in range(2):
        assert int(emu.status[r].item()) & transport
        assert bool((outs[r] == -1).all())
    # Dead: a correct call now renders -1 without waiting or publishing.
    outs = emu.run(cands, forced, n_ord, layer_idx=1, spin=SHORT_SPIN)
    for r in range(2):
        assert bool((outs[r] == -1).all())
        assert not bool(emu.windows[r][emu.slot_words : 2 * emu.slot_words].any())


@pytest.mark.gpu
def test_negative_skipping_the_tag_check_reads_a_stale_slot(k5t):
    emu = Emulation(k5t, world=2, t_cap=64, h_local=2)
    cands, _, _ = _inputs(2, 2, 32, seed=41, planes=False)
    emu.poison(slot=0, tag=0x7777)  # a generation nobody expects
    no_pub_no_acq = int(k5t.k5t_flag_no_publish | k5t.k5t_flag_no_acquire)
    outs = emu.run(cands, None, None, layer_idx=0, flags=no_pub_no_acq)
    for r in range(2):
        assert bool((outs[r] == POISON_ID).any(-1).all())


# --------------------------------------------------------------------------
# real symmetric memory (CUDA backend)
# --------------------------------------------------------------------------


@pytest.mark.distributed
def test_real_window_world_any_matches_k2(icp_group_any):
    import torch.distributed as dist

    from fmha_sm100.icp import IcpTiledExchange

    rank = dist.get_rank(icp_group_any)
    world = dist.get_world_size(icp_group_any)
    hl = 2
    with IcpTiledExchange(
        icp_group_any, tokens=2048, heads_group=world * hl, heads_local=hl, slots=3
    ) as ex:
        assert ex.provenance()["symm_backend"] == "CUDA"
        for step, T in enumerate([4, 2048, 777, 16]):
            for layer in range(57):
                all_c, forced, n_ord = _inputs(world, hl, T, seed=10 * step + layer)
                out = torch.empty((T, hl, CAND_K), dtype=torch.int32, device="cuda")
                ex.exchange_and_merge(
                    all_c[rank],
                    out=out,
                    layer_idx=layer,
                    forced=forced,
                    n_ordinary=n_ord,
                )
                ref = torch.empty_like(out)
                status = torch.zeros(1, dtype=torch.int32, device="cuda")
                carrier.collective_exchange_and_merge(
                    all_c[rank],
                    world=world,
                    rank=rank,
                    out=ref,
                    forced=forced,
                    n_ordinary=n_ord,
                    status=status,
                    transport="all_to_all",
                    group=icp_group_any,
                )
                ok = torch.equal(out, ref)
                flag = torch.tensor([int(ok)], device="cuda")
                dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=icp_group_any)
                assert bool(flag.item()), (step, layer, rank)
        ex.check_error()
