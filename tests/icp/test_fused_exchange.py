"""D3 (selector-side publish + early poll/merge) against the routes it replaces.

Oracles: the UNCHANGED full-row decode selector writes each rank's C4 rows,
then (a) K2 over the emulated ``all_to_all`` carrier and (b) K5T over the
same rows. D3 must equal both on every row and every status bit.

Single-GPU emulation: ``W`` plain device windows, one selector stream and one
merge stream per rank. ``merges_first`` launches every merge BEFORE any
selector, so the merges really spin on tags that do not exist yet -- the
early-launch ordering PDL produces in the model.

Negative controls must fire: no publish -> transport timeout and a dead
workspace; a stale generation is refused; skipping the acquire reads it; a
merge fed the other rank's window differs.

The torch-free twin of the bit-identity and negative gates is
``tests/native/fused_exchange_standalone.cu`` (runs on any GPU >= sm_70).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fmha_sm100.icp import carrier  # noqa: E402

CAND_K = 16
LONG_SPIN = 2_000_000_000
SHORT_SPIN = 2_000_000
POISON_ID = 0x00BADBAD


# --------------------------------------------------------------------------
# inputs: decode score planes whose ids collide across ranks (fragment placement)
# --------------------------------------------------------------------------


def make_invocation(world, hl, T, N, *, seed, stride=1, begin=0):
    g = torch.Generator().manual_seed(seed)
    hg = world * hl
    base = torch.randn((T, hg, N), generator=g) * 2
    scores = []
    for _ in range(world):
        noise = torch.randn((T, hg, N), generator=g) * 0.5
        keep = torch.rand((T, hg, N), generator=g) < 0.5
        s = torch.where(keep, base, base + noise)
        scores.append((torch.round(s * 8) / 8).float().cuda())  # exact ties
    nvalid = torch.randint(0, N + 1, (T,), generator=g, dtype=torch.int32)
    short = torch.rand((T,), generator=g) < 0.1
    nvalid = torch.where(short, torch.clamp(nvalid, max=min(N, 20)), nvalid)
    active = torch.rand((T,), generator=g) >= 0.08
    has_forced = (nvalid > 0) & (torch.rand((T,), generator=g) < 0.85)
    forced_col = torch.where(has_forced, nvalid - 1, torch.full_like(nvalid, -1))
    f = torch.where(
        active & (forced_col >= 0),
        begin + stride * forced_col,
        torch.full_like(forced_col, -1),
    )
    n_ord = torch.where(f >= 0, torch.clamp(f, max=CAND_K - 1), torch.zeros_like(f)).to(
        torch.int32
    )
    return dict(
        scores=scores,
        nvalid=nvalid.cuda(),
        forced_col=forced_col.cuda(),
        active=active.cuda(),
        forced=f.to(torch.int32).cuda(),
        n_ord=n_ord.cuda(),
        T=T,
        N=N,
        stride=stride,
        begin=begin,
    )


def selector_rows(sel, inv, r):
    """The unchanged full-row selector's C4 rows for rank ``r`` (int32 view)."""
    T, N = inv["T"], inv["N"]
    hg = inv["scores"][r].shape[1]
    parts = -(-N // 4096) if N > 4096 else 0
    out = torch.empty((T, hg, CAND_K, 2), dtype=torch.float32, device="cuda")
    partials = torch.empty(
        (T, hg, parts, CAND_K, 2), dtype=torch.float32, device="cuda"
    )
    sel.select_local_candidates_full_row(
        inv["scores"][r],
        inv["nvalid"],
        inv["forced_col"],
        inv["active"],
        out,
        partials,
        inv["begin"],
        False,
        global_block_stride=inv["stride"],
    )
    return out.view(torch.int32)


def oracle_k2(cands, dest, forced, n_ord):
    from fmha_sm100.icp import merge_candidates

    world = len(cands)
    recv = torch.stack(
        [carrier.pack_send_carrier(c, world)[dest] for c in cands]
    ).contiguous()
    status = torch.zeros(1, dtype=torch.int32, device=recv.device)
    out = merge_candidates(
        recv, world=world, rank=dest, forced=forced, n_ordinary=n_ord, status=status
    )
    return out, int(status.item())


# --------------------------------------------------------------------------
# single-GPU W-rank emulation
# --------------------------------------------------------------------------


class Emulation:
    def __init__(self, mod, *, world, t_cap, h_local, slots=3):
        self.mod, self.world, self.t_cap, self.hl = mod, world, t_cap, h_local
        self.hg, self.slots = world * h_local, slots
        self.slot_words = int(mod.fused_slot_words(t_cap, self.hg))
        dev = torch.device("cuda")
        self.windows = [
            torch.zeros(slots * self.slot_words, dtype=torch.int32, device=dev)
            for _ in range(world)
        ]
        self.pub = [
            torch.zeros((slots, t_cap, self.hg), dtype=torch.int32, device=dev)
            for _ in range(world)
        ]
        self.exp = [
            torch.zeros((slots, t_cap, h_local), dtype=torch.int32, device=dev)
            for _ in range(world)
        ]
        self.status = [
            torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(world)
        ]
        self.sel_streams = [torch.cuda.Stream() for _ in range(world)]
        self.merge_streams = [torch.cuda.Stream() for _ in range(world)]
        self.ptrs = [w.data_ptr() for w in self.windows]

    def select(self, inv, r, *, layer_idx, chunk=None, flags=0, mirror=None):
        T = inv["T"]
        chunk = chunk or T
        with torch.cuda.stream(self.sel_streams[r]):
            for t0 in range(0, T, chunk):
                t1 = min(T, t0 + chunk)
                self.mod.fused_select_publish(
                    inv["scores"][r][t0:t1],
                    inv["nvalid"][t0:t1],
                    inv["forced_col"][t0:t1],
                    inv["active"][t0:t1],
                    self.ptrs,
                    self.pub[r],
                    self.status[r],
                    None if mirror is None else mirror[t0:t1],
                    r,
                    self.world,
                    self.hl,
                    t0,
                    layer_idx % self.slots,
                    self.slots,
                    self.t_cap,
                    self.slot_words,
                    inv["begin"],
                    inv["stride"],
                    flags,
                )

    def merge(
        self,
        out,
        inv,
        r,
        *,
        layer_idx,
        flags=0,
        spin=LONG_SPIN,
        max_ctas=16,
        window=None,
    ):
        with torch.cuda.stream(self.merge_streams[r]):
            self.mod.fused_merge(
                out,
                self.ptrs[r if window is None else window],
                self.exp[r],
                self.status[r],
                inv["forced"],
                inv["n_ord"],
                self.world,
                layer_idx % self.slots,
                self.slots,
                self.t_cap,
                self.slot_words,
                max_ctas,
                spin,
                flags,
            )

    def run(
        self,
        inv,
        *,
        layer_idx,
        merges_first=False,
        chunk=None,
        sel_flags=0,
        merge_flags=0,
        spin=LONG_SPIN,
        max_ctas=16,
        mirrors=None,
    ):
        T = inv["T"]
        outs = [
            torch.full(
                (T, self.hl, CAND_K), 0x5A5A5A5A, dtype=torch.int32, device="cuda"
            )
            for _ in range(self.world)
        ]
        torch.cuda.synchronize()

        def selects():
            for r in range(self.world):
                self.select(
                    inv,
                    r,
                    layer_idx=layer_idx,
                    chunk=chunk,
                    flags=sel_flags,
                    mirror=None if mirrors is None else mirrors[r],
                )

        def merges():
            for r in range(self.world):
                self.merge(
                    outs[r],
                    inv,
                    r,
                    layer_idx=layer_idx,
                    flags=merge_flags,
                    spin=spin,
                    max_ctas=max_ctas,
                )

        if merges_first:
            merges()
            selects()
        else:
            selects()
            torch.cuda.synchronize()
            merges()
        torch.cuda.synchronize()
        return outs


@pytest.fixture(scope="module")
def fused():
    from fmha_sm100.icp import _build

    return _build._fused()


@pytest.fixture(scope="module")
def sel():
    from fmha_sm100.icp import _build

    return _build._select_ext()


def _pdl_flags(mod):
    if torch.cuda.get_device_capability()[0] < 9:
        return 0, 0
    early = int(mod.fused_flag_pdl | mod.fused_flag_early_trigger)
    return early, early | int(mod.fused_flag_end_wait)


# --------------------------------------------------------------------------
# bit identity
# --------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize(
    "world,hl,t_cap,cases",
    [
        (
            2,
            2,
            1024,
            [(32, 1172), (1, 64), (4, 8192), (48, 300), (257, 1172), (1024, 600)],
        ),
        (4, 1, 256, [(7, 900), (256, 17)]),
        (8, 1, 64, [(9, 500), (64, 64)]),
        (2, 4, 128, [(33, 700)]),
    ],
)
@pytest.mark.parametrize("merges_first", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
def test_d3_is_bit_identical_to_selector_plus_k2(
    fused, sel, world, hl, t_cap, cases, merges_first, stride
):
    emu = Emulation(fused, world=world, t_cap=t_cap, h_local=hl)
    sf, mf = _pdl_flags(fused)
    for layer, (T, N) in enumerate(cases):
        inv = make_invocation(
            world, hl, T, N, seed=100 * layer + T, stride=stride, begin=stride - 1
        )
        cands = [selector_rows(sel, inv, r) for r in range(world)]
        mirrors = [
            torch.empty((T, world * hl, CAND_K, 2), dtype=torch.float32, device="cuda")
            for _ in range(world)
        ]
        outs = emu.run(
            inv,
            layer_idx=layer,
            merges_first=merges_first,
            sel_flags=sf,
            merge_flags=mf,
            mirrors=mirrors,
        )
        for r in range(world):
            ref, s_ref = oracle_k2(cands, r, inv["forced"], inv["n_ord"])
            assert torch.equal(outs[r], ref), (world, T, N, r)
            assert int(emu.status[r].item()) == s_ref
            assert torch.equal(mirrors[r].view(torch.int32), cands[r])


@pytest.mark.gpu
def test_d3_equals_k5t_on_the_same_rows(fused, sel):
    from fmha_sm100.icp import _build

    k5t = _build._k5t()
    from test_tiled_exchange import Emulation as K5TEmulation

    world, hl, T = 2, 2, 64
    k5 = K5TEmulation(k5t, world=world, t_cap=256, h_local=hl)
    emu = Emulation(fused, world=world, t_cap=256, h_local=hl)
    for layer in range(7):
        inv = make_invocation(world, hl, T, 1172, seed=7 + layer)
        cands = [selector_rows(sel, inv, r) for r in range(world)]
        a = k5.run(cands, inv["forced"], inv["n_ord"], layer_idx=layer)
        b = emu.run(inv, layer_idx=layer, merges_first=bool(layer % 2))
        for r in range(world):
            assert torch.equal(a[r], b[r]), (layer, r)


@pytest.mark.gpu
def test_d3_classic_merge_flag_is_bit_identical(fused, sel):
    emu = Emulation(fused, world=2, t_cap=512, h_local=2)
    classic = int(fused.fused_flag_classic_merge)
    for layer, (T, N) in enumerate([(32, 1172), (512, 300)]):
        inv = make_invocation(2, 2, T, N, seed=40 + layer)
        a = emu.run(inv, layer_idx=2 * layer)
        b = emu.run(inv, layer_idx=2 * layer + 1, merge_flags=classic)
        for r in range(2):
            assert torch.equal(a[r], b[r])


@pytest.mark.gpu
def test_multi_chunk_token_offsets_cover_the_extent(fused, sel):
    emu = Emulation(fused, world=2, t_cap=1024, h_local=2)
    inv = make_invocation(2, 2, 1000, 400, seed=3)
    cands = [selector_rows(sel, inv, r) for r in range(2)]
    outs = emu.run(inv, layer_idx=0, chunk=256)
    for r in range(2):
        assert torch.equal(outs[r], oracle_k2(cands, r, inv["forced"], inv["n_ord"])[0])


@pytest.mark.gpu
def test_57_layer_sweeps_keep_counters_in_lockstep(fused, sel):
    emu = Emulation(fused, world=2, t_cap=512, h_local=2, slots=3)
    grids = [1, 3, 16, 64, 7]
    for step, T in enumerate([16, 512, 100, 4, 512]):
        inv = make_invocation(2, 2, T, 1172, seed=77 + step)
        cands = [selector_rows(sel, inv, r) for r in range(2)]
        refs = [oracle_k2(cands, r, inv["forced"], inv["n_ord"])[0] for r in range(2)]
        for layer in range(57):
            outs = emu.run(
                inv,
                layer_idx=layer,
                merges_first=bool(layer % 2),
                max_ctas=grids[(step + layer) % 5],
            )
            for r in range(2):
                assert torch.equal(outs[r], refs[r]), (step, layer, r)
    assert all(int(s.item()) == 0 for s in emu.status)
    # Source c's publish counter for (t, d*Hl + hl) == dest d's expect counter.
    for c in range(2):
        pub = emu.pub[c].view(3, 512, 2, 2)  # [slot, t, dest, hl]
        for d in range(2):
            assert torch.equal(pub[:, :, d, :], emu.exp[d])


@pytest.mark.gpu
def test_cuda_graph_replay_advances_generations_on_device(fused, sel):
    world, hl, T, N = 2, 2, 32, 1172
    emu = Emulation(fused, world=world, t_cap=64, h_local=hl)
    sf, mf = _pdl_flags(fused)
    static = make_invocation(world, hl, T, N, seed=1)
    outs = [
        torch.zeros((T, hl, CAND_K), dtype=torch.int32, device="cuda")
        for _ in range(world)
    ]
    graphs = [torch.cuda.CUDAGraph() for _ in range(world)]
    torch.cuda.synchronize()
    for r in range(world):
        # One stream per rank inside the graph: select then merge, as in vLLM.
        with torch.cuda.graph(graphs[r], stream=emu.sel_streams[r]):
            emu.mod.fused_select_publish(
                static["scores"][r],
                static["nvalid"],
                static["forced_col"],
                static["active"],
                emu.ptrs,
                emu.pub[r],
                emu.status[r],
                None,
                r,
                world,
                hl,
                0,
                1,
                3,
                64,
                emu.slot_words,
                0,
                1,
                sf,
            )
            emu.mod.fused_merge(
                outs[r],
                emu.ptrs[r],
                emu.exp[r],
                emu.status[r],
                static["forced"],
                static["n_ord"],
                world,
                1,
                3,
                64,
                emu.slot_words,
                8,
                LONG_SPIN,
                mf,
            )
    for rep in range(6):
        inv = make_invocation(world, hl, T, N, seed=500 + rep)
        for key in ("nvalid", "forced_col", "active", "forced", "n_ord"):
            static[key].copy_(inv[key])
        for r in range(world):
            static["scores"][r].copy_(inv["scores"][r])
        cands = [selector_rows(sel, static, r) for r in range(world)]
        torch.cuda.synchronize()
        for r in range(world):
            with torch.cuda.stream(emu.sel_streams[r]):
                graphs[r].replay()
        torch.cuda.synchronize()
        for r in range(world):
            ref, _ = oracle_k2(cands, r, static["forced"], static["n_ord"])
            assert torch.equal(outs[r], ref), rep
    assert int(emu.exp[0][1, 0, 0].item()) == 6
    assert int(emu.pub[1][1, 0, 0].item()) == 6


# --------------------------------------------------------------------------
# negative controls -- each must fire
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_negative_no_publish_times_out_and_kills_the_workspace(fused):
    emu = Emulation(fused, world=2, t_cap=64, h_local=2)
    inv = make_invocation(2, 2, 32, 300, seed=5)
    transport = int(fused.fused_status_transport)
    outs = emu.run(
        inv, layer_idx=0, spin=SHORT_SPIN, sel_flags=int(fused.fused_flag_no_publish)
    )
    for r in range(2):
        assert int(emu.status[r].item()) & transport
        assert bool((outs[r] == -1).all())
    before = emu.windows[0][emu.slot_words : 2 * emu.slot_words].clone()
    outs = emu.run(inv, layer_idx=1, spin=SHORT_SPIN)
    assert torch.equal(before, emu.windows[0][emu.slot_words : 2 * emu.slot_words])
    for r in range(2):
        assert bool((outs[r] == -1).all())


@pytest.mark.gpu
def test_negative_one_unpublished_row_times_out(fused, sel):
    # The contract says every t < T must be published: merge over T while
    # publishing only T-4 rows must fail loudly, not return a plausible answer.
    emu = Emulation(fused, world=2, t_cap=64, h_local=2)
    inv = make_invocation(2, 2, 32, 300, seed=6)
    short = dict(
        inv,
        T=28,
        scores=[s[:28] for s in inv["scores"]],
        nvalid=inv["nvalid"][:28],
        forced_col=inv["forced_col"][:28],
        active=inv["active"][:28],
    )
    for r in range(2):
        emu.select(short, r, layer_idx=0)
    torch.cuda.synchronize()
    outs = [
        torch.empty((32, 2, CAND_K), dtype=torch.int32, device="cuda") for _ in range(2)
    ]
    for r in range(2):
        emu.merge(outs[r], inv, r, layer_idx=0, spin=SHORT_SPIN)
    torch.cuda.synchronize()
    for r in range(2):
        assert int(emu.status[r].item()) & int(fused.fused_status_transport)


@pytest.mark.gpu
def test_negative_stale_generation_is_refused_and_no_acquire_reads_it(fused, sel):
    emu = Emulation(fused, world=2, t_cap=64, h_local=2)
    inv = make_invocation(2, 2, 40, 300, seed=9)
    cands = [selector_rows(sel, inv, r) for r in range(2)]
    refs = [oracle_k2(cands, r, inv["forced"], inv["n_ord"])[0] for r in range(2)]
    emu.run(inv, layer_idx=0)  # generation 1 on slot 0

    def poison():
        big = int(torch.tensor([3.0e38]).view(torch.int32))
        for w in emu.windows:
            rec = w[: emu.slot_words].view(-1, 4)
            rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3] = big, 1, POISON_ID, 1

    poison()  # tagged with the PREVIOUS generation; layer 3 expects 2
    outs = emu.run(inv, layer_idx=3, merges_first=True)
    for r in range(2):
        assert torch.equal(outs[r], refs[r])
    poison()
    outs = emu.run(
        inv,
        layer_idx=6,
        sel_flags=int(fused.fused_flag_no_publish),
        merge_flags=int(fused.fused_flag_no_acquire),
    )
    assert any(bool((o == POISON_ID).any()) for o in outs)


@pytest.mark.gpu
def test_negative_other_ranks_window_differs(fused, sel):
    emu = Emulation(fused, world=2, t_cap=64, h_local=2)
    inv = make_invocation(2, 2, 48, 600, seed=13)
    cands = [selector_rows(sel, inv, r) for r in range(2)]
    for r in range(2):
        emu.select(inv, r, layer_idx=0)
    torch.cuda.synchronize()
    wrong = torch.empty((48, 2, CAND_K), dtype=torch.int32, device="cuda")
    emu.merge(wrong, inv, 0, layer_idx=0, window=1)
    torch.cuda.synchronize()
    assert not torch.equal(wrong, oracle_k2(cands, 0, inv["forced"], inv["n_ord"])[0])


# --------------------------------------------------------------------------
# real symmetric memory (CUDA backend), against the NCCL route
# --------------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.parametrize(
    "use_pdl,end_wait", [(True, True), (True, False), (False, False)]
)
def test_real_window_matches_nccl_route(icp_group, sel, use_pdl, end_wait):
    import torch.distributed as dist

    from fmha_sm100.icp import IcpFusedExchange
    from fmha_sm100.icp.candidates import CandidateGeometry

    rank = dist.get_rank(icp_group)
    world = dist.get_world_size(icp_group)
    hl = 2
    if use_pdl and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("PDL needs sm_90+")
    with IcpFusedExchange(
        icp_group,
        tokens=2048,
        heads_group=world * hl,
        heads_local=hl,
        slots=3,
        use_pdl=use_pdl,
        end_wait=end_wait,
    ) as ex:
        assert ex.provenance()["symm_backend"] == "CUDA"
        for step, (T, N) in enumerate([(32, 1172), (2048, 300), (777, 64), (16, 8192)]):
            for layer in range(57):
                inv = make_invocation(world, hl, T, N, seed=10 * step + layer)
                geom = CandidateGeometry(
                    local_valid_blocks=inv["nvalid"],
                    forced_column=inv["forced_col"],
                    active_rows=inv["active"],
                )
                out = torch.empty((T, hl, CAND_K), dtype=torch.int32, device="cuda")
                ex.select_and_publish(inv["scores"][rank], geom, layer_idx=layer)
                ex.merge(
                    out, layer_idx=layer, forced=inv["forced"], n_ordinary=inv["n_ord"]
                )
                ref = torch.empty_like(out)
                status = torch.zeros(1, dtype=torch.int32, device="cuda")
                carrier.collective_exchange_and_merge(
                    selector_rows(sel, inv, rank),
                    world=world,
                    rank=rank,
                    out=ref,
                    forced=inv["forced"],
                    n_ordinary=inv["n_ord"],
                    status=status,
                    transport="all_to_all",
                    group=icp_group,
                )
                ok = torch.equal(out, ref)
                flag = torch.tensor([int(ok)], device="cuda")
                dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=icp_group)
                assert bool(flag.item()), (step, layer, rank)
        ex.check_error()
