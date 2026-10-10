"""The C4 carrier and the prefill NCCL route (pack -> all_to_all -> K2).

**The carrier (CPU).** ``pack_send_carrier`` turns the producer's query-major
``[Qchunk, H_group, 16, 2]`` output into C4's destination-major int32
``[W, Qchunk, H_local, 16, 2]`` send carrier. It is a **real permutation**; the
``.view``-reshape that happens to work at ``Qchunk == 1`` is kept in the package
as ``carrier._wrong_pack_view_negative_control`` and is gated here in BOTH
directions.

**The transport and collective (distributed).** ``all_to_all`` against the
``all_gather`` + head-slice reference, byte for byte, and the retained-workspace
collective against a hand-written pack -> exchange -> K2 merge.

Run::

    torchrun --nproc_per_node=2 -m pytest tests/icp/test_carrier.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from fmha_sm100.icp import (
    abi,
    carrier,
    exchange_carrier,
    forced_rows,
    invalid_carrier,
    merge_candidates,
    pack_send_carrier,
)

CAND_K = 16
INF = float("inf")

def score_bits(value: float) -> int:
    """An fp32 score as the int32 word C4 puts on the wire. A BITCAST.

    Written as a host int so it can be assigned into a CUDA int32 carrier
    without dragging a CPU tensor across the device boundary -- and so that
    nothing on this path can accidentally become a numeric conversion.
    """
    return int(torch.tensor([value], dtype=torch.float32).view(torch.int32).item())


NEG_INF_BITS = score_bits(-INF)


def make_candidates(
    T: int,
    Hg: int,
    rank: int,
    device,
    *,
    world: int = 1,
    seed: int = 0,
    domain: int = 1 << 12,
) -> torch.Tensor:
    """A rank's query-major ``[T, H_group, 16, 2]`` int32 candidates (C4).

    int32, because C4's carrier words are the fp32 score's raw bits and the
    int32 block id, **both bitcast**. The producer emits fp32 and the consumer
    reinterprets with ``.view(torch.int32)``; that is what this helper models.

    Global block ids are drawn from ONE shared domain of ``domain`` blocks,
    identical on every rank.

    USED TO READ ``ids = local * world + rank``, with a docstring explaining
    that C1 gave every block exactly one owner and that the merge's tie-free
    arithmetic relied on ids being disjoint across ranks. refined-icp-v1 C1
    deletes that: every rank holds ``R = 128/W`` rows of every logical block and
    publishes a *partial maximum* for it, so the same id arriving from several
    sources is the normal case and the merge max-reduces by id. Drawing disjoint
    ids would now test a configuration production never produces.

    Distinctness *within* one rank's row is still real and still deliberate.
    ``torch.randint`` samples with replacement, so 16 draws from 4096 collide in
    a few percent of rows; C4's producer emits each block at most once per row,
    so this helper samples without replacement -- ranking a random pool and
    keeping the first 16.
    """
    g = torch.Generator(device=device).manual_seed(seed * 1000 + rank)
    cand = torch.empty((T, Hg, CAND_K, 2), dtype=torch.float32, device=device)
    cand[..., 0] = torch.randn((T, Hg, CAND_K), device=device, generator=g)
    pool = torch.rand((T, Hg, domain), device=device, generator=g)
    ids = pool.argsort(dim=-1)[..., :CAND_K].to(torch.int32)
    cand[..., 1] = ids.view(torch.float32)
    # The bitcast the producer's consumer makes. Never `.int()`, never `.to()`.
    return cand.view(torch.int32).contiguous()


def reference(
    cand: torch.Tensor,
    group,
    rank: int,
    world: int,
    Hl: int,
    *,
    transport: str = "all_gather",
    forced: torch.Tensor | None = None,
    n_ordinary: torch.Tensor | None = None,
) -> torch.Tensor:
    """CONTRACT C7 Impl A: pack -> exchange -> the K2 merge.

    ``forced``/``n_ordinary`` default to ``None``, which is what most callers
    here want: those tests compare the **transport** -- which candidates reach
    which row -- and passing no planes keeps both arms on one contract so the
    comparison stays about the thing it names.

    They are *expressible*, which they were not while K5 reached the merge
    through the four-argument compatibility overload, and
    ``test_exchange_under_c3_forcing_is_bit_identical_to_the_reference_merge``
    populates them. That arm exists because the C3 behaviour of K5 and of K2 is
    otherwise gated only against the same Python reference, one module apart, so
    a shared misreading of C3 would pass both; comparing the two kernels against
    each other is the check that does not share an oracle with either.

    The failure paths are still not compared: K5 renders a failed row as block
    id ``0`` and K2 as all ``-1`` (``docs/STATUS.md``).
    """
    send = pack_send_carrier(cand, world)
    recv = exchange_carrier(
        send, world=world, rank=rank, transport=transport, group=group
    )
    status = torch.zeros(1, dtype=torch.int32, device=cand.device)
    out = merge_candidates(
        recv,
        world=world,
        rank=rank,
        status=status,
        forced=forced,
        n_ordinary=n_ordinary,
    )
    assert int(status.item()) == 0, "the reference merge itself failed C5/C3"
    return out


def c3_planes(T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """C3's row metadata for ``T`` rows, from each row's own query position.

    The positions are spread so that ``f = p // 128`` differs from row to row --
    a plane that held one repeated value could not tell a kernel indexing by
    ``t`` from one indexing by anything else that happens to be constant. Row 0
    lands at ``f = 0``, whose ``n_ordinary`` is ``0``: the forced block is the
    row's only valid id, which is the narrowest case the reservation has.

    Identical on every rank by construction. Forcing is a function of the query
    position, and every ICP rank holds the same queries, so a plane that
    differed across ranks would be describing different work.
    """
    positions = torch.arange(T, dtype=torch.int32, device=device) * 200 + 40
    return forced_rows(positions)


def collective_status(
    cand,
    world,
    rank,
    out,
    forced,
    n_ordinary,
    transport,
    group,
    *,
    workspace,
    status=None,
):
    """``collective_exchange_and_merge`` with this module's fixed arguments."""
    from fmha_sm100.icp import collective_exchange_and_merge

    return collective_exchange_and_merge(
        cand,
        world=world,
        rank=rank,
        out=out,
        forced=forced,
        n_ordinary=n_ordinary,
        status=status,
        transport=transport,
        group=group,
        workspace=workspace,
    )


def all_agree(flag: bool, group) -> bool:
    """True only if every rank in the group passed True.

    A per-rank assertion in a distributed test either wedges the job (the ranks
    that passed go on to the next collective) or reports a failure on one rank
    and a pass on the others. Reducing first makes the whole group fail
    together, which is also what keeps the *next* test's call sequence aligned.
    """
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


@pytest.fixture()
def icp(icp_group):
    """rank, world, and this rank's device, after the group is up."""
    return (
        dist.get_rank(icp_group),
        dist.get_world_size(icp_group),
        torch.device("cuda", torch.cuda.current_device()),
    )


# --------------------------------------------------------------------------
# the carrier. CPU: no device, no process group.
# --------------------------------------------------------------------------


def _unique_local(q: int, h_group: int) -> torch.Tensor:
    """``[Qchunk, H_group, 16, 2]`` int32, every element distinct.

    Distinct values are the point: any mis-routing of a (query, head) pair is
    then visible as a value in the wrong place, rather than hidden by two slots
    that happened to agree.
    """
    n = q * h_group * CAND_K * 2
    return torch.arange(1, n + 1, dtype=torch.int32).reshape(q, h_group, CAND_K, 2)


@pytest.mark.parametrize("W", [2, 4])
@pytest.mark.parametrize("q", [1, 2, 5])
def test_pack_send_carrier_routes_every_head_to_its_destination(W, q):
    """The pack's definition, asserted elementwise.

    Destination ``d``'s slab is global heads ``[d*Hlocal, (d+1)*Hlocal)`` for
    ALL queries (C9's rank-major head order). At ``Qchunk > 1`` that is not the
    producer's memory order, which is the whole reason this is a permutation.
    """
    h_local = abi.h_local(W)
    local = _unique_local(q, W * h_local)
    packed = pack_send_carrier(local, W)

    assert packed.dtype == torch.int32 and packed.is_contiguous()
    assert tuple(packed.shape) == abi.send_carrier_shape(W, q)
    assert tuple(packed.shape) == carrier.send_carrier_shape(W, q, h_local)
    for d in range(W):
        for hl in range(h_local):
            assert torch.equal(packed[d, :, hl], local[:, d * h_local + hl])


@pytest.mark.parametrize("W", [2, 4])
def test_the_forbidden_view_reshape_fires_at_q_gt_1_and_cannot_at_q_eq_1(W):
    """The negative control, gated in BOTH directions.

    C4 forbids reinterpreting the producer's buffer as the send carrier. The
    forbidden reshape is *accidentally correct* at ``Qchunk == 1``, where the
    query and destination axes cannot interleave, so:

    * at ``Qchunk == 1`` the control MUST agree with the pack -- if it did not,
      the control would be testing something other than the axis order, and a
      ``Qchunk == 1`` suite would be "passing" for the wrong reason;
    * at ``Qchunk > 1`` it MUST differ -- if it did not, every gate built on it
      would be vacuous and a ``.view``-based pack would ship undetected.

    Asserting only the second half would leave a control that could be firing
    for any reason at all.
    """
    h_local = abi.h_local(W)

    one = _unique_local(1, W * h_local)
    assert torch.equal(
        pack_send_carrier(one, W), carrier._wrong_pack_view_negative_control(one, W)
    ), (
        "at Qchunk == 1 the pack and the reshape must coincide; if they do not, "
        "this control is not isolating the axis order"
    )

    for q in (2, 5):
        many = _unique_local(q, W * h_local)
        packed = pack_send_carrier(many, W)
        wrong = carrier._wrong_pack_view_negative_control(many, W)
        assert packed.shape == wrong.shape
        assert not torch.equal(packed, wrong), (
            f"the forbidden .view-reshape did not diverge from the pack at "
            f"Qchunk={q}, W={W}: the gate that proves the pack is a permutation "
            "is vacuous"
        )


def test_pack_send_carrier_bitcasts_an_fp32_producer_buffer():
    """The producer emits fp32; the pack reinterprets and never converts.

    An id above ``2**24`` has no exact fp32 representation, so a single
    ``.float()`` or ``.int()`` anywhere on this path corrupts large block ids
    and leaves small ones correct -- a defect that only appears on long
    sequences.
    """
    big = (1 << 24) + 1
    # PRECONDITION: the id must actually be one a conversion would destroy, or
    # this test cannot tell a bitcast from a cast.
    assert int(torch.tensor([big], dtype=torch.float32).item()) != big

    local = torch.zeros((2, 4, CAND_K, 2), dtype=torch.float32)
    local[..., 0] = 1.5
    local[..., 1] = torch.full((2, 4, CAND_K), big, dtype=torch.int32).view(
        torch.float32
    )
    packed = pack_send_carrier(local, 2)
    assert (packed[..., 1] == big).all()
    assert (packed[..., 0] == torch.tensor([1.5]).view(torch.int32).item()).all()


def test_an_int32_and_an_fp32_producer_buffer_pack_identically():
    local = torch.zeros((3, 4, CAND_K, 2), dtype=torch.float32)
    local[..., 0] = torch.randn((3, 4, CAND_K))
    local[..., 1] = torch.randint(0, 1 << 20, (3, 4, CAND_K), dtype=torch.int32).view(
        torch.float32
    )
    assert torch.equal(
        pack_send_carrier(local, 2), pack_send_carrier(local.view(torch.int32), 2)
    )


def test_pack_send_carrier_refuses_a_word_that_is_not_four_bytes():
    with pytest.raises(TypeError, match="4 bytes"):
        pack_send_carrier(torch.zeros((2, 4, CAND_K, 2), dtype=torch.float64), 2)


def test_an_empty_rank_still_publishes_a_full_slab():
    """C4: capture padding participates with invalid records.

    A rank with no valid candidates does not skip the collective -- it sends
    exactly as many bytes as everyone else, full of ``(-inf, -1)``. The element
    counts are a pure function of ``(W, Qchunk, Hlocal)``, which is what makes
    the exchange capturable.
    """
    W, q, h_local = 4, 3, abi.h_local(4)
    empty = invalid_carrier(W, q, h_local, device="cpu")
    assert tuple(empty.shape) == abi.send_carrier_shape(W, q)
    assert empty.dtype == torch.int32
    assert (empty[..., 1] == -1).all(), "every record must be invalid"
    assert (empty[..., 0] == NEG_INF_BITS).all()
    assert (
        empty.numel()
        == W * carrier.peer_split_int32_elements(q, h_local)
        == W * abi.peer_split_int32_elements(W, q)
    )


def test_exchange_carrier_at_world_one_is_a_copy_and_not_an_alias():
    # The one place where "no collective" is not a participation failure: there
    # is no peer. It must still be a real carrier, and a distinct buffer, or a
    # later in-place publish would mutate the merge's input.
    send = _unique_local(2, 4).reshape(1, 2, 4, CAND_K, 2).contiguous()
    recv = exchange_carrier(send, world=1, rank=0)
    assert torch.equal(recv, send)
    assert recv.data_ptr() != send.data_ptr()


def test_exchange_carrier_refuses_what_the_collective_would_silently_fix():
    # TypeError/ValueError, not AssertionError: `python -O` deletes an assert,
    # and each of these three guards a silent wrong answer -- a converted
    # carrier, a strided buffer the collective would materialise in an order
    # nobody checked, and a transport name nothing implements.
    h_local = abi.h_local(2)
    good = pack_send_carrier(_unique_local(2, 2 * h_local), 2)
    with pytest.raises(TypeError, match="int32"):
        exchange_carrier(good.to(torch.float32), world=2, rank=0)
    with pytest.raises(ValueError, match="contiguous"):
        exchange_carrier(good.transpose(1, 2), world=2, rank=0)
    with pytest.raises(ValueError, match="unknown transport"):
        exchange_carrier(good, world=2, rank=0, transport="ring")


def test_the_transport_names_are_closed():
    # Both arms return byte-identical results and differ only in how much they
    # move; a third name appearing without a gate is what this pins.
    assert carrier.TRANSPORTS == ("all_to_all", "all_gather")


@pytest.mark.parametrize("q", [1, 5])
def test_packing_into_a_retained_buffer_is_the_same_pack(q):
    """``out=`` must be the allocating pack's answer, in the caller's storage.

    Both halves are load-bearing. ``torch.equal`` alone would pass for a pack
    that allocated internally and copied; ``data_ptr`` alone would pass for a
    buffer that was returned unwritten. Q=5 is here for the same reason it is in
    the pack gates above: at Q=1 the destination-major permutation coincides
    with a reshape, so a Q=1-only check cannot see a pack writing the wrong
    order into the right buffer.
    """
    W, h_local = 2, abi.h_local(2)
    local = _unique_local(q, W * h_local)
    want = pack_send_carrier(local, W)

    buf = torch.zeros_like(want)
    first = pack_send_carrier(local, W, out=buf)
    assert first is buf and torch.equal(first, want)

    # A second call must reuse the same storage -- that is the whole claim --
    # and must fully overwrite it, which the poisoning proves.
    ptr = buf.data_ptr()
    buf.fill_(-12345)
    second = pack_send_carrier(local, W, out=buf)
    assert second.data_ptr() == ptr
    assert torch.equal(second, want), "the retained buffer was not overwritten"


def test_a_retained_pack_buffer_is_checked_like_every_other_carrier():
    # Same discipline as the send carrier itself: raised (never asserted), and
    # contiguity is checked because a strided destination would be written in an
    # order nobody verified.
    W, h_local = 2, abi.h_local(2)
    local = _unique_local(2, W * h_local)
    good = pack_send_carrier(local, W)
    with pytest.raises(ValueError, match="out must be"):
        pack_send_carrier(local, W, out=good[:, :1].contiguous())
    with pytest.raises(TypeError, match="int32"):
        pack_send_carrier(local, W, out=good.to(torch.float32))
    # The right shape and dtype, strided: the one form a shape check cannot see.
    strided = torch.zeros((*good.shape, 2), dtype=torch.int32)[..., 0]
    assert tuple(strided.shape) == tuple(good.shape) and not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        pack_send_carrier(local, W, out=strided)


def test_exchange_carrier_at_world_one_still_copies_into_a_retained_buffer():
    # The `out=` form must not turn the one non-collective case into an alias:
    # returning `send` itself would let a later in-place publish mutate the
    # merge's input, which is what the allocating form's `clone()` prevents.
    send = _unique_local(2, 4).reshape(1, 2, 4, CAND_K, 2).contiguous()
    recv = torch.zeros_like(send)
    got = exchange_carrier(send, world=1, rank=0, out=recv)
    assert got is recv and torch.equal(got, send)
    assert got.data_ptr() != send.data_ptr()
    with pytest.raises(ValueError, match="must not alias"):
        exchange_carrier(send, world=1, rank=0, out=send)


def test_a_carrier_workspace_is_not_resizable_and_says_so():
    # Its buffers' element counts are a pure function of (W, Qchunk, Hlocal) --
    # C4's capture requirement -- so a workspace built for another geometry is
    # named as such rather than surfacing as a shape error on whichever tensor
    # happened to be checked first.
    ws = carrier.allocate_carrier_workspace(world=2, qchunk=4, h_local=1, device="cpu")
    assert ws.gathered is None, "an all_to_all workspace stages nothing"
    assert tuple(ws.send.shape) == carrier.send_carrier_shape(2, 4, 1)
    assert tuple(ws.recv.shape) == carrier.recv_carrier_shape(2, 4, 1)
    send, recv, gathered = carrier._workspace_for(
        ws, world=2, qchunk=4, h_local=1, transport="all_to_all"
    )
    assert send is ws.send and recv is ws.recv and gathered is None
    with pytest.raises(ValueError, match="allocated for"):
        carrier._workspace_for(ws, world=2, qchunk=5, h_local=1, transport="all_to_all")
    with pytest.raises(ValueError, match="no `gathered` buffer"):
        carrier._workspace_for(ws, world=2, qchunk=4, h_local=1, transport="all_gather")
    staged = carrier.allocate_carrier_workspace(
        world=2, qchunk=4, h_local=1, device="cpu", transport="all_gather"
    )
    assert tuple(staged.gathered.shape) == (2, *staged.send.shape)


# --------------------------------------------------------------------------
# the transport. DISTRIBUTED.
# --------------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("T", [1, 5])
def test_the_head_directed_transport_matches_the_all_gather_reference(
    icp_group, icp, T
):
    """Byte for byte, at ``Qchunk > 1`` as well as at 1.

    ``all_to_all_single`` with the fixed per-peer split converts
    destination-major to source-major by definition; the all-gather arm gets
    there by moving W times the traffic and slicing. The only thing that can
    make them differ is the pack, which is why ``T = 5`` is here.
    """
    rank, world, dev = icp
    Hl = abi.h_local(world) if world in (2, 4) else 1
    Hg = world * Hl
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=T)
    send = pack_send_carrier(cand, world)

    directed = exchange_carrier(
        send, world=world, rank=rank, transport="all_to_all", group=icp_group
    )
    gathered = exchange_carrier(
        send, world=world, rank=rank, transport="all_gather", group=icp_group
    )
    assert all_agree(torch.equal(directed, gathered), icp_group), (
        f"rank {rank}: all_to_all and all_gather disagree at T={T}"
    )
    # This rank's own slab must arrive at source index `rank`, which is the one
    # placement a symmetric split cannot get wrong by accident.
    assert all_agree(torch.equal(directed[rank], send[rank]), icp_group)


@pytest.mark.distributed
@pytest.mark.gpu
def test_the_collective_path_matches_a_hand_written_pack_exchange_merge(icp_group, icp):
    """``collective_exchange_and_merge`` is the whole consumer-facing path."""
    from fmha_sm100.icp import collective_exchange_and_merge

    rank, world, dev = icp
    Hl = abi.h_local(world) if world in (2, 4) else 1
    Hg = world * Hl
    T = 4
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=41)
    want = reference(cand, icp_group, rank, world, Hl)

    out = torch.full((T, Hl, CAND_K), -7, dtype=torch.int32, device=dev)
    status = collective_exchange_and_merge(
        cand, world=world, rank=rank, out=out, group=icp_group
    )
    ok = int(status.item()) == 0 and torch.equal(out, want)
    assert all_agree(ok, icp_group), f"rank {rank}: status={status.item()}"


# --------------------------------------------------------------------------
# the retained carrier buffers. DISTRIBUTED.
# --------------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("transport", list(carrier.TRANSPORTS))
def test_a_retained_carrier_workspace_allocates_nothing_and_agrees(
    icp_group, icp, transport
):
    """57 layers used to be 114 allocations of buffers of a fixed size.

    Two claims, and neither implies the other. **Agreement**: the workspace
    route's output is the allocating route's output, for every layer -- the
    carriers are scratch that is overwritten in full, so retaining them cannot
    change an answer. **No allocation**: the buffers keep their addresses across
    calls and the allocator's byte count does not move, which is what says the
    retention is real rather than a fresh tensor copied into at the end.
    """
    from fmha_sm100.icp import allocate_carrier_workspace

    rank, world, dev = icp
    Hl = abi.h_local(world) if world in (2, 4) else 1
    Hg = world * Hl
    T, layers = 6, 4
    cands = [
        make_candidates(T, Hg, rank, dev, world=world, seed=300 + i)
        for i in range(layers)
    ]
    forced, n_ordinary = c3_planes(T, dev)

    # The allocating route, which is the default and stays the default.
    want = []
    for cand in cands:
        out = torch.full((T, Hl, CAND_K), -7, dtype=torch.int32, device=dev)
        status = collective_status(
            cand,
            world,
            rank,
            out,
            forced,
            n_ordinary,
            transport,
            icp_group,
            workspace=None,
        )
        assert int(status.item()) == 0
        want.append(out)

    ws = allocate_carrier_workspace(
        world=world, qchunk=T, h_local=Hl, device=dev, transport=transport
    )
    out = torch.full((T, Hl, CAND_K), -7, dtype=torch.int32, device=dev)
    status = torch.zeros(1, dtype=torch.int32, device=dev)
    ptrs = (ws.send.data_ptr(), ws.recv.data_ptr())

    def allocations() -> int:
        # CUMULATIVE, not live: a per-call carrier is freed before the call
        # returns, so `memory_allocated()` cannot see it.
        return torch.cuda.memory_stats(dev)["allocation.all.allocated"]

    # One warmed call per route before the measurement: the first collective on
    # a new buffer set may allocate inside NCCL, which is not this function's
    # doing, and the JIT'd merge loads on first use.
    for ws_arg in (ws, None):
        collective_status(
            cands[0],
            world,
            rank,
            out,
            forced,
            n_ordinary,
            transport,
            icp_group,
            workspace=ws_arg,
            status=status,
        )
    torch.cuda.synchronize()

    ok = True
    for cand, expect in zip(cands, want):
        collective_status(
            cand,
            world,
            rank,
            out,
            forced,
            n_ordinary,
            transport,
            icp_group,
            workspace=ws,
            status=status,
        )
        ok &= torch.equal(out, expect) and int(status.item()) == 0
        ok &= (ws.send.data_ptr(), ws.recv.data_ptr()) == ptrs

    # The allocation window holds the calls and NOTHING else: `torch.equal`,
    # `==` and `.item()` all allocate, and a window wide enough to include the
    # comparisons above would be measuring this test rather than the route.
    torch.cuda.synchronize()
    start = allocations()
    for cand in cands:
        collective_status(
            cand,
            world,
            rank,
            out,
            forced,
            n_ordinary,
            transport,
            icp_group,
            workspace=ws,
            status=status,
        )
    torch.cuda.synchronize()
    retained = allocations() - start

    start = allocations()
    for cand in cands:
        collective_status(
            cand,
            world,
            rank,
            out,
            forced,
            n_ordinary,
            transport,
            icp_group,
            workspace=None,
            status=status,
        )
    torch.cuda.synchronize()
    allocating = allocations() - start

    assert all_agree(ok, icp_group), (
        f"rank {rank}: the retained-workspace route disagreed with the "
        f"allocating route on transport {transport!r}, or moved its buffers"
    )
    assert all_agree(retained == 0, icp_group), (
        f"rank {rank}: {layers} calls through a retained workspace made "
        f"{retained} allocations on transport {transport!r}; every buffer they "
        "need was supplied, and the carriers' size is a pure function of "
        "(W, Qchunk, Hlocal)"
    )
    assert all_agree(allocating >= 2 * layers, icp_group), (
        f"rank {rank}: the allocating route made only {allocating} allocations "
        f"for {layers} calls on transport {transport!r}, so this comparison "
        "cannot tell the two routes apart"
    )
