"""The C4 carrier, its transport, and ``IcpExchange`` against ``merge_candidates``.

Three things are gated here.

**The carrier (CPU).** ``pack_send_carrier`` turns the producer's query-major
``[Qchunk, H_group, 16, 2]`` output into C4's destination-major int32
``[W, Qchunk, H_local, 16, 2]`` send carrier. It is a **real permutation**; the
``.view``-reshape that happens to work at ``Qchunk == 1`` is kept in the package
as ``carrier._wrong_pack_view_negative_control`` and is gated here in BOTH
directions -- it must differ at ``Qchunk > 1`` and agree at ``Qchunk == 1``,
because a control that cannot fire and a control that always fires are equally
worthless. C4 names this explicitly: "Q>1 requires actual packing or direct
emission, not a token/head reshape", and "Q>=2 at BOTH W=2 and W=4 is the gate
that catches a view".

**The transport (distributed).** ``all_to_all`` against the ``all_gather`` +
head-slice reference, byte for byte.

**The fused exchange (distributed).** K5 publishes into a peer's symmetric
window, hands off with a ``.sys``-scope flag (or a generation tag inside the
payload) and merges what it finds; K2 merges the exchanged carrier. They must
produce **bit-identical** output, and that can be a ``torch.equal`` rather than
an "equivalent selection" check only because ``k5_exchange.cu`` and
``k2_merge.cu`` compile the same ``merge_topk.cuh``.

WHAT refined-icp-v1 CHANGED HERE
--------------------------------

* **The carrier is int32**, and ``IcpExchange`` takes int32 ``cand`` and
  allocates an int32 symmetric window. The wire content is unchanged -- the same
  two bitcast words -- so nothing moves in size; what changed is that a float32
  buffer is now *refused* instead of accepted, because accepting it lets a
  caller convert where it must bitcast, which silently corrupts every block id
  above ``2**24``.
* **Global block ids are no longer distinct across ranks.** ``make_candidates``
  used to draw them as ``local * world + rank``, i.e. one owner per block, and
  said so; under fragment placement every rank publishes a partial maximum for
  every block, so the *same* id arrives from several sources and the merge
  max-reduces duplicates by id before truncating. The helper now draws from one
  shared domain, and the tests that depended on disjointness say what they
  assert now.

Run::

    torchrun --nproc_per_node=2 -m pytest tests/test_exchange.py -v

Every rank must get the **whole** ``CUDA_VISIBLE_DEVICES`` list; per-rank
masking makes every rank report CUDA ordinal 0 and symmetric-memory rendezvous
fails. See ``tests/conftest.py::icp_group``.

On a workstation (``world_size < 2``) the distributed tests skip with that
reason printed, so ``pytest`` is still green -- the CPU-side carrier gates here
and the C5 gates in ``test_merge_key.py`` are the part that runs everywhere.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest
import torch
import torch.distributed as dist

import fmha_sm100.icp
from fmha_sm100.icp import (
    IcpExchange,
    MaskPreset,
    abi,
    carrier,
)
from fmha_sm100.icp import exchange as exchange_module
from fmha_sm100.icp import (
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


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


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


PRESETS = [MaskPreset.HANDSHAKE, MaskPreset.LAMPORT]

#: Every mask the kernel instantiates, presets *and* the control. Mask 0 is not
#: a preset -- it is the arm every optimisation is measured against and the arm
#: a bring-up bisects with -- so it is gated as a raw integer rather than
#: promoted into ``MaskPreset``. Gating it here is what makes "SASS-identical to
#: the pre-``opts`` kernel" a claim about something that runs.
MASKS = [0, MaskPreset.HANDSHAKE, MaskPreset.LAMPORT]


#: The masks a SHORT extent may run under: their handshake belongs to the same
#: block that derived the generation (mask 0's per-block release) or to the word
#: itself (512's tag). ``MaskPreset.HANDSHAKE`` is absent on purpose --
#: ``OPT_HIER`` publishes one release for the whole grid, so a block a short
#: launch skips falls a generation behind and the flag can then satisfy some
#: blocks of a later full launch and not others. It is refused by name, and
#: ``test_a_short_extent_is_refused_under_the_grid_wide_handshake`` is that
#: refusal's gate.
EXTENT_MASKS = [0, MaskPreset.LAMPORT]


def _mask_id(mask) -> str:
    return getattr(mask, "name", None) or f"mask{int(mask)}"


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
# the fused exchange. DISTRIBUTED.
# --------------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", MASKS, ids=_mask_id)
@pytest.mark.parametrize("T,Hl", [(1, 1), (8, 1), (64, 2)])
def test_exchange_is_bit_identical_to_the_reference_merge(
    icp_group, icp, preset, T, Hl
):
    rank, world, dev = icp
    Hg = world * Hl
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=T + Hl)
    want = reference(cand, icp_group, rank, world, Hl)

    with IcpExchange(
        icp_group,
        tokens=T,
        heads_group=Hg,
        heads_local=Hl,
        slots=4,
        preset=preset,
        device=dev,
    ) as ex:
        got = ex.exchange_and_merge(cand, layer_idx=0)
        ex.check_error()

    assert all_agree(torch.equal(got, want), icp_group), (
        f"rank {rank}: fused exchange != pack + exchange + merge at mask "
        f"{_mask_id(preset)}, T={T}, H_local={Hl}\n got={got}\nwant={want}"
    )


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", MASKS, ids=_mask_id)
@pytest.mark.parametrize("T,Hl", [(1, 1), (8, 1), (64, 2)])
def test_exchange_under_c3_forcing_is_bit_identical_to_the_reference_merge(
    icp_group, icp, preset, T, Hl
):
    """K5 against K2 with the C3 planes populated, on peers rather than on self.

    The arm above compares the two with ``forced=None``, which was the only
    thing expressible while K5 reached the merge through the four-argument
    compatibility overload. It no longer is, and this is the comparison that
    became possible.

    **Why it is not redundant with ``test_merge_key.py``'s C3 arms.** Those gate
    K5 and K2 separately, each against the same Python ``dedup_then_truncate``.
    Two kernels agreeing with one oracle is not the same claim as two kernels
    agreeing with each other: a misreading of C3 shared by the oracle and both
    kernels passes every one of those tests and fails this one. This arm has no
    oracle -- it is ``torch.equal`` between two independent implementations on a
    carrier neither of them chose.

    It also puts forcing on the **transport**. The C3 arms in the other module
    build a carrier in which every source publishes the same records; here the
    candidates are drawn per rank and cross the wire, so the reserved slot is
    exercised on rows whose pool arrived from peers.
    """
    rank, world, dev = icp
    Hg = world * Hl
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=T + Hl + 17)
    forced, n_ordinary = c3_planes(T, dev)

    want = reference(
        cand, icp_group, rank, world, Hl, forced=forced, n_ordinary=n_ordinary
    )

    # PRECONDITION: forcing must actually change this carrier's answer. If the
    # forced blocks happened to be exactly what the ordinary merge selected, a
    # `torch.equal` here would hold for a K5 that ignored the planes entirely --
    # which is the defect this whole lane is about.
    plain = reference(cand, icp_group, rank, world, Hl)
    assert all_agree(not torch.equal(want, plain), icp_group), (
        f"rank {rank}: the C3 planes did not change the reference merge's "
        f"answer at T={T}, H_local={Hl}, so this case cannot tell a kernel "
        "that honours the reserved slot from one that drops it"
    )

    with IcpExchange(
        icp_group,
        tokens=T,
        heads_group=Hg,
        heads_local=Hl,
        slots=4,
        preset=preset,
        device=dev,
    ) as ex:
        got = ex.exchange_and_merge(
            cand, layer_idx=0, forced=forced, n_ordinary=n_ordinary
        )
        ex.check_error()

    assert all_agree(torch.equal(got, want), icp_group), (
        f"rank {rank}: fused exchange under C3 forcing != pack + exchange + "
        f"merge at mask {_mask_id(preset)}, T={T}, H_local={Hl}\n"
        f" got={got}\nwant={want}"
    )
    # ... and the forced block is in every active row of K5's own output, which
    # `torch.equal` alone would not say if BOTH kernels dropped it.
    host_forced = forced.cpu().tolist()
    host_got = got.cpu()
    for t in range(T):
        if host_forced[t] < 0:
            continue
        for hl in range(Hl):
            assert host_forced[t] in host_got[t, hl].tolist(), (
                f"rank {rank}: row (t={t}, hl={hl}) is missing its forced "
                f"block {host_forced[t]}"
            )


@pytest.mark.distributed
@pytest.mark.gpu
def test_duplicate_ids_across_sources_stay_bit_identical(icp_group, icp):
    """Duplicate global ids, which are now the NORMAL case.

    USED TO ASSERT that duplicates were illegal under C4 and to inject them
    *within* one rank's list as a robustness case. That is backwards now: under
    fragment placement every rank publishes a partial maximum for every block,
    so the same id arriving from several **sources** is what production looks
    like, and the merge max-reduces by id before truncating. What is still true
    is that a duplicate makes the selection genuinely order sensitive -- two
    records with one id and different scores -- so K5 and K2 agreeing is a real
    constraint on the two loaders electing the same representative, not a
    tautology.

    The id domain is squeezed to 32 blocks so overlap is guaranteed rather than
    hoped for, and the overlap is asserted before anything else is.
    """
    rank, world, dev = icp
    T, Hl = 8, 1
    Hg = world * Hl
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=23, domain=32)

    recv = exchange_carrier(
        pack_send_carrier(cand, world),
        world=world,
        rank=rank,
        transport="all_gather",
        group=icp_group,
    )
    ids = recv[..., 1].cpu()  # [W, T, Hl, 16]
    duplicated = 0
    for t in range(T):
        for h in range(Hl):
            flat = [x for x in ids[:, t, h].flatten().tolist() if x >= 0]
            duplicated += len(flat) - len(set(flat))
    # PRECONDITION: no duplicate means this test is the plain-merge test again.
    assert all_agree(duplicated > 0, icp_group), (
        f"rank {rank}: the {world}-source pool holds no duplicate global id, so "
        "this case cannot exercise the max-by-id reduction at all"
    )

    want = reference(cand, icp_group, rank, world, Hl)
    with IcpExchange(
        icp_group,
        tokens=T,
        heads_group=Hg,
        heads_local=Hl,
        slots=4,
        preset=MaskPreset.HANDSHAKE,
        device=dev,
    ) as ex:
        got = ex.exchange_and_merge(cand, layer_idx=0)
        ex.check_error()

    assert all_agree(torch.equal(got, want), icp_group), (
        f"rank {rank}: K5 and K2 disagree on duplicate ids\n got={got}\nwant={want}"
    )


@pytest.mark.distributed
@pytest.mark.gpu
def test_output_contract(icp_group, icp):
    """C6: ``[T, H_local, 16]`` int32, ascending, ``-1`` tail, no duplicates.

    USED TO ASSERT ``len(valid) == min(16, 12 * world)``, i.e. that the pool's
    size was the sum of the per-rank valid counts. That arithmetic came from
    one-owner placement, where the ranks' id sets were disjoint by construction.
    Under fragment placement the pool is a **union**, so the depth is the number
    of DISTINCT valid ids, capped at 16 -- and the difference between the two is
    exactly what a merge without duplicate reduction would get wrong.
    """
    rank, world, dev = icp
    T, Hl = 16, 1
    Hg = world * Hl
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=3, domain=48)
    # Kill part of every row so the -1 tail is exercised rather than assumed:
    # C4 pins an invalid entry at (-inf, -1) and requires every rank to publish
    # the full shape anyway, including ranks with no valid blocks at all.
    cand[:, :, 12:, 0] = NEG_INF_BITS
    cand[:, :, 12:, 1] = -1
    # Every fourth row keeps only 3 valid candidates, so that the union across
    # the ranks is SHORTER than 16 there and longer than 16 elsewhere. Both
    # sides of `min(16, distinct)` then occur in one launch; with 12 everywhere
    # the depth would be 16 in every row and the union arithmetic below would be
    # indistinguishable from the constant 16.
    cand[::4, :, 3:, 0] = NEG_INF_BITS
    cand[::4, :, 3:, 1] = -1

    recv = exchange_carrier(
        pack_send_carrier(cand, world),
        world=world,
        rank=rank,
        transport="all_gather",
        group=icp_group,
    )
    pool_ids = recv[..., 1].cpu()

    with IcpExchange(
        icp_group, tokens=T, heads_group=Hg, heads_local=Hl, slots=4, device=dev
    ) as ex:
        out = ex.exchange_and_merge(cand)
        ex.check_error()

    assert out.shape == (T, Hl, CAND_K)
    assert out.dtype == torch.int32
    ok = True
    saw_capped, saw_short = False, False
    host = out.cpu()
    for t in range(T):
        for h in range(Hl):
            row = host[t, h].tolist()
            valid = [x for x in row if x >= 0]
            ok &= row[: len(valid)] == valid  # -1s form a tail
            ok &= valid == sorted(valid)  # ascending
            ok &= len(set(valid)) == len(valid)  # no duplicates
            distinct = {x for x in pool_ids[:, t, h].flatten().tolist() if x >= 0}
            ok &= len(valid) == min(CAND_K, len(distinct))
            ok &= set(valid) <= distinct  # nothing invented
            saw_capped |= len(distinct) >= CAND_K
            saw_short |= len(distinct) < CAND_K
    assert all_agree(ok, icp_group), f"rank {rank}: C6 violated in {host}"
    # ANTI-VACUITY: both sides of `min(16, distinct)` must actually occur --
    # a row whose union fills all 16 slots and a row whose union cannot -- or
    # the depth assertion above is only ever checking one of them.
    assert all_agree(saw_capped and saw_short, icp_group), (
        f"rank {rank}: capped={saw_capped} short={saw_short}; the pool depths "
        "this case produces cannot exercise both sides of min(16, distinct)"
    )


@pytest.mark.distributed
@pytest.mark.gpu
def test_negative_control_perturbing_one_score_changes_the_output(icp_group, icp):
    """A control that must FIRE.

    One site is perturbed: a single candidate's *score*, on every rank, in a
    head that rank owns. Everything else -- ids, geometry, the head window -- is
    left consistent. (CONTRACT C10 rejects "use a wrong ``owner()``" as a
    control precisely because a consistent repartition provably cannot change
    the answer.)

    ``+inf`` is used here as what it is -- the top of the C5 score order, a
    score a producer may legitimately emit -- and NOT as a forcing mechanism:
    refined-icp-v1 C3 removed +inf forcing, and this path carries no forced slot
    at all (K5 still calls the pre-S7 merge overload).

    The candidate is checked to be *absent* from the unperturbed output first,
    so a control that passed for the wrong reason -- because the id was already
    selected -- cannot masquerade as a working one.
    """
    rank, world, dev = icp
    T, Hl = 8, 1
    Hg = world * Hl
    mine = rank * Hl  # a head index this rank's output covers
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=11)

    # A distinct, valid, definitely-losing candidate. The marker is outside the
    # id domain `make_candidates` draws from, so no peer can publish it and the
    # duplicate reduction cannot resurrect it.
    marker = (1 << 20) + rank
    cand[:, mine, 15, 0] = score_bits(-1e30)
    cand[:, mine, 15, 1] = marker

    with IcpExchange(
        icp_group, tokens=T, heads_group=Hg, heads_local=Hl, slots=4, device=dev
    ) as ex:
        base = ex.exchange_and_merge(cand).clone()
        absent = bool((base[:, 0] != marker).all().item())
        assert all_agree(absent, icp_group), (
            f"rank {rank}: the control's marker {marker} was selected before "
            "it was perturbed, so this control would pass vacuously"
        )

        cand[:, mine, 15, 0] = score_bits(INF)
        perturbed = ex.exchange_and_merge(cand, layer_idx=1)
        ex.check_error()

    fired = bool((perturbed[:, 0] == marker).any(dim=-1).all().item())
    changed = not torch.equal(base, perturbed)
    assert all_agree(fired and changed, icp_group), (
        f"rank {rank}: perturbing one candidate's score did not change the "
        f"merged output -- the gate is vacuous.\nbase={base}\n"
        f"perturbed={perturbed}"
    )


@pytest.mark.distributed
@pytest.mark.gpu
def test_per_layer_slots_stay_bit_identical(icp_group, icp):
    """The shipping configuration: one slot per layer, ack off.

    At ``slots >= 2`` the consumer ack is dropped, which is where most of the
    exchange's win is. Its safety is a stream-order argument, not a flag, so
    the thing to gate is that a *sequence* of layers -- rotating through the
    slots more than once, with different data per layer -- still agrees with
    the reference at every step.
    """
    rank, world, dev = icp
    T, Hl, slots, layers = 8, 1, 4, 9
    Hg = world * Hl
    ok = True
    with IcpExchange(
        icp_group,
        tokens=T,
        heads_group=Hg,
        heads_local=Hl,
        slots=slots,
        preset=MaskPreset.HANDSHAKE,
        device=dev,
    ) as ex:
        assert not ex.use_ack, "slots >= 2 must turn the ack off"
        for layer in range(layers):
            cand = make_candidates(T, Hg, rank, dev, world=world, seed=100 + layer)
            got = ex.exchange_and_merge(cand, layer_idx=layer)
            want = reference(cand, icp_group, rank, world, Hl)
            ok &= torch.equal(got, want)
        ex.check_error()
    assert all_agree(ok, icp_group), (
        f"rank {rank}: a layer disagreed with the reference while rotating "
        f"{layers} layers through {slots} slots"
    )


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", PRESETS, ids=_mask_id)
def test_single_slot_turns_the_ack_back_on(icp_group, icp, preset):
    """``use_ack`` is ``slots < ACK_FREE_SLOTS``, at EVERY mask including LAMPORT.

    LAMPORT is parametrised rather than skipped because it deletes the publish
    flag (control kind 0) and leaves the consumer ack (kind 1) untouched, so
    ``slots < 2 and not LAMPORT`` would turn the ack off in the one
    configuration where it is the only protection against a rank a launch ahead
    overwriting a slot its peer has not read.
    """
    rank, world, dev = icp
    T, Hl = 4, 1
    Hg = world * Hl
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=5)
    with IcpExchange(
        icp_group,
        tokens=T,
        heads_group=Hg,
        heads_local=Hl,
        slots=1,
        preset=preset,
        device=dev,
    ) as ex:
        assert ex.use_ack, (
            "at slots == 1 the ack is the only thing stopping a rank one "
            f"launch ahead from overwriting a slot its peer has not read "
            f"(mask {_mask_id(preset)})"
        )
        got = ex.exchange_and_merge(cand)
        ex.check_error()
    want = reference(cand, icp_group, rank, world, Hl)
    assert all_agree(torch.equal(got, want), icp_group)


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", MASKS, ids=_mask_id)
def test_a_failed_launch_fills_its_rows_with_block_zero(icp_group, icp, preset):
    """The failure contract, and it INVERTS the one this suite used to assert.

    The pre-H2 kernel wrote **no** output row on a failing launch, so a timeout
    gate could assert ``out`` was untouched. That is safe as a test contract and
    unsafe as a runtime one: ``out`` is consumed by attention in the same step,
    before the host polls the error flag, and an untouched row reaching a
    block-table load holds either uninitialised memory or a previous batch's
    ids. The kernel now fills the failing block's rows with global block id
    **0** -- in range for every row that is read at all, where ``-1`` would
    dereference one element before the row.

    This is also where K5 and K2 are known to DISAGREE: K2 publishes a failed
    row as all ``-1`` (``test_merge_key.py``), K5 fills it with 0. Reconciling
    the two renderings is an open ABI decision, recorded in ``docs/STATUS.md``;
    the bit-identity tests above therefore compare the two only on the success
    path.

    Forcing the failure through the **entry-time error check** rather than
    through a spin timeout is deliberate, twice over:

    * it is collective. Every rank pre-sets its own flag, so no rank runs ahead
      of another and the session-scoped group is still aligned for the next
      test. A skew-induced timeout would leave the job wedged.
    * it gates the entry check itself, which otherwise has no gate at all. That
      check is what turns 57 layers x a one-second spin into one, and it is
      invisible on the healthy path.

    ``out`` is pre-filled with a sentinel so this cannot pass by the kernel
    writing nothing to an already-zero buffer.
    """
    rank, world, dev = icp
    T, Hl = 8, 1
    Hg = world * Hl
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=31)
    out = torch.full((T, Hl, CAND_K), -7, dtype=torch.int32, device=dev)

    with IcpExchange(
        icp_group,
        tokens=T,
        heads_group=Hg,
        heads_local=Hl,
        slots=4,
        preset=preset,
        device=dev,
    ) as ex:
        # Every rank, collectively: this workspace has already failed.
        ex._err.fill_(1)
        torch.cuda.synchronize()
        dist.barrier(group=icp_group)

        ex.exchange_and_merge(cand, out=out, layer_idx=0)
        torch.cuda.synchronize()

        filled = bool((out == 0).all().item())
        assert all_agree(filled, icp_group), (
            f"rank {rank}: a launch on a failed workspace left "
            f"{int((out != 0).sum().item())} of {out.numel()} output entries "
            f"un-filled at mask {_mask_id(preset)}; the failure contract is "
            f"block id 0 in every row of the failing blocks, not -1 and not "
            f"untouched.\nout={out}"
        )
        with pytest.raises(RuntimeError, match="k5 exchange failed"):
            ex.check_error()


@pytest.mark.distributed
@pytest.mark.gpu
def test_lamport_matches_the_reference(icp_group, icp):
    """LAMPORT at whatever degree the group has.

    This was skipped above C=2 while the harness's C=4 run was outstanding.
    That run has since landed and the mask has since been gated on a second
    architecture: mask 512 is 207/207 at C=2 and C=4 on sm_107a plus a
    10 000 x 2 cudagraph replay soak, and on GB300 sm_103a it passes 11 gates x
    3 masks {0, 12, 512} x 2 world sizes with the stale-producer poison firing
    at every mask. The kernel carries no C-degree restriction -- the mask checks
    reject bit combinations and unbuilt masks, never a world size. Gating at
    C=2 only was therefore stricter than the evidence, and a skip that outlives
    its reason is how coverage quietly disappears.

    C=8 remains unmeasured at every mask.
    """
    rank, world, dev = icp
    T, Hl = 8, 1
    Hg = world * Hl
    cand = make_candidates(T, Hg, rank, dev, world=world, seed=17)
    want = reference(cand, icp_group, rank, world, Hl)
    with IcpExchange(
        icp_group,
        tokens=T,
        heads_group=Hg,
        heads_local=Hl,
        slots=4,
        preset=MaskPreset.LAMPORT,
        device=dev,
    ) as ex:
        # 2x window: the tag rides in the upper half of every 64-bit word.
        assert ex.slot_floats == 2 * T * Hg * CAND_K * 2
        got = ex.exchange_and_merge(cand)
        ex.check_error()
    assert all_agree(torch.equal(got, want), icp_group)


# --------------------------------------------------------------------------
# the execution extent. DISTRIBUTED.
# --------------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", EXTENT_MASKS, ids=_mask_id)
@pytest.mark.parametrize("extent", [1, 8, 33])
def test_a_short_extent_is_the_full_calls_answer_on_the_rows_it_shares(
    icp_group, icp, preset, extent
):
    """One window at capacity, a call over a prefix of it, and the SAME answer.

    This is the whole point of separating the allocation from the extent: the
    receive window's strides come from the capacity, so a short call writes and
    polls exactly the words a full call would have written and polled for those
    rows. If it did not -- if the strides moved with the extent -- the short
    call would still produce a *plausible* full row from the wrong offsets,
    which is why this compares bit patterns against two independent oracles
    rather than checking that the output looks well formed:

    * the SAME workspace's full-capacity call, which shares every stride with
      it, and
    * the K2 reference merge over the prefix, which shares nothing with it.

    ``extent=33`` is not a multiple of ``tpb``: the last live block is partial,
    which is the case where a row inside a launched block is still past the
    extent.
    """
    rank, world, dev = icp
    Hl = abi.h_local(world) if world in (2, 4) else 1
    Hg = world * Hl
    T_cap = 64
    cand = make_candidates(T_cap, Hg, rank, dev, world=world, seed=extent + 3)
    # A prefix VIEW, not a copy: `cand[:n]` is contiguous, allocates nothing,
    # and is what a caller holding one capacity buffer actually passes.
    prefix = cand[:extent]
    assert prefix.data_ptr() == cand.data_ptr() and prefix.is_contiguous()

    want = reference(prefix, icp_group, rank, world, Hl)

    with IcpExchange(
        icp_group,
        tokens=T_cap,
        heads_group=Hg,
        heads_local=Hl,
        slots=4,
        preset=preset,
        device=dev,
    ) as ex:
        assert ex.capacity == T_cap
        # Neither of these moves with the extent -- they are the workspace's
        # geometry, and the generation counters are addressed through them.
        tpb, nblocks = ex.tpb, ex.nblocks
        full = ex.exchange_and_merge(cand, layer_idx=0).clone()
        short = ex.exchange_and_merge(prefix, num_tokens=extent, layer_idx=1)
        ex.check_error()
        assert (ex.tpb, ex.nblocks) == (tpb, nblocks)

    ok = (
        tuple(short.shape) == (extent, Hl, CAND_K)
        and torch.equal(short, full[:extent])
        and torch.equal(short, want)
    )
    assert all_agree(ok, icp_group), (
        f"rank {rank}: a {extent}-row call on a {T_cap}-row window disagreed "
        f"at mask {_mask_id(preset)}\nshort={short}\nfull[:{extent}]="
        f"{full[:extent]}\nwant={want}"
    )


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", EXTENT_MASKS, ids=_mask_id)
def test_interleaved_extents_keep_every_generation_monotone(icp_group, icp, preset):
    """Short and long calls ALTERNATING on one workspace, which is the point.

    A short launch runs fewer blocks, so the blocks covering the rows past its
    extent do not run at all: their generation counters (control kind 2) do not
    advance and the tags sitting in those rows' payload words stay at the value
    the last long launch wrote. The next long launch therefore expects
    ``counter + 1`` for those rows and finds a strictly older tag until its peer
    publishes -- which is exactly what has to be true, because OPT_LAMPORT's tag
    compare is ``==``: a stale tag that happened to equal the expected one would
    be accepted as arrived and the merge would run on the previous step's
    candidates. That failure is SILENT -- a well formed row of plausible block
    ids -- so it cannot be gated by looking at the output alone; it is gated
    here by driving the interleaving and comparing every call, long and short,
    against the reference.

    What makes it safe is that ``tpb`` comes from the capacity, so the row ->
    block map is the same in both launches and one block owns one row's words
    forever. A partition re-planned per extent would put two independent
    counters on one word.

    ``T_cap = 64`` with ``H_local = 1`` plans ``tpb = 8``, so the capacity grid
    is 8 blocks and the 8-row call launches exactly one: seven blocks are
    skipped per short launch, repeatedly.
    """
    rank, world, dev = icp
    Hl, T_cap, short = 1, 64, 8
    Hg = world * Hl
    ok = True
    with IcpExchange(
        icp_group,
        tokens=T_cap,
        heads_group=Hg,
        heads_local=Hl,
        slots=4,
        preset=preset,
        device=dev,
    ) as ex:
        assert not ex.use_ack, "slots=4 must leave the ack off (ACK_FREE_SLOTS)"
        assert ex.nblocks > 1, (
            f"this case needs a multi-block capacity grid to skip any blocks; "
            f"T_cap={T_cap}, H_local={Hl} planned nblocks={ex.nblocks}"
        )
        for layer in range(8):
            extent = T_cap if layer % 2 else short
            cand = make_candidates(extent, Hg, rank, dev, world=world, seed=200 + layer)
            got = ex.exchange_and_merge(cand, layer_idx=layer, num_tokens=extent)
            want = reference(cand, icp_group, rank, world, Hl)
            ok &= torch.equal(got, want)
        ex.check_error()
    assert all_agree(ok, icp_group), (
        f"rank {rank}: alternating {short}-row and {T_cap}-row calls on one "
        f"window disagreed with the reference at mask {_mask_id(preset)}"
    )


#: ``kFlagStride`` and ``kCtlKinds`` from ``csrc/k5_exchange.cu``, and its
#: ``ctl_off`` addressing. Mirrored here rather than exported because the two
#: tests below are the only readers and what they are gating IS the address: a
#: helper on the class would compute the same wrong number as the kernel if the
#: kernel's changed.
CTL_FLAG_STRIDE = 32
CTL_KIND_GENERATION = 2


def ctl_index(
    kind: int, slot: int, slots: int, b: int, nblocks: int, s: int, world: int
) -> int:
    return (((kind * slots + slot) * nblocks + b) * world + s) * CTL_FLAG_STRIDE


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", EXTENT_MASKS, ids=_mask_id)
def test_a_short_call_writes_only_the_prefix_of_its_slot(icp_group, icp, preset):
    """WHERE a short call writes, asserted on the window itself.

    The bit-identity gates above cannot see this. A kernel that took its
    receive-window strides from the EXTENT instead of the capacity is
    self-consistent within one launch -- every rank publishes and polls at the
    same wrong offsets -- so it returns the right answer and quietly scribbles
    into another slot's rows. What that costs is paid later and at random: the
    next launch to use the trampled slot finds a tag some other launch wrote,
    and OPT_LAMPORT's compare is exact, so if the two happen to be equal it
    merges stale candidates instead of waiting. That is a race, and a race is
    not something a gate can be built on. The address is.

    So this reads the symmetric window directly: a call at extent ``n`` on slot
    ``s`` must leave every word outside ``[s][:, :n]`` exactly as it found it.
    """
    rank, world, dev = icp
    Hl, T_cap, extent, slots, slot = 1, 64, 8, 4, 2
    Hg = world * Hl
    cand = make_candidates(extent, Hg, rank, dev, world=world, seed=71)

    with IcpExchange(
        icp_group,
        tokens=T_cap,
        heads_group=Hg,
        heads_local=Hl,
        slots=slots,
        preset=preset,
        device=dev,
    ) as ex:
        # `symm_mem.empty` does not zero; `__init__` does, and this is before
        # the first launch, so the whole window is a known value.
        before = ex._data.clone()
        assert not before.any(), "the window must start zeroed"
        # int32 words per candidate: OPT_LAMPORT carries its generation tag in
        # the upper half of every 64-bit payload word, hence 4 rather than 2.
        wpc = 4 if int(preset) & exchange_module.OPT_LAMPORT else 2
        assert ex.slot_floats == T_cap * Hg * CAND_K * wpc

        ex.exchange_and_merge(cand, num_tokens=extent, layer_idx=slot)
        torch.cuda.synchronize()
        ex.check_error()

        # [slots][source plane][token row][H_local][16][words]: the layout the
        # kernel's own offsets describe, taken from the CAPACITY.
        window = ex._data.view(slots, world, T_cap, Hl, CAND_K, wpc)
        touched = window != before.view_as(window)
        outside_slot = touched[[s for s in range(slots) if s != slot]]
        past_extent = touched[slot][:, extent:]
        inside = touched[slot][:, :extent]

    ok = (
        not bool(outside_slot.any())
        and not bool(past_extent.any())
        and bool(inside.any())
    )
    assert all_agree(ok, icp_group), (
        f"rank {rank}: a {extent}-row call on slot {slot} of a {T_cap}-row "
        f"window at mask {_mask_id(preset)} touched "
        f"{int(outside_slot.sum())} words of another slot and "
        f"{int(past_extent.sum())} words past its own extent "
        f"({int(inside.sum())} inside it). The window's strides must come from "
        "the allocation capacity, not from the extent."
    )


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", EXTENT_MASKS, ids=_mask_id)
def test_the_extent_does_not_move_a_blocks_generation_counter(icp_group, icp, preset):
    """One block, one counter, one word -- whatever the extent.

    Each block derives its generation as ``own_generation + 1`` from its own
    control word, which is what makes the tag on a data word strictly
    increasing and therefore makes OPT_LAMPORT's exact compare safe. That holds
    only while the block that owns a row owns it at *every* extent, so ``tpb``
    and ``nblocks`` are planned from the capacity and never from the call.

    The failure this prevents is, again, a race -- two counters writing one
    word, which collide only sometimes -- so what is gated is the address and
    not the collision: block 0's counter must be the SAME word for a short call
    and a long one, and must therefore read 2 after one of each.

    A 64-row capacity at ``H_local = 1`` plans ``tpb = 8`` and an 8-block grid,
    so the 8-row call launches one block and skips seven: the whole point of
    the arrangement.
    """
    rank, world, dev = icp
    Hl, T_cap, extent, slots = 1, 64, 8, 4
    Hg = world * Hl

    with IcpExchange(
        icp_group,
        tokens=T_cap,
        heads_group=Hg,
        heads_local=Hl,
        slots=slots,
        preset=preset,
        device=dev,
    ) as ex:
        assert ex.nblocks > 1, "this case needs a multi-block capacity grid"
        gen = ctl_index(CTL_KIND_GENERATION, 0, slots, 0, ex.nblocks, 0, world)
        assert int(ex.control_block[gen]) == 0

        short = make_candidates(extent, Hg, rank, dev, world=world, seed=81)
        ex.exchange_and_merge(short, num_tokens=extent, layer_idx=0)
        torch.cuda.synchronize()
        after_short = int(ex.control_block[gen])

        # Slots 1..3, so that slot 0's next use is not a consecutive reuse --
        # the discipline this package's `slots` argument exists for.
        for layer in (1, 2, 3):
            ex.exchange_and_merge(
                make_candidates(T_cap, Hg, rank, dev, world=world, seed=90 + layer),
                layer_idx=layer,
            )
        full = make_candidates(T_cap, Hg, rank, dev, world=world, seed=82)
        got = ex.exchange_and_merge(full, layer_idx=4)  # slot 0 again
        torch.cuda.synchronize()
        after_full = int(ex.control_block[gen])
        ex.check_error()
        want = reference(full, icp_group, rank, world, Hl)
        agrees = torch.equal(got, want)

    assert all_agree(after_short == 1, icp_group), (
        f"rank {rank}: after an {extent}-row call on slot 0, block 0's "
        f"generation counter at the CAPACITY address reads {after_short}, not "
        "1. The short call advanced a different word, so the row -> block map "
        f"moved with the extent (mask {_mask_id(preset)})."
    )
    assert all_agree(after_full == 2, icp_group), (
        f"rank {rank}: after a short and a full call on slot 0, block 0's "
        f"generation counter reads {after_full}, not 2 -- the two calls did "
        f"not share a counter (mask {_mask_id(preset)})."
    )
    assert all_agree(agrees, icp_group), (
        f"rank {rank}: the full call after a short one disagreed with the "
        f"reference at mask {_mask_id(preset)}"
    )


@pytest.mark.distributed
@pytest.mark.gpu
def test_a_short_extent_is_refused_under_the_grid_wide_handshake(icp_group, icp):
    """``MaskPreset.HANDSHAKE`` sets ``OPT_HIER``, and OPT_HIER cannot do this.

    Its release is ONE flag for the whole grid, carrying whichever block arrived
    last, and every block acquires on it. That needs all the grid's blocks to be
    on the same generation, and a block skipped by a short launch is not: it
    keeps its own counter, so a later full launch on the same slot has blocks
    expecting different values from one flag. The acquire compares ``>=``, so
    the blocks below the published generation pass instantly on another block's
    flag and the ones above it spin to their timeout -- decided by arrival
    order, which is why it is refused rather than left to a gate that would
    catch it only sometimes.

    **This refusal was found by the counter gate above, not reasoned out first:**
    mask 12 failed it with a real spin timeout at the second call on slot 0.

    The refusal is raised in Python *and* by the kernel; the Python one exists
    because under a capture the kernel's first chance to speak may be the first
    captured launch.
    """
    rank, world, dev = icp
    Hl, T_cap = 1, 64
    Hg = world * Hl
    cand = make_candidates(8, Hg, rank, dev, world=world, seed=91)
    with IcpExchange(
        icp_group,
        tokens=T_cap,
        heads_group=Hg,
        heads_local=Hl,
        slots=4,
        preset=MaskPreset.HANDSHAKE,
        device=dev,
    ) as ex:
        with pytest.raises(ValueError, match="OPT_HIER"):
            ex.exchange_and_merge(cand, num_tokens=8)
        # The full extent is still fine, and the workspace is still alive after
        # the refusal -- nothing was launched.
        got = ex.exchange_and_merge(
            make_candidates(T_cap, Hg, rank, dev, world=world, seed=92)
        )
        ex.check_error()
    assert all_agree(tuple(got.shape) == (T_cap, Hl, CAND_K), icp_group)


@pytest.mark.distributed
@pytest.mark.gpu
def test_an_extent_larger_than_the_allocation_raises(icp_group, icp):
    """Refused, never clamped, and refused before anything is launched.

    A clamp would merge the rows that fit and leave the rest of ``out`` holding
    whatever was there -- on a decode path, a previous batch's block ids, which
    a block-table load dereferences as real pages of another request's KV. The
    capacity and the extent are both host integers at this point, and this is
    the last place anything compares them.
    """
    rank, world, dev = icp
    Hl = 1
    Hg = world * Hl
    T_cap = 8
    with IcpExchange(
        icp_group, tokens=T_cap, heads_group=Hg, heads_local=Hl, slots=4, device=dev
    ) as ex:
        big = make_candidates(T_cap + 1, Hg, rank, dev, world=world)
        with pytest.raises(ValueError, match="exceeds this exchange's"):
            ex.exchange_and_merge(big, num_tokens=T_cap + 1)
        with pytest.raises(ValueError, match="num_tokens must be >= 1"):
            ex.exchange_and_merge(big, num_tokens=0)
        # The extent and the carrier must agree: `num_tokens` is the caller
        # stating the graph's constant, not a licence to slice one plane and
        # not another.
        with pytest.raises(ValueError, match="cand must be"):
            ex.exchange_and_merge(
                make_candidates(T_cap, Hg, rank, dev, world=world), num_tokens=4
            )
        cand4 = make_candidates(4, Hg, rank, dev, world=world)
        with pytest.raises(ValueError, match="out must be"):
            ex.exchange_and_merge(
                cand4,
                num_tokens=4,
                out=torch.empty((T_cap, Hl, CAND_K), dtype=torch.int32, device=dev),
            )
        forced, n_ordinary = c3_planes(T_cap, dev)
        with pytest.raises(ValueError, match="per-TOKEN-ROW plane"):
            ex.exchange_and_merge(
                cand4, num_tokens=4, forced=forced, n_ordinary=n_ordinary
            )
        # A device tensor is refused by TYPE: reading it would synchronise on
        # the submission path and make the shape depend on a live count.
        with pytest.raises(TypeError, match="host int"):
            ex.exchange_and_merge(cand4, num_tokens=torch.tensor([4], device=dev))


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


# --------------------------------------------------------------------------
# the API contract that does not need a second rank
# --------------------------------------------------------------------------


@pytest.mark.distributed
@pytest.mark.gpu
def test_shape_is_validated_not_reallocated(icp_group, icp):
    """Symmetric memory cannot be allocated during cudagraph capture, so the
    window is sized once in ``__init__`` and only *checked* on use. A wrong
    ``T`` must be an error, never a quiet re-allocation.

    The dtype refusal is the refined-icp-v1 half: the carrier is int32, and the
    tensor a caller is most likely to pass by mistake is the producer's
    **float32** buffer -- same shape, same bytes, one ``.view`` away from
    correct. USED TO PASS ``float64`` here, which is not a mistake anybody makes
    and would keep passing against a build that accepted float32.
    """
    rank, world, dev = icp
    T, Hl = 8, 1
    Hg = world * Hl
    with IcpExchange(
        icp_group, tokens=T, heads_group=Hg, heads_local=Hl, slots=2, device=dev
    ) as ex:
        assert ex.dtype == torch.int32
        with pytest.raises(ValueError, match="cand must be"):
            ex.exchange_and_merge(make_candidates(T + 1, Hg, rank, dev, world=world))
        with pytest.raises(TypeError, match="int32"):
            ex.exchange_and_merge(
                make_candidates(T, Hg, rank, dev, world=world).view(torch.float32)
            )
        prov = ex.provenance()
        assert prov["world"] == world and prov["T"] == T


@pytest.mark.distributed
@pytest.mark.gpu
def test_the_window_is_pinned_at_the_int32_carrier(icp_group, icp):
    # C4 pins the carrier's dtype, so the symmetric window's element type is not
    # a caller's choice. A float32 window would be the same size, which is
    # exactly why this has to be refused by name rather than by arithmetic.
    rank, world, dev = icp
    with pytest.raises(TypeError, match="int32"):
        IcpExchange(
            icp_group,
            tokens=4,
            heads_group=world,
            heads_local=1,
            dtype=torch.float32,
            device=dev,
        )


@pytest.mark.distributed
@pytest.mark.gpu
def test_c7_head_geometry_is_enforced(icp_group, icp):
    rank, world, dev = icp
    if world < 2:  # pragma: no cover - the fixture already skipped
        pytest.skip("needs C >= 2")
    with pytest.raises(ValueError, match="H_group"):
        IcpExchange(
            icp_group, tokens=4, heads_group=world * 2 + 1, heads_local=2, device=dev
        )


def test_arch_is_never_hard_coded_and_the_flag_form_is_per_arch(monkeypatch):
    """CPU. The arch rules decide whether the exchange kernel exists at all on
    the target, so they are gated in this lane rather than assumed.

    A prior version of the loader hard-coded ``sm_107a`` and died on GB300 with
    "no kernel image is available for execution on the device", so:

    * ``ICP_KERNEL_ARCH`` must win over everything;
    * the ``-gencode`` must be the **two-part** form -- ``-arch=sm_XXXa`` also
      emits a ``compute_XXX`` PTX entry, i.e. a silent multi-hundred-
      millisecond driver JIT on first launch instead of a build error;
    * ``code=compute_*`` (PTX-only) must never appear, for the same reason;
    * sm_103 takes the plain form, where the ``a`` form emits that PTX entry.
    """
    from fmha_sm100.icp import _build

    monkeypatch.setenv("ICP_KERNEL_ARCH", "103a")
    assert _build.arch() == "103a"
    monkeypatch.setenv("ICP_KERNEL_ARCH", "sm_107A")
    assert _build.arch() == "107a"  # normalised, prefix stripped

    for a, want in [
        ("107a", ["-gencode=arch=compute_107a,code=sm_107a"]),
        ("100a", ["-gencode=arch=compute_100a,code=sm_100a"]),
        ("90a", ["-gencode=arch=compute_90a,code=sm_90a"]),
        ("103a", ["-gencode=arch=compute_103,code=sm_103"]),
        ("103", ["-gencode=arch=compute_103,code=sm_103"]),
    ]:
        assert _build.gencode_flags(a) == want, a

    flags = _build.cuda_flags("107a")
    assert not any(f.startswith("-arch=") for f in flags)
    assert not any("code=compute_" in f for f in flags)
    # No `-std=`: cpp_extension.load appends the one its own headers need, and
    # a caller-supplied `-std=c++17` wins over it and breaks modern torch.
    assert not any(f.startswith("-std=") for f in flags)

    with pytest.raises(ValueError):
        _build.gencode_flags("blackwell")


def test_mask_presets_are_the_kernel_masks():
    """CPU: the preset values *are* the kernel's ``opts`` bits.

    ``opt_bits()`` re-checks the numbering against the running kernel on GPU; a
    silently renumbered bit would run a different optimisation under a preset's
    name, which nothing downstream can see.

    Both values are frozen because the docs and every results CSV quote them.
    The unshipped option bits were dropped from the *supported* set without
    renumbering the survivors, precisely so that these integers keep meaning
    what they meant when they were measured.

    ``LAMPORT`` moved 768 -> 512 when the kernel from the upstream campaign
    landed here: that kernel implements ``OPT_NODIV`` and dispatches no mask
    containing it, so 768 is not a mask this build can run. The bit is dormant,
    not deleted -- which is why it is still in the naming table below.
    """
    from fmha_sm100.icp import exchange

    assert int(MaskPreset.HANDSHAKE) == 12  # OPT_NOFENCE | OPT_HIER
    assert int(MaskPreset.LAMPORT) == 512  # OPT_LAMPORT

    # The preset set is closed: a third mask must not appear without a
    # measurement. Mask 0 is instantiated and is deliberately NOT a preset.
    assert [p.name for p in MaskPreset] == ["HANDSHAKE", "LAMPORT"]

    # The presets really are compositions of the mirrored bits, not literals
    # that happen to match.
    assert int(MaskPreset.HANDSHAKE) == exchange.OPT_NOFENCE | exchange.OPT_HIER
    assert int(MaskPreset.LAMPORT) == exchange.OPT_LAMPORT

    # Every preset must be a mask the kernel's dispatch switch builds, and the
    # control must be in the set: `pybind11::arg("opts") = 0` is only a legal
    # default because `case 0` exists. A build that dropped mask 0 while keeping
    # that default would raise on the first positional or default-argument
    # launch -- possibly the first *captured* one.
    assert exchange.INSTANTIATED_MASKS == {0, 12, 512}
    for preset in MaskPreset:
        assert int(preset) in exchange.INSTANTIATED_MASKS


def test_undispatched_bits_are_refused_by_name_not_as_unknown_integers():
    """CPU: the two bits the kernel implements and dispatches from nowhere.

    An anonymous "unsupported bit" would be true and useless. Each of these has
    to name itself and say why it is not reachable, because "why is this gone"
    is asked at the point it is refused and nowhere else. The kernel
    TORCH_CHECKs the same two by name; this mirror exists so the refusal happens
    at construction rather than at first launch, which under a cudagraph may be
    the first *captured* launch.
    """
    from fmha_sm100.icp import exchange

    # The naming table is a NAMING table, not a support table: both
    # un-dispatched bits are in it so a mask number still decodes.
    assert exchange._OPT_NAMES == {
        "self_elide": 1,
        "nofence": 4,
        "hier": 8,
        "nodiv": 256,
        "lamport": 512,
    }
    assert exchange.RETIRED_OPTS == exchange.OPT_SELF_ELIDE
    assert exchange.DORMANT_OPTS == exchange.OPT_NODIV
    # A bit is either dispatched or refused, never both -- a supported bit that
    # no `case` builds is the silent-fallback trap the kernel is organised
    # around.
    assert not (
        exchange.SUPPORTED_OPTS & (exchange.RETIRED_OPTS | exchange.DORMANT_OPTS)
    )

    with pytest.raises(ValueError, match="OPT_SELF_ELIDE"):
        exchange._validate_mask(513)  # the retired mask
    with pytest.raises(ValueError, match="OPT_NODIV"):
        exchange._validate_mask(768)  # the previous LAMPORT preset
    with pytest.raises(ValueError, match="not instantiated"):
        exchange._validate_mask(4)  # an ablation arm, bit supported
    with pytest.raises(ValueError, match="not instantiated"):
        exchange._validate_mask(8)
    with pytest.raises(ValueError, match="handshake"):
        exchange._validate_mask(520)  # LAMPORT | HIER: dead code
    with pytest.raises(ValueError, match="non-negative"):
        exchange._validate_mask(-1)  # checked BEFORE any named bit,
        # or -1 would be blamed on
        # OPT_SELF_ELIDE
    for mask in (0, 12, 512):
        exchange._validate_mask(mask)


# --------------------------------------------------------------------------
# the grid plan. CPU: the kernel's OWN planner, compiled for the host.
# --------------------------------------------------------------------------
#
# `k5_plan` and `k5_ctl_words` are pure host integer arithmetic that happens to
# live in a .cu, and the JIT build that exports them to Python needs a CUDA
# device. So this section lifts the three functions -- `plan_tpb`, `k5_plan`,
# `k5_ctl_words` -- and the four constants they read straight out of
# `csrc/k5_exchange.cu` BY TEXT and compiles them with the host compiler. What
# runs below is therefore the kernel's own source, not a model of it: a change
# to `plan_tpb` moves these numbers, and a change that deletes or renames one
# of the extracted definitions fails the extraction rather than passing
# silently on a stale copy.
#
# The alternative -- restating the arithmetic in Python -- would have made the
# gate a test of the restatement. This file already carries the lesson: the
# whole regression gated here is `tpb` being planned from one number while the
# grid is derived from another.

_CU_SOURCE = pathlib.Path(fmha_sm100.icp.__file__).parent / "csrc" / "k5_exchange.cu"

#: Extracted by name. Each must appear EXACTLY once as a top-level definition.
_PLANNER_CONSTANTS = ("kWarpsPerBlock", "kCtlKinds", "kFlagStride", "kDefaultMaxBlocks")
_PLANNER_FUNCTIONS = (
    "int plan_tpb(",
    "std::vector<int64_t> k5_plan(",
    "int64_t k5_ctl_words(",
)

_PLANNER_MAIN = r"""
int main() {
  std::printf("%d %d %d %d\n", kDefaultMaxBlocks, kWarpsPerBlock, kCtlKinds,
              kFlagStride);
  long long T, Hl, mb, slots, world;
  while (std::scanf("%lld %lld %lld %lld %lld", &T, &Hl, &mb, &slots,
                    &world) == 5) {
    std::vector<int64_t> p = k5_plan(T, Hl, mb);
    std::printf("%lld %lld %lld\n", (long long)p[0], (long long)p[1],
                (long long)k5_ctl_words(slots, p[1], world));
  }
  return 0;
}
"""


def _extract_planner_source() -> str:
    """The planner's own text, or an error naming what moved."""
    lines = _CU_SOURCE.read_text().splitlines()
    out = ["#include <cstdint>", "#include <cstdio>", "#include <vector>"]
    for name in _PLANNER_CONSTANTS:
        hits = [ln for ln in lines if ln.startswith(f"constexpr int {name} = ")]
        if len(hits) != 1:
            raise AssertionError(
                f"{_CU_SOURCE.name} has {len(hits)} definitions of "
                f"`constexpr int {name}`, expected exactly 1. The grid gate "
                f"reads the kernel's constants by text; if one was renamed or "
                f"made non-constexpr, update _PLANNER_CONSTANTS rather than "
                f"reimplementing it here."
            )
        out.append(hits[0])
    for signature in _PLANNER_FUNCTIONS:
        starts = [i for i, ln in enumerate(lines) if ln.startswith(signature)]
        if len(starts) != 1:
            raise AssertionError(
                f"{_CU_SOURCE.name} has {len(starts)} definitions starting "
                f"`{signature}`, expected exactly 1."
            )
        first = starts[0]
        end = next((j for j in range(first, len(lines)) if lines[j] == "}"), None)
        if end is None:
            raise AssertionError(
                f"no closing `}}` at column 0 after `{signature}` in "
                f"{_CU_SOURCE.name}; the extraction assumes the repository's "
                f"own formatting."
            )
        out.extend(lines[first : end + 1])
    return "\n".join(out) + _PLANNER_MAIN


class _KernelPlanner:
    """``k5_plan`` and ``k5_ctl_words``, as the kernel computes them."""

    def __init__(self, binary: pathlib.Path) -> None:
        self._binary = binary
        head = self._run([]).split()
        (
            self.default_max_blocks,
            self.warps_per_block,
            self.ctl_kinds,
            self.flag_stride,
        ) = (int(v) for v in head[:4])

    def _run(self, queries: list[tuple[int, int, int, int, int]]) -> str:
        stdin = "".join(f"{t} {hl} {mb} {s} {w}\n" for t, hl, mb, s, w in queries)
        done = subprocess.run(
            [str(self._binary)], input=stdin, text=True, capture_output=True, check=True
        )
        return done.stdout

    def plan(self, tokens: int, heads_local: int, max_blocks: int) -> tuple[int, int]:
        """``(tpb, nblocks)``, i.e. ``k5_plan``."""
        line = self._run([(tokens, heads_local, max_blocks, 1, 1)])
        tpb, nblocks, _ = line.splitlines()[-1].split()
        return int(tpb), int(nblocks)

    def ctl_bytes(
        self, tokens: int, heads_local: int, max_blocks: int, *, slots: int, world: int
    ) -> int:
        """The control block's size for the plan that geometry produces."""
        line = self._run(
            [(tokens, heads_local, max_blocks, slots, world)]
        ).splitlines()[-1]
        return int(line.split()[2]) * 4  # uint32 words -> bytes


@pytest.fixture(scope="session")
def k5_planner(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip(
            "no host C++ compiler on PATH, so the kernel's own `plan_tpb` "
            "cannot be compiled for the host. This gate deliberately does not "
            "fall back to a Python restatement of the arithmetic -- that would "
            "test the restatement."
        )
    build = tmp_path_factory.mktemp("k5_planner")
    source = build / "planner.cpp"
    source.write_text(_extract_planner_source())
    binary = build / "planner"
    subprocess.run(
        [compiler, "-O2", "-o", str(binary), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    return _KernelPlanner(binary)


#: The deployed geometry: 4 index heads split by TP, and one symmetric window
#: at the scheduler's token budget.
GRID_CAPACITY = 8192
GRID_GEOMETRIES = {"TP2": (2, 2), "TP4": (1, 4)}  # H_local, world
#: Both slot widths that exist. 3 is what the model binds
#: (``icp_dispatch.ICP_EXCHANGE_SLOTS``: the smallest that clears
#: ``ACK_FREE_SLOTS`` and divides the 57-layer sweep without a wrap-around
#: reuse); 57 is this package's own default and what the bench harness used.
#: The control block is linear in ``slots``, so the price differs by 19x
#: between them and quoting one alone would misstate it.
GRID_SLOTS = (3, 57)

#: A deployed ladder's decode band plus the global capacity. The point of the
#: small rungs is that they are the extents a decode step actually runs, 57
#: times per step, on the critical path.
ADMITTED_EXTENTS = (4, 16, 64, 8192)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _grid_at_every_extent(planner, heads_local: int, max_blocks: int) -> dict:
    """``ceil(T / tpb)`` per extent, for ONE window at the global capacity.

    This is `k5_exchange.cu:904-908` and nothing else: `tpb` from the capacity,
    the grid from the extent.
    """
    tpb, _ = planner.plan(GRID_CAPACITY, heads_local, max_blocks)
    return {extent: _cdiv(extent, tpb) for extent in ADMITTED_EXTENTS}


def _grid_the_per_extent_window_gave(planner, heads_local: int) -> dict:
    """What the pre-change code planned: one window PER extent.

    Before the one-window change a window was allocated at ``tokens=extent``,
    so the capacity and the extent were the same number and `plan_tpb` saw the
    extent. This reconstructs that plan exactly, at the historical
    ``max_blocks``.
    """
    grid = {}
    for extent in ADMITTED_EXTENTS:
        tpb, nblocks = planner.plan(
            extent, heads_local, exchange_module.DEFAULT_MAX_BLOCKS
        )
        assert _cdiv(extent, tpb) == nblocks  # the window WAS the extent
        grid[extent] = nblocks
    return grid


def _unwidened_band(planner, heads_local: int) -> int:
    """The largest extent the PRE-CHANGE plan did not widen the block for.

    ``plan_tpb`` widened once ``ceil(T / 64) > tpb_natural``, so a per-extent
    window planned the natural ``tpb`` for every extent up to
    ``64 * tpb_natural`` -- 256 rows at TP2, 512 at TP4 -- and widened above it.
    """
    return exchange_module.DEFAULT_MAX_BLOCKS * exchange_module.natural_tpb(
        planner.plan, heads_local
    )


@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
def test_one_window_keeps_the_grid_the_per_extent_windows_gave(k5_planner, geometry):
    """The regression itself: 64 rows must still get 16 CTAs at TP2.

    Two claims, and they are deliberately not the same claim:

    * On every extent the pre-change code did NOT widen the block for -- the
      whole decode band, 4/16/64 -- the grid is restored EXACTLY.
    * On no extent at all is the grid smaller than the per-extent windows gave.
      Above the unwidened band it is larger, because the pre-change plan widened
      there too; see the test below, which pins that number.
    """
    heads_local, _ = GRID_GEOMETRIES[geometry]
    derived = exchange_module.plan_max_blocks(
        k5_planner.plan,
        tokens=GRID_CAPACITY,
        heads_local=heads_local,
        opts=int(MaskPreset.LAMPORT),
    )
    now = _grid_at_every_extent(k5_planner, heads_local, derived)
    before = _grid_the_per_extent_window_gave(k5_planner, heads_local)
    band = _unwidened_band(k5_planner, heads_local)

    restored = {e: g for e, g in now.items() if e <= band}
    assert restored == {e: g for e, g in before.items() if e <= band}
    assert restored, "the admitted ladder has no extent inside the band"
    assert all(now[e] >= before[e] for e in now), (now, before)


@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
def test_the_capacity_rung_gains_blocks_rather_than_keeping_its_own(
    k5_planner, geometry
):
    """A full-capacity launch does NOT keep the grid it had. It gets more.

    The brief this patch answers said the derivation "restores the pre-change
    ``grid_blocks`` at every extent". That is true of every extent the
    pre-change plan left unwidened and false at the capacity itself, where the
    pre-change plan widened too: at ``T_cap = 8192`` a per-extent window planned
    ``tpb = 128`` and 64 CTAs, and this patch plans the natural ``tpb`` and
    ``T_cap / tpb_natural`` CTAs. The work is identical -- the same rows, the
    same warp count -- so this is occupancy, not extra work, and 64 CTAs
    under-fills any GB300-class part. But it is a CHANGE at the full extent, not
    a restoration, and no measurement in this repository covers it. Pinned here
    so it is a recorded consequence rather than a surprise.
    """
    heads_local, _ = GRID_GEOMETRIES[geometry]
    derived = exchange_module.plan_max_blocks(
        k5_planner.plan,
        tokens=GRID_CAPACITY,
        heads_local=heads_local,
        opts=int(MaskPreset.LAMPORT),
    )
    natural = exchange_module.natural_tpb(k5_planner.plan, heads_local)
    _, before = k5_planner.plan(
        GRID_CAPACITY, heads_local, exchange_module.DEFAULT_MAX_BLOCKS
    )
    _, after = k5_planner.plan(GRID_CAPACITY, heads_local, derived)
    assert before == exchange_module.DEFAULT_MAX_BLOCKS
    assert after == GRID_CAPACITY // natural
    assert after > before


@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
def test_the_grid_gate_fires_at_the_max_blocks_that_caused_the_regression(
    k5_planner, geometry
):
    """Discrimination for the gate above: at ``max_blocks = 64`` it FAILS.

    Not a restatement of it -- the same two dicts and BOTH of its assertions,
    with the one number under test put back. A gate that cannot fail is not a
    gate, and this one's whole content is that 64 is the wrong number.
    """
    heads_local, _ = GRID_GEOMETRIES[geometry]
    collapsed = _grid_at_every_extent(
        k5_planner, heads_local, exchange_module.DEFAULT_MAX_BLOCKS
    )
    before = _grid_the_per_extent_window_gave(k5_planner, heads_local)
    band = _unwidened_band(k5_planner, heads_local)

    # Assertion 1 of the gate above fails.
    assert {e: g for e, g in collapsed.items() if e <= band} != {
        e: g for e, g in before.items() if e <= band
    }
    # Assertion 2 of the gate above fails: the grid SHRINKS.
    assert any(collapsed[e] < before[e] for e in collapsed), (collapsed, before)
    # And at TP2 it shrinks by exactly the reported factor on the 64-row extent.
    if heads_local == 2:
        assert (collapsed[64], before[64]) == (1, 16)


def test_the_collapse_is_the_reported_sixteen_fold_one_at_tp2(k5_planner):
    """The arithmetic in the bug report, spelled out rather than derived."""
    # Before: a window per extent. 64 rows, ceil(64/64) = 1 which is not > 4.
    assert k5_planner.plan(64, 2, 64) == (4, 16)
    # After the one-window change, at the old max_blocks: the CAPACITY widens
    # the block to 128 rows, and 64 rows then fit in a single CTA.
    assert k5_planner.plan(GRID_CAPACITY, 2, 64) == (128, 64)
    assert _cdiv(64, 128) == 1
    # With the derived max_blocks the capacity plans the natural tpb again.
    derived = exchange_module.plan_max_blocks(
        k5_planner.plan,
        tokens=GRID_CAPACITY,
        heads_local=2,
        opts=int(MaskPreset.LAMPORT),
    )
    assert derived == 2048
    assert k5_planner.plan(GRID_CAPACITY, 2, derived) == (4, 2048)
    assert _cdiv(64, 4) == 16


def _tpb_as_the_kernel_plans_it(
    planner, *, extent: int, heads_local: int, max_blocks: int, from_extent: bool
) -> int:
    """``k5_exchange``'s ``tpb``. ``from_extent`` is the negative control.

    ``from_extent=False`` is `k5_exchange.cu:904`, which plans from ``T_cap``.
    ``from_extent=True`` is the change the kernel's own note at :238 forbids --
    planning from ``T`` -- and exists here so the independence assertion has a
    way to fail. It is not a hypothetical: at ``max_blocks = 64`` it is exactly
    what the pre-change per-extent windows did, which is why the control below
    passes that number rather than the derived one. (With a derived
    ``max_blocks`` the extent cannot move ``tpb`` either, since the widening
    term is then ``<= tpb_natural`` for every ``T <= T_cap`` -- so that
    configuration would be a control that cannot fire.)
    """
    tokens = extent if from_extent else GRID_CAPACITY
    return planner.plan(tokens, heads_local, max_blocks)[0]


@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
def test_tpb_and_the_row_to_block_map_do_not_move_with_the_extent(k5_planner, geometry):
    """The safety property: one row, one block, for the workspace's life.

    ``tpb`` fixes ``block = row // tpb``. If it moved with the extent, two
    independent generation counters would write one word and OPT_LAMPORT's
    exact tag compare would accept a stale tag as arrived.

    The configuration this discriminates against is the PRE-CHANGE one -- a
    window per extent, i.e. ``max_blocks = 64`` and ``tpb`` planned from ``T``
    -- which is what the control below builds. Under the derived ``max_blocks``
    alone the extent cannot move ``tpb``; see the structural test further down
    for why, and note that it means flipping ONLY the capacity source is a
    mutation this gate is not able to see.
    """
    heads_local, _ = GRID_GEOMETRIES[geometry]
    derived = exchange_module.plan_max_blocks(
        k5_planner.plan,
        tokens=GRID_CAPACITY,
        heads_local=heads_local,
        opts=int(MaskPreset.LAMPORT),
    )
    tpbs = {
        extent: _tpb_as_the_kernel_plans_it(
            k5_planner,
            extent=extent,
            heads_local=heads_local,
            max_blocks=derived,
            from_extent=False,
        )
        for extent in ADMITTED_EXTENTS
    }
    assert len(set(tpbs.values())) == 1, tpbs
    # The map itself, on the rows the small extents actually touch.
    tpb = next(iter(tpbs.values()))
    owners = [row // tpb for row in range(0, 256)]
    for extent in ADMITTED_EXTENTS:
        assert [row // tpbs[extent] for row in range(0, 256)] == owners


@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
def test_the_independence_gate_fires_on_an_extent_derived_tpb(k5_planner, geometry):
    """Discrimination for the gate above: plan ``tpb`` from ``T`` and it FAILS.

    The same dict, built the same way, from the pre-change per-extent plan.
    """
    heads_local, _ = GRID_GEOMETRIES[geometry]
    tpbs = {
        extent: _tpb_as_the_kernel_plans_it(
            k5_planner,
            extent=extent,
            heads_local=heads_local,
            max_blocks=exchange_module.DEFAULT_MAX_BLOCKS,
            from_extent=True,
        )
        for extent in ADMITTED_EXTENTS
    }
    assert len(set(tpbs.values())) > 1, tpbs
    # And the row -> block map moves with it, which is the hazard itself: row
    # 300 belongs to a different block at a different extent.
    owners = {extent: 300 // tpb for extent, tpb in tpbs.items()}
    assert len(set(owners.values())) > 1, owners


@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
def test_under_the_derived_max_blocks_no_T_below_the_capacity_can_move_tpb(
    k5_planner, geometry
):
    """Why the independence above is structural once ``max_blocks`` is derived.

    ``max_blocks = ceil(T_cap / tpb_natural)`` makes the widening term
    ``ceil(T / max_blocks) <= ceil(T_cap / max_blocks) == tpb_natural`` for
    every ``T <= T_cap``, so ``plan_tpb`` returns the natural value whether it
    is handed the extent or the capacity. The capacity/extent distinction at
    `k5_exchange.cu:904` stops being able to move the row -> block map at all --
    it remains the right line to write, but a slip in it is no longer the
    silent-corruption hazard the kernel's note at :238 describes. That is a
    consequence of this patch worth having written down; it is NOT a licence to
    plan from the extent, because ``max_blocks`` is itself capacity-derived.
    """
    heads_local, _ = GRID_GEOMETRIES[geometry]
    derived = exchange_module.plan_max_blocks(
        k5_planner.plan,
        tokens=GRID_CAPACITY,
        heads_local=heads_local,
        opts=int(MaskPreset.LAMPORT),
    )
    natural = exchange_module.natural_tpb(k5_planner.plan, heads_local)
    for tokens in (1, 4, 16, 64, 128, 1000, 4096, GRID_CAPACITY):
        assert k5_planner.plan(tokens, heads_local, derived)[0] == natural


@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
@pytest.mark.parametrize("capacity", [128, 256, 2048, 8192, 16384])
def test_the_derivation_lands_tpb_on_its_natural_value(k5_planner, geometry, capacity):
    """At both geometries and at capacities either side of the floor."""
    heads_local, _ = GRID_GEOMETRIES[geometry]
    natural = exchange_module.natural_tpb(k5_planner.plan, heads_local)
    assert natural == k5_planner.warps_per_block // heads_local
    derived = exchange_module.plan_max_blocks(
        k5_planner.plan,
        tokens=capacity,
        heads_local=heads_local,
        opts=int(MaskPreset.LAMPORT),
    )
    assert derived == max(exchange_module.DEFAULT_MAX_BLOCKS, _cdiv(capacity, natural))
    tpb, nblocks = k5_planner.plan(capacity, heads_local, derived)
    assert tpb == natural
    assert nblocks == _cdiv(capacity, natural)


def test_the_python_floor_is_still_the_kernels_own_constant(k5_planner):
    """``DEFAULT_MAX_BLOCKS`` is now a floor, but it is still that number."""
    assert exchange_module.DEFAULT_MAX_BLOCKS == k5_planner.default_max_blocks


@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
def test_opt_hier_keeps_the_old_max_blocks_because_the_kernel_bounds_its_grid(
    k5_planner, geometry
):
    """`k5_exchange.cu:932` bounds the LIVE grid by ``kDefaultMaxBlocks``.

    It compares against the kernel's constant, never against the ``max_blocks``
    argument, and OPT_HIER's own short-extent refusal pins ``T == T_cap`` -- so
    under that bit the live grid IS ``nblocks``. A derived ``max_blocks`` would
    therefore make the mask refuse its only legal launch. It is excluded, and
    this is the arithmetic that says it has to be.
    """
    heads_local, _ = GRID_GEOMETRIES[geometry]
    handshake = int(MaskPreset.HANDSHAKE)
    assert handshake & exchange_module.OPT_HIER

    kept = exchange_module.plan_max_blocks(
        k5_planner.plan, tokens=GRID_CAPACITY, heads_local=heads_local, opts=handshake
    )
    assert kept == exchange_module.DEFAULT_MAX_BLOCKS
    _, nblocks = k5_planner.plan(GRID_CAPACITY, heads_local, kept)
    assert nblocks <= k5_planner.default_max_blocks

    # And the refusal it would have walked into.
    derived = exchange_module.plan_max_blocks(
        k5_planner.plan,
        tokens=GRID_CAPACITY,
        heads_local=heads_local,
        opts=int(MaskPreset.LAMPORT),
    )
    _, would_be = k5_planner.plan(GRID_CAPACITY, heads_local, derived)
    assert would_be > k5_planner.default_max_blocks


@pytest.mark.parametrize("slots", GRID_SLOTS)
@pytest.mark.parametrize("geometry", sorted(GRID_GEOMETRIES))
def test_the_control_block_cost_of_the_derived_plan_is_the_measured_one(
    k5_planner, geometry, slots
):
    """The price, pinned as a number so a later change to it is visible.

    The control block is ``kCtlKinds * slots * nblocks * world * kFlagStride``
    uint32 words of SYMMETRIC memory, and ``nblocks`` is the only term this
    patch moves. It is not free and the numbers are not rounded here.

    For scale: at the deployed ``slots = 3`` the DATA window is
    ``T_cap * H_group * 16 * 4 * slots * 4`` bytes = 24 MiB, so the control
    block goes from 0.8% of it to 25% -- 192 KiB to 6 MiB at TP2. At the
    package default ``slots = 57`` the same ratio is 3.56 MiB to 114 MiB
    against a 456 MiB window. The ratio is what this patch fixes in place; the
    absolute number is a function of ``slots``, which it does not touch.
    """
    heads_local, world = GRID_GEOMETRIES[geometry]
    before = k5_planner.ctl_bytes(
        GRID_CAPACITY,
        heads_local,
        exchange_module.DEFAULT_MAX_BLOCKS,
        slots=slots,
        world=world,
    )
    derived = exchange_module.plan_max_blocks(
        k5_planner.plan,
        tokens=GRID_CAPACITY,
        heads_local=heads_local,
        opts=int(MaskPreset.LAMPORT),
    )
    after = k5_planner.ctl_bytes(
        GRID_CAPACITY, heads_local, derived, slots=slots, world=world
    )
    expected = {
        ("TP2", 3): (196_608, 6_291_456),
        ("TP4", 3): (393_216, 6_291_456),
        ("TP2", 57): (3_735_552, 119_537_664),
        ("TP4", 57): (7_471_104, 119_537_664),
    }[(geometry, slots)]
    assert (before, after) == expected
    # The data window this is measured against, from the kernel's own formula
    # for the mask that ships. The control block must stay a minority of it.
    data = GRID_CAPACITY * (heads_local * world) * CAND_K * 4 * slots * 4
    assert after < data
    # Both geometries land on the same figure, and that is arithmetic rather
    # than coincidence: nblocks * world = (T_cap / tpb_natural) * world =
    # T_cap * H_local * world / kWarpsPerBlock = T_cap * H_group / 8, and
    # H_group is 4 at every TP this model runs.
    assert (
        after
        == (
            k5_planner.ctl_kinds
            * slots
            * k5_planner.flag_stride
            * GRID_CAPACITY
            * (heads_local * world)
            // k5_planner.warps_per_block
        )
        * 4
    )
