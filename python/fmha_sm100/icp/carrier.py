"""refined-icp-v1 C4 candidate carrier and head-directed transport.

This module owns exactly two things:

1. **The carrier.** Packing this rank's local candidates into C4's int32
   ``[W, Qchunk, Hlocal, 16, 2]`` **destination-major** send carrier, and the
   ``[W, Qchunk, Hlocal, 16, 2]`` **source-major** receive carrier the merge
   consumes. See ``docs/CONTRACT.md`` C4.
2. **The transport.** ``all_to_all_single`` with the fixed per-peer split, plus
   the fixed all-gather + head-slice **reference** path under the same API, so
   the head-directed arm can be gated byte-for-byte against it.

Both take an optional caller-owned destination, and
:class:`CarrierWorkspace` bundles the pair. Neither carrier's size has ever
depended on a step -- C4 fixes it at ``(W, Qchunk, Hlocal)`` -- so allocating
them per call, 57 layers deep, was buying nothing. The no-workspace path is
unchanged and is still the default.

This is the NCCL-collective route, which serves prefill/mixed steps; pure
decode uses :class:`fmha_sm100.icp.tiled_exchange.IcpTiledExchange` (K5T).
:func:`collective_exchange_and_merge` is ATen-free apart from the collective:
the pack is the native ``icp_pack`` kernel and the merge is a direct K2
launch. :func:`pack_send_carrier` (the ATen permute) is kept only as the
reference the native pack is gated against.

A defect this transport does NOT detect, stated here because the call site is
where a reader will look for it: if a rank never publishes, the collective
**returns without raising** and that rank's slab arrives zero-filled, decoding
as 80 *valid* candidates for block 0 -- a legal block id, which the merge then
selects. ``init_process_group(timeout=...)`` bounds the watchdog, not the
calling thread (measured: 80.8 s against a configured 20 s). Closing this needs
a device-visible per-source arrival witness; there is none today, and nothing in
the gate set detects a regression here.

WHY A ``.view`` IS NOT A PACK
-----------------------------
The producer emits query-major ``[Qchunk, H_group, 16, 2]``: query ``q``'s four
global heads are contiguous. The send carrier needs destination ``d``'s slab --
heads ``[d*Hlocal, (d+1)*Hlocal)`` for **all** queries -- contiguous. For
``Qchunk == 1`` those coincide and a reshape is free. For ``Qchunk > 1`` they do
not: element ``(q, d)`` sits at stride ``H_group`` in one and at stride ``Q`` in
the other. C4 says so explicitly: "Q>1 requires actual packing or direct
emission, not a token/head reshape". ``pack_send_carrier`` therefore performs a
real permutation; the forbidden reshape is kept here as
:func:`_wrong_pack_view_negative_control` so the gate that proves it matters
cannot go stale. That gate fires at ``Qchunk = 5`` and, measured, does **not**
fire at ``Qchunk = 1`` -- a Q=1-only test proves nothing here.

LIFETIME (C9)
-------------
``exchange_carrier`` is collective. Every rank must call it the same number of
times in the same order with the same shapes, **including ranks whose local
fragment is empty** -- those publish 16 ``(-inf, -1)`` records and still
participate (C4: "capture padding participates with invalid records"). The
collective's element counts are a pure function of ``(W, Qchunk, Hlocal)``,
never of the live row count, so they are fixed across ranks and across replays,
which is C4's capture requirement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

# CONTRACT C4 / A0 `refined_icp_v1.K_SELECT`.
K_SELECT = 16
# CONTRACT C4: the two words of the 8-byte record.
CARRIER_WORD_SCORE_BITS = 0
CARRIER_WORD_BLOCK_ID = 1
INVALID_BLOCK_ID = -1

TRANSPORTS = ("all_to_all", "all_gather")


# --------------------------------------------------------------------------
# 1. The carrier
# --------------------------------------------------------------------------


def send_carrier_shape(W: int, qchunk: int, h_local: int) -> tuple[int, ...]:
    """C4 send carrier: axis 0 is the DESTINATION rank."""
    return (W, qchunk, h_local, K_SELECT, 2)


def recv_carrier_shape(W: int, qchunk: int, h_local: int) -> tuple[int, ...]:
    """C4 receive carrier: identical shape, axis 0 is the SOURCE rank.

    A0 pins this: "send/recv carrier SHAPES match; only the axis-0 MEANING
    differs" (`refined_icp_v1.py:453-454`). The shapes matching is what makes
    the `all_to_all_single` split symmetric.
    """
    return (W, qchunk, h_local, K_SELECT, 2)


def peer_split_int32_elements(qchunk: int, h_local: int) -> int:
    """int32 words in ONE source->destination slab (A0 `:255-257`)."""
    return 2 * qchunk * h_local * K_SELECT


def invalid_carrier(W: int, qchunk: int, h_local: int,
                    device: torch.device | str) -> torch.Tensor:
    """A fully invalid send carrier: every record is ``(-inf, -1)``.

    This is what a rank with **zero valid candidates** publishes. It is not an
    optimisation and not a skip: C4 requires fixed all-rank participation, so
    the empty rank sends exactly as many bytes as everyone else.
    """
    out = torch.empty(send_carrier_shape(W, qchunk, h_local),
                      dtype=torch.int32, device=device)
    neg_inf_bits = torch.tensor([float("-inf")],
                                dtype=torch.float32).view(torch.int32).item()
    out[..., CARRIER_WORD_SCORE_BITS] = neg_inf_bits
    out[..., CARRIER_WORD_BLOCK_ID] = INVALID_BLOCK_ID
    return out


def _check_buffer(name: str, buf: torch.Tensor, *, shape: tuple[int, ...],
                  device: torch.device) -> torch.Tensor:
    """A caller-supplied carrier buffer, checked the way the pack checks its own.

    Raised, not asserted, and checked for **contiguity** as well as shape: a
    strided buffer would be silently materialised by ``all_to_all_single`` in an
    order nobody verified, and a workspace exists precisely so that no copy
    happens behind the caller's back.
    """
    if tuple(buf.shape) != shape:
        raise ValueError(
            f"{name} must be {list(shape)}; got {list(buf.shape)}")
    if buf.dtype != torch.int32:
        raise TypeError(
            f"{name} must be int32 (C4 carrier words); got {buf.dtype}")
    if buf.device != device:
        raise ValueError(f"{name} is on {buf.device}, not {device}")
    if not buf.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return buf


def pack_send_carrier(local_cand: torch.Tensor, W: int, *,
                      out: torch.Tensor | None = None) -> torch.Tensor:
    """Query-major producer output -> C4 destination-major int32 send carrier.

    ``local_cand``: ``[Qchunk, H_group, 16, 2]``. Either

      * fp32, the shipping producer layout, where ``[...,0]`` is the score and
        ``[...,1]`` is the int32 block id **bitcast** into the fp32 slot (the
        layout ``csrc/local_candidates.cuh::store_local_candidate`` emits), or
      * int32 already, where ``[...,0]`` is the score's raw bits.

    Both are handled with ``Tensor.view(dtype)``, which is a **reinterpretation
    of the same storage** and never a numeric conversion. That is the whole
    bitcast discipline: an id above ``2**24`` has no exact fp32 representation,
    so a single ``.float()`` anywhere on this path silently corrupts large block
    ids while leaving small ones correct -- a defect that only shows up on long
    sequences.

    ``out`` is an optional destination of exactly that shape. Supplied, this
    function allocates **nothing**: the permutation is written into it and it is
    returned. The NCCL route runs once per indexer layer -- 57 times per
    invocation -- and a fresh send carrier per layer is 57 allocations of the
    same bytes, so a caller that retains one buffer and passes it here pays for
    it once. Omitted, a new tensor is allocated, which is the behaviour every
    existing caller has and the one the tests without a workspace exercise.

    Returns int32 ``[W, Qchunk, Hlocal, 16, 2]``, contiguous, destination-major.
    """
    # Raised, not asserted: `python -O` deletes an assert, and a mis-shaped
    # pack is silently wrong rather than loud. Same discipline as `merge.py`
    # and `exchange.py`, which never validate with `assert`.
    if local_cand.dim() != 4 or local_cand.shape[-1] != 2:
        raise ValueError(
            f"expected [Qchunk, H_group, 16, 2], got {tuple(local_cand.shape)}")
    q, h_group, k, _ = local_cand.shape
    if k != K_SELECT:
        raise ValueError(f"C4 fixes K={K_SELECT}, got {k}")
    if h_group % W != 0:
        raise ValueError(f"H_group {h_group} not divisible by W {W}")
    h_local = h_group // W

    if local_cand.dtype == torch.float32:
        # Bit reinterpretation of the SAME storage. Not `.to`, not `.int()`.
        src = local_cand.contiguous().view(torch.int32)
    elif local_cand.dtype == torch.int32:
        src = local_cand.contiguous()
    else:
        raise TypeError(
            f"C4 carrier words are 4 bytes; got {local_cand.dtype}")

    # [Q, W, Hl, K, 2] -> [W, Q, Hl, K, 2]. The permutation must be
    # MATERIALISED; left as a strided view, `all_to_all_single` would either
    # refuse it or (after an implicit contiguous) move the wrong bytes. C4:
    # "Q>1 requires actual packing or direct emission".
    permuted = src.view(q, W, h_local, K_SELECT, 2).permute(1, 0, 2, 3, 4)
    if out is None:
        return permuted.contiguous()
    # `copy_` into the caller's buffer is the same materialisation as
    # `.contiguous()` -- one strided read, one contiguous write -- minus the
    # allocation. It is NOT `out[...] = permuted`, which is the same thing with
    # an advanced-indexing path in front of it.
    _check_buffer("out", out, shape=send_carrier_shape(W, q, h_local),
                  device=local_cand.device)
    out.copy_(permuted)
    return out


def _wrong_pack_view_negative_control(local_cand: torch.Tensor,
                                      W: int) -> torch.Tensor:
    """NEGATIVE CONTROL ONLY. The reshape C4 forbids.

    ``[Q, H_group, 16, 2] -> view(W, Q, Hlocal, 16, 2)`` reinterprets the SAME
    storage with the axes in the wrong order. At ``Qchunk == 1`` it is
    accidentally correct, which is exactly why a Q=1-only gate proves nothing.
    For ``Qchunk > 1`` it routes query ``q``'s heads to the wrong destination.
    Never call this outside the gate.
    """
    src = (local_cand.contiguous().view(torch.int32)
           if local_cand.dtype == torch.float32 else local_cand.contiguous())
    q, h_group = src.shape[0], src.shape[1]
    return src.view(W, q, h_group // W, K_SELECT, 2).contiguous()


# --------------------------------------------------------------------------
# 2. The transport
# --------------------------------------------------------------------------


def exchange_carrier(send: torch.Tensor, *, world: int, rank: int,
                     transport: str = "all_to_all",
                     group: dist.ProcessGroup | None = None,
                     out: torch.Tensor | None = None,
                     gathered: torch.Tensor | None = None) -> torch.Tensor:
    """C4 destination-major send carrier -> source-major receive carrier.

    Both transports are collective over every rank of ``group`` and both return
    byte-identical results; they differ only in how much they move. The
    ``all_gather`` arm is the reference implementation, kept under the same API
    so the head-directed arm can be gated against it.

    ``out`` is an optional receive carrier of ``send``'s shape; ``gathered`` is
    the ``all_gather`` arm's ``[W, *send.shape]`` staging buffer, which is
    ``W`` times larger and is the reason that arm is a reference rather than a
    route. Supplied, this function allocates nothing. Both are ignored by the
    arm that does not use them, and a caller that retains them across layers
    turns 57 allocations per invocation into zero -- see
    :class:`CarrierWorkspace`.

    Supplying ``out`` does **not** make the call non-collective or reorder
    anything: the element counts stay a pure function of ``(W, Qchunk, Hlocal)``
    (C4's capture requirement), and the buffer only changes where the bytes
    land.
    """
    # Raised, not asserted: every one of these is a silent wrong answer under
    # `python -O` if the check disappears. See `pack_send_carrier`.
    if transport not in TRANSPORTS:
        raise ValueError(
            f"unknown transport {transport!r}; known: {', '.join(TRANSPORTS)}")
    if send.dtype != torch.int32:
        raise TypeError(
            f"C4 carrier is int32; got {send.dtype}. A float32 carrier here "
            "means the pack step converted instead of bitcasting.")
    if send.dim() != 5 or send.shape[0] != world or send.shape[-1] != 2:
        raise ValueError(
            f"expected [W={world}, Qchunk, Hlocal, 16, 2], got "
            f"{tuple(send.shape)}")
    if not send.is_contiguous():
        raise ValueError(
            "the send carrier must be contiguous: a strided permutation would "
            "be silently materialised by the collective in an order nobody "
            "checked")

    if out is not None:
        _check_buffer("out", out, shape=tuple(send.shape), device=send.device)
        if out.data_ptr() == send.data_ptr():
            raise ValueError(
                "the receive carrier must not alias the send carrier: "
                "`all_to_all_single` is not defined in place, and at world == 1 "
                "the copy below would be a no-op that looks like a transport")

    if world == 1:
        # Still a real carrier, not a special case: axis 0 is length 1 and the
        # only source is this rank. No collective, because there is no peer --
        # this is the one place where skipping is not a participation failure.
        # The copy stays a copy even into a caller's buffer: returning `send`
        # itself would alias the producer's storage into the merge's input.
        return send.clone() if out is None else out.copy_(send)

    if transport == "all_to_all":
        # THE head-directed transport. `all_to_all_single` with equal splits
        # sends chunk `d` of the input to rank `d` and places rank `s`'s chunk
        # at index `s` of the output -- which converts destination-major to
        # source-major exactly and by definition. The split IS the peer-slab
        # word count -- `peer_split_int32_elements` above, a pure function of
        # (W, Qchunk, Hlocal) -- so it is identical on every rank and on every
        # replay.
        recv = torch.empty_like(send) if out is None else out
        dist.all_to_all_single(recv, send, group=group)
        return recv

    # The reference. `all_gather` on dim 0 concatenates, so output index
    # `r*W + d` is rank `r`'s slab for destination `d`. This rank wants
    # `[:, rank]` -- every source's slab addressed to it. That slice is the
    # "head slicing" the pipeline doc names; it is also W times the traffic,
    # which is the whole reason the head-directed arm exists.
    if gathered is None:
        gathered = torch.empty((world, *send.shape), dtype=send.dtype,
                               device=send.device)
    else:
        _check_buffer("gathered", gathered, shape=(world, *send.shape),
                      device=send.device)
    dist.all_gather_into_tensor(gathered, send, group=group)
    sliced = gathered[:, rank]
    return sliced.contiguous() if out is None else out.copy_(sliced)


# --------------------------------------------------------------------------
# 3. the retained buffers
# --------------------------------------------------------------------------


# `eq=False`: the fields are tensors and a generated `__eq__` would return a
# tensor rather than a bool, exactly as in `candidates.CandidateWorkspace`.
@dataclass(frozen=True, slots=True, eq=False)
class CarrierWorkspace:
    """The two (or three) carriers of one NCCL route, allocated once.

    This route runs **per indexer layer** -- 57 times per invocation on this
    model -- and every one of those calls allocated a send carrier and a receive
    carrier of its own. Their size is a pure function of ``(W, Qchunk,
    Hlocal)``, which C4 already requires to be fixed across ranks and across
    replays, so nothing about them was ever per-step: the allocation was the
    only thing that varied.

    It is the same bargain :class:`fmha_sm100.icp.CandidateWorkspace` makes for the
    decode selector, and it is deliberately **optional** in the same way. With
    no workspace the route allocates per call and behaves exactly as it always
    has; that path is not deprecated and is what the eager tests use.

    ``gathered`` is the ``all_gather`` reference arm's staging buffer and is
    ``None`` for an ``all_to_all`` workspace -- ``W`` times the receive carrier
    is not something to allocate for a transport this workspace will not run.

    Keep it alive for as long as any call may use it, and do not share one
    across concurrent callers: two calls on two streams would write one buffer.
    The routes' collectives are already required to be issued in the same order
    on every rank (C9), which is the same constraint.
    """

    world: int
    qchunk: int
    h_local: int
    send: torch.Tensor
    recv: torch.Tensor
    gathered: torch.Tensor | None = field(default=None)


def allocate_carrier_workspace(*, world: int, qchunk: int, h_local: int,
                               device: torch.device | str,
                               transport: str = "all_to_all",
                               ) -> CarrierWorkspace:
    """Allocate one route's carriers, eagerly. Not collective.

    ``qchunk`` is the **capacity** row count this workspace serves, i.e. the
    largest ``Qchunk`` any call through it will present. A shorter call needs a
    differently *shaped* carrier -- the row count is baked into the collective's
    element counts, not just into a stride -- so it needs its own workspace;
    unlike :class:`fmha_sm100.icp.IcpExchange`'s symmetric window, nothing here is
    scarce, and a dict keyed by extent is the whole story.
    """
    if transport not in TRANSPORTS:
        raise ValueError(
            f"unknown transport {transport!r}; known: {', '.join(TRANSPORTS)}")
    if world < 1 or qchunk < 1 or h_local < 1:
        raise ValueError(
            f"world, qchunk and h_local must all be >= 1; got {world}, "
            f"{qchunk}, {h_local}")
    device = torch.device(device)
    shape = send_carrier_shape(world, qchunk, h_local)
    send = torch.empty(shape, dtype=torch.int32, device=device)
    recv = torch.empty(recv_carrier_shape(world, qchunk, h_local),
                       dtype=torch.int32, device=device)
    gathered = None
    if transport == "all_gather" and world > 1:
        gathered = torch.empty((world, *shape), dtype=torch.int32,
                               device=device)
    return CarrierWorkspace(world=world, qchunk=qchunk, h_local=h_local,
                            send=send, recv=recv, gathered=gathered)


def _workspace_for(workspace: CarrierWorkspace | None, *, world: int,
                   qchunk: int, h_local: int, transport: str,
                   ) -> tuple[torch.Tensor | None, torch.Tensor | None,
                              torch.Tensor | None]:
    """``(send, recv, gathered)`` from a workspace, checked against this call.

    The geometry is checked here rather than left to the buffer checks, so a
    workspace built for another layer's geometry is named as such instead of
    surfacing as a shape mismatch on whichever tensor happened to be validated
    first.
    """
    if workspace is None:
        return None, None, None
    if (workspace.world, workspace.qchunk, workspace.h_local) != (
            world, qchunk, h_local):
        raise ValueError(
            f"the workspace was allocated for (W, Qchunk, Hlocal) = "
            f"({workspace.world}, {workspace.qchunk}, {workspace.h_local}) but "
            f"this call is ({world}, {qchunk}, {h_local}). The carrier's "
            "element counts are a pure function of that triple, so a "
            "workspace is not resizable -- allocate one per geometry."
        )
    if transport == "all_gather" and world > 1 and workspace.gathered is None:
        raise ValueError(
            "this workspace holds no `gathered` buffer, so it was allocated "
            "for the all_to_all route. The all_gather reference stages W times "
            "the receive carrier, which is not allocated for a transport the "
            "workspace was not built for."
        )
    return workspace.send, workspace.recv, workspace.gathered


def pack_send_carrier_native(local_cand: torch.Tensor, out: torch.Tensor,
                             world: int) -> torch.Tensor:
    """:func:`pack_send_carrier` as one native launch into ``out``. No ATen."""
    from . import _build  # noqa: PLC0415

    _build._pack().pack_send_carrier(local_cand, out, int(world))
    return out


def collective_exchange_and_merge(local_cand: torch.Tensor, *, world: int,
                                  rank: int, out: torch.Tensor,
                                  forced: torch.Tensor | None = None,
                                  n_ordinary: torch.Tensor | None = None,
                                  status: torch.Tensor | None = None,
                                  transport: str = "all_to_all",
                                  group: dist.ProcessGroup | None = None,
                                  merge_fn: Callable[..., None] | None = None,
                                  workspace: CarrierWorkspace | None = None,
                                  ) -> torch.Tensor:
    """Native pack -> exchange -> K2 merge, over the NCCL collective.

    Returns ``status``, the C5 failure word (allocated and zeroed here only if
    the caller supplied none; never read here). ``forced``/``n_ordinary`` are
    C3's per-row int32 ``[Qchunk]`` planes. ``head_offset = rank * H_local``
    reaches K2 only so it can assert C7.

    With a ``workspace`` and a ``status`` this allocates nothing and issues no
    ATen op: the pack is ``icp_pack``, the transport is
    ``all_to_all_single`` into the retained receive carrier, and the merge is
    a direct ``k2_merge`` launch. Without a workspace the carriers are
    allocated per call (tests and scripts only).

    ``merge_fn`` overrides the merge binding for gates that drive a separately
    built extension.
    """
    from . import _build  # noqa: PLC0415

    if status is None:
        status = torch.zeros(1, dtype=torch.int32, device=out.device)
    qchunk, h_group = int(local_cand.shape[0]), int(local_cand.shape[1])
    h_local = h_group // world
    send_buf, recv_buf, gathered_buf = _workspace_for(
        workspace, world=world, qchunk=qchunk, h_local=h_local,
        transport=transport)
    if send_buf is None:
        send_buf = torch.empty(send_carrier_shape(world, qchunk, h_local),
                               dtype=torch.int32, device=local_cand.device)
    send = pack_send_carrier_native(local_cand, send_buf, world)
    recv = exchange_carrier(send, world=world, rank=rank, transport=transport,
                            group=group, out=recv_buf, gathered=gathered_buf)
    merge = _build._k2().k2_merge if merge_fn is None else merge_fn
    merge(recv, out, rank * h_local, world, rank, forced, n_ordinary, status)
    return status
