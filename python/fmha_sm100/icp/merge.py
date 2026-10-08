"""The ICP candidate merge (K2) and a torch reference for the C5 key.

This is the *unfused* path: whoever produced the candidates packs them into the
C4 carrier, exchanges them (:mod:`fmha_sm100.icp.carrier`) and calls
:func:`merge_candidates`. :class:`fmha_sm100.icp.IcpExchange` fuses the exchange
and the merge into one launch and must produce a **bit-identical** result --
``tests/test_exchange.py::test_exchange_is_bit_identical_to_the_reference_merge``
gates exactly that, and it can be a ``torch.equal`` only because both paths
compile the same ``merge_topk.cuh``.

Host ABI
--------

``refined-icp-v1.k2.2``. Eight arguments, none of them defaulted:

    k2_merge(cand, out, head_offset, world, rank, forced, n_ordinary, status)

``forced``/``n_ordinary`` are the C3 per-row planes and may be ``None``
*explicitly*, which selects the plain ordinary merge with no reservation.
``status`` is the C5 failure word and is **required**: an optional failure
channel is an opt-out from a MUST, and the default of that opt-out is the old
behaviour in which a NaN candidate silently wins the merge.

Shapes (CONTRACT C4/C6)
-----------------------

``cand``
    int32 ``[W, Qchunk, H_local, 16, 2]`` -- the **source-major receive
    carrier**. ``[..., 0]`` is the fp32 score's raw bits and ``[..., 1]`` is the
    int32 global block id, both *bitcast*, never converted, because an id above
    2^24 would not survive a float conversion. An invalid entry is
    ``(-inf, -1)``.

    Axis 2 is ``H_local``, the destination's **own** heads -- not ``H_group``.
    The pre-refinement fp32 ``[C, T, H_group, 16, 2]`` gathered tensor is
    *refused* by the kernel rather than reinterpreted, because the two differ in
    the head stride and a reinterpretation would merge a peer's heads under this
    rank's name with no symptom.
``out``
    int32 ``[Qchunk, H_local, 16]``: global block ids, **ascending**, ``-1`` at
    the tail. C7 pins ``head_offset = icp_rank * H_local``; under C4 it no
    longer addresses anything (the carrier *is* the slice) and is passed only so
    the kernel can assert it.

Duplicate ids are the normal case
---------------------------------

Under fragment placement every rank scores every logical block over its own
``R = 128/W`` rows and publishes a *partial maximum*, so the same global block
id arrives from several sources. The merge max-reduces duplicates by id
**before** truncating to 16. Truncating first -- which is what the pre-refined
merge and the M0 oracle both did -- drops winners.

Arms, and their tiers
---------------------

:class:`MergeArm` names the three implementations of the same C3/C5/C6
semantics. :data:`PRODUCTION_ARMS` holds exactly one, ``CUDA``; the Triton and
torch arms in :mod:`fmha_sm100.icp.merge_reference` are :data:`REFERENCE_ARMS` --
they exist to define and check correct behaviour, and they are opt-in by name
only. :func:`merge_backend` is the single resolution point: ``auto`` resolves to
a production arm or **raises**, never to a reference one, and an explicit
reference arm warns with :class:`IcpReferenceArmWarning` and is recorded by
:func:`last_merge_arm` so a result artifact can carry it.
"""

from __future__ import annotations

import enum
import os
import warnings

import torch

from . import _build

CAND_K = 16

#: Status bits ORed into the caller's failure word. Any non-zero value **fails
#: the invocation**; the bits are distinguished only so the failure can be
#: diagnosed. Mirrored from ``merge_topk.cuh`` by the built extension's
#: ``k2_status_nan`` / ``k2_status_row_meta`` attributes, which
#: ``tests/test_merge_key.py`` asserts these against rather than trusting this
#: comment.
ICP_STATUS_OK = 0
ICP_STATUS_NAN = 1  # C5: a NaN score on a record whose id is valid.
ICP_STATUS_ROW_META = 2  # C3: n_ordinary disagrees with min(15, M-1).

#: The forced-block sentinel. Also an inactive row's forced id, which pairs with
#: ``n_ordinary == 0`` and yields zero valid ids (C6).
ICP_NO_FORCED_BLOCK = -1

#: The C5 key is a uint64. Both :func:`canonical_key_reference` and the CUDA
#: probe return it biased by this amount, as a signed int64, so that a *signed*
#: compare reproduces the contract's *unsigned* compare. Undo it to recover the
#: literal contract value: ``int(biased) + 2**63``.
KEY_BIAS = 1 << 63

_U32 = 0xFFFFFFFF


class IcpMergeError(RuntimeError):
    """A merge invocation failed its numerical contract (C5) or its row
    metadata (C3).

    It is an error, not a fallback: the output rows it names carry no selection,
    and no selection derived from this invocation may reach attention or token
    commit.
    """

    def __init__(self, status: int):
        self.status = int(status)
        names = []
        if self.status & ICP_STATUS_NAN:
            names.append(
                "NaN score on a valid candidate (C5: NaN fails the invocation)"
            )
        if self.status & ICP_STATUS_ROW_META:
            names.append(
                "n_ordinary does not equal C3's min(15, M-1) for the row's "
                "forced block"
            )
        if not names:
            names.append(f"unknown status bits {self.status:#x}")
        super().__init__(
            f"ICP merge failed (status={self.status:#x}): " + "; ".join(names)
        )


def forced_rows(
    positions: torch.Tensor,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C3 row metadata from each row's own query position ``p``.

    ``positions`` is int32/int64 ``[Qchunk]``: row ``t``'s **exact global query
    position**. A batch-wide sequence length is never a substitute -- prefill,
    chunked-prefill, decode and ``Q > 1`` verification rows sit in one
    invocation with different ``p``, which is the whole reason these are planes
    and not scalars. ``active`` is an optional bool ``[Qchunk]``; inactive rows
    get ``(-1, 0)`` and select nothing.

    Returns ``(forced, n_ordinary)`` int32 ``[Qchunk]``::

        f          = p // 128
        M          = ceil((p+1)/128) = f + 1
        n_ordinary = min(15, M - 1) = min(15, f)

    The kernel re-derives the same formula and fails the invocation when the two
    disagree. **A caller with its own count should pass that count rather than
    call this helper** -- the cross-check is only worth something when it can
    fire, and a caller that derives both sides from here has made it vacuous for
    itself.
    """
    pos = positions.to(torch.int64)
    forced = torch.div(pos, 128, rounding_mode="floor")
    n_ordinary = torch.clamp(forced, min=0, max=CAND_K - 1)
    if active is not None:
        forced = torch.where(active, forced, torch.full_like(forced, -1))
        n_ordinary = torch.where(active, n_ordinary,
                                 torch.zeros_like(n_ordinary))
    forced = torch.where(forced < 0, torch.full_like(forced, -1), forced)
    n_ordinary = torch.where(forced < 0, torch.zeros_like(n_ordinary),
                             n_ordinary)
    return forced.to(torch.int32), n_ordinary.to(torch.int32)


def canonical_key_reference(scores: torch.Tensor,
                            ids: torch.Tensor) -> torch.Tensor:
    """CONTRACT C5's canonical key, in torch, for tests and documentation.

    ``key = (sortable(score) << 32) | ~uint32(id)``, with ``id < 0 -> 0`` and
    ``-0.0`` flushed to ``+0.0`` before the bits are taken. Score descending,
    then global block id ascending.

    Returned **biased by** :data:`KEY_BIAS` and signed, so ``sort``/``argsort``
    /``>`` on the result are the contract's ordering. Descending order of this
    tensor is the selection order.

    This is a second implementation of the key on purpose: the shipped one
    lives in ``csrc/merge_topk.cuh`` and nowhere else, and
    ``tests/test_merge_key.py`` gates this against it via the ``canonical_key``
    probe in the K2 extension. It is *not* a helper the kernels use; nothing in
    the fast path calls it.

    NaN is rejected rather than ordered (C5 forbids it). The check reads back
    from the device, so this function is not cudagraph-safe -- it is a
    reference, not a kernel.
    """
    if scores.shape != ids.shape:
        raise ValueError(
            f"scores {tuple(scores.shape)} and ids {tuple(ids.shape)} must "
            "have the same shape"
        )
    if scores.dtype != torch.float32:
        raise TypeError(f"scores must be float32, got {scores.dtype}")
    if ids.dtype != torch.int32:
        raise TypeError(f"ids must be int32, got {ids.dtype}")
    if torch.isnan(scores).any():
        raise ValueError(
            "NaN score reached the C5 key; CONTRACT C5 forbids it. The kernel "
            "does not silently discard it either: it sets ICP_STATUS_NAN and "
            "fails the invocation."
        )

    score_bits = scores.contiguous().view(torch.int32).to(torch.int64) & _U32
    # C5 amendment: -0.0 and +0.0 are equal as floats, so they must be equal as
    # keys. Without this flush they get 0x7fffffff and 0x80000000 and the
    # selector orders two values IEEE-754 declares equal -- and since a block
    # score is a max reduction, whose zero sign IEEE leaves implementation-
    # defined, two numerically identical runs could then disagree.
    score_bits = torch.where((score_bits & 0x7FFFFFFF) == 0,
                             torch.zeros_like(score_bits), score_bits)
    # IEEE-754 float bits are not monotonic as unsigned integers: negatives run
    # backwards and sit above positives. The standard fix-up makes them
    # order-preserving -- flip every bit of a negative, flip only the sign bit
    # of a positive -- so a plain unsigned compare on `sortable` is a float
    # compare.
    sortable = torch.where(
        (score_bits & 0x80000000) != 0,
        score_bits ^ _U32,
        score_bits ^ 0x80000000,
    )
    # The low half is the *complement* of the id, so that within one score a
    # larger key means a smaller id: score descending, then block id ascending.
    low = (~ids.to(torch.int64)) & _U32
    # An invalid candidate is keyed to 0 outright -- below every real key, so
    # padding can never be selected regardless of the score bits it carries.
    invalid = ids < 0
    sortable = torch.where(invalid, torch.zeros_like(sortable), sortable)
    low = torch.where(invalid, torch.zeros_like(low), low)
    # (sortable << 32 | low) - 2**63, written so no intermediate overflows
    # int64: `sortable - 2**31` is in [-2^31, 2^31), so the product is in
    # [-2^63, 2^63 - 2^32] and adding `low < 2^32` still fits.
    return (sortable - (1 << 31)) * (1 << 32) + low


def canonical_key_cuda(scores: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The **shipped** C5 key, straight out of ``merge_topk.cuh``.

    A thin probe over ``icp::canonical_key`` so a test can compare the key the
    kernels actually use against :func:`canonical_key_reference` without a
    third transcription existing anywhere. Same bias, so the two are directly
    ``torch.equal``. CUDA only.
    """
    scores = scores.contiguous()
    ids = ids.contiguous()
    out = torch.empty(scores.shape, dtype=torch.int64, device=scores.device)
    _build._k2().canonical_key(scores.reshape(-1), ids.reshape(-1),
                               out.reshape(-1))
    return out


# --------------------------------------------------------------------------
# Arms, and the one place an arm is chosen
# --------------------------------------------------------------------------


class MergeArm(enum.Enum):
    """Which implementation of the C3/C5/C6 merge runs.

    ``CUDA``
        ``csrc/k2_merge.cu`` over ``csrc/merge_topk.cuh``. The production path.
    ``TRITON`` / ``TORCH``
        :mod:`fmha_sm100.icp.merge_reference`. **REFERENCE tier**: they exist to
        define and check correct behaviour, they are opt-in by name only, and
        they are not production backends.

    The split is a *tier*, not a preference ranking. Nothing resolves to a
    reference arm on a caller's behalf -- see :func:`merge_backend`.
    """

    CUDA = "cuda"
    TRITON = "triton"
    TORCH = "torch"


#: The shipping tier. ``auto`` resolves here and nowhere else.
PRODUCTION_ARMS = frozenset({MergeArm.CUDA})

#: The reference tier: :mod:`fmha_sm100.icp.merge_reference`.
REFERENCE_ARMS = frozenset({MergeArm.TRITON, MergeArm.TORCH})

#: Every name :func:`merge_backend` accepts, in the order it lists them when it
#: refuses one. ``auto`` is a *resolution rule*, not an arm, so it has no
#: :class:`MergeArm` member.
MERGE_ARM_NAMES = ("auto", *(arm.value for arm in MergeArm))

#: Sets the default arm without touching a call site. Same refuse-by-name
#: discipline as the ``arm`` argument, and ``auto`` remains the default.
MERGE_ARM_ENV = "ICP_MERGE_ARM"


class IcpReferenceArmWarning(UserWarning):
    """A REFERENCE-tier merge arm was selected by name.

    Not advice and not a deprecation: the selection was honoured exactly as
    asked. It is emitted because running a reference implementation is a
    deliberate, visible act, and because a result taken on one must say so --
    :func:`last_merge_arm` is what a result artifact should record.
    """


_last_arm: MergeArm | None = None

#: Arms already warned about. Module state, so the warning is *one-time*
#: whatever the ambient ``warnings`` filters are. ``tests/test_merge_key.py``
#: clears it, which is the only supported reason to touch it.
_WARNED_REFERENCE_ARMS: set[MergeArm] = set()


def last_merge_arm() -> MergeArm | None:
    """The arm :func:`merge_backend` resolved most recently, or ``None``.

    Record it beside any result. A number whose arm is not recorded cannot be
    attributed to an implementation, and the whole point of the reference tier
    is that it produces the *same* numbers -- so no output comparison can
    recover the arm after the fact.
    """
    return _last_arm


def _require_k2() -> None:
    """Raise if the CUDA arm cannot take a call at all."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device is visible to this process "
            "(torch.cuda.is_available() is False)"
        )
    module = _build._k2()
    if not hasattr(module, "k2_merge"):
        raise RuntimeError(
            "the built K2 extension exports no k2_merge; it is not the "
            f"refined-icp-v1.k2.2 endpoint (exports: {sorted(dir(module))})"
        )


def _warn_reference_arm(arm: MergeArm, *, from_env: bool) -> None:
    if arm in _WARNED_REFERENCE_ARMS:
        return
    _WARNED_REFERENCE_ARMS.add(arm)
    via = f" (selected through {MERGE_ARM_ENV})" if from_env else ""
    warnings.warn(
        f"the {arm.value!r} merge arm{via} is REFERENCE tier, not a production "
        "backend: it exists to define and check correct C5/C6 behaviour, and "
        "the production path is the CUDA arm (K2). The selection was honoured; "
        "record fmha_sm100.icp.merge.last_merge_arm() beside any result taken "
        "with it.",
        IcpReferenceArmWarning,
        stacklevel=3,
    )


def merge_backend(requested: MergeArm | str | None = None) -> MergeArm:
    """Resolve a request to a :class:`MergeArm`, and record it.

    ===================  ====================================================
    ``requested``        result
    ===================  ====================================================
    ``None``             :data:`MERGE_ARM_ENV` if set, else ``"auto"``; then
                         the row for that value, refusing an unknown one by
                         name.
    ``"auto"``           ``CUDA`` if K2 is usable, else **raises**, naming the
                         underlying reason and chaining it. Never a reference
                         arm.
    ``"cuda"``           ``CUDA`` if K2 is usable, else **raises**. Never
                         degrades.
    ``"triton"``         ``TRITON``, after a one-time
                         :class:`IcpReferenceArmWarning`.
    ``"torch"``          ``TORCH``, after a one-time
                         :class:`IcpReferenceArmWarning`.
    a :class:`MergeArm`  the same as its ``.value``.
    anything else        ``ValueError`` naming the value and listing
                         :data:`MERGE_ARM_NAMES`. Never silently defaulted.
    ===================  ====================================================

    The one thing this function will not do is put production traffic on a
    reference implementation without being told to, by name. A fallback here
    would be undetectable downstream: the arms are bit-exact by construction,
    so no output comparison can tell which one ran.
    """
    global _last_arm

    from_env = False
    if requested is None:
        env = os.environ.get(MERGE_ARM_ENV)
        if env:
            requested, from_env = env, True
        else:
            requested = "auto"

    if isinstance(requested, MergeArm):
        name = requested.value
    else:
        name = str(requested).strip().lower()
    if name not in MERGE_ARM_NAMES:
        where = (f"{MERGE_ARM_ENV}={requested!r}" if from_env
                 else f"arm={requested!r}")
        raise ValueError(
            f"unknown merge arm: {where}. Legal values are "
            f"{', '.join(repr(n) for n in MERGE_ARM_NAMES)}. An unrecognised "
            "name is refused rather than defaulted: silently running 'auto' "
            "for a misspelled 'trtion' would report one arm and measure "
            "another, and the arms are bit-exact, so nothing downstream could "
            "catch it."
        )

    if name in ("auto", MergeArm.CUDA.value):
        try:
            _require_k2()
        except Exception as exc:
            if name == "auto":
                raise RuntimeError(
                    "merge arm 'auto' resolves to a production arm "
                    f"({', '.join(sorted(a.value for a in PRODUCTION_ARMS))}) "
                    f"only, and the CUDA arm (K2) is unusable here: {exc}. "
                    "`auto` will not silently route production traffic onto a "
                    "reference implementation, so it raises instead. Fix the "
                    "build, or ask for "
                    f"{' / '.join(sorted(repr(a.value) for a in REFERENCE_ARMS))} "
                    "by name -- which is a deliberate, visible act and warns."
                ) from exc
            # S7/vLLM's justification for raising instead of degrading, kept
            # verbatim: "a benchmark that silently measured Triton while
            # reporting 'K2' would be worse than a crash".
            raise RuntimeError(
                f"merge arm 'cuda' was requested but K2 is unusable: {exc}. "
                "An explicit 'cuda' raises rather than degrading to a "
                "reference arm: a benchmark that silently measured Triton "
                "while reporting 'K2' would be worse than a crash."
            ) from exc
        arm = MergeArm.CUDA
    else:
        arm = MergeArm(name)
        _warn_reference_arm(arm, from_env=from_env)

    _last_arm = arm
    return arm


# --------------------------------------------------------------------------
# The host-side contract, shared by all three arms
# --------------------------------------------------------------------------


def _prepare(cand: torch.Tensor, world: int, rank: int,
             out: torch.Tensor | None, forced: torch.Tensor | None,
             n_ordinary: torch.Tensor | None):
    """Validate one merge invocation and return ``(cand, out, W, rank, Q, Hl)``.

    Every arm goes through this, so the *argument* contract is one
    implementation even though the merge is three. An arm-dependent refusal
    would be a second way for the arms to disagree, and the point of the
    reference tier is that they cannot.
    """
    if cand.dim() != 5:
        raise ValueError(
            "cand must be the C4 receive carrier [W, Qchunk, H_local, 16, 2]; "
            f"got {tuple(cand.shape)}"
        )
    if cand.dtype != torch.int32:
        raise TypeError(
            "cand must be int32: refined-icp-v1 C4 replaced the fp32 "
            "[C, T, H_group, 16, 2] gathered tensor with the int32 "
            "[W, Qchunk, H_local, 16, 2] carrier. The wire content is the same "
            "two bitcast words, but axis 2 is H_LOCAL now, so a float32 tensor "
            "here is an unpacked carrier and is refused rather than "
            f"reinterpreted; got {cand.dtype}"
        )
    if not cand.is_cuda:
        raise ValueError("cand must be a CUDA tensor")
    sources, tokens, heads_local, cand_k, pair = cand.shape
    if cand_k != CAND_K or pair != 2:
        raise ValueError(
            f"cand must end in [16, 2] (score bits, id); got [{cand_k}, {pair}]"
        )
    world = int(world)
    rank = int(rank)
    if world != sources:
        raise ValueError(
            f"world={world} must equal the carrier's source axis W={sources}"
        )
    if not 0 <= rank < world:
        raise ValueError(f"rank must lie in [0, {world}); got {rank}")
    # One warp merges the whole candidate set for a row, so the W * 16
    # candidates must fit its budget.
    if sources * CAND_K > 128:
        raise ValueError(
            f"W={sources} exceeds the merge's 128-candidate warp budget (W<=8)"
        )
    if (forced is None) != (n_ordinary is None):
        raise ValueError(
            "forced and n_ordinary must be supplied together (C3), or both "
            "omitted for the plain ordinary merge"
        )

    if out is None:
        out = torch.empty((tokens, heads_local, CAND_K), dtype=torch.int32,
                          device=cand.device)
    else:
        if tuple(out.shape) != (tokens, heads_local, CAND_K):
            raise ValueError(
                f"out must be [{tokens}, {heads_local}, {CAND_K}]; got "
                f"{tuple(out.shape)}"
            )
        if out.dtype != torch.int32:
            raise TypeError(f"out must be int32, got {out.dtype}")

    return cand.contiguous(), out, world, rank, tokens, heads_local


def _acquire_status(status: torch.Tensor | None, device) -> tuple[torch.Tensor,
                                                                 bool]:
    """The C5 failure word, and whether this call owns it.

    Owned means allocated-and-zeroed here and read back at the end, which
    synchronises and is therefore refused under an active capture. Supplied
    means neither cleared nor read: no device->host sync, and the caller checks
    it at its own completion edge.
    """
    if status is not None:
        if status.dtype != torch.int32:
            raise TypeError(f"status must be int32, got {status.dtype}")
        if status.numel() < 1:
            raise ValueError("status must have at least one element")
        return status, False
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "a captured invocation must supply its own status tensor: the "
            "wrapper-owned form reads the word back, which is a "
            "device->host sync and is not available during CUDA-graph "
            "capture. Allocate an int32 status word, zero it inside the "
            "capture, pass it here, and check it after the replay. The "
            "alternative -- skipping the C5 check under capture -- is not "
            "offered."
        )
    return torch.zeros(1, dtype=torch.int32, device=device), True


def _raise_on_status(status: torch.Tensor, owned: bool) -> None:
    if not owned:
        return
    code = int(status.item())  # synchronises; see merge_candidates' docstring
    if code != ICP_STATUS_OK:
        raise IcpMergeError(code)


def merge_candidates(cand: torch.Tensor, *, world: int, rank: int,
                     out: torch.Tensor | None = None,
                     forced: torch.Tensor | None = None,
                     n_ordinary: torch.Tensor | None = None,
                     status: torch.Tensor | None = None,
                     arm: MergeArm | str | None = None) -> torch.Tensor:
    """Merge an exchanged C4 carrier into this rank's Top-16 (CONTRACT C6).

    Parameters
    ----------
    cand
        int32 ``[W, Qchunk, H_local, 16, 2]``, the **source-major receive
        carrier** -- the output of :func:`fmha_sm100.icp.exchange_carrier`. See the
        module docstring for the encoding and for why the pre-refinement fp32
        gathered tensor is refused rather than reinterpreted.
    world, rank
        ``W`` and this rank's index. ``world`` must equal ``cand.shape[0]``, and
        ``head_offset = rank * H_local`` is re-derived here and re-checked by
        the kernel (C7).
    out
        Optional int32 ``[Qchunk, H_local, 16]`` destination. Allocated if
        omitted -- which a captured caller must not do.
    forced, n_ordinary
        C3's per-row int32 ``[Qchunk]`` planes, or both ``None`` for the plain
        ordinary merge. Supplying one without the other is refused: a caller
        that thinks it is forcing and is not is precisely the bug class C3
        replaced. See :func:`forced_rows`.
    status
        C5's int32 failure word, at least one element.

        **Omitted:** this function allocates it, zeroes it, runs, reads it back
        -- *which synchronises* -- and raises :class:`IcpMergeError` on any
        non-zero bit. Convenient, and illegal during CUDA-graph capture, where
        it is refused rather than silently skipped.

        **Supplied:** neither cleared nor read here, so there is no device->host
        sync and the invocation is capturable. Zeroing it before the launch and
        checking it after belong to the caller. A status word nobody reads is a
        failure channel that does not exist.
    arm
        Which :class:`MergeArm` runs, resolved by :func:`merge_backend`.
        ``None`` is ``ICP_MERGE_ARM`` or ``auto``, i.e. the CUDA arm or an
        exception -- never a reference arm. ``"triton"``/``"torch"`` select
        :mod:`fmha_sm100.icp.merge_reference` and warn; they are reference-tier
        and produce the same bits, including on the failure path, which is what
        ``tests/test_merge_key.py`` gates.

    Returns
    -------
    ``out``: int32 ``[Qchunk, H_local, 16]``, ascending, ``-1``-padded. A failed
    row is published as an all ``-1`` row -- never a stale or plausible
    selection, and defined, so the CUDA and reference arms stay bit-comparable
    on the failure path too.
    """
    cand, out, world, rank, tokens, heads_local = _prepare(
        cand, world, rank, out, forced, n_ordinary)
    status, owned_status = _acquire_status(status, cand.device)

    chosen = merge_backend(arm)
    if chosen is MergeArm.CUDA:
        _build._k2().k2_merge(cand, out, rank * heads_local, world, rank,
                              forced, n_ordinary, status)
    else:
        # Imported here, not at module scope: the reference arms pull in
        # torch-heavy (and, for Triton, optional) machinery that the production
        # path must not pay for or depend on.
        from . import merge_reference  # noqa: PLC0415

        kernel = (merge_reference._triton_merge if chosen is MergeArm.TRITON
                  else merge_reference._torch_merge)
        kernel(cand, out, forced, n_ordinary, status)

    _raise_on_status(status, owned_status)
    return out
