"""REFERENCE implementations of the ICP merge -- **not** production backends.

These two arms exist to *define* and to *check* correct C5/C6 behaviour: they
are the second and third statements of the same contract, so that
``csrc/merge_topk.cuh`` is gated against something other than a paraphrase of
itself, and so that the C3 reserved slot and the C5 failure path are written
down in a form a reader can follow line by line. They are **opt-in only** --
:func:`fmha_sm100.icp.merge.merge_backend` resolves ``auto`` to a production arm or
raises, and selecting one of these by name warns
(:class:`~fmha_sm100.icp.merge.IcpReferenceArmWarning`) and is recorded by
:func:`~fmha_sm100.icp.merge.last_merge_arm`.

The production path is the CUDA arm (K2, ``csrc/k2_merge.cu``). A **fused**
implementation -- the merge folded into the exchange, as K5 already is for the
pre-refinement semantics -- is future work, and it is where the next
performance comes from. Neither arm here is a fallback for it, and neither is a
step towards it.

What they reproduce, exactly
----------------------------

Three things, all three of them load-bearing and all three gated at
``torch.equal`` against the CUDA arm in ``tests/test_merge_key.py`` section 7:

1. **The duplicate-id max-reduce, before truncation (C3).** One *representative* is
   elected per distinct global block id -- the duplicate carrying the greatest
   canonical key, exact ties broken by the smaller flat candidate index -- and
   only representatives are ranked and placed. Under fragment placement every
   rank publishes a partial maximum for *every* block, so duplicates are the
   normal case; truncating to 16 records first drops winners.
2. **The C3 reserved forced slot.** ``forced`` is excluded from the ordinary
   ranking, at most ``n_ordinary`` ordinary winners are kept, and ``forced`` is
   injected exactly once at the position its ascending id demands. ``+inf`` is
   **not** a forcing mechanism: a legitimate ordinary ``+inf`` with a smaller id
   ties with the forced block under C5 and wins the ascending-id tie-break.
3. **The C5 NaN failure path.** A NaN score on a record whose **id is valid**
   sets ``ICP_STATUS_NAN`` and fails the invocation; validity is by id, never by
   score, so an invalid record's score field is never examined. A failed row
   publishes an all ``-1`` row -- never a stale or plausible selection, and
   *defined*, which is what lets the three arms stay bit-comparable on the
   failure path too.

The flat index must agree with the kernel
-----------------------------------------

``merge_topk.cuh``'s ``load_candidates_c4`` enumerates ``idx = lane + i*32``
with ``s = idx / 16`` and ``k = idx - s*16``, and ``warp_merge_topk16`` breaks
exact key ties on that same ``lane + i*32``. Both arms here flatten the C4
carrier's ``(source, k)`` pair **source-major**, ``n = s*16 + k``, which is the
same enumeration of the same pool -- so the three arms elect the same
representative. Which duplicate is elected cannot change the emitted ids (they
are equal by definition), but it must be *exactly one*, or the placement pass
collides and a winner is silently dropped.

Two deliberate properties, so they are not "tidied" away
--------------------------------------------------------

* **Triton is never imported at module scope** -- only inside
  :func:`_merge_kernel` and :func:`_triton_merge`, i.e. only once the Triton arm
  is actually called -- so a host with no triton, or no GPU, can still import
  this module, read it and run the torch arm. A reference you cannot read
  without the thing it is a reference for is not much of one.
* **:func:`_stable_key` is not** :func:`fmha_sm100.icp.merge.canonical_key_reference`.
  Two reasons, and both matter: that one *raises* on a NaN, and this one must
  survive one (C5 fails the invocation by computing a status word, not by
  throwing out of the middle of a merge); and a second expression of the key is
  the only way a bug *in* the key can be witnessed rather than hidden in both
  copies of it. The key itself is gated against the shipped CUDA one.

Two places the arms are NOT identical, stated rather than left to be found
-------------------------------------------------------------------------

Both are outside what the gate can reach, which is exactly why they are written
here instead of relied on.

1. **``W`` outside {1, 2, 4}.** K2 instantiates only ``KPT`` 1, 2 and 4, so it
   accepts ``W*16`` of 16, 32, 64 or 128 and ``TORCH_CHECK``-fails on
   ``W ∈ {5, 6, 7}``; both arms here would run those happily. The host contract
   only enforces ``W*16 <= 128``. It is unreachable while ``H_local = 4/W``
   restricts ``W`` to {1, 2, 4}, and closing it would move the CUDA arm's
   existing failure site, so it is recorded rather than papered over.
2. **The status word's atomicity.** The CUDA and Triton arms ``atomicOr`` into
   it; the torch arm does a plain read-modify-write. Identical for one
   invocation -- which is all the gate exercises -- but two concurrent merges on
   different streams sharing one status word could lose a bit on the torch arm
   only.

Neither is a reason to prefer an arm; both are reasons not to treat "the arms
are bit-exact" as meaning "the arms are interchangeable".

The C4 conversion
-----------------

These arms' pre-refinement form consumed the fp32 ``[C, T, H_group, 16, 2]``
gathered tensor and sliced ``[head_offset, head_offset + H_local)`` out of axis
2. They now
consume the int32 C4 **receive** carrier ``[W, Qchunk, H_local, 16, 2]``,
source-major, whose axis 2 is already this rank's own heads. So there is no head
slicing and ``head_offset`` is not an index any more: a source-major carrier
physically cannot address a peer's heads. The records are unchanged -- word 0 is
the fp32 score's raw bits and word 1 the int32 global block id, both **bitcast**,
never converted, because an id at or above ``2**24`` does not survive a float
round trip.
"""

from __future__ import annotations

import torch

from .merge import (
    ICP_NO_FORCED_BLOCK,
    ICP_STATUS_NAN,
    ICP_STATUS_ROW_META,
    MergeArm,
    _acquire_status,
    _prepare,
    _raise_on_status,
    merge_backend,
)

__all__ = ["merge_candidates_torch", "merge_candidates_triton"]


# --------------------------------------------------------------------------
# the torch arm
# --------------------------------------------------------------------------


def _stable_key(score: torch.Tensor, block_id: torch.Tensor) -> torch.Tensor:
    """CONTRACT C5's canonical key as int64, biased by ``-2**63``.

    Deliberately **not** :func:`fmha_sm100.icp.merge.canonical_key_reference`,
    which *raises* on a NaN. This is the merge's own key and it has to survive a
    NaN: C5 says a NaN on a valid record fails the invocation, and failing an
    invocation means computing a status word and publishing an all ``-1`` row,
    not throwing out of the middle of a kernel. The two are otherwise the same
    expression, and ``tests/test_merge_key.py`` gates the key itself against the
    shipped CUDA one.
    """
    # .view(dtype) at equal itemsize is a pure reinterpretation of the same
    # storage -- the bitcast C4 demands, with no rounding.
    bits = score.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    # Flush -0.0 to +0.0 BEFORE the mapping. Without it the two IEEE-equal
    # zeros get different keys (0x7fffffff vs 0x80000000) and the selector
    # orders two values the contract calls equal. A block score is a max
    # reduction and IEEE-754 leaves the sign of fmax(-0.0, +0.0)
    # implementation-defined, so an unflushed key can make two numerically
    # identical runs disagree. Upstream this omission was a live CUDA-vs-Triton
    # divergence (stable_key(-0.0, 0): 0x7fffffff_ffffffff vs
    # 0x80000000_ffffffff) and it survived for weeks because the test's own
    # oracle shared it. merge_topk.cuh:105 is the authority.
    bits = torch.where((bits & 0x7FFFFFFF) == 0, torch.zeros_like(bits), bits)
    # IEEE-754 fp32 is sign-magnitude: negatives run backwards and sit above
    # positives as raw integers. Flip every bit of a negative and only the sign
    # bit of a non-negative, and one unsigned compare is the float compare.
    sortable = torch.where(bits >= 0x80000000, bits ^ 0xFFFFFFFF,
                           bits ^ 0x80000000)
    # The id COMPLEMENTED in the low half: at equal score the larger key is the
    # smaller id, so one `>` means score descending then block id ascending.
    low = (~block_id.to(torch.int64)) & 0xFFFFFFFF
    valid = block_id >= 0
    zero = torch.zeros((), dtype=torch.int64, device=score.device)
    sortable = torch.where(valid, sortable, zero)
    low = torch.where(valid, low, zero)
    # (sortable << 32 | low) - 2**63: the unsigned -> two's-complement
    # reinterpretation, which is order-preserving, so torch's *signed* compare
    # is the contract's *unsigned* one.
    return ((sortable - 0x80000000) << 32) | low


def _torch_merge(cand: torch.Tensor, out: torch.Tensor,
                 forced: torch.Tensor | None,
                 n_ordinary: torch.Tensor | None,
                 status: torch.Tensor) -> torch.Tensor:
    """The eager arm. Slow and obvious, which is what a reference has to be.

    ``cand`` is the int32 C4 receive carrier ``[W, Qchunk, H_local, 16, 2]``;
    ``out`` is int32 ``[Qchunk, H_local, 16]``; ``status`` is the C5 word, ORed
    into and never cleared or read here (no device->host sync, so this is
    capturable exactly like the CUDA arm).
    """
    W, Q, Hl, K, _ = cand.shape
    if Q == 0 or Hl == 0:
        return out  # the kernel returns before its launch for the same reason
    n_cand = W * K

    # Bitcast the whole carrier and then take the lane. Both words are
    # reinterpreted, never converted (C4): word 0 holds the fp32 score's raw
    # bits and word 1 the int32 global block id. Slicing before the .view()
    # would leave a stride-2 last axis.
    score = cand.view(torch.float32)[..., 0]
    gid = cand[..., 1]

    # Flatten (source, k) SOURCE-major into one candidate axis: n = s*K + k.
    # That is `load_candidates_c4`'s own enumeration (`s = idx / K`,
    # `k = idx - s*K`), which is what makes the positional tie-break below
    # elect the same representative the kernel elects. Axis 2 is H_local, so
    # there is no head slicing here -- the receive carrier IS this rank's slice.
    score = score.permute(1, 2, 0, 3).reshape(Q, Hl, n_cand).contiguous()
    gid = gid.permute(1, 2, 0, 3).reshape(Q, Hl, n_cand).contiguous()

    # --- C3 row metadata and the C5 NaN failure -----------------------------
    # `f`/`q` are [Q, 1, 1] so they broadcast over the head and candidate axes;
    # forcing is a function of the token row's query position only, never of
    # the head.
    if forced is None:
        f = torch.full((Q, 1, 1), ICP_NO_FORCED_BLOCK, dtype=torch.int64,
                       device=gid.device)
        q = torch.full((Q, 1, 1), K, dtype=torch.int64, device=gid.device)
        err = torch.zeros((Q, Hl), dtype=torch.int32, device=gid.device)
    else:
        f = forced.to(torch.int64).reshape(Q, 1, 1)
        q = n_ordinary.to(torch.int64).reshape(Q, 1, 1)
        # C3: min(15, M-1) with M = f+1, and 0 for an inactive row. A caller
        # that disagrees fails the invocation; it does not get a quietly
        # different number of blocks. `k2_merge_kernel` computes exactly this.
        expect = torch.where(f < 0, torch.zeros_like(f), f.clamp(max=K - 1))
        err = torch.where(
            (q != expect).reshape(Q, 1).expand(Q, Hl),
            torch.full((Q, Hl), ICP_STATUS_ROW_META, dtype=torch.int32,
                       device=gid.device),
            torch.zeros((Q, Hl), dtype=torch.int32, device=gid.device),
        )
    # C5: a NaN on a valid record fails the invocation. `isnan` is false for
    # +/-inf, which participate in the ordering normally, and the `gid >= 0`
    # conjunct is C11/C12 -- validity is by id, so an invalid record's score
    # field decides nothing.
    nan_row = (torch.isnan(score) & (gid >= 0)).any(dim=-1)  # [Q, H_local]
    err = err | torch.where(nan_row, torch.full_like(err, ICP_STATUS_NAN),
                            torch.zeros_like(err))
    fail = err != 0
    # Bitwise-OR reduce without a host sync: the bits are known, so two maxima
    # and an OR are exact, and the result is the same word `atomicOr` leaves.
    # `status` accumulates across calls; clearing it is the caller's.
    acc = (err & ICP_STATUS_NAN).amax() | (err & ICP_STATUS_ROW_META).amax()
    status[0] = status[0] | acc.to(status.dtype)

    key = _stable_key(score, gid)
    # stable=True over the source-major pool order makes "first occurrence in
    # this order" exactly the representative the CUDA and Triton arms elect:
    # maximum key, exact ties broken by the smaller flat candidate index.
    order = torch.argsort(key, dim=-1, descending=True, stable=True)
    sgid = torch.gather(gid, -1, order)  # ids in canonical key order

    # --- max-reduce duplicate global ids (refined-icp-v1 C3) ----------------
    # Re-sort the key-ordered ids by (id, key-order position): each id's group
    # is then contiguous and its first element is that id's maximum-score
    # record. int64 throughout -- `id * n_cand` overflows int32 well before
    # INT32_MAX, which is a legal block id.
    seq = torch.arange(n_cand, device=gid.device, dtype=torch.int64)
    grp = torch.argsort(sgid.to(torch.int64) * n_cand + seq, dim=-1)
    by_id = torch.gather(sgid, -1, grp)
    first = torch.ones_like(by_id, dtype=torch.bool)
    first[..., 1:] = by_id[..., 1:] != by_id[..., :-1]
    # Undo the grouping permutation to get back to key order. `sgid != f` is
    # C3's exclusion: the forced block never competes for an ordinary slot on
    # any rank, so it must not compete here either -- and dropping its whole id
    # group is what makes "inject exactly once" hold even if some producer
    # published it anyway.
    is_rep = (
        torch.gather(first, -1, torch.argsort(grp, dim=-1))
        & (sgid >= 0)
        & (sgid.to(torch.int64) != f)
    )

    # The first `q` representatives in key order. The cumsum counts only
    # distinct ids, which is what makes this dedup-then-truncate and not the
    # other way round. `q` is min(15, M-1) when forcing, so the 16th slot stays
    # RESERVED for the injection below.
    keep = is_rep & (torch.cumsum(is_rep.to(torch.int64), dim=-1) <= q)

    # Ascending ids with the -1 tail: park everything else on a sentinel, sort,
    # unpark. The sentinel is 2**31 in int64, not INT32_MAX -- INT32_MAX is a
    # legal block id and would be silently turned into a -1 tail entry.
    #
    # The forced block joins as ONE extra column before the sort, which is the
    # whole reservation: it cannot be outranked, it cannot appear twice (it was
    # excluded from the ordinary pool), and the sort puts it at its ascending
    # position. `n_cand + 1` columns in, the first K out.
    sentinel = 1 << 31
    top = sgid.to(torch.int64)
    top = torch.where(keep, top, torch.full_like(top, sentinel))
    fcol = torch.where(f >= 0, f, torch.full_like(f, sentinel))
    top = torch.cat([top, fcol.expand(Q, Hl, 1)], dim=-1)
    top, _ = torch.sort(top, dim=-1)
    top = top[..., :K]
    top = torch.where(top == sentinel, torch.full_like(top, -1), top)
    # A failed row publishes NO selection rather than a plausible one, in the
    # same rendering the kernel emits -- which is what keeps the arms
    # bit-comparable on the failure path.
    top = torch.where(fail.unsqueeze(-1), torch.full_like(top, -1), top)
    out.copy_(top)
    return out


# --------------------------------------------------------------------------
# the Triton arm
# --------------------------------------------------------------------------

#: Built on first use. Triton is imported inside :func:`_merge_kernel` and
#: nowhere else, so a host with no triton -- or no GPU at all -- can still
#: import this module, read it, and run the torch arm.
_MERGE_KERNEL = None


def _merge_kernel():
    """JIT-decorate the Triton merge, importing triton lazily.

    The kernel body is a nested ``def`` on purpose: ``@triton.jit`` has to run
    against a real ``triton.language``, and the point of this module is that
    importing it does not require one. ``tl`` is published into the module
    globals first, because that is where Triton's code generator resolves the
    body's free names (and where the ``tl.constexpr`` annotations are looked up
    when the ``def`` executes).
    """
    global _MERGE_KERNEL
    if _MERGE_KERNEL is not None:
        return _MERGE_KERNEL
    import triton as _triton  # noqa: PLC0415
    import triton.language as _tl  # noqa: PLC0415

    globals()["tl"] = _tl

    @_triton.jit
    def _icp_merge_c4_kernel(
        cand_ptr,  # int32 [W, Qchunk, H_local, K, 2] -- the C4 receive carrier
        out_ptr,  # int32 [Qchunk, H_local, K]
        forced_ptr,  # int32 [Qchunk], or any int32 pointer when not HAS_FORCED
        nord_ptr,  # int32 [Qchunk], likewise
        status_ptr,  # int32 [>=1], the C5 failure word
        stride_w,
        stride_q,
        stride_h,
        stride_k,
        stride_e,  # carrier axes ([.., 0] score bits, [.., 1] id)
        stride_oq,
        stride_oh,
        stride_ok,  # output axes
        NUM_CAND: tl.constexpr,  # W * K
        TOPK: tl.constexpr,  # K == CAND_K == 16
        BLOCK_N: tl.constexpr,  # next_power_of_2(NUM_CAND)
        HAS_FORCED: tl.constexpr,  # C3 reserved-slot mode
        STATUS_NAN: tl.constexpr,
        STATUS_ROW_META: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_h = tl.program_id(1)
        # No `h = pid_h + head_offset`: the C4 receive carrier is already this
        # rank's slice, so `pid_h` IS the address -- the structural half of the
        # C7 guard that the gathered layout had to assert at runtime.

        # --- C3 row metadata ------------------------------------------------
        # Indexed by the token row only: forcing is a function of the row's
        # query position, not of the head. `f = -1, q = 0` is an inactive row
        # (zero valid ids); `f = -1, q = TOPK` is the plain ordinary merge.
        f = -1
        q = TOPK
        err = 0
        if HAS_FORCED:
            f = tl.load(forced_ptr + pid_q).to(tl.int32)
            q = tl.load(nord_ptr + pid_q).to(tl.int32)
            expect = tl.where(f < 0, 0, tl.minimum(f, TOPK - 1))
            err = tl.where(q != expect, STATUS_ROW_META, 0)

        n = tl.arange(0, BLOCK_N)
        in_range = n < NUM_CAND
        # Candidate n is entry k of SOURCE rank s, source-major (s = n // K).
        # `load_candidates_c4` enumerates `idx = lane + i*32` with the same
        # `s = idx // K`, `k = idx - s*K`, so the positional tie-break below
        # elects the representative the CUDA arm elects.
        s = n // TOPK
        k = n % TOPK

        # int64 offsets: the carrier is W * Qchunk * H_local * 32 int32 words,
        # which can pass 2**31 elements on a large captured chunk.
        row = pid_q.to(tl.int64) * stride_q + pid_h.to(tl.int64) * stride_h
        base = (cand_ptr + row + s.to(tl.int64) * stride_w
                + k.to(tl.int64) * stride_k)
        # BITCAST, not a conversion (C4): word 0 is the fp32 score's raw bits
        # and word 1 the int32 global block id. Neither may ever be converted --
        # an id at or above 2**24 does not survive a float round trip.
        score_bits = tl.load(base, mask=in_range, other=0)
        gid = tl.load(base + stride_e, mask=in_range, other=-1)
        gid = tl.where(in_range, gid, -1)
        valid = gid >= 0
        score = score_bits.to(tl.float32, bitcast=True)

        # --- C5: NaN fails the invocation ------------------------------------
        # `score != score` is the IEEE NaN test; +/-inf are NOT NaN and go on
        # participating in the ordering. Validity is by id (C11/C12), so the
        # score field of an invalid record is never examined. Checked HERE,
        # before the key: sortable(NaN) is 0xffc00000, strictly above +inf's
        # 0xff800000, so a NaN that reaches the ranking below does not
        # misbehave -- it WINS.
        err = err | tl.where(
            tl.sum((valid & (score != score)).to(tl.int32)) > 0, STATUS_NAN, 0)
        fail = err != 0

        # --- C5 canonical key -------------------------------------------------
        # The raw score word, zero-extended to 64 bits.
        raw = score_bits.to(tl.int64) & 0xFFFFFFFF
        # Flush -0.0 to +0.0 BEFORE the mapping -- merge_topk.cuh:105. Without
        # it the two IEEE-equal zeros get different keys and the selector orders
        # two values the contract calls equal; a block score is a max reduction
        # and IEEE leaves the sign of fmax(-0.0, +0.0) implementation-defined.
        raw = tl.where((raw & 0x7FFFFFFF) == 0, 0, raw)
        sortable = tl.where(raw >= 0x80000000, raw ^ 0xFFFFFFFF,
                            raw ^ 0x80000000)
        low = (~gid.to(tl.int64)) & 0xFFFFFFFF
        # Invalid -> the unsigned key 0, i.e. strictly below every real key. (A
        # real key can only be 0 if sortable == 0, which means score bits
        # 0xffffffff, which is a NaN and is forbidden.)
        sortable = tl.where(valid, sortable, 0)
        low = tl.where(valid, low, 0)
        # Subtracting 2**31 from the high word biases the 64-bit key by -2**63,
        # the order-preserving unsigned -> two's-complement reinterpretation, so
        # the int64 comparisons below are the contract's unsigned ones. Invalid
        # entries land on INT64_MIN.
        key = ((sortable - 0x80000000) << 32) | low

        # --- max-reduce duplicate global ids (refined-icp-v1 C3) -------------
        # better[i, j] is "j outranks i": greater key, or an exact key tie
        # broken by the smaller flat candidate index n. rep[i] is "no candidate
        # carrying my id outranks me", which elects exactly one representative
        # per distinct id -- the one holding that id's maximum score, because
        # duplicates of one id share the whole low key word (~gid) and so
        # compare on the score alone. `ordinary` is `valid` minus C3's forced
        # block.
        ordinary = valid & (gid != f)
        better = (key[None, :] > key[:, None]) | (
            (key[None, :] == key[:, None]) & (n[None, :] < n[:, None])
        )
        dup = (ordinary[None, :] & ordinary[:, None]
               & (gid[None, :] == gid[:, None]))
        rep = ordinary & (tl.sum((dup & better).to(tl.int32), axis=1) == 0)

        # --- select the ordinary winners -------------------------------------
        # rank[i] = #{representatives j : j outranks i}. Representatives hold
        # pairwise distinct ids, so their keys differ in the low word and no
        # real key can tie; the positional tie-break inside `better` only keeps
        # the invalid entries -- which all share INT64_MIN -- a strict
        # permutation. `q` replaces TOPK: with q = min(15, M-1) the 16th slot is
        # RESERVED and no ordinary block can take it.
        rank = tl.sum((better & rep[None, :]).to(tl.int32), axis=1)
        win = rep & (rank < q)

        # --- C6 order: ascending by global block id --------------------------
        # pos[i] = #{winners j : gid[j] < gid[i]}. Non-winner columns are masked
        # out rather than parked on a sentinel id -- INT32_MAX is a legal block
        # id. The forced block is one more id in the same ascending order: it
        # shifts every winner above it by one slot and itself takes slot `fpos`.
        smaller = win[None, :] & (
            (gid[None, :] < gid[:, None])
            | ((gid[None, :] == gid[:, None]) & (n[None, :] < n[:, None]))
        )
        pos = tl.sum(smaller.to(tl.int32), axis=1)
        pos = pos + tl.where((f >= 0) & (f < gid), 1, 0)
        fpos = tl.sum((win & (gid < f)).to(tl.int32))

        # Gather winner -> slot in registers and store once. (Scattering into
        # the output after a separate -1 fill would race: the two stores are
        # issued by different lanes with no ordering between them.) `gid + 1`
        # keeps the empty-slot sum at 0, which becomes the -1 tail.
        o = tl.arange(0, TOPK)
        hit = (pos[None, :] == o[:, None]) & win[None, :]
        # int64 so that `+ 1` cannot overflow on gid == INT32_MAX.
        gid64 = gid.to(tl.int64)
        res = tl.sum(tl.where(hit, gid64[None, :] + 1, 0), axis=1) - 1
        # The single injection. Slot `fpos` is provably unclaimed: winners below
        # f have pos < fpos and winners above it have pos > fpos, and f is never
        # a winner because it was excluded from `ordinary`.
        #
        # `zeros + f` rather than `f.to(tl.int64)`: with HAS_FORCED False the
        # `if` above is folded away at compile time and `f` is still the Python
        # literal -1, i.e. a `tl.constexpr`, which has no tensor `.to`. Adding
        # it to a typed zero tensor is valid in both branches and is the same
        # value.
        res = tl.where((f >= 0) & (o == fpos), tl.zeros((TOPK,), tl.int64) + f,
                       res)
        # A failed row publishes NO selection rather than a plausible one, in
        # the same all--1 rendering the CUDA and torch arms emit, and the
        # failure is reported once per program.
        res = tl.where(fail, tl.full((TOPK,), -1, tl.int64), res)
        tl.atomic_or(status_ptr, err, mask=fail)

        tl.store(
            out_ptr
            + pid_q.to(tl.int64) * stride_oq
            + pid_h.to(tl.int64) * stride_oh
            + o.to(tl.int64) * stride_ok,
            res.to(out_ptr.dtype.element_ty),
        )

    _MERGE_KERNEL = _icp_merge_c4_kernel
    return _MERGE_KERNEL


def _triton_merge(cand: torch.Tensor, out: torch.Tensor,
                  forced: torch.Tensor | None,
                  n_ordinary: torch.Tensor | None,
                  status: torch.Tensor) -> torch.Tensor:
    """One Triton program per (query, local head) row. Same arguments as
    :func:`_torch_merge`, same output, same status word."""
    import triton  # noqa: PLC0415

    W, Q, Hl, K, _ = cand.shape
    if Q == 0 or Hl == 0:
        return out
    block_n = triton.next_power_of_2(W * K)
    _merge_kernel()[(Q, Hl)](
        cand,
        out,
        # When HAS_FORCED is False these two are never dereferenced; Triton
        # still needs a correctly typed int32 pointer for the argument, and
        # `out` is the one already at hand. Passing None would make the
        # argument a compile-time specialisation instead.
        forced if forced is not None else out,
        n_ordinary if n_ordinary is not None else out,
        status,
        cand.stride(0),
        cand.stride(1),
        cand.stride(2),
        cand.stride(3),
        cand.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_CAND=W * K,
        TOPK=K,
        BLOCK_N=block_n,
        HAS_FORCED=forced is not None,
        STATUS_NAN=ICP_STATUS_NAN,
        STATUS_ROW_META=ICP_STATUS_ROW_META,
        num_warps=4 if block_n <= 64 else 8,
    )
    return out


# --------------------------------------------------------------------------
# the public reference entry points
# --------------------------------------------------------------------------


def _run(arm: MergeArm, cand: torch.Tensor, world: int, rank: int,
         out: torch.Tensor | None, forced: torch.Tensor | None,
         n_ordinary: torch.Tensor | None,
         status: torch.Tensor | None) -> torch.Tensor:
    cand, out, world, rank, _, _ = _prepare(cand, world, rank, out, forced,
                                            n_ordinary)
    # Naming one of these functions IS naming the arm, so it goes through the
    # same resolution point a `merge_candidates(arm=...)` call does, in the
    # same place: one warning, and `last_merge_arm()` records it either way.
    # Nothing here can resolve to a production arm, so nothing is being routed
    # anywhere. Called for its effects and not asserted on -- `python -O`
    # deletes an assert, and the warning and the recording are the point.
    merge_backend(arm)
    status, owned = _acquire_status(status, cand.device)
    kernel = _triton_merge if arm is MergeArm.TRITON else _torch_merge
    kernel(cand, out, forced, n_ordinary, status)
    _raise_on_status(status, owned)
    return out


def merge_candidates_torch(cand: torch.Tensor, *, world: int, rank: int,
                           out: torch.Tensor | None = None,
                           forced: torch.Tensor | None = None,
                           n_ordinary: torch.Tensor | None = None,
                           status: torch.Tensor | None = None) -> torch.Tensor:
    """The **reference** torch arm of :func:`fmha_sm100.icp.merge_candidates`.

    Same signature, same argument contract (it shares the validation), same
    output and same status bits -- including on the failure path, where all
    three arms publish the same all ``-1`` row. Not a production backend; see
    the module docstring.

    ``world``/``rank`` are validated exactly as the CUDA arm validates them, but
    ``head_offset = rank * H_local`` addresses nothing here: axis 2 of the C4
    receive carrier is already this rank's own heads.

    ``status`` **supplied** is neither cleared nor read, so this is capturable;
    **omitted** it is allocated, zeroed, read back -- which synchronises -- and
    a non-zero word raises :class:`~fmha_sm100.icp.merge.IcpMergeError`. Omitting
    it under an active CUDA-graph capture is refused, never silently skipped.
    """
    return _run(MergeArm.TORCH, cand, world, rank, out, forced, n_ordinary,
                status)


def merge_candidates_triton(cand: torch.Tensor, *, world: int, rank: int,
                            out: torch.Tensor | None = None,
                            forced: torch.Tensor | None = None,
                            n_ordinary: torch.Tensor | None = None,
                            status: torch.Tensor | None = None
                            ) -> torch.Tensor:
    """The **reference** Triton arm of :func:`fmha_sm100.icp.merge_candidates`.

    Identical contract to :func:`merge_candidates_torch`; triton is imported on
    first call, not at import time, so a host without it can still import this
    module and run the torch arm. Not a production backend; see the module
    docstring.
    """
    return _run(MergeArm.TRITON, cand, world, rank, out, forced, n_ordinary,
                status)
