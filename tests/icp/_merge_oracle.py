"""Independent torch oracle for the K2/D3 candidate merge (test-only).

Moved out of the package's former reference tier: production runs the CUDA
merge in ``merge_topk.cuh``; tests compare it with this eager implementation.
"""

from __future__ import annotations

import torch
from fmha_sm100.icp.merge import (
    ICP_NO_FORCED_BLOCK,
    ICP_STATUS_NAN,
    ICP_STATUS_ROW_META,
)


def _stable_key(score: torch.Tensor, block_id: torch.Tensor) -> torch.Tensor:
    """CONTRACT C5's canonical key as int64, biased by ``-2**63``.

    Deliberately **not** :func:`fmha_sm100.icp.merge.canonical_key_reference`,
    which *raises* on a NaN. This is the merge's own key and it has to survive a
    NaN: C5 says a NaN on a valid record fails the invocation, and failing an
    invocation means computing a status word and publishing an all ``-1`` row,
    not throwing out of the middle of a kernel. The two are otherwise the same
    expression, and ``tests/icp/test_merge_key.py`` gates the key itself against the
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
