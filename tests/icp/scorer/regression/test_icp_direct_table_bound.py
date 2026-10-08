# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The direct-table live fragment bound is the SAME function as the one it replaces.

WHAT IS BEING GATED
-------------------
`DIRECT_TABLE_CONTRACT.md` §2.1 rests on one arithmetic claim: replacing the
packed page list with (rectangular table + device bound) is behaviour-preserving
*provided the kernel's bound equals `counts[j]`*, the length of the row prefix
`_pack_page_lists` used to flatten.  Three expressions have to agree for that to
hold, and they live in three different places:

  host packing (deleted)   req_pages  = clamp(ceil((ceil(L/R) - r) / W), min=0)
  kernel trip counts       lb         = ceil((ceil(L/R) - r) / W), floored
  kernel direct-table      local_blocks(L, r) = floor(L/B) + ((L mod B) > r*R)
                           (`icp_local_blocks_exact`, loader header)

The first two are the same expression; the third is the one this change
introduces, and it is the one the contract states in §4.  If it disagrees with
the others by even one block at one residue, the direct-table path either drops
a live fragment (a missing candidate, silently) or reads one past the live tail
(a stale physical page ID from an evicted request -- a *plausible wrong answer*,
which is the failure signature §C says this redesign trades into).

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
It pins the ARITHMETIC.  It does not read the CUDA source and does not prove the
kernel computes what is written here -- that needs a GPU and is Lane A's fixture
set (contract §9).  What it does buy is that the identity the contract asserts
"algebraically" is checked over the whole boundary region rather than at the five
points the document tabulates, including the TP4 interior seams that the TP2
residue set does not reach (amendment D6).

No GPU, no torch, no import of the kernel package: it is arithmetic.
"""

import math

import pytest


# ---------------------------------------------------------------------------
# the three expressions, transcribed
# ---------------------------------------------------------------------------
def host_req_pages(L, r, W):
    """The host formula the contract deletes (`indexer_msa.py:1067-1068`).

    `R` is the rank's row count, so `ceil(L / R)` is the global count of R-row
    fragments and the rank takes every W-th one starting at `r`.
    """
    R = 128 // W
    r_chunks = -(-L // R)
    return max(-(-(r_chunks - r) // W), 0)


def kernel_trip_count_blocks(L, r, W):
    """The block-cyclic form both kernel trip counts use, WITHOUT their floors.

    `compute_effective_end` (loader) and `get_full_trip_count` (mainloop) floor
    this at 1 and `icp_local_blocks` floors it at 0; the floors are a separate,
    deliberate asymmetry reconciled by `is_empty_work`, so they are excluded
    here -- this test is about the unfloored function.
    """
    R = 128 // W
    gb = -(-L // R)
    return math.ceil((gb - r) / W)


def direct_local_blocks(L, r, W):
    """`icp_local_blocks_exact`, the contract §4 form.  STRICTLY greater-than."""
    R = 128 // W
    B = R * W
    return L // B + (1 if (L % B) > r * R else 0)


def writer_owned_blocks(L, r, W):
    """Ground truth, straight from the WRITER's ownership gate.

    `fused_indexer_nvfp4_kv_write.cu:665-669` stores position `p` on rank `r`
    iff `(p % 128) / R == r`.  A logical block `b` is materialised on rank `r`
    iff the rank owns at least one position `p < L` inside it, and the bound the
    loader needs is the count of such blocks -- which, because ownership is a
    CONTIGUOUS row range inside the page rather than block-cyclic across pages,
    is a prefix count.  Derived by enumeration, deliberately: it shares no
    algebra with the three expressions above, so an error common to all of them
    would still be caught.
    """
    R = 128 // W
    B = R * W
    owned = set()
    for p in range(L):
        if (p % B) // R == r:
            owned.add(p // B)
    if not owned:
        return 0
    assert owned == set(range(max(owned) + 1)), (
        "ownership is not a block prefix -- the whole redesign rests on it being one"
    )
    return max(owned) + 1


WORLD_SIZES = (2, 4, 8)


def _lengths_for(W):
    """Every residue near a seam, plus a long tail, plus 0."""
    R = 128 // W
    seams = set()
    for b in range(3):  # three compound pages
        for k in range(W + 1):  # every in-page row boundary
            for d in (-1, 0, 1):
                seams.add(b * 128 + k * R + d)
    seams |= {0, 1, 4, 65, 127, 128, 129, 192, 1024, 150000}
    return sorted(x for x in seams if x >= 0)


@pytest.mark.parametrize("W", WORLD_SIZES)
def test_direct_bound_equals_the_host_formula_it_replaces(W):
    for L in _lengths_for(W):
        for r in range(W):
            assert direct_local_blocks(L, r, W) == host_req_pages(L, r, W), (
                f"W={W} r={r} L={L}: direct={direct_local_blocks(L, r, W)} "
                f"host={host_req_pages(L, r, W)}"
            )


@pytest.mark.parametrize("W", WORLD_SIZES)
def test_direct_bound_equals_the_kernel_trip_count_form(W):
    """The clamp and the trip count must not be able to disagree.

    `is_empty_work` takes the kernel-wide empty decision from the trip-count
    form while the loader clamps page lookups with the direct form.  If they
    could differ, a work item could survive the empty check and then clamp to
    page -1, or be skipped while having live pages.
    """
    for L in _lengths_for(W):
        for r in range(W):
            assert direct_local_blocks(L, r, W) == kernel_trip_count_blocks(L, r, W)


@pytest.mark.parametrize("W", (2, 4))
def test_direct_bound_equals_the_writer_ownership_gate(W):
    """Against enumeration of the writer's own predicate, not against algebra."""
    for L in range(0, 3 * 128 + 5):
        for r in range(W):
            assert direct_local_blocks(L, r, W) == writer_owned_blocks(L, r, W), (
                f"W={W} r={r} L={L}"
            )


def test_strictly_greater_than_is_load_bearing():
    """`>=` instead of `>` is the whole bug, and it is invisible at most lengths.

    At `L == B*q + r*R` rank `r`'s first row of block `q` is the token at global
    position `L` -- one past the end -- so `>=` would claim a block whose index
    rows the writer has not written.  This asserts the two spellings actually
    differ, so the strictness in the contract is not decorative.
    """
    W, r = 2, 1
    R, B = 128 // W, 128
    L = B * 3 + r * R  # exactly on the seam
    ge_form = L // B + (1 if (L % B) >= r * R else 0)
    assert direct_local_blocks(L, r, W) == 3
    assert ge_form == 4
    # and the contract's own table (§4), verbatim
    assert [direct_local_blocks(x, 0, 2) for x in (4, 64, 65, 128, 192)] == [
        1,
        1,
        1,
        1,
        2,
    ]
    assert [direct_local_blocks(x, 1, 2) for x in (4, 64, 65, 128, 192)] == [
        0,
        0,
        1,
        1,
        1,
    ]


def test_empty_shard_is_reachable_and_is_exactly_zero():
    """The capture-time state (`seq_len == query_len == 4`) has zero rank-1 rows.

    A zero bound is what `is_empty_work` keys on, and what makes the loader's
    `min(page, bound - 1)` clamp unreachable rather than -1.  It must be zero on
    the nose, not "small".
    """
    for W in WORLD_SIZES:
        assert direct_local_blocks(4, 0, W) == 1
        for r in range(1, W):
            assert direct_local_blocks(4, r, W) == 0
