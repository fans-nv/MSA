"""The C5 canonical key, and the merge on the refined-icp-v1 C4 carrier.

C5 is ``key = (sortable(score) << 32) | ~uint32(id)``, ``id < 0 -> 0``, with
``-0.0`` flushed to ``+0.0`` before the bits are taken: score descending, then
global block id ascending.

**Why the flush is not pedantry.** A block score is a *max reduction*, and
IEEE-754 leaves the sign of ``fmax(-0.0, +0.0)`` implementation-defined. ICP
reduces over a different grouping than the unsharded path *by construction*, so
without the flush two numerically identical runs can produce opposite zero
signs for the same block, get different keys (``0x7fffffff`` vs ``0x80000000``)
and disagree on the Top-16. Upstream, the same omission survived for weeks in
two of five implementations because the *test's own oracle* shared it.

So the key is never gated by one transcription against another:

1. **Literals.** A handful of keys computed once, by hand, and written out as
   integers. A literal cannot drift in lockstep with anything.
2. **Properties.** Ordering, ties, signed zeros and invalid ids checked as
   *behaviour*, independently of how the key is built.
3. **Differential, on GPU.** ``canonical_key_reference`` (torch) against
   ``canonical_key_cuda``, which is a probe over ``icp::canonical_key`` in
   ``csrc/merge_topk.cuh`` -- the key the kernels actually use, not a copy of
   it.

WHAT refined-icp-v1 CHANGED HERE
--------------------------------

The key itself did not move. Everything around it did, and three assertions in
this file used to state things that are now false:

* **Duplicate global ids are the normal case.** Every rank holds ``R = 128/W``
  rows of *every* logical block and publishes a partial maximum per block, so
  the same id arrives from several sources and the merge max-reduces duplicates
  by id **before** truncating to 16. The pre-refinement merge could truncate
  first because C1 gave each block exactly one owner; that premise is gone.
* **``+inf`` is not a forcing mechanism.** C3 reserves an output slot for the
  forced block ``f = p//128``, excludes ``f`` from ordinary ranking on every
  rank, and injects it exactly once. A *legitimate* ordinary ``+inf`` with a
  smaller id ties with a ``+inf``-forced block under C5 and wins the
  ascending-id tie-break, which is why the reservation exists.
* **NaN fails the invocation.** ``sortable_bits(NaN)`` is ``0xffc00000``,
  strictly above ``+inf``, so a NaN candidate used to win silently. It now sets
  ``ICP_STATUS_NAN``, the row is published as all ``-1`` and the wrapper-owned
  status form raises :class:`IcpMergeError`. Validity is by **id**, never by
  score, so a NaN on an invalid record is not a failure.

The carrier is int32 ``[W, Qchunk, H_local, 16, 2]`` and the host ABI is
``refined-icp-v1.k2.2``; the pre-C4 fp32 gathered tensor is *refused* rather
than reinterpreted.

MUTATION CHECK. Delete the ``-0.0`` flush at ``csrc/merge_topk.cuh:105`` (the
``((raw & 0x7fffffffu) == 0u) ? 0u : raw`` in ``sortable_bits``) and
``test_signed_zeros_agree_on_gpu`` must fail. Removing the matching
``torch.where`` in ``canonical_key_reference`` must fail
``test_signed_zeros_are_one_key`` on CPU. Delete pass 0 of
``warp_merge_topk16`` and ``test_duplicate_ids_are_max_reduced_before_truncation``
must fail. Replace the reserved slot with ``+inf`` forcing and
``test_a_forced_block_outranks_sixteen_legitimate_infinities`` must fail. If any
still passes, the gate is not gating.
"""

from __future__ import annotations

import math

import pytest
import torch
from fmha_sm100.icp import (
    ICP_STATUS_NAN,
    ICP_STATUS_OK,
    ICP_STATUS_ROW_META,
    KEY_BIAS,
    IcpMergeError,
    canonical_key_cuda,
    canonical_key_reference,
    forced_rows,
    merge_candidates,
)

INF = float("inf")
NAN = float("nan")
CAND_K = 16

# --- 1. literals -----------------------------------------------------------
# (score, id) -> the C5 key, biased by -2**63 so a signed compare is the
# contract's unsigned compare. Derived once by hand from the contract text;
# e.g. score 1.0 has bits 0x3f800000, which is positive so sortable is
# 0x3f800000 ^ 0x80000000 = 0xbf800000, and ~5 is 0xfffffffa, giving the
# unsigned key 0xbf800000_fffffffa and the biased value below.
KEY_LITERALS = [
    (1.0, 5, 4575657225703391226),  # 0xbf800000fffffffa
    (-1.0, 0, -4575657221408423937),  # 0x407fffffffffffff
    (2.5, 0, 4620693221977096191),  # 0xc0200000ffffffff
    (0.0, 7, 4294967288),  # 0x80000000fffffff8
    (-0.0, 7, 4294967288),  # the same key: the flush
    (INF, 3, 9187343244130779132),  # 0xff800000fffffffc
    (-INF, -1, -(1 << 63)),  # invalid -> key 0
    (-0.0, -1, -(1 << 63)),  # invalid wins over the score
]


def _ref(scores, ids, device="cpu"):
    return canonical_key_reference(
        torch.tensor(scores, dtype=torch.float32, device=device),
        torch.tensor(ids, dtype=torch.int32, device=device),
    )


def test_key_matches_hand_computed_literals():
    scores = [s for s, _, _ in KEY_LITERALS]
    ids = [i for _, i, _ in KEY_LITERALS]
    want = [k for _, _, k in KEY_LITERALS]
    got = _ref(scores, ids)
    assert got.tolist() == want


def test_bias_round_trips_to_the_contract_value():
    # The contract's value is unsigned; KEY_BIAS is how it is carried in an
    # int64 tensor without losing the ordering.
    (biased,) = _ref([1.0], [5]).tolist()
    assert biased + KEY_BIAS == 0xBF800000FFFFFFFA


# --- 2. properties ---------------------------------------------------------


def test_order_is_score_descending_then_id_ascending():
    # Independent of the key's construction: sort by the key and compare with
    # the contract's stated order.
    scores = [0.5, 2.0, 2.0, -1.0, 2.0, 0.5]
    ids = [9, 4, 1, 0, 7, 2]
    keys = _ref(scores, ids).tolist()
    by_key = sorted(range(len(ids)), key=lambda j: -keys[j])
    by_contract = sorted(range(len(ids)), key=lambda j: (-scores[j], ids[j]))
    assert by_key == by_contract
    # and spell the expected answer out, so a bug in both sorts cannot agree
    assert [ids[j] for j in by_key] == [1, 4, 7, 2, 9, 0]


def test_exact_ties_break_by_id_ascending():
    ids = [40, 7, 7000, 1]
    keys = _ref([1.25] * 4, ids).tolist()
    assert [ids[j] for j in sorted(range(4), key=lambda j: -keys[j])] == [
        1,
        7,
        40,
        7000,
    ]
    # ties really are ties on the score half: only the low 32 bits differ
    highs = {(k + KEY_BIAS) >> 32 for k in keys}
    assert len(highs) == 1


def test_signed_zeros_are_one_key():
    # MUTATION: drop the flush in `canonical_key_reference` and this fails.
    pos = _ref([0.0], [11])
    neg = _ref([-0.0], [11])
    assert torch.equal(pos, neg)
    assert math.copysign(1.0, -0.0) == -1.0  # the input really was -0.0


def test_signed_zero_does_not_reorder_against_a_neighbour():
    # The failure the flush prevents is an *ordering* one, so check the order,
    # not only the key: -0.0 must sit between the positive and negative values
    # in exactly the same place +0.0 does.
    for zero in (0.0, -0.0):
        keys = _ref([1.0, zero, -1.0], [1, 2, 3]).tolist()
        assert keys[0] > keys[1] > keys[2]


def test_infinities_bound_every_finite_score():
    # USED TO SAY: "C3 forces a block by giving it +inf, so +inf must beat every
    # finite score regardless of id". The ORDERING claim is still true and is
    # still asserted; the REASON is not. refined-icp-v1 C3 removed +inf forcing
    # outright, because a *legitimate* ordinary +inf with a smaller id ties with
    # the forced block under this very ordering and wins the ascending-id
    # tie-break. The forced block now rides a reserved output slot --
    # `test_a_forced_block_outranks_sixteen_legitimate_infinities` is where that
    # is gated. What +inf still is: the top of the score order, and a score a
    # producer may legitimately emit.
    keys = _ref([INF, 3.4e38, -3.4e38, -INF], [0, 0, 0, 0]).tolist()
    assert keys[0] > keys[1] > keys[2] > keys[3]
    # and, in the same breath, the fact that makes +inf useless as a forcing
    # mechanism: at equal score the SMALLER id wins.
    tied = _ref([INF, INF], [7, 4096]).tolist()
    assert tied[0] > tied[1]


def test_invalid_id_is_the_minimum_key_whatever_the_score():
    keys = _ref([INF, 1.0, -INF], [-1, -1, -1]).tolist()
    assert keys == [-(1 << 63)] * 3
    # ... and strictly below any valid candidate, so padding can never win
    valid = _ref([-INF], [0]).item()
    assert valid > -(1 << 63)


def test_nan_is_rejected_not_ordered():
    with pytest.raises(ValueError, match="NaN"):
        _ref([NAN], [0])


def test_dtypes_are_enforced():
    with pytest.raises(TypeError):
        canonical_key_reference(
            torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.int32)
        )
    with pytest.raises(TypeError):
        canonical_key_reference(
            torch.zeros(4, dtype=torch.float32), torch.zeros(4, dtype=torch.int64)
        )


# --- 3. C3 row metadata, on the host ---------------------------------------


def test_forced_rows_derives_c3_metadata_from_each_rows_own_position():
    # A batch-wide sequence length is never a substitute: prefill, chunked
    # prefill, decode and Q>1 verification rows sit in ONE invocation with
    # different p, which is why these are planes and not scalars.
    positions = torch.tensor(
        [0, 127, 128, 129, 128 * 15, 128 * 40 + 7], dtype=torch.int32
    )
    forced, n_ordinary = forced_rows(positions)
    assert forced.dtype == torch.int32 and n_ordinary.dtype == torch.int32
    assert forced.tolist() == [0, 0, 1, 1, 15, 40]
    # n_ordinary = min(15, f): the 16th slot is RESERVED for the forced block,
    # so the ordinary count saturates one below K.
    assert n_ordinary.tolist() == [0, 0, 1, 1, 15, 15]


def test_forced_rows_marks_an_inactive_row_as_selecting_nothing():
    positions = torch.tensor([128 * 3, 128 * 3], dtype=torch.int32)
    active = torch.tensor([True, False])
    forced, n_ordinary = forced_rows(positions, active)
    assert forced.tolist() == [3, -1]
    assert n_ordinary.tolist() == [3, 0]


# --- 4. the wrapper's argument contract, without a device ------------------


def test_merge_refuses_the_pre_refinement_fp32_gathered_tensor():
    """C4: refused, never reinterpreted.

    The fp32 ``[C, T, H_group, 16, 2]`` tensor has the same rank and the same
    trailing pair as the C4 carrier, so a reinterpretation would *run* -- and
    read a peer's heads under this rank's name, because axis 2 is H_group there
    and H_local here. The dtype is the only thing that distinguishes them, so
    the dtype is a refusal.
    """
    gathered = torch.zeros((2, 4, 4, CAND_K, 2), dtype=torch.float32)
    with pytest.raises(TypeError, match="int32"):
        merge_candidates(gathered, world=2, rank=0)


def test_merge_refuses_a_carrier_that_is_not_five_dimensional():
    with pytest.raises(ValueError, match=r"\[W, Qchunk, H_local, 16, 2\]"):
        merge_candidates(
            torch.zeros((4, CAND_K, 2), dtype=torch.int32), world=1, rank=0
        )


# --- 5. differential against the shipped CUDA key --------------------------


@pytest.mark.gpu
def test_random_scores_agree_on_gpu():
    g = torch.Generator(device="cuda").manual_seed(0)
    n = 1 << 16
    scores = torch.randn(n, device="cuda", generator=g, dtype=torch.float32)
    scores[::97] = INF  # a legitimate saturated score, not a forcing
    scores[1::97] = -INF  # padding scores
    ids = torch.randint(
        -1, 1 << 20, (n,), device="cuda", generator=g, dtype=torch.int32
    )
    assert torch.equal(
        canonical_key_cuda(scores, ids), canonical_key_reference(scores, ids)
    )


@pytest.mark.gpu
def test_exact_ties_agree_on_gpu():
    # Every score identical: the whole key is then the id half, which is the
    # part a "morally equivalent" reimplementation is most likely to get wrong.
    n = 4096
    scores = torch.full((n,), 0.125, device="cuda", dtype=torch.float32)
    ids = torch.arange(n, device="cuda", dtype=torch.int32)
    assert torch.equal(
        canonical_key_cuda(scores, ids), canonical_key_reference(scores, ids)
    )


@pytest.mark.gpu
def test_signed_zeros_agree_on_gpu():
    # MUTATION: remove the flush at csrc/merge_topk.cuh:105 and this must fail.
    scores = torch.tensor([0.0, -0.0, 0.0, -0.0], device="cuda", dtype=torch.float32)
    ids = torch.tensor([3, 3, 8, 8], device="cuda", dtype=torch.int32)
    cuda = canonical_key_cuda(scores, ids)
    assert torch.equal(cuda, canonical_key_reference(scores, ids))
    assert cuda[0].item() == cuda[1].item()
    assert cuda[2].item() == cuda[3].item()


# --------------------------------------------------------------------------
# 6. the merge on the C4 carrier. GPU.
# --------------------------------------------------------------------------
#
# Every reference below is plain Python over the same record list the carrier
# was built from. They are deliberately several, because the point of most of
# these tests is that two *orders of operations* give different answers and the
# kernel takes the contract's one.


def _pool(records, *, forced=None):
    """id -> max score over every source. C3's max-by-id reduction."""
    best: dict[int, float] = {}
    for score, gid in records:
        if gid < 0 or (forced is not None and forced >= 0 and gid == forced):
            continue  # C11/C12: validity is by id; C3 excludes the forced id
        if gid not in best or score > best[gid]:
            best[gid] = score
    return best


def _row(ids):
    """An id list rendered as C6's output row: ascending, -1-padded."""
    ids = sorted(ids)
    assert len(ids) <= CAND_K
    return ids + [-1] * (CAND_K - len(ids))


def dedup_then_truncate(records, *, forced=None, n_ordinary=CAND_K):
    """THE CONTRACT (C3/C6): max-reduce by id, then take the winners."""
    ranked = sorted(
        _pool(records, forced=forced).items(), key=lambda kv: (-kv[1], kv[0])
    )
    winners = [gid for gid, _ in ranked[:n_ordinary]]
    if forced is not None and forced >= 0:
        winners.append(forced)
    return _row(winners)


def truncate_then_dedup(records, *, n_ordinary=CAND_K):
    """THE PRE-REFINEMENT ORDER, and the M0 oracle's. Kept as a control: it is
    what the merge does if pass 0 of ``warp_merge_topk16`` is deleted."""
    ranked = sorted((r for r in records if r[1] >= 0), key=lambda r: (-r[0], r[1]))[
        :n_ordinary
    ]
    return _row(set(gid for _, gid in ranked))


def min_reduce_then_truncate(records, *, n_ordinary=CAND_K):
    """A control for the DIRECTION of the reduction: identical to the contract
    except that a duplicate keeps its worst score instead of its best."""
    worst: dict[int, float] = {}
    for score, gid in records:
        if gid < 0:
            continue
        if gid not in worst or score < worst[gid]:
            worst[gid] = score
    ranked = sorted(worst.items(), key=lambda kv: (-kv[1], kv[0]))
    return _row([gid for gid, _ in ranked[:n_ordinary]])


def plus_inf_forcing(records, *, forced, n_ordinary=CAND_K):
    """The forcing mechanism refined-icp-v1 C3 REMOVED: give the forced block
    ``+inf`` and let it compete for one of the 16 ordinary slots."""
    pool = _pool(records)
    pool[forced] = INF
    ranked = sorted(pool.items(), key=lambda kv: (-kv[1], kv[0]))
    return _row([gid for gid, _ in ranked[:n_ordinary]])


def _from_sources(sources):
    """``fill`` for the ``c4_carrier`` fixture, plus the flat record list."""

    def fill(s, q, h):
        return sources[s]

    return fill, [r for src in sources for r in src]


def _plane(value, *, qchunk, device):
    return torch.full((qchunk,), int(value), dtype=torch.int32, device=device)


@pytest.mark.gpu
def test_the_status_bits_are_the_kernels_own(k2_module):
    # The Python constants are a mirror of `merge_topk.cuh`; assert them against
    # the built object rather than against the comment that says so.
    assert int(k2_module.k2_status_nan) == ICP_STATUS_NAN
    assert int(k2_module.k2_status_row_meta) == ICP_STATUS_ROW_META
    assert ICP_STATUS_OK == 0
    assert k2_module.k2_abi_version == "refined-icp-v1.k2.2"
    assert k2_module.k2_carrier == "refined-icp-v1.C4"


@pytest.mark.gpu
def test_merge_output_is_ascending_with_a_minus_one_tail(cuda_device, c4_carrier):
    """C6, over a random multi-source carrier.

    USED TO ASSERT, in `make_candidates`' spirit, that the ids reaching the
    merge were distinct by construction. Under fragment placement they are not:
    the ids below are drawn from ONE global domain on every source, so the same
    id arrives several times, and "no duplicates in the output" is now a
    property the kernel has to *establish* rather than inherit.
    """
    W, T, Hl = 4, 8, 2
    rng = torch.Generator().manual_seed(7)

    sources = {}
    for s in range(W):
        for t in range(T):
            for h in range(Hl):
                # Rows of 2 keep the pool below 16 so the -1 tail is exercised;
                # the id domain is 256 wide so that the full rows collide across
                # sources many times over rather than by luck.
                n = 2 if t % 3 == 0 else CAND_K
                ids = torch.randint(0, 1 << 8, (n,), generator=rng).tolist()
                scores = torch.randn(n, generator=rng).tolist()
                sources[(s, t, h)] = list(zip(scores, ids))

    cand = c4_carrier(
        lambda s, t, h: sources[(s, t, h)],
        world=W,
        qchunk=T,
        heads_local=Hl,
        device=cuda_device,
    )
    out = merge_candidates(cand, world=W, rank=1)
    assert out.shape == (T, Hl, CAND_K) and out.dtype == torch.int32

    host = out.cpu()
    duplicates_seen = 0
    short_rows = full_rows = 0
    for t in range(T):
        for h in range(Hl):
            records = [r for s in range(W) for r in sources[(s, t, h)]]
            ids = [gid for _, gid in records]
            duplicates_seen += len(ids) - len(set(ids))
            row = host[t, h].tolist()
            valid = [x for x in row if x >= 0]
            assert row[: len(valid)] == valid, "the -1s must be a tail"
            assert valid == sorted(valid), "ids must be ascending"
            assert len(set(valid)) == len(valid), "no duplicates"
            assert row == dedup_then_truncate(records)
            short_rows += len(valid) < CAND_K
            full_rows += len(valid) == CAND_K
    # ANTI-VACUITY: if the draw produced no cross-source duplicate at all, this
    # test degenerates into the pre-refinement case and proves nothing about
    # pass 0.
    assert duplicates_seen > 0, (
        "no duplicate global id was drawn, so this case cannot distinguish the "
        "deduplicating merge from the pre-refinement one"
    )
    # ... and both row depths have to occur, or "the -1s are a tail" and "the
    # row is full" are each being checked against one case only.
    assert short_rows and full_rows, (short_rows, full_rows)


@pytest.mark.gpu
def test_merge_selects_by_the_canonical_key(cuda_device, c4_carrier):
    # Ties everywhere, so the selection is decided entirely by the id half of
    # the key -- a merge that broke ties by anything else (SMEM staging slot,
    # source rank, arrival order) would differ here and nowhere else.
    W = 2
    sources = [
        [(1.0, gid) for gid in range(s * CAND_K, (s + 1) * CAND_K)] for s in range(W)
    ]
    fill, records = _from_sources(sources)
    cand = c4_carrier(fill, world=W, device=cuda_device)
    out = merge_candidates(cand, world=W, rank=0)
    # 32 distinct candidates at one score: the 16 smallest ids win.
    assert out[0, 0].tolist() == list(range(CAND_K))
    assert out[0, 0].tolist() == dedup_then_truncate(records)


@pytest.mark.gpu
def test_duplicate_ids_are_max_reduced_before_truncation(cuda_device, c4_carrier):
    """C3's reduction, in both of its halves.

    Block 500 arrives from two sources: a losing score on one and a winning one
    on the other. Under the contract the surviving representative carries the
    **cross-source maximum**, so 500 wins a slot; under a min-reduction (or a
    "first source wins" rule) it does not, and a different id takes that slot.

    MUTATION: delete pass 0 of ``warp_merge_topk16`` and this fails -- both
    copies of 500 then reach pass 2, compute the same output slot, and one
    overwrites the other, leaving a -1 inside the valid prefix.
    """
    W = 2
    sources = [
        [(-5.0, 500)] + [(0.5 - 0.01 * i, i) for i in range(15)],
        [(+5.0, 500)] + [(0.5 - 0.01 * (15 + i), 15 + i) for i in range(15)],
    ]
    fill, records = _from_sources(sources)

    want = dedup_then_truncate(records)
    if_min = min_reduce_then_truncate(records)
    # PRECONDITION: the case must discriminate, or "the kernel matched `want`"
    # says nothing about the direction of the reduction.
    assert want != if_min, "this case cannot tell max-by-id from min-by-id"
    assert 500 in want and 500 not in if_min

    cand = c4_carrier(fill, world=W, device=cuda_device)
    got = merge_candidates(cand, world=W, rank=0)[0, 0].tolist()
    assert got == want
    assert got.count(500) == 1, "a duplicate must be reduced, not carried twice"


@pytest.mark.gpu
def test_deduplication_changes_which_ids_win(cuda_device, c4_carrier):
    """Duplicates straddling the 16-way cutoff.

    The previous test shows the reduction picks the right score. This one shows
    the reduction must happen **before** the truncation: eight ids are duplicated
    across the two sources, so truncating to 16 records first spends 16 slots on
    8 distinct blocks and the row comes back half empty, while deduplicating
    first fills all 16 with a strictly different id set.
    """
    W = 2
    sources = [
        [(1.0, gid) for gid in range(8)] + [(0.1, 200 + i) for i in range(8)],
        [(0.9, gid) for gid in range(8)] + [(0.5, 100 + i) for i in range(8)],
    ]
    fill, records = _from_sources(sources)

    want = dedup_then_truncate(records)
    if_truncated_first = truncate_then_dedup(records)
    # PRECONDITION: the two orders must actually disagree here.
    assert want != if_truncated_first
    assert want == _row(list(range(8)) + [100 + i for i in range(8)])
    assert if_truncated_first.count(-1) == 8, (
        "the control must leave the row half empty, or it is not the failure "
        "mode this test is about"
    )

    cand = c4_carrier(fill, world=W, device=cuda_device)
    got = merge_candidates(cand, world=W, rank=0)[0, 0].tolist()
    assert got == want
    assert -1 not in got, "C6 forbids a -1 inside the valid prefix"


# --- the C3 reserved forced slot ------------------------------------------


@pytest.mark.gpu
def test_the_forced_block_is_excluded_from_ordinary_ranking(cuda_device, c4_carrier):
    """C3: ``f`` takes no ordinary slot, on any rank, and is injected once.

    ``f = 3`` carries the best score in the carrier -- a producer that published
    it anyway, which C3 tolerates precisely because the merge excludes it. If it
    were allowed to compete it would win an ordinary slot *and* be injected, so
    the row would carry it twice (or come back short); excluded, it costs
    nothing and block 2 keeps the third ordinary slot.
    """
    W, f = 2, 3
    n_ordinary = min(CAND_K - 1, f)  # C3: min(15, f)
    sources = [
        [(9.0, f), (0.9, 0), (0.8, 1)],
        [(0.7, 2), (0.6, 4), (0.5, 5)],
    ]
    fill, records = _from_sources(sources)

    want = dedup_then_truncate(records, forced=f, n_ordinary=n_ordinary)
    if_it_competed = dedup_then_truncate(records, n_ordinary=n_ordinary)
    # PRECONDITION: with f competing, it would displace an ordinary winner --
    # so a kernel that failed to exclude it gives a different answer here.
    assert want != if_it_competed
    assert want == _row([0, 1, 2, f]) and f in if_it_competed
    assert 2 not in if_it_competed

    cand = c4_carrier(fill, world=W, device=cuda_device)
    out = merge_candidates(
        cand,
        world=W,
        rank=0,
        forced=_plane(f, qchunk=1, device=cuda_device),
        n_ordinary=_plane(n_ordinary, qchunk=1, device=cuda_device),
    )
    got = out[0, 0].tolist()
    assert got == want
    assert got.count(f) == 1, "exactly one injection, even when a producer leaks f"


@pytest.mark.gpu
def test_a_forced_block_outranks_sixteen_legitimate_infinities(cuda_device, c4_carrier):
    """``+inf`` is NOT a forcing mechanism -- the reserved slot is.

    Sixteen ordinary candidates carry a legitimate ``+inf`` and ids **smaller**
    than ``f``. Under the pre-refinement mechanism (score ``f`` ``+inf`` and let
    it compete) every one of them ties with ``f`` on the score half of the C5
    key and wins the ascending-id tie-break, so ``f`` is displaced and the
    selection is wrong -- silently, because the row is full and well formed.
    With the reserved slot, 15 of the infinities win ordinary slots and ``f``
    takes the 16th regardless of anything anyone scores.
    """
    W, f = 2, 100
    n_ordinary = min(CAND_K - 1, f)  # == 15
    sources = [[(INF, gid) for gid in range(8)], [(INF, gid) for gid in range(8, 16)]]
    fill, records = _from_sources(sources)

    want = dedup_then_truncate(records, forced=f, n_ordinary=n_ordinary)
    if_plus_inf_forced = plus_inf_forcing(records, forced=f)
    # PRECONDITION: the old mechanism must actually lose the forced block here,
    # or this test cannot tell the two mechanisms apart.
    assert f not in if_plus_inf_forced, (
        "the +inf control did not displace the forced block, so this case does "
        "not discriminate"
    )
    assert want == _row(list(range(15)) + [f])

    cand = c4_carrier(fill, world=W, device=cuda_device)
    out = merge_candidates(
        cand,
        world=W,
        rank=0,
        forced=_plane(f, qchunk=1, device=cuda_device),
        n_ordinary=_plane(n_ordinary, qchunk=1, device=cuda_device),
    )
    got = out[0, 0].tolist()
    assert got == want
    assert got.count(f) == 1
    # The forced block sits in the slot its ASCENDING position demands, not at
    # the end: C6 says the row is ascending and says nothing about provenance.
    assert got == sorted(got)


@pytest.mark.gpu
def test_an_inactive_row_selects_nothing(cuda_device, c4_carrier):
    # C6: an inactive row is (forced=-1, n_ordinary=0) and yields zero valid
    # ids, even though its carrier slots are full of perfectly good candidates.
    W = 2
    sources = [
        [(1.0, gid) for gid in range(CAND_K)],
        [(2.0, gid) for gid in range(CAND_K, 2 * CAND_K)],
    ]
    fill, _ = _from_sources(sources)
    cand = c4_carrier(fill, world=W, qchunk=2, device=cuda_device)

    forced = torch.tensor([5, -1], dtype=torch.int32, device=cuda_device)
    n_ordinary = torch.tensor([5, 0], dtype=torch.int32, device=cuda_device)
    out = merge_candidates(cand, world=W, rank=0, forced=forced, n_ordinary=n_ordinary)
    assert out[1, 0].tolist() == [-1] * CAND_K
    # PRECONDITION: the active row on the same launch must NOT be empty, or
    # "the inactive row is empty" could be true of a kernel that wrote nothing.
    assert out[0, 0].tolist() != [-1] * CAND_K


# --- the C3 / C5 failure paths --------------------------------------------


def _ordinary_sources(W=2):
    return [
        [(1.0 - 0.01 * (s * CAND_K + k), s * CAND_K + k) for k in range(CAND_K)]
        for s in range(W)
    ]


@pytest.mark.gpu
def test_n_ordinary_disagreeing_with_the_c3_formula_fails_the_invocation(
    cuda_device, c4_carrier
):
    """C3's cross-check, which only means something when it can fire.

    ``n_ordinary`` is redundant -- the kernel could derive ``min(15, f)`` from
    ``f`` -- and it is carried anyway so a caller that derived its count from a
    batch-wide sequence length instead of the row's own ``p`` is caught. Here
    the caller claims one more ordinary winner than ``f`` allows.
    """
    W, f = 2, 5
    fill, _ = _from_sources(_ordinary_sources(W))
    cand = c4_carrier(fill, world=W, device=cuda_device)
    forced = _plane(f, qchunk=1, device=cuda_device)

    # PRECONDITION: the same invocation with C3's own count must SUCCEED, or the
    # failure below could be caused by anything in this carrier.
    ok = merge_candidates(
        cand,
        world=W,
        rank=0,
        forced=forced,
        n_ordinary=_plane(min(15, f), qchunk=1, device=cuda_device),
    )
    assert ok[0, 0].tolist() != [-1] * CAND_K

    with pytest.raises(IcpMergeError, match="n_ordinary") as excinfo:
        merge_candidates(
            cand,
            world=W,
            rank=0,
            forced=forced,
            n_ordinary=_plane(min(15, f) + 1, qchunk=1, device=cuda_device),
        )
    assert excinfo.value.status == ICP_STATUS_ROW_META

    # ... and the failing row is published as all -1, not as a plausible
    # selection: a caller-owned status word makes the same call capturable, and
    # the output is defined on the failure path too.
    status = torch.zeros(1, dtype=torch.int32, device=cuda_device)
    out = merge_candidates(
        cand,
        world=W,
        rank=0,
        forced=forced,
        n_ordinary=_plane(min(15, f) + 1, qchunk=1, device=cuda_device),
        status=status,
    )
    assert int(status.item()) & ICP_STATUS_ROW_META
    assert out[0, 0].tolist() == [-1] * CAND_K


@pytest.mark.gpu
def test_a_nan_on_a_valid_record_fails_the_invocation(cuda_device, c4_carrier):
    """C5. ``sortable_bits(NaN)`` sits above ``+inf``, so before this the NaN
    candidate silently WON the merge -- a wrong selection with no symptom."""
    W = 2
    clean = _ordinary_sources(W)
    poisoned = [list(src) for src in clean]
    poisoned[1][0] = (NAN, poisoned[1][0][1])  # a VALID id, a NaN score

    # PRECONDITION: the same carrier without the NaN must succeed and select
    # something, so the failure is attributable to the NaN and not to the shape.
    baseline = merge_candidates(
        c4_carrier(_from_sources(clean)[0], world=W, device=cuda_device),
        world=W,
        rank=0,
    )
    assert baseline[0, 0].tolist() != [-1] * CAND_K

    cand = c4_carrier(_from_sources(poisoned)[0], world=W, device=cuda_device)
    with pytest.raises(IcpMergeError, match="NaN") as excinfo:
        merge_candidates(cand, world=W, rank=0)
    assert excinfo.value.status == ICP_STATUS_NAN

    status = torch.zeros(1, dtype=torch.int32, device=cuda_device)
    out = merge_candidates(cand, world=W, rank=0, status=status)
    assert int(status.item()) & ICP_STATUS_NAN
    assert out[0, 0].tolist() == [-1] * CAND_K


@pytest.mark.gpu
def test_a_nan_on_an_invalid_record_is_not_a_failure(cuda_device, c4_carrier):
    """C11/C12: validity is by **id**, never by score.

    An invalid record's score field carries no meaning, so a NaN sitting in one
    -- which is what an uninitialised or recycled padding slot looks like -- must
    not fail an invocation that is otherwise perfectly well formed.
    """
    W = 2
    clean = _ordinary_sources(W)
    clean[0] = clean[0][:12]  # 4 padding slots on source 0
    padded = [list(src) for src in clean]
    padded[0] = padded[0] + [(NAN, -1)] * 4  # NaN on an INVALID record

    fill_clean, records = _from_sources(clean)
    fill_padded, _ = _from_sources(padded)

    # PRECONDITION: the row must actually contain the NaN-bearing records, and
    # the two carriers must otherwise be the same selection problem.
    assert any(math.isnan(s) for s, gid in padded[0] if gid < 0)
    want = dedup_then_truncate(records)

    quiet = merge_candidates(
        c4_carrier(fill_clean, world=W, device=cuda_device), world=W, rank=0
    )
    noisy = merge_candidates(
        c4_carrier(fill_padded, world=W, device=cuda_device), world=W, rank=0
    )  # must NOT raise
    assert quiet[0, 0].tolist() == want
    assert torch.equal(quiet, noisy)


@pytest.mark.gpu
def test_a_supplied_status_word_is_neither_cleared_nor_read(cuda_device, c4_carrier):
    """The capturable form: no device->host sync, so no exception either.

    The wrapper-owned form is a convenience that synchronises; the supplied form
    is the one a captured caller must use, and zeroing and checking the word are
    then the caller's job. A status word nobody reads is a failure channel that
    does not exist -- so the bits have to be observable, which is what this
    asserts.
    """
    W = 2
    clean = _ordinary_sources(W)
    clean[0][0] = (NAN, clean[0][0][1])
    cand = c4_carrier(_from_sources(clean)[0], world=W, device=cuda_device)

    status = torch.full((1,), 0x40, dtype=torch.int32, device=cuda_device)
    merge_candidates(cand, world=W, rank=0, status=status)  # no raise
    code = int(status.item())
    assert code & ICP_STATUS_NAN, "the kernel must OR its bits in"
    assert code & 0x40, "the caller's pre-existing bits must survive"


@pytest.mark.gpu
def test_omitting_status_under_capture_is_refused(cuda_device, c4_carrier):
    """The one thing the convenience form may not do is silently skip the check.

    Reading the word back is a device->host sync, which capture forbids; the
    alternative -- capturing the launch and not checking C5 -- is not offered.
    """
    W = 2
    cand = c4_carrier(
        _from_sources(_ordinary_sources(W))[0], world=W, device=cuda_device
    )
    out = torch.empty((1, 1, CAND_K), dtype=torch.int32, device=cuda_device)
    status = torch.zeros(1, dtype=torch.int32, device=cuda_device)
    merge_candidates(cand, world=W, rank=0, out=out, status=status)  # warm up

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="status"), torch.cuda.graph(graph):
        merge_candidates(cand, world=W, rank=0, out=out)


@pytest.mark.gpu
def test_the_merge_addresses_this_ranks_own_heads(cuda_device, c4_carrier):
    """C4/C7 at ``H_local == 2``, where a head permutation is not the identity.

    The receive carrier holds only this rank's heads, so ``hl`` IS the address
    and no ``head_offset`` arithmetic is left to get wrong -- but a loader that
    strided by ``H_group`` (the pre-C4 layout) would read the neighbouring head
    and this is where that shows.
    """
    W, Hl = 2, 2
    per_head = {0: [(1.0, 11), (0.5, 12)], 1: [(1.0, 21), (0.5, 22)]}
    cand = c4_carrier(
        lambda s, q, h: per_head[h] if s == 0 else [],
        world=W,
        heads_local=Hl,
        device=cuda_device,
    )
    out = merge_candidates(cand, world=W, rank=1)  # head_offset = 1 * Hl = 2
    assert out[0, 0].tolist()[:2] == [11, 12]
    assert out[0, 1].tolist()[:2] == [21, 22]
