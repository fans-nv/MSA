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

BOTH BACKENDS ARE GATED FOR C3, not just K2. The reserved forced slot is a
property of a merged row, and K5 produces merged rows too -- through the same
``merge_topk.cuh``, after a symmetric-memory exchange rather than after a
collective. Its arms are the ``test_k5_*`` group and they take ``icp_group_any``,
so they run at ``world_size == 1`` as well; ``tests/test_exchange.py`` keeps the
two-rank fixture for the claims that need peers.

MUTATION CHECK. Delete the ``-0.0`` flush at ``csrc/merge_topk.cuh:105`` (the
``((raw & 0x7fffffffu) == 0u) ? 0u : raw`` in ``sortable_bits``) and
``test_signed_zeros_agree_on_gpu`` must fail. Removing the matching
``torch.where`` in ``canonical_key_reference`` must fail
``test_signed_zeros_are_one_key`` on CPU. Delete pass 0 of
``warp_merge_topk16`` and ``test_duplicate_ids_are_max_reduced_before_truncation``
must fail. Replace the reserved slot with ``+inf`` forcing and
``test_a_forced_block_outranks_sixteen_legitimate_infinities`` must fail. Send
K5 back to the four-argument compatibility overload in ``csrc/k5_exchange.cu``
and every ``test_k5_*`` arm must fail while the bit-identity arms of
``tests/test_exchange.py`` keep passing -- a mutation that breaks both is
measuring the kernel's liveness, not its forcing. If any still passes, the gate
is not gating.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, NamedTuple

import pytest
import torch
import torch.distributed as dist

from fmha_sm100.icp import (
    ICP_STATUS_NAN,
    ICP_STATUS_OK,
    ICP_STATUS_ROW_META,
    KEY_BIAS,
    IcpExchange,
    IcpMergeError,
    MaskPreset,
    _build,
    canonical_key_cuda,
    canonical_key_reference,
    forced_rows,
    merge_candidates,
)
from fmha_sm100.icp import merge as merge_module
from fmha_sm100.icp.merge import (
    MERGE_ARM_ENV,
    MERGE_ARM_NAMES,
    PRODUCTION_ARMS,
    REFERENCE_ARMS,
    IcpReferenceArmWarning,
    MergeArm,
    last_merge_arm,
    merge_backend,
)
from fmha_sm100.icp.merge_reference import (
    merge_candidates_torch,
    merge_candidates_triton,
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
    with pytest.raises(RuntimeError, match="status"):
        with torch.cuda.graph(graph):
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


# --- the C3 reserved forced slot, through K5 -------------------------------
#
# Everything above reaches the merge through K2. K5 reaches the SAME
# ``merge_topk.cuh`` through the fused symmetric-memory exchange, and the
# reserved slot has to hold on both arms or the two backends select different
# blocks for the same row. It is gated here rather than in
# ``tests/test_exchange.py`` because it is a statement about a merged ROW, which
# is what this module is about; ``test_exchange.py`` gates the transport.
#
# These take ``icp_group_any`` and therefore run at ``world_size == 1`` as well.
# That is deliberate and it is not a weakening: at W=1 the exchange still
# publishes into its own symmetric window, hands off and reads itself back, so
# the row reaching the merge is a real exchanged row and the reserved slot is
# exercised end to end. What W=1 cannot show is anything about the transport --
# a global id arriving from two different ranks, the head-directed
# ``all_to_all``, the interleave. Those are ``test_exchange.py``'s and they keep
# the two-rank fixture. Run this file under ``torchrun --nproc_per_node=2`` as
# well and these arms widen to the cross-rank case for free, because the
# carrier below is built so that ``want`` does not depend on the world size.


K5_MASKS = [0, MaskPreset.HANDSHAKE, MaskPreset.LAMPORT]


def _k5_mask_id(mask) -> str:
    return getattr(mask, "name", None) or f"mask{int(mask)}"


def _k5_local_cand(records, *, world, heads_local, qchunk, device):
    """A rank's query-major ``[Qchunk, W*Hl, 16, 2]`` int32 candidates (C4).

    **Every (token, head) slab carries the same record list.** K5 publishes head
    slab ``[p*Hl, (p+1)*Hl)`` to destination ``p``, so after the exchange every
    destination row's pool is ``world`` copies of ``records`` -- and C3's
    max-by-id reduction collapses those copies back to the record set itself.
    The expectation is therefore the same row on every rank, at every head and
    at every world size, which is what lets one test gate W=1 on a workstation
    and W>=2 under ``torchrun`` without being two different tests.

    Both words are written **bitcast**, never converted, for the reason
    ``conftest.py``'s ``c4_carrier`` gives: an id above ``2**24`` does not
    survive a float round trip.
    """
    assert len(records) <= CAND_K
    heads_group = world * heads_local
    scores = torch.full((qchunk, heads_group, CAND_K), -INF, dtype=torch.float32)
    ids = torch.full((qchunk, heads_group, CAND_K), -1, dtype=torch.int32)
    for k, (score, gid) in enumerate(records):
        scores[:, :, k] = score
        ids[:, :, k] = gid
    cand = torch.empty((qchunk, heads_group, CAND_K, 2), dtype=torch.int32)
    cand[..., 0] = scores.view(torch.int32)
    cand[..., 1] = ids
    return cand.to(device).contiguous()


def _k5_forced_and_unforced(
    group, device, preset, records, *, forced, n_ordinary, qchunk=1, heads_local=1
):
    """K5's row for ``records``, with the C3 planes and without them.

    The second return value is the CONTROL, and it is not a simulation of the
    defect -- it *is* the defect: ``forced=None`` selects the plain ordinary
    merge, which is what K5 did on every row before the planes existed. Every
    test below asserts both that the forced arm is right and that the unforced
    arm is the specific wrong row, so neither direction can pass vacuously.

    ``forced`` and ``n_ordinary`` are lists of length ``qchunk``: per token row,
    never per head.
    """
    world = dist.get_world_size(group)
    heads_group = world * heads_local
    cand = _k5_local_cand(
        records, world=world, heads_local=heads_local, qchunk=qchunk, device=device
    )
    plane = torch.tensor(forced, dtype=torch.int32, device=device)
    counts = torch.tensor(n_ordinary, dtype=torch.int32, device=device)
    with IcpExchange(
        group,
        tokens=qchunk,
        heads_group=heads_group,
        heads_local=heads_local,
        slots=4,
        preset=preset,
        device=device,
    ) as ex:
        got = ex.exchange_and_merge(cand, layer_idx=0, forced=plane, n_ordinary=counts)
        # A different slot, because no two consecutive launches on a workspace
        # may use the same one.
        control = ex.exchange_and_merge(cand, layer_idx=1)
        ex.check_error()
    return got, control


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", K5_MASKS, ids=_k5_mask_id)
def test_k5_excludes_the_forced_block_from_ordinary_ranking(
    icp_group_any, cuda_device, preset
):
    """The K5 arm of ``test_the_forced_block_is_excluded_from_ordinary_ranking``.

    Same records and same ``f``. ``f = 3`` carries the best score in the
    carrier -- a producer that published it anyway, which C3 tolerates because
    the merge excludes it -- so a merge that let it compete would rank it *and*
    inject it.
    """
    f = 3
    n_ordinary = min(CAND_K - 1, f)  # C3: min(15, f)
    records = [(9.0, f), (0.9, 0), (0.8, 1), (0.7, 2), (0.6, 4), (0.5, 5)]

    want = dedup_then_truncate(records, forced=f, n_ordinary=n_ordinary)
    if_it_competed = dedup_then_truncate(records, n_ordinary=n_ordinary)
    # PRECONDITION, as in the K2 arm: with f competing it displaces an ordinary
    # winner, so this case can tell the two apart.
    assert want == _row([0, 1, 2, f])
    assert want != if_it_competed and 2 not in if_it_competed

    got, control = _k5_forced_and_unforced(
        icp_group_any, cuda_device, preset, records, forced=[f], n_ordinary=[n_ordinary]
    )

    assert got[0, 0].tolist() == want
    assert got[0, 0].tolist().count(f) == 1, (
        "exactly one injection, even when a producer leaks f"
    )
    # THE CONTROL, pinned to the exact row the unwired kernel returns rather
    # than to "something different": sixteen ordinary winners and no
    # reservation, so the six ids all place and nothing is held back.
    assert control[0, 0].tolist() == _row([0, 1, 2, 3, 4, 5])
    assert control[0, 0].tolist() != want, (
        "the unforced arm agrees with the forced one, so this case cannot "
        "tell a wired K5 from an unwired one"
    )


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", K5_MASKS, ids=_k5_mask_id)
def test_k5_keeps_the_forced_block_against_sixteen_legitimate_infinities(
    icp_group_any, cuda_device, preset
):
    """The K5 arm of ``test_a_forced_block_outranks_sixteen_legitimate_infinities``.

    This is also the defect in its user-visible form. Sixteen ordinary
    candidates carry a legitimate ``+inf`` and ids smaller than ``f``, so
    without the reservation they take all sixteen slots and ``f`` -- the query's
    **own** block -- is not in the row at all. The consumer's trip count comes
    from ``kv_len`` and never from this tensor, so a full, well formed,
    ascending row with the query's own block missing is read as if it were
    correct: no crash, no NaN, attention silently skipping the block it is
    positioned in.
    """
    f = 100
    n_ordinary = min(CAND_K - 1, f)  # == 15
    records = [(INF, gid) for gid in range(CAND_K)]

    want = dedup_then_truncate(records, forced=f, n_ordinary=n_ordinary)
    assert want == _row(list(range(15)) + [f])

    got, control = _k5_forced_and_unforced(
        icp_group_any, cuda_device, preset, records, forced=[f], n_ordinary=[n_ordinary]
    )

    assert got[0, 0].tolist() == want
    assert got[0, 0].tolist().count(f) == 1
    # C6 says the row is ascending and says nothing about provenance: the forced
    # block sits where its id demands, not at the end.
    assert got[0, 0].tolist() == sorted(got[0, 0].tolist())
    # THE CONTROL. The unwired kernel returns a full row of the sixteen
    # infinities and drops f entirely -- both halves asserted, because "the row
    # differs" would also be satisfied by a row that merely reordered.
    assert control[0, 0].tolist() == _row(list(range(CAND_K)))
    assert f not in control[0, 0].tolist(), (
        "the unforced arm kept the forced block anyway, so this case does not "
        "discriminate"
    )


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.parametrize("preset", K5_MASKS, ids=_k5_mask_id)
def test_k5_forcing_is_indexed_by_token_row_and_not_by_head(
    icp_group_any, cuda_device, preset
):
    """C3's planes are per TOKEN ROW, which only bites at ``H_local > 1``.

    K5 merges one warp per (token, head), so the obvious wrong address is the
    warp's own row index. At ``H_local == 1`` that is the same number as ``t``
    and every other test here would still pass. At ``H_local == 2`` it is not:
    a kernel indexing by the warp row would give the second head of token 0 the
    metadata of token 1, and token 1 is the inactive row -- so the two heads of
    one token would disagree about a value that depends on the query position
    alone.

    The inactive row is C6's ``(forced=-1, n_ordinary=0)``: zero valid ids, even
    though its carrier slots are full of perfectly good candidates. Its all--1
    rendering is the same one K2 emits (``test_an_inactive_row_selects_nothing``)
    and is safe for the same reason -- a padded row's ``kv_len`` is zero, so the
    consumer reads none of it.
    """
    f = 3
    n_ordinary = min(CAND_K - 1, f)
    records = [(9.0, f), (0.9, 0), (0.8, 1), (0.7, 2), (0.6, 4), (0.5, 5)]
    want = dedup_then_truncate(records, forced=f, n_ordinary=n_ordinary)

    got, control = _k5_forced_and_unforced(
        icp_group_any,
        cuda_device,
        preset,
        records,
        forced=[f, -1],
        n_ordinary=[n_ordinary, 0],
        qchunk=2,
        heads_local=2,
    )

    assert got.shape == (2, 2, CAND_K)
    for hl in range(2):
        assert got[0, hl].tolist() == want, f"active row, head {hl}"
        assert got[1, hl].tolist() == [-1] * CAND_K, f"inactive row, head {hl}"
    # PRECONDITION: the same launch's active row must not be empty, or "the
    # inactive row selects nothing" would also be true of a kernel that wrote
    # nothing at all.
    assert got[0, 0].tolist() != [-1] * CAND_K
    # THE CONTROL: with no planes both rows are ordinary, so the inactive row is
    # indistinguishable from the active one. That is what makes the plane, and
    # not the carrier, the thing under test.
    assert control[1, 0].tolist() == _row([0, 1, 2, 3, 4, 5])
    assert control[1, 0].tolist() != [-1] * CAND_K


@pytest.mark.distributed
@pytest.mark.gpu
def test_k5_refuses_an_ordinary_count_that_disagrees_with_the_c3_formula(
    icp_group_any, cuda_device
):
    """The K5 arm of ``test_n_ordinary_disagreeing_with_the_c3_formula_...``.

    ``n_ordinary`` is redundant -- the kernel could derive ``min(15, f)`` from
    ``f`` -- and it is carried anyway so a caller that derived its count from a
    batch-wide sequence length instead of the row's own ``p`` is caught.

    It is refused rather than clamped because the disagreement is not a tuning
    question: ``n_ordinary = 16`` on a forced row makes pass 2 of the merge
    write ``out[16]``, one int32 into the next row. K2 refuses the same input
    and reports it through its C5 status word; K5 has no status word and
    reports it through the error flag it already has. **The refusal is the same
    and the rendering is not** -- K5 fills the failed row with block id 0 where
    K2 publishes all ``-1`` -- so this test asserts the rendering explicitly
    rather than borrowing K2's.
    """
    world = dist.get_world_size(icp_group_any)
    f = 5
    records = [(1.0 - 0.01 * k, k) for k in range(CAND_K)]
    cand = _k5_local_cand(
        records, world=world, heads_local=1, qchunk=1, device=cuda_device
    )
    forced = torch.tensor([f], dtype=torch.int32, device=cuda_device)

    def count(value):
        return torch.tensor([value], dtype=torch.int32, device=cuda_device)

    with IcpExchange(
        icp_group_any,
        tokens=1,
        heads_group=world,
        heads_local=1,
        slots=4,
        preset=MaskPreset.LAMPORT,
        device=cuda_device,
    ) as ex:
        # PRECONDITION: the same launch with C3's own count must SUCCEED, or the
        # failure below could be caused by anything in this carrier.
        ok = ex.exchange_and_merge(
            cand, layer_idx=0, forced=forced, n_ordinary=count(min(CAND_K - 1, f))
        )
        ex.check_error()
        assert f in ok[0, 0].tolist()
        assert ok[0, 0].tolist() != [0] * CAND_K

        bad = ex.exchange_and_merge(
            cand, layer_idx=1, forced=forced, n_ordinary=count(min(CAND_K - 1, f) + 1)
        )
        with pytest.raises(RuntimeError, match="k5 exchange failed"):
            ex.check_error()
        assert bad[0, 0].tolist() == [0] * CAND_K, (
            "a refused row must be rendered as K5 renders every failure -- "
            "block id 0, not K2's all--1 row and not left untouched"
        )


# --------------------------------------------------------------------------
# 7. the three arms, bit for bit
# --------------------------------------------------------------------------
#
# `fmha_sm100.icp.merge_reference` carries a Triton and a torch statement of the
# same C3/C5/C6 semantics. They are REFERENCE tier -- they are not backends and
# nothing routes to them -- and their whole value is that they must agree with
# the shipped CUDA arm *exactly*, on the success path and on the failure path
# alike. A reference that is only approximately the contract gates nothing.
#
# Every case below mirrors one of the merge tests in section 6 and computes its
# expectation with the SAME plain-Python oracles those tests use
# (`dedup_then_truncate` and friends), not with a copied-out constant, so the
# table cannot drift away from them. Each carries a `precondition` that fails
# loudly if the case cannot discriminate -- a bit-exactness test whose input
# every implementation answers identically by luck is worse than no test,
# because it reports a guarantee it never checked.


class ArmCase(NamedTuple):
    name: str  # and the section-6 test it mirrors
    fill: Callable  # fill(source, q, h) -> [(score, gid), ...]
    world: int
    rank: int
    qchunk: int
    heads_local: int
    forced: list[int] | None  # the C3 planes, per token row
    n_ordinary: list[int] | None
    want_rows: dict  # (q, h) -> the C6 row, spelled out
    want_status: int
    precondition: Callable  # () -> None, asserted before the merge

    def planes(self, device):
        if self.forced is None:
            return {}
        return {
            "forced": torch.tensor(self.forced, dtype=torch.int32, device=device),
            "n_ordinary": torch.tensor(
                self.n_ordinary, dtype=torch.int32, device=device
            ),
        }


ARMS = (MergeArm.CUDA, MergeArm.TRITON, MergeArm.TORCH)

EMPTY_ROW = [-1] * CAND_K


def _case(
    name,
    *,
    sources,
    world,
    want_rows,
    precondition,
    rank=0,
    qchunk=1,
    heads_local=1,
    forced=None,
    n_ordinary=None,
    want_status=ICP_STATUS_OK,
    fill=None,
):
    return ArmCase(
        name=name,
        fill=fill if fill is not None else _from_sources(sources)[0],
        world=world,
        rank=rank,
        qchunk=qchunk,
        heads_local=heads_local,
        forced=forced,
        n_ordinary=n_ordinary,
        want_rows=want_rows,
        want_status=want_status,
        precondition=precondition,
    )


def _case_selects_by_the_canonical_key():
    """Mirrors test_merge_selects_by_the_canonical_key."""
    world = 2
    sources = [
        [(1.0, gid) for gid in range(s * CAND_K, (s + 1) * CAND_K)]
        for s in range(world)
    ]
    records = [r for src in sources for r in src]

    def precondition():
        # Ties everywhere, so the id half of the key is the only thing that can
        # decide -- an arm that broke ties by source rank or arrival order
        # differs here and nowhere else.
        assert len({score for score, _ in records}) == 1
        ids = [gid for _, gid in records]
        assert len(set(ids)) == len(ids) == 2 * CAND_K

    return _case(
        "selects_by_the_canonical_key",
        sources=sources,
        world=world,
        want_rows={(0, 0): dedup_then_truncate(records)},
        precondition=precondition,
    )


def _case_duplicates_are_max_reduced():
    """Mirrors test_duplicate_ids_are_max_reduced_before_truncation."""
    world = 2
    sources = [
        [(-5.0, 500)] + [(0.5 - 0.01 * i, i) for i in range(15)],
        [(+5.0, 500)] + [(0.5 - 0.01 * (15 + i), 15 + i) for i in range(15)],
    ]
    records = [r for src in sources for r in src]
    want = dedup_then_truncate(records)

    def precondition():
        assert want != min_reduce_then_truncate(records), (
            "this case cannot tell max-by-id from min-by-id"
        )
        assert 500 in want and want.count(500) == 1

    return _case(
        "duplicates_are_max_reduced",
        sources=sources,
        world=world,
        want_rows={(0, 0): want},
        precondition=precondition,
    )


def _case_dedup_before_truncation():
    """Mirrors test_deduplication_changes_which_ids_win."""
    world = 2
    sources = [
        [(1.0, gid) for gid in range(8)] + [(0.1, 200 + i) for i in range(8)],
        [(0.9, gid) for gid in range(8)] + [(0.5, 100 + i) for i in range(8)],
    ]
    records = [r for src in sources for r in src]
    want = dedup_then_truncate(records)

    def precondition():
        control = truncate_then_dedup(records)
        assert want != control
        assert control.count(-1) == 8, (
            "the control must leave the row half empty, or this case is not "
            "about the order of dedup and truncation"
        )
        assert -1 not in want

    return _case(
        "dedup_before_truncation",
        sources=sources,
        world=world,
        want_rows={(0, 0): want},
        precondition=precondition,
    )


def _case_forced_is_excluded_from_ordinary_ranking():
    """Mirrors test_the_forced_block_is_excluded_from_ordinary_ranking."""
    world, f = 2, 3
    n_ordinary = min(CAND_K - 1, f)
    sources = [
        [(9.0, f), (0.9, 0), (0.8, 1)],
        [(0.7, 2), (0.6, 4), (0.5, 5)],
    ]
    records = [r for src in sources for r in src]
    want = dedup_then_truncate(records, forced=f, n_ordinary=n_ordinary)

    def precondition():
        # f carries the best score in the carrier -- a producer that published
        # it anyway. If it were allowed to compete it would win an ordinary
        # slot AND be injected.
        if_it_competed = dedup_then_truncate(records, n_ordinary=n_ordinary)
        assert want != if_it_competed
        assert want == _row([0, 1, 2, f])
        assert f in if_it_competed and 2 not in if_it_competed

    return _case(
        "forced_is_excluded_from_ordinary_ranking",
        sources=sources,
        world=world,
        forced=[f],
        n_ordinary=[n_ordinary],
        want_rows={(0, 0): want},
        precondition=precondition,
    )


def _case_forced_beats_sixteen_infinities():
    """Mirrors test_a_forced_block_outranks_sixteen_legitimate_infinities."""
    world, f = 2, 100
    n_ordinary = min(CAND_K - 1, f)
    sources = [[(INF, gid) for gid in range(8)], [(INF, gid) for gid in range(8, 16)]]
    records = [r for src in sources for r in src]
    want = dedup_then_truncate(records, forced=f, n_ordinary=n_ordinary)

    def precondition():
        assert f not in plus_inf_forcing(records, forced=f), (
            "the +inf control did not displace the forced block, so this case "
            "does not discriminate the reserved slot from +inf forcing"
        )
        assert want == _row(list(range(15)) + [f]) == sorted(want)

    return _case(
        "forced_beats_sixteen_infinities",
        sources=sources,
        world=world,
        forced=[f],
        n_ordinary=[n_ordinary],
        want_rows={(0, 0): want},
        precondition=precondition,
    )


def _case_inactive_row():
    """Mirrors test_an_inactive_row_selects_nothing.

    Two rows in one invocation with DIFFERENT C3 metadata, which is the whole
    reason those are planes and not scalars.
    """
    world = 2
    sources = [
        [(1.0, gid) for gid in range(CAND_K)],
        [(2.0, gid) for gid in range(CAND_K, 2 * CAND_K)],
    ]
    records = [r for src in sources for r in src]
    active = dedup_then_truncate(records, forced=5, n_ordinary=5)

    def precondition():
        # The active row on the same launch must NOT be empty, or "the inactive
        # row is empty" would also be true of an arm that wrote nothing at all.
        assert active != EMPTY_ROW

    return _case(
        "inactive_row",
        sources=sources,
        world=world,
        qchunk=2,
        forced=[5, -1],
        n_ordinary=[5, 0],
        want_rows={(0, 0): active, (1, 0): EMPTY_ROW},
        precondition=precondition,
    )


def _case_this_ranks_own_heads():
    """Mirrors test_the_merge_addresses_this_ranks_own_heads."""
    per_head = {0: [(1.0, 11), (0.5, 12)], 1: [(1.0, 21), (0.5, 22)]}

    def fill(s, q, h):
        return per_head[h] if s == 0 else []

    def precondition():
        # The two heads must expect different ids, or an arm that strided by
        # H_group (the pre-C4 layout) would answer this correctly by accident.
        assert per_head[0] != per_head[1]

    return _case(
        "this_ranks_own_heads",
        sources=None,
        fill=fill,
        world=2,
        rank=1,
        heads_local=2,
        want_rows={(0, 0): _row([11, 12]), (0, 1): _row([21, 22])},
        precondition=precondition,
    )


def _case_nan_on_a_valid_record():
    """Mirrors test_a_nan_on_a_valid_record_fails_the_invocation.

    THE FAILURE PATH. All three arms must publish the same all--1 row and the
    same status bits: the failed-row rendering is defined precisely so that the
    arms stay bit-comparable here too.
    """
    world = 2
    clean = _ordinary_sources(world)
    poisoned = [list(src) for src in clean]
    poisoned[1][0] = (NAN, poisoned[1][0][1])  # a VALID id, a NaN score
    clean_records = [r for src in clean for r in src]

    def precondition():
        # The same carrier without the NaN selects something, so an all--1 row
        # is attributable to the NaN and not to the shape.
        assert dedup_then_truncate(clean_records) != EMPTY_ROW
        score, gid = poisoned[1][0]
        assert math.isnan(score) and gid >= 0

    return _case(
        "nan_on_a_valid_record",
        sources=poisoned,
        world=world,
        want_rows={(0, 0): EMPTY_ROW},
        want_status=ICP_STATUS_NAN,
        precondition=precondition,
    )


def _case_nan_on_an_invalid_record():
    """Mirrors test_a_nan_on_an_invalid_record_is_not_a_failure.

    C11/C12: validity is by **id**, never by score, so an arm that tested
    validity by score would fail an invocation that is perfectly well formed.
    """
    world = 2
    clean = _ordinary_sources(world)
    clean[0] = clean[0][:12]
    padded = [list(src) for src in clean]
    padded[0] = padded[0] + [(NAN, -1)] * 4  # NaN on an INVALID record
    records = [r for src in clean for r in src]

    def precondition():
        assert any(math.isnan(s) for s, gid in padded[0] if gid < 0)
        assert dedup_then_truncate(records) != EMPTY_ROW

    return _case(
        "nan_on_an_invalid_record",
        sources=padded,
        world=world,
        want_rows={(0, 0): dedup_then_truncate(records)},
        precondition=precondition,
    )


def _case_row_metadata_disagreement():
    """Mirrors test_n_ordinary_disagreeing_with_the_c3_formula_fails.

    The other half of the failure path: C3's cross-check, whose bits and whose
    all--1 row must be the same on all three arms.
    """
    world, f = 2, 5
    sources = _ordinary_sources(world)
    records = [r for src in sources for r in src]

    def precondition():
        # With C3's own count the same carrier succeeds and selects something,
        # so the failure is attributable to the count and to nothing else.
        assert (
            dedup_then_truncate(records, forced=f, n_ordinary=min(15, f)) != EMPTY_ROW
        )
        assert min(15, f) + 1 != min(15, f)

    return _case(
        "row_metadata_disagreement",
        sources=sources,
        world=world,
        forced=[f],
        n_ordinary=[min(15, f) + 1],
        want_rows={(0, 0): EMPTY_ROW},
        want_status=ICP_STATUS_ROW_META,
        precondition=precondition,
    )


def _case_mixed_depth_rows():
    """Mirrors test_merge_output_is_ascending_with_a_minus_one_tail.

    W=4, 8 query rows, 2 local heads, ids drawn from one 256-wide global domain
    on every source so duplicates arrive across sources many times over, and
    both short and full rows occur. This is the only case in the table with
    more than one source-major candidate block per lane (W*16 = 64), so it is
    also the only one that exercises the KPT=2 instantiation of the kernel
    against the arms' flat enumeration.
    """
    world, qchunk, heads_local = 4, 8, 2
    rng = torch.Generator().manual_seed(7)
    sources = {}
    for s in range(world):
        for q in range(qchunk):
            for h in range(heads_local):
                n = 2 if q % 3 == 0 else CAND_K
                ids = torch.randint(0, 1 << 8, (n,), generator=rng).tolist()
                scores = torch.randn(n, generator=rng).tolist()
                sources[(s, q, h)] = list(zip(scores, ids))

    want_rows = {}
    duplicates_seen = short_rows = full_rows = 0
    for q in range(qchunk):
        for h in range(heads_local):
            records = [r for s in range(world) for r in sources[(s, q, h)]]
            ids = [gid for _, gid in records]
            duplicates_seen += len(ids) - len(set(ids))
            row = dedup_then_truncate(records)
            want_rows[(q, h)] = row
            valid = [x for x in row if x >= 0]
            short_rows += len(valid) < CAND_K
            full_rows += len(valid) == CAND_K

    def precondition():
        assert duplicates_seen > 0, (
            "no duplicate global id was drawn, so this case cannot distinguish "
            "the deduplicating merge from the pre-refinement one"
        )
        assert short_rows and full_rows, (short_rows, full_rows)
        for row in want_rows.values():
            valid = [x for x in row if x >= 0]
            assert row[: len(valid)] == valid == sorted(valid)
            assert len(set(valid)) == len(valid)

    return _case(
        "mixed_depth_rows",
        sources=None,
        fill=lambda s, q, h: sources[(s, q, h)],
        world=world,
        rank=1,
        qchunk=qchunk,
        heads_local=heads_local,
        want_rows=want_rows,
        precondition=precondition,
    )


THREE_ARM_CASES = [
    _case_selects_by_the_canonical_key(),
    _case_duplicates_are_max_reduced(),
    _case_dedup_before_truncation(),
    _case_forced_is_excluded_from_ordinary_ranking(),
    _case_forced_beats_sixteen_infinities(),
    _case_inactive_row(),
    _case_this_ranks_own_heads(),
    _case_nan_on_a_valid_record(),
    _case_nan_on_an_invalid_record(),
    _case_row_metadata_disagreement(),
    _case_mixed_depth_rows(),
]


def _quietly(fn, *args, **kwargs):
    """Run something that selects a reference arm, without the warning.

    The warning is asserted where it is the subject (section 8). Here it is
    noise, and silencing it is narrow: only :class:`IcpReferenceArmWarning`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IcpReferenceArmWarning)
        return fn(*args, **kwargs)


@pytest.mark.gpu
@pytest.mark.parametrize("case", THREE_ARM_CASES, ids=lambda c: c.name)
def test_the_three_arms_agree_bit_for_bit(cuda_device, c4_carrier, case):
    case.precondition()
    cand = c4_carrier(
        case.fill,
        world=case.world,
        qchunk=case.qchunk,
        heads_local=case.heads_local,
        device=cuda_device,
    )

    outputs, statuses = {}, {}
    for arm in ARMS:
        # A caller-owned status word: nothing raises, so the FAILURE cases
        # produce a comparable output and a readable status on every arm.
        status = torch.zeros(1, dtype=torch.int32, device=cuda_device)
        outputs[arm] = _quietly(
            merge_candidates,
            cand,
            world=case.world,
            rank=case.rank,
            status=status,
            arm=arm,
            **case.planes(cuda_device),
        )
        statuses[arm] = int(status.item())
        assert last_merge_arm() is arm

    # First the contract, spelled out, so three arms agreeing on a WRONG answer
    # is still a failure.
    for (q, h), want in case.want_rows.items():
        assert outputs[MergeArm.CUDA][q, h].tolist() == want, (q, h)
    assert statuses[MergeArm.CUDA] == case.want_status

    # Then the bit-exactness, over the whole output tensor rather than the rows
    # the case spells out.
    for arm in (MergeArm.TRITON, MergeArm.TORCH):
        assert torch.equal(outputs[arm], outputs[MergeArm.CUDA]), (
            f"{arm.value} differs from cuda on case {case.name}:\n"
            f"{outputs[arm].cpu()}\nvs\n{outputs[MergeArm.CUDA].cpu()}"
        )
        assert statuses[arm] == statuses[MergeArm.CUDA], arm


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [c for c in THREE_ARM_CASES if c.want_status != ICP_STATUS_OK],
    ids=lambda c: c.name,
)
def test_every_arm_raises_the_same_failure_when_it_owns_the_status_word(
    cuda_device, c4_carrier, case
):
    """The wrapper-owned form, on the failure path, on all three arms.

    The bits are already compared through a supplied status word above; this is
    the other mode, where the invocation must FAIL rather than return a status
    nobody looks at.
    """
    case.precondition()
    assert case.want_status != ICP_STATUS_OK  # the parametrisation, asserted
    cand = c4_carrier(
        case.fill,
        world=case.world,
        qchunk=case.qchunk,
        heads_local=case.heads_local,
        device=cuda_device,
    )
    for arm in ARMS:
        with pytest.raises(IcpMergeError) as excinfo:
            _quietly(
                merge_candidates,
                cand,
                world=case.world,
                rank=case.rank,
                arm=arm,
                **case.planes(cuda_device),
            )
        assert excinfo.value.status == case.want_status, arm


@pytest.mark.gpu
def test_every_arm_ors_into_a_supplied_status_word(cuda_device, c4_carrier):
    """A supplied word is ORed into, never cleared and never read -- on every
    arm, or the capturable form means something different per arm."""
    W = 2
    poisoned = _ordinary_sources(W)
    poisoned[0][0] = (NAN, poisoned[0][0][1])
    cand = c4_carrier(_from_sources(poisoned)[0], world=W, device=cuda_device)

    for arm in ARMS:
        status = torch.full((1,), 0x40, dtype=torch.int32, device=cuda_device)
        _quietly(
            merge_candidates, cand, world=W, rank=0, status=status, arm=arm
        )  # no raise
        code = int(status.item())
        assert code & ICP_STATUS_NAN, f"{arm} must OR its bits in"
        assert code & 0x40, f"{arm} must leave the caller's bits alone"


@pytest.mark.gpu
def test_the_reference_entry_points_are_the_same_call(cuda_device, c4_carrier):
    """``merge_candidates_torch`` / ``_triton`` and ``arm=`` are one path.

    Two spellings of the same selection must not become two implementations of
    the argument contract, so they are checked to produce the same tensor and
    to record the same arm.
    """
    W = 2
    cand = c4_carrier(
        _from_sources(_ordinary_sources(W))[0], world=W, device=cuda_device
    )
    for arm, fn in (
        (MergeArm.TORCH, merge_candidates_torch),
        (MergeArm.TRITON, merge_candidates_triton),
    ):
        direct = _quietly(fn, cand, world=W, rank=0)
        assert last_merge_arm() is arm
        through = _quietly(merge_candidates, cand, world=W, rank=0, arm=arm)
        assert torch.equal(direct, through)


def test_the_reference_module_imports_on_a_host_with_no_triton():
    """The lazy import, checked by denying triton rather than by reading the
    source.

    A host without triton must still be able to import the module, read it and
    run the torch arm. In-process this cannot be checked -- triton may already
    be imported by something else -- so it is checked in a child with
    ``sys.modules['triton'] = None``, which makes ``import triton`` raise.
    """
    import os
    import subprocess
    import sys

    from fmha_sm100.icp import merge_reference

    program = (
        "import sys; sys.modules['triton'] = None\n"
        "try:\n"
        "    import triton\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('the denial did not take')\n"
        "from fmha_sm100.icp.merge_reference import merge_candidates_torch\n"
        "print('ok')\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(merge_reference.__file__)))
    done = subprocess.run(
        [sys.executable, "-c", program], cwd=root, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().endswith("ok")


# --------------------------------------------------------------------------
# 8. arm selection
# --------------------------------------------------------------------------
#
# The arms are bit-exact by construction -- section 7 is precisely that gate --
# so NO output comparison can recover which one ran. That is what makes the
# resolution rules load-bearing rather than ergonomic: `auto` must never reach a
# reference arm, an explicit `cuda` must never degrade, an explicit reference
# arm must be visible, and an unknown name must be refused rather than
# defaulted.


@pytest.fixture()
def fresh_arm_warnings():
    """The reference-arm warning is one-time per arm, by design and by module
    state. A test that wants to observe it has to clear that state."""
    merge_module._WARNED_REFERENCE_ARMS.clear()
    yield
    merge_module._WARNED_REFERENCE_ARMS.clear()


def _k2_is_unusable(monkeypatch, reason="nvcc is not on PATH in this container"):
    """Simulate an unbuildable K2 without breaking anything else.

    `torch.cuda.is_available` is forced True as well, so the failure under test
    is the *loader's*, and the test says the same thing on a GPU node and on a
    workstation.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*args, **kwargs):
        raise RuntimeError(reason)

    monkeypatch.setattr(_build, "_k2", boom)
    return reason


def test_the_two_tiers_partition_the_arms():
    assert PRODUCTION_ARMS == {MergeArm.CUDA}
    assert REFERENCE_ARMS == {MergeArm.TRITON, MergeArm.TORCH}
    assert PRODUCTION_ARMS | REFERENCE_ARMS == set(MergeArm)
    assert not (PRODUCTION_ARMS & REFERENCE_ARMS)
    assert MERGE_ARM_NAMES == ("auto", "cuda", "triton", "torch")


@pytest.mark.gpu
def test_auto_resolves_to_a_production_arm(monkeypatch):
    monkeypatch.delenv(MERGE_ARM_ENV, raising=False)
    for requested in (None, "auto", "AUTO "):
        assert merge_backend(requested) in PRODUCTION_ARMS
        assert merge_backend(requested) is MergeArm.CUDA
        assert last_merge_arm() is MergeArm.CUDA


def test_auto_raises_and_names_the_reason_when_k2_is_unusable(monkeypatch):
    """`auto` never degrades to a reference arm -- it fails loudly.

    The message has to carry the real reason: "the merge is slow" with no
    explanation is how a build failure becomes a permanent silent fallback.
    """
    monkeypatch.delenv(MERGE_ARM_ENV, raising=False)
    reason = _k2_is_unusable(monkeypatch)
    for requested in (None, "auto"):
        with pytest.raises(RuntimeError) as excinfo:
            merge_backend(requested)
        message = str(excinfo.value)
        assert reason in message, "the underlying reason must be named"
        assert "reference" in message, (
            "the message must say that auto will not route production traffic "
            "onto a reference implementation"
        )
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert reason in str(excinfo.value.__cause__)


def test_explicit_cuda_raises_rather_than_degrading(monkeypatch):
    # A benchmark that silently measured Triton while reporting "K2" would be
    # worse than a crash.
    reason = _k2_is_unusable(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        merge_backend("cuda")
    assert reason in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


@pytest.mark.gpu
def test_a_merge_call_does_not_degrade_either(monkeypatch, cuda_device, c4_carrier):
    """The same rule at the call site, not only in the resolver."""
    W = 2
    cand = c4_carrier(
        _from_sources(_ordinary_sources(W))[0], world=W, device=cuda_device
    )
    _k2_is_unusable(monkeypatch)
    with pytest.raises(RuntimeError, match="nvcc"):
        merge_candidates(cand, world=W, rank=0)


@pytest.mark.parametrize("arm", sorted(REFERENCE_ARMS, key=lambda a: a.value))
def test_a_reference_arm_is_honoured_but_warns_and_is_recorded(arm, fresh_arm_warnings):
    with pytest.warns(IcpReferenceArmWarning) as record:
        assert merge_backend(arm.value) is arm
    assert last_merge_arm() is arm
    message = str(record[0].message)
    assert arm.value in message and "REFERENCE" in message
    assert "production" in message

    # ... and it is ONE-TIME: a per-call warning is a warning a caller filters
    # out wholesale, which is how the tier stops being visible at all.
    with warnings.catch_warnings():
        warnings.simplefilter("error", IcpReferenceArmWarning)
        assert merge_backend(arm.value) is arm


def test_an_unknown_arm_is_refused_by_name(fresh_arm_warnings):
    before = merge_module._last_arm
    with pytest.raises(ValueError) as excinfo:
        merge_backend("trtion")
    message = str(excinfo.value)
    assert "trtion" in message
    for name in MERGE_ARM_NAMES:
        assert name in message, "the legal arms must be listed"
    # Refused, not defaulted: a misspelled 'triton' that silently resolved to
    # 'auto' would report one arm and measure another, and no output comparison
    # could catch it.
    assert merge_module._last_arm is before


def test_the_env_var_is_honoured(monkeypatch, fresh_arm_warnings):
    monkeypatch.setenv(MERGE_ARM_ENV, "torch")
    with pytest.warns(IcpReferenceArmWarning, match=MERGE_ARM_ENV):
        assert merge_backend() is MergeArm.TORCH
    assert last_merge_arm() is MergeArm.TORCH
    # an explicit argument still wins over the environment
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IcpReferenceArmWarning)
        assert merge_backend("triton") is MergeArm.TRITON


def test_an_unknown_env_value_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(MERGE_ARM_ENV, "k2")
    with pytest.raises(ValueError) as excinfo:
        merge_backend()
    message = str(excinfo.value)
    assert MERGE_ARM_ENV in message and "k2" in message
    assert "auto" in message


@pytest.mark.gpu
def test_the_default_merge_path_is_still_the_cuda_arm(
    cuda_device, c4_carrier, monkeypatch
):
    """`arm=None` is `auto` is K2, and the call records it."""
    monkeypatch.delenv(MERGE_ARM_ENV, raising=False)
    W = 2
    cand = c4_carrier(
        _from_sources(_ordinary_sources(W))[0], world=W, device=cuda_device
    )
    out = merge_candidates(cand, world=W, rank=0)
    assert last_merge_arm() is MergeArm.CUDA
    assert out[0, 0].tolist() != EMPTY_ROW


# --------------------------------------------------------------------------
# 9. the register baseline
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_the_merge_has_no_stack_or_local_memory_on_the_target_architecture():
    from fmha_sm100.icp import _build

    _build._k2()
    profile = _build.merge_register_profile()
    assert set(profile) == {1, 2, 4}, (
        f"missing/unexpected KPT resources on sm_{_build.arch()}: {profile}; "
        "cuobjdump must be on PATH"
    )
    for kpt, got in profile.items():
        assert got["stack"] == 0 and got["local"] == 0, (
            f"sm_{_build.arch()} KPT={kpt} spills: {got}"
        )


@pytest.mark.gpu
def test_the_merge_register_profile_is_the_recorded_baseline():
    """Assert the profile rather than leave it in a comment.

    ``KPT = ceil(W*16 / 32)``, so KPT 1/2/4 is ``W`` 2/4/8 and only 1 and 2 are
    supported degrees.

    **The baseline is 22/30/32, and it used to be 22/28/32.** The +2 at KPT=2
    arrives with the int32 C4 loader, bisected compile-only to M1 -- S1
    22/28/32, S7 22/28/32, M1 22/30/32, on both ``-gencode`` forms and
    reproduced against M1's unmodified file. The C4 carrier is required by the
    frozen ABI, so the loader is not optional.

    **Do not "fix" this back to 28 without re-measuring.** Two
    semantics-preserving rewrites of the loader's address arithmetic both give
    24/28/32: the two registers can be *moved* from W=4 to W=2, not recovered.
    A change that makes KPT=2 read 28 has almost certainly taken 24 at KPT=1,
    which this test would catch and a comment would not.

    Zero spills is the part that is not negotiable -- a spill is a different
    kind of regression from two registers, and 30 registers at 256 threads is
    7680 of an SM's 65536, so the occupancy bound does not move.
    """
    from fmha_sm100.icp import _build

    if _build.arch().removeprefix("sm_").rstrip("a") != "103":
        pytest.skip(
            f"the 22/30/32 register baseline is recorded for sm_103, not "
            f"sm_{_build.arch()}; exact target counts need an independent "
            "source/toolchain-qualified baseline. The separate spill gate "
            "still runs on every target architecture."
        )
    _build._k2()
    profile = _build.merge_register_profile()
    assert profile, (
        "no resource usage could be read; cuobjdump must be on PATH for this "
        "gate to mean anything"
    )
    assert set(profile) == set(_build.MERGE_REGISTER_BASELINE), (
        f"instantiated KPTs changed: {sorted(profile)}"
    )
    for kpt, want in sorted(_build.MERGE_REGISTER_BASELINE.items()):
        got = profile[kpt]
        assert got["stack"] == 0 and got["local"] == 0, (
            f"KPT={kpt} spills: {got}. A spill is not a tuning question."
        )
        assert got["regs"] == want, (
            f"KPT={kpt} uses {got['regs']} registers, baseline {want}. "
            "If this is deliberate, re-measure ALL of KPT 1/2/4 and move the "
            "baseline in _build.MERGE_REGISTER_BASELINE with the numbers -- "
            "the two registers at KPT=2 can be moved to KPT=1 but not removed."
        )
