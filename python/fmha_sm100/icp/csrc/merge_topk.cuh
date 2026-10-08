// Shared device code for the ICP candidate merge (CONTRACT C4/C5/C6).
//
// This header is the SINGLE implementation of the canonical key. K2 (the NCCL
// reference merge) and K5 (the fused symmetric-memory exchange + merge) both
// include it, so the two paths cannot drift apart -- the K5-vs-K2 parity gate
// (`tests/test_exchange.py::test_exchange_is_bit_identical_to_the_reference_merge`)
// would be meaningless if each had its own key.
//
// Key (CONTRACT C5; the torch statement of the same rule is
// `icp_kernels/merge_reference.py::_stable_key`):
//     (sortable(score_bits) << 32) | ~uint32(id),   id < 0 -> 0
// score descending, then global block id ascending. `+inf` sorts above every
// finite score -- but that is NOT how the forced block survives. Under
// refined-icp-v1 C3 the forced block is carried by a RESERVED SLOT, because a
// valid ordinary `+inf` with a smaller id beats it on the ascending-id
// tie-break. And NaN no longer "must never reach here": it is DETECTED here
// and fails the invocation (C5). Both are described in the block below.
//
// Merge algorithm: one warp per (token, head) row, rank-based selection rather
// than a bitonic sort. Pass 0 elects one *representative* per distinct global
// block id (the duplicate carrying the maximum score). Pass 1: every
// representative counts how many representatives outrank it; rank < 16 wins.
// Pass 2: the winners count how many winners have a smaller block id, which
// places them ascending directly. O(N^2/32) with N <= 128 is ~128 comparisons
// per lane per pass -- far below the launch overhead, and obviously correct,
// which is what a reference has to be.
//
// DUPLICATE GLOBAL IDS ARE THE NORMAL CASE (refined-icp-v1, C3). Pass 0 did not
// exist while M0's CONTRACT C1 held: back then each block had exactly one owner
// and "block ids are distinct across ranks by construction" was true, so no
// tie-breaking was needed inside the merge at all. The refined design deletes
// that assumption -- every rank computes a partial maximum for *every* global
// block. The governing text is the `refined-icp-v1` amendment's C1/C2/C3
// (`indexer-DCP/docs/history/CONTRACT.md`): "across all W*16 incoming records,
// max-reduce duplicates before selecting distinct ordinary winners";
// truncating to 16 records before deduplication is not valid.
//
// Without pass 0 this is a SILENT WRONG ANSWER, not a hang or an assert: two
// winners sharing a gid compute the same `pos` in pass 2, so one of them
// overwrites the other and one output slot keeps its -1 pre-fill. The caller
// sees a short top-16 with a -1 inside the valid prefix, which C6 forbids --
// and main readers dereference the count-sized prefix with no negative-id
// guard.
//
// ---------------------------------------------------------------------------
// THE TWO REFINED-ICP-V1 RULES THIS HEADER IMPLEMENTS ON TOP OF PASS 0:
//
// (1) THE RESERVED FORCED SLOT (C3). The merge takes the row's forced block
//     `f = p//128` and the number of ordinary winners `q = min(15, M-1)` the
//     caller wants, and it does the reservation itself: `f` is excluded from
//     the ordinary ranking, at most `q` ordinary winners are selected, and `f`
//     is injected exactly once into the slot its ascending position demands.
//     A merge that instead returned 16 ordinary winners in ASCENDING-ID order
//     destroys the key order, so a caller holding 16 ordinary ids plus `f`
//     cannot tell which of the 16 to drop -- the 17-into-16 gap.
//
//     `f` is NOT forced with `+inf`: a valid ordinary `+inf` with a smaller id
//     ties with it under C5 and wins the ascending-id tie-break, displacing
//     the forced block. A reserved slot cannot be displaced by any score.
//
// (2) NaN FAILS THE INVOCATION (C5). `sortable_bits(NaN)` is 0xffc00000,
//     strictly ABOVE `+inf`'s 0xff800000, so an undetected NaN candidate
//     silently WINS the merge. The merge reports errors through a caller-owned
//     status word and emits an all-`-1` row for the failing row, so nothing
//     plausible reaches the reader. The host wrapper turns a non-zero status
//     into a raised exception; see
//     `icp_kernels/merge.py::merge_candidates`.

#pragma once

#include <cstdint>

namespace icp {

// Up to C = 8 with K = 16.
constexpr int kMaxCandidates = 128;
constexpr int kTopK = 16;
constexpr unsigned kFullMask = 0xffffffffu;

// No forced block for this row. Also what an INACTIVE row carries -- an
// inactive row is `(kNoForcedBlock, 0)`, i.e. no ordinary winners and no
// injection, which is C6's "inactive rows have zero valid IDs". A row with no
// forcing at all -- the plain ordinary merge -- is `(kNoForcedBlock, kTopK)`.
constexpr int32_t kNoForcedBlock = -1;

// Status bits ORed into the caller's status word. Any non-zero value FAILS THE
// INVOCATION; they are distinguished only so the failure can be diagnosed.
constexpr int32_t kStatusOk = 0;
constexpr int32_t kStatusNaN = 1;       // C5: a NaN score on a VALID record.
constexpr int32_t kStatusRowMeta = 2;   // C3: ordinary count disagrees with f.

// fp32 -> a uint32 whose UNSIGNED order is the float order.
//
// IEEE-754 fp32 is sign-magnitude, so the raw bits already order correctly
// *within* a sign but not across it, and negatives run backwards. Flipping
// every bit of a negative and only the sign bit of a non-negative folds both
// halves onto one ascending unsigned range. The map is monotone and injective,
// so it decides no ordering itself -- it only lets one integer compare make the
// decision the float compare would have made.
__device__ __forceinline__ uint32_t sortable_bits(float score) {
  uint32_t raw = __float_as_uint(score);
  // -0.0f and +0.0f compare equal as floats; make them equal as keys too.
  uint32_t bits = ((raw & 0x7fffffffu) == 0u) ? 0u : raw;
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

// The CONTRACT C5 key. Score in the high 32 bits so it dominates; the id
// COMPLEMENTED in the low 32, so that at equal score the LARGER key is the
// SMALLER id. One `>` on the uint64 therefore means "outranks" -- score
// descending, then global block id ascending -- and no comparison anywhere
// downstream needs a separate tie-break.
//
// An invalid id maps to key 0, which is strictly below every real candidate: a
// real one has a non-zero high half, because `sortable_bits` returns 0 only for
// the raw pattern 0xffffffff, a NaN, which must never reach here. A padding
// slot can therefore never outrank data.
__device__ __forceinline__ uint64_t canonical_key(float score, int32_t gid) {
  if (gid < 0) return 0ull;
  return (static_cast<uint64_t>(sortable_bits(score)) << 32) |
         static_cast<uint64_t>(~static_cast<uint32_t>(gid));
}

// Merge one row's `n` candidates, held `kpt` per lane in a strided layout
// (lane L holds indices L, L+32, L+64, ...), into `out[0..15]`: the selected
// global block ids, ascending, -1 at the tail.
//
// `forced`      the row's forced block id `f = p//128`, or `kNoForcedBlock`.
//               When >= 0 it is EXCLUDED from the ordinary ranking (C3 says
//               "exclude on EVERY rank", and excluding it here as well makes
//               the single injection true even if a producer leaks it) and
//               then injected exactly once.
// `n_ordinary`  how many ordinary winners to keep: `min(15, M-1)` when forcing,
//               `kTopK` for the plain ordinary merge, 0 for an inactive row.
//               The caller MUST pass `n_ordinary <= kTopK - 1` whenever
//               `forced >= 0`; `k2_merge_kernel` checks that against C3's own
//               formula and fails the invocation rather than overflowing the
//               16 slots.
//
// `out` must be the base of this row's 16 int32 outputs. All 32 lanes of the
// warp must call this.
template <int KPT>
__device__ __forceinline__ void warp_merge_topk16(const uint64_t (&key)[KPT],
                                                  const int32_t (&gid)[KPT],
                                                  int lane, int32_t forced,
                                                  int n_ordinary,
                                                  int32_t* out) {
  // --- pass 0: max-reduce duplicate global ids (refined-icp-v1 C3) ----------
  // `rep[i]` is true iff candidate i is the representative of its global block
  // id: no other candidate with the same id carries a strictly greater key, and
  // among exact key ties the smallest flat candidate index wins. That elects
  // exactly one representative per distinct id, and because every duplicate of
  // one id shares the same low key word (`canonical_key` puts ~id there), "the
  // greatest key" is "the greatest score" -- the max-by-id C3 asks for.
  //
  // The flat candidate index of slot i on lane L is L + i*32, matching
  // `load_candidates` below and the `n = c*K + k` enumeration the Triton and
  // torch arms tie-break on, so all three arms elect the same representative.
  // (Which duplicate is elected cannot change the emitted ids -- they are equal
  // by definition -- but it must be *exactly one*, or pass 2 collides again.)
  //
  // `gid[i] != forced` is C3's exclusion. The forced block never competes
  // for an ordinary slot on any rank, so it also must not compete here -- and
  // dropping it from the pool is what makes "inject exactly once" hold even if
  // some producer published it anyway.
  bool rep[KPT];
#pragma unroll
  for (int i = 0; i < KPT; ++i) rep[i] = (gid[i] >= 0) && (gid[i] != forced);

#pragma unroll 1
  for (int j = 0; j < 32; ++j) {
#pragma unroll
    for (int s = 0; s < KPT; ++s) {
      const uint64_t okey = __shfl_sync(kFullMask, key[s], j);
      const int32_t ogid = __shfl_sync(kFullMask, gid[s], j);
      const int oidx = j + s * 32;
#pragma unroll
      for (int i = 0; i < KPT; ++i) {
        // ogid >= 0 is implied by ogid == gid[i] once gid[i] >= 0 is known, and
        // rep[i] already carries gid[i] >= 0.
        const bool same = rep[i] && (ogid == gid[i]);
        const bool beats =
            (okey > key[i]) || ((okey == key[i]) && (oidx < lane + i * 32));
        if (same && beats) rep[i] = false;
      }
    }
  }

  // The deduplicated pool. A non-representative is retired to exactly what an
  // invalid record looks like -- key 0, id -1 -- because that is precisely its
  // semantics once its id's maximum has been taken elsewhere. Key 0 is strictly
  // below every real key (a real key is 0 only if ~id == 0 and sortable == 0,
  // i.e. id == -1, which is already invalid, or a NaN score, which C5 forbids).
  //
  // Materialising the pool rather than re-testing `rep` inside passes 1 and 2
  // also ends `key`/`gid`'s liveness here, which is what keeps the register
  // count at the pre-fix numbers: measured sm_103a 22/28/32 for KPT 1/2/4
  // against a pre-fix 23/28/32, 0 spill bytes either way, WITH THE FP32
  // GATHERED LOADER. The int32 C4 loader below costs +2 at KPT=2, so the
  // current baseline for the whole merge is 22/30/32 -- see
  // `_build.MERGE_REGISTER_BASELINE`, which is asserted by a test rather than
  // left in a comment. This pass-0 result is unaffected by that; it is about
  // materialising the pool, and it still holds.
  uint64_t rkey[KPT];
  int32_t rgid[KPT];
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    rkey[i] = rep[i] ? key[i] : 0ull;
    rgid[i] = rep[i] ? gid[i] : -1;
  }

  // --- pass 1: rank among all (deduplicated) candidates --------------------
  //
  // Every surviving candidate counts how many candidates strictly outrank it.
  // `j` walks the 32 lanes and `s` the KPT slots each lane holds, so the pair
  // (j, s) broadcasts all 32*KPT keys of the row to every lane exactly once. A
  // candidate with rank < 16 is a winner. The lane loop stays rolled
  // (`unroll 1`) so the 32 rounds are one loop body rather than 32 copies of
  // the inner KPT x KPT block.
  //
  // Unchanged from the one-owner version, and exact for the same reason it was
  // then: the surviving ids are pairwise distinct, so their keys differ in the
  // low word and no real key can tie. What changed is *why* that holds -- pass
  // 0 establishes it, where C1 used to supply it for free.
  int rank[KPT];
#pragma unroll
  for (int i = 0; i < KPT; ++i) rank[i] = 0;

#pragma unroll 1
  for (int j = 0; j < 32; ++j) {
#pragma unroll
    for (int s = 0; s < KPT; ++s) {
      const uint64_t other = __shfl_sync(kFullMask, rkey[s], j);
#pragma unroll
      for (int i = 0; i < KPT; ++i) rank[i] += (other > rkey[i]) ? 1 : 0;
    }
  }

  // C3: `n_ordinary` replaces the hard-wired kTopK. That single substitution is
  // the reservation: with q = min(15, M-1) ordinary winners the 16th slot
  // cannot be taken by an ordinary block, so the forced injection below always
  // has a slot and never evicts anyone.
  bool win[KPT];
#pragma unroll
  for (int i = 0; i < KPT; ++i)
    win[i] = (rgid[i] >= 0) && (rank[i] < n_ordinary);

  // Pre-fill the tail; winners overwrite their slots below.
  if (lane < kTopK) out[lane] = -1;
  __syncwarp();

  // --- pass 2: position among the winners, ascending by block id -----------
  //
  // Same broadcast shape, now counting only winners with a smaller block id, so
  // a winner's count IS its output slot, and the winners land in distinct slots
  // with no tie-break -- which is why the tail pre-fill above is the only thing
  // that has to write the -1s.
  //
  // Exact only because the winners are representatives and therefore hold
  // pairwise distinct ids: `pos` is a strict permutation of [0, #winners) and
  // every output slot below #winners is written exactly once. This is the step
  // that silently dropped a winner before pass 0 existed. It used to be exact
  // for a different reason -- "each block has exactly one owner", pre-refinement
  // C1 -- and that premise is gone: under fragment placement every rank scores
  // every block, so duplicate ids are the normal case, not an anomaly.
  //
  // C3: the forced block is one more id in the same ascending order, so it
  // shifts every winner above it by one slot -- that is the `forced < rgid[i]`
  // seed -- and takes the slot `fpos` = #{winners with a smaller id}. `fpos` is
  // accumulated inside the SAME broadcast loop: every lane sees every winner's
  // id, so every lane computes the identical total and no extra warp reduction
  // is needed. (`forced` is never a winner, so `fpos` collides with no `pos`:
  // winners below `forced` have pos < fpos, winners above it have pos > fpos.)
  int pos[KPT];
#pragma unroll
  // The `forced >= 0` half is NOT redundant: with no forcing `forced` is -1,
  // which is smaller than every valid id, and the seed would shift every winner
  // by one slot and leave slot 0 holding the -1 pre-fill.
  for (int i = 0; i < KPT; ++i)
    pos[i] = (forced >= 0 && forced < rgid[i]) ? 1 : 0;
  int fpos = 0;

#pragma unroll 1
  for (int j = 0; j < 32; ++j) {
#pragma unroll
    for (int s = 0; s < KPT; ++s) {
      // Non-winners contribute a sentinel that can never be "smaller".
      const int32_t og = __shfl_sync(kFullMask, win[s] ? rgid[s] : INT32_MAX, j);
      // INT32_MAX < forced is false for every legal id, so the sentinel cannot
      // inflate fpos -- and a WINNER carrying INT32_MAX cannot be confused with
      // it either, because that winner's id differs from `forced` by
      // construction.
      fpos += (og < forced) ? 1 : 0;
#pragma unroll
      for (int i = 0; i < KPT; ++i) pos[i] += (og < rgid[i]) ? 1 : 0;
    }
  }

#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    if (win[i]) out[pos[i]] = rgid[i];
  }
  // Exactly one injection, by exactly one lane. `forced < 0` (no forcing, or an
  // inactive row) writes nothing at all.
  if (lane == 0 && forced >= 0) out[fpos] = forced;
  __syncwarp();
}

// `warp_merge_topk16<1>` with the same output bits, in log-depth shuffle
// networks instead of three 32-round broadcast loops: match_any for the
// duplicate max-reduce, a 32-lane bitonic sort for the rank, a second one for
// the ascending id order. Requires one candidate per lane (n <= 32).
__device__ __forceinline__ void warp_merge_topk16_net(uint64_t key, int32_t gid,
                                                      int lane, int32_t forced,
                                                      int n_ordinary, int32_t* out) {
  // Pass 0: representative of each id = max key, ties to the lowest lane,
  // exactly the classic predicate with flat index == lane.
  bool rep = (gid >= 0) && (gid != forced);
  // Only candidates that can still be representatives walk their group, so
  // invalid padding (one large gid == -1 group) costs no rounds.
  const unsigned group = __match_any_sync(kFullMask, gid);
  unsigned others = rep ? group & ~(1u << lane) : 0u;
  while (__any_sync(kFullMask, others != 0u)) {
    const int src = others ? __ffs(others) - 1 : lane;
    const uint64_t okey = __shfl_sync(kFullMask, key, src);
    if (others) {
      if ((okey > key) || (okey == key && src < lane)) rep = false;
      others &= others - 1u;
    }
  }
  uint64_t v = rep ? key : 0ull;
  // Pass 1: descending bitonic sort; surviving keys are distinct (distinct ids
  // in the low word), so lane i holds the key of rank i.
#pragma unroll
  for (int width = 2; width <= 32; width <<= 1) {
#pragma unroll
    for (int distance = width >> 1; distance > 0; distance >>= 1) {
      const uint64_t other = __shfl_xor_sync(kFullMask, v, distance);
      const bool descending = (lane & width) == 0;
      const bool lower = (lane & distance) == 0;
      v = (descending == lower) ? (v > other ? v : other) : (v < other ? v : other);
    }
  }
  const bool win = (v != 0ull) && (lane < n_ordinary);
  const int32_t wid = win ? static_cast<int32_t>(~static_cast<uint32_t>(v)) : -1;
  const int fpos =
      __popc(__ballot_sync(kFullMask, win && forced >= 0 && wid < forced));
  // Pass 2: ascending sort of the winners' ids; non-winners sort last.
  uint32_t u = win ? static_cast<uint32_t>(wid) : 0xffffffffu;
#pragma unroll
  for (int width = 2; width <= 32; width <<= 1) {
#pragma unroll
    for (int distance = width >> 1; distance > 0; distance >>= 1) {
      const uint32_t other = __shfl_xor_sync(kFullMask, u, distance);
      const bool ascending = (lane & width) == 0;
      const bool lower = (lane & distance) == 0;
      u = (ascending == lower) ? (u < other ? u : other) : (u > other ? u : other);
    }
  }
  if (lane < kTopK) out[lane] = -1;
  __syncwarp();
  if (u != 0xffffffffu) {
    const int32_t id = static_cast<int32_t>(u);
    out[lane + ((forced >= 0 && forced < id) ? 1 : 0)] = id;
  }
  if (lane == 0 && forced >= 0) out[fpos] = forced;
  __syncwarp();
}

// COMPATIBILITY OVERLOAD -- the plain ordinary merge, no forcing.
//
// IT HAS NO CALLER IN THIS REPOSITORY. It used to have exactly one: K5 reached
// the merge through this arity and therefore carried PRE-REFINEMENT semantics,
// which meant the C3 reserved slot was absent from every row K5 produced.
// `k5_exchange.cu` now calls the six-argument form and passes
// `(kNoForcedBlock, kTopK)` itself when its caller supplies no planes.
//
// It is kept rather than deleted because this header is also compiled by the
// measurement harness, which is not in this repository (see the OPT_SELF_ELIDE
// and OPT_NODIV refusals in `k5_exchange.cu`), and because "no forcing" is one
// pair of values that is better spelled once here than at each call site.
// Nothing in this tree depends on it, so removing it is a decision for whoever
// owns that harness, not a cleanup to be taken in passing.
template <int KPT>
__device__ __forceinline__ void warp_merge_topk16(const uint64_t (&key)[KPT],
                                                  const int32_t (&gid)[KPT],
                                                  int lane, int32_t* out) {
  warp_merge_topk16<KPT>(key, gid, lane, kNoForcedBlock, kTopK, out);
}

// Load this lane's share of a row of candidates laid out as
// cand[c][t][h][k][2], where [...,0] is the fp32 score and [...,1] is the
// int32 global block id bitcast to fp32 (CONTRACT C4, never a conversion).
//
// The row is the C*K candidates at a fixed (t, h), flattened as idx = c*K + k
// and handed out strided: lane L takes idx L, L+32, L+64, ... That is the
// layout `warp_merge_topk16` assumes, and it makes consecutive lanes read
// consecutive candidates. Slots past the row's end are filled with (key 0,
// gid -1), which both passes above already treat as "never wins".
//
// C5: `nan_seen` is set (never cleared) if any VALID record this lane loads
// carries a NaN score. C5 makes that an invocation-level failure, and it has to
// be caught HERE: `canonical_key` maps NaN to 0xffc00000 << 32, strictly above
// `+inf`'s 0xff800000, so by the time the merge runs a NaN candidate is just an
// ordinary record that wins. Validity is by ID, never by score (C11/C12), so an
// invalid record's score field is not examined at all.
template <int KPT>
__device__ __forceinline__ void load_candidates(const float* __restrict__ cand,
                                                int C, int T, int Hg, int K,
                                                int t, int h, int lane,
                                                uint64_t (&key)[KPT],
                                                int32_t (&gid)[KPT],
                                                bool& nan_seen) {
  const int n = C * K;
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    const int idx = lane + i * 32;
    if (idx < n) {
      const int c = idx / K;
      const int k = idx - c * K;
      const long base =
          ((((long)c * T + t) * Hg + h) * K + k) * 2;
      const float score = cand[base];
      const int32_t g = __float_as_int(cand[base + 1]);
      // `score != score` is the IEEE NaN test; it does not fire on +/-inf,
      // which C5 says participate in the ordering normally.
      nan_seen = nan_seen || (g >= 0 && score != score);
      key[i] = canonical_key(score, g);
      gid[i] = g;
    } else {
      key[i] = 0ull;
      gid[i] = -1;
    }
  }
}

// COMPATIBILITY OVERLOAD, for callers that do not implement the C5 NaN failure
// path (K5). The flag is dead here, so the NaN test costs those callers
// nothing -- but it also means they have no failure path. Same recorded gap as
// the merge overload above.
template <int KPT>
__device__ __forceinline__ void load_candidates(const float* __restrict__ cand,
                                                int C, int T, int Hg, int K,
                                                int t, int h, int lane,
                                                uint64_t (&key)[KPT],
                                                int32_t (&gid)[KPT]) {
  bool unchecked = false;
  load_candidates<KPT>(cand, C, T, Hg, K, t, h, lane, key, gid, unchecked);
}

// ---------------------------------------------------------------------------
// THE refined-icp-v1 C4 CARRIER LOADER.
//
// C4 (`indexer-DCP/docs/history/CONTRACT.md`) replaces the M0 gathered tensor
// with an
// int32 bit-preserving carrier `[W, Qchunk, Hlocal, 16, 2]`, DESTINATION-major
// at send and SOURCE-major at receive, holding only the destination's own
// heads. This function reads the RECEIVE form:
//
//     cand[s][q][hl][k][0] = raw FP32 score bits   (int32, BITCAST)
//     cand[s][q][hl][k][1] = int32 global block id (-1 = invalid)
//
// where `s` is the SOURCE rank and `hl` is a head local to THIS rank, i.e. the
// global head `rank*Hlocal + hl` (C7's rank-major head order). There is no
// `head_offset` here BY CONSTRUCTION: a source-major carrier physically cannot
// address a peer's heads, which is the structural half of the C7 guard that
// `k2_merge.cu` and `k5_exchange.cu` otherwise have to assert at runtime (see
// their `head_offset == rank * Hl` TORCH_CHECKs).
//
// ADDITIVE, like the forced-slot overloads above. `load_candidates` is
// untouched, so the M0 gathered path still builds and the two layouts can be
// cross-checked against each other inside one binary, with the gathered merge
// as the ground truth.
//
// WHY THE ADDRESS ARITHMETIC IS THE SAME EXPRESSION. Substituting `Hg := Hl`
// and `h := hl` into `load_candidates`'s
//     ((((long)c * T + t) * Hg + h) * K + k) * 2
// gives this function's
//     ((((long)s * Q + q) * Hl + hl) * K + k) * 2
// which is also, character for character, the address K5 already computes for
// its PUSH receive window (`k5_exchange.cu`, section 5). C4's whole point is
// that K2's
// post-transport input and K5's receive window become ONE layout; the merge
// therefore needs a new ELEMENT TYPE and a new CALL SITE, not new addressing.
//
// DTYPE, NOT CONVERSION. Word 0 is loaded as int32 and bitcast with
// `__int_as_float`. The old fp32 carrier stored the ID via `__float_as_int`;
// the new one stores the SCORE via a bitcast in the other direction. The bits
// on the wire are identical either way (`abi/refined_icp_v1.py`'s
// `CARRIER_WORD_SCORE_BITS`: "raw FP32 bits, BITCAST. Never a numeric cast.").
// An id above 2^24 does not
// survive a float ROUND TRIP, which is why neither word may ever be converted.
//
// `nan_seen` carries the C5 failure path through unchanged: a NaN score on a
// VALID record must fail the invocation, and it has to be caught here because
// `canonical_key` maps NaN above `+inf`. The test is on the bitcast float, not
// on the raw word, so it is the same IEEE test the gathered loader makes.
template <int KPT>
__device__ __forceinline__ void load_candidates_c4(
    const int32_t* __restrict__ cand, int W, int Q, int Hl, int K, int q,
    int hl, int lane, uint64_t (&key)[KPT], int32_t (&gid)[KPT],
    bool& nan_seen) {
  const int n = W * K;
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    const int idx = lane + i * 32;
    if (idx < n) {
      const int s = idx / K;
      const int k = idx - s * K;
      const long base = ((((long)s * Q + q) * Hl + hl) * K + k) * 2;
      // Bitcast both words; `cand[base]` holds the FP32 score's raw bits.
      const float score = __int_as_float(cand[base]);
      const int32_t g = cand[base + 1];
      // `score != score` is the IEEE NaN test; it does not fire on +/-inf,
      // which C5 says participate in the ordering normally. Validity is by ID,
      // so an invalid record's score field is not examined at all.
      nan_seen = nan_seen || (g >= 0 && score != score);
      key[i] = canonical_key(score, g);
      gid[i] = g;
    } else {
      key[i] = 0ull;
      gid[i] = -1;
    }
  }
}

// COMPATIBILITY OVERLOAD, for a caller with no C5 failure path. Same recorded
// gap as the overloads above: taking this arity means the caller has no NaN
// failure path, not that NaN cannot occur.
template <int KPT>
__device__ __forceinline__ void load_candidates_c4(
    const int32_t* __restrict__ cand, int W, int Q, int Hl, int K, int q,
    int hl, int lane, uint64_t (&key)[KPT], int32_t (&gid)[KPT]) {
  bool unchecked = false;
  load_candidates_c4<KPT>(cand, W, Q, Hl, K, q, hl, lane, key, gid, unchecked);
}

}  // namespace icp
