// The selector's RADIX-SELECT arm -- the one that ships.
//
// Same selection problem, same canonical key (`merge_topk.cuh::canonical_key`,
// CONTRACT C5) and same partition/combine structure as `local_candidates.cuh`.
// Only the inner algorithm differs: MSB-first radix SELECT, one 256-bin
// histogram per 8-bit digit, a threshold bin, and a 16-element rank at the end,
// instead of a full block sort of all 128 * Items slots. It keeps the sort
// arm's provisioning -- 128 threads, one CTA per row-partition -- and never
// stages the threshold bin into shared memory, refining the bin with another
// histogram pass instead.
//
// Two producer kernels, both selecting into `sm.ranked`:
//   `local_candidates_radix_kernel`   register depth from the host ladder.
//   `local_candidates_capped_kernel`  register depth a fixed cap, the row
//                                     streamed in chunks, so depth and shared
//                                     footprint stop tracking capacity.
//
// THE fp32 VALUE PAYLOAD IS DROPPED. The score is recovered from `key >> 32` by
// inverting `sortable_bits`, exactly, for every input except -0.0, which comes
// back as +0.0. That is not a loss: C5 mandates the flush, because a block
// score is a max reduction and IEEE-754 leaves the sign of `fmax(-0.0, +0.0)`
// implementation-defined, so two shardings may legitimately disagree on it.
// CONSEQUENCE: a gate that bit-compares this arm's candidate tensor against a
// -0.0-preserving reference must flush the reference too. See CONTRACT.md.
#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "local_candidates.cuh"  // kLocalThreads, kLocalPartition, PDL helpers
#include "merge_topk.cuh"        // canonical_key, sortable_bits, kTopK

namespace icp {

// Negative-control hooks, COMPILED OUT of the shipping extension.
// `ICP_RS_NEGATIVE_CONTROL` is defined only by `local_candidates_rs_control.cu`,
// which builds into its own extension (`icp_select_control`, see
// `icp_kernels/_build.py::DEFINES`). Without it `ICP_RS_CTL(n)` is the literal
// `false`, the extra kernel parameter does not exist, and ptxas reports
// byte-identical register/spill/shared numbers -- which the gate checks.
//
// A gate nobody has seen fail is not a gate (CONTRACT C10). Each control
// perturbs exactly ONE site and leaves everything else consistent:
//   1  the selector skips its last partition            -> STALE partials
//   2  exclusion ignores forced_col[] and drops valid-1 -> wrong forced block
//      (refined-icp-v1: was "forcing ignores owns[token]"; owns[] no longer
//       exists, so the control was re-pointed at the exclusion that replaced
//       the +inf forcing. Same single site, same observable.)
//   3  the global id drops scan_block_begin            -> C10 control #1
//      (refined-icp-v1: was `(rank+1)%world`. world/rank no longer form the id;
//       dropping the scan offset is the refined-design port mistake with the
//       same signature -- every id low by a constant.)
//   4  the combine reads count-16 candidates            -> drops a partition
//   5  the select keeps 15, not 16                      -> drops one winner
//      (VACUOUS by construction, and that is a CONTRACT PROPERTY, not a hole:
//       refined C3 reserves one of the 16 output slots for the forced block, so
//       the receiver never consumes more than min(15, M-1) ordinary candidates
//       and a local list of 15 is already exact. Control 9 is its firing
//       sibling. Do not "fix" control 5 by weakening it.)
//   6  the bounded arm's live-count bound is short by one -> observable only on
//                                                            a row shorter than
//                                                            16
//   9  the select keeps 14, not 16                      -> fires; control 5's
//                                                          sibling, see above
// Control 1 is the one that needs a poisoned scratch to fire at all: a skipped
// write leaves the previous launch's CORRECT value behind, so against a warm
// buffer the perturbation is invisible.
// refined-icp-v1: the definition moved to `local_candidates.cuh`, which this
// file already includes, so the SORT arm is gated by the same macros instead
// of having no controls at all. The control numbering did not change.
// Controls 7 and 8, which apply to both arms, are documented at the definition
// site in `local_candidates.cuh`.

constexpr int kRadixBits = 8;
constexpr int kRadixBins = 1 << kRadixBits;   // 256
constexpr int kRadixPasses = 64 / kRadixBits; // 8, worst case
constexpr int kBinsPerLane = kRadixBins / 32; // 8

// Exact inverse of `merge_topk.cuh::sortable_bits`, up to the C5 -0.0 flush:
// sortable_bits maps -0.0 to 0x80000000 (the +0.0 key), so this returns +0.0.
__device__ __forceinline__ float score_from_sortable(uint32_t s) {
  const uint32_t bits = (s & 0x80000000u) ? (s ^ 0x80000000u) : ~s;
  return __uint_as_float(bits);
}

// Same output contract as `store_local_candidate`, without the value payload.
__device__ __forceinline__ void store_local_candidate_keyonly(float* out,
                                                              int slot,
                                                              uint64_t key) {
  const int32_t gid =
      key ? static_cast<int32_t>(~static_cast<uint32_t>(key)) : -1;
  out[slot * 2] =
      key ? score_from_sortable(static_cast<uint32_t>(key >> 32)) : -CUDART_INF_F;
  out[slot * 2 + 1] = __int_as_float(gid);
}

struct RadixSelectSmem {
  int hist[kRadixBins];
  int bin;        // threshold digit
  int above;      // candidates strictly above the threshold digit
  int bin_count;  // candidates inside the threshold digit
  int win_count;
  uint64_t win_keys[kTopK];  // collection order (atomicAdd), unordered
  uint64_t ranked[kTopK];    // descending, slot j == j-th largest
};

// Select the top `kTopK` of the block's register-resident keys and leave them in
// `sm.ranked` in descending key order, padding with 0 (== CONTRACT C4 invalid).
//
// `live_items` is a BLOCK-UNIFORM bound: register `i` is skipped entirely when
// `i >= live_items`, which is what makes the cost O(valid) rather than
// O(capacity). Registers `i < live_items` all participate, including the lanes
// inside the last partly-filled register whose key is 0; those sort last and
// emit (-inf, -1), which is the same answer as excluding them.
//
// `live_count` is the block-uniform number of live ELEMENTS, or -1 for "do not
// bound `wanted`". A row holding fewer than 16 live blocks can reach a `wanted`
// of 16 only through the zero padding, whose keys are identical, so no digit
// separates them and the loop walks all 8 radix digits before falling out to
// the padding path. With fewer than 16 live keys "top 16" IS "all of them", so
// selecting `live_count` and letting sm.ranked's zero padding fill the rest
// with (-inf, -1) is the same answer, reached in one pass.
//
// Exactness. Let S(d) = sum of counts of digits >= d at the current shift.
// Choose b with S(b) >= need > S(b+1). Then every alive key whose digit exceeds
// b is a definite winner (there are S(b+1) < need of them), every alive key
// whose digit is below b is a definite loser, and the residual requirement
// `need - S(b+1)` is <= count[b] by construction, so the recursion is
// well-founded. After all 8 digits the alive set is a set of *identical* 64-bit
// keys; since the ids WITHIN one row are distinct (see the note below) and a
// distinct id gives a distinct low half, identical keys can only be key 0,
// whose output is (-inf, -1) -- which is exactly what the zero padding of
// `sm.ranked` writes.
//
// refined-icp-v1: this argument survives, but for a NARROWER reason, and the
// difference matters one level up. WITHIN one rank's list the ids are still
// distinct, because they are now `scan_block_begin + global_block_stride * column` over distinct
// columns of one row -- so the select is still exact. ACROSS ranks they are no
// longer distinct: refined C1 gives every rank a fragment of every logical
// block, so W ranks routinely emit the SAME id with different partial maxima.
// That is the RECEIVER's problem and it is handled there:
// `merge_topk.cuh::warp_merge_topk16` pass 0 elects one representative per
// distinct global id before ranking. Do not carry this row's distinctness
// argument across the transport -- it is a property of one rank's row only.
template <int Items>
__device__ __forceinline__ void radix_select_top16(const uint64_t (&keys)[Items],
                                                   int live_items,
                                                   int live_count,
                                                   RadixSelectSmem& sm
                                                       ICP_RS_CTL_PARAM) {
  static_assert(Items <= 32, "the alive/win masks are 32-bit");
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  // `live_count` is the block-uniform number of live ELEMENTS in this
  // partition, or -1 for "do not bound". `wanted` was unconditionally kTopK,
  // and when a row holds fewer than 16 live blocks the threshold `need` can
  // only be reached by the ZERO PADDING
  // (`merge_topk.cuh::canonical_key(-inf, -1) == 0`). Those keys are
  // identical, so no digit ever separates them, `bin_count <= need` never
  // holds, and the loop walks all 8 radix digits before falling out to the
  // padding path the comment below describes. Decode never sees this (valid
  // ~= 234 at 120k / C=4); prefill sees it on the first chunk of every
  // request and throughout short-context serving.
  //
  // Exactness: with fewer than 16 live keys, "top 16" IS "all of them", so
  // selecting `live_count` and letting sm.ranked's zero padding fill slots
  // live_count..15 with (-inf, -1) is the same answer, reached in one pass.
  // refined-icp-v1: control 5 is PROVABLY VACUOUS under this contract and
  // control 9 is its non-vacuous sibling. The final output is
  // min(15, M-1) ordinary winners plus one RESERVED slot for the forced block,
  // so the receiver never consumes more than 15 ordinary candidates. A local
  // list of 15 is therefore already exact: if block b were displaced from a
  // rank's top-15 by 15 locally better blocks, each of those 15 has a global
  // maximum at least its score on this rank, hence all 15 outrank b globally
  // and b is not in the global top-15 either. Keeping 16 is one slot of slack
  // that M0 needed (disjoint per-rank block sets) and refined C1 does not.
  // Dropping to 14 crosses the real boundary and is detected.
  const int base_wanted =
      ICP_RS_CTL(9) ? kTopK - 2 : (ICP_RS_CTL(5) ? kTopK - 1 : kTopK);
  const bool bounded = (live_count >= 0 && live_count < base_wanted);
  // Guarded at `live_count >= 1` so control 6 cannot drive `need` negative and
  // turn a wrong answer into an assert.
  const int wanted =
      (bounded && ICP_RS_CTL(6) && live_count >= 1)
          ? live_count - 1
          : (bounded ? live_count : base_wanted);

  // Nothing to select. TWO distinct ways to get here, and the second one only
  // became reachable with refined C3:
  //
  //   live_items <= 0   this partition holds no slots at all.
  //   wanted   <= 0     it holds slots but ZERO ORDINARY candidates, because
  //                     the row's only live column IS the excluded (forced)
  //                     one. That is every query position in logical block 0:
  //                     f = p//128 = 0, M = 1, so the live extent is 1 column
  //                     and `short_row` passes `remaining - excluded_here` = 0.
  //                     It is the first 128 tokens of every request, not a
  //                     corner case.
  //
  // Without the `wanted <= 0` arm the bin scan below cannot fire -- its straddle
  // test is `exclusive < need && need <= inclusive`, and at need == 0 no lane
  // satisfies `exclusive < 0` -- so `sm.bin` is never written and the
  // `assert(bin >= 0)` below fires, killing the context. Measured on GB300
  // sm_103 before this guard: that device-side assertion fired at Items=16 and
  // cascaded into 84 test failures, because a poisoned context fails
  // everything after it.
  //
  // The answer here is exact, not a fallback: with no ordinary candidates the
  // local list is empty, which is what `sm.ranked` full of key 0 means -- every
  // slot decodes to C4's invalid record (-inf, -1). The receiver then reserves
  // the forced slot and injects f, which is the whole selection for such a row.
  //
  // It also hardens control 6 (`live_count - 1`), which reaches wanted == 0 at
  // live_count == 1; the pre-existing `live_count >= 1` guard stopped it going
  // negative but not zero.
  if (live_items <= 0 || wanted <= 0) {
    if (tid < kTopK) sm.ranked[tid] = 0ull;
    __syncthreads();
    return;
  }

  uint32_t alive =
      live_items >= 32 ? 0xffffffffu : ((1u << live_items) - 1u);
  uint32_t win = 0u;
  int need = wanted;

  for (int shift = 64 - kRadixBits; shift >= 0; shift -= kRadixBits) {
#pragma unroll
    for (int i = tid; i < kRadixBins; i += kLocalThreads) sm.hist[i] = 0;
    __syncthreads();

    // Warp-aggregated shared atomics. Attention logits share a sign and an
    // exponent, so a whole CTA can land in one bin and a plain atomicAdd would
    // serialise; __match_any_sync collapses each warp's same-digit lanes into
    // one add.
#pragma unroll
    for (int i = 0; i < Items; ++i) {
      if (i < live_items) {
        const bool a = (alive >> i) & 1u;
        const int digit =
            a ? static_cast<int>((keys[i] >> shift) & (kRadixBins - 1)) : -1;
        const unsigned peers =
            __match_any_sync(0xffffffffu, static_cast<unsigned>(digit));
        if (digit >= 0 && (__ffs(peers) - 1) == lane) {
          atomicAdd(&sm.hist[digit], __popc(peers));
        }
      }
    }
    __syncthreads();

    // One warp reduces the 256 bins: lane l owns the 8 bins at 8*(31-l), so
    // ascending lane == descending digit and an ordinary inclusive scan gives
    // the descending running total.
    if (tid < 32) {
      const int base = (31 - lane) * kBinsPerLane;
      int chunk = 0;
#pragma unroll
      for (int j = 0; j < kBinsPerLane; ++j) chunk += sm.hist[base + j];
      int inclusive = chunk;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += other;
      }
      const int exclusive = inclusive - chunk;
      // Exactly one lane straddles `need`: the pass-1 population is
      // live_items*128 >= 128 >= kTopK, and every later pass runs on a bin
      // whose count is > need (otherwise the previous pass exited).
      if (exclusive < need && need <= inclusive) {
        int accumulated = exclusive;
        int chosen = -1, above = 0, count = 0;
#pragma unroll
        for (int j = kBinsPerLane - 1; j >= 0; --j) {
          const int c = sm.hist[base + j];
          if (chosen < 0 && accumulated + c >= need) {
            chosen = base + j;
            above = accumulated;
            count = c;
          }
          accumulated += c;
        }
        sm.bin = chosen;
        sm.above = above;
        sm.bin_count = count;
      }
    }
    __syncthreads();

    const int bin = sm.bin;
    const int above = sm.above;
    const int bin_count = sm.bin_count;
    assert(bin >= 0);

#pragma unroll
    for (int i = 0; i < Items; ++i) {
      if (i < live_items && ((alive >> i) & 1u)) {
        const int digit = static_cast<int>((keys[i] >> shift) & (kRadixBins - 1));
        if (digit > bin) {
          win |= 1u << i;
          alive &= ~(1u << i);
        } else if (digit < bin) {
          alive &= ~(1u << i);
        }
      }
    }
    need -= above;
    if (bin_count <= need) {
      win |= alive;
      alive = 0u;
      need -= bin_count;
      break;
    }
    __syncthreads();  // everyone has read sm.bin before the next pass clears
  }
  // `need > 0` here means the alive set is a run of identical keys that the
  // 64-bit prefix never separated, which can only be key 0; the zero padding
  // below supplies those slots.

  if (tid == 0) sm.win_count = 0;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    if ((win >> i) & 1u) {
      const int slot = atomicAdd(&sm.win_count, 1);
      if (slot < kTopK) sm.win_keys[slot] = keys[i];
    }
  }
  __syncthreads();

  if (tid < kTopK) {
    const int found = min(sm.win_count, kTopK);
    const uint64_t mine = tid < found ? sm.win_keys[tid] : 0ull;
    int rank = 0;
#pragma unroll
    for (int j = 0; j < kTopK; ++j) {
      const uint64_t other = j < found ? sm.win_keys[j] : 0ull;
      rank += (other > mine || (other == mine && j < tid)) ? 1 : 0;
    }
    // Strict-greater plus an index tiebreak is a permutation of 0..15. Equal
    // keys can only be key 0, which all emit (-inf, -1), so the collection
    // order that atomicAdd produced never reaches the output.
    sm.ranked[rank] = mine;
  }
  __syncthreads();
}

// Drop-in for `local_candidates_kernel`: identical signature, identical refined
// C3 exclusion (per token, EVERY rank, no +inf, `forced_col[token]`, before
// truncation), identical refined C1 global id formation at emit, identical NaN
// rejection, identical PDL. The two arms are changed together on purpose: the
// dispatch gate compares their outputs, so a divergence here is a silent
// selection difference between the arm that is tested and the arm that ships.
// `short_row` bounds the select by this partition's live extent.
template <int Items>
__global__ void __launch_bounds__(kLocalThreads) local_candidates_radix_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    float* __restrict__ candidates, int heads, int blocks, int partitions,
    int scan_block_begin, int global_block_stride, bool use_pdl, bool short_row ICP_RS_CTL_PARAM) {
  __shared__ RadixSelectSmem sm;

  local_pdl_wait(use_pdl);
  const int row = blockIdx.x / partitions;
  const int partition = blockIdx.x % partitions;
  const int token = row / heads;
  if (ICP_RS_CTL(1) && partitions > 1 && partition == partitions - 1) {
    local_pdl_complete(use_pdl);
    return;
  }
  if (threadIdx.x == 0 && active[token]) {
    assert(nvalid[token] >= 0 && nvalid[token] <= blocks);
    assert(forced_col[token] < 0 ||
           forced_col[token] < min(nvalid[token], blocks));
  }
  const int valid = active[token] ? max(0, min(nvalid[token], blocks)) : 0;
  // refined C3: exclusion is unconditional on every rank; there is no owner.
  // Control 2 now perturbs the EXCLUSION rather than the +inf forcing it
  // replaced: it excludes column valid-1 no matter what the metadata says,
  // which is the refined analogue of the vendor's scalar `force_end_blocks=1`
  // -- right only on rows whose forced block happens to be the last live one.
  const int excluded =
      ICP_RS_CTL(2) ? valid - 1 : (active[token] ? forced_col[token] : -1);
  const int start = partition * kLocalPartition;
  const int remaining = valid - start;
  const int live_items =
      remaining <= 0
          ? 0
          : min(Items, (remaining + kLocalThreads - 1) / kLocalThreads);
  // TRUNCATION TRAPS. Both halves of "the launch covers the row", asserted on
  // the device because the host may not ask the device anything here (N2, the
  // nonblocking execution policy): a rung that does not span its partition, and
  // a partition COUNT that does not span the row. The second is not implied by
  // the first -- every partition that IS launched can be fully spanned while
  // the row runs past the last one, which is what an extent between
  // kLocalPartition and `valid` produces -- and neither is visible in the
  // output: the dropped columns simply never become candidates.
  //
  // Both bands now size the launch from `max_local_blocks`, and `valid` is
  // clamped to `blocks` below, so neither can fire from the shipping entry
  // points. That is the point: this is the statement of the invariant the host
  // guarantees by construction, kept so that a future launcher which re-derives
  // the extent from per-step data traps instead of silently truncating.
  if (threadIdx.x == 0) {
    assert(min(remaining, kLocalPartition) <= kLocalThreads * Items);
    assert(static_cast<int64_t>(valid) <=
           static_cast<int64_t>(partitions) * kLocalPartition);
  }
  // The excluded column is not an ordinary candidate, so the short-row bound
  // must not count it: otherwise `wanted` exceeds the number of separable keys
  // and the select walks all 8 digits to reach the same answer through the zero
  // padding. Correct either way, so this is a bound-tightness fix, not a
  // selection fix -- stated explicitly so nobody later "fixes" it the other way.
  const int excluded_here =
      (excluded >= start && excluded < min(valid, start + kLocalPartition)) ? 1
                                                                            : 0;

  uint64_t keys[Items];
  bool valid_scores = true;
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    keys[i] = 0ull;
    if (i < live_items) {
      const int local = start + threadIdx.x + i * kLocalThreads;
      const bool live =
          local < (ICP_RS_CTL(7) ? valid + 1 : valid) && local != excluded;
      // refined C1: absolute logical id. Control 3 stays a global-id
      // perturbation, now expressed against the refined formation -- it drops
      // the scan offset, which is exactly the mistake a port of the M0 kernel
      // makes when a wave does not start at block 0.
      const int32_t gid =
          live ? (ICP_RS_CTL(3) ? local : scan_block_begin + global_block_stride * local) : -1;
      const float raw_score =
          live ? scores[static_cast<int64_t>(row) * blocks + local]
               : -CUDART_INF_F;
      valid_scores = valid_scores && (!live || !isnan(raw_score));
      keys[i] = canonical_key(raw_score, gid);
    }
  }
  assert(valid_scores);

  radix_select_top16<Items>(
      keys, live_items,
      short_row ? min(remaining, kLocalThreads * Items) - excluded_here : -1,
      sm ICP_RS_CTL_ARG);
  // Control 8: skip ONE live producer write. See local_candidates.cuh.
  if (threadIdx.x < kTopK && !(ICP_RS_CTL(8) && threadIdx.x == kTopK - 1)) {
    store_local_candidate_keyonly(
        candidates + static_cast<int64_t>(blockIdx.x) * kTopK * 2, threadIdx.x,
        sm.ranked[threadIdx.x]);
  }
  __syncthreads();
  local_pdl_complete(use_pdl);
}

// The same radix select with the register array depth DECOUPLED from the
// capacity: `Items` is a fixed cap and the row is streamed in chunks of
// `kLocalThreads * Items - kTopK` fresh elements, folding the running top-16
// back into slots 0..15 of the next chunk. Register depth, unrolled body length
// and shared footprint are then constants.
//
// Exactness is the partitioned combine's argument one level down: a chunk
// publishes only its best 16, and a chunk's 17th element cannot be in the row's
// top-16 because 16 elements of that same chunk already precede it under the
// same total order. Winners from earlier chunks cannot be lost because they are
// re-admitted in slots 0..15 of every subsequent chunk.
//
// The emit reads `keys[0]`, not `sm.ranked`: a row with no live elements at all
// never enters the loop, and `keys[0]` is initialised to 0 == C4's invalid
// entry, where `sm.ranked` would still be uninitialised shared memory.
template <int Items>
__global__ void __launch_bounds__(kLocalThreads) local_candidates_capped_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    float* __restrict__ candidates, int heads, int blocks, int partitions,
    int scan_block_begin, int global_block_stride, bool use_pdl ICP_RS_CTL_PARAM) {
  __shared__ RadixSelectSmem sm;
  constexpr int kFresh = kLocalThreads * Items - kTopK;

  local_pdl_wait(use_pdl);
  const int row = blockIdx.x / partitions;
  const int partition = blockIdx.x % partitions;
  const int token = row / heads;
  if (ICP_RS_CTL(1) && partitions > 1 && partition == partitions - 1) {
    local_pdl_complete(use_pdl);
    return;
  }
  if (threadIdx.x == 0 && active[token]) {
    assert(nvalid[token] >= 0 && nvalid[token] <= blocks);
    assert(forced_col[token] < 0 ||
           forced_col[token] < min(nvalid[token], blocks));
  }
  const int valid = active[token] ? max(0, min(nvalid[token], blocks)) : 0;
  const int excluded =
      ICP_RS_CTL(2) ? valid - 1 : (active[token] ? forced_col[token] : -1);
  const int start = partition * kLocalPartition;
  const int end = min(valid, start + kLocalPartition);

  uint64_t keys[Items];
#pragma unroll
  for (int i = 0; i < Items; ++i) keys[i] = 0ull;
  bool valid_scores = true;

  for (int base = start; base < end; base += kFresh) {
    // Slots 0..15 carry the previous chunk's winners; 16.. take fresh elements.
    // The last chunk is short, so bound the select the way the single-shot
    // kernel does.
    const int span = end - base + kTopK;
    const int live_items =
        min(Items, (span + kLocalThreads - 1) / kLocalThreads);
#pragma unroll
    for (int i = 0; i < Items; ++i) {
      const int slot = threadIdx.x + i * kLocalThreads;
      if (slot >= kTopK) {
        const int local = base + slot - kTopK;
        // refined C3 exclusion, every rank, no +inf.
        const bool live =
            local < (ICP_RS_CTL(7) ? end + 1 : end) && local != excluded;
        // refined C1 absolute id; control 3 drops the scan offset.
        const int32_t gid =
            live ? (ICP_RS_CTL(3) ? local : scan_block_begin + global_block_stride * local) : -1;
        const float raw_score =
            live ? scores[static_cast<int64_t>(row) * blocks + local]
                 : -CUDART_INF_F;
        valid_scores = valid_scores && (!live || !isnan(raw_score));
        keys[i] = canonical_key(raw_score, gid);
      }
    }
    radix_select_top16<Items>(keys, live_items, -1, sm ICP_RS_CTL_ARG);
    if (threadIdx.x < kTopK) keys[0] = sm.ranked[threadIdx.x];
    __syncthreads();
  }
  assert(valid_scores);

  // Control 8: skip ONE live producer write. See local_candidates.cuh.
  if (threadIdx.x < kTopK && !(ICP_RS_CTL(8) && threadIdx.x == kTopK - 1)) {
    store_local_candidate_keyonly(
        candidates + static_cast<int64_t>(blockIdx.x) * kTopK * 2, threadIdx.x,
        keys[0]);
  }
  __syncthreads();
  local_pdl_complete(use_pdl);
}

// Same streaming structure and same exactness argument as
// `combine_local_candidates_kernel`: the previous 16 winners are re-admitted in
// slots 0..15 of every window and the remaining slots take fresh partials.
template <int Items>
__global__ void __launch_bounds__(kLocalThreads)
    combine_local_candidates_radix_kernel(const float* __restrict__ partials,
                                          float* __restrict__ out,
                                          int partitions,
                                          bool use_pdl ICP_RS_CTL_PARAM) {
  __shared__ RadixSelectSmem sm;
  constexpr int kCapacity = kLocalThreads * Items;
  constexpr int kFresh = kCapacity - kTopK;
  const int64_t count = static_cast<int64_t>(partitions) * kTopK -
                        (ICP_RS_CTL(4) ? kTopK : 0);
  const float* input = partials + static_cast<int64_t>(blockIdx.x) * count * 2;
  uint64_t keys[Items];
#pragma unroll
  for (int i = 0; i < Items; ++i) keys[i] = 0ull;

  local_pdl_wait(use_pdl);
  for (int64_t begin = 0; begin < count; begin += kFresh) {
#pragma unroll
    for (int i = 0; i < Items; ++i) {
      const int slot = threadIdx.x + i * kLocalThreads;
      if (slot >= kTopK) {
        const int64_t index = begin + slot - kTopK;
        const bool live = index < count;
        const float score = live ? input[index * 2] : -CUDART_INF_F;
        const int32_t gid = live ? __float_as_int(input[index * 2 + 1]) : -1;
        keys[i] = canonical_key(score, gid);
      }
    }
    radix_select_top16<Items>(keys, Items, -1, sm ICP_RS_CTL_ARG);
    if (threadIdx.x < kTopK) keys[0] = sm.ranked[threadIdx.x];
    __syncthreads();
  }
  if (threadIdx.x < kTopK) {
    store_local_candidate_keyonly(
        out + static_cast<int64_t>(blockIdx.x) * kTopK * 2, threadIdx.x,
        keys[0]);
  }
  __syncthreads();
  local_pdl_complete(use_pdl);
}

}  // namespace icp
