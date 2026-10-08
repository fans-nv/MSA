// The selector's SORT arm: dense block scores -> this rank's local Top-16.
//
// Input  `scores`     [T, H_group, N] fp32, this rank's block scores, with
//                     N = the scan window's width in LOGICAL blocks.
// Output `candidates` [T, H_group, 16, 2] fp32 (CONTRACT C4): [..., 0] the
//                     score, [..., 1] the int32 GLOBAL block id bitcast to
//                     fp32. Invalid entries are (-inf, -1).
//
// One CTA of 128 threads per (row, partition), where a row is a (token, head)
// pair and a partition is at most kLocalPartition consecutive blocks. A
// partitioned launch writes per-partition partials that
// `combine_local_candidates_kernel` folds into the final 16.
//
// `Items` is the compile-time register depth, so one CTA covers 128 * Items
// blocks; the host ladder picks it. The key is `merge_topk.cuh::canonical_key`,
// the same one K2 and K5 merge with, so a candidate produced here and a
// candidate merged there are ordered identically.
//
// This is the older of the two arms; `local_candidates_rs.cuh` carries the one
// that ships. Both are changed together and must stay semantically identical,
// because the dispatch gate compares them.
//
// refined-icp-v1. Column `j` of a score row names the ABSOLUTE
// logical block `scan_block_begin + j`, not an M0 block-cyclic local ordinal
// `b//C`. Under refined C1 every rank holds a fragment of every logical block,
// so the score row is W times wider than it was under M0 -- `ceil(N_tok/128)`
// columns, not `ceil(N_tok/128/W)`. At a 262144-token context that is 2048
// columns, and at 1M it is 8192. The `Items` ladder in `local_candidates.cu`
// is indexed by that width and was re-derived against it at its definition
// site; `docs/PERFORMANCE.md` section 7.1 carries the register/spill evidence.
//
// KNOWN, AND OUT OF SCOPE HERE (recorded, not solved, not measured): this kernel
// costs O(capacity), not O(valid), and the refined design multiplies capacity
// by W. That is a design-level performance problem for the lane that owns the
// score-wave seam, not a correctness defect here.
#pragma once

#include <cassert>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>

#include "merge_topk.cuh"

// ---------------------------------------------------------------------------
// Negative-control hooks, shared by BOTH selector arms. COMPILED OUT of the
// shipping extension: without -DICP_RS_NEGATIVE_CONTROL, `ICP_RS_CTL(n)` is the
// literal `false`, the extra kernel parameter does not exist, and ptxas reports
// byte-identical register/spill/shared numbers -- which the gate checks.
//
// These lived in `local_candidates_rs.cuh` and covered the radix arm only, so
// the sort arm had no controls at all and its gate was vacuous by construction.
// refined-icp-v1 hoists them here, into the header the radix arm already
// includes, so ONE definition serves both and the two arms are gated alike.
// The #ifndef makes the hoist compatible with the
// `local_candidates_rs_control.cu` build, which defines nothing itself.
//
// Controls 1-6 are the radix arm's (documented at their sites in
// `local_candidates_rs.cuh`). Control 7 is new and applies to BOTH arms:
//
//   7  the score load's live bound is `local < valid + 1`  -> a consumer that
//      reads ONE column past `valid`. It is observable only when the invalid
//      tail is POISONED with a dominant finite value first; against an
//      -inf-filled or zero-filled tail it is silently harmless, which is why
//      the old blanket score-buffer fill made this whole class of bug
//      invisible. `docs/CONTRACT.md` section 5 states that discipline.
// ---------------------------------------------------------------------------
#ifndef ICP_RS_CTL
#ifdef ICP_RS_NEGATIVE_CONTROL
#define ICP_RS_CTL_PARAM , int control
#define ICP_RS_CTL_ARG , control
#define ICP_RS_CTL(n) (control == (n))
#else
#define ICP_RS_CTL_PARAM
#define ICP_RS_CTL_ARG
#define ICP_RS_CTL(n) false
#endif
#endif

namespace icp {

constexpr int kLocalThreads = 128;
constexpr int kLocalPartition = 4096;

// A CTA that owns a whole partition must span one, so the partitioned rung is
// this and nothing else.
constexpr int kPartitionedItems = kLocalPartition / kLocalThreads;

// Distinct from `int blocks` so that sizing a row stride with it is a compile
// error rather than a silently corrupt read.
struct LiveExtent {
  int64_t value;
};

// The eager prefill path's whole launch geometry. Both the partition count and
// the rung come from the live extent; capacity is only ever the row stride.
struct PrefillLaunch {
  int64_t partitions;  // 0 when one CTA covers the row
  int items;
};

__host__ inline PrefillLaunch prefill_launch(LiveExtent live) {
  if (live.value > kLocalPartition) {
    return PrefillLaunch{
        (live.value + kLocalPartition - 1) / kLocalPartition,
        kPartitionedItems};
  }
  for (int items = 1; items < kPartitionedItems; items *= 2) {
    if (live.value <= static_cast<int64_t>(kLocalThreads) * items) {
      return PrefillLaunch{0, items};
    }
  }
  return PrefillLaunch{0, kPartitionedItems};
}

// Instantiating a partitioned launch at a shorter rung drops the tail of every
// partition with no other symptom, so make it a compile error.
template <int Items>
struct PartitionedRung {
  static_assert(kLocalThreads * Items >= kLocalPartition,
                "a partitioned CTA must span kLocalPartition blocks");
  static constexpr int items = Items;
};

__device__ __forceinline__ void local_pdl_wait(bool use_pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (use_pdl) cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void local_pdl_complete(bool use_pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (use_pdl) cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ void store_local_candidate(float* out, int slot,
                                                      uint64_t key,
                                                      float score) {
  // Key 0 is C4's invalid entry. The id is bitcast, never converted: an id
  // above 2^24 would not survive a float conversion.
  const int32_t gid =
      key ? static_cast<int32_t>(~static_cast<uint32_t>(key)) : -1;
  out[slot * 2] = key ? score : -CUDART_INF_F;
  out[slot * 2 + 1] = __int_as_float(gid);
}

// Each CTA owns a disjoint <=4096-block partition. CUB retains the original
// score as a value: recovering it from the canonical key would lose -0 bits.
// Input arrangement does not matter; striped output puts the winners in lanes
// 0..15. The total order is exactly the one used by the K2/K5 global merge.
// ---------------------------------------------------------------------------
// refined-icp-v1. THREE semantic changes against the M0 kernel. The governing
// text is the refined-icp-v1 amendment in
// `indexer-DCP/docs/history/CONTRACT.md`; `docs/CONTRACT.md` section 10 states
// the same three where they bite.
//
// 1. GLOBAL ID. Was `local*world + rank` (C1 block-cyclic ownership, one owner
//    per block). refined C1 deletes block-cyclic ownership: every rank holds a
//    fragment of EVERY logical block, so the column index is an offset into an
//    absolute scan window and the id is `scan_block_begin + global_block_stride * local`. `world` and
//    `rank` no longer participate in id formation and are gone from the
//    signature -- deliberately, so a caller still passing them fails to compile
//    rather than silently producing M0 ids.
//
// 2. FORCING. Was `live && owner && local == diagonal ? +inf : raw`. refined C3
//    removes +inf forcing outright: a valid ordinary +inf with a SMALLER global
//    id outranks the forced block under C5 (score desc, then id asc) and
//    displaces it, which is a wrong selection, not a tie. The forced block is
//    now EXCLUDED from ordinary ranking on every rank (not just an owner); the
//    receiver reserves its final output slot and injects it exactly once after
//    max-by-id duplicate reduction. An excluded column emits key 0, i.e. C4's
//    invalid entry (-inf,-1) -- the same thing a dead lane emits.
//
// 3. owns[]/diag[]. refined C1 makes `diagonal_owned` unconditionally true, so
//    `owns` is DELETED rather than pinned to 1. `diag` (an M0 local ordinal
//    `b//C`) becomes `forced_col`, the column of block `p//128` inside THIS
//    scan window, i.e. `f - scan_block_begin`, or any negative value when the
//    forced block lies outside the window. Both the parameter and the
//    duck-typed vLLM attribute are RENAMED (`icp_diagonal_local` ->
//    `icp_forced_column_v1`, `icp_owns_diagonal` -> removed) because the
//    binding is duck-typed: keeping the old names with new meanings would bind
//    silently and mis-select. AttributeError is the intended failure mode.
// ---------------------------------------------------------------------------
template <int Items>
__global__ void local_candidates_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    float* __restrict__ candidates, int heads, int blocks, int partitions,
    int scan_block_begin, int global_block_stride, bool use_pdl ICP_RS_CTL_PARAM) {
  using Sort = cub::BlockRadixSort<uint64_t, kLocalThreads, Items, float>;
  __shared__ typename Sort::TempStorage storage;

  local_pdl_wait(use_pdl);
  const int row = blockIdx.x / partitions;
  const int partition = blockIdx.x % partitions;
  const int token = row / heads;
  // Control 1, as in the radix arm: skip the last partition, leaving STALE
  // partials. Added for refined-icp-v1 so the sort arm is gated like the radix
  // arm rather than not at all; it only fires against poisoned scratch.
  if (ICP_RS_CTL(1) && partitions > 1 && partition == partitions - 1) {
    local_pdl_complete(use_pdl);
    return;
  }
  if (threadIdx.x == 0 && active[token]) {
    assert(nvalid[token] >= 0 && nvalid[token] <= blocks);
    // refined C3: a NEGATIVE forced column is legal and means "the forced block
    // is not in this scan window" (bounded waves). A non-negative one must
    // name a live column of this row.
    assert(forced_col[token] < 0 ||
           forced_col[token] < min(nvalid[token], blocks));
  }
  // Clamped so an invalid row cannot read out of bounds while its device
  // assertion is still being reported.
  const int valid = active[token] ? max(0, min(nvalid[token], blocks)) : 0;
  // Unconditional on every rank: refined C1 has no owner, so there is no
  // `owner &&` guard left to get wrong. -1 disables exclusion for this row.
  const int excluded = active[token] ? forced_col[token] : -1;
  const int start = partition * kLocalPartition;
  // A rung that does not span its partition truncates the row silently, and so
  // does a partition COUNT that does not span the row -- the second is not
  // implied by the first. Both are device assertions for the reason §10 of
  // `docs/CONTRACT.md` gives for the rest of the preconditions: checking them
  // on the host means reading device values, which is a submission-path
  // synchronisation on the eager band and uncapturable on the other. The radix
  // arm carries the same pair; the two arms are changed together.
  if (threadIdx.x == 0) {
    assert(min(valid - start, kLocalPartition) <= kLocalThreads * Items);
    assert(static_cast<int64_t>(valid) <=
           static_cast<int64_t>(partitions) * kLocalPartition);
  }
  uint64_t keys[Items];
  float values[Items];
  bool valid_scores = true;
#pragma unroll
  for (int i = 0; i < Items; ++i) {
    const int local = start + threadIdx.x + i * kLocalThreads;
    // refined C3: the forced column takes no part in ORDINARY ranking, on every
    // rank. Dropping it here (rather than scoring it +inf) is what keeps a
    // valid ordinary +inf from displacing it downstream.
    const bool live =
        local < (ICP_RS_CTL(7) ? valid + 1 : valid) && local != excluded;
    // refined C1: absolute logical block id, no world/rank term.
    const int32_t gid = live ? scan_block_begin + global_block_stride * local : -1;
    const float raw_score =
        live ? scores[static_cast<int64_t>(row) * blocks + local]
             : -CUDART_INF_F;
    // NaN rejection still reads the RAW cell of every live ordinary column. The
    // excluded column is not read at all: its score is not an output under C3
    // and the receiver never needs it.
    valid_scores = valid_scores && (!live || !isnan(raw_score));
    keys[i] = canonical_key(raw_score, gid);
    values[i] = raw_score;
  }
  assert(valid_scores);
  Sort(storage).SortDescendingBlockedToStriped(keys, values);
  // Control 8 skips ONE live producer write (the last winner slot). Like
  // control 1 it is invisible unless the output was poisoned first: a skipped
  // store leaves whatever the previous launch wrote, which on a warm buffer is
  // usually the CORRECT value. That is the no-fill contract's whole point.
  if (threadIdx.x < kTopK && !(ICP_RS_CTL(8) && threadIdx.x == kTopK - 1)) {
    store_local_candidate(
        candidates + static_cast<int64_t>(blockIdx.x) * kTopK * 2, threadIdx.x,
        keys[0], values[0]);
  }
  __syncthreads();
  local_pdl_complete(use_pdl);
}

// Repeated bounded reductions keep the scratch bound independent of row length:
// the previous 16 winners occupy the first 16 slots of each iteration and the
// remaining slots admit fresh partials. Discarding a partial loser is exact
// because 16 distinct candidates already precede it under the same total order.
template <int Items>
__global__ void combine_local_candidates_kernel(
    const float* __restrict__ partials, float* __restrict__ out, int partitions,
    bool use_pdl ICP_RS_CTL_PARAM) {
  using Sort = cub::BlockRadixSort<uint64_t, kLocalThreads, Items, float>;
  __shared__ typename Sort::TempStorage storage;
  constexpr int kCapacity = kLocalThreads * Items;
  constexpr int kFresh = kCapacity - kTopK;
  // Control 4 (as in the radix arm): read count-16, i.e. drop one partition's
  // worth of partials. Uniform across both arms so the gate is not vacuous on
  // the sort side.
  const int64_t count = static_cast<int64_t>(partitions) * kTopK -
                        (ICP_RS_CTL(4) ? kTopK : 0);
  const float* input = partials + static_cast<int64_t>(blockIdx.x) *
                                      static_cast<int64_t>(partitions) * kTopK * 2;
  uint64_t keys[Items];
  float values[Items];
  keys[0] = 0;
  values[0] = -CUDART_INF_F;

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
        values[i] = score;
      }
    }
    Sort(storage).SortDescendingBlockedToStriped(keys, values);
    // CUB requires a block barrier before reusing the sort's temporary storage.
    __syncthreads();
  }
  if (threadIdx.x < kTopK) {
    store_local_candidate(out + static_cast<int64_t>(blockIdx.x) * kTopK * 2,
                          threadIdx.x, keys[0], values[0]);
  }
  __syncthreads();
  local_pdl_complete(use_pdl);
}

}  // namespace icp
