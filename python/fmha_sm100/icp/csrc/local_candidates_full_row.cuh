// Exact whole-row Top-16 for the refined ICP candidate contract.
//
// The initial coarse-histogram / small-boundary strategy is inspired by the
// Apache-2.0 FlashInfer/MSA sparse_topk_select.cuh implementation, whose
// histogram algorithm is derived from NVIDIA TensorRT-LLM indexerTopK.cu
// (Copyright 2019-2026 NVIDIA CORPORATION; 2021 NAVER Corp.; 2024-2026
// FlashInfer team). This implementation retains ICP's complete canonical key
// for every final comparison; half precision is ONLY a monotone partition.
//
// One fixed CTA per (token, head), independent of live counts. Device-side
// loops cover the entire live prefix, so capacity padding needs no fill and
// a capacity of 8192 does not launch an empty second partition or a combine.
// There is no compile-time register array proportional to row capacity.
#pragma once

#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "local_candidates_rs.cuh"

namespace icp {

constexpr int kFullRowThreads = 512;
constexpr int kFullRowCachedItems = 3;
constexpr int kFullRowCoarseBits = 10;
constexpr int kFullRowCoarseBins = 1 << kFullRowCoarseBits;
constexpr int kFullRowFinishCapacity = 128;
constexpr int kFullRowHistogramStorage =
    kFullRowCoarseBins + kFullRowCoarseBins / 32;

// One padding word per 32 logical bins makes the threshold warp's per-lane
// 32-bin (coarse) or 8-bin (exact) chunks visit different shared-memory banks.
__device__ __forceinline__ int full_row_histogram_index(int bin) {
  return bin + (bin >> 5);
}

struct FullRowSelectSmem {
  int histogram[kFullRowHistogramStorage];
  int threshold;
  int above;
  int boundary_count;
  int candidate_count;
  int warp_prefix[kFullRowThreads / 32];
  uint64_t candidate_keys[kFullRowFinishCapacity];
};

// Monotone, not injective: overflow, underflow and ordinary rounding can put
// different FP32 scores in the same bin. The exact finish/fallback resolves
// ALL such collisions, including canonical +/-0 and ties at +/-infinity.
__device__ __forceinline__ int full_row_coarse_bin(float score) {
  if (score == 0.0f) score = 0.0f;
  const uint16_t raw = __half_as_ushort(__float2half_rn(score));
  const uint16_t ordered =
      (raw & 0x8000u) ? static_cast<uint16_t>(~raw)
                      : static_cast<uint16_t>(raw ^ 0x8000u);
  return ordered >> (16 - kFullRowCoarseBits);
}

// All lanes call this, including masked lanes, so match_any's full mask is
// valid. One shared atomic per distinct bin per warp, rather than one per
// element, matters for the common score-exponent and all-equal cases.
__device__ __forceinline__ void full_row_histogram_add(int digit, int* hist) {
  const int lane = threadIdx.x & 31;
  const unsigned peers =
      __match_any_sync(0xffffffffu, static_cast<unsigned>(digit));
  if (digit >= 0 && lane == __ffs(peers) - 1) {
    atomicAdd(hist + full_row_histogram_index(digit), __popc(peers));
  }
}

template <int Bins>
__device__ __forceinline__ void full_row_find_threshold(
    int wanted, FullRowSelectSmem& sm) {
  static_assert(Bins % 32 == 0, "histogram is distributed across one warp");
  if constexpr (Bins == kFullRowCoarseBins) {
    // The common coarse pass uses every warp: two bins per thread rather
    // than 32 serial bins in a single warp. A pair of warp scans forms the
    // block prefix without an expensive full-histogram transpose.
    constexpr int kBinsPerThread = Bins / kFullRowThreads;
    constexpr int kWarps = kFullRowThreads / 32;
    static_assert(Bins % kFullRowThreads == 0);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int base = (kFullRowThreads - 1 - tid) * kBinsPerThread;
    int chunk = 0;
#pragma unroll
    for (int j = 0; j < kBinsPerThread; ++j) {
      chunk += sm.histogram[full_row_histogram_index(base + j)];
    }
    int inclusive = chunk;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) inclusive += other;
    }
    if (lane == 31) sm.warp_prefix[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
      const int warp_count = lane < kWarps ? sm.warp_prefix[lane] : 0;
      int prefix = warp_count;
#pragma unroll
      for (int offset = 1; offset < kWarps; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, prefix, offset);
        if (lane >= offset) prefix += other;
      }
      if (lane < kWarps) sm.warp_prefix[lane] = prefix - warp_count;
    }
    __syncthreads();
    const int exclusive = sm.warp_prefix[warp] + inclusive - chunk;
    if (exclusive < wanted && wanted <= exclusive + chunk) {
      int accumulated = exclusive;
      int threshold = -1;
      int above = 0;
      int boundary_count = 0;
#pragma unroll
      for (int j = kBinsPerThread - 1; j >= 0; --j) {
        const int count = sm.histogram[full_row_histogram_index(base + j)];
        if (threshold < 0 && accumulated + count >= wanted) {
          threshold = base + j;
          above = accumulated;
          boundary_count = count;
        }
        accumulated += count;
      }
      sm.threshold = threshold;
      sm.above = above;
      sm.boundary_count = boundary_count;
    }
    __syncthreads();
    return;
  }
  constexpr int kPerLane = Bins / 32;
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const int base = (31 - lane) * kPerLane;
    int chunk = 0;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      chunk += sm.histogram[full_row_histogram_index(base + j)];
    }
    int inclusive = chunk;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
      if (lane >= offset) inclusive += other;
    }
    const int exclusive = inclusive - chunk;
    if (exclusive < wanted && wanted <= inclusive) {
      int accumulated = exclusive;
      int threshold = -1;
      int above = 0;
      int boundary_count = 0;
#pragma unroll
      for (int j = kPerLane - 1; j >= 0; --j) {
        const int count = sm.histogram[full_row_histogram_index(base + j)];
        if (threshold < 0 && accumulated + count >= wanted) {
          threshold = base + j;
          above = accumulated;
          boundary_count = count;
        }
        accumulated += count;
      }
      sm.threshold = threshold;
      sm.above = above;
      sm.boundary_count = boundary_count;
    }
  }
  __syncthreads();
}

// Collective bounded compaction. Every warp enters once per uniform input
// chunk, and the threshold construction proves the total is <=128 before
// any stores occur. Atomic collection order never affects the exact key rank.
__device__ __forceinline__ void full_row_append_key(
    uint64_t key, bool keep, FullRowSelectSmem& sm) {
  const int lane = threadIdx.x & 31;
  const unsigned mask = __ballot_sync(0xffffffffu, keep);
  // Ballot returns the same mask to every lane. Empty warps can return
  // collectively without an atomic, shuffle or candidate-address arithmetic.
  if (mask == 0u) return;
  int base = 0;
  if (lane == 0) {
    base = atomicAdd(&sm.candidate_count, __popc(mask));
  }
  base = __shfl_sync(0xffffffffu, base, 0);
  if (keep) {
    const unsigned lower_lanes = (1u << lane) - 1u;
    const int index = base + __popc(mask & lower_lanes);
    assert(index < kFullRowFinishCapacity);
    sm.candidate_keys[index] = key;
  }
}

// Where a finished row's 16 records go. The local sink is the C4 candidate
// row; the fused exchange supplies a sink that writes the owner's window.
struct LocalCandidateSink {
  float* out;
  __device__ __forceinline__ void store(int slot, uint64_t key) const {
    store_local_candidate_keyonly(out, slot, key);
  }
};

template <class Sink>
__device__ __forceinline__ void full_row_finish_to(
    const Sink& sink, int wanted, FullRowSelectSmem& sm) {
  const int tid = threadIdx.x;
  const int count = sm.candidate_count;
  assert(count >= wanted && count <= kFullRowFinishCapacity);
  if (count <= 32) {
    // The common boundary has 16--32 candidates. Warp 0 finishes them with
    // 15 register/shuffle exchanges instead of a shared-memory rank loop;
    // zeros pad the unused lanes and sort below every valid key, including
    // ordinary -infinity. The full 64-bit score/ID key decides every exchange.
    if (tid < 32) {
      uint64_t key = tid < count ? sm.candidate_keys[tid] : 0ull;
#pragma unroll
      for (int width = 2; width <= 32; width <<= 1) {
#pragma unroll
        for (int distance = width >> 1; distance > 0; distance >>= 1) {
          const uint64_t other =
              __shfl_xor_sync(0xffffffffu, key, distance);
          const bool descending = (tid & width) == 0;
          const bool lower_lane = (tid & distance) == 0;
          if (descending == lower_lane) {
            key = key > other ? key : other;
          } else {
            key = key < other ? key : other;
          }
        }
      }
      if (tid < kTopK) {
        sink.store(tid, tid < wanted ? key : 0ull);
      }
    }
    __syncthreads();
    return;
  }
  if (count <= 64) {
    // A few rows have 33--64 boundary candidates and determine the kernel's
    // tail latency. Sort each half with its own warp, then find each key's
    // position in the other sorted half with a bounded binary search. Its
    // complete rank is the local lane plus the number of larger remote keys.
    // Distinct affine IDs make every real canonical key unique; zero padding
    // sorts last and never competes with a valid -infinity candidate.
    uint64_t key = 0ull;
    if (tid < 64) {
      const int lane = tid & 31;
      key = tid < count ? sm.candidate_keys[tid] : 0ull;
#pragma unroll
      for (int width = 2; width <= 32; width <<= 1) {
#pragma unroll
        for (int distance = width >> 1; distance > 0; distance >>= 1) {
          const uint64_t other =
              __shfl_xor_sync(0xffffffffu, key, distance);
          const bool descending = (lane & width) == 0;
          const bool lower_lane = (lane & distance) == 0;
          if (descending == lower_lane) {
            key = key > other ? key : other;
          } else {
            key = key < other ? key : other;
          }
        }
      }
      sm.candidate_keys[tid] = key;
    }
    __syncthreads();
    if (tid < 64 && key != 0ull) {
      const int lane = tid & 31;
      const int other_warp_base = ((tid >> 5) ^ 1) * 32;
      int first = 0;
      int last = 32;
      while (first < last) {
        const int middle = (first + last) >> 1;
        if (sm.candidate_keys[other_warp_base + middle] > key) {
          first = middle + 1;
        } else {
          last = middle;
        }
      }
      const int rank = lane + first;
      if (rank < wanted) sink.store(rank, key);
    }
    if (tid >= wanted && tid < kTopK) {
      sink.store(tid, 0ull);
    }
    __syncthreads();
    return;
  }
  if (tid < count) {
    const uint64_t mine = sm.candidate_keys[tid];
    int rank = 0;
    for (int j = 0; j < count; ++j) {
      // Local affine global IDs are unique, hence all nonzero keys are
      // unique. The secondary staging-position comparison makes this helper
      // a total-order rank even if it is reused with equal invalid keys.
      const uint64_t other = sm.candidate_keys[j];
      rank += (other > mine || (other == mine && j < tid));
    }
    if (rank < wanted) sink.store(rank, mine);
  }
  if (tid >= wanted && tid < kTopK) {
    sink.store(tid, 0ull);
  }
  __syncthreads();
}


__device__ __forceinline__ void full_row_finish(
    float* out, int wanted, FullRowSelectSmem& sm) {
  full_row_finish_to(LocalCandidateSink{out}, wanted, sm);
}

// One row's exact selection, emitted through `sink`. Every return path has
// written all 16 slots and passed a block barrier; PDL is the caller's.
template <class Sink>
__device__ __forceinline__ void full_row_select_row(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    int row, int heads, int blocks, int scan_block_begin,
    int global_block_stride, const Sink& sink, FullRowSelectSmem& sm) {
  const int token = row / heads;
  const int tid = threadIdx.x;
  const bool row_active = active[token];
  assert(blockDim.x == kFullRowThreads);
  // The host dispatch selects this kernel only for capacity <=8192 and sends
  // larger capacities to the generic partitioned implementation. That proof
  // permits int32 in-row iteration; only the complete tensor row offset needs
  // int64. Keep this guard if the dispatch boundary is changed in the future.
  assert(blocks >= 0 && blocks <= 8192);
  if (tid == 0 && row_active) {
    assert(nvalid[token] >= 0 && nvalid[token] <= blocks);
    assert(forced_col[token] < 0 || forced_col[token] < nvalid[token]);
  }
  const int valid = row_active ? max(0, min(nvalid[token], blocks)) : 0;
  const int excluded = row_active ? forced_col[token] : -1;
  const int ordinary = valid - (excluded >= 0 ? 1 : 0);
  const int wanted = min(kTopK, ordinary);
  const float* row_scores = scores + static_cast<int64_t>(row) * blocks;

  if (wanted <= 0) {
    if (tid < kTopK) sink.store(tid, 0ull);
    __syncthreads();
    return;
  }

  if (tid == 0) sm.candidate_count = 0;

  // Tiny rows need no histogram. Bounds and exclusion still precede every
  // score load; a forced score and all capacity padding may contain NaNs.
  if (ordinary <= kFullRowFinishCapacity) {
    __syncthreads();
    for (int base = 0; base < valid; base += kFullRowThreads) {
      const int local = base + tid;
      const bool live = local < valid && local != excluded;
      uint64_t key = 0ull;
      if (live) {
        const float score = row_scores[local];
        assert(!isnan(score));
        key = canonical_key(score, scan_block_begin + global_block_stride * local);
      }
      full_row_append_key(key, live, sm);
    }
    __syncthreads();
    full_row_finish_to(sink, wanted, sm);
    return;
  }

  // First pass: one coarse, monotone histogram over the LIVE row. The 16th
  // element's bin plus all higher bins is an exact superset of the answer.
  for (int i = tid; i < kFullRowCoarseBins; i += kFullRowThreads) {
    sm.histogram[full_row_histogram_index(i)] = 0;
  }
  __syncthreads();
  bool valid_scores = true;
  // The 150K-history profiles fit in three slots per thread. Cache this small
  // prefix across the histogram so compaction needs neither a second score
  // load nor another FP16 conversion. Larger DEVICE live counts retain the
  // streaming path within the same fixed launch; capacity never truncates a
  // row. Separate prefetch and histogram loops expose independent loads.
  const bool cache_row = valid <= kFullRowThreads * kFullRowCachedItems;
  float cached_scores[kFullRowCachedItems] = {};
  int cached_bins[kFullRowCachedItems] = {-1, -1, -1};
  if (cache_row) {
#pragma unroll
    for (int item = 0; item < kFullRowCachedItems; ++item) {
      const int local = tid + item * kFullRowThreads;
      if (local < valid && local != excluded) {
        cached_scores[item] = row_scores[local];
      }
    }
#pragma unroll
    for (int item = 0; item < kFullRowCachedItems; ++item) {
      if (item * kFullRowThreads < valid) {
        const int local = tid + item * kFullRowThreads;
        if (local < valid && local != excluded) {
          const float score = cached_scores[item];
          valid_scores = valid_scores && !isnan(score);
          cached_bins[item] = full_row_coarse_bin(score);
        }
        full_row_histogram_add(cached_bins[item], sm.histogram);
      }
    }
  } else {
    for (int base = 0; base < valid; base += kFullRowThreads) {
      const int local = base + tid;
      const bool live = local < valid && local != excluded;
      int bin = -1;
      if (live) {
        const float score = row_scores[local];
        valid_scores = valid_scores && !isnan(score);
        bin = full_row_coarse_bin(score);
      }
      full_row_histogram_add(bin, sm.histogram);
    }
  }
  assert(valid_scores);
  __syncthreads();
  full_row_find_threshold<kFullRowCoarseBins>(wanted, sm);
  const int coarse_threshold = sm.threshold;
  const bool coarse_finish =
      sm.above + sm.boundary_count <= kFullRowFinishCapacity;

  uint64_t final_prefix = 0ull;
  if (!coarse_finish) {
    // Exact fallback for arbitrarily large boundary bins (all equal scores,
    // half overflow/underflow, +/-inf, etc.). Refine the COMPLETE canonical
    // 64-bit key. Higher prefixes remain definite winners; the current bin
    // holds every unresolved candidate. No payloads or full-row register
    // arrays survive between passes.
    uint64_t prefix = 0ull;
    uint64_t mask = 0ull;
    int need = wanted;
    int selected = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
      // All threads consumed the coarse/previous threshold before reusing the
      // scalar fields and histogram for this pass.
      __syncthreads();
      for (int i = tid; i < 256; i += kFullRowThreads) {
        sm.histogram[full_row_histogram_index(i)] = 0;
      }
      __syncthreads();
      for (int base = 0; base < valid; base += kFullRowThreads) {
        const int local = base + tid;
        const bool live = local < valid && local != excluded;
        int digit = -1;
        if (live) {
          const uint64_t key = canonical_key(
              row_scores[local], scan_block_begin + global_block_stride * local);
          if ((key & mask) == prefix) digit = (key >> shift) & 255u;
        }
        full_row_histogram_add(digit, sm.histogram);
      }
      __syncthreads();
      full_row_find_threshold<256>(need, sm);
      const int above = sm.above;
      const int count = sm.boundary_count;
      selected += above;
      need -= above;
      prefix |= static_cast<uint64_t>(sm.threshold) << shift;
      mask |= 0xffull << shift;
      if (selected + count <= kFullRowFinishCapacity) {
        final_prefix = prefix;
        break;
      }
      // At shift zero the unique affine IDs make every live key distinct.
      // The final bin therefore has count one and must have taken the break.
      assert(shift > 0);
    }
  }

  // Reload only the valid prefix and compact the proven <=128-key superset.
  // Every forced or invalid location is masked before even loading its score.
  if (tid == 0) sm.candidate_count = 0;
  __syncthreads();
  if (cache_row && coarse_finish) {
#pragma unroll
    for (int item = 0; item < kFullRowCachedItems; ++item) {
      if (item * kFullRowThreads < valid) {
        // Invalid/forced slots have bin -1 because their original score load
        // was masked. Every valid coarse bin is nonnegative.
        const bool keep = cached_bins[item] >= coarse_threshold;
        uint64_t key = 0ull;
        if (keep) {
          const int local = tid + item * kFullRowThreads;
          key = canonical_key(cached_scores[item],
                              scan_block_begin + global_block_stride * local);
        }
        full_row_append_key(key, keep, sm);
      }
    }
  } else {
    for (int base = 0; base < valid; base += kFullRowThreads) {
      const int local = base + tid;
      const bool live = local < valid && local != excluded;
      uint64_t key = 0ull;
      bool keep = false;
      if (live) {
        const float score = row_scores[local];
        if (coarse_finish) {
          keep = full_row_coarse_bin(score) >= coarse_threshold;
          // Most live scores are already excluded by the coarse threshold.
          // Form the exact score/ID key only for the retained boundary set.
          if (keep) {
            key = canonical_key(score,
                                scan_block_begin + global_block_stride * local);
          }
        } else {
          key = canonical_key(score,
                              scan_block_begin + global_block_stride * local);
          keep = key >= final_prefix;
        }
      }
      full_row_append_key(key, keep, sm);
    }
  }
  __syncthreads();
  full_row_finish_to(sink, wanted, sm);
}

__global__ void __launch_bounds__(kFullRowThreads) local_candidates_full_row_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    float* __restrict__ out, int heads, int blocks, int scan_block_begin,
    int global_block_stride, bool use_pdl) {
  __shared__ FullRowSelectSmem sm;
  local_pdl_wait(use_pdl);
  const int row = blockIdx.x;
  full_row_select_row(scores, nvalid, forced_col, active, row, heads, blocks,
                      scan_block_begin, global_block_stride,
                      LocalCandidateSink{out + static_cast<int64_t>(row) * kTopK * 2},
                      sm);
  local_pdl_complete(use_pdl);
}

}  // namespace icp
