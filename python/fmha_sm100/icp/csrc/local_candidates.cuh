// Exact whole-row local Top-16 device code, shared by the prefill selector
// kernels (local_candidates.cu) and the D3 decode publisher (fused_exchange.cuh).
//
// Input  `scores` [T, H_group, N] fp32; column j is block scan_block_begin + stride * j.
// Output [T, H_group, 16, 2] fp32 (CONTRACT C4): score, int32 id bitcast; invalid (-inf, -1).
// The forced column is excluded (C3); NaN in the live prefix traps.
//
// One CTA per (token, head) row: a coarse fp16 histogram bounds a <=128-key
// superset, an exact 64-bit radix refinement handles any wider boundary, and
// the final order always compares full canonical keys. The coarse-histogram
// strategy is inspired by the Apache-2.0 FlashInfer/MSA sparse_topk_select.cuh,
// derived from NVIDIA TensorRT-LLM indexerTopK.cu (Copyright 2019-2026 NVIDIA
// CORPORATION; 2021 NAVER Corp.; 2024-2026 FlashInfer team).
#pragma once

#include <cassert>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "merge_topk.cuh"

namespace icp {

constexpr int kFullRowThreads = 512;        // decode CTA
constexpr int kFullRowCachedItems = 3;      // decode's per-thread score cache
constexpr int kPrefillThreads = 128;        // prefill CTA
constexpr int kPrefillPrefetchItems = 4;    // prefill's streaming tile
constexpr int kMaxRowBlocks = 8192;         // int32 in-row iteration bound
constexpr int kTinyRowBlocks = 32;          // one warp per row at or below this
constexpr int kFourWarpMaxBlocks = 128;     // four-warp finish at or below this
constexpr int kFullRowCoarseBits = 10;
constexpr int kFullRowCoarseBins = 1 << kFullRowCoarseBits;
constexpr int kFullRowFinishCapacity = 128;
constexpr int kFullRowHistogramStorage =
    kFullRowCoarseBins + kFullRowCoarseBins / 32;

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

// Inverse of sortable_bits up to the C5 -0.0 flush.
__device__ __forceinline__ float score_from_sortable(uint32_t s) {
  const uint32_t bits = (s & 0x80000000u) ? (s ^ 0x80000000u) : ~s;
  return __uint_as_float(bits);
}

// Key 0 is C4's invalid record; the id is bitcast because ids above 2^24 do not survive fp32.
__device__ __forceinline__ void store_local_candidate_keyonly(float* out,
                                                              int slot,
                                                              uint64_t key) {
  const int32_t gid =
      key ? static_cast<int32_t>(~static_cast<uint32_t>(key)) : -1;
  out[slot * 2] =
      key ? score_from_sortable(static_cast<uint32_t>(key >> 32)) : -CUDART_INF_F;
  out[slot * 2 + 1] = __int_as_float(gid);
}

struct LocalCandidateSink {
  float* out;
  __device__ __forceinline__ void store(int slot, uint64_t key) const {
    store_local_candidate_keyonly(out, slot, key);
  }
};

// One padding word per 32 bins spreads the threshold warp's chunks across banks.
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

// Monotone but not injective; the exact finish resolves every collision.
__device__ __forceinline__ int full_row_coarse_bin(float score) {
  if (score == 0.0f) score = 0.0f;
  const uint16_t raw = __half_as_ushort(__float2half_rn(score));
  const uint16_t ordered =
      (raw & 0x8000u) ? static_cast<uint16_t>(~raw)
                      : static_cast<uint16_t>(raw ^ 0x8000u);
  return ordered >> (16 - kFullRowCoarseBits);
}

// All lanes must call (match_any's full mask); one atomic per distinct bin per warp.
__device__ __forceinline__ void full_row_histogram_add(int digit, int* hist) {
  const int lane = threadIdx.x & 31;
  const unsigned peers =
      __match_any_sync(0xffffffffu, static_cast<unsigned>(digit));
  if (digit >= 0 && lane == __ffs(peers) - 1) {
    atomicAdd(hist + full_row_histogram_index(digit), __popc(peers));
  }
}

// The `wanted`-th largest coarse bin, by a block-wide scan over Threads threads.
template <int Threads>
__device__ __forceinline__ void full_row_coarse_threshold(int wanted,
                                                          FullRowSelectSmem& sm) {
  constexpr int kBinsPerThread = kFullRowCoarseBins / Threads;
  constexpr int kWarps = Threads / 32;
  static_assert(kFullRowCoarseBins % Threads == 0);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int base = (Threads - 1 - tid) * kBinsPerThread;
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
}

// The `wanted`-th largest of 256 digit bins, scanned by warp 0.
__device__ __forceinline__ void full_row_digit_threshold(int wanted,
                                                         FullRowSelectSmem& sm) {
  constexpr int kPerLane = 256 / 32;
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

// Warp-collective bounded compaction; the threshold proves the total is <=128.
__device__ __forceinline__ void full_row_append_key(
    uint64_t key, bool keep, FullRowSelectSmem& sm) {
  const int lane = threadIdx.x & 31;
  const unsigned mask = __ballot_sync(0xffffffffu, keep);
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

// Ranks the compacted keys and writes all 16 slots. Zero padding sorts below
// every valid key, including -inf; distinct ids make every real key unique.
template <class Sink>
__device__ __forceinline__ void full_row_finish_to(
    const Sink& sink, int wanted, FullRowSelectSmem& sm) {
  const int tid = threadIdx.x;
  const int count = sm.candidate_count;
  assert(count >= wanted && count <= kFullRowFinishCapacity);
  if (count <= 32) {
    if (tid < 32) {
      const uint64_t key =
          warp_sort_desc(tid < count ? sm.candidate_keys[tid] : 0ull, tid);
      if (tid < kTopK) {
        sink.store(tid, tid < wanted ? key : 0ull);
      }
    }
    __syncthreads();
    return;
  }
  if (count <= 64) {
    // Two sorted warps; a key's rank is its lane plus the larger keys in the other warp.
    uint64_t key = 0ull;
    if (tid < 64) {
      const int lane = tid & 31;
      key = warp_sort_desc(tid < count ? sm.candidate_keys[tid] : 0ull, lane);
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
      // Staging position breaks ties, so this is a total order even on equal keys.
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

// Prefill's finish; FourWarpFinish ranks 65--128 keys as four sorted warps (a
// key's rank is its lane plus the larger keys in the other three) instead of
// the quadratic rank. Needs four complete warps.
template <bool FourWarpFinish>
__device__ __forceinline__ void prefill_full_row_finish(
    float* out, int wanted, FullRowSelectSmem& sm) {
  if constexpr (!FourWarpFinish) {
    full_row_finish_to(LocalCandidateSink{out}, wanted, sm);
  } else {
    const int count = sm.candidate_count;
    assert(count >= wanted && count <= kFullRowFinishCapacity);
    if (count <= 64) {
      full_row_finish_to(LocalCandidateSink{out}, wanted, sm);
      return;
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    uint64_t key = 0ull;
    if (tid < 128) {
      key = warp_sort_desc(tid < count ? sm.candidate_keys[tid] : 0ull, lane);
      sm.candidate_keys[tid] = key;
    }
    __syncthreads();
    if (tid < 128 && key != 0ull) {
      int rank = lane;
#pragma unroll
      for (int other_warp = 0; other_warp < 4; ++other_warp) {
        if (other_warp != warp) {
          int first = 0;
          int last = 32;
          while (first < last) {
            const int middle = (first + last) >> 1;
            if (sm.candidate_keys[other_warp * 32 + middle] > key) {
              first = middle + 1;
            } else {
              last = middle;
            }
          }
          rank += first;
        }
      }
      if (rank < wanted) store_local_candidate_keyonly(out, rank, key);
    }
    if (tid >= wanted && tid < kTopK) {
      store_local_candidate_keyonly(out, tid, 0ull);
    }
    __syncthreads();
  }
}

struct FullRowExtent {
  int valid;
  int excluded;
  int ordinary;
  int wanted;
};

// The row's live prefix and C3 exclusion; bounds and exclusion precede every
// score load because padding and the forced score may hold NaN.
template <int Threads>
__device__ __forceinline__ FullRowExtent full_row_extent(
    const int32_t* __restrict__ nvalid, const int32_t* __restrict__ forced_col,
    const bool* __restrict__ active, int token, int blocks) {
  const int tid = threadIdx.x;
  const bool row_active = active[token];
  assert(blockDim.x == Threads);
  assert(blocks >= 0 && blocks <= kMaxRowBlocks);
  if (tid == 0 && row_active) {
    assert(nvalid[token] >= 0 && nvalid[token] <= blocks);
    assert(forced_col[token] < 0 || forced_col[token] < nvalid[token]);
  }
  const int valid = row_active ? max(0, min(nvalid[token], blocks)) : 0;
  const int excluded = row_active ? forced_col[token] : -1;
  const int ordinary = valid - (excluded >= 0 ? 1 : 0);
  const int wanted = min(kTopK, ordinary);
  return {valid, excluded, ordinary, wanted};
}

// Exact fallback for a coarse boundary wider than 128 keys: refine the full
// 64-bit key MSB-first until the boundary fits, and return the winning prefix.
template <int Threads>
__device__ __forceinline__ uint64_t full_row_exact_prefix(
    const float* __restrict__ row_scores, int valid, int excluded, int wanted,
    int scan_block_begin, int global_block_stride, FullRowSelectSmem& sm) {
  const int tid = threadIdx.x;
  uint64_t final_prefix = 0ull;
  uint64_t prefix = 0ull;
  uint64_t mask = 0ull;
  int need = wanted;
  int selected = 0;
  for (int shift = 56; shift >= 0; shift -= 8) {
    __syncthreads();
    for (int i = tid; i < 256; i += Threads) {
      sm.histogram[full_row_histogram_index(i)] = 0;
    }
    __syncthreads();
    for (int base = 0; base < valid; base += Threads) {
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
    full_row_digit_threshold(need, sm);
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
    // Unique affine ids make the shift-0 bin a single key, so it always breaks.
    assert(shift > 0);
  }
  return final_prefix;
}

// Decode: one 512-thread CTA per row, emitted through `sink`. Every return path
// has written all 16 slots and passed a block barrier; PDL is the caller's.
template <class Sink>
__device__ __forceinline__ void full_row_select_row(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    int row, int heads, int blocks, int scan_block_begin,
    int global_block_stride, const Sink& sink, FullRowSelectSmem& sm) {
  const int token = row / heads;
  const int tid = threadIdx.x;
  const auto [valid, excluded, ordinary, wanted] =
      full_row_extent<kFullRowThreads>(nvalid, forced_col, active, token, blocks);
  const float* row_scores = scores + static_cast<int64_t>(row) * blocks;

  if (wanted <= 0) {
    if (tid < kTopK) sink.store(tid, 0ull);
    __syncthreads();
    return;
  }

  if (tid == 0) sm.candidate_count = 0;

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

  for (int i = tid; i < kFullRowCoarseBins; i += kFullRowThreads) {
    sm.histogram[full_row_histogram_index(i)] = 0;
  }
  __syncthreads();
  bool valid_scores = true;
  // Rows up to 1536 live columns keep scores and bins in registers across the threshold.
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
  full_row_coarse_threshold<kFullRowThreads>(wanted, sm);
  const int coarse_threshold = sm.threshold;
  const bool coarse_finish =
      sm.above + sm.boundary_count <= kFullRowFinishCapacity;
  const uint64_t final_prefix =
      coarse_finish ? 0ull
                    : full_row_exact_prefix<kFullRowThreads>(
                          row_scores, valid, excluded, wanted, scan_block_begin,
                          global_block_stride, sm);

  if (tid == 0) sm.candidate_count = 0;
  __syncthreads();
  if (cache_row && coarse_finish) {
#pragma unroll
    for (int item = 0; item < kFullRowCachedItems; ++item) {
      if (item * kFullRowThreads < valid) {
        // Masked loads left bin -1, below every valid coarse bin.
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

}  // namespace icp
