// Prefill-specific CTA specialization of the exact full-row selector.
//
// Keep the qualified decode source unchanged while measuring occupancy and
// throughput at thousands of query rows. Canonical keys, bounded compaction,
// exact 64-bit fallback and final ordering remain the shared implementation.
// Only CTA ownership of the scan and the cooperative coarse histogram varies.
#pragma once

#include "local_candidates_full_row.cuh"

namespace icp {

template <int Threads>
__device__ __forceinline__ void prefill_full_row_coarse_threshold(
    int wanted, FullRowSelectSmem& sm) {
  static_assert(Threads == 128 || Threads == 256);
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

// Four independently sorted warps cover the 65--128-candidate boundary. Each
// key's exact global rank is its local lane plus the number of larger keys in
// the other three sorted warps. This replaces the quadratic shared-key rank
// only on the explicit prefill experiment; <=64 keeps the qualified finish.
template <bool FourWarpFinish>
__device__ __forceinline__ void prefill_full_row_finish(
    float* out, int wanted, FullRowSelectSmem& sm) {
  if constexpr (!FourWarpFinish) {
    full_row_finish(out, wanted, sm);
  } else {
    const int count = sm.candidate_count;
    assert(count >= wanted && count <= kFullRowFinishCapacity);
    if (count <= 64) {
      full_row_finish(out, wanted, sm);
      return;
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    uint64_t key = 0ull;
    if (tid < 128) {
      // The predicate admits four complete warps, so every shuffle's full
      // mask is valid. Zero padding sorts below valid -infinity keys.
      key = tid < count ? sm.candidate_keys[tid] : 0ull;
#pragma unroll
      for (int width = 2; width <= 32; width <<= 1) {
#pragma unroll
        for (int distance = width >> 1; distance > 0; distance >>= 1) {
          const uint64_t other = __shfl_xor_sync(0xffffffffu, key, distance);
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

template <int Threads, int CachedItems, bool FourWarpFinish = false>
__global__ void __launch_bounds__(Threads) prefill_candidates_full_row_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    float* __restrict__ out, int heads, int blocks, int scan_block_begin,
    int global_block_stride, bool use_pdl) {
  static_assert(Threads == 128 || Threads == 256);
  static_assert(CachedItems >= 0 && CachedItems * Threads <= 1536);
  __shared__ FullRowSelectSmem sm;
  local_pdl_wait(use_pdl);
  const int row = blockIdx.x;
  const int token = row / heads;
  const int tid = threadIdx.x;
  const bool row_active = active[token];
  assert(blockDim.x == Threads);
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
  float* row_out = out + static_cast<int64_t>(row) * kTopK * 2;

  if (wanted <= 0) {
    if (tid < kTopK) store_local_candidate_keyonly(row_out, tid, 0ull);
    __syncthreads();
    local_pdl_complete(use_pdl);
    return;
  }

  if (tid == 0) sm.candidate_count = 0;

  // Tiny rows need no histogram. Bounds and exclusion still precede every
  // score load; a forced score and all capacity padding may contain NaNs.
  if (ordinary <= kFullRowFinishCapacity) {
    __syncthreads();
    // ordinary<=128 implies valid<=129: one forced hole may be inside the
    // prefix. Bound the loop statically so the compiler need not spill a
    // general loop's row-offset/length temporaries. The 128-thread variant
    // MUST still visit column128 when the excluded column lies before it.
    constexpr int kTinyItems = (kFullRowFinishCapacity + Threads) / Threads;
#pragma unroll
    for (int item = 0; item < kTinyItems; ++item) {
      if (item * Threads < valid) {
        const int local = item * Threads + tid;
        const bool live = local < valid && local != excluded;
        uint64_t key = 0ull;
        if (live) {
          const float score = row_scores[local];
          assert(!isnan(score));
          key = canonical_key(score, scan_block_begin + global_block_stride * local);
        }
        full_row_append_key(key, live, sm);
      }
    }
    __syncthreads();
    prefill_full_row_finish<FourWarpFinish>(row_out, wanted, sm);
    local_pdl_complete(use_pdl);
    return;
  }

  // First pass: one coarse, monotone histogram over the LIVE row. The 16th
  // element's bin plus all higher bins is an exact superset of the answer.
  for (int i = tid; i < kFullRowCoarseBins; i += Threads) {
    sm.histogram[full_row_histogram_index(i)] = 0;
  }
  __syncthreads();
  bool valid_scores = true;
  // Retain three 10-bit coarse bins per word across the threshold step.
  // Independent score prefetches live only inside the first-pass scope; final
  // compaction reloads exact FP32 values solely for the <=128-key superset.
  // This fixed cache admission still depends on the exact device prefix.
  const bool cache_row = CachedItems > 0 && valid <= Threads * CachedItems;
  constexpr int kBinsPerWord = 3;
  constexpr int kPackedWords = (CachedItems + kBinsPerWord - 1) / kBinsPerWord;
  static_assert(kBinsPerWord * kFullRowCoarseBits <= 32);
  uint32_t packed_bins[kPackedWords > 0 ? kPackedWords : 1] = {};
  if (cache_row) {
    // Uninitialized slots are never consumed: the histogram loop repeats the
    // population loop's exact live/non-forced predicate. This array goes dead
    // before thresholding and is not a full-row cache across selector passes.
    float prefetched_scores[CachedItems > 0 ? CachedItems : 1];
#pragma unroll
    for (int item = 0; item < CachedItems; ++item) {
      const int local = tid + item * Threads;
      if (local < valid && local != excluded) {
        prefetched_scores[item] = row_scores[local];
      }
    }
#pragma unroll
    for (int item = 0; item < CachedItems; ++item) {
      if (item * Threads < valid) {
        const int local = tid + item * Threads;
        int bin = -1;
        if (local < valid && local != excluded) {
          const float score = prefetched_scores[item];
          valid_scores = valid_scores && !isnan(score);
          bin = full_row_coarse_bin(score);
          constexpr uint32_t kBinMask = (1u << kFullRowCoarseBits) - 1u;
          const int word = item / kBinsPerWord;
          const int shift = (item % kBinsPerWord) * kFullRowCoarseBits;
          packed_bins[word] |= (static_cast<uint32_t>(bin) & kBinMask) << shift;
        }
        // Invalid fields need no sentinel: the repeated live/non-forced
        // compaction predicate excludes them before reading an exact score.
        full_row_histogram_add(bin, sm.histogram);
      }
    }
  } else {
    if constexpr (CachedItems == 0) {
      // A fixed four-score prefetch exposes independent coalesced loads before
      // the dependent conversion/histogram work. This is a streaming tile,
      // not a row-sized register cache. In particular, the same exact prefix
      // and forced-column predicates protect every load in the partial tile.
      constexpr int kPrefetchItems = 4;
      for (int base = 0; base < valid; base += Threads * kPrefetchItems) {
        float prefetched[kPrefetchItems] = {};
#pragma unroll
        for (int item = 0; item < kPrefetchItems; ++item) {
          const int local = base + item * Threads + tid;
          if (local < valid && local != excluded) {
            prefetched[item] = row_scores[local];
          }
        }
#pragma unroll
        for (int item = 0; item < kPrefetchItems; ++item) {
          // The item predicate is uniform across each CTA, so every lane
          // enters match_any, including masked lanes of the final chunk.
          if (base + item * Threads < valid) {
            const int local = base + item * Threads + tid;
            const bool live = local < valid && local != excluded;
            int bin = -1;
            if (live) {
              const float score = prefetched[item];
              valid_scores = valid_scores && !isnan(score);
              bin = full_row_coarse_bin(score);
            }
            full_row_histogram_add(bin, sm.histogram);
          }
        }
      }
    } else {
      for (int base = 0; base < valid; base += Threads) {
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
  }
  assert(valid_scores);
  __syncthreads();
  prefill_full_row_coarse_threshold<Threads>(wanted, sm);
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
    for (int item = 0; item < CachedItems; ++item) {
      if (item * Threads < valid) {
        const int local = tid + item * Threads;
        const bool live = local < valid && local != excluded;
        constexpr uint32_t kBinMask = (1u << kFullRowCoarseBits) - 1u;
        const int shift = (item % kBinsPerWord) * kFullRowCoarseBits;
        const int bin = static_cast<int>(
            (packed_bins[item / kBinsPerWord] >> shift) & kBinMask);
        const bool keep = live && bin >= coarse_threshold;
        uint64_t key = 0ull;
        if (keep) {
          // Bounds and forced exclusion dominate this sparse reload. Only
          // the exact coarse superset is loaded; canonical FP32/ID ordering
          // remains unchanged and valid -infinity still has a nonzero key.
          const float score = row_scores[local];
          key = canonical_key(score,
                              scan_block_begin + global_block_stride * local);
        }
        full_row_append_key(key, keep, sm);
      }
    }
  } else {
    if constexpr (CachedItems == 0) {
      // Keep this four-score tile scoped to compaction: the first-pass tile
      // is dead before thresholding. Exact prefix/exclusion guards precede
      // every prefetch, including the final partial tile and forced hole.
      constexpr int kPrefetchItems = 4;
      for (int base = 0; base < valid; base += Threads * kPrefetchItems) {
        float prefetched[kPrefetchItems] = {};
#pragma unroll
        for (int item = 0; item < kPrefetchItems; ++item) {
          const int local = base + item * Threads + tid;
          if (local < valid && local != excluded) {
            prefetched[item] = row_scores[local];
          }
        }
#pragma unroll
        for (int item = 0; item < kPrefetchItems; ++item) {
          // CTA-uniform admission keeps every lane in append's ballot,
          // including invalid and excluded lanes contributing keep=false.
          if (base + item * Threads < valid) {
            const int local = base + item * Threads + tid;
            const bool live = local < valid && local != excluded;
            uint64_t key = 0ull;
            bool keep = false;
            if (live) {
              const float score = prefetched[item];
              if (coarse_finish) {
                keep = full_row_coarse_bin(score) >= coarse_threshold;
                if (keep) {
                  key = canonical_key(
                      score, scan_block_begin + global_block_stride * local);
                }
              } else {
                key = canonical_key(
                    score, scan_block_begin + global_block_stride * local);
                keep = key >= final_prefix;
              }
            }
            full_row_append_key(key, keep, sm);
          }
        }
      }
    } else {
      for (int base = 0; base < valid; base += Threads) {
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
  }
  __syncthreads();
  prefill_full_row_finish<FourWarpFinish>(row_out, wanted, sm);
  local_pdl_complete(use_pdl);
}


// Static small-capacity admission: one complete warp ranks each score row.
// Four rows share a CTA, including at the final partial CTA. Missing rows do
// no metadata or score reads but still participate in CTA-level PDL ordering.
__global__ void __launch_bounds__(128) prefill_candidates_tiny_warp_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    float* __restrict__ out, int heads, int blocks, int row_count,
    int scan_block_begin, int global_block_stride, bool use_pdl) {
  local_pdl_wait(use_pdl);
  assert(blockDim.x == 128 && blocks >= 0 && blocks <= 32);
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * 4 + (threadIdx.x >> 5);
  const bool has_row = row < row_count;
  uint64_t key = 0ull;
  int wanted = 0;
  if (has_row) {
    // Only the at-most-three absent row ordinals may exceed int32. The
    // validated real-row range admits a cheaper 32-bit token division.
    const int token = static_cast<int>(row) / heads;
    if (active[token]) {
      const int count = nvalid[token];
      const int excluded = forced_col[token];
      if (lane == 0) {
        assert(count >= 0 && count <= blocks);
        assert(excluded < 0 || excluded < count);
      }
      const int valid = max(0, min(count, blocks));
      wanted = min(kTopK, valid - (excluded >= 0 ? 1 : 0));
      // Prefix and exclusion are checked before forming a score load. A
      // forced score, inactive row and all capacity padding may contain NaNs.
      if (lane < valid && lane != excluded) {
        const float score = scores[row * blocks + lane];
        assert(!isnan(score));
        key = canonical_key(score, scan_block_begin + global_block_stride * lane);
      }
    }
  }
  // Every warp, including missing rows, executes all shuffles with all lanes.
  // Canonical keys are unique and positive for ordinary candidates; zero
  // padding sorts behind valid -infinity and either sign of canonical zero.
#pragma unroll
  for (int width = 2; width <= 32; width <<= 1) {
#pragma unroll
    for (int distance = width >> 1; distance > 0; distance >>= 1) {
      const uint64_t other = __shfl_xor_sync(0xffffffffu, key, distance);
      const bool descending = (lane & width) == 0;
      const bool lower_lane = (lane & distance) == 0;
      if (descending == lower_lane) {
        key = key > other ? key : other;
      } else {
        key = key < other ? key : other;
      }
    }
  }
  if (has_row && lane < kTopK) {
    store_local_candidate_keyonly(out + row * kTopK * 2, lane,
                                  lane < wanted ? key : 0ull);
  }
  __syncthreads();
  local_pdl_complete(use_pdl);
}

}  // namespace icp
