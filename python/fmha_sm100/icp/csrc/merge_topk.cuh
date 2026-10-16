// Canonical candidate key and the per-row merge shared by every ICP kernel.
//
// Key (CONTRACT C5): (sortable(score) << 32) | ~uint32(id); invalid id -> 0.
// One unsigned `>` is "outranks": score descending, then block id ascending.
// Merge (C3/C6): max-reduce duplicate ids, keep `n_ordinary` ordinary winners,
// inject the forced block once, emit ids ascending with a -1 tail. A NaN on a
// valid record fails the row through the caller's status word.
#pragma once

#include <cstdint>

namespace icp {

constexpr int kTopK = 16;
constexpr unsigned kFullMask = 0xffffffffu;

// An inactive row is (kNoForcedBlock, 0); a plain ordinary merge is (kNoForcedBlock, kTopK).
constexpr int32_t kNoForcedBlock = -1;

// Bits ORed into the caller's status word; any non-zero value fails the invocation.
constexpr int32_t kStatusOk = 0;
constexpr int32_t kStatusNaN = 1;      // C5: a NaN score on a valid record.
constexpr int32_t kStatusRowMeta = 2;  // C3: ordinary count disagrees with f.

// fp32 -> uint32 whose unsigned order is the float order; -0 and +0 share a key.
__device__ __forceinline__ uint32_t sortable_bits(float score) {
  uint32_t raw = __float_as_uint(score);
  uint32_t bits = ((raw & 0x7fffffffu) == 0u) ? 0u : raw;
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ uint64_t canonical_key(float score, int32_t gid) {
  if (gid < 0) return 0ull;
  return (static_cast<uint64_t>(sortable_bits(score)) << 32) |
         static_cast<uint64_t>(~static_cast<uint32_t>(gid));
}

// C3's ordinary quota min(15, M-1) with M = f+1; 0 for an inactive row.
__device__ __forceinline__ int ordinary_quota(int32_t forced) {
  return (forced < 0) ? 0 : (forced < kTopK - 1 ? forced : kTopK - 1);
}

// 32-lane bitonic sort, descending across `lane` 0..31; all 32 lanes must call.
__device__ __forceinline__ uint64_t warp_sort_desc(uint64_t key, int lane) {
#pragma unroll
  for (int width = 2; width <= 32; width <<= 1) {
#pragma unroll
    for (int distance = width >> 1; distance > 0; distance >>= 1) {
      const uint64_t other = __shfl_xor_sync(kFullMask, key, distance);
      const bool descending = (lane & width) == 0;
      const bool lower_lane = (lane & distance) == 0;
      if (descending == lower_lane) {
        key = key > other ? key : other;
      } else {
        key = key < other ? key : other;
      }
    }
  }
  return key;
}

// One row's `32 * KPT` candidates, lane L holding flat indices L, L+32, ...,
// merged into out[0..15]. `n_ordinary <= kTopK - 1` whenever `forced >= 0`.
template <int KPT>
__device__ __forceinline__ void warp_merge_topk16(const uint64_t (&key)[KPT],
                                                  const int32_t (&gid)[KPT],
                                                  int lane, int32_t forced,
                                                  int n_ordinary,
                                                  int32_t* out) {
  // Pass 0: one representative per id (max key, ties to the lowest flat index);
  // the forced block never competes for an ordinary slot.
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
        const bool same = rep[i] && (ogid == gid[i]);
        const bool beats =
            (okey > key[i]) || ((okey == key[i]) && (oidx < lane + i * 32));
        if (same && beats) rep[i] = false;
      }
    }
  }

  // Materialised pool: ends key/gid liveness here, which holds the register count.
  uint64_t rkey[KPT];
  int32_t rgid[KPT];
#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    rkey[i] = rep[i] ? key[i] : 0ull;
    rgid[i] = rep[i] ? gid[i] : -1;
  }

  // Pass 1: rank among distinct representatives.
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

  bool win[KPT];
#pragma unroll
  for (int i = 0; i < KPT; ++i)
    win[i] = (rgid[i] >= 0) && (rank[i] < n_ordinary);

  if (lane < kTopK) out[lane] = -1;
  __syncwarp();

  // Pass 2: ascending slot among winners; the forced id shifts larger winners by one.
  int pos[KPT];
#pragma unroll
  for (int i = 0; i < KPT; ++i)
    pos[i] = (forced >= 0 && forced < rgid[i]) ? 1 : 0;
  int fpos = 0;

#pragma unroll 1
  for (int j = 0; j < 32; ++j) {
#pragma unroll
    for (int s = 0; s < KPT; ++s) {
      const int32_t og = __shfl_sync(kFullMask, win[s] ? rgid[s] : INT32_MAX, j);
      fpos += (og < forced) ? 1 : 0;
#pragma unroll
      for (int i = 0; i < KPT; ++i) pos[i] += (og < rgid[i]) ? 1 : 0;
    }
  }

#pragma unroll
  for (int i = 0; i < KPT; ++i) {
    if (win[i]) out[pos[i]] = rgid[i];
  }
  if (lane == 0 && forced >= 0) out[fpos] = forced;
  __syncwarp();
}

// warp_merge_topk16<1> with the same output bits via log-depth shuffle networks.
__device__ __forceinline__ void warp_merge_topk16_net(uint64_t key, int32_t gid,
                                                      int lane, int32_t forced,
                                                      int n_ordinary, int32_t* out) {
  bool rep = (gid >= 0) && (gid != forced);
  // Only potential representatives walk their id group, so padding costs no rounds.
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
  // Surviving keys are distinct, so after the sort lane i holds rank i.
  const uint64_t v = warp_sort_desc(rep ? key : 0ull, lane);
  const bool win = (v != 0ull) && (lane < n_ordinary);
  const int32_t wid = win ? static_cast<int32_t>(~static_cast<uint32_t>(v)) : -1;
  const int fpos =
      __popc(__ballot_sync(kFullMask, win && forced >= 0 && wid < forced));
  // Ascending sort of the winners' ids; non-winners sort last.
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

// Loads this lane's share of a source-major C4 receive carrier
// cand[s][q][hl][k][{score bits, id}]; both words are bitcast, never converted.
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
      const float score = __int_as_float(cand[base]);
      const int32_t g = cand[base + 1];
      // Validity is by id; an invalid record's score is never examined.
      nan_seen = nan_seen || (g >= 0 && score != score);
      key[i] = canonical_key(score, g);
      gid[i] = g;
    } else {
      key[i] = 0ull;
      gid[i] = -1;
    }
  }
}

}  // namespace icp
