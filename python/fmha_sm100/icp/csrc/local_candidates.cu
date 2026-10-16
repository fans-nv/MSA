// The prefill local selector kernels and their host launch.
#include <torch/extension.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <limits>

#include "local_candidates.cuh"

namespace icp {

// Prefill: a 128-thread CTA per row for occupancy at thousands of rows, with a
// four-score streaming tile instead of decode's register cache.
template <bool FourWarpFinish>
__global__ void __launch_bounds__(kPrefillThreads) prefill_candidates_full_row_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    float* __restrict__ out, int heads, int blocks, int scan_block_begin,
    int global_block_stride, bool use_pdl) {
  constexpr int Threads = kPrefillThreads;
  __shared__ FullRowSelectSmem sm;
  local_pdl_wait(use_pdl);
  const int row = blockIdx.x;
  const int token = row / heads;
  const int tid = threadIdx.x;
  const auto [valid, excluded, ordinary, wanted] =
      full_row_extent<Threads>(nvalid, forced_col, active, token, blocks);
  const float* row_scores = scores + static_cast<int64_t>(row) * blocks;
  float* row_out = out + static_cast<int64_t>(row) * kTopK * 2;

  if (wanted <= 0) {
    if (tid < kTopK) store_local_candidate_keyonly(row_out, tid, 0ull);
    __syncthreads();
    local_pdl_complete(use_pdl);
    return;
  }

  if (tid == 0) sm.candidate_count = 0;

  if (ordinary <= kFullRowFinishCapacity) {
    __syncthreads();
    // Static bound (valid <= 129) avoids spilling a general loop's temporaries.
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

  for (int i = tid; i < kFullRowCoarseBins; i += Threads) {
    sm.histogram[full_row_histogram_index(i)] = 0;
  }
  __syncthreads();
  bool valid_scores = true;
  for (int base = 0; base < valid; base += Threads * kPrefillPrefetchItems) {
    float prefetched[kPrefillPrefetchItems] = {};
#pragma unroll
    for (int item = 0; item < kPrefillPrefetchItems; ++item) {
      const int local = base + item * Threads + tid;
      if (local < valid && local != excluded) {
        prefetched[item] = row_scores[local];
      }
    }
#pragma unroll
    for (int item = 0; item < kPrefillPrefetchItems; ++item) {
      // CTA-uniform predicate: every lane enters match_any.
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
  assert(valid_scores);
  __syncthreads();
  full_row_coarse_threshold<Threads>(wanted, sm);
  const int coarse_threshold = sm.threshold;
  const bool coarse_finish =
      sm.above + sm.boundary_count <= kFullRowFinishCapacity;
  const uint64_t final_prefix =
      coarse_finish ? 0ull
                    : full_row_exact_prefix<Threads>(
                          row_scores, valid, excluded, wanted, scan_block_begin,
                          global_block_stride, sm);

  if (tid == 0) sm.candidate_count = 0;
  __syncthreads();
  for (int base = 0; base < valid; base += Threads * kPrefillPrefetchItems) {
    float prefetched[kPrefillPrefetchItems] = {};
#pragma unroll
    for (int item = 0; item < kPrefillPrefetchItems; ++item) {
      const int local = base + item * Threads + tid;
      if (local < valid && local != excluded) {
        prefetched[item] = row_scores[local];
      }
    }
#pragma unroll
    for (int item = 0; item < kPrefillPrefetchItems; ++item) {
      // CTA-uniform predicate: every lane enters append's ballot.
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
  __syncthreads();
  prefill_full_row_finish<FourWarpFinish>(row_out, wanted, sm);
  local_pdl_complete(use_pdl);
}

// Prefill rows of at most 32 columns: one warp ranks each row, four rows per
// CTA. Absent rows read nothing but still join the shuffles and PDL ordering.
__global__ void __launch_bounds__(128) prefill_candidates_tiny_warp_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nvalid,
    const int32_t* __restrict__ forced_col, const bool* __restrict__ active,
    float* __restrict__ out, int heads, int blocks, int row_count,
    int scan_block_begin, int global_block_stride, bool use_pdl) {
  local_pdl_wait(use_pdl);
  assert(blockDim.x == 128 && blocks >= 0 && blocks <= kTinyRowBlocks);
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * 4 + (threadIdx.x >> 5);
  const bool has_row = row < row_count;
  uint64_t key = 0ull;
  int wanted = 0;
  if (has_row) {
    // Real rows are below row_count, so int32 division is exact.
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
      if (lane < valid && lane != excluded) {
        const float score = scores[row * blocks + lane];
        assert(!isnan(score));
        key = canonical_key(score, scan_block_begin + global_block_stride * lane);
      }
    }
  }
  key = warp_sort_desc(key, lane);
  if (has_row && lane < kTopK) {
    store_local_candidate_keyonly(out + row * kTopK * 2, lane,
                                  lane < wanted ? key : 0ull);
  }
  __syncthreads();
  local_pdl_complete(use_pdl);
}

}  // namespace icp

namespace {

template <int Threads, typename Kernel, typename... Args>
void launch_local(Kernel kernel, int grid, cudaStream_t stream, bool use_pdl,
                  Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(Threads);
  config.stream = stream;
  cudaLaunchAttribute attribute{};
  if (use_pdl) {
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attribute;
    config.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

void check_tensor(const at::Tensor& tensor, const char* name,
                  const at::Tensor& scores, at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.device() == scores.device(), name,
              " must be on the scores device");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Shape, dtype, aliasing and int32-range checks; returns the stream to launch on.
cudaStream_t check_selector_arguments(const at::Tensor& scores,
                                      const at::Tensor& nvalid,
                                      const at::Tensor& forced_col,
                                      const at::Tensor& active,
                                      const at::Tensor& out,
                                      int64_t scan_block_begin,
                                      int64_t global_block_stride, bool use_pdl) {
  check_tensor(scores, "scores", scores, at::kFloat);
  check_tensor(nvalid, "nvalid", scores, at::kInt);
  check_tensor(forced_col, "forced_col", scores, at::kInt);
  check_tensor(active, "active", scores, at::kBool);
  check_tensor(out, "out", scores, at::kFloat);
  TORCH_CHECK(scores.dim() == 3, "scores must be [T,H,N]");
  TORCH_CHECK(scan_block_begin >= 0, "scan_block_begin must be non-negative");
  TORCH_CHECK(global_block_stride == 1 || global_block_stride == 2,
              "global_block_stride must be 1 or 2");
  const int64_t tokens = scores.size(0);
  const int64_t heads = scores.size(1);
  const int64_t blocks = scores.size(2);
  constexpr int64_t kMaxInt = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(heads > 0 && tokens * heads <= kMaxInt,
              "T * H must be a positive int32 grid");
  TORCH_CHECK(blocks <= icp::kMaxRowBlocks, "the full-row selector covers N <= ",
              icp::kMaxRowBlocks, " columns; got ", blocks);
  TORCH_CHECK(blocks == 0 ||
                  scan_block_begin + global_block_stride * (blocks - 1) <= kMaxInt,
              "global block IDs must fit int32");
  for (const auto& tensor : {nvalid, forced_col, active}) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == tokens,
                "metadata must have shape [T]");
  }
  TORCH_CHECK(out.dim() == 4 && out.size(0) == tokens && out.size(1) == heads &&
                  out.size(2) == icp::kTopK && out.size(3) == 2,
              "out must be [T,H,16,2]");
  for (const auto& input : {scores, nvalid, forced_col, active}) {
    at::assert_no_overlap(out, input);
  }
  if (use_pdl) {
    TORCH_CHECK(at::cuda::getDeviceProperties(scores.get_device())->major >= 9,
                "PDL requires compute capability >= 9.0");
  }
  return c10::cuda::getCurrentCUDAStream(scores.get_device());
}

}  // namespace

// Prefill: the row width alone picks the kernel (warp per row, then the
// 128-thread CTA with or without the four-warp finish).
void select_prefill_candidates(at::Tensor scores, at::Tensor nvalid,
                               at::Tensor forced_col, at::Tensor active,
                               at::Tensor out, int64_t scan_block_begin,
                               bool use_pdl, int64_t global_block_stride) {
  const auto stream = check_selector_arguments(
      scores, nvalid, forced_col, active, out, scan_block_begin,
      global_block_stride, use_pdl);
  const int64_t heads = scores.size(1);
  const int64_t blocks = scores.size(2);
  const int rows = static_cast<int>(scores.size(0) * heads);
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(scores.device());
  const auto launch = [&](auto kernel, int grid, auto... row_count) {
    launch_local<icp::kPrefillThreads>(
        kernel, grid, stream, use_pdl, scores.data_ptr<float>(),
        nvalid.data_ptr<int32_t>(), forced_col.data_ptr<int32_t>(),
        active.data_ptr<bool>(), out.data_ptr<float>(), static_cast<int>(heads),
        static_cast<int>(blocks), row_count..., static_cast<int>(scan_block_begin),
        static_cast<int>(global_block_stride), use_pdl);
  };
  if (blocks <= icp::kTinyRowBlocks) {
    launch(icp::prefill_candidates_tiny_warp_kernel, (rows + 3) / 4, rows);
  } else if (blocks <= icp::kFourWarpMaxBlocks) {
    launch(icp::prefill_candidates_full_row_kernel<true>, rows);
  } else {
    launch(icp::prefill_candidates_full_row_kernel<false>, rows);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select_prefill_candidates", &select_prefill_candidates,
        "Prefill full-row local Top-16 (C4), kernel chosen by row width.",
        pybind11::arg("scores"), pybind11::arg("nvalid"),
        pybind11::arg("forced_col"), pybind11::arg("active"),
        pybind11::arg("out"), pybind11::arg("scan_block_begin"),
        pybind11::arg("use_pdl"), pybind11::arg("global_block_stride") = 1);
}
