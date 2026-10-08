// The selector's host side: validation, the capacity ladder and the launches.
//
// Input  `scores`   [T, H_group, N] fp32, N = max_local_blocks.
// Output `out`      [T, H_group, 16, 2] fp32 (CONTRACT C4).
// Scratch `partials` [T, H_group, S, 16, 2] fp32, S = ceil(N / kLocalPartition)
//                   when N exceeds one partition and 0 otherwise.
//
// Two launchers, because the bands do not share a launch policy. Decode is
// cudagraph-captured: `select_local_candidates_impl` takes the rung and the
// partition count from CAPACITY and keeps every arm reachable for the A/B.
// Prefill is eager: its public caller supplies the startup capacity as the
// extent. Bounded radix remains the default; full-row selection is explicit.
//
// Both arms live in ONE translation unit and ONE extension so that an A/B is
// interleavable inside a single process and a single timing call, and so the
// two arms cannot acquire different build fingerprints.
#include <torch/extension.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <limits>
#include <vector>

#include "local_candidates.cuh"
#include "local_candidates_rs.cuh"
#include "local_candidates_full_row.cuh"
#include "local_candidates_prefill_full_row.cuh"

namespace {

// Which kernel was actually launched, for the dispatch gate. A dispatch that
// silently fell back to one arm would report one kernel's time under another's
// name, and because all three producers are bit-exact no output comparison can
// catch it.
//
// The identifier is the HOST STUB ADDRESS, not the register/shared footprint:
// two rungs of different arms can share a footprint, so a footprint comparison
// cannot tell those arms apart. Footprint is reported as corroboration only.
//
// OFF by default: cudaFuncGetAttributes is a host call and would land in the
// eager timing path.
struct LaunchRecord {
  unsigned long long symbol = 0;
  int regs = -1;
  int smem = -1;
  int grid = -1;
  long long count = 0;
  int threads = -1;
};
LaunchRecord g_last_launch;
bool g_record_launches = false;
bool g_record_armed = false;

template <int Threads = icp::kLocalThreads, typename Kernel, typename... Args>
void launch_local(Kernel kernel, int grid, cudaStream_t stream, bool use_pdl,
                  Args... args) {
  if (g_record_armed) {
    // Only the SELECT launch is recorded; the combine would otherwise
    // overwrite it on partitioned capacities.
    g_record_armed = false;
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    g_last_launch = {reinterpret_cast<unsigned long long>(
                         reinterpret_cast<const void*>(kernel)),
                     attributes.numRegs,
                     static_cast<int>(attributes.sharedSizeBytes), grid,
                     g_last_launch.count + 1, Threads};
  }
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

template <int Items>
void select_partition(const at::Tensor& scores, const at::Tensor& nvalid,
                      const at::Tensor& forced_col, const at::Tensor& active,
                      at::Tensor& result, int heads, int blocks, int partitions,
                      int grid, int scan_block_begin, int global_block_stride,
                      bool use_pdl, cudaStream_t stream) {
  launch_local(icp::local_candidates_kernel<Items>, grid, stream, use_pdl,
               scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),
               forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),
               result.data_ptr<float>(), heads, blocks, partitions,
               scan_block_begin, global_block_stride, use_pdl);
}

template <int Items>
void select_partition_radix(const at::Tensor& scores, const at::Tensor& nvalid,
                            const at::Tensor& forced_col,
                            const at::Tensor& active, at::Tensor& result,
                            int heads, int blocks, int partitions, int grid,
                            int scan_block_begin, int global_block_stride,
                            bool use_pdl, bool short_row, cudaStream_t stream) {
  launch_local(icp::local_candidates_radix_kernel<Items>, grid, stream, use_pdl,
               scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),
               forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),
               result.data_ptr<float>(), heads, blocks, partitions,
               scan_block_begin, global_block_stride, use_pdl, short_row);
}

template <int Items>
void select_capped(const at::Tensor& scores, const at::Tensor& nvalid,
                   const at::Tensor& forced_col, const at::Tensor& active,
                   at::Tensor& result, int heads, int blocks, int partitions,
                   int grid, int scan_block_begin, int global_block_stride,
                   bool use_pdl, cudaStream_t stream) {
  launch_local(icp::local_candidates_capped_kernel<Items>, grid, stream, use_pdl,
               scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),
               forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),
               result.data_ptr<float>(), heads, blocks, partitions,
               scan_block_begin, global_block_stride, use_pdl);
}

void check_tensor(const at::Tensor& tensor, const char* name,
                  const at::Tensor& scores, at::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.device() == scores.device(), name,
              " must be on the scores device");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Shape, dtype, aliasing and int32-range checks, shared by both bands. The
// partition count is a parameter because it is the one thing the two disagree
// on: decode takes it from capacity, prefill from the live extent.
void check_selector_arguments(const at::Tensor& scores, const at::Tensor& nvalid,
                              const at::Tensor& forced_col,
                              const at::Tensor& active, const at::Tensor& out,
                              const at::Tensor& partials,
                              int64_t scan_block_begin, int64_t partitions,
                              int64_t global_block_stride) {
  check_tensor(scores, "scores", scores, at::kFloat);
  check_tensor(nvalid, "nvalid", scores, at::kInt);
  check_tensor(forced_col, "forced_col", scores, at::kInt);
  check_tensor(active, "active", scores, at::kBool);
  check_tensor(out, "out", scores, at::kFloat);
  check_tensor(partials, "partials", scores, at::kFloat);
  TORCH_CHECK(scores.dim() == 3, "scores must be [T,H,N]");
  // refined-icp-v1: the old (world, rank) pair no longer forms a global id, so
  // its [1,8]/[0,C) checks are gone. What replaces them is the absolute scan
  // origin and stride: the caller chooses the physical-profile mapping.
  TORCH_CHECK(scan_block_begin >= 0, "scan_block_begin must be non-negative");
  TORCH_CHECK(global_block_stride == 1 || global_block_stride == 2,
              "global_block_stride must be 1 or 2");
  const int64_t tokens = scores.size(0);
  const int64_t heads = scores.size(1);
  const int64_t blocks = scores.size(2);
  constexpr int64_t kMaxInt = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(heads > 0 && heads <= kMaxInt && tokens <= kMaxInt,
              "T must fit int32 and H must be a positive int32");
  // refined C1 id bound: the largest id this launch can emit is
  // scan_block_begin + global_block_stride * (blocks - 1).
  TORCH_CHECK(scan_block_begin <= kMaxInt && blocks <= kMaxInt &&
                  (blocks == 0 || scan_block_begin +
                      global_block_stride * (blocks - 1) <= kMaxInt),
              "global block IDs must fit int32");
  TORCH_CHECK(tokens * heads <= kMaxInt / std::max<int64_t>(partitions, 1),
              "selector grid exceeds int32 CUDA grid bound");
  for (const auto& tensor : {nvalid, forced_col, active}) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == tokens,
                "metadata must have shape [T]");
  }
  TORCH_CHECK(out.dim() == 4 && out.size(0) == tokens && out.size(1) == heads &&
                  out.size(2) == icp::kTopK && out.size(3) == 2,
              "out must be [T,H,16,2]");
  TORCH_CHECK(
      partials.dim() == 5 && partials.size(0) == tokens &&
          partials.size(1) == heads && partials.size(2) == partitions &&
          partials.size(3) == icp::kTopK && partials.size(4) == 2,
      "partials must be [T,H,S,16,2], S=ceil(N/4096) for N>4096, else 0");
  for (const auto& input : {scores, nvalid, forced_col, active}) {
    at::assert_no_overlap(out, input);
    at::assert_no_overlap(partials, input);
  }
  at::assert_no_overlap(out, partials);
}

// The DECODE launcher: cudagraph-captured, so the geometry may depend on
// nothing but capacity, and every arm stays reachable for the A/B.
//
// `cap` selects the arm: 0 the sort arm, -1 radix select, -2 radix select with
// the live-count bound, > 0 radix select at a fixed register cap.
void select_local_candidates_impl(at::Tensor scores, at::Tensor nvalid,
                                  at::Tensor forced_col, at::Tensor active,
                                  at::Tensor out, at::Tensor partials,
                                  int64_t scan_block_begin, bool use_pdl,
                                  int64_t cap, int64_t global_block_stride) {
  TORCH_CHECK(cap == 0 || cap == -1 || cap == -2 || cap == -3 || cap == 2 || cap == 4 ||
                  cap == 8 || cap == 16,
              "cap must be 0 (sort), -1 (radix), -2 (radix, short-row bound), -3 (full row) "
              "or one of {2,4,8,16}");
  const bool radix = cap != 0;
  const int64_t tokens = scores.dim() == 3 ? scores.size(0) : 0;
  const int64_t heads = scores.dim() == 3 ? scores.size(1) : 0;
  const int64_t blocks = scores.dim() == 3 ? scores.size(2) : 0;
  const int64_t partitions =
      blocks > icp::kLocalPartition
          ? (blocks + icp::kLocalPartition - 1) / icp::kLocalPartition
          : 0;
  check_selector_arguments(scores, nvalid, forced_col, active, out, partials,
                           scan_block_begin, partitions, global_block_stride);
  const int64_t launch_partitions = std::max<int64_t>(partitions, 1);
  if (tokens == 0) return;

  const c10::cuda::CUDAGuard guard(scores.device());
  if (use_pdl) {
    TORCH_CHECK(at::cuda::getDeviceProperties(scores.get_device())->major >= 9,
                "PDL requires compute capability >= 9.0");
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  const int rows = static_cast<int>(tokens * heads);
  const int grid = static_cast<int>(rows * launch_partitions);
  auto& destination = partitions ? partials : out;
  g_record_armed = g_record_launches;

  if (cap == -3 && blocks <= 8192) {
    launch_local<icp::kFullRowThreads>(icp::local_candidates_full_row_kernel, rows, stream, use_pdl,
                 scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),
                 forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),
                 out.data_ptr<float>(), static_cast<int>(heads),
                 static_cast<int>(blocks), static_cast<int>(scan_block_begin),
                 static_cast<int>(global_block_stride), use_pdl);
    return;
  }
  // Preserve the bounded partitioned implementation for larger capacities.
  if (cap == -3) cap = -2;

  if (cap > 0) {
    // The capped arm's whole point: no capacity ladder, the register depth IS
    // the cap.
#define SELECT_CAPPED(ITEMS)                                                  \
  select_capped<ITEMS>(scores, nvalid, forced_col, active, destination, heads, \
                       blocks, launch_partitions, grid,                        \
                       static_cast<int>(scan_block_begin),                   \
                       static_cast<int>(global_block_stride), use_pdl, stream)
    if (cap == 2) {
      SELECT_CAPPED(2);
    } else if (cap == 4) {
      SELECT_CAPPED(4);
    } else if (cap == 8) {
      SELECT_CAPPED(8);
    } else {
      SELECT_CAPPED(16);
    }
#undef SELECT_CAPPED
  } else {
#define SELECT_PARTITION(ITEMS)                                                \
  do {                                                                         \
    if (radix) {                                                               \
      select_partition_radix<ITEMS>(scores, nvalid, forced_col, active,        \
                                    destination, heads, blocks,                \
                                    launch_partitions, grid,                   \
                                    static_cast<int>(scan_block_begin),        \
                                    static_cast<int>(global_block_stride),     \
                                    use_pdl, cap == -2, stream);               \
    } else {                                                                   \
      select_partition<ITEMS>(scores, nvalid, forced_col, active, destination, \
                              heads, blocks, launch_partitions, grid,          \
                              static_cast<int>(scan_block_begin),              \
                              static_cast<int>(global_block_stride), use_pdl, \
                              stream);                                         \
    }                                                                          \
  } while (0)
  // -------------------------------------------------------------------------
  // The `Items` ladder, RE-DERIVED for refined-icp-v1's wider score row.
  //
  // A CTA is kLocalThreads(128) threads holding `Items` registers each, so it
  // covers 128*Items columns; the rung is the smallest Items with
  // 128*Items >= min(blocks, kLocalPartition). That rule is unchanged. What
  // changed is the ARGUMENT: refined C1 makes every rank see every logical
  // block, so `blocks` is ceil(N_tok/128), not M0's ceil(N_tok/128/W) -- W
  // times wider, for W = 2 or 4.
  //
  //   context     M0 blocks (W=4)  rung | refined blocks  rung
  //   32768             64          1   |      256          2
  //   131072           256          2   |     1024          8
  //   262144           512          4   |     2048         16   <-- production
  //   1048576         2048         16   |     8192         32 + 2 partitions
  //
  // Two conclusions, both deliberate:
  //
  // 1. NO BOUNDARY MOVES. The ladder is already a pure function of `blocks` and
  //    stays exact at the wider width; every rung still satisfies
  //    128*Items >= blocks, and blocks > kLocalPartition still routes to the
  //    partitioned path where a CTA must span kLocalPartition, i.e. Items=32
  //    (kLocalPartition/kLocalThreads). Widening does not require an edit here,
  //    and inventing one would be churn. This comment IS the re-derivation.
  //
  // 2. THE PRODUCTION POINT MOVES ONTO THE Items=16 RUNG. At the 262144-token
  //    production context refined `blocks` is exactly 2048, the last rung
  //    before partitioning, which under M0/W=4 was only reached at 1M. The only
  //    pre-existing build data for this selector is sm_90a, a non-target arch,
  //    where `local_candidates_radix_kernel<16>` is the ONE rung reported to
  //    spill (8 bytes of local memory; every other rung reports zero). So the
  //    width growth lands production on the single rung with a known spill, on
  //    an architecture nobody had measured. It has since been measured on the
  //    target: on GB300 sm_103a the 8-byte STACK frame IS present at Items=16
  //    (and at 32, and on two capped rungs), with LOCAL:0 -- the table is in
  //    `docs/PERFORMANCE.md` section 7.1. Do NOT "fix" a spill
  //    by benchmarking -- the selector's O(capacity) cost under a W-times-wider
  //    domain is a separate, larger design problem that S3 is forbidden to
  //    solve or to measure.
  // -------------------------------------------------------------------------
  if (blocks <= 128) {
    SELECT_PARTITION(1);
  } else if (blocks <= 256) {
    SELECT_PARTITION(2);
  } else if (blocks <= 512) {
    SELECT_PARTITION(4);
  } else if (blocks <= 1024) {
    SELECT_PARTITION(8);
  } else if (blocks <= 2048) {
    SELECT_PARTITION(16);
  } else {
    SELECT_PARTITION(32);
  }
#undef SELECT_PARTITION
  }

  if (partitions) {
    const int64_t candidate_count = partitions * icp::kTopK + icp::kTopK;
#define COMBINE_PARTITIONS(ITEMS)                                              \
  do {                                                                         \
    if (radix) {                                                               \
      launch_local(icp::combine_local_candidates_radix_kernel<ITEMS>, rows,    \
                   stream, use_pdl, partials.data_ptr<float>(),                \
                   out.data_ptr<float>(), static_cast<int>(partitions),        \
                   use_pdl);                                                   \
    } else {                                                                   \
      launch_local(icp::combine_local_candidates_kernel<ITEMS>, rows, stream,  \
                   use_pdl, partials.data_ptr<float>(),                        \
                   out.data_ptr<float>(), static_cast<int>(partitions),        \
                   use_pdl);                                                   \
    }                                                                          \
  } while (0)
    if (candidate_count <= 128) {
      COMBINE_PARTITIONS(1);
    } else if (candidate_count <= 256) {
      COMBINE_PARTITIONS(2);
    } else if (candidate_count <= 512) {
      COMBINE_PARTITIONS(4);
    } else if (candidate_count <= 1024) {
      COMBINE_PARTITIONS(8);
    } else if (candidate_count <= 2048) {
      COMBINE_PARTITIONS(16);
    } else {
      COMBINE_PARTITIONS(32);
    }
#undef COMBINE_PARTITIONS
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_prefill_full_row_config(int64_t arm, int64_t threads,
                                   int64_t cached_items, bool four_warp_finish) {
  TORCH_CHECK(arm == -2 || arm == -3,
              "prefill arm must be -2 (RADIX_BOUNDED) or -3 (RADIX_FULL_ROW)");
  TORCH_CHECK((threads == 512 && cached_items == 3) ||
                  (threads == 128 && (cached_items == 0 || cached_items == 12)) ||
                  (threads == 256 && (cached_items == 0 || cached_items == 6)),
              "unsupported prefill full-row configuration; use (512,3), "
              "(128,0), (128,12), (256,0), or (256,6)");
  TORCH_CHECK(arm == -3 || (threads == 512 && cached_items == 3),
              "non-default full-row controls require RADIX_FULL_ROW");
  TORCH_CHECK(!four_warp_finish || (arm == -3 && threads == 128),
              "four-warp finish requires a 128-thread RADIX_FULL_ROW");
}

// The PREFILL launcher: its own entry symbol and bounded/full-row arms.
//
// Prefill is never cudagraph-captured, so it may allocate and take `T` per
// call. Its EXTENT, though, is no longer per-call data: the Python entry point
// passes `max_local_blocks` -- ceil(max_model_len/128), a startup constant --
// because deriving it from this step's metadata would mean reading the device
// counts on the host to prove it bounds them, and the nonblocking execution
// policy forbids that on the submission path whether or not the call is
// captured (N2, 2026-09-13). `live_blocks` therefore arrives equal to `blocks`
// from the shipping caller; the TORCH_CHECK below still admits a smaller one,
// and the kernel's two truncation assertions (`local_candidates_rs.cuh`) trap
// on the device if such an extent does not cover a row, so a hand-driven launch
// cannot truncate silently either.
void select_prefill_candidates_impl(at::Tensor scores, at::Tensor nvalid,
                                    at::Tensor forced_col, at::Tensor active,
                                    at::Tensor out, at::Tensor partials,
                                    int64_t scan_block_begin, bool use_pdl,
                                    int64_t live_blocks,
                                    int64_t global_block_stride, int64_t arm,
                                    int64_t full_row_threads,
                                    int64_t full_row_cached_items,
                                    bool full_row_four_warp_finish) {
  check_prefill_full_row_config(arm, full_row_threads, full_row_cached_items,
                                full_row_four_warp_finish);
  const int64_t tokens = scores.dim() == 3 ? scores.size(0) : 0;
  const int64_t heads = scores.dim() == 3 ? scores.size(1) : 0;
  const int64_t blocks = scores.dim() == 3 ? scores.size(2) : 0;
  TORCH_CHECK(live_blocks >= 0 && live_blocks <= blocks,
              "live_blocks must lie in [0, N]; it is an upper bound over every "
              "row in the launch on the live local ordinals, and prefill always "
              "knows it");
  TORCH_CHECK(arm != -3 || live_blocks == blocks,
              "RADIX_FULL_ROW prefill requires live_blocks == N (startup "
              "capacity), matching the public PrefillPlan entry point");
  const icp::PrefillLaunch launch =
      icp::prefill_launch(icp::LiveExtent{live_blocks});
  // Bounding the partition count is what keeps the combine's streaming loop
  // single-iteration, so this entry point never reaches a path no gate can run.
  constexpr int64_t kMaxCombine = icp::kLocalThreads * icp::kPartitionedItems;
  TORCH_CHECK(launch.partitions * icp::kTopK + icp::kTopK <= kMaxCombine,
              "prefill supports at most ", kMaxCombine / icp::kTopK - 1,
              " partitions; got ", launch.partitions);
  check_selector_arguments(scores, nvalid, forced_col, active, out, partials,
                           scan_block_begin, launch.partitions, global_block_stride);
  if (tokens == 0) return;

  const c10::cuda::CUDAGuard guard(scores.device());
  if (use_pdl) {
    TORCH_CHECK(at::cuda::getDeviceProperties(scores.get_device())->major >= 9,
                "PDL requires compute capability >= 9.0");
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  const int64_t launch_partitions = std::max<int64_t>(launch.partitions, 1);
  const int rows = static_cast<int>(tokens * heads);
  const int grid = static_cast<int>(rows * launch_partitions);
  auto& destination = launch.partitions ? partials : out;
  g_record_armed = g_record_launches;

  if (arm == -3 && blocks <= 8192) {
    if (full_row_threads == 128 && blocks <= 32) {
      const int tiny_grid = static_cast<int>((tokens * heads + 3) / 4);
      launch_local<128>(
          icp::prefill_candidates_tiny_warp_kernel, tiny_grid, stream, use_pdl,
          scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),
          forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),
          out.data_ptr<float>(), static_cast<int>(heads),
          static_cast<int>(blocks), rows, static_cast<int>(scan_block_begin),
          static_cast<int>(global_block_stride), use_pdl);
    } else if (full_row_threads == 512) {
      launch_local<icp::kFullRowThreads>(
          icp::local_candidates_full_row_kernel, rows, stream, use_pdl,
          scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),
          forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),
          out.data_ptr<float>(), static_cast<int>(heads),
          static_cast<int>(blocks), static_cast<int>(scan_block_begin),
          static_cast<int>(global_block_stride), use_pdl);
    } else {
#define SELECT_PREFILL_FULL_ROW(THREADS, CACHED, MERGE4)                         \
  launch_local<THREADS>(                                                       \
      icp::prefill_candidates_full_row_kernel<THREADS, CACHED, MERGE4>,          \
      rows, stream,                                                           \
      use_pdl, scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),            \
      forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),                 \
      out.data_ptr<float>(), static_cast<int>(heads),                          \
      static_cast<int>(blocks), static_cast<int>(scan_block_begin),             \
      static_cast<int>(global_block_stride), use_pdl)
      if (full_row_four_warp_finish && full_row_cached_items == 0) {
        SELECT_PREFILL_FULL_ROW(128, 0, true);
      } else if (full_row_four_warp_finish) {
        SELECT_PREFILL_FULL_ROW(128, 12, true);
      } else if (full_row_threads == 128 && full_row_cached_items == 0) {
        SELECT_PREFILL_FULL_ROW(128, 0, false);
      } else if (full_row_threads == 128) {
        SELECT_PREFILL_FULL_ROW(128, 12, false);
      } else if (full_row_cached_items == 0) {
        SELECT_PREFILL_FULL_ROW(256, 0, false);
      } else {
        SELECT_PREFILL_FULL_ROW(256, 6, false);
      }
#undef SELECT_PREFILL_FULL_ROW
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  // Larger full-row capacities retain the exact bounded prefill fallback.

#define SELECT_PREFILL(ITEMS)                                                  \
  select_partition_radix<ITEMS>(scores, nvalid, forced_col, active,            \
                                destination, heads, blocks, launch_partitions, \
                                grid, static_cast<int>(scan_block_begin),      \
                                static_cast<int>(global_block_stride),        \
                                use_pdl, /*short_row=*/true, stream)
  if (launch.partitions) {
    TORCH_CHECK(launch.items == icp::kPartitionedItems,
                "a partitioned prefill launch must take the rung that spans a "
                "partition");
    SELECT_PREFILL(icp::PartitionedRung<icp::kPartitionedItems>::items);
  } else if (launch.items == 1) {
    SELECT_PREFILL(1);
  } else if (launch.items == 2) {
    SELECT_PREFILL(2);
  } else if (launch.items == 4) {
    SELECT_PREFILL(4);
  } else if (launch.items == 8) {
    SELECT_PREFILL(8);
  } else if (launch.items == 16) {
    SELECT_PREFILL(16);
  } else {
    SELECT_PREFILL(32);
  }
#undef SELECT_PREFILL

  if (launch.partitions) {
    const int64_t candidate_count = launch.partitions * icp::kTopK + icp::kTopK;
#define COMBINE_PREFILL(ITEMS)                                                \
  launch_local(icp::combine_local_candidates_radix_kernel<ITEMS>, rows,       \
               stream, use_pdl, partials.data_ptr<float>(),                   \
               out.data_ptr<float>(),                                         \
               static_cast<int>(launch.partitions), use_pdl)
    if (candidate_count <= 128) {
      COMBINE_PREFILL(1);
    } else if (candidate_count <= 256) {
      COMBINE_PREFILL(2);
    } else if (candidate_count <= 512) {
      COMBINE_PREFILL(4);
    } else if (candidate_count <= 1024) {
      COMBINE_PREFILL(8);
    } else if (candidate_count <= 2048) {
      COMBINE_PREFILL(16);
    } else {
      COMBINE_PREFILL(32);
    }
#undef COMBINE_PREFILL
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// The attributes of the kernel `cap` WOULD launch at this capacity. The ladder
// is deliberately a second copy of the dispatch above: if the two ever drift,
// the gate's `last_launch_info() == selector_kernel_attributes(...)` assertion
// fires, which is the whole point of having it.
std::vector<int64_t> selector_kernel_attributes(int64_t cap, int64_t blocks) {
  cudaFuncAttributes a{};
  unsigned long long symbol = 0;
#define Q(SYM)                                                       \
  do {                                                               \
    symbol = reinterpret_cast<unsigned long long>((const void*)(SYM)); \
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, (const void*)(SYM)));   \
  } while (0)
  if (cap == -3 && blocks <= 8192) {
    Q(icp::local_candidates_full_row_kernel);
  } else if (cap > 0) {
    if (cap == 2) {
      Q(icp::local_candidates_capped_kernel<2>);
    } else if (cap == 4) {
      Q(icp::local_candidates_capped_kernel<4>);
    } else if (cap == 8) {
      Q(icp::local_candidates_capped_kernel<8>);
    } else {
      Q(icp::local_candidates_capped_kernel<16>);
    }
  } else if (cap < 0) {
    if (blocks <= 128) {
      Q(icp::local_candidates_radix_kernel<1>);
    } else if (blocks <= 256) {
      Q(icp::local_candidates_radix_kernel<2>);
    } else if (blocks <= 512) {
      Q(icp::local_candidates_radix_kernel<4>);
    } else if (blocks <= 1024) {
      Q(icp::local_candidates_radix_kernel<8>);
    } else if (blocks <= 2048) {
      Q(icp::local_candidates_radix_kernel<16>);
    } else {
      Q(icp::local_candidates_radix_kernel<32>);
    }
  } else {
    if (blocks <= 128) {
      Q(icp::local_candidates_kernel<1>);
    } else if (blocks <= 256) {
      Q(icp::local_candidates_kernel<2>);
    } else if (blocks <= 512) {
      Q(icp::local_candidates_kernel<4>);
    } else if (blocks <= 1024) {
      Q(icp::local_candidates_kernel<8>);
    } else if (blocks <= 2048) {
      Q(icp::local_candidates_kernel<16>);
    } else {
      Q(icp::local_candidates_kernel<32>);
    }
  }
#undef Q
  return {static_cast<int64_t>(symbol), a.numRegs,
          static_cast<int64_t>(a.sharedSizeBytes)};
}

// The attributes of the kernel the PREFILL entry would launch at this live
// extent and explicit arm. Full-row's extent must be the startup capacity,
// exactly as required by its native launch guard and the public caller.
std::vector<int64_t> prefill_kernel_attributes(int64_t live_blocks, int64_t arm,
                                             int64_t full_row_threads,
                                             int64_t full_row_cached_items,
                                             bool full_row_four_warp_finish) {
  check_prefill_full_row_config(arm, full_row_threads, full_row_cached_items,
                                full_row_four_warp_finish);
  TORCH_CHECK(live_blocks >= 0, "live_blocks must be non-negative");
  cudaFuncAttributes a{};
  unsigned long long symbol = 0;
#define QP(SYM)                                                        \
  do {                                                                 \
    symbol = reinterpret_cast<unsigned long long>((const void*)(SYM)); \
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, (const void*)(SYM)));     \
  } while (0)
  if (arm == -3 && live_blocks <= 8192) {
    if (full_row_threads == 128 && live_blocks <= 32) {
      QP(icp::prefill_candidates_tiny_warp_kernel);
    } else if (full_row_threads == 512) {
      QP(icp::local_candidates_full_row_kernel);
    } else if (full_row_four_warp_finish && full_row_cached_items == 0) {
      QP((icp::prefill_candidates_full_row_kernel<128, 0, true>));
    } else if (full_row_four_warp_finish) {
      QP((icp::prefill_candidates_full_row_kernel<128, 12, true>));
    } else if (full_row_threads == 128 && full_row_cached_items == 0) {
      QP((icp::prefill_candidates_full_row_kernel<128, 0>));
    } else if (full_row_threads == 128) {
      QP((icp::prefill_candidates_full_row_kernel<128, 12>));
    } else if (full_row_cached_items == 0) {
      QP((icp::prefill_candidates_full_row_kernel<256, 0>));
    } else {
      QP((icp::prefill_candidates_full_row_kernel<256, 6>));
    }
  } else if (live_blocks > icp::kLocalPartition) {
    QP(icp::local_candidates_radix_kernel<
        icp::PartitionedRung<icp::kPartitionedItems>::items>);
  } else if (live_blocks <= 128) {
    QP(icp::local_candidates_radix_kernel<1>);
  } else if (live_blocks <= 256) {
    QP(icp::local_candidates_radix_kernel<2>);
  } else if (live_blocks <= 512) {
    QP(icp::local_candidates_radix_kernel<4>);
  } else if (live_blocks <= 1024) {
    QP(icp::local_candidates_radix_kernel<8>);
  } else if (live_blocks <= 2048) {
    QP(icp::local_candidates_radix_kernel<16>);
  } else {
    QP(icp::local_candidates_radix_kernel<32>);
  }
#undef QP
  return {static_cast<int64_t>(symbol), a.numRegs,
          static_cast<int64_t>(a.sharedSizeBytes)};
}

void set_launch_recording(bool enabled) { g_record_launches = enabled; }

std::vector<int64_t> last_launch_info() {
  return {static_cast<int64_t>(g_last_launch.symbol), g_last_launch.regs,
          g_last_launch.smem, g_last_launch.grid, g_last_launch.count};
}

int64_t last_launch_threads() { return g_last_launch.threads; }

void select_local_candidates(at::Tensor scores, at::Tensor nvalid,
                             at::Tensor forced_col, at::Tensor active,
                             at::Tensor out, at::Tensor partials,
                             int64_t scan_block_begin, bool use_pdl,
                             int64_t global_block_stride) {
  select_local_candidates_impl(scores, nvalid, forced_col, active, out,
                               partials, scan_block_begin, use_pdl, /*cap=*/0,
                               global_block_stride);
}

void select_local_candidates_radix(at::Tensor scores, at::Tensor nvalid,
                                   at::Tensor forced_col, at::Tensor active,
                                   at::Tensor out, at::Tensor partials,
                                   int64_t scan_block_begin, bool use_pdl,
                                   int64_t global_block_stride) {
  select_local_candidates_impl(scores, nvalid, forced_col, active, out,
                               partials, scan_block_begin, use_pdl, /*cap=*/-1,
                               global_block_stride);
}

void select_local_candidates_radix_sr(at::Tensor scores, at::Tensor nvalid,
                                      at::Tensor forced_col, at::Tensor active,
                                      at::Tensor out, at::Tensor partials,
                                      int64_t scan_block_begin, bool use_pdl,
                                      int64_t global_block_stride) {
  select_local_candidates_impl(scores, nvalid, forced_col, active, out,
                               partials, scan_block_begin, use_pdl, /*cap=*/-2,
                               global_block_stride);
}

void select_local_candidates_capped(at::Tensor scores, at::Tensor nvalid,
                                    at::Tensor forced_col, at::Tensor active,
                                    at::Tensor out, at::Tensor partials,
                                    int64_t scan_block_begin, bool use_pdl,
                                    int64_t cap, int64_t global_block_stride) {
  TORCH_CHECK(cap > 0, "the capped producer needs a positive cap");
  select_local_candidates_impl(scores, nvalid, forced_col, active, out,
                               partials, scan_block_begin, use_pdl, cap,
                               global_block_stride);
}

void select_local_candidates_full_row(at::Tensor scores, at::Tensor nvalid,
                                      at::Tensor forced_col, at::Tensor active,
                                      at::Tensor out, at::Tensor partials,
                                      int64_t scan_block_begin, bool use_pdl,
                                      int64_t global_block_stride) {
  select_local_candidates_impl(scores, nvalid, forced_col, active, out,
                               partials, scan_block_begin, use_pdl, /*cap=*/-3,
                               global_block_stride);
}

void select_prefill_candidates(at::Tensor scores, at::Tensor nvalid,
                               at::Tensor forced_col, at::Tensor active,
                               at::Tensor out, at::Tensor partials,
                               int64_t scan_block_begin, bool use_pdl,
                               int64_t live_blocks,
                               int64_t global_block_stride, int64_t arm,
                               int64_t full_row_threads,
                               int64_t full_row_cached_items,
                               bool full_row_four_warp_finish) {
  select_prefill_candidates_impl(scores, nvalid, forced_col, active, out,
                                 partials, scan_block_begin, use_pdl,
                                 live_blocks, global_block_stride, arm,
                                 full_row_threads, full_row_cached_items,
                                 full_row_four_warp_finish);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("candidate_mapping_abi_version") = 2;
  m.attr("supported_global_block_strides") = pybind11::make_tuple(1, 2);
  m.def(
      "select_prefill_candidates", &select_prefill_candidates,
      "Prefill selection: arm=-2 is bounded radix (default); arm=-3 selects "
      "the whole-row kernel up to N=8192 and bounded radix above it. "
      "The public caller always passes startup capacity as live_blocks. "
      "The full-row arm requires that equality; the bounded arm also admits "
      "a smaller extent with device assertions against truncated rows. "
      "Retained output/scratch shapes are unchanged for both arms. Explicit "
      "full_row_threads/full_row_cached_items controls preserve the default "
      "512/3 and admit 128/0, 128/12, 256/0, 256/6 prefill specializations. "
      "full_row_four_warp_finish explicitly enables an exact four-warp merge "
      "for 65--128 candidates on the 128-thread specializations only.",
      pybind11::arg("scores"), pybind11::arg("nvalid"),
      pybind11::arg("forced_col"), pybind11::arg("active"),
      pybind11::arg("out"), pybind11::arg("partials"),
      pybind11::arg("scan_block_begin"), pybind11::arg("use_pdl"),
      pybind11::arg("live_blocks"),
      pybind11::arg("global_block_stride") = 1,
      pybind11::arg("arm") = -2,
      pybind11::arg("full_row_threads") = 512,
      pybind11::arg("full_row_cached_items") = 3,
      pybind11::arg("full_row_four_warp_finish") = false);
  m.def("prefill_kernel_attributes", &prefill_kernel_attributes,
        "[symbol_address, numRegs, sharedSizeBytes] of the select kernel the "
        "prefill entry would launch at this extent and arm.",
        pybind11::arg("live_blocks"), pybind11::arg("arm") = -2,
        pybind11::arg("full_row_threads") = 512,
        pybind11::arg("full_row_cached_items") = 3,
        pybind11::arg("full_row_four_warp_finish") = false);
  m.def(
      "select_local_candidates", &select_local_candidates,
      "Sort arm: dense scores and geometry to canonical local top-16 "
      "score/ID pairs, by a full cub::BlockRadixSort of every slot.",
      pybind11::arg("scores"), pybind11::arg("nvalid"),
      pybind11::arg("forced_col"), pybind11::arg("active"),
      pybind11::arg("out"), pybind11::arg("partials"),
      pybind11::arg("scan_block_begin"),
      pybind11::arg("use_pdl") = false,
      pybind11::arg("global_block_stride") = 1);
  m.def(
      "select_local_candidates_radix", &select_local_candidates_radix,
      "Radix arm: the same contract by keys-only radix SELECT instead of a "
      "full block sort. Scores are reconstructed from the canonical key, so "
      "-0.0 emits as +0.0 (CONTRACT C5's mandated flush).",
      pybind11::arg("scores"), pybind11::arg("nvalid"),
      pybind11::arg("forced_col"), pybind11::arg("active"),
      pybind11::arg("out"), pybind11::arg("partials"),
      pybind11::arg("scan_block_begin"),
      pybind11::arg("use_pdl") = false,
      pybind11::arg("global_block_stride") = 1);
  m.def(
      "select_local_candidates_radix_sr", &select_local_candidates_radix_sr,
      "The SHIPPING arm: the same radix select with `wanted` bounded by the "
      "live element count, so a row shorter than 16 exits after one digit "
      "instead of walking all 8 over the zero padding. Same contract, same "
      "output.",
      pybind11::arg("scores"), pybind11::arg("nvalid"),
      pybind11::arg("forced_col"), pybind11::arg("active"),
      pybind11::arg("out"), pybind11::arg("partials"),
      pybind11::arg("scan_block_begin"),
      pybind11::arg("use_pdl") = false,
      pybind11::arg("global_block_stride") = 1);
  m.def(
      "select_local_candidates_capped", &select_local_candidates_capped,
      "Radix select with the register depth capped and the row streamed, so "
      "cost stops tracking the capture-time capacity.",
      pybind11::arg("scores"), pybind11::arg("nvalid"),
      pybind11::arg("forced_col"), pybind11::arg("active"),
      pybind11::arg("out"), pybind11::arg("partials"),
      pybind11::arg("scan_block_begin"),
      pybind11::arg("use_pdl"), pybind11::arg("cap"),
      pybind11::arg("global_block_stride") = 1);
  m.def("select_local_candidates_full_row", &select_local_candidates_full_row,
        "One CTA per decode row, exact canonical score/ID selection with device bounds.",
        pybind11::arg("scores"), pybind11::arg("nvalid"),
        pybind11::arg("forced_col"), pybind11::arg("active"),
        pybind11::arg("out"), pybind11::arg("partials"),
        pybind11::arg("scan_block_begin"), pybind11::arg("use_pdl") = false,
        pybind11::arg("global_block_stride") = 1);
  m.def("selector_kernel_attributes", &selector_kernel_attributes,
        "[symbol_address, numRegs, sharedSizeBytes] of the select kernel this "
        "(cap, blocks) would launch. cap: 0=sort, <0=radix (-1 unbounded, -2 "
        "live-count bounded -- one kernel, so both report the same symbol), "
        ">0=capped.",
        pybind11::arg("cap"), pybind11::arg("blocks"));
  m.def("set_launch_recording", &set_launch_recording,
        "Record the launched select kernel's attributes. Off by default: the "
        "query is a host call and would land in the eager timing path.",
        pybind11::arg("enabled"));
  m.def("last_launch_info", &last_launch_info,
        "[symbol_address, numRegs, sharedSizeBytes, gridDim.x, "
        "launches_recorded] of the last recorded select launch.");
  m.def("last_launch_threads", &last_launch_threads,
        "Actual blockDim.x of the last recorded select launch; -1 before any "
        "launch is recorded. Existing last_launch_info keeps its five fields.");
}
