// TEST ONLY: the selector's negative-control build.
//
// Built into its own extension with -DICP_RS_NEGATIVE_CONTROL=1. Never part of
// the shipping selector, which compiles local_candidates_rs.cuh WITHOUT that
// macro, under which every ICP_RS_CTL(n) is the literal `false` and the extra
// kernel parameter does not exist. The controls themselves are listed at the
// macro definition in local_candidates_rs.cuh.
#ifndef ICP_RS_NEGATIVE_CONTROL
#error "this file exists only to build the negative controls"
#endif

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "local_candidates_rs.cuh"

namespace {

template <typename Kernel, typename... Args>
void launch_local(Kernel kernel, int grid, cudaStream_t stream, bool use_pdl,
                  Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(icp::kLocalThreads);
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

}  // namespace

void select_local_candidates_control(at::Tensor scores, at::Tensor nvalid,
                                     at::Tensor forced_col, at::Tensor active,
                                     at::Tensor out, at::Tensor partials,
                                     int64_t scan_block_begin, bool use_pdl,
                                     int64_t control, int64_t cap,
                                     int64_t global_block_stride) {
  TORCH_CHECK(global_block_stride == 1 || global_block_stride == 2,
              "global_block_stride must be 1 or 2");
  const int64_t tokens = scores.size(0);
  const int64_t heads = scores.size(1);
  const int64_t blocks = scores.size(2);
  const int64_t partitions =
      blocks > icp::kLocalPartition
          ? (blocks + icp::kLocalPartition - 1) / icp::kLocalPartition
          : 0;
  const int64_t launch_partitions = std::max<int64_t>(partitions, 1);
  const c10::cuda::CUDAGuard guard(scores.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  const int rows = static_cast<int>(tokens * heads);
  const int grid = static_cast<int>(rows * launch_partitions);
  auto& destination = partitions ? partials : out;
  const int ctl = static_cast<int>(control);

  // One extra branch for the capped arm keeps the gate a single build.
#define SELECT_CAPPED(ITEMS)                                                 \
  launch_local(icp::local_candidates_capped_kernel<ITEMS>, grid, stream,     \
               use_pdl, scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),\
               forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),      \
               destination.data_ptr<float>(),                                \
               static_cast<int>(heads), static_cast<int>(blocks),            \
               static_cast<int>(launch_partitions),                          \
               static_cast<int>(scan_block_begin),                            \
               static_cast<int>(global_block_stride), use_pdl, ctl)
  // The SORT arm. refined-icp-v1 hoisted the ICP_RS_CTL macros into
  // `local_candidates.cuh` precisely so that this arm is gated at all -- before
  // that the macros lived in `local_candidates_rs.cuh` and the sort arm's gate
  // was vacuous by construction. Driving it here is what makes that hoist real
  // rather than inert: controls 1, 4, 7 and 8 apply to both arms, and `cap == 0`
  // is the sort arm in the shipping launcher, so the two dispatches agree.
#define SELECT_SORT(ITEMS)                                                   \
  launch_local(icp::local_candidates_kernel<ITEMS>, grid, stream,            \
               use_pdl, scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),\
               forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),      \
               destination.data_ptr<float>(),                                \
               static_cast<int>(heads), static_cast<int>(blocks),            \
               static_cast<int>(launch_partitions),                          \
               static_cast<int>(scan_block_begin),                            \
               static_cast<int>(global_block_stride), use_pdl, ctl)
  // `short_row` is fixed true: the shipped entry is the BOUNDED radix select,
  // so a control driven with it false would gate a path nothing takes.
#define SELECT(ITEMS)                                                        \
  launch_local(icp::local_candidates_radix_kernel<ITEMS>, grid, stream,      \
               use_pdl, scores.data_ptr<float>(), nvalid.data_ptr<int32_t>(),\
               forced_col.data_ptr<int32_t>(), active.data_ptr<bool>(),      \
               destination.data_ptr<float>(),                                \
               static_cast<int>(heads), static_cast<int>(blocks),            \
               static_cast<int>(launch_partitions),                          \
               static_cast<int>(scan_block_begin),                            \
               static_cast<int>(global_block_stride), use_pdl,                \
               /*short_row=*/true, ctl)
#define SELECT_LADDER(MACRO)                                                 \
  do {                                                                       \
    if (blocks <= 128) {                                                     \
      MACRO(1);                                                              \
    } else if (blocks <= 256) {                                              \
      MACRO(2);                                                              \
    } else if (blocks <= 512) {                                              \
      MACRO(4);                                                              \
    } else if (blocks <= 1024) {                                             \
      MACRO(8);                                                              \
    } else if (blocks <= 2048) {                                             \
      MACRO(16);                                                             \
    } else {                                                                 \
      MACRO(32);                                                             \
    }                                                                        \
  } while (0)
  if (cap > 0) {
    if (cap == 2) {
      SELECT_CAPPED(2);
    } else if (cap == 4) {
      SELECT_CAPPED(4);
    } else if (cap == 8) {
      SELECT_CAPPED(8);
    } else {
      SELECT_CAPPED(16);
    }
  } else if (cap == 0) {
    SELECT_LADDER(SELECT_SORT);
  } else {
    SELECT_LADDER(SELECT);
  }
#undef SELECT_LADDER
#undef SELECT
#undef SELECT_SORT
#undef SELECT_CAPPED

  if (partitions) {
    const int64_t candidate_count = partitions * icp::kTopK + icp::kTopK;
    // The combine is the radix one for every arm, as in the shipping launcher
    // for cap != 0; at cap == 0 the shipping launcher uses the sort combine.
    // Control 4 perturbs the same site in both, so gating one is enough and
    // gating the wrong one would be worse than gating neither -- stated so the
    // asymmetry with the select dispatch above is deliberate rather than a slip.
#define COMBINE(ITEMS)                                                        \
  launch_local(icp::combine_local_candidates_radix_kernel<ITEMS>, rows,       \
               stream, use_pdl, partials.data_ptr<float>(),                   \
               out.data_ptr<float>(), static_cast<int>(partitions), use_pdl,  \
               ctl)
    if (candidate_count <= 128) {
      COMBINE(1);
    } else if (candidate_count <= 256) {
      COMBINE(2);
    } else if (candidate_count <= 512) {
      COMBINE(4);
    } else if (candidate_count <= 1024) {
      COMBINE(8);
    } else if (candidate_count <= 2048) {
      COMBINE(16);
    } else {
      COMBINE(32);
    }
#undef COMBINE
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("select_local_candidates_control", &select_local_candidates_control,
        "TEST ONLY: the selector with one deliberately perturbed site. `cap` "
        "picks the arm exactly as the shipping launcher does: 0 sort, -1/-2 "
        "radix, >0 capped.",
        pybind11::arg("scores"), pybind11::arg("nvalid"),
        pybind11::arg("forced_col"), pybind11::arg("active"),
        pybind11::arg("out"), pybind11::arg("partials"),
        pybind11::arg("scan_block_begin"), pybind11::arg("use_pdl"),
        pybind11::arg("control"), pybind11::arg("cap") = 0,
        pybind11::arg("global_block_stride") = 1);
}
