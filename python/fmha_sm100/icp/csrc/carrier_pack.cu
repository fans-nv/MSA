// Native C4 send-carrier pack: query-major [Q, W*Hl, 16, 2] -> destination-major
// [W, Q, Hl, 16, 2]. Replaces the ATen permute + copy_ on the NCCL route.
// Each (q, d, hl) record list is 128 contiguous bytes in both layouts, so the
// copy moves 16-byte words; threads walk the DESTINATION so stores coalesce.
// The words are copied opaquely: score bits and ids are never converted.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kVecPerRow = 8;  // 16 records * 2 words * 4 B / 16 B

__global__ __launch_bounds__(kThreads) void carrier_pack_kernel(
    const int4* __restrict__ src, int4* __restrict__ dst, int Q, int W, int Hl,
    long total) {
  const long i = (long)blockIdx.x * kThreads + threadIdx.x;
  if (i >= total) return;
  const long row = i / kVecPerRow;
  const int v = (int)(i - row * kVecPerRow);
  const long per_dest = (long)Q * Hl;
  const int d = (int)(row / per_dest);
  const long rem = row - (long)d * per_dest;
  const long q = rem / Hl;
  const int hl = (int)(rem - q * Hl);
  const long src_row = (q * W + d) * Hl + hl;
  dst[i] = src[src_row * kVecPerRow + v];
}

}  // namespace

void pack_send_carrier(at::Tensor local_cand, at::Tensor out, int64_t world) {
  TORCH_CHECK(local_cand.is_cuda() && out.is_cuda(), "pack: tensors must be CUDA");
  TORCH_CHECK(local_cand.device() == out.device(), "pack: device mismatch");
  TORCH_CHECK(local_cand.element_size() == 4 &&
                  (local_cand.scalar_type() == at::kInt ||
                   local_cand.scalar_type() == at::kFloat),
              "pack: carrier words are 4-byte int32/fp32 (bitcast)");
  TORCH_CHECK(out.scalar_type() == at::kInt, "pack: out must be int32");
  TORCH_CHECK(local_cand.is_contiguous() && out.is_contiguous(),
              "pack: tensors must be contiguous");
  TORCH_CHECK(local_cand.dim() == 4 && local_cand.size(2) == 16 &&
                  local_cand.size(3) == 2,
              "pack: local_cand must be [Q, H_group, 16, 2]");
  TORCH_CHECK(world >= 1, "pack: world must be >= 1");
  const int Q = (int)local_cand.size(0);
  const int Hg = (int)local_cand.size(1);
  TORCH_CHECK(Hg % world == 0, "pack: H_group ", Hg, " not divisible by W ", world);
  const int Hl = Hg / (int)world;
  TORCH_CHECK(out.dim() == 5 && out.size(0) == world && out.size(1) == Q &&
                  out.size(2) == Hl && out.size(3) == 16 && out.size(4) == 2,
              "pack: out must be [W, Q, H_local, 16, 2]");
  const long total = (long)world * Q * Hl * kVecPerRow;
  // Empty tensors report data_ptr() == nullptr, so the alias check below would refuse Q == 0.
  if (total == 0) return;
  TORCH_CHECK(local_cand.data_ptr() != out.data_ptr(), "pack: out aliases input");
  TORCH_CHECK((reinterpret_cast<uintptr_t>(local_cand.data_ptr()) % 16) == 0 &&
                  (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16) == 0,
              "pack: buffers must be 16-byte aligned");
  const c10::cuda::CUDAGuard guard(local_cand.device());
  const long blocks = (total + kThreads - 1) / kThreads;
  carrier_pack_kernel<<<blocks, kThreads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const int4*>(local_cand.data_ptr()),
      reinterpret_cast<int4*>(out.data_ptr()), Q, (int)world, Hl, total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_send_carrier", &pack_send_carrier,
        "C4 query-major -> destination-major send carrier (native, no ATen)",
        pybind11::arg("local_cand"), pybind11::arg("out"), pybind11::arg("world"));
}
