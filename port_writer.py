"""Reproduce the bounded writer port from the two recorded input commits.

Run once against the untouched writer paths; unrelated worktree files are not
read or modified. The result still requires the recorded review/test gates.
"""

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DST = ROOT / "worktrees/vllm-icp-public"
REF = ROOT / "worktrees/vllm-msa-upstream"
PATH = "csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu"
BASE = "242e4213fc9845ff6fe607af1aee626fd8acc990"


def original(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=DST, text=True)


def replace(text, old, new):
    assert text.count(old) == 1, (old[:160], text.count(old))
    return text.replace(old, new)


def between(text, begin, end):
    start = text.index(begin)
    return text[start : text.index(end, start)]


def function(text, name, template=True):
    name_at = text.index(name + "(")
    start = text.rfind("template <", 0, name_at) if template else text.rfind("__device__", 0, name_at)
    masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', lambda m: " " * len(m[0]), text, flags=re.S)
    opening = masked.index("{", name_at)
    depth = 1
    end = opening + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return text[start:end]


src = original(PATH)
ref = (REF / PATH).read_text()
(DST / "csrc/libtorch_stable/fused_minimax_m3_icp.cuh").write_bytes(
    (REF / "csrc/libtorch_stable/fused_minimax_m3_icp.cuh").read_bytes()
)
src = replace(src, "namespace vllm {\nnamespace minimax_m3_fused_ops {", """#if !defined(USE_ROCM) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  #define VLLM_MINIMAX_M3_NVFP4
#endif

#include "fused_minimax_m3_icp.cuh"

namespace vllm {
namespace minimax_m3_fused_ops {""")
src = replace(src, function(src, "normAndRope"), function(ref, "normAndRope"))
src = replace(src, function(src, "storeElemsFp8", template=False), function(ref, "storeElemsFp8"))
src = replace(src, function(src, "storeScaledQElemsFp8"), function(ref, "storeScaledQElemsFp8"))
helpers = "#ifdef VLLM_MINIMAX_M3_NVFP4\n" + function(ref, "nvfp4Reciprocal", template=False) + "\n\n" + function(ref, "quantizeNvfp4Quad") + "\n#endif\n\n"
store = function(src, "storeNvfp4CacheElems")
address_at = store.index("  uint8_t* const slot =")
quant_at = store.index("  // Quantize the model-dtype")
quant = store[quant_at:address_at]
quant = replace(quant, "  uint8_t sf_byte;\n", "")
quant = replace(quant, "  uint16_t packed;\n", "")
new_store = store[:quant_at].replace("template <typename scalar_t>", "template <typename scalar_t, bool kIcp = false>") + """  uint8_t sf_byte;
  uint16_t packed;
  if constexpr (kIcp) {
#ifdef VLLM_MINIMAX_M3_NVFP4
    // Retain ICP's qualified quantization, sharing the public slot stores.
    packed = quantizeNvfp4Quad<scalar_t>(
        elems, nvfp4Reciprocal(scale), sf_byte);
#endif
  } else {
""" + quant + "  }\n\n" + store[address_at:]
src = replace(src, store, helpers + new_store)

old_kernel = function(src, "fusedMiniMaxM3QNormRopeKVInsertKernel")
body = old_kernel
body = replace(body, "bool kProcessIndex, bool kFp8Idx>", "bool kProcessIndex, bool kFp8Idx, bool kIcp = false,\n          bool kFlatRow = false, bool kWriteMeta = false>")
body = replace(body, "__global__ void fusedMiniMaxM3QNormRopeKVInsertKernel(", "__device__ __forceinline__ void fusedMiniMaxM3Body(")
body = replace(body, "float const* __restrict__ kv_v_scale) {", "float const* __restrict__ kv_v_scale, IcpWriterParams const icp) {")
metadata = between(ref, "    if constexpr (kIcp && kWriteMeta)", " else if constexpr (kWriteMeta)")
body = replace(body, "    int const warpsPerBlock =", "    // Head CTAs publish ICP live metadata and ABI-3 work in this grid.\n" + metadata + "\n    int const warpsPerBlock =")
body = replace(body, "    int const globalWarpIdx = blockIdx.x * warpsPerBlock + (threadIdx.x / 32);", """    int const metadata_blocks =
        kWriteMeta ? icp.metadata.meta_blocks + icp.metadata.plan.blocks : 0;
    int const globalWarpIdx =
        (blockIdx.x - static_cast<unsigned>(metadata_blocks)) * warpsPerBlock +
        (threadIdx.x / 32);""")
body = replace(body, """    unsigned const warp_u = globalWarpIdx, slots_u = slots_per_token;
    int const tokenIdx = warp_u / slots_u;
    int const slot = warp_u % slots_u;""", """    int tokenIdx, slot;
    if constexpr (kIcp) {
      tokenIdx = globalWarpIdx / slots_per_token;
      slot = globalWarpIdx % slots_per_token;
    } else {
      unsigned const warp_u = globalWarpIdx, slots_u = slots_per_token;
      tokenIdx = warp_u / slots_u;
      slot = warp_u % slots_u;
    }""")
row_begin = body.index("    if (isQ) {")
row_end = body.index("    // NVFP4: the per-layer", row_begin)
rows = body[row_begin:row_end]
rows = replace(rows, "    scalar_t* store_ptr = row_ptr;", "    store_ptr = row_ptr;")
rows = replace(rows, "} else if (isQ && q_fp8_out != nullptr) {", "} else if (!kIcp && isQ && q_fp8_out != nullptr) {")
flat = between(ref, "    scalar_t* store_ptr = nullptr;\n    if constexpr (kFlatRow)", "      if (isQ) {\n        row_ptr =")
body = body[:row_begin] + flat + rows + "    }\n\n" + body[row_end:]
body = replace(body, "normAndRope<scalar_t>(", "normAndRope<scalar_t, kIcp>(")
stores_at = body.index("      if constexpr (kFp8Idx) {")
icp_stores = between(ref, "    if constexpr (kIcp) {\n      if (isQ)", " else if (!isV) {")
body = body[:stores_at] + "    }\n" + icp_stores + " else if (!isV) {\n" + body[stores_at:]
body = replace(body, "storeNvfp4CacheElems<scalar_t>(", "storeNvfp4CacheElems<scalar_t, kIcp>(")
signature = old_kernel[:old_kernel.index(") {") + 1]
ordinary = signature + """ {
  fusedMiniMaxM3Body<scalar_t, cache_t, kv_dt, kNvfp4, out_idx_t, kHasIndex,
                    kInsertKV, kProcessIndex, kFp8Idx>(
      qkv, q_out, q_fp8_out, index_q_out, q_norm_w, k_norm_w, iq_norm_w,
      ik_norm_w, cos_sin_cache, positions, slot_mapping, index_slot_mapping,
      kv_cache, index_cache, eps, q_fp8_inv_scale, rotary_dim, num_tokens, nq,
      nkv, niq, block_size, kv_s_block, kv_s_head, kv_s_token, kv_s_dim,
      kv_k_scale, kv_v_scale, IcpWriterParams{});
}"""
icp_kernel = function(ref, "fusedMiniMaxM3IcpKernel")
icp_kernel = replace(icp_kernel, """fusedMiniMaxM3Body<scalar_t, uint8_t, Fp8KVCacheDataType::kAuto, uint8_t,
                     true, kInsertKV, true, true, kInsertKV, kWriteMeta, true,
                     kFlatRow>""", """fusedMiniMaxM3Body<scalar_t, uint8_t, Fp8KVCacheDataType::kAuto, kInsertKV,
                    uint8_t, true, kInsertKV, true, true, true, kFlatRow,
                    kWriteMeta>""")
icp_kernel = replace(icp_kernel, "      StepMetaParams{}, false, icp);", "      icp);")
src = replace(src, old_kernel, body + "\n\n" + ordinary + "\n\n#ifdef VLLM_MINIMAX_M3_NVFP4\n// Keep the qualified ICP occupancy contract on its opt-in entry only.\n" + icp_kernel + "\n#endif")
src = replace(src, "float const* kv_v_scale, cudaStream_t stream) {", "float const* kv_v_scale, cudaStream_t stream,\n    bool const enable_pdl, IcpWriterParams const& icp) {")
src = replace(src, """  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);""", """  int const main_grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  int const grid = main_grid + icp.metadata.meta_blocks + icp.metadata.plan.blocks;""")
src = replace(src, "config.numAttrs = (sm_version >= 90) ? 1 : 0;", "config.numAttrs = (enable_pdl && sm_version >= 90) ? 1 : 0;")
launch = between(ref, "  #ifdef VLLM_MINIMAX_M3_NVFP4\n  if (icp.world_size", "  #define LAUNCH_M")
launch = re.sub(r"\bk_scale\b", "kv_k_scale", launch)
launch = re.sub(r"\bv_scale\b", "kv_v_scale", launch)
src = replace(src, "  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)", launch + "  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)") if src.count("  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)") == 1 else src.replace("  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)", launch + "  #define LAUNCH(HAS_INDEX, INSERT, PROCESS_INDEX, FP8, OUT_T)", 1)
src = replace(src, "      kv_v_scale_ptr, stream)", "      kv_v_scale_ptr, stream, enable_pdl, icp)")
native_tail = "    " + between(ref, "int64_t index_block_tokens,", ") {\n  STD_TORCH_CHECK(qkv.is_cuda()")
src = replace(src, "    std::optional<torch::stable::Tensor> kv_v_scale) {", "    std::optional<torch::stable::Tensor> kv_v_scale,\n" + native_tail + ") {")
validation = between(ref, "  vllm::minimax_m3_fused_ops::IcpWriterParams icp{};", "  // Per-step decode metadata")
validation = replace(validation, "    STD_TORCH_CHECK(!write_step_meta && !pdl_early_trigger,\n                    \"ICP cannot use legacy step metadata or early PDL\");\n", "")
validation = re.sub(r"\bnvfp4\b", "nvfp4_kv", validation)
validation = re.sub(r"\bk_scale\b", "kv_k_scale", validation)
validation = re.sub(r"\bv_scale\b", "kv_v_scale", validation)
# The public writer already checks slot payload geometry; ICP additionally
# requires a common device and room for all slots in the compound parent page.
validation = replace(validation, "      auto const& idx = *index_cache;", """      STD_TORCH_CHECK(
          same_device(*kv_cache) && same_device(*kv_k_scale) &&
              same_device(*kv_v_scale) && kv_s_block >= 2 * nkv * block_size * 72,
          "ICP main cache/scales must be on the qkv device with full head slots");
      auto const& idx = *index_cache;""")
live = between(ref, "  if (write_icp_metadata) {", "  const torch::stable::accelerator::DeviceGuard device_guard")
live = re.sub(r"\bnvfp4\b", "nvfp4_kv", live)
live = live.replace("vllm::minimax_m3_fused_ops::kBlockSize", "vllm::minimax_m3_fused_ops::kIcpPlanThreads")
src = replace(src, "  STD_TORCH_CHECK(!nvfp4_kv || insert_kv,", validation + live + "  STD_TORCH_CHECK(!nvfp4_kv || insert_kv || use_icp,")
src = replace(src, "      nvfp4_kv ? kv_k_scale->const_data_ptr<float>() : nullptr;", "      nvfp4_kv && insert_kv ? kv_k_scale->const_data_ptr<float>() : nullptr;")
src = replace(src, "      nvfp4_kv ? kv_v_scale->const_data_ptr<float>() : nullptr;", "      nvfp4_kv && insert_kv ? kv_v_scale->const_data_ptr<float>() : nullptr;")
src = replace(src, "  if (nvfp4_kv) {\n#ifndef USE_ROCM\n    VLLM_STABLE", """  if (use_icp) {
    STD_TORCH_CHECK(vllm::minimax_m3_fused_ops::getSMVersion() >= 100,
                    "ICP writer requires SM100 or newer");
  }

  if (nvfp4_kv) {
#ifndef USE_ROCM
    VLLM_STABLE""")
(DST / PATH).write_text(src)

# Add declarations/schema without changing any existing public prefix.
path = "csrc/libtorch_stable/ops.h"
text = original(path)
decl_tail = between((REF / path).read_text(), "    int64_t index_block_tokens,", ");\n\n#ifdef VLLM_ENABLE_FUSED_KDA_DECODE")
text = replace(text, "    std::optional<torch::stable::Tensor> kv_v_scale);", "    std::optional<torch::stable::Tensor> kv_v_scale,\n" + decl_tail + ");")
(DST / path).write_text(text)
path = "csrc/libtorch_stable/torch_bindings.cpp"
text = original(path)
schema_tail = between((REF / path).read_text(), '      "int index_block_tokens=0, "', '\n\n#ifdef VLLM_ENABLE_FUSED_KDA_DECODE')
text = replace(text, '      "Tensor? kv_k_scale=None, Tensor? kv_v_scale=None) -> ()");', '      "Tensor? kv_k_scale=None, Tensor? kv_v_scale=None, "\n' + schema_tail)
(DST / path).write_text(text)

# Python keeps the original positional call, with only an optional ICP tail.
path = "vllm/_custom_ops.py"
text = original(path)
pyref = (REF / path).read_text()
protocol = between(pyref, "class MiniMaxM3IcpDevicePlan(Protocol):", "def fused_minimax_m3_qknorm_rope_kv_insert(")
begin = text.index("def fused_minimax_m3_qknorm_rope_kv_insert(")
end = text.index("\ndef fused_kda_decode(", begin)
wrapper = text[begin:end]
keyword_tail = between(pyref[pyref.index("def fused_minimax_m3_qknorm"):], "    *,\n    index_block_tokens", ") -> None:\n")
wrapper = replace(wrapper, "    kv_v_scale: torch.Tensor | None = None,\n", "    kv_v_scale: torch.Tensor | None = None,\n" + keyword_tail)
icp_doc = """
    ``index_world_size=2`` enables TP2/Q32/KV2/I4/P128/R64 ICP with
    ``enable_pdl=False`` and unit Q scale. Main KV retains the per-head slot
    layout above, including the parent page stride. ICP preserves its
    reciprocal-multiply NVFP4 quantization and direct FP32-to-FP8 index Q;
    ordinary calls retain model-dtype-rounded index Q and public NVFP4 math.
    Main Q is also stored in model dtype (in ``q_out`` when supplied).
    Rank ownership affects only the FP8 ``[pages,64,128]`` index fragment.

    ``write_icp_metadata=True`` adds live metadata and optional ABI-3 plan
    CTAs to the same producer grid. ``icp_device_plan`` is preallocated; its
    generation advances only after a successful native launch.
"""
wrapper = replace(wrapper, '    """\n    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(', icp_doc + '    """\n    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(')
dispatch_prepare = between(pyref, "    # Keep the ordinary call compatible", "    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(")
wrapper = replace(wrapper, "    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(", dispatch_prepare + "    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(")
wrapper = replace(wrapper, "        kv_v_scale,\n    )", "        kv_v_scale,\n        **icp_kwargs,\n    )")
after = between(pyref[pyref.index("    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert("):], "    if icp_device_plan is not None:", "\n\ndef fused_kda_decode(")
wrapper = wrapper.rstrip() + "\n" + after + "\n\n"
text = text[:begin] + protocol + wrapper + text[end:]
text = replace(text, "from typing import TYPE_CHECKING, Literal\n", "from typing import TYPE_CHECKING, Literal, Protocol\n")
(DST / path).write_text(text)
