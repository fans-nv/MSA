/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <climits>

#include "gmem_bounds_check.h"
#include "cutlass_utils.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "fmha_common.hpp"
#include "gpu_trace.h"
#include "fmha_fusion.hpp"
#include "k2_nvfp4_kv.hpp"

#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
#include <cuda.h>
#endif


#if MSA_NVFP4_KV_MODE >= 2
#define K2_TAG_SEQ (pipeline_kv_producer_state.count())

#define K2_TAG_ISV(x) (x)

#define K2_TAG_STAGE(is_v_, page_)                                                       \
  storage.stage_tag[pipeline_kv_producer_state.index()] =                                \
      k2::StageTag::make(K2_TAG_ISV(is_v_), K2_TAG_SEQ,                                  \
                         static_cast<uint32_t>(page_))
#else
#define K2_TAG_STAGE(is_v_, page_) do { } while (0)
#endif

GPU_TRACE_SCOPE_DEC(LOAD_Q);
GPU_TRACE_SCOPE_DEC(LOAD_Q_WAIT);
GPU_TRACE_SCOPE_DEC(LOAD_K);
GPU_TRACE_SCOPE_DEC(LOAD_V);

namespace cutlass::fmha::collective {

using namespace cute;

template <typename T>
struct MaskPackFactor { static constexpr int value = 1; };
template <int P>
struct MaskPackFactor<PackedCausalMask<P>> { static constexpr int value = P; };

template <class Element, class CollectiveMmaQK, class CollectiveMmaPV, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class TensorStorage, class PipelineQ,
          class PipelineKV, class Mask, class TileShape, int KVPageSize = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Off>
struct Sm100FmhaLoadTmaWarpspecialized {
  // P0: OnlyScore mode never reads V — skip every V slot in the KV pipeline.
  // mma() gates the matching wait_V/release_V under `kNeedOutput` so producer
  // (loader) and consumer (mma) stay step-locked on the pipeline state.
  static constexpr bool kNeedV = (kSparseAttnMode != SparseAttnMode::OnlyScore
                               && kSparseAttnMode != SparseAttnMode::OnlyScoreIcp);

  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;

  static constexpr int kNumKvSub = get<1>(TileShape{}) / get<1>(TileShapeQK{});

  static constexpr int kKvEventsPerTile = (kNeedV ? 2 : 1) * kNumKvSub;

  static_assert(!k2::kDequantProducesKV || (KVPageSize > 0),
                "MSA_NVFP4_KV_MODE >= 2 requires the paged KV path (KVPageSize > 0); the "
                "non-paged load lambdas do not write a k2::StageTag.");

  // ---- refined-icp-v1: THE LIVE FRAGMENT BOUND (DIRECT_TABLE_CONTRACT §4) ----------
  // How many logical blocks of this request THIS RANK actually holds index rows for,
  // derived from the EXACT device KV length `L` (get<1>(problem_shape), which under ICP
  // is the GLOBAL length -- see the mainloop's icp_local_blocks comment):
  //
  //     local_blocks(L, r) = floor(L / B) + ((L mod B) > r * R)
  //
  // with R = this rank's row count inside a compound page and B = R * W the compound
  // page's token count (128 in production; both are DERIVED here rather than written as
  // literals so a variant built at another page granularity cannot silently disagree).
  // The comparison is STRICTLY greater-than: at L == B*q + r*R rank r's first row of
  // block q is the token at global position B*q + r*R, which is position L, i.e. one
  // past the last live token.  `>=` would hand the loader a block whose index rows the
  // writer never wrote (fused_indexer_nvfp4_kv_write.cu:665-669 owns position p iff
  // (p % B) / R == r, so it writes that row only once p reaches L).
  //
  // WHY THIS IS NOT A SECOND, COMPETING COUNT.  It is algebraically identical to the
  // block-cyclic form the trip counts use, `ceil((ceil(L/R) - r) / W)`
  // (`compute_effective_end` below, and the mainloop's `icp_local_blocks`): writing
  // L = qB + s with 0 <= s < B gives ceil(L/R) = qW + ceil(s/R), and since
  // ceil(s/R) - r lies in [-(W-1), W] the outer ceil contributes 1 exactly when
  // ceil(s/R) > r, i.e. when s > r*R.  Proven for every L >= 0 and r in [0, W).  The
  // two forms may therefore never disagree, which is what lets the kernel-wide
  // empty-work decision (kernel header `is_empty_work`, which uses the OTHER form)
  // dominate the page clamp below.
  //
  // `kv_len` IS `segment_lens[batch_idx]`, a live device read (`apply_variable_length`,
  // fmha_fusion.hpp:369-378).  It is deliberately NOT the host length vector the work
  // decomposition was sized from: the caller may overwrite the plan's `kv_segment_lens`
  // IN PLACE with tighter, exact lengths after the plan is built -- vLLM's
  // `seq_lens_cpu_upper_bound` still counts REJECTED Eagle3 drafts, so as a bound it is
  // optimistic, and at every compound-page boundary a rejected draft crosses it would
  // buy one column past the request's live end.  Every consumer of `get<1>` tightens
  // together with this one: `compute_effective_end`, the mainloop's
  // `get_full_trip_count` / `icp_local_blocks`, `is_empty_work`, and the W5 validity
  // pass all read the same value, so the loader and the MMA stay step-locked and the
  // advertised wave stays consistent with what was scored.  Nothing here may be
  // re-derived from the host list.
  //
  // PRECONDITION: W >= 2 and KVPageSize > 0, i.e. an ICP variant with a direct table.
  // That is the only place this is called from, and the host refuses the call
  // otherwise (`page_size * W == 128` in the run binding), so it is not re-checked
  // here -- a device-side guard could only return a wrong-but-quiet 0.
  CUTLASS_DEVICE static int icp_local_blocks_exact(int kv_len, int packed_rank_c) {
    int W = packed_rank_c & 15;
    int r = packed_rank_c >> 4;
    int R = KVPageSize;      // rows of the compound page this rank owns (ABI P1/P2)
    int B = R * W;           // the compound page's token extent
    int full = kv_len / B;
    int rem = kv_len - full * B;
    return full + ((rem > r * R) ? 1 : 0);
  }

  // K2: lifted VERBATIM out of run() so that the Dequant warpgroup and the Load warp
  // compute `mask_tile_count` from ONE piece of code.  DESIGN §4.2 names divergence here
  // -- through get_effective_trip_count, kEnablePaddingSkip, or the prologue/epilogue
  // asymmetry -- as the hang risk.  Two copies of this arithmetic is exactly how that
  // happens; there is now one.
  // ParamsT is a template parameter only because `Params` is declared further down this
  // class; it is always Load::Params.
  template <class BlkCoord, class ProblemShape, class ParamsT>
  CUTLASS_DEVICE static int compute_effective_end(BlkCoord const& blk_coord,
                                                  ProblemShape const& problem_shape,
                                                  ParamsT const& params, int kv_tile_end) {
    int kv_len = get<1>(problem_shape);
    int full_trip;
    if constexpr (kSparseAttnMode == SparseAttnMode::Sparse) {
      constexpr int full_tile_kv = get<1>(TileShape{});
      int valid_sparse_blocks = (kv_len + KVPageSize - 1) / KVPageSize;
      valid_sparse_blocks = min(valid_sparse_blocks, params.kv_block_num);
      valid_sparse_blocks = max(valid_sparse_blocks, 1);
      full_trip = (valid_sparse_blocks * KVPageSize + full_tile_kv - 1) / full_tile_kv;
    } else if constexpr (kSparseAttnMode == SparseAttnMode::OnlyScoreIcp) {
      // ICP: mirror the mainloop's get_full_trip_count ICP branch exactly; loader,
      // mainloop and kv_event_count() are all step-locked on this one count.
      constexpr int full_tile_kv = get<1>(TileShape{});
      int C = params.kv_block_num & 15;
      int r = params.kv_block_num >> 4;
      int gb_avail = (kv_len + KVPageSize - 1) / KVPageSize;
      int offset_q = Mask::get_qo_offset(problem_shape);
      int q_end = (int(get<0>(blk_coord)) + 1) * int(get<0>(TileShape{}));
      int gb_causal = (q_end + offset_q + KVPageSize - 1) / KVPageSize;
      int gb = gb_avail < gb_causal ? gb_avail : gb_causal;
      int lb = (gb - r + C - 1) / C;
      // Floor stays 1, step-locked with the identical floor in the mainloop's
      // get_trip_count(); flooring at 0 hangs mma()'s prologue on a causally-
      // empty tile.
      if (lb < 1) lb = 1;
      full_trip = (lb * KVPageSize + full_tile_kv - 1) / full_tile_kv;
    } else {
      full_trip = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    }
    return full_trip < kv_tile_end ? full_trip : kv_tile_end;
  }

#if MSA_NVFP4_KV_MODE >= 3
  // ===================================================================================
  // K2 / DESIGN §4.3 -- KV INGRESS: TWO 1-D BULK COPIES, NO TENSORMAP.
  //
  // One `(page, kv_head, side)` is 8192 contiguous packed-e2m1 bytes plus 1024 contiguous
  // e4m3 block-scale bytes, at two base+stride pairs supplied from Python
  // (k2::Nvfp4KvViews).  Both issues are credited to the SAME staging mbarrier, whose
  // expected transaction count producer_acquire already set to
  // PipelineKvStage::Params::transaction_bytes == 9216
  // ($CUT/include/cutlass/pipeline/sm90_pipeline.hpp, PipelineTmaAsync::producer_acquire
  // -> arrive_and_expect_tx).  Nothing else is needed: no cuTensorMap, no 4-bit data type,
  // no swizzle mode, and the ring is written LINEARLY, which is exactly what K1's dequant
  // loop wants.
  //
  // WHY NOT A 4-D cuTensorMap (rev 1's proposal): CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
  // requires boxDim[0] == 128 EXACTLY (64 and 256 both rejected, VERIFIED-LIVE R6 §7.2)
  // and PADS 8 four-bit values into 16 B (cuda.h:23272), so the staging ring would be
  // 16384 B/stage instead of 8192 -- silently doubling the SMEM problem.  R6 §7.2 also
  // recorded a doc-vs-driver divergence, i.e. cuTensorMapEncodeTiled returning
  // CUDA_SUCCESS is NOT proof of a legal descriptor.
  //
  // WHY NO gather4: tiles_per_page = 128/128 = 1, so one KV tile IS one page and the top-k
  // indirection is an ordinary integer coordinate (physical_page, computed by the caller
  // exactly as the fp8 path computes it).  The trtllm-gen `tmaDescSf = nullptr` gather4
  // blocker therefore does not apply here either.
  //
  // SPARSITY SEMANTICS ARE PRESERVED EXACTLY: the caller has already mapped an invalid
  // logical page to physical page 0, and the fp8 path does the same, relying on the mask
  // (mainloop:1226-1239 pushes invalid pages to INT_MAX/2, :907-908 applies it).  We copy
  // page 0's fp4 bytes for the same reason it copies page 0's fp8 bytes.
  // ===================================================================================
  template <bool kIsV, class ParamsT, class TensorStorageType, class BarrierT>
  CUTLASS_DEVICE static void k2_bulk_load_stage(ParamsT const& params,
                                                TensorStorageType& storage, BarrierT* mbar,
                                                int stage, int page, int head) {
    static_assert(k2::kStageBytes == 8192 + 1024, "mode 3 staging stage must be 9216 B");
    uint8_t* dst = storage.smem_kv_stage.data() + (size_t)stage * k2::kStageBytes;
    uint64_t* mb = reinterpret_cast<uint64_t*>(mbar);
    cute::SM90_BULK_COPY_G2S::copy(params.nvfp4_kv.data(kIsV, page, head), mb, dst, 8192);
    cute::SM90_BULK_COPY_G2S::copy(params.nvfp4_kv.scale(kIsV, page, head), mb, dst + 8192, 1024);
  }
#endif

  template <class BlkCoord, class ProblemShape, class ParamsT>
  CUTLASS_DEVICE static int kv_event_count(BlkCoord const& blk_coord,
                                           ProblemShape const& problem_shape,
                                           ParamsT const& params, int kv_tile_begin,
                                           int kv_tile_end) {
    int m = compute_effective_end(blk_coord, problem_shape, params, kv_tile_end) - kv_tile_begin;
    return m > 0 ? m * kKvEventsPerTile : 0;
  }

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
  static constexpr uint32_t NumStagesQ = PipelineQ::Stages;
  static constexpr int kTransactionBytesKV =
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);

  // (N, D, (H_R, H_G))
  using ShapeQ = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideQ = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;
  using LayoutQ = cute::Layout<ShapeQ, StrideQ>;

  // Paged: (page_size, D, H_kv, total_page_num) — 4-mode flat
  // Non-paged: (N, D, (H_R, H_G)) — 3-mode hierarchical
  using ShapeK = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, int32_t, int32_t, int32_t>,
      cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>>;
  using StrideK = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, _1, int32_t, int32_t>,
      cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>>;
  using ShapeV = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, int32_t, int32_t, int32_t>,
      cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>>;
  using StrideV = std::conditional_t<(KVPageSize > 0),
      cute::Shape<_1, int32_t, int32_t, int32_t>,
      cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>>;
  using LayoutK = cute::Layout<ShapeK, StrideK>;
  using LayoutV = cute::Layout<ShapeV, StrideV>;
  struct Arguments {
    const Element* ptr_Q;
    LayoutQ layout_Q;
    const Element* ptr_K;
    LayoutK layout_K;
    const Element* ptr_V;
    LayoutV layout_V;
    int* kv_indices = nullptr;
    int* kv_page_indptr = nullptr;
    int* kv_block_indexes = nullptr;
    int kv_block_num = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int kv_page_indptr_size = 0;
    int kv_indices_size = 0;
    int kv_block_indexes_numel = 0;
#endif
    int pack_factor = 1;
    int q_stride_n_original = 0;
    int q_stride_h_original = 0;
    int h_r_original = 0;
    // ---- refined-icp-v1: THE DIRECT COMPOUND-PAGE TABLE ----------------------------
    // The ORDINARY rectangular block table, consumed directly instead of the packed
    // per-rank page list.  `icp_block_table` is the table's STABLE BASE (offset 0, so
    // the address is invariant to the live request count -- contract I-6), and
    // `icp_block_table_row_stride` is an ADDRESS PITCH ONLY (contract I-1): it says
    // where request i's row starts, and NOTHING about how many of that row's entries
    // are live.  Null selects the packed-list path, which is unchanged.
    // Placed AFTER h_r_original and BEFORE the nvfp4 block on purpose: every positional
    // brace initialiser of this aggregate (fmha_cutlass_sm100.cuh:308, :329) stops at
    // h_r_original, so appending here cannot re-interpret one of their elements.
    int* icp_block_table = nullptr;
    int icp_block_table_row_stride = 0;
    // WHY A ROW ORIGIN EXISTS AT ALL.  "batch index IS table row" holds for the FIRST
    // query chunk and for no other.  The indexer chunks one invocation by query rows
    // (`icp_outer_chunk_tokens` is 1024 at 8k context but 128 at 1M, because the score
    // plane costs `H_group * 8192 * 5` bytes per row), so a 512-token decode graph at
    // 1M is FOUR invocations and chunk 3's batch index 0 is table row 384.  Addressing
    // it as `batch_idx * row_stride` would read request 0's pages for it: in bounds,
    // another tenant, finite and plausible.
    // It is a HOST int and it is I-3-legal: on the decode band `require_uniform` makes
    // `row_begin = t0 / query_len` a PER-GRAPH constant that does not depend on the
    // live request count, and prefill never enters a FULL graph
    // (`cudagraph_decode_phase_only`).  The table POINTER is still the base at storage
    // offset 0, which is what I-6 actually protects -- a moving BASE is frozen wrong by
    // capture, a per-graph constant ORIGIN is not.
    int icp_block_table_row_begin = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int icp_block_table_size = 0;
#endif
#if MSA_NVFP4_KV_MODE >= 3
    // K2: the packed NVFP4 KV pool, as four independent views.  See k2_nvfp4_kv.hpp.
    k2::Nvfp4KvViews nvfp4_kv{};
#endif
  };

  // using ShapeLseT = cute::Shape<int32_t, int32_t>;
  // using StrideLseT = cute::Shape<_1, int64_t>;
  // using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using ClusterLayout_VMNK =
      decltype(tiled_divide(make_layout(Shape<_1, _1, _1>{}),
                            make_tile(typename CollectiveMmaQK::TiledMma::AtomThrID{})));
  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

  static constexpr int kPackFactor = MaskPackFactor<Mask>::value;
  static constexpr bool kEnablePackedQTMA = (kPackFactor > 1) && ((kPackFactor & (kPackFactor - 1)) == 0);

  struct Params {
    TMA_Q tma_load_Q;
    LayoutQ layout_Q;
    TMA_K tma_load_K;
    LayoutK layout_K;
    TMA_V tma_load_V;
    LayoutV layout_V;
    int* kv_indices = nullptr;
    int* kv_page_indptr = nullptr;
    int* kv_block_indexes = nullptr;
    int kv_block_num = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int kv_page_indptr_size = 0;
    int kv_indices_size = 0;
    int kv_block_indexes_numel = 0;
#endif
    const Element* ptr_Q_orig = nullptr;
    int q_stride_n_orig = 0;
    int q_stride_h_orig = 0;
    int h_r_orig = 0;
    cute::TmaDescriptor tma_desc_q_pack;
    // refined-icp-v1 direct table; see Arguments above.  Appended AFTER
    // tma_desc_q_pack, which is where `to_underlying_arguments`' positional brace list
    // ends, for the same reason.
    int* icp_block_table = nullptr;
    int icp_block_table_row_stride = 0;
    int icp_block_table_row_begin = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int icp_block_table_size = 0;
#endif
#if MSA_NVFP4_KV_MODE >= 3
    k2::Nvfp4KvViews nvfp4_kv{};
#endif
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                        void* workspace) {
    static_assert(is_variable_length_v<tuple_element_t<0, ProblemShape>>);
    static_assert(is_variable_length_v<tuple_element_t<1, ProblemShape>>);
    auto ptr_Q = args.ptr_Q;
    auto ptr_K = args.ptr_K;
    auto ptr_V = args.ptr_V;
    LayoutQ layout_Q = args.layout_Q;
    LayoutK layout_K = args.layout_K;
    LayoutV layout_V = args.layout_V;

    auto mQ = make_tensor(make_gmem_ptr(ptr_Q), layout_Q);
    auto mK = make_tensor(make_gmem_ptr(ptr_K), layout_K);
    auto mV = make_tensor(make_gmem_ptr(ptr_V), layout_V);

    auto cluster_layout_vmnk =
        tiled_divide(make_layout(Shape<_1, _1, _1>{}),
                     make_tile(typename CollectiveMmaQK::TiledMma::AtomThrID{}));
    TMA_Q tma_load_Q = make_tma_atom_A_sm100<Element>(
        GmemTiledCopyQ{}, mQ, SmemLayoutQ{}(_, _, _, _0{}), TileShapeQK{},
        typename CollectiveMmaQK::TiledMma{}, cluster_layout_vmnk);
    TMA_K tma_load_K = make_tma_atom_B_sm100<Element>(
        GmemTiledCopyKV{}, mK, SmemLayoutK{}(_, _, _, _0{}), TileShapeQK{},
        typename CollectiveMmaQK::TiledMma{}, cluster_layout_vmnk);
    TMA_V tma_load_V = make_tma_atom_B_sm100<Element>(
        GmemTiledCopyKV{}, mV, SmemLayoutV{}(_, _, _, _0{}), TileShapePV{},
        typename CollectiveMmaPV::TiledMma{}, cluster_layout_vmnk);

    cute::TmaDescriptor tma_desc_q_pack{};
    if constexpr (kEnablePackedQTMA) {
      constexpr int tile_m = get<0>(TileShapeQK{});
      constexpr int tile_k = get<2>(TileShapeQK{});
      int total_seq = get<0>(shape(layout_Q)) / kPackFactor;
      int total_heads = args.h_r_original * get<1>(get<2>(shape(layout_Q)));
      int dim = get<1>(shape(layout_Q));

      auto tma_dtype = cute::TMA::to_CUtensorMapDataType<Element>();
      constexpr int box_dim0 = 128 / (int)sizeof(Element);
      uint64_t gDim[3] = {(uint64_t)dim, (uint64_t)total_heads, (uint64_t)total_seq};
      uint64_t gStride[2] = {
          (uint64_t)(args.q_stride_h_original * (int)sizeof(Element)),
          (uint64_t)(args.q_stride_n_original * (int)sizeof(Element))};
      uint32_t bDim[3] = {(uint32_t)box_dim0, (uint32_t)kPackFactor, (uint32_t)(tile_m / kPackFactor)};
      uint32_t eStride[3] = {1, 1, 1};

      cuTensorMapEncodeTiled(
          reinterpret_cast<CUtensorMap*>(&tma_desc_q_pack),
          tma_dtype, 3, (void*)args.ptr_Q,
          gDim, gStride, bDim, eStride,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    Params p{tma_load_Q, layout_Q, tma_load_K, layout_K, tma_load_V, layout_V,
                  args.kv_indices, args.kv_page_indptr, args.kv_block_indexes, args.kv_block_num,
#ifdef FMHA_GMEM_BOUNDS_CHECK
                  args.kv_page_indptr_size, args.kv_indices_size, args.kv_block_indexes_numel,
#endif
                  args.ptr_Q, args.q_stride_n_original, args.q_stride_h_original, args.h_r_original,
                  tma_desc_q_pack};
    // refined-icp-v1 direct table: copied EXPLICITLY for exactly the reason the
    // `p.nvfp4_kv = args.nvfp4_kv` line below exists.  Both members sit past the end of
    // the brace list above and both have default member initialisers, so omitting this
    // compiles clean, links clean, and silently runs the whole batch against a null
    // table -- i.e. the packed path, on a plan that has no packed list.
    p.icp_block_table = args.icp_block_table;
    p.icp_block_table_row_stride = args.icp_block_table_row_stride;
    p.icp_block_table_row_begin = args.icp_block_table_row_begin;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    p.icp_block_table_size = args.icp_block_table_size;
#endif
#if MSA_NVFP4_KV_MODE >= 3
    // K3: `nvfp4_kv` is the last member of BOTH aggregates and both have a default member
    // initialiser, so the brace list above leaves p.nvfp4_kv default-constructed (four
    // nullptrs).  It must be copied EXPLICITLY.  Omitting this line compiles clean, links
    // clean, and dies in the first bulk copy as `unspecified launch failure` with nothing
    // pointing at the cause -- which is exactly what it did before this line existed.
    p.nvfp4_kv = args.nvfp4_kv;
#endif
    return p;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (kEnablePackedQTMA)
      cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&params.tma_desc_q_pack));
    else if constexpr (kPackFactor == 1)
      cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
  }

  static constexpr int kTransactionBytesQ =
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);

  CUTLASS_DEVICE
  static void load_Q_cp_async(
      int q_tile_index, int qo_head_idx, int qo_segment_offset, int qo_len,
      Params const& params, auto const& params_problem_shape,
      TensorStorage& storage, PipelineQ& pipeline_q,
      typename PipelineQ::PipelineState& pipeline_q_producer_state,
      uint32_t lane_predicate
      #ifdef GPU_TRACE_ENABLED
        , gpu_trace::Recorder& _gt_rec
      #endif
      ) {

    constexpr int tile_m = get<0>(TileShapeQK{});
    constexpr int tile_k = get<2>(TileShapeQK{});
    int stage = pipeline_q_producer_state.index();

    int h_r_orig = params.h_r_orig;
    int h_r_packed = h_r_orig / kPackFactor;
    int kv_head = qo_head_idx / h_r_packed;
    int rem_h = qo_head_idx % h_r_packed;

    int tile_base = q_tile_index * tile_m;

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    auto sQ_grouped = group_modes<0, rank(SmemLayoutQ{}) - 1>(sQ);

    int lane_id = threadIdx.x % 32;
    int packed_len = qo_segment_offset + qo_len;

    {
      GPU_TRACE_SCOPE(LOAD_Q);
      constexpr int kVecElems = 16 / sizeof(Element);
      constexpr int k_iters = tile_k / kVecElems;
      int valid_m = min(tile_m, packed_len - (qo_segment_offset + tile_base));
      if (valid_m < 0) valid_m = 0;
      int total_ops = valid_m * k_iters;

      for (int idx = lane_id; idx < total_ops; idx += 32) {
        int m = idx / k_iters;
        int k = (idx % k_iters) * kVecElems;

        int packed_pos = qo_segment_offset + tile_base + m;
        int orig_pos = packed_pos / kPackFactor;
        int pack_idx = packed_pos % kPackFactor;
        int orig_head = kv_head * h_r_orig + rem_h * kPackFactor + pack_idx;

        const Element* gmem_src = params.ptr_Q_orig
                                + orig_pos * params.q_stride_n_orig
                                + orig_head * params.q_stride_h_orig
                                + k;

        Element* smem_dst = reinterpret_cast<Element*>(&sQ_grouped(m + tile_m * k, stage));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            smem_dst, gmem_src);
      }
    }
  }

  CUTLASS_DEVICE
  static void load_Q_wait(PipelineQ& pipeline_q,
      typename PipelineQ::PipelineState& pipeline_q_producer_state,
      uint32_t lane_predicate
      #ifdef GPU_TRACE_ENABLED
        , gpu_trace::Recorder& _gt_rec
      #endif
      ) {

    GPU_TRACE_SCOPE(LOAD_Q_WAIT);
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();

    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      cutlass::arch::ClusterTransactionBarrier::complete_transaction(
          tma_barrier, cute::block_rank_in_cluster(), kTransactionBytesQ);
    }
  }

  template <bool IsSplitKV = false, bool LoadV = true, class BlkCoord, class ProblemShape, class ParamsProblemShape, class TensorStorageType>
  CUTLASS_DEVICE void load(BlkCoord const& blk_coord, ProblemShape const& problem_shape,
                           Params const& params, ParamsProblemShape const& params_problem_shape,
                           int const& work_idx,
                           TensorStorageType& storage, PipelineQ& pipeline_q,
                           typename PipelineQ::PipelineState& pipeline_q_producer_state,
                           PipelineKV& pipeline_kv,
                           typename PipelineKV::PipelineState& pipeline_kv_producer_state,
                           int kv_tile_begin, int kv_tile_end
                           #ifdef GPU_TRACE_ENABLED
                             , gpu_trace::Recorder& _gt_rec
                           #endif
                          ) {
    // GET_GPU_TRACE(true);

    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int kv_len = get<1>(problem_shape);
    int qo_segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];
    int kv_segment_offset = get<1>(params_problem_shape).segment_offsets[batch_idx];

    int effective_end = compute_effective_end(blk_coord, problem_shape, params, kv_tile_end);
    int mask_tile_count = effective_end - kv_tile_begin;
    if constexpr (IsSplitKV) {
      if (mask_tile_count <= 0) return;
    }
    int start_kv_index = effective_end - 1;

    using X = Underscore;

    Tensor mQ = params.tma_load_Q.get_tma_tensor(params.layout_Q.shape());

    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);
    ThrMMA mma_pv = typename CollectiveMmaPV::TiledMma{}.get_slice(0);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Element* kv_dst = storage.smem_kv.data();

    Tensor sK = make_tensor(make_smem_ptr(kv_dst), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(kv_dst), SmemLayoutV{});

    auto gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShapeQK{}), qo_head_idx, qo_segment_offset,
                                    qo_len);
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ);
    auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 3>(sQ),
                                      group_modes<0, 3>(tSgQ_qdl));

    uint32_t lane_predicate = cute::elect_one_sync();

    static constexpr int num_q_sub = get<0>(TileShape{}) / get<0>(TileShapeQK{});
    static constexpr int num_kv_sub = kNumKvSub;

    int q0_index = num_q_sub * get<0>(blk_coord);
    int q1_index = num_q_sub * get<0>(blk_coord) + 1;
    int kv_tile_index = start_kv_index * num_kv_sub;

    auto run_loads = [&](auto load_K_tile, auto load_V_tile) {

      if constexpr (kEnablePackedQTMA) {
        static_assert(num_kv_sub > 1);
        static_assert(num_q_sub == 1);

        pipeline_q.producer_acquire(pipeline_q_producer_state);
        {
          GPU_TRACE_SCOPE(LOAD_Q);
          if (lane_predicate) {
            constexpr int tile_m = get<0>(TileShapeQK{});
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            uint32_t smem_int_mbar = cute::cast_smem_ptr_to_uint(&(*tma_barrier));
            uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&params.tma_desc_q_pack);

            int h_r_packed = params.h_r_orig / kPackFactor;
            int kv_head = qo_head_idx / h_r_packed;
            int rem_h = qo_head_idx % h_r_packed;
            int first_head = kv_head * params.h_r_orig + rem_h * kPackFactor;
            int tile_base = q0_index * tile_m;
            int orig_token_start = (qo_segment_offset + tile_base) / kPackFactor;

            int stage = pipeline_q_producer_state.index();
            constexpr int stage_bytes = kTransactionBytesQ;
            uint32_t smem_base = cute::cast_smem_ptr_to_uint(storage.smem_q.data())
                                  + stage * stage_bytes;

            constexpr int box_dim0 = 128 / (int)sizeof(Element);
            constexpr int dim_iters = get<2>(TileShapeQK{}) / box_dim0;
            constexpr int chunk_bytes = box_dim0 * tile_m * (int)sizeof(Element);
            for (int di = 0; di < dim_iters; di++) {
              uint32_t smem_int_ptr = smem_base + di * chunk_bytes;
              asm volatile(
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
                  "cp.async.bulk.tensor.3d.shared::cta.global.tile"
#else
                  "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
#endif
                  ".mbarrier::complete_tx::bytes.L2::cache_hint"
                  " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                  :
                  : "r"(smem_int_ptr), "l"(gmem_int_desc),
                    "r"(di * box_dim0), "r"(first_head), "r"(orig_token_start),
                    "r"(smem_int_mbar), "l"((uint64_t)0)
                  : "memory");
            }
          }
        }
        ++pipeline_q_producer_state;

        load_K_tile(kv_tile_index);
        load_K_tile(kv_tile_index + 1);
      } else if constexpr (kPackFactor > 1) {
        static_assert(num_kv_sub > 1);
        static_assert(num_q_sub == 1);

        load_K_tile(kv_tile_index);
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        load_Q_cp_async(q0_index, qo_head_idx, qo_segment_offset, qo_len,
                        params, params_problem_shape, storage, pipeline_q,
                        pipeline_q_producer_state, lane_predicate
                        #ifdef GPU_TRACE_ENABLED
                          , _gt_rec
                        #endif
                        );
        load_K_tile(kv_tile_index + 1);
        load_Q_wait(pipeline_q, pipeline_q_producer_state, lane_predicate
          #ifdef GPU_TRACE_ENABLED
            , _gt_rec
          #endif
          );
        ++pipeline_q_producer_state;
      } else {
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        {
          GPU_TRACE_SCOPE(LOAD_Q);
          if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_Q.with(*tma_barrier, 0), tQgQ(_, q0_index),
                 tQsQ(_, pipeline_q_producer_state.index()));
          }
        }
        ++pipeline_q_producer_state;

        if constexpr (num_q_sub > 1) {
          pipeline_q.producer_acquire(pipeline_q_producer_state);
          GPU_TRACE_SCOPE(LOAD_Q);
          if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_Q.with(*tma_barrier, 0), tQgQ(_, q1_index),
                 tQsQ(_, pipeline_q_producer_state.index()));
          }
          ++pipeline_q_producer_state;
        }
        load_K_tile(kv_tile_index);
        if constexpr (num_kv_sub > 1) load_K_tile(kv_tile_index + 1);
      }

      if constexpr (num_kv_sub > 1) {
        // K-split: pre(KK) already done above; loop body is VKVK; epi is VV
        int v_tile_index = kv_tile_index;
        kv_tile_index -= 2;

        mask_tile_count -= 1;
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
          if constexpr (kNeedV) load_V_tile(v_tile_index);
          load_K_tile(kv_tile_index);
          if constexpr (kNeedV) load_V_tile(v_tile_index + 1);
          load_K_tile(kv_tile_index + 1);
          v_tile_index = kv_tile_index;
          kv_tile_index -= 2;
        }

        if constexpr (kNeedV) {
          load_V_tile(v_tile_index);
          load_V_tile(v_tile_index + 1);
        }
      } else {
        if constexpr (kNeedV) load_V_tile(kv_tile_index);
        kv_tile_index -= 1;

        mask_tile_count -= 1;
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
          load_K_tile(kv_tile_index);
          if constexpr (kNeedV) load_V_tile(kv_tile_index);
          kv_tile_index -= 1;
        }
      }
    };

    if constexpr (KVPageSize > 0) {
      // Paged KV: K/V shape (page_size, D, num_kv_heads, total_page_num)
      Tensor mK = params.tma_load_K.get_tma_tensor(params.layout_K.shape());
      Tensor mV = params.tma_load_V.get_tma_tensor(params.layout_V.shape());

      int h_r = get<3, 0, 0>(params_problem_shape);
      int kv_head_idx = qo_head_idx / h_r;

      // ---- refined-icp-v1: ROW ADDRESS AND LIVE BOUND ARE TWO DIFFERENT THINGS ------
      // The packed-list path derives BOTH from one CSR: `indptr[b]` is where the row
      // starts and `indptr[b+1] - indptr[b]` is how many pages exist.  The direct-table
      // path (DIRECT_TABLE_CONTRACT §5.1) separates them, because the table is
      // RECTANGULAR and its row length carries no information about live pages:
      //
      //   row start  = (row_begin + batch_idx) * row_stride   <- address pitch (I-1)
      //   live bound = local_blocks(L, r)                     <- EXACT device KV length
      //
      // Substituting the row capacity for the second is the failure this redesign is
      // most exposed to: the tail of a row holds physical page IDs left behind by
      // EVICTED requests, which are valid addresses pointing at another tenant's data,
      // so the symptom is a plausible wrong answer rather than a fault
      // (DIRECT_COMPOUND_PAGE_TABLE.md l.119-121, contract I-1).
      //
      // Direct mode is selected by the TABLE POINTER, not by a flag, and only inside
      // the ICP variant: for every other `kSparseAttnMode` the left conjunct is a
      // compile-time false and the whole branch folds away, so the packed path emits
      // exactly the instructions it did before.
      constexpr bool kIcpDirectEligible =
          (kSparseAttnMode == SparseAttnMode::OnlyScoreIcp);
      const bool use_direct_table =
          kIcpDirectEligible && (params.icp_block_table != nullptr);

      const int* page_table = params.kv_indices;
      int kv_page_start = 0;
      int num_pages_batch = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
      int page_table_size = params.kv_indices_size;
#endif
      if (use_direct_table) {
        // The whole request->row mapping, and it is still not a tensor: a per-graph
        // host origin plus the batch index.  `row_begin` is 0 for the first query
        // chunk and `t0 / query_len` for the rest -- see `icp_block_table_row_begin`
        // in Arguments for why chunk 3 of a 1M-context decode graph is row 384 and
        // not row 0.  The table pointer itself is still the base at offset 0 (I-6).
        page_table = params.icp_block_table;
        kv_page_start = (params.icp_block_table_row_begin + batch_idx)
                      * params.icp_block_table_row_stride;
        num_pages_batch = icp_local_blocks_exact(kv_len, params.kv_block_num);
#ifdef FMHA_GMEM_BOUNDS_CHECK
        page_table_size = params.icp_block_table_size;
#endif
      } else {
        kv_page_start = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx, params.kv_page_indptr_size);
        if constexpr (kSparseAttnMode != SparseAttnMode::Sparse) {
          int kv_page_end = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx + 1, params.kv_page_indptr_size);
          num_pages_batch = kv_page_end - kv_page_start;
        }
      }
      // ONE checked read for both paths, so the clamp and the load can never be applied
      // to different arrays.  Without -DFMHA_GMEM_BOUNDS_CHECK both arms expand to the
      // same `__ldg` and the branch disappears; with it, the two arrays report under
      // their own names because they fail for different reasons.
      auto load_page_entry = [&](int idx) -> int {
#ifdef FMHA_GMEM_BOUNDS_CHECK
        if (use_direct_table) return BLOCK_TABLE_LOAD(page_table, idx, page_table_size);
        return KV_INDICES_LOAD(page_table, idx, page_table_size);
#else
        return KV_INDICES_LOAD(page_table, idx, 0);
#endif
      };
      constexpr int effective_tile_kv = get<1>(TileShapeQK{});

      // ---- refined-icp-v1 --------------------------------------------------
      // A compound page contributes only R = 128/W index rows to THIS rank
      // (ABI P1/P2), so KVPageSize may be SMALLER than a 128-row compute tile.
      // Two regimes:
      //   KVPageSize >= tile : one page feeds `tiles_per_page` tiles (shipped).
      //   KVPageSize <  tile : one tile needs `pages_per_tile` DIFFERENT pages.
      // The refined route avoids the second regime entirely by BOUNDED FRAGMENT
      // LOADING -- jit.py binds tile_kv to the page size, so effective_tile_kv
      // == KVPageSize and tiles_per_page == 1 legitimately. `pages_per_tile` is
      // computed and asserted anyway so that a future variant which does enter
      // the second regime FAILS THE BUILD instead of silently misaddressing.
      static_assert(KVPageSize > 0, "paged path only");
      static_assert(effective_tile_kv > 0, "TileShapeQK KV extent must be > 0");
      static_assert(KVPageSize >= effective_tile_kv
                        ? (KVPageSize % effective_tile_kv == 0)
                        : (effective_tile_kv % KVPageSize == 0),
                    "refined-icp-v1: KVPageSize and TileShapeQK's KV extent must "
                    "divide one another. A remainder leaves part of the compute "
                    "tile unfed by any page, and no later masking can repair an "
                    "unfed tile.");
      constexpr int tiles_per_page = (KVPageSize >= effective_tile_kv)
                                   ? (KVPageSize / effective_tile_kv) : 1;
      constexpr int pages_per_tile = (KVPageSize >= effective_tile_kv)
                                   ? 1 : (effective_tile_kv / KVPageSize);
      // THE blocking-defect guard. A compile-time ZERO divisor is UB and nvcc
      // does NOT fault: it emits only warning #39-D / #179-D, returns 0, and
      // DELETES the dependent address computation, so `logical_page` and
      // `sub_tile` become undefined and the loader silently reads the wrong
      // page. Measured in this very kernel at KVPageSize=64/32: 12 warnings,
      // ninja rc=0, a loadable .so, a launch and a sync that both report no
      // CUDA error, and the stores simply gone. Nothing crashes.
      static_assert(tiles_per_page >= 1 && pages_per_tile >= 1,
                    "refined-icp-v1: tiles_per_page/pages_per_tile must be "
                    ">= 1; a compile-time zero divisor is UB and nvcc silently "
                    "deletes the dependent address computation.");
      // TMA DESCRIPTOR BOUND. The gmem tensor's mode-0 extent is KVPageSize
      // (`shape_K = make_shape(KVPageSize, ...)` in FwdRunner::run, paged arm)
      // while the TMA atom's box comes from TileShapeQK. A box taller than the
      // tensor extent is an out-of-bounds descriptor.
      static_assert(effective_tile_kv <= KVPageSize || pages_per_tile > 1,
                    "refined-icp-v1: the TMA box (TileShapeQK KV extent) is "
                    "taller than the paged tensor's mode-0 extent (KVPageSize) "
                    "and no multi-page gather is implemented for it.");
      (void)pages_per_tile;
      // ---- end refined-icp-v1 ----------------------------------------------

      int kv_block_offset = 0;
      if constexpr (kSparseAttnMode == SparseAttnMode::Sparse) {
        int num_kv_heads_val = get<3, 0, 1>(params_problem_shape);
        kv_block_offset = batch_idx * num_kv_heads_val * params.kv_block_num
                        + kv_head_idx * params.kv_block_num;
      }

      // Keep full 4D tensor (page_size, D, H_kv, P) — select head at copy time
      auto gK = local_tile(mK, select<1, 2>(TileShapeQK{}), make_coord(_, _0{}));
      auto gV = local_tile(mV, select<1, 2>(TileShapePV{}), make_coord(_0{}, _));

      Tensor tSgK_kdl = mma_qk.partition_B(gK);
      Tensor tOgV_dkl = mma_pv.partition_B(gV);
      auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                         group_modes<0, 3>(sK), group_modes<0, 3>(tSgK_kdl));
      auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                         group_modes<0, 3>(sV), group_modes<0, 3>(tOgV_dkl));

      auto load_K_tile = [&](int tile_idx) {
        int logical_page = tile_idx / tiles_per_page;
        int sub_tile = tile_idx % tiles_per_page;
        int page_for_lookup;
        if constexpr (kSparseAttnMode == SparseAttnMode::Sparse) {
          int sparse_idx = (logical_page < params.kv_block_num)
              ? __ldg(&params.kv_block_indexes[kv_block_offset + logical_page])
              : -1;
          page_for_lookup = (sparse_idx >= 0) ? sparse_idx : 0;
        } else {
          // SPECULATIVE-LOAD CLAMP.  The pipeline is allowed to run ahead of the
          // masked extent, so `logical_page` can exceed the live count; it is pulled
          // back onto the LAST LIVE page, never onto the row's capacity.  Under the
          // direct table `num_pages_batch` is the device-derived live bound above, so
          // this clamp is exactly the one the packed indptr difference used to give.
          // `num_pages_batch == 0` would make this -1; that work item never reaches
          // here, because the kernel-wide `is_empty_work` decision
          // (sm100_fmha_fwd_kernel_tma_warpspecialized.hpp:404-416) skips it in all
          // five warp roles from the same device data.
          page_for_lookup = logical_page;
          page_for_lookup = min(page_for_lookup, num_pages_batch - 1);
        }
        int physical_page = load_page_entry(kv_page_start + page_for_lookup);
        { GPU_TRACE_SCOPE(LOAD_K); pipeline_kv.producer_acquire(pipeline_kv_producer_state); }
        if (lane_predicate) {
          K2_TAG_STAGE(/*is_v=*/false, physical_page);
          auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
#if MSA_NVFP4_KV_MODE >= 3
          k2_bulk_load_stage</*kIsV=*/false>(params, storage, tma_barrier,
                                             pipeline_kv_producer_state.index(),
                                             physical_page, kv_head_idx);
#else
          copy(params.tma_load_K.with(*tma_barrier, 0),
               tKgK(_, sub_tile, kv_head_idx, physical_page),
               tKsK(_, pipeline_kv_producer_state.index()));
#endif
        }
        ++pipeline_kv_producer_state;
      };

      auto load_V_tile = [&](int tile_idx) {
        int logical_page = tile_idx / tiles_per_page;
        int sub_tile = tile_idx % tiles_per_page;
        int page_for_lookup;
        if constexpr (kSparseAttnMode == SparseAttnMode::Sparse) {
          int sparse_idx = (logical_page < params.kv_block_num)
              ? __ldg(&params.kv_block_indexes[kv_block_offset + logical_page])
              : -1;
          page_for_lookup = (sparse_idx >= 0) ? sparse_idx : 0;
        } else {
          // Same clamp, same reasoning, as load_K_tile above.
          page_for_lookup = logical_page;
          page_for_lookup = min(page_for_lookup, num_pages_batch - 1);
        }
        int physical_page = load_page_entry(kv_page_start + page_for_lookup);
        { GPU_TRACE_SCOPE(LOAD_V); pipeline_kv.producer_acquire(pipeline_kv_producer_state); }
        if (lane_predicate) {
          K2_TAG_STAGE(/*is_v=*/true, physical_page);
          auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
#if MSA_NVFP4_KV_MODE >= 3
          k2_bulk_load_stage</*kIsV=*/true>(params, storage, tma_barrier,
                                            pipeline_kv_producer_state.index(),
                                            physical_page, kv_head_idx);
#else
          if constexpr (LoadV) {
            copy(params.tma_load_V.with(*tma_barrier, 0),
                 tVgV(_, sub_tile, kv_head_idx, physical_page),
                 tVsV(_, pipeline_kv_producer_state.index()));
          } else {
            cutlass::arch::ClusterTransactionBarrier::complete_transaction(
                tma_barrier, cute::block_rank_in_cluster(), kTransactionBytesKV);
          }
#endif
        }
        ++pipeline_kv_producer_state;
      };

      run_loads(load_K_tile, load_V_tile);
    } else {
      // Non-paged KV
      Tensor mK = params.tma_load_K.get_tma_tensor(params.layout_K.shape());
      Tensor mV = params.tma_load_V.get_tma_tensor(params.layout_V.shape());

      auto gK = get_local_tile_tensor(mK, select<1, 2>(TileShapeQK{}), qo_head_idx,
                                      kv_segment_offset, kv_len);
      auto gV = get_local_tile_t_tensor(mV, select<1, 2>(TileShapePV{}), qo_head_idx,
                                        kv_segment_offset, kv_len);

      Tensor tSgK_kdl = mma_qk.partition_B(gK);
      Tensor tOgV_dkl = mma_pv.partition_B(gV);
      auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                         group_modes<0, 3>(sK), group_modes<0, 3>(tSgK_kdl));
      auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                         group_modes<0, 3>(sV), group_modes<0, 3>(tOgV_dkl));

      auto load_K_tile = [&](int tile_idx) {
        { GPU_TRACE_SCOPE(LOAD_K); pipeline_kv.producer_acquire(pipeline_kv_producer_state); }
        if (lane_predicate) {
          auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
          copy(params.tma_load_K.with(*tma_barrier, 0), tKgK(_, tile_idx),
               tKsK(_, pipeline_kv_producer_state.index()));
        }
        ++pipeline_kv_producer_state;
      };

      auto load_V_tile = [&](int tile_idx) {
        { GPU_TRACE_SCOPE(LOAD_V); pipeline_kv.producer_acquire(pipeline_kv_producer_state); }
        if (lane_predicate) {
          auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
          if constexpr (LoadV) {
            copy(params.tma_load_V.with(*tma_barrier, 0), tVgV(_, tile_idx),
                 tVsV(_, pipeline_kv_producer_state.index()));
          } else {
            cutlass::arch::ClusterTransactionBarrier::complete_transaction(
                tma_barrier, cute::block_rank_in_cluster(), kTransactionBytesKV);
          }
        }
        ++pipeline_kv_producer_state;
      };

      run_loads(load_K_tile, load_V_tile);
    }
    // RELEASE_GPU_TRACE;
  }
};

}  // namespace cutlass::fmha::collective
