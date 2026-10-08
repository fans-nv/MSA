// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gmem_bounds_check.h"

struct FMHACutlassSM100Params {
  void* workspace_buffer_ptr;
  void* q_ptr;
  void* k_ptr;
  void* v_ptr;
  int* qo_segment_lens_ptr;
  int* kv_segment_lens_ptr;
  int* qo_segment_offsets_ptr;
  int* kv_segment_offsets_ptr;
  uint64_t* packed_work_range_ptr;
  uint64_t* packed_work_info_ptr;
  void* o_ptr;
  int mask_mode_code;
  float sm_scale;
  float scale_q;
  float scale_k;
  float scale_v;
  float o_scale;
  int num_qo_heads;
  int num_kv_heads;
  int head_dim_qk;
  int head_dim_vo;
  int q_stride_n;
  int q_stride_h;
  int k_stride_n;
  int k_stride_h;
  int k_stride_t;
  int v_stride_n;
  int v_stride_h;
  int v_stride_t;
  int batch_size;
  int total_qo_len;
  int total_kv_len;
  int max_qo_len;
  int* qo_offsets_ptr;
  cudaStream_t stream;
  int num_kv_splits;
  int* kv_tile_begin_ptr;
  int* kv_tile_end_ptr;
  int* kv_split_ptr;
  void* workspace_o_ptr;
  float* workspace_lse_ptr;
  int* num_kv_splits_per_row_ptr;
  int* kv_indices_ptr;
  int* kv_page_indptr_ptr;
  int total_page_num;
  float* max_score_ptr;
  int max_k_tiles;
  int max_score_stride_t = 0;
  int max_score_stride_h = 0;
  int max_score_stride_k = 0;
  int* kv_block_indexes_ptr;
  int kv_block_num;
  int pack_factor = 1;
  int h_r_original = 0;
  int q_stride_n_original = 0;
  int q_stride_h_original = 0;
  float* max_score_direct_ptr = nullptr;
  int total_qo_len_orig = 0;
  void* o_direct_ptr = nullptr;
  int num_qo_heads_orig = 0;
  int num_ctas = 0;
  // TMA fused direct-O unpack: enabled when pack_factor > 1, ptr_O_direct present,
  // qo_len uniform across batches, and FMHA_DISABLE_TMA_DIRECT_O env var not set.
  // When false, epilogue falls back to vec16 software scatter.
  bool tma_direct_o_enabled = false;

  void* nvfp4_k_data_ptr = nullptr;
  void* nvfp4_k_scale_ptr = nullptr;
  void* nvfp4_v_data_ptr = nullptr;
  void* nvfp4_v_scale_ptr = nullptr;
  int64_t nvfp4_page_stride = 0;         // bytes between consecutive PHYSICAL pages
  int nvfp4_head_stride_data = 8192; // bytes between kv heads, data region
  int nvfp4_head_stride_scale = 1024;// bytes between kv heads, scale region
  float dequant_g_k = 1.0f;
  float dequant_g_v = 1.0f;
  const float* nvfp4_k_global_scale = nullptr;
  const float* nvfp4_v_global_scale = nullptr;

  // ---- refined-icp-v1: the OUT-OF-BAND VALIDITY PLANE ------------------------------
  // ABI `refined-icp-v1.abi.1` clauses W4/W8/W9: validity is a uint8 plane
  // shape-matched to the score plane, with its own explicit strides.  An in-band
  // sentinel is FORBIDDEN, because -inf is a legal representable score (W12) and a
  // consumer reading `max_score` alone cannot tell "scored, and its maximum is -inf"
  // from "past the end".  These fields are additive at the tail and default to
  // neutral, so a variant that does not use them is byte-identical to the previous
  // kernel.
  //
  // Output-ABI version of the ICP score wave.  1 = score plane only (superseded);
  // 2 = score plane + out-of-band uint8 validity plane.  Bumping this is what makes
  // the change explicit rather than silent: nothing may change units, layout or
  // validity encoding without moving this number.
  static constexpr int kIcpScoreAbiVersion = 2;
  uint8_t* valid_score_ptr = nullptr;
  int valid_score_stride_t = 0;
  int valid_score_stride_h = 0;
  int valid_score_stride_k = 0;

  // ---- refined-icp-v1: THE DIRECT COMPOUND-PAGE TABLE ------------------------------
  // DIRECT_TABLE_CONTRACT §5.  An ADDED input mode, not a cutover: with
  // `icp_block_table_ptr == nullptr` the kernel consumes the packed page list exactly as
  // before, which is what the `icp_c == 1` production route needs (it uses the same CSR).
  //
  // `icp_block_table_ptr` is the ordinary rectangular block table's STABLE BASE.  It is
  // the base, never a view that begins at the first live request: a view offset would
  // move with the live request count and be frozen wrong by a graph capture (I-6).
  // `icp_block_table_row_stride` is an ADDRESS PITCH -- `max_blocks_per_req`, a startup
  // constant.  It must NEVER be used as a live page count: the tail of a row holds
  // physical page IDs from evicted requests, i.e. valid addresses pointing at another
  // tenant's data, so the failure signature of misusing it is a plausible wrong answer
  // and not a fault (I-1).  The live bound is device-derived in the loader from the exact
  // KV length (`icp_local_blocks_exact`).
  //
  // `icp_block_table_row_begin` is the table row this CALL's batch index 0 maps to, and
  // it is not always 0: the indexer chunks one invocation by query rows, so a multi-chunk
  // decode graph hands later chunks a batch whose index 0 is row `t0 / query_len`.  A
  // host int rather than a mapping tensor, because on the decode band uniform query
  // length makes it a PER-GRAPH constant independent of the live request count -- which
  // is what I-3 admits, and is why the table POINTER (not the origin) is the thing I-6
  // pins at storage offset 0.
  //
  // INPUT-ABI version of the direct-table entry.  Separate from kIcpScoreAbiVersion,
  // which versions the OUTPUT wave: the two move independently, and a Python half that
  // advertises this input mode over a kernel half that ignores it would run the whole
  // batch against a null table -- i.e. silently take the packed path on a plan that has
  // no packed list.  api.py refuses to run unless it can see these symbols in the
  // overlay sources (`_FMHA_HAS_ICP_DIRECT_TABLE`), and asserts it.
  static constexpr int kIcpDirectTableAbiVersion = 1;
  int* icp_block_table_ptr = nullptr;
  int icp_block_table_row_stride = 0;
  int icp_block_table_row_begin = 0;

  GMEM_BOUNDS_FIELD
};

using FMHAVariantFn = cudaError_t (*)(const FMHACutlassSM100Params&);

cudaError_t fmha_reduction_bf16(const void* ptr_O_partial, void* ptr_O,
                                const float* ptr_lse,
                                const int* num_kv_splits_per_row,
                                float scale_softmax_log2, float inv_scale_o,
                                int num_kv_splits, int total_qo_len, int num_qo_heads,
                                int head_dim_vo, int stride_o_n, int stride_o_h,
                                int stride_partial_n, int stride_partial_h,
                                cudaStream_t stream);
