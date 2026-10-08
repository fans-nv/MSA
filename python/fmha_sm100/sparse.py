# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Package-relative public sparse attention, indexer and KV-outer APIs.

The CuTe modules load only through this package. Their imports remain valid
when MSA is vendored, without registering ambiguous top-level module names.
"""

from __future__ import annotations

# Sparse attention forward / decode (cute/interface.py).
from .cute.interface import (  # noqa: E402
    SparseDecodePagedAttentionWrapper,
    sparse_atten_func,
    sparse_atten_nvfp4_kv_func,
    sparse_atten_nvfp4_kv_cache_func,
    sparse_decode_atten_func,
)

# CSR + schedule construction (cute/sparse_index_utils.py).
from .cute.sparse_index_utils import build_k2q_csr  # noqa: E402

# SM100 fused CSR builder (cute/src/sm100/prepare_k2q_csr.py).
from .cute.src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100  # noqa: E402

# FP4 block-score indexer (cute/fp4_indexer_interface.py).
# Returns per-(Hq, kv_block, q) max scores; topK selection + q2k construction
# remain caller-owned downstream steps.
from .cute.fp4_indexer_interface import fp4_indexer_block_scores  # noqa: E402

# Q8KV4/Q8KV8 paged indexers over the vLLM index-K cache, with fused TopK
# (cute/q8_indexer_interface.py).
from .cute.q8_indexer_interface import (  # noqa: E402
    BatchDecodeIndexerQ8KV4Wrapper,
    BatchDecodeIndexerQ8KV8Wrapper,
    BatchPrefillIndexerQ8KV8Wrapper,
    bind_indexer_module_loader,
)

from .jit import get_indexer_module  # noqa: E402

# The cute/ modules cannot import this package by name when it is vendored under
# another package, so hand them this package's csrc JIT loader.
bind_indexer_module_loader(get_indexer_module)

# NVFP4 quantization helpers used to feed the FP4 indexer / NVFP4 attention
# (cute/quantize.py).
from .cute.quantize import (  # noqa: E402
    Nvfp4QuantizedTensor,
    dequantize_nvfp4_128x4_to_bf16,
    nvfp4_global_scale_from_amax,
    quantize_bf16_to_nvfp4_128x4,
    quantize_kv_bf16_to_nvfp4_128x4,
    swizzle_nvfp4_scale_to_128x4,
)

# Fireworks KV-outer (KV-stationary) block-sparse prefill (fmha_sm100/kvouter/).
from .kvouter import can_run_sparse_kvouter, kvouter_attention  # noqa: E402

__all__ = [
    # attention
    "sparse_atten_func",
    "sparse_atten_nvfp4_kv_func",
    "sparse_atten_nvfp4_kv_cache_func",
    "sparse_decode_atten_func",
    "SparseDecodePagedAttentionWrapper",
    # indexing / CSR
    "fp4_indexer_block_scores",
    "BatchDecodeIndexerQ8KV4Wrapper",
    "BatchDecodeIndexerQ8KV8Wrapper",
    "BatchPrefillIndexerQ8KV8Wrapper",
    "build_k2q_csr",
    "SparseK2qCsrBuilderSm100",
    # kv-outer prefill
    "kvouter_attention",
    "can_run_sparse_kvouter",
    # nvfp4 quantization helpers
    "Nvfp4QuantizedTensor",
    "quantize_bf16_to_nvfp4_128x4",
    "quantize_kv_bf16_to_nvfp4_128x4",
    "dequantize_nvfp4_128x4_to_bf16",
    "swizzle_nvfp4_scale_to_128x4",
    "nvfp4_global_scale_from_amax",
]
