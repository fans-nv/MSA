"""Lazy ICP facade for MSA's shared NVFP4 sparse prefill implementation.

The attention/combine kernels and CSR builder live only in
``fmha_sm100.cute``. Both ordinary MSA and ICP use that source tree and its AOT
cache. The CSR extension has a source/architecture-specific MSA identity, so a
framework's older FP8/reference copy cannot overwrite it. Strict serving
prewarm requires AOT caching to remain enabled and verifies the effective paths.
"""

from __future__ import annotations

__all__ = ["build_k2q_csr", "sparse_atten_nvfp4_kv_func", "PREWARM_VARIANTS"]

COMPONENT = "nvfp4"
K2Q_EXTENSION = "msa_build_k2q_csr"

#: The admitted TP2 compound-page variant: two local KV heads, D128, GQA16,
#: top-16 of P128 blocks, 45056-byte pages, BF16 query, paged causal KV with
#: seqused_k and no global scales. ``(q_dtype, qhead_per_kv, topk)``.
PREWARM_VARIANTS = (("bfloat16", 16, 16),)
AOT_FAMILIES = ("sparse_forward_sm100_csr_varlen_nvfp4_kv_cache", "combine")

def _configure():
    """Load the shared MSA cache policy without mutating global configuration."""
    from ....cute.src.common import aot_cache  # noqa: F401, PLC0415


def aot_dir(arch=None):
    import pathlib

    from ....cute.src.common import aot_cache  # noqa: PLC0415

    # Shared dev AOT includes its schema and toolchain namespace. Using the
    # loader's resolved path also respects its frozen import-time environment.
    return pathlib.Path(aot_cache._AOT_CACHE_DIR)


def k2q_extension_name(arch=None):
    """A separate extension identity for each packaged source/architecture."""
    from ... import _cache  # noqa: PLC0415

    arch = _cache.normalize_arch(arch or _cache.key_arch())
    return (f"{K2Q_EXTENSION}_{arch}_"
            f"{_cache.component_digest(COMPONENT)[:16]}")


def k2q_extension_path(arch=None):
    """Where ``cpp_extension`` keeps the k2q CSR builder's ``.so``."""
    import pathlib  # noqa: PLC0415

    from torch.utils.cpp_extension import _get_build_directory  # noqa: PLC0415

    name = k2q_extension_name(arch)
    return pathlib.Path(_get_build_directory(name, False)) / f"{name}.so"


def _load_k2q_extension():
    from ... import _jit_guard  # noqa: PLC0415

    if not k2q_extension_path().is_file():
        _jit_guard.on_compile(COMPONENT, k2q_extension_name(),
                              where="attention.nvfp4_prefill")
    from ....cute.src.sm100 import build_k2q_csr  # noqa: F401, PLC0415


def __getattr__(name):
    if name == "sparse_atten_nvfp4_kv_func":
        _configure()
        from ....cute.interface import sparse_atten_nvfp4_kv_cache_func  # noqa: PLC0415

        return sparse_atten_nvfp4_kv_cache_func
    if name == "build_k2q_csr":
        _configure()
        _load_k2q_extension()
        from ....cute.sparse_index_utils import build_k2q_csr  # noqa: PLC0415

        return build_k2q_csr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
