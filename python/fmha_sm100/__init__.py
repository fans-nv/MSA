# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""MiniMax sparse attention and optional indexer context parallelism.

Kernel APIs load on first access. Importing the package does not initialize CUDA
or compile native extensions. Legacy TVM global functions register by default;
frameworks with a vendored MSA can set ``MSA_REGISTER_TVM_FFI=0`` before import
to leave those global names with the existing owner.
"""
import os
from importlib import import_module

__version__ = "0.1.1"
_DENSE = {"fmha_sm100", "fmha_sm100_plan", "sparse_topk_select"}
_SPARSE = {"sparse_atten_func", "sparse_atten_nvfp4_kv_func",
           "sparse_atten_nvfp4_kv_cache_func",
           "sparse_decode_atten_func", "SparseDecodePagedAttentionWrapper",
           "fp4_indexer_block_scores", "build_k2q_csr", "SparseK2qCsrBuilderSm100",
           "BatchDecodeIndexerQ8KV4Wrapper", "BatchDecodeIndexerQ8KV8Wrapper",
           "BatchPrefillIndexerQ8KV8Wrapper", "kvouter_attention",
           "can_run_sparse_kvouter"}
_MODULES = {"icp", "icp_decode_score"}
_WARMUP = {"warmup", "plan_warmup"}
__all__ = sorted(_DENSE | _SPARSE | _MODULES | _WARMUP | {"register_tvm_ffi"})


def __getattr__(name):
    if name in _WARMUP:
        return getattr(import_module(".msa_warmup", __name__), name)
    if name in _DENSE:
        return getattr(import_module(".api", __name__), name)
    if name in _SPARSE:
        return getattr(import_module(".sparse", __name__), name)
    if name in _MODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})


def _tvm_to_torch(x):
    import ctypes
    import torch

    if type(x).__module__ == "tvm_ffi.core" and type(x).__name__ == "Tensor":
        dtype = {"float8_e4m3fn": torch.float8_e4m3fn,
                 "float8_e5m2": torch.float8_e5m2}.get(str(x.dtype))
        if dtype is not None:
            capsule = x._to_dlpack()
            get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
            get_ptr.restype = ctypes.c_void_p
            get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
            # apache-tvm-ffi<=0.1.10 labels FP8 capsules as DLPack bool.
            # Reinterpret their bytes as int8 before recovering the FP8 view.
            dtype_addr = get_ptr(capsule, b"dltensor") + 20
            ctypes.c_uint8.from_address(dtype_addr).value = 0
            ctypes.c_uint8.from_address(dtype_addr + 1).value = 8
            ctypes.c_uint16.from_address(dtype_addr + 2).value = 1
            return torch.from_dlpack(capsule).view(dtype)
        return torch.from_dlpack(x)
    if type(x).__module__ == "tvm_ffi.container" and type(x).__name__ == "Map":
        return {str(key): _tvm_to_torch(x[key]) for key in x}
    return x


_registered_ffi_namespaces = set()


def register_tvm_ffi(namespace="minfer.ops"):
    """Register legacy TVM entry points explicitly, without replacing an owner.

    Import calls this by default, preserving the legacy TVM global API.
    ``MSA_REGISTER_TVM_FFI=0`` disables automatic registration. A vendored MSA may own
    ``minfer.ops``; choose another namespace when both copies need TVM globals.
    Calling again for a namespace registered by this module is a no-op.
    """
    if namespace in _registered_ffi_namespaces:
        return
    import tvm_ffi

    names = [f"{namespace}.{name}" for name in sorted(_DENSE)]
    occupied = [name for name in names
                if tvm_ffi.get_global_func(name, allow_missing=True) is not None]
    if occupied:
        raise RuntimeError(
            "TVM functions already registered; select a separate namespace: "
            + ", ".join(occupied))
    for name in sorted(_DENSE):
        def invoke(*args, _name=name):
            function = getattr(import_module(".api", __name__), _name)
            return function(*[_tvm_to_torch(arg) for arg in args])

        tvm_ffi.register_global_func(f"{namespace}.{name}", invoke)
    _registered_ffi_namespaces.add(namespace)


if os.environ.get("MSA_REGISTER_TVM_FFI", "1") != "0":
    try:
        register_tvm_ffi()
    except ImportError:
        # The TVM dependency is optional for CPU-only planning/source inspection.
        pass
