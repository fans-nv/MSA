# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""CPU contracts for the shared MSA/ICP import and dispatch surfaces.

Native numerical tests live in tests/icp. These tests never compile or launch a
kernel. TVM coexistence exercises its real process-global registry; the legacy
owner callbacks stand in for the unmaterialized serving-image vendor package.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"


def _isolated(code, tmp_path, *, register="0"):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(PYTHON_ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        MSA_REGISTER_TVM_FFI=register,
        ICP_CACHE_ROOT=str(tmp_path / "icp"),
        ICP_KERNEL_ARCH="107a",
        TORCH_EXTENSIONS_DIR=str(tmp_path / "extensions"),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_icp_import_is_lazy_and_aliases_share_implementation(tmp_path):
    _isolated(
        """
import sys
import fmha_sm100
import fmha_sm100.icp as icp
from fmha_sm100.icp import local_indexer
assert icp.ICP_INTEGRATION_ABI == 1
assert 'torch' not in sys.modules
assert 'cutlass' not in sys.modules
assert 'tvm_ffi' not in sys.modules
from fmha_sm100.icp.scorer.prefill import api, jit
from fmha_sm100 import api as shared_api, jit as shared_jit
assert api is shared_api and jit is shared_jit
assert api.ICP_DEVICE_PLAN_ABI_VERSION == 3
from fmha_sm100.icp.scorer.decode import icp_decode_score as decode
from fmha_sm100 import icp_decode_score as decode_alias
assert decode_alias is decode
assert decode.ICP_DECODE_SCORE_LAUNCH_ABI_VERSION == 3
import torch
assert not torch.cuda.is_initialized()
assert 'cutlass' not in sys.modules
assert 'src' not in sys.modules and 'interface' not in sys.modules
""",
        tmp_path,
    )


def test_default_tvm_registration_keeps_lazy_callable_api(tmp_path):
    pytest.importorskip("tvm_ffi")
    _isolated(
        """
import sys, types
import tvm_ffi
import fmha_sm100
assert 'fmha_sm100.api' not in sys.modules
fake_api = types.ModuleType('fmha_sm100.api')
fake_api.fmha_sm100_plan = lambda x: x + 7
fake_api.fmha_sm100 = lambda x: (x, x + 1)
fake_api.sparse_topk_select = lambda x: x + 9
sys.modules['fmha_sm100.api'] = fake_api
assert tvm_ffi.get_global_func('minfer.ops.fmha_sm100_plan')(4) == 11
assert tvm_ffi.get_global_func('minfer.ops.sparse_topk_select')(4) == 13
assert list(tvm_ffi.get_global_func('minfer.ops.fmha_sm100')(4)) == [4, 5]
fmha_sm100.register_tvm_ffi()  # idempotent for our own registration
import torch
assert not torch.cuda.is_initialized()
assert 'cutlass' not in sys.modules
""",
        tmp_path,
        register="1",
    )


@pytest.mark.parametrize("vendor_first", [False, True])
def test_explicit_opt_out_preserves_existing_tvm_owner(tmp_path, vendor_first):
    pytest.importorskip("tvm_ffi")
    _isolated(
        f"""
import tvm_ffi
names = ('fmha_sm100', 'fmha_sm100_plan', 'sparse_topk_select')
def vendor_import():
    for name in names:
        tvm_ffi.register_global_func('minfer.ops.' + name, lambda: 'vendor')
if {vendor_first!r}:
    vendor_import()
import fmha_sm100.icp
if not {vendor_first!r}:
    vendor_import()
for name in names:
    assert tvm_ffi.get_global_func('minfer.ops.' + name)() == 'vendor'
import fmha_sm100
try:
    fmha_sm100.register_tvm_ffi()
except RuntimeError as error:
    assert 'already registered' in str(error)
else:
    raise AssertionError('canonical registration replaced the vendor')
for name in names:
    assert tvm_ffi.get_global_func('minfer.ops.' + name)() == 'vendor'
""",
        tmp_path,
    )


def test_generic_fmha_and_nvfp4_variants_remain_separate_from_icp():
    from fmha_sm100 import api, jit
    import inspect
    import torch

    assert "nvfp4_kv" not in inspect.signature(api._fmha_sm100).parameters
    assert "kv_dtype" in inspect.signature(jit.get_fmha_variant).parameters
    for mode, expected in enumerate(("Sparse", "Full", "OnlyScore", "Off")):
        name, params = jit._variant_key_from_runtime(
            jit._BFLOAT16_CODE,
            128,
            False,
            mode,
            128,
            False,
            1,
        )
        assert params["sparse_mode"] == expected
        assert params["kv_mode"] == 0
        assert name.split("_")[3] == str(mode), "ordinary prebuild/cache names changed"
        assert jit._variant_params_from_name(name) == params
    ordinary = (
        jit._dlpack_dtype_code(torch.float8_e4m3fn),
        128,
        True,
        0,
        128,
        False,
        1,
    )
    plain_name, plain = jit._variant_key_from_runtime(*ordinary)
    assert jit._variant_key_from_runtime(*ordinary, kv_dtype="fp8") == (
        plain_name,
        plain,
    )
    nvfp4_name, nvfp4 = jit._variant_key_from_runtime(*ordinary, kv_dtype="nvfp4")
    assert nvfp4_name == plain_name + "_nvfp4_v1"
    assert nvfp4["kv_mode"] == 3
    assert jit._variant_params_from_name(nvfp4_name) == nvfp4
    _, params = jit._variant_key_from_runtime(
        jit._BFLOAT16_CODE,
        128,
        False,
        4,
        64,
        False,
        1,
    )
    assert params["sparse_mode"] == "OnlyScoreIcp"
    with pytest.raises(ValueError, match="Impossible"):
        jit._variant_key_from_runtime(
            jit._BFLOAT16_CODE,
            128,
            False,
            4,
            64,
            True,
            1,
        )
    assert list(inspect.signature(api.sparse_topk_select).parameters) == [
        "max_score",
        "topk",
        "num_valid_pages",
        "output",
        "force_begin_blocks",
        "force_end_blocks",
        "max_score_layout",
        "block_table",
    ]


def test_sparse_surface_preserves_original_quantization_helpers(tmp_path):
    # Isolate native-leaf substitutes from subsequent real import tests. Python
    # caches imported children on their parent modules as well as in sys.modules.
    _isolated(
        """
import importlib, sys, types
# Quantization is the real public module. Only native/DSL-facing leaves are
# substituted while verifying how the sparse facade binds and exports them.
leaves = {
    'fmha_sm100.cute.interface': [
        'SparseDecodePagedAttentionWrapper', 'sparse_atten_func',
        'sparse_atten_nvfp4_kv_func', 'sparse_atten_nvfp4_kv_cache_func',
        'sparse_decode_atten_func',
    ],
    'fmha_sm100.cute.sparse_index_utils': ['build_k2q_csr'],
    'fmha_sm100.cute.src.sm100.prepare_k2q_csr': ['SparseK2qCsrBuilderSm100'],
    'fmha_sm100.cute.fp4_indexer_interface': ['fp4_indexer_block_scores'],
    'fmha_sm100.cute.q8_indexer_interface': [
        'BatchDecodeIndexerQ8KV4Wrapper', 'BatchDecodeIndexerQ8KV8Wrapper',
        'BatchPrefillIndexerQ8KV8Wrapper',
    ],
    'fmha_sm100.kvouter': ['can_run_sparse_kvouter', 'kvouter_attention'],
}
for name, symbols in leaves.items():
    module = types.ModuleType(name)
    for symbol in symbols:
        setattr(module, symbol, object())
    sys.modules[name] = module
bound_loaders = []
sys.modules['fmha_sm100.cute.q8_indexer_interface'].bind_indexer_module_loader = (
    bound_loaders.append
)
old_path = list(sys.path)
sparse = importlib.import_module('fmha_sm100.sparse')
quantize = importlib.import_module('fmha_sm100.cute.quantize')
for name in (
    'quantize_bf16_to_nvfp4_128x4', 'quantize_kv_bf16_to_nvfp4_128x4',
    'dequantize_nvfp4_128x4_to_bf16', 'swizzle_nvfp4_scale_to_128x4',
    'nvfp4_global_scale_from_amax', 'Nvfp4QuantizedTensor',
):
    assert name in sparse.__all__
    assert getattr(sparse, name) is getattr(quantize, name)
from fmha_sm100 import jit
assert bound_loaders == [jit.get_indexer_module]
for module_name in ('fmha_sm100.cute.q8_indexer_interface', 'fmha_sm100.kvouter'):
    for name in leaves[module_name]:
        assert name in sparse.__all__
        assert getattr(sparse, name) is getattr(sys.modules[module_name], name)
assert sys.path == old_path
assert 'src' not in sys.modules and 'interface' not in sys.modules
""",
        tmp_path,
    )


def test_generic_topk_loader_uses_public_native_exports(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from fmha_sm100 import jit

    ffi = pytest.importorskip("tvm_ffi")
    initialized, compiled = [], []
    public_module = SimpleNamespace(
        sparse_topk_select=lambda: None,
        sparse_topk_select_init=lambda: initialized.append(True),
    )
    monkeypatch.setattr(jit, "_modules", {})

    def build_module(key):
        compiled.append(key)
        return tmp_path / "sparse_topk_select.so"

    monkeypatch.setattr(jit, "build_module", build_module)
    monkeypatch.setattr(ffi, "load_module", lambda path: public_module)
    assert jit.get_sparse_topk_module() is public_module
    assert jit.get_sparse_topk_module() is public_module
    assert compiled == ["sparse_topk"]
    assert initialized == [True]


def test_ordinary_warmup_builds_csr_through_the_canonical_namespace(tmp_path):
    _isolated(
        """
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.utils.cpp_extension as extensions
def forbidden(*args, **kwargs):
    raise AssertionError('warmup host control initialized CUDA')
torch.cuda._lazy_init = forbidden
compiled = []
def load(*args, **kwargs):
    compiled.extend(Path(source).name for source in kwargs['sources'])
    return SimpleNamespace()
extensions.load = load
from fmha_sm100 import plan_warmup
items = plan_warmup('bf16', None, 32, 2, 16, sparse_decode=False,
                    prefill_backend='cute_dsl')
assert not compiled, 'constructing a warmup plan must not compile it'
builder, = [item for item in items if item.label == 'k2q CSR builder']
old_path = list(sys.path)
builder.build()
assert 'build_k2q_csr.cu' in compiled
assert sys.path == old_path
assert 'src' not in sys.modules and 'interface' not in sys.modules
""",
        tmp_path,
    )
