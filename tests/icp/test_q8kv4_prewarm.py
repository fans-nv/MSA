# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Host checks for Q8KV4 prewarm, split-policy defaults and callback ownership."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fmha_sm100 import _jit_cache
from fmha_sm100.decode_q8kv4 import _build_utils, interface, jit
from fmha_sm100.icp import _jit_guard, prewarm


def _isolated(code, tmp_path):
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "python"),
        PYTHONDONTWRITEBYTECODE="1",
        MSA_REGISTER_TVM_FFI="0",
        ICP_CACHE_ROOT=str(tmp_path / "icp"),
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


@pytest.mark.parametrize(
    "splits,mode,expected",
    [
        (None, None, (8, "streamk")),
        (1, None, (1, "legacy")),
        (1, "streamk", (1, "streamk")),
    ],
)
def test_default_split_selection_and_explicit_streamk1(
    monkeypatch, splits, mode, expected
):
    monkeypatch.delenv("MSA_Q8KV4_SPLIT_MODE", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=148),
    )
    monkeypatch.setattr(jit, "_target_arch", lambda *_: "107a")
    monkeypatch.setattr(interface, "_make_backend_plan", lambda *args, **kw: kw)
    result = interface._prepare_decode_plan(
        1,
        1,
        device=0,
        num_q_heads=32,
        num_kv_heads=2,
        topk=16,
        num_kv_splits=splits,
        split_mode=mode,
    )
    assert (result["num_kv_splits"], result["split_mode"]) == expected


@pytest.mark.parametrize("old_vendor_first", [False, True])
def test_native_callbacks_survive_legacy_vendor_import_order(
    tmp_path, old_vendor_first
):
    _isolated(
        f"""
import types
import torch
import tvm_ffi
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA initialized')
torch.cuda._lazy_init = forbidden
names = ('jit_get_plan', 'jit_get_reduction', 'jit_get_fmha_fwd_sparse_variant')
def legacy_vendor():
    for name in names:
        tvm_ffi.register_global_func('fmha_sm100.decode_q8kv4.' + name,
                                    lambda *args: 'old-vendor', override=True)
if {old_vendor_first!r}:
    legacy_vendor()
import fmha_sm100.decode_q8kv4 as package
package.get_plan_fn = lambda *_: types.SimpleNamespace(plan='new-canonical')
if not {old_vendor_first!r}:
    legacy_vendor()
assert tvm_ffi.get_global_func('fmha_sm100.decode_q8kv4.native_v2.jit_get_plan')() == 'new-canonical'
for name in names:
    assert tvm_ffi.get_global_func('fmha_sm100.decode_q8kv4.' + name)() == 'old-vendor'
assert not torch.cuda.is_initialized()
""",
        tmp_path,
    )


def test_host_extension_identity_is_namespace_specific_and_path_is_read_only(
    q8_cache, tmp_path, monkeypatch
):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    canonical = interface.cpp_extension_name()
    path = interface.cpp_extension_path()
    assert path.name == canonical + ".so"
    assert not path.parent.exists()
    monkeypatch.setattr(interface, "__package__", "vendor.fmha_sm100.decode_q8kv4")
    assert interface.cpp_extension_name() != canonical
    flags = " ".join(interface._cpp_compile_flags())
    assert "vendor.fmha_sm100.decode_q8kv4.native_v2" in flags
    assert "MSA_Q8KV4_PYTHON_PACKAGE" in flags


def test_host_extension_identity_tracks_resolved_compilers(q8_cache, monkeypatch):
    before = interface.cpp_extension_name()
    nvcc = _build_utils.cuda_home() / "bin/nvcc"
    nvcc.write_text(nvcc.read_text() + "# a different toolkit at the same path\n")
    assert interface.cpp_extension_name() != before
    before = interface.cpp_extension_name()
    monkeypatch.setenv("CXX", "c++ -DMSA_TOOLCHAIN_IDENTITY_TEST=1")
    assert interface.cpp_extension_name() != before


def test_host_extension_load_waits_for_other_rank_to_finish_linking(
    q8_cache, monkeypatch
):
    # The producer holds exactly the lock used by the real builder. A consumer
    # must not open the partial library even though its final path already exists.
    path = interface.cpp_extension_path()
    entered = threading.Event()
    loaded = threading.Event()

    def load(completed):
        assert completed.read_bytes() == b"complete"
        loaded.set()
        return "loaded"

    monkeypatch.setattr(interface, "_load_or_build_cpp_backend", load)

    def consumer():
        entered.set()
        return interface._jit_compile_cpp_backend()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with _build_utils.build_lock(path.parent):
            path.write_bytes(b"partially linked")
            pending = executor.submit(consumer)
            assert entered.wait(5)
            assert not loaded.wait(0.1)
            path.write_bytes(b"complete")
        assert pending.result(timeout=5) == "loaded"


@pytest.fixture
def q8_cache(tmp_path, monkeypatch):
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin/nvcc"
    nvcc.write_text(
        "#!/bin/sh\nprintf 'Cuda compilation tools, release 13.4, V13.4.0\\n'\n"
    )
    nvcc.chmod(0o700)
    monkeypatch.setenv("CUDA_HOME", str(toolkit))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))
    monkeypatch.setenv("ICP_CACHE_ROOT", str(tmp_path / "icp"))
    monkeypatch.delenv("MINFER_FMHA_CACHE_DIR", raising=False)
    monkeypatch.delenv(jit.DISABLE_QMUL4_ENV, raising=False)
    _build_utils.cuda_home.cache_clear()
    _build_utils.cuda_version.cache_clear()
    _jit_cache.clear_memo()
    result = _build_utils.qmul4_result_path(
        "107a", probe_root=jit._cache_root() / "capability_probes"
    )
    result.parent.mkdir(parents=True)
    result.write_text("unsupported\n")
    yield result
    _build_utils.cuda_home.cache_clear()
    _build_utils.cuda_version.cache_clear()
    _jit_cache.clear_memo()


def _publish_fixture(recipe, source):
    recipe.namespace.ensure()
    recipe.dir.mkdir(parents=True)
    library = recipe.dir / "fixture.so"
    library.write_bytes(b"artifact integrity fixture, never loaded")
    entry = {
        "schema": _jit_cache.SCHEMA,
        "recipe": recipe.recipe,
        "records": [
            {
                "library": library.name,
                "inputs": {
                    str(source): hashlib.sha256(source.read_bytes()).hexdigest()
                },
            }
        ],
    }
    (recipe.dir / "entry.json").write_text(json.dumps(entry))
    return library


def test_q8kv4_verification_is_read_only_and_checks_each_recipe(
    q8_cache, tmp_path, monkeypatch
):
    source = tmp_path / "native-input.h"
    source.write_text("the compiler's recorded input")
    recipes = jit.icp_prewarm_recipes("107a", read_only=True)
    assert len(recipes) == 3
    out = []
    for name, recipe in recipes.items():
        library = _publish_fixture(recipe, source)
        out.append(prewarm._artifact("q8kv4", name, library))
        out.append(
            prewarm._artifact("q8kv4", name + ".entry.json", recipe.dir / "entry.json")
        )
    cpp = interface.cpp_extension_path()
    cpp.parent.mkdir(parents=True)
    cpp.write_bytes(b"host extension fixture, never loaded")
    out += [
        prewarm._artifact("q8kv4", "cpp_ext", cpp),
        prewarm._artifact("q8kv4", "qmul4.result", q8_cache),
        prewarm._artifact(
            "q8kv4", "namespace.json", recipes["plan"].namespace.path / "manifest.json"
        ),
    ]
    monkeypatch.setitem(prewarm.BUILDERS, "q8kv4", lambda arch: out)
    monkeypatch.setattr(prewarm, "toolchain", lambda: {})
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_107a")
    prewarm.build("107a", components=("q8kv4",), commit="fixture")
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("verification invoked a compiler or mutated the cache")

    monkeypatch.setattr(_build_utils, "_probe_qmul4", forbidden)
    monkeypatch.setattr(_jit_cache.Recipe, "build", forbidden)
    monkeypatch.setattr(jit, "_run_ninja", forbidden)
    assert prewarm.check("107a", check_toolchain=False, components=("q8kv4",)) == []
    assert before == {
        str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }

    source.write_text("changed native compiler input")
    _jit_cache.clear_memo()
    problems = prewarm.check("107a", check_toolchain=False, components=("q8kv4",))
    assert (
        sum("no current dev JIT source record" in problem for problem in problems) == 3
    )
    q8_cache.unlink()
    assert any(
        "capability record" in problem
        for problem in prewarm.check(
            "107a", check_toolchain=False, components=("q8kv4",)
        )
    )


@pytest.mark.parametrize("kind", ["host", "forward", "plan", "capability"])
def test_runtime_fail_policy_blocks_every_q8kv4_compiler_entry(
    q8_cache, tmp_path, monkeypatch, kind
):
    monkeypatch.setenv("ICP_RUNTIME_JIT", "fail")
    monkeypatch.setattr(jit, "_dequant_mode", lambda arch: "fp16_fallback")
    recipes = jit.icp_prewarm_recipes("107a", read_only=True)
    namespace = recipes["plan"].namespace
    monkeypatch.setattr(jit, "_namespace", lambda: namespace)
    monkeypatch.setattr(jit, "_cuda_home", lambda: Path(__file__).parent)

    def forbidden(*args, **kwargs):
        pytest.fail("fail policy reached the compiler")

    monkeypatch.setattr(jit, "_run_ninja", forbidden)
    run = _build_utils.subprocess.run

    def version_only(args, **kwargs):
        if args[-1] != "--version":
            forbidden()
        return run(args, **kwargs)

    monkeypatch.setattr(_build_utils.subprocess, "run", version_only)
    with pytest.raises(_jit_guard.RuntimeJitError):
        if kind == "host":
            interface._jit_compile_cpp_backend()
        elif kind == "forward":
            jit.JitSpec(
                "decode_attention_q8kv4", False, "fp16_fallback", "107a", 16, 3
            ).build()
        elif kind == "plan":
            jit._build_fixed_module(
                "decode_attention_plan",
                jit._SOURCES / "decode_attention_plan.cu",
                "plan",
                "107a",
            )
        else:
            _build_utils._probe_qmul4(
                tmp_path / "nvcc", "107a", tmp_path / "uncached-probe"
            )
