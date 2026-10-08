"""Cache keying, runtime-JIT policy and the prewarm verifier, on the CPU.

Each property has its negative control: a source edit MUST move the key, a
missing or altered artifact MUST fail verification, ``fail`` MUST refuse.
"""

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from contextlib import nullcontext

import pytest
import torch

from fmha_sm100.icp import _cache, _jit_guard, prewarm

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def pkg_copy(tmp_path, monkeypatch):
    """A private copy of the package sources so edits never touch the tree."""
    dst = tmp_path / "fmha_sm100"
    shutil.copytree(
        _cache.SOURCE_ROOT, dst, ignore=shutil.ignore_patterns("__pycache__", "cutlass")
    )
    monkeypatch.setattr(_cache, "SOURCE_ROOT", dst)
    monkeypatch.setattr(_cache, "PACKAGE", dst / "icp")
    return dst


@pytest.fixture
def root(tmp_path, monkeypatch):
    r = tmp_path / "cache"
    monkeypatch.setenv(_cache.ROOT_ENV, str(r))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "torch-cache"))
    for env in (*_cache.LEGACY_ENV.values(), _jit_guard.POLICY_ENV):
        monkeypatch.delenv(env, raising=False)
    return r


@pytest.mark.parametrize(
    "component,edit",
    [
        ("icp", "icp/csrc/merge_topk.cuh"),
        ("icp", "icp/_build.py"),
        ("fmha", "csrc/include/sm100_fmha_reduction.hpp"),
        ("fmha", "jit.py"),
        ("fmha", "_jit_cache.py"),
        ("decode", "icp/scorer/decode/_icp_decode_score_kernel.py"),
        ("nvfp4", "cute/src/sm100/fwd/atten_fwd_nvfp4_kv.py"),
    ],
)
def test_a_source_edit_moves_only_its_components_directory(
    pkg_copy, root, component, edit
):
    before = {c: _cache.component_dir(c, "107a") for c in _cache.COMPONENTS}
    with open(pkg_copy / edit, "a") as f:
        f.write("\n// edit\n" if not edit.endswith(".py") else "\n# edit\n")
    after = {c: _cache.component_dir(c, "107a") for c in _cache.COMPONENTS}
    assert before[component] != after[component]
    assert {c for c in _cache.COMPONENTS if before[c] != after[c]} == {component}


def test_arch_is_in_the_key_and_normalised(root):
    assert _cache.component_dir("icp", "sm_107A").parent.name == "sm_107a"
    assert _cache.component_dir("icp", "103") != _cache.component_dir("icp", "107a")
    assert _cache.normalize_arch((10, 7)) == "107a"
    with pytest.raises(ValueError):
        _cache.normalize_arch("sm_x")


def test_cutlass_header_edit_invalidates_the_fmha_prewarm(pkg_copy, root):
    header = pkg_copy / "cutlass/include/cutlass/cutlass.h"
    header.parent.mkdir(parents=True)
    header.write_text("// first reviewed header\n")
    before = _cache.component_digest("fmha")
    header.write_text("// changed compiler input\n")
    assert _cache.component_digest("fmha") != before


def test_fmha_prewarm_records_dev_libraries_and_source_records(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fmha_sm100 import jit

    def compiled(name):
        directory = tmp_path / "v2" / "toolchain" / (name + "-recipe")
        directory.mkdir(parents=True)
        library = directory / (name + "-inputs.so")
        library.write_bytes(b"compiled object")
        (directory / "entry.json").write_text("{}")
        return library

    seen = []

    def compile_variant(name, params):
        assert params["sparse_mode"] == "OnlyScoreIcp"
        seen.append((params["page_size"], params["single_wg"]))
        return compiled(name)

    namespace = tmp_path / "v2/toolchain"
    namespace.mkdir(parents=True)
    (namespace / "manifest.json").write_text("{}")
    monkeypatch.setattr(jit._variant_manager, "compile_locked", compile_variant)
    monkeypatch.setattr(jit, "build_module", compiled)
    monkeypatch.setattr(jit, "_namespace", lambda: SimpleNamespace(path=namespace))
    records = prewarm._build_fmha()
    assert set(seen) == {(64, "true"), (64, "false"), (32, "true"), (32, "false")}
    assert {item["name"] for item in records} == prewarm._required_artifact_names(
        "fmha"
    )
    assert all(Path(item["path"]).is_file() for item in records)


def test_key_arch_never_initialises_cuda():
    pytest.importorskip("torch")
    # A fresh process is essential: earlier GPU tests may already own a context.
    # Record forbidden calls even if key_arch catches the exception internally.
    code = """
import os
import torch
from fmha_sm100.icp import _cache
assert not torch.cuda.is_initialized()
calls = []
def forbidden(*args, **kwargs):
    calls.append((args, kwargs))
    raise AssertionError('architecture lookup initialized or probed CUDA')
torch.cuda._lazy_init = forbidden
torch.cuda.get_device_capability = forbidden
os.environ['ICP_KERNEL_ARCH'] = 'sm_107a'
assert _cache.key_arch() == '107a'
del os.environ['ICP_KERNEL_ARCH']
_cache.key_arch()
assert not calls
assert not torch.cuda.is_initialized()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def test_legacy_override_is_verbatim(root, monkeypatch, tmp_path):
    monkeypatch.setenv("ICP_KERNEL_CACHE", str(tmp_path / "legacy"))
    assert _cache.component_dir("icp", "107a") == tmp_path / "legacy"
    with pytest.raises(SystemExit, match="unkeyed"):
        prewarm.build("107a", components=())


def test_policy_defaults_follow_the_manifest(root, monkeypatch):
    monkeypatch.setenv("ICP_KERNEL_ARCH", "107a")
    assert _jit_guard.policy() == "allow"
    path = _cache.manifest_path("107a")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert _jit_guard.policy() == "warn"
    monkeypatch.setenv(_jit_guard.POLICY_ENV, "fail")
    with pytest.raises(_jit_guard.RuntimeJitError, match="refused"):
        _jit_guard.on_compile("icp", "icp_k2", arch="107a")
    assert _jit_guard.events()[-1]["policy"] == "fail"
    monkeypatch.setenv(_jit_guard.POLICY_ENV, "sometimes")
    with pytest.raises(ValueError):
        _jit_guard.policy()


def test_build_dir_is_keyed_and_a_cold_build_consults_the_policy(root, monkeypatch):
    pytest.importorskip("torch")
    from fmha_sm100.icp import _build

    monkeypatch.setenv("ICP_KERNEL_ARCH", "107a")
    monkeypatch.setenv(_jit_guard.POLICY_ENV, "fail")
    directory = Path(_build.build_dir("icp_k2"))
    assert directory.parent == _cache.component_dir("icp", "107a")
    monkeypatch.setattr(_build, "_loaded", {})
    with pytest.raises(_jit_guard.RuntimeJitError):
        _build._load_extension("icp_k2")
    # A present artifact is a cache hit: the policy is not consulted.
    (directory / "icp_k2.so").write_bytes(b"")
    import torch.utils.cpp_extension as ext

    monkeypatch.setattr(ext, "load", lambda **kw: "loaded")
    assert _build._load_extension("icp_k2") == "loaded"


def _fake_manifest(arch, artifact):
    names = sorted(prewarm._required_artifact_names("icp"))
    manifest = {
        "schema": prewarm.SCHEMA,
        "arch": arch,
        "cache_root": str(_cache.root()),
        "components": {
            c: {
                "digest": _cache.component_digest(c),
                "files": _cache.file_hashes(c),
                "dir": str(_cache.component_dir(c, arch)),
                "artifacts": names,
            }
            for c in ("icp",)
        },
        "toolchain": {},
        "environment": prewarm.environment(),
        "artifacts": [
            {
                "component": "icp",
                "name": name,
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            for name in names
        ],
    }
    path = _cache.manifest_path(arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    return path


def test_verify_passes_clean_and_fails_each_negative_control(pkg_copy, root):
    arch = "107a"
    first = prewarm.check(arch, check_toolchain=False)[0]
    assert first.startswith("no prewarm manifest")
    artifact = _cache.component_dir("icp", arch) / "icp_k2_sm107a" / "icp_k2.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"object")
    _fake_manifest(arch, artifact)
    assert prewarm.check(arch, check_toolchain=False, components=("icp",)) == []

    artifact.write_bytes(b"tampered")
    problems = prewarm.check(arch, check_toolchain=False)
    assert any("sha256 differs" in p for p in problems)
    artifact.unlink()
    assert any("missing" in p for p in prewarm.check(arch, check_toolchain=False))
    artifact.write_bytes(b"object")
    with open(pkg_copy / "icp" / "csrc" / "k2_merge.cu", "a") as f:
        f.write("\n// dev edit\n")
    problems = prewarm.check(arch, check_toolchain=False)
    assert any("csrc/k2_merge.cu" in p and "will JIT" in p for p in problems)


def test_partial_manifest_cannot_pass_the_full_startup_check(pkg_copy, root, tmp_path):
    artifact = tmp_path / "fixture.so"
    artifact.write_bytes(b"object")
    _fake_manifest("107a", artifact)
    assert prewarm.check("107a", check_toolchain=False, components=("icp",)) == []
    problems = prewarm.check("107a", check_toolchain=False)
    assert any(
        "missing required prewarm components" in p and "nvfp4" in p for p in problems
    )


def test_build_records_the_complete_component_inventory(
    pkg_copy, root, tmp_path, monkeypatch
):
    artifact = tmp_path / "fixture.so"
    artifact.write_bytes(b"object")
    names = sorted(prewarm._required_artifact_names("icp"))
    records = [prewarm._artifact("icp", name, artifact) for name in names]
    monkeypatch.setitem(prewarm.BUILDERS, "icp", lambda arch: records)
    monkeypatch.setattr(prewarm, "toolchain", lambda: {})
    for key, value in (
        ("ICP_RUNTIME_JIT", "allow"),
        ("ICP_KERNEL_ARCH", "107a"),
        ("CUTE_DSL_ARCH", "sm_107a"),
    ):
        monkeypatch.setenv(key, value)
    manifest = prewarm.build("107a", components=("icp",), commit="fixture")
    assert manifest["components"]["icp"]["artifacts"] == names
    assert prewarm.check("107a", check_toolchain=False, components=("icp",)) == []
    assert prewarm.check("107a", check_toolchain=False)
    monkeypatch.setitem(prewarm.BUILDERS, "icp", lambda arch: [])
    with pytest.raises(RuntimeError, match="invalid inventory"):
        prewarm.build("107a", components=("icp",))


@pytest.mark.parametrize("remove_declared", [False, True])
def test_missing_artifact_records_fail_even_if_the_file_exists(
    pkg_copy, root, tmp_path, remove_declared
):
    artifact = tmp_path / "fixture.so"
    artifact.write_bytes(b"object")
    path = _fake_manifest("107a", artifact)
    manifest = json.loads(path.read_text())
    removed = manifest["artifacts"].pop()
    if remove_declared:
        manifest["components"]["icp"]["artifacts"].remove(removed["name"])
    path.write_text(json.dumps(manifest))
    problems = prewarm.check("107a", check_toolchain=False, components=("icp",))
    expected = "required profiles" if remove_declared else "artifact inventory differs"
    assert any(expected in p for p in problems)


def test_old_manifest_schema_requires_rebuilding(pkg_copy, root, tmp_path):
    artifact = tmp_path / "fixture.so"
    artifact.write_bytes(b"object")
    path = _fake_manifest("107a", artifact)
    manifest = json.loads(path.read_text())
    manifest["schema"] = "icp-kernels.prewarm.v1"
    path.write_text(json.dumps(manifest))
    assert any("rebuild" in p for p in prewarm.check("107a", check_toolchain=False))


def test_changed_torch_cache_directory_fails_with_old_artifacts_present(
    pkg_copy, root, tmp_path, monkeypatch
):
    artifact = tmp_path / "fixture.so"
    artifact.write_bytes(b"object")
    _fake_manifest("107a", artifact)
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "new-torch-cache"))
    problems = prewarm.check("107a", check_toolchain=False, components=("icp",))
    assert artifact.is_file()
    assert any("environment TORCH_EXTENSIONS_DIR" in p for p in problems)


def test_effective_cache_paths_and_disabled_aot_are_not_silently_accepted(
    pkg_copy, root, tmp_path, monkeypatch
):
    artifact = tmp_path / "fixture.so"
    artifact.write_bytes(b"object")
    path = _fake_manifest("107a", artifact)
    manifest = json.loads(path.read_text())
    manifest["cache_root"] = str(tmp_path / "old-cache-root")
    manifest["components"]["icp"]["dir"] = str(tmp_path / "old-component-dir")
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv("ICP_DECODE_AOT", "0")
    problems = prewarm.check("107a", check_toolchain=False)
    for expected in ("effective cache root", "runtime cache directory", "disables"):
        assert any(expected in p for p in problems)


def _aot_export_fixture(directory, monkeypatch, *, keys=None):
    from fmha_sm100.cute.src.common import aot_cache
    from fmha_sm100.icp.attention.nvfp4_prefill._prewarm import required_aot_keys

    class Compiled:
        def export_to_c(self, path, function_name):
            Path(path).write_bytes(f"object of {function_name}".encode())

    monkeypatch.setattr(aot_cache, "_AOT_CACHE_DIR", str(directory))
    monkeypatch.setattr(aot_cache, "_AOT_DISABLE", False)
    for key in required_aot_keys("107a") if keys is None else keys:
        aot_cache.save_aot(key, Compiled(), sources=["src/common/seqlen_info.py"])


def _nvfp4_manifest(tmp_path, monkeypatch):
    from fmha_sm100.icp.attention.nvfp4_prefill import k2q_extension_path

    artifact = tmp_path / "fixture.so"
    artifact.write_bytes(b"object")
    path = _fake_manifest("107a", artifact)
    manifest = json.loads(path.read_text())
    directory = _cache.component_dir("nvfp4", "107a")
    aot_directory = directory / "aot/v2/host-fixture"
    _aot_export_fixture(aot_directory, monkeypatch)
    k2q = k2q_extension_path("107a")
    k2q.parent.mkdir(parents=True, exist_ok=True)
    k2q.write_bytes(b"object")
    paths = {"k2q_ext": k2q, "aot-manifest.json": aot_directory / "manifest.json"}
    for obj in aot_directory.glob("*.o"):
        paths[obj.stem] = obj
        paths[obj.stem + ".json"] = obj.with_suffix(".json")
    manifest["components"] = {
        "nvfp4": {
            "digest": _cache.component_digest("nvfp4"),
            "files": _cache.file_hashes("nvfp4"),
            "dir": str(directory),
            "artifacts": list(paths),
        }
    }
    manifest["artifacts"] = [
        prewarm._artifact("nvfp4", name, destination)
        for name, destination in paths.items()
    ]
    path.write_text(json.dumps(manifest))
    return path, manifest, aot_directory


def test_nvfp4_prewarm_rejects_disabled_aot_even_if_manifest_matches(
    pkg_copy, root, tmp_path, monkeypatch
):
    from fmha_sm100.cute.src.common import aot_cache

    monkeypatch.setenv("MM_SPARSE_ATTN_AOT_DISABLE", "0")
    path, manifest, _ = _nvfp4_manifest(tmp_path, monkeypatch)
    assert prewarm.check("107a", check_toolchain=False, components=("nvfp4",)) == []

    # Changing the environment after import cannot retarget a frozen loader.
    with monkeypatch.context() as patch:
        patch.setattr(aot_cache, "_AOT_CACHE_DIR", str(tmp_path / "stale-aot"))
        problems = prewarm.check("107a", check_toolchain=False, components=("nvfp4",))
        assert any("runtime artifact path" in problem for problem in problems)
    with monkeypatch.context() as patch:
        patch.setattr(aot_cache, "_AOT_DISABLE", True)
        assert prewarm.check("107a", check_toolchain=False, components=("nvfp4",)) == [
            "NVFP4 AOT loader was imported with caching disabled"
        ]

    # Matching recorded environment cannot legitimize a cache-disable switch.
    monkeypatch.setenv("MM_SPARSE_ATTN_AOT_DISABLE", "1")
    manifest["environment"] = prewarm.environment()
    path.write_text(json.dumps(manifest))
    assert prewarm.check("107a", check_toolchain=False, components=("nvfp4",)) == [
        "MM_SPARSE_ATTN_AOT_DISABLE=1 disables prewarmed NVFP4 artifacts"
    ]


def test_nvfp4_aot_import_rejects_other_architectures_and_sources(
    tmp_path, monkeypatch
):
    from fmha_sm100.icp.attention.nvfp4_prefill._prewarm import _import_aot_objects

    source = tmp_path / "source/sm_107a/nvfp4-digest/aot/v2/toolchain"
    _aot_export_fixture(source, monkeypatch)
    for identity in ("sm_100a/nvfp4-digest", "sm_107a/nvfp4-other"):
        target = tmp_path / "target" / identity / "aot/v2/toolchain"
        with pytest.raises(SystemExit, match="sources, arch or DSL differ"):
            _import_aot_objects(source, target)
        assert not target.exists()
    target = tmp_path / "target/sm_107a/nvfp4-digest/aot/v2/toolchain"
    _import_aot_objects(source, target)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == {
        p.name: p.read_bytes() for p in source.iterdir()
    }


def test_incomplete_nvfp4_export_is_rejected_before_copying(tmp_path):
    from fmha_sm100.icp.attention.nvfp4_prefill._prewarm import _import_aot_objects

    source = tmp_path / "source/sm_107a/nvfp4-digest/aot/v2/toolchain"
    source.mkdir(parents=True)
    (source / "combine_fixture.o").write_bytes(b"object")
    target = tmp_path / "target/sm_107a/nvfp4-digest/aot/v2/toolchain"
    with pytest.raises(SystemExit, match="missing AOT families"):
        _import_aot_objects(source, target)
    assert not target.exists()


def test_nvfp4_export_without_source_metadata_is_not_a_cache_hit(tmp_path, monkeypatch):
    from fmha_sm100.icp.attention.nvfp4_prefill._prewarm import _import_aot_objects

    source = tmp_path / "source/sm_107a/nvfp4-digest/aot/v2/toolchain"
    target = tmp_path / "target/sm_107a/nvfp4-digest/aot/v2/toolchain"
    _aot_export_fixture(source, monkeypatch)
    next(source.glob("combine_*.json")).unlink()
    with pytest.raises(SystemExit, match="invalid AOT metadata"):
        _import_aot_objects(source, target)
    assert not target.exists()


def _captured_production_call(
    monkeypatch, *, compact=False, capability=(10, 7), q_dtype="bfloat16"
):
    """Run real public forward/combine host paths; substitute only native leaves."""
    from fmha_sm100.cute import interface
    from fmha_sm100.cute.src.common import aot_cache
    from fmha_sm100.cute.src.sm100.prepare_scheduler import SparseAttentionSchedule
    from fmha_sm100.icp.attention import nvfp4_prefill
    from fmha_sm100.icp.attention.nvfp4_prefill import _prewarm

    combine = importlib.import_module("fmha_sm100.cute.src.sm100.fwd.combine")
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("host prewarm control tried to compile or initialize CUDA")

    def load(key):
        return lambda *args: calls.append((key, args))

    def csr(q2k, cu_q, cu_k, block, **kwargs):
        heads, rows, topk = q2k.shape
        indices = torch.zeros((heads, rows * topk), dtype=torch.int32)
        schedule = SparseAttentionSchedule(
            enabled=True,
            scheduler_metadata=torch.zeros((1, 6), dtype=torch.int32),
            work_count=torch.ones(1, dtype=torch.int32),
            qsplit_indices=torch.zeros_like(indices),
            split_counts=torch.ones((rows, heads), dtype=torch.int32),
        )
        return (
            torch.zeros((heads, kwargs["total_rows"] + 1), dtype=torch.int32),
            indices,
            schedule,
        )

    def attend(q, k, v, k_sf, v_sf, *args, **kwargs):
        if compact:
            k, v, k_sf, v_sf = (t.contiguous() for t in (k, v, k_sf, v_sf))
        return interface.sparse_atten_nvfp4_kv_cache_func(
            q, k, v, k_sf, v_sf, *args, **kwargs
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "_lazy_init", forbidden)
        patch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
        patch.setattr(torch.cuda.nvtx, "range", lambda name: nullcontext())
        patch.setattr(interface.cute, "compile", forbidden)
        patch.setattr(interface, "_compile_cache", {})
        patch.setattr(combine, "_combine_compile_cache", {})
        patch.setattr(aot_cache, "try_load_aot", load)
        # Set module dictionaries directly: probing the lazy CSR attribute
        # would itself import its native extension before the replacement.
        patch.setitem(nvfp4_prefill.__dict__, "build_k2q_csr", csr)
        patch.setitem(nvfp4_prefill.__dict__, "sparse_atten_nvfp4_kv_func", attend)
        _, inputs = _prewarm.production_call(
            q_dtype, q_lens=(2, 1), k_lens=(256, 128), device="cpu"
        )
    return calls, inputs


def test_production_fixture_matches_compound_abi_and_actual_loader_keys(monkeypatch):
    from fmha_sm100.icp.attention.nvfp4_prefill._prewarm import required_aot_keys

    calls, inputs = _captured_production_call(monkeypatch)
    assert len(calls) == 2
    assert {key for key, _ in calls} == set(required_aot_keys("107a"))
    expected = {"k": (0, 64), "k_sf": (16384, 8), "v": (18432, 64), "v_sf": (34816, 8)}
    storage_pointer = inputs["k"].untyped_storage().data_ptr()
    for name, (offset, width) in expected.items():
        value = inputs[name]
        assert value.shape == (6, 2, 128, width)
        assert value.dtype == torch.uint8
        assert value.stride() == (45056, 128 * width, width, 1)
        assert value.storage_offset() == offset
        assert value.untyped_storage().data_ptr() == storage_pointer
    assert inputs["q"].dtype == torch.bfloat16
    assert inputs["q"].shape == (3, 32, 128)
    # Launch arguments are the actual cache views, and globals stay absent.
    for index, name in enumerate(("k", "v", "k_sf", "v_sf")):
        assert calls[0][1][index] is inputs[name]
    assert calls[0][1][4:6] == (None, None)
    assert calls[1][0][1:3] == ((10, 7), 3)


@pytest.mark.parametrize("variant", ["compact", "sm100", "fp8_q", "dequant"])
def test_import_rejects_other_valid_variants_from_actual_loaders(
    tmp_path, monkeypatch, variant
):
    from fmha_sm100.icp.attention.nvfp4_prefill._prewarm import _import_aot_objects

    monkeypatch.setenv("MINIMAX_KVFP4_FP8_PAIR_DEQUANT", "1")
    with monkeypatch.context() as patch:
        if variant == "dequant":
            patch.setenv("MINIMAX_KVFP4_FP8_PAIR_DEQUANT", "0")
        calls, _ = _captured_production_call(
            patch,
            compact=variant == "compact",
            capability=(10, 0) if variant == "sm100" else (10, 7),
            q_dtype="float8_e4m3fn" if variant == "fp8_q" else "bfloat16",
        )
    source = tmp_path / "source/sm_107a/nvfp4-digest/aot/v2/toolchain"
    target = tmp_path / "target/sm_107a/nvfp4-digest/aot/v2/toolchain"
    _aot_export_fixture(source, monkeypatch, keys=[key for key, _ in calls])
    with pytest.raises(SystemExit, match="missing required production AOT keys"):
        _import_aot_objects(source, target)
    assert not target.exists()


def test_import_requires_exact_loader_filenames(tmp_path, monkeypatch):
    from fmha_sm100.icp.attention.nvfp4_prefill._prewarm import _import_aot_objects

    source = tmp_path / "source/sm_107a/nvfp4-digest/aot/v2/toolchain"
    target = tmp_path / "target/sm_107a/nvfp4-digest/aot/v2/toolchain"
    _aot_export_fixture(source, monkeypatch)
    obj = next(source.glob("sparse_forward*.o"))
    # Same family, valid metadata/key/hash suffix, wrong loader filename.
    obj.rename(
        obj.with_name(
            obj.stem.rsplit("_", 1)[0] + "_other_" + obj.name.rsplit("_", 1)[1]
        )
    )
    metadata = obj.with_suffix(".json")
    metadata.rename(
        metadata.with_name(
            metadata.stem.rsplit("_", 1)[0]
            + "_other_"
            + metadata.name.rsplit("_", 1)[1]
        )
    )
    with pytest.raises(SystemExit, match="missing required production AOT keys"):
        _import_aot_objects(source, target)
    assert not target.exists()


def test_verify_requires_production_profiles_in_recorded_inventory(
    pkg_copy, root, tmp_path, monkeypatch
):
    from fmha_sm100.cute.src.common import aot_cache

    path, manifest, directory = _nvfp4_manifest(tmp_path, monkeypatch)
    assert prewarm.check("107a", check_toolchain=False, components=("nvfp4",)) == []
    removed = next(
        name
        for name in manifest["components"]["nvfp4"]["artifacts"]
        if name.startswith("sparse_forward") and not name.endswith(".json")
    )
    # Keep valid matching files on disk; remove only their sealed inventory.
    manifest["components"]["nvfp4"]["artifacts"] = [
        name
        for name in manifest["components"]["nvfp4"]["artifacts"]
        if name not in (removed, removed + ".json")
    ]
    manifest["artifacts"] = [
        entry
        for entry in manifest["artifacts"]
        if entry["name"] not in (removed, removed + ".json")
    ]
    # Retain a fully recorded forward family, but for compact buffers. A
    # directory-only profile check must not borrow the unrecorded right key.
    compact_calls, _ = _captured_production_call(monkeypatch, compact=True)
    compact_key = compact_calls[0][0]
    _aot_export_fixture(directory, monkeypatch, keys=[compact_key])
    compact_base = Path(aot_cache._key_to_path(compact_key))
    for suffix in (".o", ".json"):
        name = compact_base.name + (suffix if suffix == ".json" else "")
        manifest["components"]["nvfp4"]["artifacts"].append(name)
        manifest["artifacts"].append(
            prewarm._artifact("nvfp4", name, compact_base.with_suffix(suffix))
        )
    path.write_text(json.dumps(manifest))
    assert (directory / (removed + ".o")).is_file()
    assert any(
        "inventory omits required production AOT profiles" in problem
        for problem in prewarm.check(
            "107a", check_toolchain=False, components=("nvfp4",)
        )
    )


def test_verify_refuses_changed_pair_dequant_setting(
    pkg_copy, root, tmp_path, monkeypatch
):
    monkeypatch.setenv("MINIMAX_KVFP4_FP8_PAIR_DEQUANT", "1")
    _nvfp4_manifest(tmp_path, monkeypatch)
    assert prewarm.check("107a", check_toolchain=False, components=("nvfp4",)) == []
    monkeypatch.setenv("MINIMAX_KVFP4_FP8_PAIR_DEQUANT", "0")
    problems = prewarm.check("107a", check_toolchain=False, components=("nvfp4",))
    assert any(
        "environment MINIMAX_KVFP4_FP8_PAIR_DEQUANT" in problem for problem in problems
    )
    assert any("production AOT" in problem for problem in problems)


def test_verify_cli_exit_codes(root, tmp_path):
    env = {**os.environ, _cache.ROOT_ENV: str(root)}
    run = [
        sys.executable,
        "-m",
        "fmha_sm100.icp.prewarm",
        "verify",
        "--arch",
        "107a",
        "--no-toolchain",
    ]
    assert (
        subprocess.run(
            [*run, "--policy", "fail"], env=env, cwd=REPO, capture_output=True
        ).returncode
        == 1
    )
    assert (
        subprocess.run(
            [*run, "--policy", "warn"], env=env, cwd=REPO, capture_output=True
        ).returncode
        == 0
    )


def test_decode_aot_lookup_is_keyed_and_can_be_disabled(root, monkeypatch):
    from fmha_sm100.icp.scorer.decode import icp_decode_score as d

    assert d._load_aot_scan(1, 0, 4, True, (10, 7)) is None
    path = d.aot_path(1, 0, 4, True, "107a")
    assert path.parent == _cache.component_dir("decode", "107a")
    assert path.name == "icp_decode_scan_q1_r0_s4_pdl.so"
    assert len(d.AOT_PROFILES) == 8
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    monkeypatch.setenv("ICP_DECODE_AOT", "0")
    assert d._load_aot_scan(1, 0, 4, True, (10, 7)) is None


def test_decode_export_refuses_a_mismatched_dsl_arch(root, monkeypatch):
    from fmha_sm100.icp.scorer.decode import icp_decode_score as d

    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_103a")
    with pytest.raises(RuntimeError, match="CUTE_DSL_ARCH"):
        d.export_aot_scan(1, 0, 4, True, "107a")
