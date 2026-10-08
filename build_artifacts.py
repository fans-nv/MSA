"""Build source-only public-review artifacts from clean commits; no GPU work.

Run with review/v13-formal/.venv/bin/python after both candidate branches are
committed. Outputs are append-only evidence: existing output paths are refused.
The installed probe uses an isolated target directory and does not build vLLM.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
MSA = ROOT / "worktrees/msa-dev-public"
VLLM = ROOT / "worktrees/vllm-icp-public"
ARTIFACTS = AUDIT / "artifacts"
LOGS = AUDIT / "logs/artifacts-final"
BUILD = AUDIT / "build-source"
UNPACKED = AUDIT / "build-from-sdist"
INSTALL = AUDIT / "installed-wheel"
MANIFEST = AUDIT / "manifest.json"
CUTLASS_COMMIT = "098de2a652cf8f00fd70b2df54051c7eccbb855a"
MSA_COMPAT_BASE = "f82fb1759fb0b0f1fd9798c4730484feeb1e3df6"
CACHE_COMPONENTS = ("icp", "fmha", "decode", "nvfp4", "q8kv4")
# Include C++ bindings and templates, not only CUDA and Python kernel sources.
SOURCE_SUFFIXES = {
    ".py", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh",
    ".hpp", ".hxx", ".inc", ".inl", ".ipp", ".jinja", ".json",
}
BINARY_SUFFIXES = {".so", ".o", ".a", ".ptx", ".cubin", ".pyc", ".pyo"}
SOURCES = (
    ("msa", MSA, "work/icp-dev-public-compat",
     "ed4e40efcb5895aba1a3554d62cd1dcaf77920cb"),
    ("vllm", VLLM, "work/icp-public-vllm",
     "242e4213fc9845ff6fe607af1aee626fd8acc990"),
)

# Keep this packet independent of audit scripts from earlier review packets.
CORE = (
    "vllm/v1/attention/backend.py",
    "vllm/v1/kv_cache_interface.py",
    "vllm/v1/worker/gpu/attn_utils.py",
    "vllm/v1/worker/gpu/cudagraph_utils.py",
    "vllm/v1/worker/gpu/dp_utils.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu/shutdown.py",
    "vllm/v1/worker/utils.py",
)
OFFLOAD = (
    "vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py",
    "vllm/v1/simple_kv_offload/copy_backend.py",
    "vllm/v1/simple_kv_offload/cuda_mem_ops.py",
    "vllm/v1/simple_kv_offload/worker.py",
)
RUNTIME_TESTS = (
    "tests/v1/cudagraph/test_cudagraph_manager.py",
    "tests/v1/worker/test_attn_utils.py",
    "tests/v1/worker/test_gpu_model_runner_v2_cudagraph_profiling.py",
    "tests/v1/worker/test_gpu_ubatch_slicing.py",
)
WRITER = (
    "csrc/libtorch_stable/fused_minimax_m3_icp.cuh",
    "csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu",
    "csrc/libtorch_stable/ops.h",
    "csrc/libtorch_stable/torch_bindings.cpp",
    "vllm/_custom_ops.py",
    "tests/kernels/test_fused_minimax_m3_icp_writer_api.py",
    "tests/kernels/test_fused_minimax_m3_icp_writer.py",
    "tests/kernels/test_minimax_m3_icp_device_plan.py",
)
MODEL_CI = ".buildkite/test_areas/models_basic.yaml"


def git(tree: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(tree), *args], text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_payload(name: str) -> bool:
    path = Path(name)
    return (path.suffix in SOURCE_SUFFIXES or path.name == "COMMIT"
            or path.name.startswith(("LICENSE", "NOTICE")))


def review_parts(all_paths: set[str]) -> dict[str, set[str]]:
    parts = {
        "01-offload": {*OFFLOAD, "tests/v1/simple_kv_offload/test_shutdown.py"},
        "02-runtime": {*CORE, *RUNTIME_TESTS},
        "03-writer": set(WRITER),
    }
    assigned = set().union(*parts.values())
    assert assigned <= all_paths, f"Missing review paths: {assigned - all_paths}"
    assert len(assigned) == sum(map(len, parts.values())), "Overlapping review parts"
    parts["04-model-integration"] = all_paths - assigned
    assert MODEL_CI in parts["04-model-integration"], "Missing model CI tethering"
    return parts


def read_headers(headers: Path) -> tuple[bytes, dict]:
    inventory_bytes = (headers / "SOURCE.json").read_bytes()
    inventory = json.loads(inventory_bytes)
    assert inventory["schema"] == "fmha_sm100.cutlass-sources.v1"
    assert inventory["source_id"] == "git:" + CUTLASS_COMMIT
    paths = list(headers.rglob("*"))
    assert not [path for path in paths if path.is_symlink()], "CUTLASS symlink"
    materialized = {path.relative_to(headers).as_posix() for path in paths if path.is_file()}
    assert materialized == {*inventory["files"], "SOURCE.json"}, "Unlisted CUTLASS input"
    for relative, expected_hash in inventory["files"].items():
        path = headers / relative
        assert path.resolve().is_relative_to(headers.resolve()), relative
        assert digest(path) == expected_hash, relative
    return inventory_bytes, inventory


def main() -> None:
    if not __debug__:
        raise RuntimeError("Verification requires Python without -O")
    assert not any(os.environ.get(name) for name in
                   ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")), "Git environment override"
    components = {}
    for name, tree, branch, base in SOURCES:
        assert not git(tree, "status", "--porcelain"), f"Uncommitted source: {tree}"
        head = git(tree, "rev-parse", "HEAD")
        assert git(tree, "rev-parse", branch) == head, branch
        assert git(tree, "symbolic-ref", "--short", "HEAD") == branch, branch
        subprocess.run(["git", "-C", str(tree), "merge-base", "--is-ancestor", base, head],
                       check=True)
        components[name] = dict(head=head, tree=git(tree, "rev-parse", head + "^{tree}"),
                                branch=branch, base=base)
    subprocess.run(["git", "-C", str(MSA), "merge-base", "--is-ancestor",
                    MSA_COMPAT_BASE, components["msa"]["head"]], check=True)
    paths = set(git(VLLM, "diff", "--name-only", components["vllm"]["base"],
                    components["vllm"]["head"]).splitlines())
    parts = review_parts(paths)
    for target in (ARTIFACTS, LOGS, BUILD, UNPACKED, INSTALL, MANIFEST):
        assert not target.exists(), f"Refusing to replace evidence: {target}"

    # git archive does not expand the CUTLASS gitlink. Verify its pinned,
    # independently inventoried header payload before materializing a snapshot.
    headers = MSA / "python/fmha_sm100/cutlass"
    inventory_bytes, inventory = read_headers(headers)
    gitlink = git(MSA, "ls-tree", components["msa"]["head"], "python/fmha_sm100/cutlass")
    assert gitlink.split() == ["160000", "commit", CUTLASS_COMMIT,
                               "python/fmha_sm100/cutlass"], gitlink
    ARTIFACTS.mkdir()
    LOGS.mkdir(parents=True)
    BUILD.mkdir()
    commit = components["msa"]["head"]
    archive = subprocess.check_output(["git", "-C", str(MSA), "archive", commit])
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(BUILD, filter="data")
    build_headers = BUILD / "python/fmha_sm100/cutlass"
    build_headers.mkdir(parents=True, exist_ok=True)
    for relative, expected_hash in inventory["files"].items():
        content = (headers / relative).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash, relative
        destination = build_headers / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (build_headers / "SOURCE.json").write_bytes(inventory_bytes)
    (ARTIFACTS / "cutlass-SOURCE.json").write_bytes(inventory_bytes)
    (BUILD / "python/fmha_sm100/icp/COMMIT").write_text(commit + "\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               SOURCE_DATE_EPOCH=git(MSA, "show", "-s", "--format=%ct", commit),
               UV_CACHE_DIR=str(AUDIT / ".cache/artifacts/uv"),
               XDG_CACHE_HOME=str(AUDIT / ".cache/artifacts"))
    env.pop("PYTHONPATH", None)
    commands = []

    def run(name: str, command: list[str], *, cwd: Path = BUILD) -> None:
        with (LOGS / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        commands.append(dict(name=name, command=command, cwd=str(cwd),
                             exit_code=result.returncode,
                             environment={key: env.get(key) for key in (
                                 "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "SOURCE_DATE_EPOCH",
                                 "VLLM_TARGET_DEVICE", "MSA_REGISTER_TVM_FFI",
                                 "ICP_KERNEL_ARCH", "UV_CACHE_DIR", "XDG_CACHE_HOME",
                                 "GIT_INDEX_FILE")}))
        (LOGS / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {LOGS / (name + '.log')}")

    def reconstruct(name: str, tree: Path, base: str, patches: list[Path],
                    expected_tree: str) -> dict:
        # A private index and --cached apply leave the source checkout/index alone.
        env["GIT_INDEX_FILE"] = str(LOGS / f"{name}-index")
        try:
            run(f"{name}-base", ["git", "read-tree", base], cwd=tree)
            for index, patch in enumerate(patches):
                assert patch.stat().st_size, f"Empty review patch: {patch}"
                run(f"{name}-apply-{index + 1}",
                    ["git", "apply", "--cached", str(patch)], cwd=tree)
            actual = subprocess.check_output(["git", "write-tree"], cwd=tree,
                                             env=env, text=True).strip()
            assert actual == expected_tree, f"{name}: reconstructed tree differs"
        finally:
            env.pop("GIT_INDEX_FILE")
        return dict(base=base, reconstructed_tree=actual, expected_tree=expected_tree,
                    patches=[path.name for path in patches])

    run("sdist", [sys.executable, "-c",
                  "from setuptools.build_meta import build_sdist; "
                  f"build_sdist({str(ARTIFACTS)!r})"])
    sdist, = ARTIFACTS.glob("*.tar.gz")
    UNPACKED.mkdir()
    with tarfile.open(sdist) as source:
        source.extractall(UNPACKED, filter="data")
    sdist_root, = UNPACKED.iterdir()
    run("wheel-from-sdist", [sys.executable, "-c",
                            "from setuptools.build_meta import build_wheel; "
                            f"build_wheel({str(ARTIFACTS)!r})"], cwd=sdist_root)
    wheel, = ARTIFACTS.glob("*.whl")

    expected = {}
    for path in (BUILD / "python/fmha_sm100").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and source_payload(path.name):
            expected[path.relative_to(BUILD / "python").as_posix()] = path.read_bytes()
    assert any(name.startswith("fmha_sm100/decode_q8kv4/") and name.endswith(".cpp")
               for name in expected), "Q8KV4 C++ bindings missing from source inventory"
    with zipfile.ZipFile(wheel) as source:
        wheel_files = {name: source.read(name) for name in source.namelist()}
    with tarfile.open(sdist) as source:
        sdist_files = {"/".join(item.name.split("/")[1:]): source.extractfile(item).read()
                       for item in source.getmembers() if item.isfile()}
    for files in (wheel_files, sdist_files):
        assert not [name for name in files if Path(name).suffix in BINARY_SUFFIXES]
    for name, content in expected.items():
        assert wheel_files[name] == content, f"Wheel input differs: {name}"
        assert sdist_files["python/" + name] == content, f"Sdist input differs: {name}"
    wheel_sources = {name for name in wheel_files
                     if name.startswith("fmha_sm100/") and source_payload(name)}
    sdist_sources = {name.removeprefix("python/") for name in sdist_files
                     if name.startswith("python/fmha_sm100/") and source_payload(name)}
    assert wheel_sources == set(expected), "Unexpected or missing wheel source"
    assert sdist_sources == set(expected), "Unexpected or missing sdist source"
    assert not any(name.startswith(("icp_kernels/", "fmha_sm100_icp/"))
                   for name in wheel_files)
    assert not any("fused_minimax_m3_qknorm_rope_kv_insert" in name for name in wheel_files)
    payload_hashes = {name: hashlib.sha256(content).hexdigest()
                      for name, content in sorted(expected.items())}
    (ARTIFACTS / "source-payload-sha256.json").write_text(
        json.dumps(payload_hashes, indent=2) + "\n")
    licenses = sorted(name for name in wheel_files if ".dist-info/licenses/" in name)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "Apache-2.0.txt",
                 "BSD-3-Clause-CUTLASS.txt", "BSD-3-Clause-NVIDIA.txt"):
        assert any(path.endswith("/" + name) for path in licenses), name
    for name in licenses:
        relative = name.split(".dist-info/licenses/", 1)[1]
        content = (BUILD / relative).read_bytes()
        assert wheel_files[name] == content, f"Wheel license differs: {relative}"
        assert sdist_files[relative] == content, f"Sdist license differs: {relative}"

    run("install", ["uv", "pip", "install", "--python", sys.executable,
                    "--no-index", "--no-deps", "--target", str(INSTALL), str(wheel)])
    env.update(PYTHONPATH=f"{INSTALL}:{VLLM}", VLLM_TARGET_DEVICE="cpu",
               MSA_REGISTER_TVM_FFI="0", ICP_KERNEL_ARCH="107a")
    probe_report = LOGS / "installed-probe.json"
    probe = '''
import importlib.metadata, json
from pathlib import Path
import torch
attempts = []
def forbid_cuda(*args, **kwargs):
    attempts.append("cuda_init")
    raise AssertionError("Installed host API initialized CUDA")
torch.cuda._lazy_init = forbid_cuda
from vllm.models.minimax_m3.nvidia.msa_icp import require_msa_icp
package = require_msa_icp()
from fmha_sm100.icp import CandidateExchange, exchange_plan, local_indexer, _cache
assert package.ICP_INTEGRATION_ABI == 1
assert package.PUBLIC_VLLM_ABI == 1
assert _cache.COMPONENTS == COMPONENTS
installed = Path(package.__file__).resolve().parent
assert installed == Path(INSTALL) / "fmha_sm100/icp"
assert (installed / "COMMIT").read_text().strip() == COMMIT
assert CandidateExchange.__module__ == "fmha_sm100.icp.candidate_exchange"
assert exchange_plan.select_token_capacity([4, 8, 16], 5) == 8
cache_files = {name: _cache.file_hashes(name) for name in COMPONENTS}
cache_digests = {name: _cache.component_digest(name) for name in COMPONENTS}
assert all(cache_files.values())
assert any(name.startswith("decode_q8kv4/") and name.endswith(".cpp")
           for name in cache_files["q8kv4"])
saved_package, saved_root = _cache.PACKAGE, _cache.SOURCE_ROOT
try:
    _cache.PACKAGE = Path(SOURCE) / "python/fmha_sm100/icp"
    _cache.SOURCE_ROOT = _cache.PACKAGE.parent
    assert all(cache_files[name] == _cache.file_hashes(name) for name in COMPONENTS)
    assert all(cache_digests[name] == _cache.component_digest(name) for name in COMPONENTS)
finally:
    _cache.PACKAGE, _cache.SOURCE_ROOT = saved_package, saved_root
assert not attempts and not torch.cuda.is_initialized()
result = {"installed_package":str(installed),
 "version":importlib.metadata.version("fmha-sm100"),
 "msa_integration_abi":package.ICP_INTEGRATION_ABI,
 "public_vllm_abi":package.PUBLIC_VLLM_ABI,
 "vllm_msa_admission":"passed", "jit_source_payload_identity":"passed",
 "components":list(COMPONENTS),
 "component_counts":{name:len(files) for name,files in cache_files.items()},
 "component_digests":cache_digests,
 "cuda_initialization_attempts":attempts}
Path(REPORT).write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result, indent=2))
'''
    bindings = (f"INSTALL={str(INSTALL)!r}\nSOURCE={str(BUILD)!r}\nCOMMIT={commit!r}\n"
                f"COMPONENTS={CACHE_COMPONENTS!r}\nREPORT={str(probe_report)!r}\n")
    run("installed-probe", [sys.executable, "-c", bindings + probe], cwd=AUDIT)

    reconstructions = {}
    for name, tree, branch, base in SOURCES:
        head = components[name]["head"]
        assert git(tree, "rev-parse", branch) == head, f"Branch changed: {branch}"
        patch = ARTIFACTS / f"{name}-full.patch"
        patch.write_bytes(subprocess.check_output(
            ["git", "-C", str(tree), "diff", "--binary", base, head]))
        reconstructions[name] = reconstruct(
            name + "-full", tree, base, [patch], components[name]["tree"])
        bundle = ARTIFACTS / f"{name}.bundle"
        run(f"{name}-bundle", ["git", "bundle", "create", str(bundle),
                              branch, f"^{base}"], cwd=tree)
        run(f"{name}-bundle-verify", ["git", "bundle", "verify", str(bundle)], cwd=tree)
        bundle_heads = git(tree, "bundle", "list-heads", str(bundle))
        assert bundle_heads.split() == [head, "refs/heads/" + branch], bundle_heads
    incremental = ARTIFACTS / "msa-public-compat.patch"
    incremental.write_bytes(subprocess.check_output(
        ["git", "-C", str(MSA), "diff", "--binary", MSA_COMPAT_BASE, commit]))
    compat = reconstruct("msa-public-compat", MSA, MSA_COMPAT_BASE,
                         [incremental], components["msa"]["tree"])
    compat["head"] = commit
    slice_patches = []
    for name, part_paths in parts.items():
        patch = ARTIFACTS / f"vllm-{name}.patch"
        patch.write_bytes(subprocess.check_output(
            ["git", "-C", str(VLLM), "diff", "--binary", components["vllm"]["base"],
             components["vllm"]["head"], "--", *sorted(part_paths)]))
        slice_patches.append(patch)
    slices = reconstruct("vllm-review-parts", VLLM, components["vllm"]["base"],
                         slice_patches, components["vllm"]["tree"])
    slices.update(parts={name: sorted(part_paths) for name, part_paths in parts.items()},
                  scope="Exact cumulative source equivalence; intermediate GPU builds not qualified.")
    manifest = {
        "sources": components, "published": False, "gpu_qualified": False,
        "public_distribution_pending_rights_clearance": True,
        "msa_upstream": "https://github.com/vllm-project/MSA/tree/dev",
        "vllm_upstream": "https://github.com/vllm-project/vllm/tree/main",
        "cutlass_commit": CUTLASS_COMMIT,
        "msa_integration_abi": 1, "public_vllm_abi": 1, "device_plan_abi": 3,
        "cache_components": list(CACHE_COMPONENTS),
        "verified_source_payload_files": len(expected),
        "wheel_file_count": len(wheel_files), "sdist_file_count": len(sdist_files),
        "cutlass_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "cutlass_inventory_files": len(inventory["files"]),
        "licenses": licenses,
        "artifacts": {path.name: dict(bytes=path.stat().st_size, sha256=digest(path))
                      for path in sorted(ARTIFACTS.iterdir())},
        "installed_probe": json.loads(probe_report.read_text()),
        "full_patch_reconstructions": reconstructions,
        "msa_public_compatibility": compat,
        "vllm_review_parts": slices,
        "builder_sha256": digest(Path(__file__)),
    }
    for name, tree, branch, _ in SOURCES:
        assert not git(tree, "status", "--porcelain"), f"Build changed source: {tree}"
        assert git(tree, "rev-parse", "HEAD") == components[name]["head"], name
        assert git(tree, "rev-parse", branch) == components[name]["head"], branch
    assert read_headers(headers)[0] == inventory_bytes, "CUTLASS inputs changed during build"
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
