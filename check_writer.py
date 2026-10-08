"""CPU contracts and real native TU compile, without CUDA execution."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
tree = root / "worktrees/vllm-icp-public"
audit = Path(__file__).parent
tag = sys.argv[1]
mode = sys.argv[2]
proof = audit / "logs" / tag
proof.mkdir(parents=True, exist_ok=False)
python = root / "review/v13-formal/.venv/bin/python"
torch_include = python.parent.parent / "lib/python3.12/site-packages/torch/include"
cuda = Path("/tmp/cu134/nvidia/cu13")
source = "csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu"
paths = [source, "csrc/libtorch_stable/fused_minimax_m3_icp.cuh",
         "csrc/libtorch_stable/ops.h", "csrc/libtorch_stable/torch_bindings.cpp",
         "vllm/_custom_ops.py"]
tests = ["tests/kernels/test_fused_minimax_m3_icp_writer_api.py",
         "tests/kernels/test_minimax_m3_icp_device_plan.py",
         "tests/kernels/test_fused_minimax_m3_icp_writer.py",
         "tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py"]
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before = {p: digest(tree / p) for p in paths + tests}
env_overrides = {
    "PYTHONPATH": str(tree), "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": "",
    "VLLM_TARGET_DEVICE": "cpu", "XDG_CACHE_HOME": str(proof / "cache"),
    "VLLM_CACHE_ROOT": str(proof / "cache/vllm"),
    "VLLM_CONFIG_ROOT": str(proof / "cache/config"),
    "HF_HOME": str(proof / "cache/huggingface"),
    "TORCH_EXTENSIONS_DIR": str(proof / "cache/torch"),
}
commands = {}
if mode == "native":
    common = [str(cuda / "bin/nvcc"), "-std=c++17", "-O3", "-DNDEBUG",
              "--expt-relaxed-constexpr", "-DENABLE_FP8", "-DUSE_CUDA",
              "-DTORCH_TARGET_VERSION=0x020B000000000000ULL", "-Xcompiler=-fPIC",
              "-Xptxas=-v", f"-I{torch_include}", f"-I{cuda}/include",
              f"-I{cuda}/include/cccl",
              f"-I{root}/review/upstream-consolidation-20261007/logs/writer-sm107-cu134-r2/cublas-declarations",
              "-Icsrc", "-Icsrc/libtorch_stable"]
    for arch in ("107a", "107f"):
        commands[arch] = common + ["-gencode", f"arch=compute_{arch},code=sm_{arch}",
                                 "-MD", "-MF", str(proof / f"writer-{arch}.d"),
                                 "-c", source, "-o", str(proof / f"writer-{arch}.o")]
elif mode == "cpu":
    commands["pytest"] = [str(python), "-m", "pytest", "--noconftest", "-p",
                          "no:cacheprovider", "-q", "-rs", *tests,
                          f"--junitxml={proof}/pytest.xml"]
    commands["ruff"] = [str(python.parent / "ruff"), "check", paths[-1], *tests]
    commands["format"] = [str(python.parent / "ruff"), "format", "--check", paths[-1], *tests]
else:
    raise ValueError(mode)
(proof / "commands.json").write_text(json.dumps({"cwd": str(tree), "commands": commands,
    "environment_overrides": env_overrides, "scope": "CPU contracts or release-like per-TU compile. No CUDA runtime, full build, numerical or performance qualification."}, indent=2) + "\n")


def run(item):
    name, command = item
    started = time.monotonic()
    with (proof / f"{name}.log").open("w") as log:
        result = subprocess.run(command, cwd=tree, env=os.environ | env_overrides,
                                stdout=log, stderr=subprocess.STDOUT)
    value = {"exit_code": result.returncode, "seconds": time.monotonic() - started}
    print(name, value, flush=True)
    return name, value


with ThreadPoolExecutor(max_workers=2) as pool:
    results = dict(pool.map(run, commands.items()))
after = {p: digest(tree / p) for p in paths + tests}
objects = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in proof.glob("*.o")}
result = {"base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip(),
          "results": results, "sources_before": before, "sources_after": after,
          "sources_unchanged": before == after, "objects": objects}
(proof / "result.json").write_text(json.dumps(result, indent=2) + "\n")
raise SystemExit(any(v["exit_code"] for v in results.values()) or before != after)
