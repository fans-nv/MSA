"""Reproduce the owned public-port CPU/model type checks in the existing env."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("mode", choices=("model", "types"))
parser.add_argument("--tree", type=Path)
args = parser.parse_args()
audit = Path(__file__).resolve().parent
root = audit.parent.parent
tree = (args.tree or root / "worktrees/vllm-icp-public").resolve()
log = audit / "logs" / args.tag
log.mkdir()
cache = audit / ".cache" / args.tag
files = [
    "vllm/models/minimax_m3/common/compound_page.py",
    "vllm/models/minimax_m3/common/compound_spec.py",
    "vllm/models/minimax_m3/common/indexer_icp.py",
    "vllm/models/minimax_m3/nvidia/indexer_icp.py",
    "vllm/models/minimax_m3/nvidia/model.py",
    "vllm/models/minimax_m3/nvidia/msa_icp.py",
    "vllm/models/minimax_m3/nvidia/msa_icp_main.py",
    "vllm/models/minimax_m3/nvidia/ops/icp_dispatch.py",
    "vllm/models/minimax_m3/nvidia/ops/icp_metadata.py",
    "vllm/models/minimax_m3/nvidia/sparse_attention_icp.py",
]
files += [str(p.relative_to(tree)) for p in sorted(
    (tree / "tests/models/minimax_m3").glob("test_*.py")
)]
files = [f for f in files if (tree / f).is_file()]

def hashes():
    return {f: hashlib.sha256((tree / f).read_bytes()).hexdigest() for f in files}

before = hashes()
venv = root / "review/v13-formal/.venv/bin"
env = os.environ.copy()
overrides = dict(
    PATH=f"{venv}:{env['PATH']}",
    PYTHONPATH=f"{tree}:{root / 'worktrees/msa-dev-public/python'}",
    PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    VLLM_TARGET_DEVICE="cpu", CUDA_VISIBLE_DEVICES="", MSA_REGISTER_TVM_FFI="0",
    ICP_KERNEL_ARCH="107a", MYPY_CACHE_DIR=str(cache / "mypy"),
    XDG_CACHE_HOME=str(cache), VLLM_CACHE_ROOT=str(cache / "vllm"),
    VLLM_CONFIG_ROOT=str(cache / "vllm-config"), HF_HOME=str(cache / "huggingface"),
    TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
    ICP_CACHE_ROOT=str(cache / "icp"),
)
env.update(overrides)
if args.mode == "model":
    command = [str(venv / "python"), "-m", "pytest", "-q", "-rs", "--tb=short",
               "--noconftest", "-p", "no:cacheprovider", "tests/models/minimax_m3",
               f"--junitxml={log / 'pytest.xml'}"]
else:
    command = [str(venv / "python"), "tools/pre_commit/mypy.py", "3.11", *files]
(log / "command.json").write_text(json.dumps(command, indent=2) + "\n")
(log / "environment.json").write_text(json.dumps(overrides, indent=2) + "\n")
(log / "cwd.txt").write_text(str(tree) + "\n")
with (log / "output.log").open("w") as output:
    result = subprocess.run(command, cwd=tree, env=env, stdout=output,
                            stderr=subprocess.STDOUT)
after = hashes()
(log / "result.json").write_text(json.dumps({
    "exit_code": result.returncode, "sources_unchanged": before == after,
    "before": before, "after": after,
}, indent=2) + "\n")
print((log / "output.log").read_text())
raise SystemExit(result.returncode or (0 if before == after else 3))
