"""Run the public vLLM configured type hook with the existing uv environment."""

import argparse
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("--tree", type=Path)
parser.add_argument("--files", nargs="+")
args = parser.parse_args()
audit = Path(__file__).resolve().parent
root = audit.parent.parent
tree = (args.tree or root / "worktrees/vllm-icp-public").resolve()
log = audit / "logs" / args.tag
log.mkdir()
cache = audit / ".cache" / args.tag
files = [
    file
    for file in (args.files or (audit / "runtime-files.txt").read_text().splitlines())
    if (tree / file).is_file()
]
env = os.environ.copy()
venv = root / "review/v13-formal/.venv/bin"
env.update(
    PATH=f"{venv}:{env['PATH']}",
    PYTHONPATH=f"{tree}:{root / 'worktrees/msa-dev-public/python'}",
    PYTHONDONTWRITEBYTECODE="1",
    VLLM_TARGET_DEVICE="cpu",
    CUDA_VISIBLE_DEVICES="",
    MYPY_CACHE_DIR=str(cache / "mypy"),
    XDG_CACHE_HOME=str(cache),
    VLLM_CACHE_ROOT=str(cache / "vllm"),
    VLLM_CONFIG_ROOT=str(cache / "vllm-config"),
    HF_HOME=str(cache / "huggingface"),
    PRE_COMMIT_HOME=str(cache / "pre-commit"),
    UV_CACHE_DIR=str(cache / "uv"),
)
command = [str(venv / "python"), "tools/pre_commit/mypy.py", "3.11", *files]
(log / "command.txt").write_text(repr(command) + "\n")
(log / "cwd.txt").write_text(str(tree) + "\n")
with (log / "output.log").open("w") as output:
    result = subprocess.run(
        command, cwd=tree, env=env, stdout=output, stderr=subprocess.STDOUT
    )
(log / "exit-code.txt").write_text(str(result.returncode) + "\n")
print((log / "output.log").read_text())
raise SystemExit(result.returncode)
