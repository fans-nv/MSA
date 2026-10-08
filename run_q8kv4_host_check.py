"""Run the CPU host-extension probe with all writable caches in this packet."""

import json
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

audit = Path(__file__).resolve().parent
root = audit.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tag")
parser.add_argument("--python", type=Path, default=Path(sys.executable))
parser.add_argument("--extra-site-packages", type=Path)
args = parser.parse_args()
tag = args.tag
assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", tag)
logs = audit / "logs" / tag
cache = audit / ".cache" / tag
assert not logs.exists() and not cache.exists(), "Refusing to replace evidence"
logs.mkdir()
cache.mkdir(parents=True)
overrides = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "CUDA_HOME": "/tmp/cu134/nvidia/cu13",
    "CCACHE_DIR": str(cache / "ccache"),
    "CCACHE_TEMPDIR": str(cache / "ccache-tmp"),
    "XDG_CACHE_HOME": str(cache),
    "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
    "ICP_CACHE_ROOT": str(cache / "icp"),
    "VLLM_TARGET_DEVICE": "cpu",
}
env = dict(os.environ, **overrides)
Path(overrides["CCACHE_TEMPDIR"]).mkdir()
records = []
for mode in ("build", "reload"):
    command = [str(args.python), str(audit / "q8kv4_host_extension_check.py"),
               "--source", str(root / "worktrees/msa-dev-public"),
               "--output", str(logs / f"{mode}-result.json")]
    if mode == "build":
        command.append("--build")
    if args.extra_site_packages is not None:
        command.extend(["--extra-site-packages", str(args.extra_site_packages)])
    with (logs / f"{mode}.log").open("w") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=300)
    records.append({"mode": mode, "command": command, "cwd": str(root),
                    "environment_overrides": overrides, "exit_code": result.returncode})
    (logs / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"{mode}: rc={result.returncode}; {logs / (mode + '.log')}", flush=True)
    if result.returncode:
        print((logs / f"{mode}.log").read_text()[-12000:])
        raise SystemExit(result.returncode)
