"""Validate optional-MSA model-test behavior using the unchanged CPU environment."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


audit = Path(__file__).resolve().parent
root = audit.parent.parent
tree = root / "worktrees/vllm-icp-public"
venv = root / "review/v13-formal/.venv/bin"
log = audit / "logs/model-no-companion-r1"
log.mkdir()
cache = audit / ".cache/model-no-companion-r1"
files = sorted((tree / "vllm/models/minimax_m3").rglob("*.py"))
files += sorted((tree / "tests/models/minimax_m3").rglob("*.py"))


def hashes():
    return {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


before = hashes()
env = os.environ.copy()
overrides = {
    "PATH": f"{venv}:{env['PATH']}",
    "PYTHONPATH": str(tree),
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "VLLM_TARGET_DEVICE": "cpu",
    "CUDA_VISIBLE_DEVICES": "", "MSA_REGISTER_TVM_FFI": "0",
    "ICP_KERNEL_ARCH": "107a", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "XDG_CACHE_HOME": str(cache), "VLLM_CACHE_ROOT": str(cache / "vllm"),
    "VLLM_CONFIG_ROOT": str(cache / "config"), "HF_HOME": str(cache / "hf"),
    "TORCH_EXTENSIONS_DIR": str(cache / "extensions"),
    "ICP_CACHE_ROOT": str(cache / "icp"),
}
env.update(overrides)
(log / "environment.json").write_text(json.dumps(overrides, indent=2) + "\n")
(log / "cwd.txt").write_text(str(tree) + "\n")
probe = [str(venv / "python"), "-c", """
import importlib.metadata as m, importlib.util as u, json, sys
assert u.find_spec('fmha_sm100') is None, 'Unexpected companion MSA is installed'
print(json.dumps({'executable': sys.executable, 'sys_path': sys.path,
    'fmha_sm100': None, 'tblib_available': u.find_spec('tblib') is not None,
    'legacy_icp_kernels_available': u.find_spec('icp_kernels') is not None,
    'versions': {p: m.version(p) for p in ('torch', 'pytest', 'transformers', 'apache-tvm-ffi')}}, indent=2))
"""]
probe_result = subprocess.run(probe, cwd=tree, env=env, text=True, capture_output=True)
(log / "probe-command.json").write_text(json.dumps(probe, indent=2) + "\n")
(log / "probe.json").write_text(probe_result.stdout)
(log / "probe-stderr.log").write_text(probe_result.stderr)
probe_result.check_returncode()

# Test normal collection first. The sealed source-only environment may lack
# dependencies required by the repository-wide conftest (record that limit).
collect = [str(venv / "python"), "-m", "pytest", "--collect-only", "-q",
           "-p", "no:cacheprovider", "tests/models/minimax_m3"]
with (log / "normal-collection.log").open("w") as output:
    collected = subprocess.run(collect, cwd=tree, env=env, stdout=output,
                               stderr=subprocess.STDOUT)
(log / "normal-collection-command.json").write_text(json.dumps(collect, indent=2) + "\n")
command = [str(venv / "python"), "-m", "pytest", "-q", "-rs", "--tb=short",
           "--noconftest", "-p", "no:cacheprovider", "tests/models/minimax_m3",
           f"--junitxml={log / 'pytest.xml'}"]
(log / "command.json").write_text(json.dumps(command, indent=2) + "\n")
with (log / "output.log").open("w") as output:
    result = subprocess.run(command, cwd=tree, env=env, stdout=output,
                            stderr=subprocess.STDOUT)
after = hashes()
counts = {}
if (log / "pytest.xml").exists():
    suites = ET.parse(log / "pytest.xml").getroot().findall("testsuite")
    counts = {name: sum(int(s.get(name, "0")) for s in suites)
              for name in ("tests", "failures", "errors", "skipped")}
(log / "result.json").write_text(json.dumps({
    "exit_code": result.returncode, "normal_collection_exit_code": collected.returncode,
    "pytest_counts": counts, "uses_noconftest": True,
    "sources_unchanged": before == after, "before": before, "after": after,
}, indent=2) + "\n")
print((log / "normal-collection.log").read_text())
print((log / "output.log").read_text())
raise SystemExit(result.returncode or (0 if before == after else 3))
