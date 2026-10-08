"""Bind completed host/build checks to the public port's final source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[1]
TREES = {
    "msa": ROOT / "worktrees/msa-dev-public",
    "vllm": ROOT / "worktrees/vllm-icp-public",
}


def git(tree: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(tree), *args])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msa-tag", required=True)
    parser.add_argument("--integration-tag", required=True)
    parser.add_argument("--q8kv4-build", type=Path, required=True)
    parser.add_argument("--q8kv4-reload", type=Path, required=True)
    args = parser.parse_args()
    output = AUDIT / "final-validation.json"
    assert not output.exists(), "Refusing to replace a sealed receipt"
    manifest_path = AUDIT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, tree in TREES.items():
        assert not git(tree, "status", "--porcelain").strip(), name
        for suffix, key in (("", "head"), ("^{tree}", "tree")):
            assert git(tree, "rev-parse", "HEAD" + suffix).decode().strip() == manifest["sources"][name][key]

    gates = {}
    for gate, tag in (("msa", args.msa_tag), ("integration", args.integration_tag)):
        assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", tag), tag
        directory = AUDIT / "logs" / tag / gate
        assert (directory / "exit-code.txt").read_text().strip() == "0", gate
        assert (directory / "sources-unchanged.txt").read_text().strip() == "1", gate
        for name, tree in TREES.items():
            before = (directory / f"{name}-before-head.txt").read_text().strip()
            head = manifest["sources"][name]["head"]
            patch = directory / f"{name}-before-working.patch"
            assert git(tree, "diff", "--binary", before, head) == patch.read_bytes(), (gate, name)
            inventory = directory / f"{name}-before-working-files.sha256"
            for row in inventory.read_text().splitlines():
                expected, relative = row.split("  ", 1)
                assert digest(tree / relative) == expected, (gate, name, relative)
        report = ET.parse(directory / "pytest.xml").getroot()
        suites = [report] if report.tag == "testsuite" else list(report.iter("testsuite"))
        counts = {key: sum(int(suite.get(key, "0")) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        assert counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0
        counts["passed"] = counts["tests"] - counts["skipped"]
        summaries = [line for line in (directory / "output.log").read_text().splitlines()
                     if re.search(r"\b\d+ passed\b", line)]
        gates[gate] = {
            "path": directory.relative_to(AUDIT).as_posix(),
            "counts": counts, "summary": summaries[-1],
            "final_tree_matches_tested_sources": True,
            "files": {path.name: digest(path) for path in sorted(directory.iterdir()) if path.is_file()},
        }

    format_path = AUDIT / "logs/writer-format21-r1/result.json"
    formatting = json.loads(format_path.read_text())
    native_path = AUDIT / formatting["reused_native_proof"]
    native = json.loads(native_path.read_text())
    assert native["sources_unchanged"]
    assert native["sources_before"] == native["sources_after"]
    assert all(record["exit_code"] == 0 for record in native["results"].values())
    for relative, expected in formatting["final_source_sha256"].items():
        assert digest(TREES["vllm"] / relative) == expected, relative
    for relative, comparison in formatting["comparisons"].items():
        assert comparison["tokens_equal"] and comparison["preprocessor_directives_equal"]
        assert native["sources_after"][relative] == comparison["before_sha256"]
        assert formatting["final_source_sha256"][relative] == comparison["after_sha256"]
    codegen_path = AUDIT / formatting["reused_ordinary_instruction_proof"]
    codegen = json.loads(codegen_path.read_text())
    for arch in ("107a", "107f"):
        result = codegen["results"][arch]
        assert result["common"] == result["identical_instruction_text"] == 76
        assert result["different_symbols"] == []

    host_extensions = {}
    for mode, path in (("build", args.q8kv4_build), ("reversed-cache-load", args.q8kv4_reload)):
        path = path.resolve()
        assert path.is_relative_to(AUDIT)
        record = json.loads(path.read_text())
        assert record["status"] == "passed" and record["mode"] == mode
        assert not record["cuda_initialized"] and not record["device_kernels_built_or_run"]
        for relative, expected in record["source_hashes"].items():
            assert digest(TREES["msa"] / relative) == expected, relative
        host_extensions[mode] = {"path": path.relative_to(AUDIT).as_posix(), "sha256": digest(path)}

    receipts = [
        "logs/vllm-commit-r2.json", "logs/model-no-companion-r1/result.json",
        "writer-source-parity.json", "model-parity.json",
        "runtime-source-preservation.json", "runtime-type-comparison.json",
        "model-mypy-comparison-r3.json", "config-type-comparison.json",
        formatting["reused_binding_proof"], formatting["reused_boxing_proof"],
    ]
    data = {
        "manifest_sha256": digest(manifest_path), "sources": manifest["sources"],
        "host_gates": gates,
        "writer_native_compile": {
            "path": native_path.relative_to(AUDIT).as_posix(), "sha256": digest(native_path),
            "architectures": sorted(native["results"]),
            "format_only_bridge": {"path": format_path.relative_to(AUDIT).as_posix(), "sha256": digest(format_path)},
            "ordinary_instruction_comparison": {"path": codegen_path.relative_to(AUDIT).as_posix(), "sha256": digest(codegen_path)},
        },
        "q8kv4_host_extension": host_extensions,
        "supporting_receipts": {relative: digest(AUDIT / relative) for relative in receipts},
        "gpu_execution": False, "performance_qualified": False, "published": False,
        "limitations": [
            "Local vLLM host suites disable repository conftest and third-party plugin autoload.",
            "CUDA/device-dependent skips are not passes.",
            "Native compilation and ordinary instruction equality are not numerical or serving measurements.",
            "Public writer-to-reader, graph, distributed, model-evaluation and performance qualification remain pending.",
        ],
    }
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"host_gates": {name: value["summary"] for name, value in gates.items()},
                      "source_bindings": "passed", "gpu_execution": False}, indent=2))


if __name__ == "__main__":
    main()
