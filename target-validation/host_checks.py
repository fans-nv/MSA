"""Reproducible stdlib-only guard tests; never run target execution or import CUDA.

Fixtures are temporary tiny Git repositories and inert compiler files. They
exercise refusal/plan logic, not a real build. --actual-defaults additionally
checks the sealed candidate's three read-only plans without changing its pin;
--unsealed-defaults instead verifies that an unpinned driver refuses all modes.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "check_sm107.py"
FORBIDDEN = {"torch", "cutlass", "cuda", "fmha_sm100", "vllm"}


class NoDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise AssertionError(
                f"Device dependency imported in host check: {fullname}"
            )


sys.meta_path.insert(0, NoDeviceImports())
spec = importlib.util.spec_from_file_location("reviewed_target_driver", DRIVER)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def git(tree, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Host test fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-C",
            str(tree),
            *args,
        ],
        text=True,
        stderr=subprocess.STDOUT,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).strip()


class DriverGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vllm-public-driver-host-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(
            vllm=self.root / "vllm",
            msa=self.root / "msa",
            mode="build",
            manifest=self.root / "manifest.json",
            target_spec=self.root / "target.json",
            output=self.root / "evidence",
            build_evidence=None,
        )
        sources = {}
        for name in ("vllm", "msa"):
            tree = getattr(self.args, name)
            tree.mkdir()
            (tree / "tracked.txt").write_text(name + "\n")
            (tree / ".gitignore").write_text("python/fmha_sm100/cutlass/\n")
            git(tree, "init", "--quiet", "--initial-branch=fixture")
            git(tree, "add", ".")
            git(tree, "commit", "--quiet", "-m", "local host fixture")
            sources[name] = {
                "head": git(tree, "rev-parse", "HEAD"),
                "tree": git(tree, "rev-parse", "HEAD^{tree}"),
            }
        self.headers = self.args.msa / "python/fmha_sm100/cutlass"
        self.headers.mkdir(parents=True)
        (self.headers / "header.h").write_text("// inert header\n")
        inventory = {"files": {"header.h": driver.sha(self.headers / "header.h")}}
        (self.headers / "SOURCE.json").write_text(json.dumps(inventory))
        self.manifest = {
            "sources": sources,
            "cutlass_inventory_sha256": driver.sha(self.headers / "SOURCE.json"),
        }
        self.args.manifest.write_text(json.dumps(self.manifest))
        self.pin = patch.object(
            driver, "MANIFEST_SHA256", driver.sha(self.args.manifest)
        )
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def test_valid_source_and_header_inventory(self):
        self.assertEqual(driver.source_checks(self.args), self.manifest)

    def test_unsealed_and_changed_manifest_are_refused(self):
        with patch.object(driver, "MANIFEST_SHA256", None):
            with self.assertRaisesRegex(RuntimeError, "not sealed"):
                driver.source_checks(self.args)
        self.args.manifest.write_text(self.args.manifest.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "manifest changed"):
            driver.source_checks(self.args)

    def test_wrong_source_path_is_refused(self):
        self.args.vllm = self.args.msa
        with self.assertRaisesRegex(RuntimeError, "vllm: wrong HEAD"):
            driver.source_checks(self.args)

    def test_dirty_source_is_refused(self):
        (self.args.vllm / "tracked.txt").write_text("unreviewed change\n")
        with self.assertRaisesRegex(RuntimeError, "source is not clean"):
            driver.source_checks(self.args)

    def test_changed_header_and_extra_header_are_refused(self):
        header = self.headers / "header.h"
        header.write_text("// changed\n")
        with self.assertRaisesRegex(RuntimeError, "CUTLASS bytes changed"):
            driver.source_checks(self.args)
        header.write_text("// inert header\n")
        (self.headers / "extra.h").write_text("// extra\n")
        with self.assertRaisesRegex(RuntimeError, "incomplete or has extra"):
            driver.source_checks(self.args)

    def test_symlink_header_is_refused(self):
        header = self.headers / "header.h"
        target = self.root / "external.h"
        target.write_bytes(header.read_bytes())
        header.unlink()
        header.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "contains a symlink"):
            driver.source_checks(self.args)

    def test_all_dry_modes_preserve_files_and_exclude_timing_suites(self):
        before = {p: driver.sha(p) for p in self.root.rglob("*") if p.is_file()}
        for mode in ("build", "correctness", "distributed"):
            with self.subTest(mode=mode):
                argv = [
                    str(DRIVER),
                    "--mode",
                    mode,
                    "--manifest",
                    str(self.args.manifest),
                    "--vllm",
                    str(self.args.vllm),
                    "--msa",
                    str(self.args.msa),
                    "--output",
                    str(self.args.output),
                ]
                capture = io.StringIO()
                with (
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(capture),
                ):
                    self.assertEqual(driver.main(), 0)
                plan = json.loads(capture.getvalue())
                self.assertTrue(plan["dry_run"])
                self.assertFalse(plan["benchmark"])
                self.assertFalse(self.args.output.exists())
                for command in plan["commands"]:
                    self.assertFalse(
                        any(
                            value.startswith(tuple(driver.SEPARATE_REVIEW_SUITES))
                            for value in command["argv"]
                        )
                    )
        after = {p: driver.sha(p) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_internal_worker_cannot_bypass_explicit_execution_flag(self):
        with patch.object(
            sys, "argv", [str(DRIVER), "--worker-plan", "/never/read.json"]
        ):
            with self.assertRaisesRegex(RuntimeError, "explicit --execute"):
                driver.main()

    def test_unsealed_main_refuses_before_reading_sources_or_probing_target(self):
        with (
            patch.object(driver, "MANIFEST_SHA256", None),
            patch.object(
                driver, "read_command", side_effect=AssertionError("source probe")
            ),
            patch.object(
                driver, "target_checks", side_effect=AssertionError("target probe")
            ),
        ):
            for mode in ("build", "correctness", "distributed"):
                for execute in (False, True):
                    with self.subTest(mode=mode, execute=execute):
                        argv = [str(DRIVER), "--mode", mode]
                        if execute:
                            argv.append("--execute")
                        with patch.object(sys, "argv", argv):
                            with self.assertRaisesRegex(RuntimeError, "not sealed"):
                                driver.main()
            with patch.object(
                sys,
                "argv",
                [str(DRIVER), "--execute", "--worker-plan", "/never/read.json"],
            ):
                with self.assertRaisesRegex(RuntimeError, "not sealed"):
                    driver.main()

    def test_public_correctness_plan_has_writer_and_q8_contracts(self):
        self.args.mode = "correctness"
        plans = driver.commands(self.args, {}, self.args.output)
        writer = next(p for p in plans if p["name"] == "vllm-writer")
        self.assertIn(
            "tests/kernels/test_fused_minimax_m3_icp_writer.py", writer["argv"]
        )
        self.assertNotIn(
            "tests/kernels/test_fused_minimax_m3_nvfp4_writer.py", writer["argv"]
        )
        host = next(p for p in plans if p["name"] == "msa-dev-host-contracts")
        self.assertIn("tests/icp/test_q8kv4_prewarm.py", host["argv"])
        self.assertIn("tests/icp/test_prewarm_cache.py", host["argv"])

    def test_existing_and_source_nested_output_paths_are_refused(self):
        self.args.output.mkdir()
        with self.assertRaisesRegex(RuntimeError, "replace an evidence"):
            driver.execution_paths(self.args, self.args.output)
        for tree in (self.args.vllm, self.args.msa):
            with self.subTest(tree=tree):
                with self.assertRaisesRegex(RuntimeError, "outside candidate trees"):
                    driver.execution_paths(self.args, tree / "new-evidence")
                self.assertFalse((tree / "new-evidence").exists())

    def test_environment_pins_toolkits_and_discards_inherited_overrides(self):
        inherited = {
            "NVCC_PREPEND_FLAGS": "wrong",
            "CUTE_DSL_ARCH": "sm_100a",
            "CUTLASS_DSL_OPT_LEVEL": "wrong",
            "ICP_RUNTIME_JIT": "fail",
            "MINIMAX_KVFP4_FP8_PAIR_DEQUANT": "0",
            "RANK": "7",
            "FMHA_SM100_DECODE_Q8KV4_DISABLE_QMUL4": "1",
            "MSA_Q8KV4_SPLIT_MODE": "legacy",
            "CC": "/wrong/cc",
            "CUDAHOSTCXX": "/wrong/cuda-host",
        }
        with patch.dict(os.environ, inherited):
            env, overrides = driver.environment(
                self.args,
                {"cuda_home": "/pinned/cuda", "cxx": "/pinned/cxx"},
                self.args.output,
            )
        for name in ("CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH"):
            self.assertEqual(env[name], "/pinned/cuda")
        self.assertEqual(env["RANK"], "0")
        for name in inherited.keys() - {"RANK"}:
            self.assertNotIn(name, env)
        self.assertEqual(overrides["ICP_KERNEL_ARCH"], "107a")
        self.assertEqual(overrides["CXX"], "/pinned/cxx")
        self.assertEqual(overrides["CUDACXX"], "/pinned/cuda/bin/nvcc")
        self.assertEqual(overrides["FMHA_SM100_DECODE_Q8KV4_ARCH"], "107a")
        for name in ("CUTLASS_ROOT", "CUTLASS_PATH"):
            self.assertEqual(overrides[name], str(self.headers))

    def test_junit_empty_skipped_and_failed_results_are_refused(self):
        path = self.root / "result.xml"
        for cases in (
            "",
            "<testcase><skipped/></testcase>",
            "<testcase><failure/></testcase>",
            "<testcase><error/></testcase>",
        ):
            with self.subTest(cases=cases):
                path.write_text(f"<testsuite>{cases}</testsuite>")
                with self.assertRaisesRegex(RuntimeError, "Incomplete correctness"):
                    driver.junit_summary(path)
        path.write_text("<testsuite><testcase/></testsuite>")
        self.assertEqual(driver.junit_summary(path)["passed"], 1)

    def build_fixture(self):
        build = self.root / "build"
        build.mkdir()
        tools = self.root / "tools"
        (tools / "bin").mkdir(parents=True)
        paths = {}
        for name in ("nvcc", "ptxas", "cuobjdump", "cmake", "ninja", "cxx"):
            paths[name] = tools / "bin" / name
            paths[name].write_text("inert tool fixture " + name)
        python = self.root / "env/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("inert uv Python fixture")
        (python.parent.parent / "pyvenv.cfg").write_text("uv = host-fixture\n")
        values = {
            "CMAKE_HOME_DIRECTORY": str(self.args.vllm),
            "CMAKE_INSTALL_PREFIX": str(self.args.vllm),
            "VLLM_PYTHON_EXECUTABLE": str(python),
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_GENERATOR": "Ninja",
            "CMAKE_SUPPRESS_REGENERATION": "ON",
            "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
            "CMAKE_BUILD_WITH_INSTALL_RPATH": "ON",
            "FETCHCONTENT_FULLY_DISCONNECTED": "ON",
            "CMAKE_CUDA_COMPILER": str(paths["nvcc"]),
            "CMAKE_MAKE_PROGRAM": str(paths["ninja"]),
            "CMAKE_CXX_COMPILER": str(paths["cxx"]),
            "CMAKE_CUDA_HOST_COMPILER": str(paths["cxx"]),
        }
        source = self.args.vllm / "csrc/libtorch_stable" / driver.WRITER
        source.parent.mkdir(parents=True)
        source.write_text("// host fixture, never compiled\n")
        output = build / "writer.o"
        entry = {
            "file": str(source),
            "directory": str(build),
            "arguments": [
                str(paths["nvcc"]),
                "-ccbin",
                str(paths["cxx"]),
                "--generate-code=arch=compute_107f,code=[sm_107f]",
                "-DTORCH_EXTENSION_NAME=_C_stable_libtorch",
                "-c",
                str(source),
                "-o",
                str(output),
            ],
        }
        specification = {
            "hostname": socket.gethostname(),
            "allocation_id": "host-fixture",
            "image_identity": "host-fixture",
            "python": str(python),
            "cuda_home": str(tools),
            "nvcc_version": "13.4.92",
            "cmake": str(paths["cmake"]),
            "torch_cuda": "13.4",
            "cxx": str(paths["cxx"]),
            "packages": dict.fromkeys(driver.REQUIRED_PACKAGES, "fixture"),
            "gpu_uuids": ["GPU-fixture-0", "GPU-fixture-1"],
            "build_dir": str(build),
            "jobs": 1,
            "timeout_seconds": 5,
            **{name + "_sha256": driver.sha(path) for name, path in paths.items()},
        }
        return build, values, entry, specification, output

    def check_target_fixture(self, build, values, entry, specification, expected):
        (build / "CMakeCache.txt").write_text(
            "\n".join(f"{k}:STRING={v}" for k, v in values.items())
        )
        (build / "compile_commands.json").write_text(json.dumps([entry]))
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(driver, "read_command", return_value="fixture V13.4.92"),
            patch.object(
                driver, "probe_json", side_effect=RuntimeError("HOST_PROBE_BOUNDARY")
            ) as probe,
        ):
            with self.assertRaisesRegex(RuntimeError, expected):
                driver.target_checks(self.args, specification, {})
            self.assertEqual(probe.call_count, int(expected == "HOST_PROBE_BOUNDARY"))

    def test_cuda_host_compiler_must_be_absolute_and_pinned_before_probe(self):
        build, values, entry, specification, _ = self.build_fixture()
        for invalid in ("", "c++", str(self.root / "wrong-cxx")):
            with self.subTest(invalid=invalid):
                values["CMAKE_CUDA_HOST_COMPILER"] = invalid
                self.check_target_fixture(
                    build, values, entry, specification, "Pin CMAKE_CUDA_HOST_COMPILER"
                )

    def test_q8_host_compiler_must_match_native_build_compiler_before_probe(self):
        build, values, entry, specification, _ = self.build_fixture()
        for invalid in ("c++", str(self.root / "wrong-cxx")):
            with self.subTest(invalid=invalid):
                specification["cxx"] = invalid
                self.check_target_fixture(
                    build, values, entry, specification, "Pin the Q8KV4 CXX compiler"
                )

    def test_writer_command_must_explicitly_use_pinned_host_compiler(self):
        build, values, entry, specification, _ = self.build_fixture()
        original = entry["arguments"][:]
        for selector in ([], ["-ccbin", "c++"], ["--compiler-bindir=/wrong/compiler"]):
            with self.subTest(selector=selector):
                entry["arguments"] = original[:1] + selector + original[3:]
                self.check_target_fixture(
                    build,
                    values,
                    entry,
                    specification,
                    "explicitly select the pinned CUDA host",
                )
        for selector in (
            ["-ccbin", values["CMAKE_CXX_COMPILER"]],
            ["--compiler-bindir=" + values["CMAKE_CXX_COMPILER"]],
        ):
            with self.subTest(selector=selector):
                entry["arguments"] = original[:1] + selector + original[3:]
                self.check_target_fixture(
                    build, values, entry, specification, "HOST_PROBE_BOUNDARY"
                )

    def test_stale_object_is_refused_before_target_probe(self):
        build, values, entry, specification, output = self.build_fixture()
        output.write_bytes(b"old object")
        self.check_target_fixture(
            build, values, entry, specification, "Writer object already exists"
        )

    def test_tool_hash_mismatch_is_refused_before_probe(self):
        build, values, entry, specification, _ = self.build_fixture()
        specification["nvcc_sha256"] = "0" * 64
        self.check_target_fixture(
            build, values, entry, specification, "nvcc: tool hash mismatch"
        )

    def test_owned_descendant_is_cleaned_after_launcher_exit(self):
        output = self.root / "process-evidence"
        output.mkdir()
        code = """import os, signal, time
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print('OWNED_CHILD='+str(os.getpid()), flush=True)
    time.sleep(60)
else:
    time.sleep(0.2)
"""
        started = time.monotonic()
        result = driver.run_logged(
            {
                "name": "owned-group",
                "argv": [sys.executable, "-c", code],
                "cwd": str(self.root),
            },
            dict(os.environ),
            output,
            5,
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertLess(time.monotonic() - started, 16)
        child = next(
            line.removeprefix("OWNED_CHILD=")
            for line in (output / "owned-group.log").read_text().splitlines()
            if line.startswith("OWNED_CHILD=")
        )
        stat = Path("/proc") / child / "stat"
        # SIGKILL delivery is asynchronous with reaping the already-exited
        # launcher. Allow the owned child a bounded scheduling interval.
        deadline = time.monotonic() + 2
        while True:
            try:
                alive = stat.read_text().split()[2] != "Z"
            except FileNotFoundError:
                alive = False
            if not alive or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        self.assertFalse(alive)


class RecordedResult(unittest.TextTestResult):
    records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"check": test.id(), "status": "passed"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({"check": test.id(), "status": "failed"})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({"check": test.id(), "status": "error"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    actual_mode = parser.add_mutually_exclusive_group()
    actual_mode.add_argument("--actual-defaults", action="store_true")
    actual_mode.add_argument("--unsealed-defaults", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace host-check evidence")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DriverGuards)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    actual = []
    if (args.actual_defaults or args.unsealed_defaults) and result.wasSuccessful():
        guard = """import importlib.abc, runpy, sys
class NoDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','cutlass','cuda','fmha_sm100','vllm'}:
            raise AssertionError('device import in dry run: '+fullname)
sys.meta_path.insert(0, NoDeviceImports())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        with tempfile.TemporaryDirectory(prefix="vllm-public-actual-dry-") as temporary:
            for mode in ("build", "correctness", "distributed"):
                output = Path(temporary) / mode
                argv = [
                    sys.executable,
                    "-c",
                    guard,
                    str(DRIVER),
                    "--mode",
                    mode,
                    "--output",
                    str(output),
                ]
                run = subprocess.run(
                    argv,
                    text=True,
                    capture_output=True,
                    timeout=60,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                expected_refusal = (
                    run.returncode == 2 and "not sealed" in run.stderr
                    if args.unsealed_defaults
                    else run.returncode == 0
                )
                passed = expected_refusal and not output.exists()
                actual.append(
                    {
                        "mode": mode,
                        "status": "passed" if passed else "failed",
                        "returncode": run.returncode,
                        "stdout": run.stdout,
                        "stderr": run.stderr,
                        "output_created": output.exists(),
                        "expected_unsealed_refusal": args.unsealed_defaults,
                    }
                )
    record = {
        "driver_sha256": driver.sha(DRIVER),
        "host_checks_sha256": driver.sha(__file__),
        "python": sys.executable,
        "python_version": sys.version,
        "device_imports_forbidden": True,
        "gpu_or_build_execution": False,
        "fixture_success": result.wasSuccessful(),
        "fixture_tests_run": result.testsRun,
        "fixture_checks": result.records,
        "actual_candidate_dry_runs": actual,
        "actual_candidate_pin_verified": bool(args.actual_defaults and actual),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    return (
        0
        if result.wasSuccessful() and all(item["status"] == "passed" for item in actual)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
