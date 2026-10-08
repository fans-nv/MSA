"""Prepare or run bounded SM107 build/correctness checks; dry run is default."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
WORKSPACE = AUDIT.parents[1]
MANIFEST_SHA256: str | None = (
    "0c2a96870fed4b22fe550fc4402b5cd9cd2867922b20f6fd5c4f1387d484d1b1"
)
WRITER = "fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu"
NATIVE_MODULE = "_C_stable_libtorch"
REQUIRED_PACKAGES = {
    "torch",
    "nvidia-cutlass-dsl",
    "quack-kernels",
    "apache-tvm-ffi",
    "pytest",
}
WRITER_TESTS = [
    "tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py",
    "tests/kernels/test_fused_minimax_m3_icp_writer.py",
    "tests/kernels/test_minimax_m3_icp_device_plan.py",
    "tests/kernels/test_fused_minimax_m3_icp_writer_api.py",
]
DEV_HOST_TESTS = [
    "tests/regression/test_icp_consolidation.py",
    "tests/regression/test_icp_dev_compatibility.py",
    "tests/jit_cache/test_cute_aot_cache.py",
    "tests/icp/test_q8kv4_prewarm.py",
    "tests/icp/test_prewarm_cache.py",
]
DEV_TOPK_TESTS = [
    "tests/smoke/test_sparse_topk_forced.py::test_block_table_gather_after_sort",
    "tests/smoke/test_sparse_topk_large_boundary_bin.py::test_per_token_valid_prefixes_select_exact_topk",
]
# These existing suites synchronize and time repeated GPU runs. They are
# intentionally excluded from this correctness-only command plan.
SEPARATE_REVIEW_SUITES = ["tests/q8kv4", "tests/q8kv4_prefill"]
DISTRIBUTED_TESTS = [
    "tests/icp/test_exchange.py::test_the_head_directed_transport_matches_the_all_gather_reference",
    "tests/icp/test_tiled_exchange.py::test_real_window_world_any_matches_k2",
    "tests/icp/test_fused_exchange.py::test_real_window_matches_nccl_route",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_command(argv, *, cwd=None, env=None):
    # Read-only preflight commands must not refresh a Git index or write pyc.
    environment = dict(os.environ if env is None else env)
    environment.update(GIT_OPTIONAL_LOCKS="0", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.check_output(
        [str(value) for value in argv],
        cwd=cwd,
        env=environment,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=60,
    ).strip()


def git(tree, *arguments):
    return read_command(["git", "-C", tree, *arguments])


def probe_json(argv, env):
    transcript = read_command(argv, env=env)
    records = [
        line.removeprefix("TARGET_CHECK_JSON=")
        for line in transcript.splitlines()
        if line.startswith("TARGET_CHECK_JSON=")
    ]
    require(len(records) == 1, "Target probe did not emit exactly one result")
    result = json.loads(records[0])
    result["argv"] = [str(value) for value in argv]
    result["transcript"] = transcript
    return result


def source_checks(args):
    require(
        MANIFEST_SHA256 is not None,
        "Public review manifest is not sealed; execution and dry plans are refused until reviewed repinning.",
    )
    require(
        sha(args.manifest) == MANIFEST_SHA256,
        "Review manifest changed; review and repin the driver.",
    )
    manifest = json.loads(args.manifest.read_text())
    for name in ("vllm", "msa"):
        tree = getattr(args, name)
        expected = manifest["sources"][name]
        require(
            git(tree, "rev-parse", "HEAD") == expected["head"], f"{name}: wrong HEAD"
        )
        require(
            git(tree, "rev-parse", "HEAD^{tree}") == expected["tree"],
            f"{name}: wrong tree",
        )
        require(not git(tree, "status", "--porcelain"), f"{name}: source is not clean")
    headers = args.msa / "python/fmha_sm100/cutlass"
    inventory = headers / "SOURCE.json"
    require(
        sha(inventory) == manifest["cutlass_inventory_sha256"],
        "Wrong staged CUTLASS inventory",
    )
    entries = json.loads(inventory.read_text())["files"]
    require(not headers.is_symlink(), "CUTLASS root must be materialized")
    paths = list(headers.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "CUTLASS contains a symlink")
    actual = {path.relative_to(headers).as_posix() for path in paths if path.is_file()}
    require(
        actual == set(entries) | {"SOURCE.json"},
        "CUTLASS inventory is incomplete or has extra files",
    )
    require(
        all(sha(headers / name) == value for name, value in entries.items()),
        "CUTLASS bytes changed",
    )
    return manifest


def environment(args, spec, output):
    env = dict(os.environ)
    # Discard inherited tuning, old cache roots, test selectors and rank state.
    prefixes = (
        "ICP_",
        "MSA_ICP_",
        "MSA_Q8KV4_",
        "FMHA_SM100_DECODE_Q8KV4_",
        "MINIMAX_",
        "MINFER_FMHA_",
        "MM_SPARSE_ATTN_",
        "VLLM_",
        "CUTE_DSL_",
        "CUTLASS_DSL_",
        "NVCC_",
        "TORCHELASTIC_",
    )
    for key in list(env):
        if key.startswith(prefixes) or key in {
            "PYTEST_ADDOPTS",
            "PYTEST_PLUGINS",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "VLLM_USE_PRECOMPILED",
            "TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES",
            "GPU_TRACE",
            "SM_TIMING",
            "FMHA_GMEM_CHECK",
            "CC",
            "CUDAHOSTCXX",
        }:
            env.pop(key)
    python = spec.get("python", "<UV_PYTHON>")
    python_bin = str(Path(python).parent) if "python" in spec else "<UV_ENV_BIN>"
    cuda = spec.get("cuda_home", "<CUDA_HOME>")
    overrides = {
        "PATH": f"{python_bin}:{cuda}/bin:{env.get('PATH', '')}",
        "PYTHONPATH": f"{args.vllm}:{args.msa / 'python'}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "VLLM_TARGET_DEVICE": "cuda",
        "CUDA_HOME": cuda,
        "CUDA_TOOLKIT_PATH": cuda,
        "CUDA_PATH": cuda,
        "CUDACXX": f"{cuda}/bin/nvcc",
        "CXX": spec.get("cxx", "<CXX>"),
        "CUTLASS_ROOT": str(args.msa / "python/fmha_sm100/cutlass"),
        "CUTLASS_PATH": str(args.msa / "python/fmha_sm100/cutlass"),
        "FMHA_SM100_DECODE_Q8KV4_ARCH": "107a",
        "CUDA_VISIBLE_DEVICES": ",".join(spec.get("gpu_uuids", ["<GPU_UUIDS>"])),
        "TORCH_CUDA_ARCH_LIST": "10.7",
        "ICP_KERNEL_ARCH": "107a",
        "MSA_REGISTER_TVM_FFI": "0",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(output / "cache"),
        "UV_CACHE_DIR": str(output / "cache/uv"),
        "TORCH_EXTENSIONS_DIR": str(output / "cache/torch-extensions"),
        "ICP_CACHE_ROOT": str(output / "cache/msa-icp"),
        "VLLM_CACHE_ROOT": str(output / "cache/vllm"),
        "VLLM_CONFIG_ROOT": str(output / "cache/vllm-config"),
        "HF_HOME": str(output / "cache/huggingface"),
        "TRITON_CACHE_DIR": str(output / "cache/triton"),
        "CUDA_CACHE_PATH": str(output / "cache/cuda"),
        "TMPDIR": str(output / "tmp"),
    }
    env.update(overrides)
    return env, overrides


def commands(args, spec, output):
    python = spec.get("python", "<UV_PYTHON>")
    build = spec.get("build_dir", "<PRECONFIGURED_BUILD_DIR>")
    cmake = spec.get("cmake", "<CMAKE>")
    result = []

    def add(name, argv, cwd, reports=()):
        result.append(
            dict(
                name=name,
                argv=[str(arg) for arg in argv],
                cwd=str(cwd),
                reports=list(reports),
            )
        )

    def pytest(name, targets, cwd, *, vllm=False):
        report = str(output / f"{name}.xml")
        argv = [python, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider"]
        if vllm:
            argv.append("--noconftest")
        add(name, [*argv, *targets, f"--junitxml={report}"], cwd, [report])

    if args.mode == "build":
        add(
            "build-native-writer",
            [
                cmake,
                "--build",
                build,
                "--target",
                NATIVE_MODULE,
                "--parallel",
                spec.get("jobs", "<JOBS>"),
                "--verbose",
            ],
            args.vllm,
        )
        add(
            "install-native-writer",
            [
                cmake,
                "--install",
                build,
                "--component",
                NATIVE_MODULE,
                "--prefix",
                args.vllm,
            ],
            args.vllm,
        )
    elif args.mode == "correctness":
        pytest("vllm-writer", WRITER_TESTS, args.vllm, vllm=True)
        pytest("msa-dev-host-contracts", DEV_HOST_TESTS, args.msa)
        pytest("msa-dev-topk", DEV_TOPK_TESTS, args.msa)
        pytest(
            "msa-shared-layouts",
            ["tests/regression/test_nvfp4_scale_compatibility.py"],
            args.msa,
        )
        pytest(
            "msa-local-selection",
            [
                "tests/icp/test_candidates.py::test_the_forced_column_is_excluded_on_every_rank",
                "tests/icp/test_candidates.py::test_the_global_id_is_the_scan_origin_plus_the_column",
                "tests/icp/test_candidates.py::test_the_decode_entry_point_is_capturable",
                "tests/icp/test_fused_exchange.py::test_d3_is_bit_identical_to_selector_plus_k2",
                "tests/icp/test_fused_exchange.py::test_multi_chunk_token_offsets_cover_the_extent",
                "tests/icp/test_fused_exchange.py::test_cuda_graph_replay_advances_generations_on_device",
            ],
            args.msa,
        )
    else:
        add(
            "distributed-transport",
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=2",
                HERE / "check_sm107.py",
                "--execute",
                "--worker-plan",
                output / "worker-plan.json",
            ],
            args.msa,
            [str(output / f"distributed-rank-{rank}.xml") for rank in range(2)],
        )
    return result


def cache_values(path):
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            name, value = line.split("=", 1)
            values[name.split(":", 1)[0]] = value
    return values


def target_checks(args, spec, env):
    required = {
        "hostname",
        "allocation_id",
        "image_identity",
        "python",
        "cuda_home",
        "nvcc_version",
        "nvcc_sha256",
        "ptxas_sha256",
        "cuobjdump_sha256",
        "cmake",
        "cmake_sha256",
        "ninja_sha256",
        "cxx",
        "cxx_sha256",
        "torch_cuda",
        "packages",
        "gpu_uuids",
        "build_dir",
        "jobs",
        "timeout_seconds",
    }
    require(
        required <= spec.keys(),
        f"Missing target inputs: {sorted(required - spec.keys())}",
    )
    require(socket.gethostname() == spec["hostname"], "Wrong target hostname")
    require(
        spec["allocation_id"] and spec["image_identity"],
        "Allocation and image identities are required",
    )
    if os.environ.get("SLURM_JOB_ID"):
        require(
            os.environ["SLURM_JOB_ID"] == spec["allocation_id"],
            "Slurm allocation ID mismatch",
        )
    require(
        type(spec["jobs"]) is int and spec["jobs"] > 0,
        "jobs must be a positive integer",
    )
    require(
        type(spec["timeout_seconds"]) is int and spec["timeout_seconds"] > 0,
        "timeout_seconds must be positive",
    )
    release = tuple(int(part) for part in spec["nvcc_version"].split(".")[:2])
    require(release >= (13, 4), "Direct SM107 targeting requires CUDA 13.4 or newer")
    require(spec["torch_cuda"], "Pin the reviewed PyTorch CUDA build explicitly")
    uuids = spec["gpu_uuids"]
    require(uuids and len(set(uuids)) == len(uuids), "List distinct full GPU UUIDs")
    require(
        all(value.startswith("GPU-") for value in uuids),
        "GPU UUIDs, not mutable ordinals, are required",
    )
    if args.mode == "distributed":
        require(
            len(uuids) == 2,
            "Distributed mode requires exactly two visible peer-connected GPUs",
        )
    python = Path(spec["python"])
    require(
        python.is_absolute() and python.is_file(),
        "Provide an existing uv-managed Python executable",
    )
    config = python.parent.parent / "pyvenv.cfg"
    require(
        config.is_file() and "uv =" in config.read_text(),
        "Python must belong to a uv-managed environment",
    )
    require(
        REQUIRED_PACKAGES <= spec["packages"].keys(),
        "Pin every required package in target-spec.packages",
    )
    build = Path(spec["build_dir"])
    require(build.is_absolute(), "build_dir must be absolute")
    cache = cache_values(build / "CMakeCache.txt")
    checks = {
        "CMAKE_HOME_DIRECTORY": str(args.vllm),
        "CMAKE_INSTALL_PREFIX": str(args.vllm),
        "VLLM_PYTHON_EXECUTABLE": str(python),
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_GENERATOR": "Ninja",
        "CMAKE_SUPPRESS_REGENERATION": "ON",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_BUILD_WITH_INSTALL_RPATH": "ON",
        "FETCHCONTENT_FULLY_DISCONNECTED": "ON",
    }
    require(
        all(cache.get(key) == value for key, value in checks.items()),
        f"Preconfigured CMake inputs differ: {[(key, cache.get(key), value) for key, value in checks.items() if cache.get(key) != value]}",
    )
    for language in ("CUDA", "CXX"):
        require(
            not cache.get(f"CMAKE_{language}_COMPILER_LAUNCHER"),
            "Configure without compiler launchers for fresh native-build evidence",
        )
    require(
        not cache.get("CMAKE_LIBRARY_OUTPUT_DIRECTORY")
        or Path(cache["CMAKE_LIBRARY_OUTPUT_DIRECTORY"]).resolve() == build.resolve(),
        "Native library output must be in the build directory",
    )
    cuda = Path(spec["cuda_home"])
    require(
        Path(cache["CMAKE_CUDA_COMPILER"]).resolve() == (cuda / "bin/nvcc").resolve(),
        "Wrong CMake nvcc",
    )
    tools = {
        "nvcc": cuda / "bin/nvcc",
        "ptxas": cuda / "bin/ptxas",
        "cuobjdump": cuda / "bin/cuobjdump",
        "cmake": Path(spec["cmake"]),
        "ninja": Path(cache["CMAKE_MAKE_PROGRAM"]),
        "cxx": Path(cache["CMAKE_CXX_COMPILER"]),
    }
    require(
        Path(spec["cxx"]).is_absolute()
        and Path(spec["cxx"]).resolve() == tools["cxx"].resolve(),
        "Pin the Q8KV4 CXX compiler to the reviewed CMake CXX executable",
    )
    require(
        cache.get("CMAKE_CUDA_HOST_COMPILER")
        and Path(cache["CMAKE_CUDA_HOST_COMPILER"]).is_absolute()
        and Path(cache["CMAKE_CUDA_HOST_COMPILER"]).resolve() == tools["cxx"].resolve(),
        "Pin CMAKE_CUDA_HOST_COMPILER to the reviewed CXX compiler executable",
    )
    for name, path in tools.items():
        require(
            path.is_absolute() and path.is_file(), f"Missing absolute tool path: {name}"
        )
        require(sha(path) == spec[f"{name}_sha256"], f"{name}: tool hash mismatch")
    versions = {name: read_command([path, "--version"]) for name, path in tools.items()}
    require(f"V{spec['nvcc_version']}" in versions["nvcc"], "nvcc version mismatch")
    require(f"V{spec['nvcc_version']}" in versions["ptxas"], "ptxas version mismatch")
    database = json.loads((build / "compile_commands.json").read_text())
    writer_commands = [
        entry for entry in database if Path(entry["file"]).name == WRITER
    ]
    require(writer_commands, "Writer missing from compile_commands.json")
    writer_objects = []
    for entry in writer_commands:
        source = Path(entry["directory"]) / entry["file"]
        require(
            source.resolve() == args.vllm / "csrc/libtorch_stable" / WRITER,
            "Writer compiled from another source tree",
        )
        command = entry.get("command", shlex.join(entry.get("arguments", [])))
        argv = entry.get("arguments") or shlex.split(command)
        require(
            Path(argv[0]).resolve() == tools["nvcc"].resolve(),
            "Writer compiler differs from the pinned nvcc",
        )
        host_compilers = []
        for index, argument in enumerate(argv):
            if argument in ("-ccbin", "--compiler-bindir"):
                require(index + 1 < len(argv), "Missing CUDA host compiler argument")
                host_compilers.append(argv[index + 1])
            elif argument.startswith(("-ccbin=", "--compiler-bindir=")):
                host_compilers.append(argument.split("=", 1)[1])
        require(
            len(host_compilers) == 1
            and Path(host_compilers[0]).is_absolute()
            and Path(host_compilers[0]).resolve() == tools["cxx"].resolve(),
            "Writer must explicitly select the pinned CUDA host compiler executable",
        )
        require(
            "compute_107f" in command and "sm_107f" in command,
            "Writer lacks resolved SM107f native FP4 target",
        )
        require(NATIVE_MODULE in command, "Writer belongs to an unexpected target")
        require(argv.count("-o") == 1, "Writer needs one explicit object output")
        object_path = (Path(entry["directory"]) / argv[argv.index("-o") + 1]).resolve()
        require(
            object_path.is_relative_to(build.resolve()),
            "Writer output is outside the build tree",
        )
        if args.mode == "build":
            require(
                not object_path.exists(),
                "Writer object already exists: configure a fresh build directory",
            )
        writer_objects.append(str(object_path))
    if args.mode == "build":
        require(
            not list(build.glob(f"{NATIVE_MODULE}*.so")),
            "Native build library already exists: configure a fresh build directory",
        )
    # The first Python probe is metadata-only. It cannot initialize CUDA.
    code = """
import importlib.metadata as m, json, os, sys
print('TARGET_CHECK_JSON='+json.dumps({'python':sys.version,
 'packages':{n:m.version(n) for n in json.loads(sys.argv[1])},
 'dsl_environment':{k:v for k,v in os.environ.items() if k.startswith(('CUTE_DSL_', 'CUTLASS_DSL_'))}}))
"""
    versions_python = probe_json(
        [python, "-c", code, json.dumps(spec["packages"])], env
    )
    require(
        versions_python["packages"] == spec["packages"],
        "Target Python dependency versions differ",
    )
    # Only --execute reaches hardware discovery; no kernel, build or output writes yet.
    code = """
import json, torch
p=[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
print('TARGET_CHECK_JSON='+json.dumps({'torch_cuda':torch.version.cuda,'gpus':[{'uuid':str(x.uuid),'name':x.name,'cc':[x.major,x.minor]} for x in p],
 'peer_access':all(torch.cuda.can_device_access_peer(i,j) for i in range(len(p)) for j in range(len(p)) if i!=j)}))
"""
    hardware = probe_json([python, "-c", code], env)
    require(hardware["torch_cuda"] == spec["torch_cuda"], "torch CUDA version mismatch")
    require(
        [gpu["uuid"] for gpu in hardware["gpus"]] == uuids,
        "Visible GPU UUID order mismatch",
    )
    require(
        all(gpu["cc"] == [10, 7] for gpu in hardware["gpus"]),
        "Visible devices are not SM107",
    )
    if args.mode == "distributed":
        require(hardware["peer_access"], "The requested GPU pair lacks peer access")
    return dict(
        tools={
            name: dict(path=str(path), sha256=sha(path), version=versions[name])
            for name, path in tools.items()
        },
        python=versions_python,
        hardware=hardware,
        writer_commands=writer_commands,
        writer_objects=writer_objects,
        cmake_cache_sha256=sha(build / "CMakeCache.txt"),
        compile_commands_sha256=sha(build / "compile_commands.json"),
    )


def native_file(vllm):
    files = list((vllm / "vllm").glob(f"{NATIVE_MODULE}*.so"))
    require(
        len(files) == 1,
        f"Expected exactly one installed {NATIVE_MODULE} shared library",
    )
    return files[0]


def check_build_evidence(args, spec):
    require(
        args.build_evidence is not None,
        "Correctness modes require --build-evidence from this driver",
    )
    record = json.loads(args.build_evidence.read_text())
    require(
        record["status"] == "passed" and record["mode"] == "build",
        "Build evidence did not pass",
    )
    require(
        record["manifest_sha256"] == MANIFEST_SHA256,
        "Build used another source manifest",
    )
    require(
        record["target_spec_sha256"] == sha(args.target_spec),
        "Build used another target specification",
    )
    require(
        record["driver_sha256"] == sha(__file__), "Build used another driver revision"
    )
    require(record.get("fresh_native_build"), "Build lacks fresh-compilation evidence")
    binary = native_file(args.vllm)
    require(
        sha(binary) == record["native_binary"]["sha256"],
        "Installed writer differs from built binary",
    )
    return record


def check_fresh_native_build(spec, verified, output):
    """Establish writer compilation and a native SM107 image before installing."""
    transcript = (output / "build-native-writer.log").read_text()
    objects = []
    for entry, filename in zip(verified["writer_commands"], verified["writer_objects"]):
        path = Path(filename)
        require(path.is_file(), "The fresh writer object was not produced")
        argv = entry.get("arguments") or shlex.split(entry["command"])
        output_arg = argv[argv.index("-o") + 1]
        require(
            any(
                WRITER in line
                and output_arg in line
                and str(Path(spec["cuda_home"]) / "bin/nvcc") in line
                for line in transcript.splitlines()
            ),
            "Verbose build log does not show the writer compile invocation",
        )
        command = [
            str(Path(spec["cuda_home"]) / "bin/cuobjdump"),
            "--list-elf",
            str(path),
        ]
        listing = read_command(command)
        require("sm_107" in listing, "Writer object has no native SM107 ELF image")
        objects.append(
            dict(
                path=str(path),
                sha256=sha(path),
                bytes=path.stat().st_size,
                image_argv=command,
                image_listing=listing,
            )
        )
    libraries = list(Path(spec["build_dir"]).glob(f"{NATIVE_MODULE}*.so"))
    require(len(libraries) == 1, "Expected one freshly linked native build library")
    library = libraries[0]
    return dict(
        objects=objects,
        library=dict(
            path=str(library), sha256=sha(library), bytes=library.stat().st_size
        ),
    )


def junit_summary(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    counts = {"tests": len(cases), "passed": 0, "skipped": 0, "failed": 0}
    for case in cases:
        if case.find("skipped") is not None:
            counts["skipped"] += 1
        elif case.find("failure") is not None or case.find("error") is not None:
            counts["failed"] += 1
        else:
            counts["passed"] += 1
    require(
        cases and not counts["skipped"] and not counts["failed"],
        f"Incomplete correctness gate: {path}: {counts}",
    )
    return counts


def run_logged(command, env, output, timeout):
    start = time.time()
    log = output / f"{command['name']}.log"
    timed_out = False
    with log.open("w") as stream:
        process = subprocess.Popen(
            command["argv"],
            cwd=command["cwd"],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # An exited leader can still have descendants. Always clean only
            # our new session, including when the driver receives Ctrl-C.
            terminate_owned_group(process)
        code = process.returncode
    return dict(
        **command,
        exit_code=code,
        timed_out=timed_out,
        elapsed_seconds=time.time() - start,
        log_sha256=sha(log),
    )


def terminate_owned_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def worker(path):
    require(
        os.environ.get("WORLD_SIZE") == "2" and os.environ.get("TORCHELASTIC_RUN_ID"),
        "Worker requires this driver's torchrun launch",
    )
    rank = int(os.environ["RANK"])
    require(
        rank in (0, 1) and int(os.environ["LOCAL_RANK"]) == rank,
        "Unexpected worker mapping",
    )
    plan = json.loads(path.read_text())
    os.chdir(plan["cwd"])
    import pytest

    return pytest.main(
        [
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            *plan["tests"],
            f"--junitxml={plan['output']}/distributed-rank-{rank}.xml",
        ]
    )


def execution_paths(args, output):
    """Reject unsafe output placement before target probes or directory creation."""
    require(
        args.target_spec is not None and args.output is not None,
        "--execute requires --target-spec and --output",
    )
    require(not output.exists(), "Refusing to replace an evidence directory")
    require(
        not any(output.is_relative_to(tree) for tree in (args.vllm, args.msa)),
        "Evidence/cache directory must be outside candidate trees",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("build", "correctness", "distributed"), default="correctness"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run only after the resolved target plan is reviewed",
    )
    parser.add_argument("--manifest", type=Path, default=AUDIT / "manifest.json")
    parser.add_argument(
        "--vllm", type=Path, default=WORKSPACE / "worktrees/vllm-icp-public"
    )
    parser.add_argument(
        "--msa", type=Path, default=WORKSPACE / "worktrees/msa-dev-public"
    )
    parser.add_argument("--target-spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-evidence", type=Path)
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_plan is not None:
        require(args.execute, "Internal distributed worker requires explicit --execute")
        require(
            MANIFEST_SHA256 is not None,
            "Public review manifest is not sealed; distributed workers are refused.",
        )
        return worker(args.worker_plan)
    args.vllm, args.msa = args.vllm.resolve(), args.msa.resolve()
    manifest = source_checks(args)
    spec = json.loads(args.target_spec.read_text()) if args.target_spec else {}
    output = args.output.resolve() if args.output else Path("<FRESH_OUTPUT_DIR>")
    env, overrides = environment(args, spec, output)
    planned = commands(args, spec, output)
    plan = dict(
        mode=args.mode,
        manifest_sha256=MANIFEST_SHA256,
        sources=manifest["sources"],
        commands=planned,
        environment=overrides,
        removed_environment=sorted(set(os.environ) - set(env)),
        target_spec=spec,
        benchmark=False,
        separately_reviewed_suites=SEPARATE_REVIEW_SUITES,
        dry_run=not args.execute,
        unresolved="Supply reviewed target-spec, fresh output and build-evidence for correctness modes"
        if not spec
        else None,
    )
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    execution_paths(args, output)
    # All source/toolchain/hardware checks complete before mkdir/build/install.
    verified = target_checks(args, spec, env)
    if args.mode != "build":
        check_build_evidence(args, spec)
    source_checks(args)
    output.mkdir(parents=True)
    (output / "tmp").mkdir()
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "manifest.json").write_bytes(args.manifest.read_bytes())
    (output / "target-spec.json").write_bytes(args.target_spec.read_bytes())
    (output / "driver.py").write_bytes(Path(__file__).read_bytes())
    if args.mode == "distributed":
        (output / "worker-plan.json").write_text(
            json.dumps(
                dict(cwd=str(args.msa), output=str(output), tests=DISTRIBUTED_TESTS)
            )
        )
    record = dict(
        status="running",
        mode=args.mode,
        manifest_sha256=MANIFEST_SHA256,
        target_spec_sha256=sha(args.target_spec),
        driver_sha256=sha(__file__),
        preflight=verified,
        commands=[],
        benchmark=False,
        gpu_qualified=False,
    )

    def interrupted(signum, frame):
        raise RuntimeError(f"Driver received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode != "build":
            code = """
import hashlib,json,pathlib,sys
from vllm.models.minimax_m3.nvidia.msa_icp import require_icp_writer, require_msa_icp
require_icp_writer(); require_msa_icp()
import vllm._C_stable_libtorch as native
import fmha_sm100.icp as msa_icp
p=pathlib.Path(native.__file__).resolve()
assert p.parent == pathlib.Path(sys.argv[1])/'vllm', p
m=pathlib.Path(msa_icp.__file__).resolve()
assert m.parent == pathlib.Path(sys.argv[2])/'python/fmha_sm100/icp', m
print('TARGET_CHECK_JSON='+json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'writer_abi':3,'public_vllm_abi':msa_icp.PUBLIC_VLLM_ABI,'msa_icp_path':str(m)}))
"""
            loaded = probe_json([spec["python"], "-c", code, args.vllm, args.msa], env)
            require(
                loaded["sha256"] == sha(native_file(args.vllm)),
                "Loaded another native writer",
            )
            record["loaded_writer"] = loaded
        for command in planned:
            result = run_logged(command, env, output, spec["timeout_seconds"])
            record["commands"].append(result)
            require(
                result["exit_code"] == 0 and not result["timed_out"],
                f"{command['name']} failed; inspect its log",
            )
            if command["name"] == "build-native-writer":
                record["fresh_native_build"] = check_fresh_native_build(
                    spec, verified, output
                )
            result["junit"] = {
                path: dict(**junit_summary(path), sha256=sha(path))
                for path in command["reports"]
            }
            source_checks(args)
        binary = native_file(args.vllm)
        if args.mode == "build":
            require(
                sha(binary) == record["fresh_native_build"]["library"]["sha256"],
                "Installed library differs from the freshly linked build library",
            )
        record["native_binary"] = dict(
            path=str(binary), sha256=sha(binary), bytes=binary.stat().st_size
        )
        record["status"] = "passed"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = repr(error)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"Refused/failed: {error}", file=sys.stderr)
        sys.exit(2)
