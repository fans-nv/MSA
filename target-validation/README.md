# Public vLLM / MSA SM107 correctness preparation

`check_sm107.py` defaults to a host-only dry run after the final manifest is sealed: it verifies the frozen review
manifest, both clean Git revisions and the exact staged CUTLASS inventory, then
prints commands. It uses only the standard library; it does not import torch,
initialize CUDA, build, install, reserve resources or write evidence directories.
The manifest is pinned by SHA256 inside the driver. A changed candidate requires
reviewing and repinning this packet.

The source pin is sealed to manifest SHA256
`0c2a96870fed4b22fe550fc4402b5cd9cd2867922b20f6fd5c4f1387d484d1b1`,
binding vLLM `516bd9a9` and MSA dev `0f658079`. The final host receipt verifies
19 fixture guards and all three actual candidate dry plans. This preparation
does not authorize target execution or establish GPU correctness.

The new defaults are `worktrees/vllm-icp-public` (public base
`242e4213fc9845ff6fe607af1aee626fd8acc990`) and `worktrees/msa-dev-public`
(additive public compatibility over the previously sealed MSA dev candidate
`f82fb1759fb0b0f1fd9798c4730484feeb1e3df6`). Final candidate heads come from the
new manifest. Fixture checks use a temporary manifest only in their own test
process; they cannot repin or bypass the release driver.

Inspect all three plans with the existing uv environment:

```bash
review/v13-formal/.venv/bin/python review/vllm-public-consolidation-20261008/target-validation/check_sm107.py --mode build
review/v13-formal/.venv/bin/python review/vllm-public-consolidation-20261008/target-validation/check_sm107.py --mode correctness
review/v13-formal/.venv/bin/python review/vllm-public-consolidation-20261008/target-validation/check_sm107.py --mode distributed
```

No target execution has been authorized or performed by preparing this packet.
Resolve the node, allocation/resource ID, container image, GPU UUIDs, toolchain,
timeouts and build parallelism before reviewing an execution command. Run inside
the already authorized target environment; this driver has no SSH, scheduler,
container-management, download or dependency-install steps.

## Target prerequisites

Use clean target copies at the exact vLLM/MSA heads and Git trees in the new
review manifest, plus its materialized CUTLASS inventory. The MSA ancestry
remains dev `ed4e40efcb5895aba1a3554d62cd1dcaf77920cb`. Override `--vllm` and `--msa` for their
target paths. The initial editable vLLM installation, uv environment, test
dependencies and local CMake dependencies must already exist.

Prepare a **fresh build directory** following the candidate's
[`docs/contributing/incremental_build.md`](../../../worktrees/vllm-icp-public/docs/contributing/incremental_build.md).
The driver deliberately does **not** configure CMake, generate presets, fetch
dependencies or refresh source checkouts. The reviewed `release` preset must
point `CMAKE_CUDA_COMPILER`, `VLLM_PYTHON_EXECUTABLE` and `CMAKE_INSTALL_PREFIX`
at the selected toolkit, uv Python and vLLM source directory. Configure with
`TORCH_CUDA_ARCH_LIST=10.7`, Release/Ninja, and these standard CMake cache values:

```text
CMAKE_EXPORT_COMPILE_COMMANDS=ON
CMAKE_BUILD_WITH_INSTALL_RPATH=ON
CMAKE_SUPPRESS_REGENERATION=ON
FETCHCONTENT_FULLY_DISCONNECTED=ON
CMAKE_CUDA_COMPILER_LAUNCHER=
CMAKE_CXX_COMPILER_LAUNCHER=
```

Set `CMAKE_CUDA_HOST_COMPILER` and target-spec `cxx` to the same absolute
compiler executable as `CMAKE_CXX_COMPILER`. `cxx` also pins the Q8KV4 host
extension through `CXX`. The writer's resolved `-ccbin`/`--compiler-bindir` must
name that executable; relying on NVCC's PATH default is refused. Its bytes are
pinned by `cxx_sha256`.

All dependencies must be materialized before configuring with disconnected
FetchContent. Source changes require an explicit reviewed reconfiguration;
automatic regeneration is disabled in this packet. No configure command is run
by the driver. Compiler launchers are disabled to establish fresh compilation
instead of a compiler-cache hit. The writer object and native build library
must not already exist; the driver never deletes existing build outputs. Leave
the library output at the build-directory default.
`CMAKE_BUILD_WITH_INSTALL_RPATH=ON` keeps build/install bytes identical instead
of rewriting the RPATH at installation. See `CMakeLists.txt` and `cmake/utils.cmake` for the actual
`_C_stable_libtorch` target and install component. `torch.ops._C` is the operator
namespace, **not** the CMake target to rebuild.

Supply a JSON target specification. Every placeholder below must be resolved;
`jobs` and `timeout_seconds` must become positive integers. Package versions
are exact installed distribution versions, including any local/nightly suffix.
Use the same specification for the build and both correctness modes.

```json
{
  "hostname": "<exact socket.gethostname() on target>",
  "allocation_id": "<reviewed allocation or externally managed resource ID>",
  "image_identity": "<reviewed immutable image digest>",
  "python": "/absolute/uv-environment/bin/python",
  "cuda_home": "/absolute/cuda-toolkit",
  "nvcc_version": "<exact V-version reported by nvcc and ptxas>",
  "nvcc_sha256": "<sha256 of cuda_home/bin/nvcc>",
  "ptxas_sha256": "<sha256 of cuda_home/bin/ptxas>",
  "cuobjdump_sha256": "<sha256 of cuda_home/bin/cuobjdump>",
  "cmake": "/absolute/path/to/cmake",
  "cmake_sha256": "<sha256>",
  "ninja_sha256": "<sha256 of CMAKE_MAKE_PROGRAM>",
  "cxx": "/absolute/path/to/the-reviewed-CXX-compiler",
  "cxx_sha256": "<sha256 of CMAKE_CXX_COMPILER>",
  "torch_cuda": "<exact torch.version.cuda>",
  "packages": {
    "torch": "<exact version>",
    "nvidia-cutlass-dsl": "<exact version>",
    "quack-kernels": "<exact version>",
    "apache-tvm-ffi": "<exact version>",
    "pytest": "<exact version>"
  },
  "gpu_uuids": ["GPU-<first UUID>", "GPU-<second UUID>"],
  "build_dir": "/absolute/preconfigured-vllm-build",
  "jobs": "<reviewed compile parallelism>",
  "timeout_seconds": "<reviewed maximum seconds per command>"
}
```

Direct SM107 compilation needs CUDA 13.4 or newer. The driver assumes **no**
deployed toolkit version: it verifies the exact reviewed version and tool
hashes from this specification. A CUDA 13.4 candidate check does not qualify
the historical CUDA 13.5 production image. Image identity and non-Slurm
resource identity are operator-supplied provenance; hostname, GPU UUIDs,
capabilities, package versions and compiler hashes are verified. If
`SLURM_JOB_ID` exists, it must match `allocation_id`.

The child environment pins `CUDA_HOME`, `CUDA_TOOLKIT_PATH` and `CUDA_PATH` to
the reviewed toolkit. Inherited `NVCC_*`, `CUTE_DSL_*` and `CUTLASS_DSL_*`
overrides and MSA instrumentation/tuning flags, including Q8KV4 overrides, are
cleared; the plan records removed keys. `CUDACXX` selects the pinned NVCC,
`CXX` selects the reviewed host compiler, `CUTLASS_ROOT`/`CUTLASS_PATH` select
the inventoried headers, and Q8KV4 architecture is explicitly `107a`. Inherited
`CC`/`CUDAHOSTCXX` overrides are removed. The ordinary MSA CUDA recipes retain
their existing compiler rules; inspect their emitted commands in the target
evidence as part of toolchain qualification. The Python probe records DSL variables established by the
reviewed environment's startup hooks. Single-process rank state is explicit;
torchrun replaces it for its two workers.

## Execution modes for subsequent review

Use a fresh output directory outside both candidate trees. First inspect the
resolved plan **without** `--execute`. After the target and exact commands have
been reviewed, append `--execute` to the applicable command below. Replace
`TARGET_PYTHON`, source paths and other uppercase placeholders with real values.

```bash
TARGET_PYTHON check_sm107.py --mode build --vllm VLLM_SOURCE --msa MSA_SOURCE --target-spec TARGET_SPEC.json --output BUILD_EVIDENCE
TARGET_PYTHON check_sm107.py --mode correctness --vllm VLLM_SOURCE --msa MSA_SOURCE --target-spec TARGET_SPEC.json --build-evidence BUILD_EVIDENCE/result.json --output CORRECTNESS_EVIDENCE
TARGET_PYTHON check_sm107.py --mode distributed --vllm VLLM_SOURCE --msa MSA_SOURCE --target-spec TARGET_SPEC.json --build-evidence BUILD_EVIDENCE/result.json --output DISTRIBUTED_EVIDENCE
```

The **build** mode compiles only `_C_stable_libtorch` and installs that component
into the existing editable vLLM tree. It replaces its native shared library and
modifies build outputs; tracked sources must remain clean. Preflight checks the
actual writer compile command contains native `compute_107f`/`sm_107f`, as
selected by this candidate's CMake rules. A verbose compiler invocation, a
newly produced writer object, and its SM107 ELF image from `cuobjdump
--list-elf` are required before installation. Object hashes and the fresh
linked library hash are retained; installation must yield identical library
bytes. A Ninja no-op or an old installed library cannot pass this build gate.
MSA's separate JIT target remains
`ICP_KERNEL_ARCH=107a`. The driver does not change either rule.

The **correctness** mode runs the existing public norm/RoPE/cache tests, the
new `test_fused_minimax_m3_icp_writer.py` per-head compound-byte, fragment,
index-Q rounding and native argument tests, plus native device-plan ABI3 tests
(including zero-token metadata publication),
the shared legacy/cache scale-layout reference comparisons, and selected local
selector/D3 tests including graph replay. It verifies the loaded writer's path,
hash and real operator schema against the recorded build, plus the imported
MSA source path. The build must use this exact driver revision and target
specification. Cold MSA JIT builds
are allowed inside the fresh evidence cache. These are pytest correctness
checks; no timing benchmark is included.

For the public migration, this mode also runs CPU ordinary-API/import/cache
contracts and the new Q8KV4 split-default, callback namespace, host-extension
identity/lock, runtime-JIT refusal and exact prewarm-artifact checks. These are
`tests/icp/test_q8kv4_prewarm.py` and `tests/icp/test_prewarm_cache.py`; their
mocked native leaves do not qualify actual main-attention execution. It also
runs two untimed generic top-k oracles: block-table gathering after
selection and per-token valid-prefix selection. These supplement the shared
legacy/cache NVFP4 numerical tests. They do not execute dev's ordinary Q8KV4
attention matrix.

The **distributed** mode runs the selected carrier, K5T and D3 transport
comparisons under local `torch.distributed.run`, exactly two ranks with the
same complete UUID list. It requires two SM107 GPUs with bidirectional peer
access. Each rank writes its own JUnit report. It does not launch a server or
create a scheduler allocation.

Every selected test must execute and pass: skips, empty JUnit reports, errors
and timeouts fail the gate. Logs retain argv/cwd, resolved environment, source
manifest, target specification, driver hash, tool hashes, actual GPU identity,
native binary hash, exit codes and JUnit counts/hashes. Existing output
directories are refused. On timeout or interruption only the driver's own
subprocess group is terminated, with bounded escalation for lingering child
processes even if their launcher exited; no allocation or unrelated process is stopped, and evidence is
retained. There is no automatic cleanup.

The selected tests exercise the public writer and canonical MSA readers
**separately**. They do not feed the new public per-head writer's outputs into
canonical Q8KV4 decode or NVFP4 prefill, so writer-to-reader correctness remains
an explicit missing integration gate. Prepared D3/selector graph tests cover
those components only: there is no full MiniMax model graph/replay, promoted
prefill/mixed-step, speculative verification, CPU offload or end-to-end model
accuracy qualification here. The full scorer/transport matrix and paired
performance comparison remain pending. The selected public writer tests do not
measure its kernel-launch count; that performance-sensitive contract needs its
own target evidence. A passing result
remains `gpu_qualified: false` for the release as a whole. Benchmarks require a
separate fully resolved and explicitly approved launch plan.

## Reproducible host guard checks

Run with the existing uv environment and a fresh result filename:

```bash
PYTHONDONTWRITEBYTECODE=1 review/v13-formal/.venv/bin/python review/vllm-public-consolidation-20261008/target-validation/host_checks.py --actual-defaults --output review/vllm-public-consolidation-20261008/target-validation/host-checks-new.json
```

The script uses only the standard library and blocks imports of torch, CUDA,
CUTLASS, MSA, and vLLM. Temporary tiny Git repositories test the exact manifest,
source cleanliness, source-path identity and CUTLASS inventory guards. Inert
compiler files test tool hashes, explicit host-compiler selection and stale
object refusal before any Python or device probe. Other checks cover all
three dry plans, output placement, environment cleanup, incomplete JUnit
refusal, and cleanup of a subprocess group created solely by the test. The
last check takes approximately ten seconds and stops only its own child.
The script also checks that unsealed `--execute` invocations stop at the pin
guard. It never reaches target execution, builds native code, initializes
CUDA, manages nodes or modifies candidate sources.

The command above also checks all three real candidate plans with device
imports forbidden and verifies that no evidence/output directory is created.
`host-checks-final-r1.json` records the successful fixture checks and actual
candidate dry runs, plus exact driver/test hashes. The earlier unsealed
refusal receipt remains separate; it does not qualify a release source pin.
`port-origin.json` records the frozen dev driver/harness origin. The old
evidence packet is unchanged.

## Separately reviewed ordinary-dev qualification

`tests/q8kv4/conftest.py` and `tests/q8kv4_prefill/conftest.py` compile their
GPU variants up front. Their `run_timed` helper synchronizes and invokes the
operation twice: a first call that may include cold JIT and a second measured
call, with a 30-second deadlock threshold. Because these suites deliberately
record runtime measurements, they are **not** included in this driver's
automatic correctness commands.

The following is a preparation template, not an approved launch:

```text
<TARGET_PYTHON> -m pytest -q -rs -p no:cacheprovider -m 'not full'
  tests/q8kv4 tests/q8kv4_prefill
  --junitxml=<FRESH_ORDINARY_DEV_EVIDENCE>/pytest.xml
```

Before running it, resolve and review the exact command and collected case
list, allocation and node, image digest, final vLLM/MSA commits, selected GPU
UUID, complete environment, output directory, timeout, and child cleanup.
There is no server and no model traffic: server topology/restart, request
rate, concurrency, ISL/OSL, and prompt/request counts are not applicable to
these kernel tests. The helper protocol is one initial invocation plus one
timed invocation per helper call; individual tests can invoke the helper more
than once. Record those case/helper counts and the independent suite repetition
count explicitly in the reviewed plan. Do not describe a suite repetition as
the helper's second invocation. No allocation cleanup is authorized by this
template.

The ordinary-dev coverage should retain both block-scale shifts (0 and 3),
nonuniform K/V global scales, per-head slot layout and parent-page gaps,
reordered pages and invalid tails, fallback/forced dispatch, mixed
decode/prefill batches, repeated graph updates, and reusable plans. This is
in addition to the admitted ICP writer-to-reader, prewarm, numerical,
distributed, model-accuracy and paired-performance qualification. None of
those pending checks is established by host guard tests or compile-only
writer evidence.
