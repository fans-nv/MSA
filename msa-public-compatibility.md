# MSA dev public compatibility

Commit `0f658079c9ea9ab63c76b975b7ee4dd54a198334` adds public-vLLM compatibility
over dev consolidation `f82fb1759fb0b0f1fd9798c4730484feeb1e3df6`.
The delta is 13 files, +722/-65 lines. Shared scorer and transport kernel bodies
are unchanged by this compatibility delta.

Main attention uses the canonical MSA package. Public per-head K/V slots are
aliased directly, with no forward repack. Both the historical and public
NVFP4 profiles are included in strict prewarm, including actual strides,
calibration flags and architecture-specific combine options. Independent
review and the SM100/103/107 dispatcher-key controls are recorded in
[public-nvfp4-review.md](public-nvfp4-review.md).

Q8KV4 exposes an additive `split_mode` argument. ICP requests stream-k with
one split; ordinary automatic and explicit-split defaults retain their prior
behavior. Native callbacks include the owning Python package and `native_v2`,
so the exact older vendored MSA's unversioned registrations cannot replace
them. Host extensions include source, namespace, Python/Torch/TVM ABI and
resolved compiler/toolkit identity. A shared file lock covers both build and
cached import, avoiding partial-library loads across ranks.

Strict prewarm now covers five components. Q8KV4 inventories the actual host
extension, plan, both GQA16/shift3 forward libraries, their source records,
namespace record and cached QMUL4 verdict. Verification uses the runtime's
CUDA discovery and recipe lookup. A missing verdict or library fails before
serving JIT; compiler version queries do not compile a device probe.

The final broad MSA host gate passes **470 tests**, with 149 skips and 413
deselections, against unchanged committed sources. Focused Q8/prewarm tests
pass 57 cases. The real native host probe additionally builds and imports two
package namespaces, verifies separate pybind plan types and callback symbols,
then reloads them in reverse order with the build entry disabled and
`ICP_RUNTIME_JIT=fail`. Both passes match the final source hashes.

Native evidence is in [q8kv4-cpp-r4](logs/q8kv4-cpp-r4/commands.json),
[build result](logs/q8kv4-cpp-r4/build-result.json) and
[cache-only reload](logs/q8kv4-cpp-r4/reload-result.json). It uses the existing
uv environment `/tmp/laneX-venv` (Torch 2.13.0+cu129), the existing formal
environment's TVM-FFI 0.1.14.post1, CUDA 13.4 headers and workspace-local caches.
Neither environment was modified. CUDA initialization was forbidden, and no
device kernel was compiled or executed by this host probe.

Earlier attempts remain visible: r1 failed on a read-only ccache temp path;
its escalated retry was cancelled before producing execution evidence. r3
used workspace-local caches but exposed missing CUDA C10 headers in the
CPU-only review Torch. r4 resolved that environment limitation with the
existing CUDA-enabled Torch. No production-source change was needed between
these attempts. This native host result does not establish target decode,
graph replay or numerical/performance correctness.
