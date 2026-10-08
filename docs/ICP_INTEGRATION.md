# Indexer context parallelism in MSA

This candidate adds `fmha_sm100.icp` to `vllm-project/MSA` **dev** at
`ed4e40efcb5895aba1a3554d62cd1dcaf77920cb`. Its input is the frozen ICP source
at `icp-kernels` `4232bce`; `ICP_INTEGRATION_ABI = 1` is the capability gate.
The MSA distribution remains version 0.1.1. Install this candidate artifact
explicitly; a published version with that number alone does not prove support.

## Ownership and stable imports

MSA owns the local index scorer, selection, carrier packing, transport and merge.
The serving engine owns the KV writer, live metadata production, buffer lifetime,
phase selection and graph capture. No KV writer CUDA or standalone `icp_kernels`
package is included here.

| API | Implementation |
| --- | --- |
| `fmha_sm100.icp.scorer.prefill.api` / `.jit` | aliases of shared `fmha_sm100.api` / `.jit`; no second FMHA tree |
| `fmha_sm100.icp.scorer.decode.icp_decode_score` | prepared CuTe score scan; launch ABI 3, score ABI 2 |
| `fmha_sm100.icp.local_indexer` | host planning and bound launch adapters |
| `fmha_sm100.icp.candidates` | local full-row selector |
| `fmha_sm100.icp.CandidateExchange` | D3, K5T and NCCL-plus-K2 route adapter |
| `fmha_sm100.icp.attention.nvfp4_prefill` | lazy facade over shared `fmha_sm100.cute` CSR/attention/combine |

Producer device plans retain ABI 3 and live-plan ABI 1. The serving engine must
check the compiled writer schema and native carrier/scorer ABIs separately from
the Python integration marker. The initial writer producer is not a PDL launch.

D3 retains publish per selector window and a final early merge. K5T retains its
separate selector/exchange launch sequence and tile generations. NCCL retains
packing, collective exchange and K2 merge. The row/tile generation protocols,
launch arguments, padding writes and wait/publish ordering are not unified.
Legacy K5 remains available for compatibility. No tuning or new performance
claim is attached to this ownership move.

## Existing MSA APIs

Dense/full/ordinary score/sparse APIs, split-KV reduction and original 128x4 NVFP4
quantization helpers remain available. The original
`sparse_atten_nvfp4_kv_func(k_scale_128x4=..., v_scale_128x4=...)` retains its
rank-2 scale contract. The added `sparse_atten_nvfp4_kv_cache_func(k_scale=...,
v_scale=...)` accepts the rank-3/4 K-linear/V-token-quad cache views. Both use one
shared kernel with compile-time scale addressing; the ICP facade binds the cache
entry directly. Layout is explicit in the in-memory/AOT key, and the cache-layout
forward has its own AOT family so legacy exports cannot satisfy ICP prewarm.
No scale repacking occurs on either path. Dev's Q8KV4 decode/prefill adapters,
Q8KV4/Q8KV8 indexers, per-head NVFP4 views, ordinary `kv_mode3` dispatch, top-k
and warmup APIs remain available. ICP uses its current full-row selector and
adds `OnlyScoreIcp` to the shared FMHA implementation. Its live metadata and
device-plan arguments are confined to that specialization; ordinary native
entry signatures and quantization stay with dev.

The ICP cache entry explicitly selects dev's strided `vllm` layout. The ordinary
FMHA NVFP4 route continues to require dev's per-head K/V slot layout. These are
different contracts for multiple KV heads and are never silently reinterpreted.

The CuTe implementation is a real qualified subpackage. It no longer inserts
`cute/` into `sys.path` or publishes global `src` / `interface` modules.

Legacy TVM `minfer.ops.*` functions still register on default MSA import. A
serving engine with its own vendored MSA must set `MSA_REGISTER_TVM_FFI=0`
**before the initial canonical import**. It may restore the environment after
import; the canonical package will then leave the vendored global names alone.
`register_tvm_ffi(namespace=...)` allows an explicit separate namespace and
rejects occupied names. Python APIs do not require these globals. With the
opt-out, bare MSA/ICP imports also avoid loading torch/TVM; neither import mode
initializes CUDA or imports the CuTe kernel implementation.

## Source artifacts and CUTLASS

JIT requires native source files, the ABI JSON, and materialized CUTLASS headers
inside the installed package. The source tree's upstream CUTLASS gitlink is
preserved. For a normal checkout, initialize its pinned submodule. For a reviewed
header snapshot, stage the exact selected inputs before building:

```bash
python tools/stage_cutlass.py /path/to/pinned/cutlass \
  --source-id <immutable-commit-or-archive-sha256>
python -m build --sdist --wheel
```

The staging tool copies `include/`, `tools/util/include/`, original license/notice
files and a SHA256 inventory; it refuses to replace a different nonempty tree.
An archive without a root license requires `--license-file` with its existing
license text. The selected dev snapshot contains 908 files from the exact submodule commit
`098de2a652cf8f00fd70b2df54051c7eccbb855a` (CUTLASS 4.8), plus `SOURCE.json`.

Runtime dependencies are declared in `pyproject.toml`; the selected validation
stack uses torch 2.14, CUTLASS DSL 4.8 and Quack 0.6.5. Source/CPU checks do not
qualify other compiler/driver/architecture combinations. The wheel contains no
compiled CUDA artifacts. See `THIRD_PARTY_NOTICES.md` for original rights;
imported sources whose donor default was Proprietary require rights-holder
clearance before public distribution.

## Fresh prewarm and deployment environment

Namespace/source changes invalidate old caches. Build a fresh manifest from the
installed MSA artifact, using the final review commit, fixed cache mount paths,
compiler/Python/dependency versions and target architecture:

```bash
export ICP_CACHE_ROOT=/opt/msa-icp-cache
export TORCH_EXTENSIONS_DIR=/opt/msa-torch-extensions
export ICP_KERNEL_ARCH=107a
export MSA_REGISTER_TVM_FFI=0
unset MINFER_FMHA_CACHE_DIR ICP_KERNEL_CACHE
unset MM_SPARSE_ATTN_AOT_DISABLE
python -m fmha_sm100.icp.prewarm build --arch 107a --commit <review-commit>
python -m fmha_sm100.icp.prewarm verify --arch 107a --policy fail
```

These are deployment instructions, not an authorization to run a benchmark.
The four components are `icp`, `fmha`, `decode` and `nvfp4`; the vLLM writer is
compiled by vLLM. FMHA prewarm covers the ICP scorer variants and existing
plan/top-k modules, not every ordinary MSA attention variant.

The NVFP4 attention/combine AOT objects need one production-form CUDA call, or
`ICP_NVFP4_AOT_IMPORT` pointing to an already generated matching
`sm_<arch>/nvfp4-<digest>/aot/v2/<toolchain>` directory. Both forward and combine
families, each object's source metadata, and the toolchain manifest are
required. The exact admitted profile uses TP2, two local KV heads, BF16 Q and
45056-byte compound pages; prewarm preserves the K/V/scale view offsets and
strides. Verification requires the precise forward and SM107 combine keys and
their filenames in the recorded inventory, including
`MINIMAX_KVFP4_FP8_PAIR_DEQUANT`. A compact-KV export with the same kernel family
does not satisfy this profile. A flat export without architecture/source/toolchain identity is
rejected. Prewarm records the actual native and AOT cache paths and metadata
used by dev's loaders, rather than assuming the old fixed artifact names.

Older private-package recipes forced canonical AOT caching on while setting
`MM_SPARSE_ATTN_AOT_DISABLE=1` for the vendored MSA. That recipe must change:
shared canonical MSA honors its normal AOT environment and strict verification
rejects disabled AOT. Unset that flag (or set it to `0`) for this integration.
Keep `MM_SPARSE_ATTN_AOT_CACHE` unset for the default source/architecture layout.
Verification checks complete inventories, sources, toolchain, effective paths
and artifact hashes; a partial prewarm does not satisfy full serving admission.

CPU regression/import/package checks are separate from pending target-image
native compilation, ordinary MSA attention regression, ICP numerical/distributed
correctness, CUDA graph replay, strict prewarm and performance requalification.
