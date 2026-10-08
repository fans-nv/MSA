# Generic runtime and CPU-offload port

The runtime port is staged in `worktrees/vllm-icp-public`, based on public vLLM
`242e4213fc9845ff6fe607af1aee626fd8acc990`. Its reference is the frozen
`worktrees/vllm-msa-upstream` at `53e229bc28df277f0906b94a082293c60a2bc2e7`.
This record covers the generic runtime and related tests only. Model, config,
native writer and MSA changes have separate owners and validation.

The staged scope is 12 production files (+204/-23) and five test files
(+481/-3). Nothing was committed or published by this agent. Exact file hashes
and the local CPU toolchain are in [runtime-source-hashes.json](runtime-source-hashes.json).

## Design and patch boundary

| Change | Purpose and public behavior preserved |
|---|---|
| `KVCacheSpec.uses_raw_page_view`, byte shape, LBHNC and merge guards | Expose the entire compound page to existing allocation, zeroing and copy lifetimes. The model owns the compound spec. Public `has_layer_views`, `uses_slot_mapping`, ordinary per-layer layouts and kernel splitting are retained. No new block manager, index allocation or copy engine. |
| Optional `release_kv_cache` hook at existing layer-cache clearing | Release derived aliases before the backing cache is replaced or freed. Temporary profiling cleanup does not close model-scoped peer resources. |
| Optional model-resource shutdown walker | After graph/device draining, close shared model/draft resources once, before normal layer-cache release and later process-group teardown. Public `GPUWorker.shutdown` already drains the KV connector before runner shutdown; that ordering is unchanged. |
| Default-off backend `cudagraph_decode_phase_only` trait | Restrict only marked FULL descriptors. Actual prefills fall back to PIECEWISE/eager; unknown phase fails closed. Public bounded-varlen, max-query bounds, speculative descriptors and one-token prefill promotion remain intact. |
| Actual-phase agreement in the existing DP metadata collective | Carry one additional CPU integer per rank and preserve the agreed phase in optional `DPSyncState.has_prefill`. No new collective or GPU launch. Both redispatch and sync reuse consult the agreed phase. |
| Separate CPU-offload prerequisite | Serialize accepted queue submissions against the FIFO shutdown sentinel, join the submission thread, synchronize DMA streams, then unregister pinned buffers and release their owners. Failure retries retain unfinished ownership. Public store/event bookkeeping and the disk backend remain unchanged. |

Keep the offload fix separate: its four production files (+78/-19) and
`tests/v1/simple_kv_offload/test_shutdown.py` address a pre-existing lifetime
race. The eight other runtime files (+126/-4) and four test updates form the
generic ICP integration boundary.

The public runner already preserves `has_prefill` and `is_prefilling_np` when
it promotes a prompt tail to a decode-compatible shape. The port passes that
actual phase into graph admission; it does not disable promotion for ordinary
backends. `DefaultModelState.prepare_attn` still forwards the actual CPU flags
to `CommonAttentionMetadata.is_prefilling`.

The DP adaptation is necessary for the generic hook: passing a per-rank phase
through the old reference implementation would allow a decode rank to
redispatch to FULL while a promoted-prefill rank selected PIECEWISE. Tests now
exercise both ranks and reuse of the same collective result. ICP itself still
admits DP1; this generic correctness check does not qualify ICP with DP>1.

[runtime-source-preservation.json](runtime-source-preservation.json) records
14 successful AST/byte comparisons against the public base, including the
allocator, COW copying, view construction, bounded-varlen checks, batch gathering,
prefill promotion, input preparation and ordinary metadata forwarding.

## Validation performed here

All commands use the existing uv-managed `review/v13-formal/.venv/bin/python`,
the new public worktree on `PYTHONPATH`, `VLLM_TARGET_DEVICE=cpu`, empty
`CUDA_VISIBLE_DEVICES`, and caches under this review directory. No dependency
stubs or dependency downloads were used.

| Gate | Result | Evidence |
|---|---|---|
| Focused runtime suite | 212 passed, 10 device-dependent skips, 1 source-version warning; 11.40s; rc0; runtime unchanged during run | [runtime-r1](logs/runtime-r1/runtime/output.log), [JUnit](logs/runtime-r1/runtime/pytest.xml), [exact command](logs/runtime-r1/runtime/command.txt) |
| Shutdown tests after explicit thread non-None assertion for type checking | 6 passed, 216 deselected, 1 source-version warning; 9.15s; rc0 | [shutdown-r2](logs/runtime-shutdown-r2/runtime/output.log) |
| Ruff 0.14.0 check and format; `git diff --check` | All passed | [static checks](logs/runtime-static-final/) |
| Public configured mypy 1.20.2 / Python 3.11 | Candidate and exact public base each report the same 19 diagnostics; zero new or resolved diagnostics | [comparison](runtime-type-comparison.json), [candidate](logs/runtime-types-py311-candidate-r1/output.log), [base](logs/runtime-types-py311-base-r1/output.log) |

The runtime suite includes the public registry, graph manager, profiling,
attention/cache utilities, batch ordering and microbatch-slicing suites, plus
the ported CPU-offload shutdown tests. Added cases cover raw-page shared-storage
COW, incompatible layouts/block splitting/merging, phase-aware backend scoping,
Q1/Q4 promoted-prefill admission with ordinary-backend controls, and DP agreement
reuse with exactly one mocked collective. The ten skips require CUDA, Triton,
real graph pools or DBO device execution; they are not counted as passes.

The full runtime gate preceded one formatting-only test change and the explicit
non-None thread assertion; the affected shutdown subset was rerun afterward.
There were no subsequent production changes. The parent owns the final combined
model/runtime validation after all owners freeze.

For reproducibility:

```bash
bash review/vllm-public-consolidation-20261008/run_runtime_checks.sh NEW_TAG
review/v13-formal/.venv/bin/python \
  review/vllm-public-consolidation-20261008/run_runtime_types.py NEW_TYPE_TAG
```

The type comparison uses an exact `git archive` of base `242e4213` restricted to
`vllm`, `tests`, the type runner and `pyproject.toml`. Diagnostics are compared
by file and message, retaining source line numbers in the JSON. The 19 inherited
diagnostics concern Torch typing, existing capture overrides, runner optionals,
and two existing NumPy annotations. They were not silently repaired in this port.
An earlier type attempt used a missing PATH override and then the old packet's
Python3.10 target; both non-authoritative logs remain recorded. The public hook
explicitly targets Python3.11, which is the comparison above.

## Qualification still required

These CPU results do not prove CUDA graph replay, DMA correctness, collective
resource teardown, native writer numerical parity or performance. Before target
execution, review the complete command/environment plan for this new public
source revision. Correctness gates should cover full-page prefix/COW/offload
round trips, profiling-pool release followed by final allocation, Q1–Q4 decode
replay with actual prefill fallbacks, and normal/error TP2 resource shutdown.
The new public writer/layout also needs its own native build and numerical
oracles; old frozen-source compile evidence does not qualify changed files.

No GPU execution, benchmark, node operation, allocation change, external post,
push or commit was performed in this runtime port.
