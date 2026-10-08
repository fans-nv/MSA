# Independent runtime review

Reviewed the staged public-vLLM runtime/offload delta against `242e4213fc9845ff6fe607af1aee626fd8acc990`. No actionable issue remains in this bounded review. This was source and existing-test inspection; no GPU, graph replay, DMA, distributed workload, or benchmark was run.

## Actual phase and graph selection

- `vllm/v1/worker/gpu/model_runner.py:1296` computes actual request prefill state before shape promotion. Lines 1314–1339 preserve `is_prefilling_np` and `has_prefill` when a one-token prompt extension, including one padded with three placeholder drafts, is eligible for a Q1/Q4 decode-shaped graph.
- `model_runner.py:1749` passes actual phase through `dispatch_cg_and_sync_dp`; `vllm/v1/worker/gpu/cudagraph_utils.py:135` admits a decode-phase-only descriptor only with explicit `has_prefill=False`. Missing phase fails closed at line 540, and line 566 implements the final replay check used by `model_runner.py:2007`.
- The capability defaults to false for ordinary builders. It changes only FULL descriptors, leaving ordinary promotion and PIECEWISE execution intact. `tests/v1/cudagraph/test_cudagraph_manager.py:340` covers Q1/Q4 and bounded-varlen selection; line 372 calls actual `gather_batch_req_state` for promoted prompt tails with restricted and ordinary managers.
- `vllm/v1/worker/gpu/model_states/default.py:229` still forwards the actual CPU `is_prefilling_np` unchanged to common attention metadata. The ICP builder reads it at `vllm/models/minimax_m3/nvidia/indexer_icp.py:2197`, so shape promotion cannot choose D3 for a prompt-containing invocation. Main-attention decode-shaped rows may still use Q8KV4; index scoring/transport follows actual phase.

## DP agreement and reuse

`vllm/v1/worker/gpu/dp_utils.py` carries phase in a seventh row of the existing CPU coordination tensor. Any rank's prefill produces an agreed true value; unknown remains unknown unless another rank reports true. Both final redispatch paths consume that agreement, and reused `DPSyncState` supplies the same phase without a second collective. Eager reuse stays eager. The public speculator's `dataclasses.replace` also preserves the added field. Ordinary draft graph managers do not opt into the ICP restriction. The control at `tests/v1/cudagraph/test_cudagraph_manager.py:411` covers either prefill rank, both local ranks, no-prefill, and same-batch reuse, asserting one collective. The ICP model itself still admits DP1 only.

## Raw cache and resource lifetime

- `vllm/v1/kv_cache_interface.py:314` obtains raw pages directly from `page_size_bytes`, refuses kernel-block splitting, and requires LBHNC at line 341. Public slot-mapping/layer-view behavior remains separate. The model-owned compound spec supplies its own merge contract, while ordinary `FullAttentionSpec.merge` refuses accidental loss of raw-page state.
- Worker shutdown first drains the KV connector (`vllm/v1/worker/gpu_worker.py:1562`). The CPU connector's DMA backend closes submission under a lock, appends a FIFO sentinel, joins the owner thread, and synchronizes its streams before unregistration (`vllm/v1/simple_kv_offload/copy_backend.py:97`). `_flush_and_sync_all` only waits on events; it does not submit after backend shutdown. Manually registered owners are dropped only after successful unregistration, supporting retry after failure (`vllm/v1/simple_kv_offload/worker.py:386`). Disk behavior is unchanged.
- `model_runner.py:2355` synchronizes, removes graphs and builders, then calls deduplicated final model hooks before dropping layer cache aliases. Temporary profiling release calls only `release_kv_cache`. Existing CPU lifecycle tests cover this ordering and shared roots.
- A separate model-owned lifetime issue found during review was fixed: the new main-decode schedule pool no longer uses a strong global cache. It is a weak registry with explicit final layer ownership release. `test_main_decode_plan_lifetime_follows_live_models_not_temporary_kv` proves temporary KV release retains schedules, closing one model preserves another's shared owner, and the last close releases schedules even if model objects survive the runner. The revised model suite passes 196 CPU tests; repository mypy reports only the same 26 inherited ordinary-model diagnostics and no errors in the nine new model modules or thirteen model test files.

Target GPU qualification remains necessary for stream/PDL ordering, CUDA graph replay, actual registered-memory DMA, cross-rank operation, and numerical/performance equivalence.
