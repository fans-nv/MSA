# Indexer context parallelism in MSA

`fmha_sm100.icp` is the TP2 indexer-context-parallel (ICP) sparse indexer for a serving
engine's compound KV layout: each rank scores its 64-row fragment of every 128-token page,
selects local top-16 candidates and exchanges them with the other rank. The engine owns the
KV writer (which also writes the per-step live metadata and the OnlyScoreIcp device plans),
config admission, graph capture and lifetime.

## Public API

```python
from fmha_sm100.icp import IcpIndexer, IcpStep

indexer = IcpIndexer(group=..., rank=..., topk_buffer=..., num_sparse_layers=...,
                     index_head_dim=..., max_model_len=..., max_num_batched_tokens=...,
                     max_num_seqs=..., cudagraph_capture_sizes=...)  # collective
indexer.allocate(max_decode_query_len=...)     # once, before graph capture
step = indexer.plan(...)                       # per forward, host only
writer(**step.writer_args)                     # the engine's fused KV writer, first sparse layer
indexer.select(step, index_q, index_k, layer=i, scale=s)  # per sparse layer
indexer.close()                                # collective, at shutdown
```

Uniform decode with query length at most 4 scores with the CuTe decode scan; every other
step scores with the OnlyScoreIcp FMHA variant. Pure-decode steps exchange through the fused
symmetric-memory route (D3); steps with prefill rows use pack + `all_to_all_single` + K2.
D3 needs the CUDA symmetric-memory backend: leave `TORCH_SYMMMEM` and `NVSHMEM_MAX_TEAMS`
unset. The device-plan layout the writer fills is ABI 4.

## Prewarm

Build every kernel into the image and verify it in a fresh process; the engine calls
`prewarm.check()` at startup and refuses to serve while anything would compile:

```bash
export ICP_CACHE_ROOT=/opt/msa-icp-cache
python -m fmha_sm100.icp.prewarm build --arch 107a
python -m fmha_sm100.icp.prewarm verify --arch 107a   # prints ICP_PREWARM_VERIFIED
```

`prewarm.warm()` and `prewarm.check()` run the same loads in-process; `check()`
sets `ICP_RUNTIME_JIT=fail` and returns one line per artifact that would compile.
ICP extensions and decode scans are keyed by a source digest under `ICP_CACHE_ROOT`; the
FMHA, Q8KV4 and NVFP4 prefill artifacts use their usual MSA caches.
