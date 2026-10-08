# Public MiniMax M3 ICP port

The model port is based on public vLLM `242e4213fc9845ff6fe607af1aee626fd8acc990`. The qualified-path reference is `53e229bc28df277f0906b94a082293c60a2bc2e7`; shared indexing/transport remains in the dev-based MSA candidate, with additive public compatibility in `worktrees/msa-dev-public`. No old ordinary MSA backend, private Q8KV4 source tree, or 94-file prerequisite is copied.

## Cache and writer contract

The raw TP2 compound page remains 45,056 bytes: 36,864 bytes of main KV and 8,192 bytes of rank-local R64 index keys. Compound ABI 2 explicitly changes main KV to public vLLM's alternating per-head K/V slots. The device-plan ABI remains 3 and index fragment geometry is unchanged.

| View | Base byte offset | Shape per page | Byte strides, including page |
| --- | ---: | --- | --- |
| K data | 0 | `(2,128,64)` | `(45056,18432,64,1)` |
| K scales | 8192 | `(2,128,8)` | `(45056,18432,8,1)` |
| V data | 9216 | `(2,128,64)` | `(45056,18432,64,1)` |
| V scales | 17408 | `(2,128,8)` | `(45056,18432,8,1)` |
| Index keys | 36864 | `(64,128)` | `(45056,128,1)` |

MSA's existing `nvfp4_head_slot_views` produces these aliases once at cache binding. Forward performs no cache repack. The writer uses the public `kv_k_scale`/`kv_v_scale` prefix and the unchanged optional ICP tail; first sparse layer ownership still fuses live metadata and the device plan into the same non-PDL producer launch. The ICP specialization emits BF16 main Q and unscaled FP8 main Q together, and preserves the qualified index-Q conversion. Ordinary public writer arithmetic and query behavior stay separate and unchanged. Calibrated device K/V globals are consumed by both main-attention routes. The writer rejects nonunit FP8-Q scale.

## Phase and API ownership

| Phase | Index scoring/selection/transport | Main attention |
| --- | --- | --- |
| Pure decode, uniform Q1–Q4 | Existing CuTe H4 scorer; D3 fused selection/publish by default; one final merge/exchange per layer | Canonical `fmha_sm100.decode_q8kv4.run_decode` with retained shape plan, live sequence lengths/page list, and caller-owned output |
| Any invocation containing prefill | Existing bounded FMHA `OnlyScoreIcp` windows for every row; one NCCL candidate exchange after all windows | Decode prefix uses Q8KV4; prefill suffix uses canonical ICP CSR builder and shared NVFP4 forward/combine |

Index phase comes from the runner's actual CPU `is_prefilling`, independently of query length and public one-token promotion. The ordinary public MSA implementation and CUTLASS adapter are unchanged. New `msa_icp_main.py` contains only host plans, cache aliases, and dispatch. It explicitly requests `num_kv_splits=1, split_mode="streamk", block_scale_shift=3`; MSA's additive split-mode option preserves all ordinary defaults. Plans are shared per device/head geometry and prebuilt for graph buckets. The shared registry is weak: live layers and metadata builders own the schedules, temporary KV profiling release preserves them, and final model shutdown releases layer ownership without invalidating another live model. Page-list offsets are retained per table stride and built outside capture, with no per-layer metadata kernel. All Q8KV4 native/JIT modules are loaded before capture.

Admission requires explicit `msa_icp` plus `cutlass`, BF16 activations, plain NVFP4 main cache, FP8 index keys, TP2/DCP1/PCP1/PP1/DP1, P128/R64, Q64/KV4/indexH4/D128/topK16, and Rubin SM107. Existing offload, sparse block, speculative Q1–Q4 and transport policy constraints are retained. Main QKV remains TP-sharded; the root-owned projection extension replicates all four index-Q heads only when requested by ICP.

Speculative admission is an FP8 `eagle`/`eagle3` drafter using a supported public draft architecture. For example, `Eagle3LlamaForCausalLM` and `Eagle3MiniMaxM2ForCausalLM` resolve to the public Llama-derived Eagle model, which does not construct `MiniMaxM3DecoderLayer`. Draft overrides preserve the target's indexer setting but those classes do not consume it. MiniMax MTP does reuse the MiniMax decoder layer; `method="mtp"` is excluded by ICP admission. This is not an assertion of support for an unregistered MiniMax M3 Eagle architecture.

Startup checks MSA integration ABI 1 and additive `PUBLIC_VLLM_ABI=1`, then the actual writer schema (public scale names plus ICP/device-plan tail). Strict prewarm runs before native main-attention loading; it covers the public stride/global-scale AOT keys and the canonical main-decode artifacts. Shared Q8KV4 callback/host-extension identities must remain isolated from later vendored MSA imports; the MSA compatibility work owns this requirement.

## Validation limits

CPU tests cover both-rank index-Q checkpoint loading, D3 windows and single exchange, writer ownership/replay binding, exact cache aliases/canaries, phase-specific BF16/FP8 query routing, K/V globals, view-only main dispatch, and plan reuse/capture-miss refusal. The reproducible driver is `run_model_checks.py`; source hashes and results are retained with each log. Existing model-loading and public ordinary correctness tests remain intact. Target native writer-to-reader correctness, model accuracy, graph replay, distributed correctness and performance still require separately authorized target validation; host passes make none of those GPU claims.
