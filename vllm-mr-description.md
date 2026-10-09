# Integrate MiniMax-M3 ICP with compound cache pages and the existing KV writer

The ICP route previously owned a separate cache-write implementation and
duplicated runtime assumptions. This change adds an opt-in ICP mode to the
existing fused MiniMax-M3 QK-normalization, RoPE and KV-insert operation, and
uses the companion MSA package for indexing and candidate communication.
The model retains main NVFP4 KV data and rank-local index fragments inside one
compound cache page.

Target: vllm-project/vllm main at the validated pin
`ab905a885dfbfc60a2c02286cc9c608c93884de3`.
Review head: `19afac40273aa84c2693ca1083995c4156aca4e2`.
This is a pinned-base review; upstream main has advanced since validation began.

[GitHub diff](https://github.com/fans-nv/vllm/compare/review%2Ficp-base-20261009...review%2Ficp-consolidation-20261009)
· [GitLab diff](https://gitlab-master.nvidia.com/fans/vllm/-/compare/review%2Ficp-base-20261009...review%2Ficp-consolidation-20261009)

The existing ordinary writer call keeps its default behavior. Explicit ICP
arguments select fragment ownership, the qualified ICP quantization policy,
live metadata and optional device-plan production. Tests cover native ABI,
whole-page bytes, scale layout and downstream reader composition.

Core runtime changes expose all compound-page bytes to allocation, binding,
copy/offload and cache lifecycle handling. Graph metadata and plans track the
current cache binding; model-owned candidate exchange is drained before
release. Small worker hooks handle profiling teardown/rebind, final shutdown,
and metadata slicing. The model-specific index Q/K projection replication
contract is explicit.

The final model-local repair copies NVFP4 cache configuration as FP8 only for
dense attention. Without it, the first real model boot selected unsupported
dense NVFP4 on SM107. Sparse attention keeps its original NVFP4 configuration.
Five focused CPU tests pass, the change was independently reviewed, and the
patch changes no native compiled input.

Fresh core, MoE, FlashAttention and supporting native modules were built
against the unchanged Rubin environment; all import origins and artifact
hashes were bound per arm. No image vLLM native library was reused as a
candidate build. Corrected ICP module decode correctness, 34 writer/reader
cases, two-rank transport, actual producer graph snapshots, model cache and
resource lifecycle, Eagle3 routing and autotune policy admission passed.

Two complete GSM8K pairs used 16 warmups plus all 1,319 examples per boot,
C16, zero-shot adaptive chat and a 512-token cap. Historical/candidate scores
were 95.5269/95.0720% and 95.9060/95.6027%; mean paired difference was
−0.3791 percentage points. There is no equivalence claim, and H/B includes
image/backend/UGPU differences. See [README](README.md) for evidence scope.

Open: the ordinary ICP-off whole-model route fails with a CUDA launch error;
the first failing operation has not yet been localized. Passing standalone
ordinary MSA tests do not close that gap. Six-pair module/model performance
validation and observer/JIT admission remain incomplete. The branch is
published for human review, not merge approval.

AI assistance was used. Before opening an upstream PR, the human submitter
must review every changed line, run the remaining checks, complete the
repository's duplicate-work checks and PR checklist, and include the final
model evaluation and performance results. No upstream PR has been opened.
