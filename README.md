# ICP consolidation: human review, 2026-10-09

These are the corrected source revisions under active validation. Publishing
them enables human review; it does not assert merge readiness or retained
performance. Earlier review branches remain unchanged. AI assistance was used.

| Repository | Pinned upstream base | Review head |
| --- | --- | --- |
| MSA | vllm-project/MSA **dev**, `968061ab16cb14991a5c7d4c0e8eabf531be0dcf` | `cd20e206a3373e8d258da8cd403b4b3399b1f714` |
| vLLM | vllm-project/vllm main, `ab905a885dfbfc60a2c02286cc9c608c93884de3` | `19afac40273aa84c2693ca1083995c4156aca4e2` |

Both repositories use `review/icp-base-20261009` and
`review/icp-consolidation-20261009`. MSA's base is the requested dev branch.
The vLLM base is the pinned revision used for builds and validation; main has
subsequently advanced. No validated source was rebased during measurement.

| Repository | GitHub comparison | GitLab comparison |
| --- | --- | --- |
| MSA | [fans-nv/MSA](https://github.com/fans-nv/MSA/compare/review%2Ficp-base-20261009...review%2Ficp-consolidation-20261009) | [fans/MSA](https://gitlab-master.nvidia.com/fans/MSA/-/compare/review%2Ficp-base-20261009...review%2Ficp-consolidation-20261009) |
| vLLM | [fans-nv/vllm](https://github.com/fans-nv/vllm/compare/review%2Ficp-base-20261009...review%2Ficp-consolidation-20261009) | [fans/vllm](https://gitlab-master.nvidia.com/fans/vllm/-/compare/review%2Ficp-base-20261009...review%2Ficp-consolidation-20261009) |

Publication results and any incomplete destination are recorded in
[publication.json](publication.json). The comparisons contain 147 changed MSA
files (+38,871/−581 lines) and 52 changed vLLM files (+10,977/−198 lines),
including tests, packaging, documentation, and licenses. This remains a
substantial review; the final correctness repairs themselves are small.

## Components and final repairs

MSA owns the local indexer, candidate selection/merge, TP candidate transport,
device plans, and source-keyed prewarm/cache admission. Prefill scoring uses
MSA's shared FMHA implementation; sparse NVFP4 attention and decode adapters
reuse the shared MSA implementations. vLLM owns the existing fused QK-norm,
RoPE, and cache writer with an explicit ICP mode, plus the model and runtime
integration. There is no separate ICP writer binary in the public candidate.

The final MSA commit fixes an inherited four-stage decode scorer race: an
async-proxy fence orders shared-memory reads before the producer can reuse a
TMA stage. The unchanged inherited scorer failed exact standalone tests;
the fence-only variant passed. The committed regression covers long context,
both TP ranks, eager execution, and changed-query graph replay. A new source
key forces fresh corrected AOT artifacts.

The final vLLM commit restores the model-local dense FP8 policy when sparse
attention uses NVFP4. It passes a copied cache configuration to dense
attention; the sparse configuration and native inputs remain unchanged.
This resolves the first model boot's unsupported dense NVFP4 path on SM107.

The runtime changes expose the complete compound page to cache allocation,
binding and offload; keep graph metadata and plans bound to the current cache;
and drain/release owned transport resources during profiling, rebinding and
shutdown. Model-specific index-Q/K projection replication is explicit.

## Validation status

Fresh native vLLM core, MoE, FlashAttention and supporting targets were built
against the unchanged Rubin CUDA 13.5/Torch/FlashInfer environment. MSA and
vLLM source/import/native hashes were recorded per arm. No old vLLM native
binary was substituted for a candidate build.

- Corrected full-indexer decode correctness passed all 36 C8/C12/C16 shapes,
  eager and changed-query graph execution, on both ranks.
- Writer-to-reader integration passed 34 numerical/layout cases, including
  nonuniform scores and changed TopK. Real two-rank transport passed.
- Actual loaded first/later-layer producer admission compared signed eager
  query bytes, complete touched compound pages and exact TopK against graph
  replays. Corrected reference and candidate passed all four cases.
- Ordinary public MSA versus pinned dev produced bitwise-equal inputs and
  five outputs from separate fresh caches; all five CPU FP32 reference checks
  passed the unchanged 0.9999 threshold. A NaN-producing GPU reference failure
  is preserved and is not counted as a passing test.
- ICP-on model boot, cache ownership/lifecycle, producer RPC and Eagle3
  routing checks passed. Autotune policy admission passed separately.
- Two complete GSM8K accuracy pairs were independently rescored. Each boot
  used 16 warmups and all 1,319 scored examples, C16, zero-shot adaptive chat,
  temperature 0, top-p 1, and at most 512 generated tokens.

| Paired round | Historical image | Corrected candidate | Candidate − historical |
| --- | ---: | ---: | ---: |
| 1 | 1260/1319 (95.5269%) | 1254/1319 (95.0720%) | −0.4549 percentage points |
| 2 | 1265/1319 (95.9060%) | 1261/1319 (95.6027%) | −0.3033 percentage points |

Mean paired difference: **−0.3791 percentage points**. Both candidate boots
exceeded 95%. These two pairs do not establish accuracy equivalence. The
historical arm has a different image, attention backend and UGPU policy, so
this is a whole-system comparison. Original scorer defects and original
failed runs remain recorded separately; corrected controls are labeled
`R_fixed`, `B_fixed` and `K_fixed`.

## Open before merge

Module timing is in progress, with some adverse early cells showing slower
candidate results. The full module and whole-model campaigns and their
analysis remain incomplete. Six independent pairs and the declared warmup and
measurement counts remain required. Incomplete runs, clock violations,
compilation during measurement, and unqualified observer overhead cannot
support a performance-retention claim. Prefill/mixed and legacy-core coverage
must be reported separately from completed decode coverage.

The ordinary **ICP-off whole-model path remains failing** with a CUDA launch
error. This is separate from the passing standalone ordinary MSA comparison.
The error surfaces after ordinary sparse attention and requires a fresh-cache,
synchronous diagnostic run to identify the first failing operation. No
production workaround has been applied and no cause is claimed.

Upstream submission remains pending the listed model, performance and
compatibility checks and the human review requested by the user. No upstream
PR or MR was opened by this publication.

[MSA MR description](msa-mr-description.md) ·
[vLLM MR description](vllm-mr-description.md) ·
[Evidence snapshot](validation-snapshot.json)
