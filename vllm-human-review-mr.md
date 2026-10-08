<!-- markdownlint-disable -->

# Draft: [MiniMax M3] Consolidate ICP integration with MSA dev

## Overview

Add optional TP2 indexer context parallelism using companion MSA dev kernels
and vLLM's existing fused KV writer. Prepare this change for human review in
the author's GitHub forks; target-GPU and serving validation remain pending.

Candidate: [vLLM `516bd9a9`](https://github.com/fans-nv/vllm/commit/516bd9a9d9962aff25582f774a49640e022bf348),
paired with [MSA `0f658079`](https://github.com/fans-nv/MSA/commit/0f658079c9ea9ab63c76b975b7ee4dd54a198334).
Compare `review/icp-consolidation-20261008` against `review/icp-base-20261008`.

## Claims

- Consolidate index scoring, selection and candidate exchange in MSA dev;
  reuse canonical Q8KV4 decode and NVFP4 prefill/combine.
- Extend the existing fused vLLM writer for ICP fragments and live metadata,
  retaining its public argument prefix and ordinary device instructions.
- Limit generic runtime production changes to eight files (+126/-4): raw
  cache views, actual-phase graph admission and resource lifetime. Keep the
  CPU-offload drain/unpin fix in a separate review slice.

No GPU correctness, accuracy or retained-performance claim is made for this
public port.

## Validation

| Check | Result |
| --- | --- |
| Combined vLLM host suite | 425 passed, 144 skipped |
| MSA host suite | 470 passed, 149 skipped, 413 deselected |
| Configured vLLM hooks | All applicable hooks passed |
| Native writer | SM107a/f compilation passes; 55-argument CPU boxing passes |
| Ordinary writer code | All 76 instruction streams per architecture match the public base |
| Q8KV4 host extension | Both namespaces build/load, including reversed cache-only loading |
| GPU numerics, full-model graphs, TP2, model evaluation and performance | Not run; required before upstream submission |

The [review notes](https://github.com/fans-nv/vllm/blob/review/icp-review-notes-20261008/README.md)
contain source-bound receipts, commands and limitations. In the recorded workspace:

```bash
bash review/vllm-public-consolidation-20261008/run_checks.sh NEW_TAG integration
bash review/vllm-public-consolidation-20261008/run_checks.sh NEW_TAG msa
```

vLLM host tests used `--noconftest`; native/collective leaves are mocked in
some host contracts. The model tests are wired into CPU CI, with companion
cases skipped when MSA is absent. Skips are not passes.

## Details

ICP is opt-in through `minimax_m3_msa_indexer_backend=msa_icp` with explicit
`minimax_m3_msa_decode_backend=cutlass`. Admission is TP2/ICP2,
DCP1/PCP1/PP1/DP1, SM107, BF16 activations, NVFP4 main KV, FP8 index keys,
P128/R64 and top-16 sparse blocks. Compound pages use public per-head K/V
slots without forward repacking. Actual request phase prevents promoted
prefills from selecting decode-only FULL graphs/transport.

MSA is based on dev `ed4e40e`; the later `968061a` update is not included in
this tested snapshot. The full vLLM feature is 52 files (+10,941/-197),
including model integration, tests and CI; it needs a substantive human
maintainability review despite the small generic-runtime delta.

The related [TP4/TP8 whole-block proposal](https://github.com/vllm-project/vllm/pull/59980)
has a different fragment/topology scope. Current status and duplicate-work
coordination must be refreshed before upstream submission. No existing review
threads are claimed resolved by this draft. Codex and parallel agents assisted
with this work; a human must review and own the final submission.

---

<details>
<summary> Pull Request Checklist </summary>

- [x] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [ ] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [ ] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>**
