# ICP consolidation: human review

Review the paired source branches below. This is a GitHub fork review candidate:
CPU tests, native compilation and packaging checks passed; GPU correctness,
distributed execution, graph replay, model evaluation and retained serving
performance have not been verified for these commits.

| Repository | Candidate | Pinned review base | Review diff |
| --- | --- | --- | --- |
| [fans-nv/MSA](https://github.com/fans-nv/MSA) | `0f658079c9ea9ab63c76b975b7ee4dd54a198334` | MSA dev `ed4e40efcb5895aba1a3554d62cd1dcaf77920cb` | [MSA changes](https://github.com/fans-nv/MSA/compare/review%2Ficp-base-20261008...review%2Ficp-consolidation-20261008) |
| [fans-nv/vllm](https://github.com/fans-nv/vllm) | `516bd9a9d9962aff25582f774a49640e022bf348` | Public main `242e4213fc9845ff6fe607af1aee626fd8acc990` | [vLLM changes](https://github.com/fans-nv/vllm/compare/review%2Ficp-base-20261008...review%2Ficp-consolidation-20261008) |

Both source branches are named `review/icp-consolidation-20261008`; both
comparison bases are `review/icp-base-20261008`. Use the pinned base for this
review. MSA's base is an ancestor of `vllm-project/MSA:dev`; at the latest check,
dev had advanced by one commit to `968061ab16cb14991a5c7d4c0e8eabf531be0dcf`
(FP8 sparse prefill with strided K/V). This candidate was not rebased after
validation. Existing repository defaults are not the comparison targets.

The `review/icp-review-notes-20261008` branch contains these notes and selected
evidence, separate from the exact tested source commits. The publication
receipt in the preparing workspace records remote verification after pushing.
Both GitHub destination repositories are public. Fork publication is
requested by the author; source-release provenance review remains open
and no new license grant is asserted.

## Review order and ownership

1. [MSA compatibility](msa-public-compatibility.md) and
   [model contract](model-design.md): local scoring, selection, candidate
   publication/exchange/merge live in MSA dev; main attention reuses canonical
   Q8KV4 decode and NVFP4 prefill/combine. Inspect the compound-page ABI,
   calibrated scales, strict prewarm inventory and namespace isolation.
2. [Writer](writer-design.md): the existing vLLM fused writer owns main KV,
   rank-local index keys and live metadata/device-plan inputs. Inspect ordinary
   versus ICP arithmetic and the preserved public operator prefix.
3. [Runtime](runtime-review.md) and [independent review](runtime-independent-review.md):
   raw-page views, actual-prefill graph admission and resource lifetime. The
   generic production boundary is eight files, +126/-4 lines.
4. Review the [offload](artifacts/vllm-01-offload.patch),
   [runtime](artifacts/vllm-02-runtime.patch),
   [writer](artifacts/vllm-03-writer.patch) and
   [model](artifacts/vllm-04-model-integration.patch) slices. The separate
   CPU-offload production fix is four files, +78/-19 lines. The cumulative
   slice tree matches the source candidate; intermediate builds are unqualified.

The full vLLM port is 52 files, +10,941/-197 lines including tests and CI.
The full MSA dev consolidation is 147 files, +38,818/-581 lines including
relocated implementations, tests and packaging. Review the complete feature's
maintenance cost as well as its smaller core-runtime boundary.
[Per-file scope](source-scope.json).

## Validation and remaining work

| Completed check | Result |
| --- | --- |
| vLLM model/runtime/writer host suite | 425 passed, 144 skipped |
| MSA host suite | 470 passed, 149 skipped, 413 deselected |
| Model contracts without companion MSA | 144 passed, 30 skipped |
| Configured vLLM commit hooks | All applicable hooks passed; no bypass |
| CUDA writer compilation | SM107a and SM107f pass; 76 ordinary instruction streams per architecture match the public base |
| Q8KV4 native host extension | Two namespaces build/import; reversed cache-only import passes without CUDA initialization |
| Packaging | 1,138 source payload files agree across sdist, wheel and isolated installation |
| Target preparation | 19 host guards and three candidate dry plans pass; no target execution |

[Final validation](final-validation.json) binds results to the source trees.
[Manifest](manifest.json) SHA256 is
`0c2a96870fed4b22fe550fc4402b5cd9cd2867922b20f6fd5c4f1387d484d1b1`.
The manifest's `published: false` records the sealed local checkpoint; the
later publication receipt supersedes that field only for GitHub fork branch
publication. It does not change any qualification result.

Local vLLM host runs used `--noconftest` and disabled plugin autoload. Some
contracts mock native leaves or collectives. Companion-only CI cases skip
without MSA. Skips, compilation and code-preservation checks are not GPU
numerical or serving evidence.

Before upstream submission, review the [complete checklist](human-review-checklist.md),
validate public writer-to-reader numerics, TP2 transport/lifetime, full-model
graphs and mixed/prefill/speculative paths, run `VLLM_GPU_SYNC_CHECK=error`,
model evaluations and an approved paired performance protocol. Resolve the
MSA source-release provenance items and refresh duplicate-work coordination.
No benchmark launch or allocation operation is part of this publication.

## MR preparation and evidence contents

[vLLM MR description](vllm-human-review-mr.md) retains the upstream template and
unchecked validation/ownership gates. [MSA MR description](msa-mr-draft.md)
describes the companion implementation. They are prepared descriptions; no
upstream PR or MR has been opened by this publication.

These changes and notes were prepared with Codex and parallel-agent assistance.
The human submitter must review the diff and own the final submission.

This notes branch carries selected text evidence and source patches. Compiled
objects, cache directories, wheels/sdists and Git bundles remain in the original
workspace packet and are not placed in the source repositories. The artifact
manifest records their hashes. Evidence logs retain their original local paths.
The reproduction drivers expect the recorded workspace layout: this packet at
`review/vllm-public-consolidation-20261008`, source trees under `worktrees/`, and
the uv-managed review environment. Read each driver and its target requirements
before rerunning it. `REVIEW_FILES.json` inventories the published notes.
