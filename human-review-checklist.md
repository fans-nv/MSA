# Internal human review: ICP consolidation into MSA and public vLLM

**Prepared for review in `fans/MSA` and `fans/vLLM`; not upstream merge-ready.**
The candidates are committed, source-bound host checks and artifact checks pass,
and the configured vLLM commit hooks pass. GPU numerical correctness, full-model
graph replay, TP2 execution, synchronization checks, model evaluation and serving
performance remain pending. This report applies the repository's
[PR checklist](https://gitlab-master.nvidia.com/fans/vllm/-/blob/516bd9a9d9962aff25582f774a49640e022bf348/.agents/skills/pr-checklist/SKILL.md)
to existing evidence; it does not claim human approval or repeat the tests.

| Candidate | Frozen commit | Base |
| --- | --- | --- |
| vLLM | `516bd9a9d9962aff25582f774a49640e022bf348` | Public `242e4213fc9845ff6fe607af1aee626fd8acc990` |
| MSA | `0f658079c9ea9ab63c76b975b7ee4dd54a198334` | Dev `ed4e40efcb5895aba1a3554d62cd1dcaf77920cb`; public compatibility delta from `f82fb1759fb0b0f1fd9798c4730484feeb1e3df6` |

Codex and parallel agents assisted with implementation, tests, analysis and review.
The human submitter must review every changed line, understand the contracts,
own the decisions and confirm the required validation before an upstream PR.
No existing PR/MR review threads were supplied for this report.

## Re-review

This is an initial internal review packet. There are no supplied conversation,
review-submission or inline threads to resolve. When feedback arrives, record
each concrete concern and its disposition, and update the description to match
the resulting source. Before any upstream PR, refresh duplicate-work checks;
the related TP4/TP8 whole-block proposal and this TP2 fragment design require
explicit scope coordination, not an assumption that they are equivalent.

## 1. Design fit

- **1.1 Core impact — bounded, still requires runtime review.** Eight generic
  production files change by **+126/-4**: raw page views, actual-phase graph
  admission, alias release and final resource close. Model-specific scoring,
  layout and transport policy stay outside those files. The four-file
  CPU-offload fix (**+78/-19**) is a separate slice. Review the graph/DP phase
  agreement and shutdown ordering even though ICP itself admits DP1.
  [Runtime design and evidence](runtime-review.md).
- **1.2 Reuse — substantially consolidated.** MSA owns indexing, selection,
  publication, exchange and merge; its existing Q8KV4/NVFP4 implementations own
  main attention. vLLM extends its existing fused writer and cache lifecycle.
  No standalone `icp-kernels` runtime dependency or copied main-decode tree is
  introduced. Canonical and existing vendored MSA coexist through explicit
  callback/build identities. [Ownership and compatibility](msa-public-compatibility.md).
- **1.3 Complexity — high.** The complete vLLM change is 52 files,
  **+10,941/-197** including tests; production `vllm/` and `csrc/` alone contain
  29 changed files, **+7,001/-194**. The small generic runtime boundary does not
  make the full feature small. Review the new model indexer, compound layout,
  writer specialization and distributed lifetime as distinct units. MSA's
  larger total includes relocated implementations and tests. Human reviewers
  should assess whether the demonstrated benefit will justify this maintenance
  burden; new end-to-end performance evidence is not available.
  [Exact scope](source-scope.json).
- **1.4 Correctness/compatibility — explicit admission, target proof pending.**
  ICP is opt-in for TP2/ICP2, DCP1/PCP1/PP1/DP1, SM107, BF16 activations,
  NVFP4 main KV, FP8 index keys and P128/R64. Ordinary defaults remain.
  Compound ABI 2 uses public alternating per-head K/V slots; integration ABI 1,
  device-plan ABI 3 and `PUBLIC_VLLM_ABI=1` prevent silent contract substitution.
  CPU controls cover public API preservation, actual-prefill routing and
  calibrated globals. The new writer's bytes have not yet been exercised
  through the canonical readers on GPU. [Model contracts](model-design.md).
- **1.5 Tradeoffs — qualify asynchronous behavior and performance.** Compound
  pages reuse allocation/copy ownership; strict prewarm and retained plans add
  startup and deployment requirements. FULL graphs are restricted by actual
  phase for the opt-in backend. Host scalar reads examined here use CPU
  metadata; producer warmup explicitly synchronizes at startup. This does not
  establish the absence of steady-state synchronization: **run the affected
  serving configuration with `VLLM_GPU_SYNC_CHECK=error`**, including replay,
  speculative verification and relevant offload transitions. Quantify launch
  count, memory cost and end-to-end gains on the agreed target workloads.
- **1.6 MRV1 deprecation — targets MRV2.** New GPU runner hooks are in
  `vllm/v1/worker/gpu/`; no MRV1 feature port is proposed. Shared cache-interface
  changes still need review for ordinary consumers. Public prefill promotion,
  input preparation and ordinary metadata forwarding have preservation checks.
  [Independent runtime review](runtime-independent-review.md).

## 2. Testing and validation

**Host and compilation evidence is complete for this packet; serving validation
is incomplete.** [Final receipt](final-validation.json) binds these results to
the frozen sources; [manifest](manifest.json) binds artifacts and patch trees.

| Existing evidence | Result and limit |
| --- | --- |
| Combined vLLM host suite | 425 passed, 144 skipped; source unchanged |
| MSA host suite | 470 passed, 149 skipped, 413 deselected; source unchanged |
| vLLM model suite without companion | 144 passed, 30 skipped; optional-dependency behavior covered |
| Writer native/ABI checks | SM107a/f compile; 55-argument CPU boxing; all 76 ordinary instruction streams per architecture match the public base |
| Q8KV4 host extension | Both namespaces build/import and reversed cache-only loading pass without CUDA initialization |
| Packaging and target preparation | Sdist → wheel → isolated install preserves 1,138 source payload files; five cache components; exact patch reconstruction; 19 driver guards and three dry plans pass |

- **2.1 Coverage/effectiveness — useful contracts, incomplete device coverage.**
  Tests cover cache alias/stride canaries, fragment ownership, ABI rejection,
  actual-prefill admission, index-Q replication, plan lifetime, cache tampering
  and offload shutdown ordering. Some host tests mock native leaves,
  collectives or graph behavior; they do not demonstrate real transport,
  attention or serving execution. vLLM host runs use `--noconftest` with plugin
  autoload disabled. Review mock coupling and redundant wiring tests before
  upstream submission; retain independent public writer numerical oracles.
- **2.2 Reliability — host checks pass; GPU tolerance/order still to review.**
  Namespace and cache-order regressions have negative controls, including
  reversed import order. Byte equality is intentional for cache packing and
  candidate ordering, not a general requirement for floating-point attention.
  Target runs must assess numerical tolerances, repeated replay/generation
  behavior, multi-rank failure cleanup and test-order/resource interactions.
  A single local host pass is not a flakiness assessment.
- **2.3 CI integration — tethered with a documented gap.**
  `.buildkite/test_areas/models_basic.yaml` runs the new model directory in the
  existing CPU job; the test-tethering hook passes. Companion-dependent tests
  skip when MSA is unavailable. Public CI therefore does not currently establish
  the paired feature or SM107/TP2 behavior. Arrange companion installation and
  target coverage, or attach independently reproduced target evidence before
  upstream submission. Skips and deselections are not passes.

## 3. Code quality and style

- **3.1 Comments — preserve contracts, trim historical narrative in review.**
  Layout, fragment ownership, live bounds and cache/lifetime assumptions need
  explanation. The large model indexer contains extensive commentary; reviewers
  should keep the correctness rationale and assess whether iteration history
  can move to the description. No broad comment cleanup is included here.
- **3.2 Documentation/examples — review docs exist; upstream user docs pending.**
  This packet contains model/writer/runtime contracts, exact reproduction
  commands and limits; MSA includes `docs/ICP_INTEGRATION.md`. The vLLM diff has
  an optional requirement file, but no new feature page or validated serving
  example in `docs/`/`examples/`. Add a concise opt-in example and supported
  configuration matrix when the target flow has been validated. The published
  MSA version alone is insufficient; installation must provide the admitted
  companion capabilities.
- **3.3 Helpers — human maintainability pass remains.** Stable dependency,
  cache-binding and lifetime helpers clarify cross-repository contracts.
  Review one-use forwarding helpers in the large model integration for needless
  indirection; this checklist does not certify an exhaustive helper audit or
  propose changing the frozen implementation without review.
- **3.4 Formatting — completed.** Actual vLLM commit hooks passed, including
  Ruff, mypy, SPDX, test tethering and clang-format **21.1.2**, with no hook
  bypass. The writer's final formatting is linked to native evidence through
  token/preprocessor equality. Avoid unrelated reformatting in follow-up fixes.
  [Hook receipt](logs/vllm-commit-r2.json), [writer record](writer-design.md).

## 4. Pull request contents

- **4.1 Description/context — template and companion draft prepared.**
  [vLLM draft](vllm-human-review-mr.md) and [MSA draft](msa-mr-draft.md) describe
  the feature, ownership and admission. The vLLM description retains the
  repository's [PR template](https://gitlab-master.nvidia.com/fans/vllm/-/blob/516bd9a9d9962aff25582f774a49640e022bf348/.github/PULL_REQUEST_TEMPLATE.md)
  sections and checklist, with exact companion links, measured results and
  unchecked human/target-validation gates. Before upstream submission, add
  finalized predecessor/blocking links and refresh duplicate-work context.
- **4.2 Claims/evidence — narrow claims to what was measured.** Claim optional
  integration, ownership consolidation, host contracts, packaging and the
  recorded compile/code-preservation results. **Do not claim retained serving
  performance or GPU correctness for the public port.** Historical ICP results
  do not qualify the changed public writer/layout. Add model-evaluation and
  paired-performance tables with baseline/candidate revisions, configuration,
  commands, repetitions and variability after approved runs.
- **4.3 Root cause — keep the independent fixes identifiable.** Actual CPU
  request phase prevents short/promoted prefills from taking decode-only graph
  and transport paths. Explicit compound layout/fragment geometry prevents
  wrong slot interpretation. The separate offload fix serializes accepted
  submissions before the shutdown sentinel, drains DMA before unregistering
  buffers, and retains ownership on failure. Host tests exercise those
  contracts; a device/offload reproducer remains pending. No NIXL/PD flow is
  changed.
- **4.4 Details — review in four slices.** Read
  [offload](artifacts/vllm-01-offload.patch),
  [runtime](artifacts/vllm-02-runtime.patch),
  [writer](artifacts/vllm-03-writer.patch), then
  [model integration](artifacts/vllm-04-model-integration.patch), alongside the
  MSA companion. Their cumulative tree is verified; intermediate slice builds
  are not qualified. Focus on cache ABI/strides, qualified ICP versus ordinary
  writer arithmetic, prewarm identities, actual-phase routing and final resource
  lifetime. The offload fix can receive separate review.
- **4.5 Contribution requirements — internal preparation only.** The frozen
  vLLM commit has `Co-authored-by: Codex` and human `Signed-off-by:` trailers;
  both drafts disclose AI assistance. The
  [local contributing guide](https://gitlab-master.nvidia.com/fans/vllm/-/blob/516bd9a9d9962aff25582f774a49640e022bf348/docs/contributing/README.md)
  and [AGENTS.md](https://gitlab-master.nvidia.com/fans/vllm/-/blob/516bd9a9d9962aff25582f774a49640e022bf348/AGENTS.md) require human
  ownership, end-to-end validation, model evaluations and duplicate-work checks
  before upstream submission. Imported MSA source-release provenance still
  needs rights-holder clearance before public distribution; preserved notices
  and internal review do not grant a new license.

## Human follow-up before upstream submission

1. Review every changed line and agree the model/MSA boundaries and separate
   offload slice; record ownership and resolve subsequent review threads.
2. Review a concrete target plan, then validate public writer → canonical
   Q8KV4/NVFP4 readers, GPU numerics, full-model graph replay, actual-prefill and
   mixed batches, supported speculative Q1–Q4, TP2 transport/lifetime and
   relevant prefix/COW/offload round trips. Run `VLLM_GPU_SYNC_CHECK=error`.
   The [prepared driver](target-validation/README.md) covers only selected
   components, not this complete model matrix.
3. Run end-to-end serving and model evaluations. Obtain separate approval for
   the fully resolved benchmark protocol and attach paired performance results;
   no benchmark or allocation action is authorized by this report.
4. Complete companion/GPU CI coverage or attach external target evidence, add
   validated user documentation, clear source-release rights, refresh upstream
   duplicate-work coordination, and complete the PR template/checklist.
