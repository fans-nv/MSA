# Add ICP indexing and candidate transport to MSA

ICP indexing currently depends on a separate kernel package. This change
places its scorer, local candidate selection, merge/transport and device-plan
interfaces in `fmha_sm100.icp`, while sharing MSA's FMHA prefill and sparse
attention implementations. vLLM retains cache production and model/runtime
ownership through the companion integration.

Target: **vllm-project/MSA dev** at
`968061ab16cb14991a5c7d4c0e8eabf531be0dcf`.
Review head: `cd20e206a3373e8d258da8cd403b4b3399b1f714`.

[GitHub diff](https://github.com/fans-nv/MSA/compare/review%2Ficp-base-20261009...review%2Ficp-consolidation-20261009)
· [GitLab diff](https://gitlab-master.nvidia.com/fans/MSA/-/compare/review%2Ficp-base-20261009...review%2Ficp-consolidation-20261009)

The package includes explicit ABI validation, source-keyed native/AOT caches,
prewarm and strict-cache admission, package-relative imports, and regression
coverage for ordinary MSA behavior. ICP's source-dependent cache keys and
transport lifetime are explicit rather than implicit process globals.

Validation found an inherited four-stage decode score race at long context.
The final commit inserts `fence_view_async_shared()` before empty-stage
arrival so subsequent TMA writes cannot race the consumer's generic shared
reads. The fence-only isolated variant passed 24 exact checks after the
unchanged scorer failed all 12 baseline checks. The committed regression
covers both ranks and changed-query graph replay; corrected artifacts have a
new source key. Original references are retained and corrected controls are
named explicitly.

Validation on the unchanged Rubin base passed all 60 candidate core cases
(36 decode, 24 prefill/mixed), all 18 guards with clean finalization,
writer/reader composition, TP transport and actual loaded-producer graph checks.
Ordinary MSA and pinned dev produced bitwise-equal tested outputs in five fresh
cache variants against the independent CPU oracle.

Performance retention is not established: five of six module pairs completed;
C16/Q1/64K core graph was +13.7% slower and C12/Q1/150K later producer graph
was +16.5%, each slower in all five pairs. All samples remain. Primary graph
warmup qualification, K/B and prefill/mixed timing remain open. K/R/B CUPTI
profiles are diagnostic, with no standalone bandwidth or causal claim.

The companion ordinary ICP-off model route still fails. The synchronous retry
failed in IPC setup before model execution; no final r3 packet was authored or
launched. Whole-model decode collected one pair with zero accepted scheduler
windows. GSM8K's two candidate scores were 95.0720%/95.6027%, with mean paired
difference −0.3791 percentage points versus H; no equivalence claim.

See [the final review summary](README.md) and
[performance report](evidence/hecate-723744/PERFORMANCE-REPORT.md).
This source is published for human review; merge readiness is not established.
AI assistance was used. Upstream submission remains pending the listed checks
and the human review requested by the user; no upstream PR has been opened.
