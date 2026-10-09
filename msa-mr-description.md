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

Validation completed on the existing Rubin image: all 36 decode module
shapes at C8/C12/C16 passed eager/graph correctness; 34 writer-to-reader cases
and real two-rank transport passed; loaded first/later-layer signed-input
producer/graph admission passed. Separate ordinary MSA and pinned-dev caches
produced bitwise-equal outputs in five tested variants, each passing an
independent CPU FP32 oracle at the unchanged threshold. Exact receipts and
limitations are summarized in [README](README.md).

Two complete model accuracy pairs produced candidate scores 95.0720% and
95.6027%, versus historical 95.5269% and 95.9060%. The mean difference was
−0.3791 percentage points. This does not establish equivalence; the historical
image is a whole-system reference. Six-pair performance analysis is pending,
and the companion vLLM ordinary ICP-off model route has an unresolved CUDA
failure. This branch is for human review and is not presented as merge-ready.

AI assistance was used. The human submitter must review the code and final
validation evidence. Duplicate-work checks and upstream submission review
remain pending; this draft has not been submitted as an upstream PR.
