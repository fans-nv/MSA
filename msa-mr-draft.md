# Add ICP indexing and candidate exchange to MSA dev

MiniMax M3 indexer context parallelism partitions each P128 index-key block
across two ranks, scores the local R64 fragment, and exchanges compact top-k
candidates. This consolidates the indexing and transport implementation under
`fmha_sm100.icp` on MSA **dev**, removing the serving integration's dependency
on a separate `icp-kernels` package. KV writing stays in vLLM's existing fused
writer.

The existing shared FMHA scorer gains an additive `OnlyScoreIcp` specialization.
Decode retains the CuTe H4 scorer; selection and candidate publication share
the existing fused window path, followed by one final merge/exchange. Main
attention uses MSA's shared Q8KV4 decode and NVFP4 sparse prefill/combine. The
existing scorer and transport schedules, bounded score storage, candidate
ordering and tuning are preserved in the consolidation.

The public-vLLM compatibility layer adds the following:

- A separate public compound-page prewarm profile with alternating per-head
  K/V slots and calibrated device K/V globals. The historical profile remains
  supported; no cache repack is added.
- An explicit Q8KV4 `split_mode` option so ICP can retain stream-k with one
  split; ordinary split selection keeps its existing defaults.
- Namespaced native callbacks and host-extension identities, allowing the
  canonical package and an older vendored MSA to coexist. Host-library loading
  and building use the same process lock.
- Strict prewarm inventory for the Q8KV4 host extension, plan, both GQA16
  shift-3 forward modules, source records and cached QMUL4 capability. Cache
  validation covers the resolved toolchain and package source inputs.
- Additive `PUBLIC_VLLM_ABI=1` admission for the companion vLLM integration;
  integration ABI 1 and device-plan ABI 3 remain intact.

The changes are based on dev `ed4e40efcb5895aba1a3554d62cd1dcaf77920cb`.
The review packet exports both the full dev consolidation and the public
compatibility delta from `f82fb1759fb0b0f1fd9798c4730484feeb1e3df6`.
Final validation results and exact source revisions are recorded in the
packet manifest and validation receipt. CPU contracts and compilation do not
establish target correctness or retained serving performance; GPU numerical,
graph, distributed and paired performance evidence must accompany submission.

Prepared with Codex assistance and parallel agent review; not submitted.
The human submitter must review the changes and resolve the source-release
provenance items inherited from the earlier consolidation packet. Those
items remain open in this GitHub fork review publication. Existing license
notices are preserved; publication does not assert a new license grant.
