# Independent public NVFP4 adapter/prewarm review

Read-only review of `msa-dev-public` changes in
`python/fmha_sm100/icp/{__init__,attention/nvfp4_prefill/__init__,attention/nvfp4_prefill/_prewarm}.py`
and `tests/icp/test_prewarm_cache.py`, with the actual shared reader, combine,
NVFP4 view helper and strict manifest checker as references.

No outstanding finding remains in this scope after the combine-key correction.

- The public profile derives views with the existing MSA head-slot helper from
  alternating K/V slots. K data/K SF/V data/V SF offsets are
  0/8,192/9,216/17,408 bytes, page stride 45,056 and head stride 18,432.
  The backing page is transferred before views are made, retaining its stride.
- The real forward dispatcher receives K calibration. It suppresses V tensor
  calibration there and supplies it to combine exactly once, as the existing
  dev implementation specifies. The new test compares actual tensor storage
  aliases after scalar normalization.
- Exact AOT keys include the actual strides, calibration-presence flags,
  pair-dequantization environment, architecture and combine staging. Both the
  old region-major/no-calibration profile and new public profile remain required.
  Object basenames, serialized keys, companion JSON files and declared sealed
  inventory all participate in admission; a historical-only export is rejected.
- Review found the new calibrated public combine key initially hard-coded
  `min_blocks_per_mp=0` for non-SM107 architectures. The real dispatcher uses
  3 there. Root corrected it to 3 for public/non-SM107 and 0 otherwise, then
  expanded actual-loader controls to SM100, SM103 and SM107. Source inspection
  confirms the correction matches the runtime branch.

The final focused host run at `logs/public-nvfp4-r3/nvfp4` passed 15 tests with
26 deselected, exit code 0 and unchanged source snapshots. These tests substitute native
load/compile leaves and forbid CUDA initialization; they verify host routing,
keys and strict inventory, not device numerical execution or performance.
