# Four-stage decode scorer correction

The original B and R implementations fail the exact indexer oracle at
B12/Q1/150,000 tokens. Original source trees and failed receipts are preserved.
The failure is in local scoring before candidate selection or transport. For
example, request10/head0/block235 scored245 instead of188 and displaced a block
whose true score is234. Integer-valued FP8 inputs make these FP32 dot products
exact; no tolerance change is justified or used.

`decode-score-isolated-r1` reproduces the defect without a writer, selector, or
transport. The default four-stage scorer fails every one of12 sign-changing
checks across the two rank fragments (99–190 wrong live cells per call).
One-, two-, and eight-stage diagnostic variants pass36 checks. All48 query and
cache byte comparisons pass.

An isolated `cute.arch.fence_view_async_shared()` immediately before each
consumer arrival on the empty-stage barrier passes24 checks with the original
four stages and split128. This orders generic shared-memory accesses before
the TMA async proxy can reuse the stage. The isolated pass is not full module
admission. Production defaults, inputs, tolerances, and query/head layouts stay
the same.

The controlled consolidation comparison will explicitly use **R_fixed** and
**B_fixed**, each with this same minimal scorer correction. Original **R** and
**B** remain available as defective historical evidence. New source commits,
patch digests, cache manifests, and bound import paths must be recorded before
admitting either fixed arm. Correctness gates are retained.

The immutable r3 matrix and the existing complete-call timing amendment remain
unchanged:36 decode shapes first, then18 prefill and6 mixed shapes, plus18
untimed guards; C8/12/16 and all specified context/query lengths; six independent
paired rounds;20 decode warmups and100 samples of10 complete calls in eager and
graph modes;10 prefill/mixed warmups and50 samples of one call in eager mode.
Untimed rank-order barriers remain outside each complete-call event and host
interval. Maximum memory clock4752MHz and continuous telemetry remain required.
There is no performance result or performance-retention claim yet.

The loaded-producer comparison must also bind the separately documented,
historical-equivalent dense-FP8 model helper restoration used for model startup.
Its source identities must be explicit in both fixed arms; it does not alter
the native writer or sparse kernels. K remains a separately identified legacy
comparison with its own dependency compatibility gates.
