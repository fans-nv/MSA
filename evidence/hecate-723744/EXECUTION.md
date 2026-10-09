This campaign resumes the ICP consolidation validation using the existing Rubin
base image and exact candidate/reference revisions already built in 721392.
The user replaced the former benchmark-approval instructions on 2026-10-08:
execution within the authorized task needs no additional launch approval.
Parent allocation cancellation/release/requeue remains forbidden without an
explicit instruction identifying the target allocation and action.

Actual execution uses RUNNING allocation 723744, partition batch-xdr, nodes
hecate0010 and hecate0011, ending 2026-10-09 02:45:19 PDT. The original drain deadline
was 02:25:19 PDT. Later finite packets below use separately recorded future
drain profiles; the parent allocation end time is unchanged. The clock-control prerequisite passed on both node GPU0/1
pairs at 22:45 PDT: use `ops/run_with_clocks_r3.py` and its per-run telemetry.
No additional launch approval is required. Earlier pending-node estimates and
failed clock-control attempts below remain historical preparation evidence.

Execution update, 2026-10-09 02:46 PDT (09:46 UTC):

After all owned work had completed, allocation723744 reached its scheduled
walltime. Slurm accounting records TIMEOUT at09:45:25 UTC; see
[final parent accounting](ops/allocation-final-r1.json). No agent terminated,
released or requeued the allocation.

Final read-only inventories at 09:42:48 UTC on hecate0010 and 09:43:09 UTC
on hecate0011 found zero GPU compute processes on all four GPUs of each node.
The runtime inventory also found no owned vLLM/torchrun/clock-wrapper processes.
Every workload launcher has been reaped; the parent allocation remained RUNNING.
No parent cancellation, requeue or release was performed. See
[kernel-node cleanup](kernels/final-node0010-cleanup-r1.json) and
[runtime-node cleanup](runtime-ops/runtime-idle-final-r1.json).

All performance collection has finished. Five complete R_fixed/B_fixed module
pairs contain all 216 cells/arm and all raw timing intervals. The original
controller was reaped after its deadline guard; a separately reviewed immutable
prefix completed only the missing B4 arm. No sixth half-pair was started. All
ten arm clocks/cleanup pass. The primary warmup actually executes 20 uncaptured
calls then capture and one validation replay, not 20 graph replay warmups.
The [performance report](PERFORMANCE-REPORT.md) records adverse cells and
this qualification without changing the original results.

The candidate model arm completed at 09:05:04 UTC with successful owned-child
and clock cleanup. R/B each completed 72 warmup and 360 measured requests,
ISL65536/OSL1024, C8/C12/C16, 2C warmup and 10C measured per cell, offered
rate infinity subject to C. One fresh TP2/ICP2 Eagle3 k=3 server per arm was
reused across the three cells. All original pure-decode windows fail their
full-cohort scheduler check. The [one-pair report](analysis/model-decode-one-pair-r1/REPORT.md)
labels whole-request rates and the separate post-hoc diagnostic. No accepted
pure-decode pair is claimed. Verbose JIT audits pass only for monitored APIs.

K/R/B CUPTI core diagnostics subsequently completed on hecate0011 GPU0/1.
Each fresh two-rank process used the existing Rubin base and source-bound
native/JIT artifacts, 12 shapes C8/C12/C16 × Q1/Q4 × 64K/150K, eager and
graph, 20 warmups of the selected callable and ten profiled complete calls.
There are no client requests or request rate in this kernel-only protocol.
One process per arm is a diagnostic repetition, not a primary timing pair.
All 144 traces, 1440 CPU call scopes and 4320 GPU core kernels were independently
reviewed. Per-arm wrapper timeout was 420 seconds, profiler child timeout
300 seconds; the full 480-second cleanup admission guard was retained. R/B/K
clock cleanup completed around 09:07, 09:17 and 09:29 UTC respectively.
All 4752 MHz telemetry and owned-lock resets passed.

Exact finite launch commands, executed from the local workspace, were:

```sh
ssh fans-mfa@login-hecate 'bash /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/kernels-tail-prefix-r1/launch-reviewed-prefix.sh'
ssh fans-mfa@login-hecate 'bash /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/kernels-diagnostics-r2/launch-R_fixed-r3.sh'
ssh fans-mfa@login-hecate 'bash /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/kernels-diagnostics-r2/launch-B_fixed-r3.sh'
ssh fans-mfa@login-hecate 'bash /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/kernels-diagnostics-K-r1/launch-K_fixed.sh'
```

The scripts bind allocation723744, exact UUIDs, container SHA
`5bdfa010c240579625c4e21ffdb95e41132cd18ae2b1223d717b3eb06b92f0d3`,
source/native manifests, outputs and profiles. Their result tags are
`module-five-pair-prefix-r1`, `core-cupti-R_fixed-r2`,
`core-cupti-B_fixed-r2`, and `core-cupti-K_fixed-r1`, under the campaign's
remote `results/`; clocks are under `ops/diagnostic-clocks/` for profiles.
K's driver differs from the reviewed R/B version only by enabling the bound
K arm. Separate cache holds prevented concurrent mutation of a shared arm's
artifacts; all those holds have now been released. Stages overlap under PDL;
CUPTI results do not measure standalone NVLink bandwidth or cross-node NICs.

Both public arms passed all 24 prefill/mixed untimed core cases. K passed all
36 decode cases. R's 18 guards completed every numerical check on both ranks,
but reached the 600-second child cap during final distributed shutdown, about
two seconds after final numerical receipts. This does not prove a deadlock.
B's matching final untimed group launched at 09:31:36 UTC, passed all 18 cases
on both ranks and completed its top-level result at 09:41:45 UTC. Root reaped
the owned launcher successfully. Its exact command was:

```sh
ssh fans-mfa@login-hecate 'bash /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/kernels-coverage-r2/launch-B_fixed-core-guards-node0010-r2.sh'
```

That packet uses hecate0010 GPU0/1, source-bound candidate environment in the
same base container, strict cached execution and all 18 frozen guard cases.
Each case has three eager oracle checks and, when supported, three changed-query
graph checks per rank. It is one correctness process, with no timed requests,
request rate or performance warmups. The child timeout is 600 seconds and
Slurm child step 720 seconds. Outputs are
`results/B_fixed-core-guards-node0010-r2`. Cleanup is scoped to its owned
processes and step; no parent allocation command is included.

The ordinary ICP-off synchronous fresh-cache diagnostic r2 was attempted at
09:19 UTC, using the unchanged ten correctness requests and one fresh server.
It failed before any model workers or requests because its private TMPDIR
made the Unix IPC socket address too long. Workload ended09:20:03.741 UTC;
clock resets completed09:20:04.636. This is a harness failure, not kernel
validation. The supported `VLLM_RPC_BASE_PATH=/tmp` correction was identified
for a proposed separately bounded retry, but no finalized r3 packet was
authored or launched before its admission cutoff. The original ordinary model CUDA failure remains open.

```sh
ssh fans-mfa@login-hecate 'bash /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/runtime-ordinary-diagnostic-r2/launch.sh'
```

Packet SHA is `a5bc7a31f3fdd05a292c197c6367cb494e81c2c4289d811f63ee06d152488b6e`.
TP2/ICP-off, eager, no speculation, CUDA_LAUNCH_BLOCKING=1; ten request cases
use C2 at ISL2048/4096 and C8 at ISL2048/2049/4096/4097/8192/8193/16384/16385,
OSL32, rate infinity, zero warmup and one drained cohort per case. Wrapper
workload cap1010 seconds plus480-second cleanup guard; separate drain09:44:19.
All thirteen mutable caches were private. Exact worker/client/RPC/cleanup
budgets and output paths remain in the packet. Original logs and the
[offline failure receipt](analysis/ordinary-r2-startup-failure.json) are preserved.

The source/base refs remain published in all four authorized repositories.
Review evidence is being refreshed; validated source heads are unchanged.
No parent allocation was terminated, cancelled, requeued or released.

Historical execution update, 2026-10-09 01:52 PDT (08:52 UTC):

Both full GSM8K paired rounds completed, with H/B scores 1260/1254 and
1265/1261 out of 1319 per boot. All four fresh boots passed request counts,
4752 MHz telemetry and owned cleanup. Independent rescoring gives a mean
candidate difference of −0.37908 percentage points. Both candidate runs
exceed 95%; equivalence is not established.

The reference 64K/1024 C8/C12/C16 model run completed 72 warmup and 360
measured requests at 08:44:49 UTC. All 30 client windows meet their duration
and token counts, but the final one or two intersecting scheduler envelopes
have fewer than C requests. Reduced scheduling precedes the first client
completion by approximately 0.8–15 ms. These windows remain invalid for the
predeclared pure-decode metric. No retroactive trimming or substitute metric
is accepted. The verbose JIT audit finds no monitored JIT-event overlap in
the measured populations; its scope is instrumented APIs only.

Root reaped the original model controller after its deadline guard refused
the original candidate launch. It had verified reference cleanup, released
the reference cache hold, verified module B3 cleanup and acquired the exact
candidate cache hold. A separate readiness receipt at 08:49:59 UTC confirms
the original and replacement candidate outputs are packet-only, all four
hecate0011 GPUs are idle, and allocation 723744 remains RUNNING. Root then
launched the reviewed candidate packet at approximately 08:51 UTC:

```sh
ssh fans-mfa@login-hecate 'bash /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/runtime-runs/model-R_fixed_vs_B_fixed-decode-r1-B_fixed-jitverbose-drain0935/packet/clocked-launch.sh'
```

Packet SHA256 is
`4416aa79d26fe5cc9657ec6fcfec4c2682ea8d7f681f6acaea78194736b5ab54`.
It uses the existing Rubin base image, candidate MSA `cd20e206` and vLLM
`19afac40`, TP2/ICP2, real Eagle3 k=3, and hecate0011 GPU0/1. It starts a fresh
server for all three cells without restarting between them. Every cell uses
ISL65536/OSL1024, 2C warmup requests and 10C measured requests in drained
cohorts, with offered rate infinity subject to C8/C12/C16. This is one
independent candidate boot, not ten repetitions. It preserves the original
datasets, algorithm flags and full counts. The three clients retain 300-second
caps, and the workload has a 1970-second cap. Only the output tag and separate
profile change: drain 09:35:19 UTC, profile SHA256
`41864ad79be128c112ff1b8add00270b2cc12bd926a38d51ccdd833d95962924`.
The full 480-second cleanup guard and owned-child/clock-reset behavior remain;
the parent allocation end time is unchanged. B's cache hold is released only
after its successful child and clock cleanup.

Three module pairs and pair3 B have completed all 216 cells/arm; pair3 R
started at 08:44:50 UTC. Raw timing outliers remain included. Future-only
kernel tails are staged for an exact completed-arm handoff after root reaps
the original controller; none has been activated at this snapshot.

Source, exact base and review-notes refs are pushed and remotely verified in
all four fans-nv GitHub and fans GitLab repositories. Notes commit is
`1235abd7d4635914d0146b1c9e140264b5593673`. Validation remains incomplete.

Historical execution update, 2026-10-09 01:05 PDT (08:05 UTC):

- The first complete R_fixed/B_fixed module performance pair produced all
  216 timing cells per arm. R took 615.02 seconds and B took 608.27 seconds,
  including fresh model startup. The next B process also completed all 216
  cells in 601.52 seconds; its paired R process started at 08:01:19 UTC.
  Completed runs passed continuous 4752 MHz telemetry and clock reset checks.
  The frozen six-pair analysis is incomplete. Rare large timing intervals
  remain in the arithmetic-mean estimator; no outliers have been removed.
- Both corrected arms passed the independent loaded first/later producer
  checks, including changed-query graph replay against separately computed
  eager results. Five ICP-enabled model boots passed: candidate no-spec FULL,
  candidate no-spec eager, candidate Eagle3, reference no-spec FULL, and
  reference Eagle3. The separate reference Eagle3 autotune-ON policy boot
  subsequently passed all 11 requests and both producer RPCs, stable cache
  ownership, actual graph routing, real speculative acceptance, and cleanup.
  This remains a scoped correctness result, not performance qualification.
- Historical H completed the first full GSM8K run: 1260/1319 correct
  (95.5269%), with 16 disjoint warmups, zero request errors or invalid answers,
  and 70 answers reaching the inherited 512-token cap. Root independently
  rescored every response against the pinned dataset and scorer. This is the
  inherited zero-shot adaptive-chat recipe, not a different few-shot recipe.
  Candidate B accuracy run 1 started on hecate0011 at approximately 08:03 UTC;
  two independent paired accuracy rounds remain required.
- The ordinary ICP-off whole-model path still fails with a CUDA launch error.
  The separate ordinary MSA kernel comparison below passes its independent
  CPU oracle, but does not qualify the failing whole-model route.
- Corrected source/base review refs have been pushed to fans-nv/MSA and
  fans-nv/vllm. GitLab publication and exact remote verification are tracked
  by the source agent. These refs are available for review; validation is not
  complete and they are not described as ready to merge.

The root-owned module controller is `kernels/continue_decode_r1.py`, bound to
the hashed plan `kernels/continue-decode-r1.json`. Its remote output is
`results/module-continuation-r1`. It verifies each complete receipt and clock
cleanup before advancing, waits for explicit per-arm cache holds, and has no
allocation cancellation code. Runtime owns the B_fixed hold during two fresh
candidate accuracy boots; the module controller can run the adjacent R arms
in parallel. Future timeout-only amendments preserve workload counts and
independent repetitions; an expired budget records incomplete work.

Current candidate accuracy launch is:

```sh
ssh fans-mfa@login-hecate 'bash /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/runtime-runs/accuracy-H_vs_B_fixed-r1-B_fixed-bound-r4/packet/clocked-launch.sh'
```

The bound packet SHA256 is
`ae109f5f39387e6672dc406cc1093c5d294ad448b03414631e19a1aa5e16d5aa`.
It uses allocation 723744, hecate0011 GPU0/1, the existing Rubin base image,
candidate MSA `cd20e20` and vLLM `19afac40`, TP2/ICP2 and real Eagle3 k=3.
It runs 16 warmups then 1319 scored requests at C16, temperature zero,
top-p 1, normal EOS and maximum OSL512; ISLs are the exact pinned GSM8K chat
prompts. Request rate is bounded by C16 with no external rate limit. This is
one independent boot, not 1319 independent model repetitions. Client timeout
is 1920 seconds and clock-wrapper timeout 4330 seconds, with the existing
480-second drain guard. The wrapper records clocks and stops only its owned
workload processes/steps before resetting its own clock locks. The parent
allocation is preserved.

The first B model boot exposed a missing model-local dense FP8 override.
Versioned B2 (`19afac40273aa84c2693ca1083995c4156aca4e2`) and R2
(`dda98ad2fbd6c6485114e82617c784cac26f36d3`) restore the historical policy:
sparse NVFP4 and a copied FP8 cache config only for dense attention. Both
passed five focused CPU tests. Their compiled inputs and same-arm native
binaries are unchanged; source, native and test receipts are recorded in
`source/dense-fp8/manifest.json`. Original B/R snapshots remain immutable.
Kernel-only checks still identify the exact original snapshot they execute.

The full module matrix then exposed a real inherited decode scorer race at
C12/Q1/150K. Both original B and R failed; the same scorer bytes also occur in
K. A minimal async-proxy fence before releasing a consumed TMA slot corrects
the ordering while retaining four pipeline stages. Original sources and
failures remain unchanged. New controls are explicitly named B_fixed MSA
`cd20e206a3373e8d258da8cd403b4b3399b1f714`, R_fixed MSA
`fa64b032b459b6df8b345836694f6a7401ffec22`, and K_fixed ICP
`26708f883b0a14792b0c1440931789aaa5e91c1a`. These are not relabelled unchanged
references. See `kernels/SCORER-FIX-AMENDMENT.md` and
`source/decode-fence/manifest.json` for the evidence and exact source binding.

At 23:20 PDT, both B_fixed and R_fixed passed all 36 primary decode cases on
both ranks, each with three eager checks and three changed-query graph
replays, exact outputs and cache canaries, and runtime JIT forbidden. Each
arm separately passed its five-component/46-artifact prewarm and strict fresh
reload plus the new focused eager/graph regressions. K_fixed also passed its
own five-component prewarm and strict reload after recovering the exact
historical headers from the immutable Rubin base image. Loaded-producer
validation subsequently passed as recorded above; complete paired performance
remains required.

Ordinary MSA compatibility has separate fresh native-cache provenance for
the candidate and upstream dev `968061ab16cb14991a5c7d4c0e8eabf531be0dcf`.
Q/K/V and all five tested outputs are bitwise equal across those builds. Both
pass the independent CPU FP32 oracle at the original 0.9999 threshold. The
original GPU SDPA reference produces nonfinite values in both runs and its
failed assertion is preserved; this is not labelled a broad pytest pass.
See `results/ordinary-msa-isolated-comparison-r1.json`.

The corrected B_fixed model completed no-speculation FULL/eager and real
Eagle3 boots, both loaded writer/indexer checks and real profiling
release/rebind/final close. Fixed-source greedy comparisons are not uniformly
token-identical; the records distinguish close-logit ties and observed
batching differences from unexplained discrepancies. Complete task accuracy,
broader model-level lifecycle coverage and performance remain open. No
performance retention claim is admitted. Updated per-boot evidence is in
`runtime-ops/ADMISSION-R4.md` and `runtime-runs/`.

At 21:33 PDT, allocation 723744 was PENDING with estimated start 21:54 PDT.
No predicted hostname is a GPU execution binding. Read the live job at startup,
then inventory each assigned node. The exact first step is:

```sh
srun --jobid=723744 --account=coreai_comparch_inferencex --partition=batch-spx,batch-xdr --job-name=icp723744-inventory --overlap --exact --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=1 --mem=1G --time=00:02:00 --output=/lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/logs/inventory-%N.log --error=/lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/logs/inventory-%N.err /usr/bin/python3 /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/ops/node_inventory.py --expected-job 723744 --output-root /lustre/fsw/coreai_comparch_inferencex/fans/campaigns/icp-public-723744/inventory
```

This host-only step reads GPU configuration and process state. It does not load
a model, change clocks, benchmark or cancel anything. It uses no container.
The subsequent allocation resolver records the actual partition, nodes, UUIDs,
expiry and drain deadline; module work owns the first node and E2E owns the
second. The protocol forbids other campaign work on an entire measured node,
including builds and GPU2/3 workloads. Unrelated
allocations 721691/721713 are not used.

Preparation and run ownership:

- Kernel agent: local indexer, real transport and full production-module
  correctness, then complete R/B and K/B performance comparisons. See
  `kernels/NEXT-STEPS.md` for exact payloads and frozen counts.
- Runtime agent: model load and token correctness, actual loaded first/later
  producers, graph/lifecycle/DMA checks, then H/B model performance and R/B
  sentinels. See the runtime preparation packet for exact boot commands.
- Build agent: exact K baseline staging/build preparation and independent
  source/image/native identity checks. Existing B/R native binaries are reused.
- Root: node and clock admission, scheduling, timeout/drain enforcement,
  telemetry, collection and interpretation of results.

Source and model identities remain those in the frozen 721392 protocol. The
rebased candidate is MSA d19d2c79228f02f2238eb75d0a665c6609cc5c63 and vLLM
2916ec5151e2c74e695ab0917a6a8f92174de5c7. B/R/K use the existing Rubin base image
SHA256 5bdfa010c240579625c4e21ffdb95e41132cd18ae2b1223d717b3eb06b92f0d3.
Historical H remains a separately identified immutable reference image; its
environment/backend differences are recorded and never treated as identical.

The authoritative scope remains `plans/hecate-721392/PERFORMANCE-MATRIX.json`:
36 decode, 18 prefill, six mixed module shapes, plus 18 untimed guards. Module
decode uses 20 warmup calls and 100 samples of ten complete calls; prefill/mixed
uses ten warmups and 50 one-call samples. There are six fresh paired process
rounds, with all original boundaries and eager/graph modes. Module request rate
and OSL do not apply; exact batch, query-row and history vectors are recorded.

Whole-model decode is first, C8/C12/C16: 2K/16K/64K/150K and frozen ragged
histories, OSL4096 for 2K/ragged and1024 otherwise, infinite offered request rate
subject to the fixed concurrency, 2C warmup requests and10C measured requests.
Prefill uses 16K/64K/150K with OSL1, 3C warmup and12C measured. Mixed uses a3:1
decode/prefill cohort, 2C warmup and10C measured. Each phase group has six fresh
paired server rounds; requests within a round are turnovers, not independent
repetitions. Complete counts and exact mixed/ragged vectors remain in the
frozen matrix and workload-count files. No server is reused across arms or
independent rounds. The full campaign is not assumed to fit one five-hour job;
record incomplete cells explicitly and preserve decode priority.

Material execution changes from the former unexecuted packet are documented:
R/B decode can start before K/B while K is built; no GPU-GRES flags on Hecate;
new allocation, node identities and output roots; larger correctness step
budget to cover the documented worst case; separate complete-call event timing
with untimed TP ordering for isolated ACK-free transport. Shapes, warmup counts,
sample counts and six independent pairs are unchanged.

Performance requires4752MHz memory clocks and fresh sampled telemetry on both
GPUs. `ops/run_with_clocks_r3.py` uses the qualified host command
`sudo -n /usr/bin/nvidia-smi -lmc 4752,4752 -i INDEX`, after checking the UUID
mapping. The wrapper runs on the host and supervises the owned container step.
A failed control does not admit timing, but does not prevent
untimed correctness. Its cleanup resets only locks it successfully set to
driver defaults; nvidia-smi cannot recover an unknown earlier lock policy.
That cleanup behavior differs from claiming exact restoration of an unknown
policy and is explicit in every result. It records baseline XML, commands,
200ms sampled telemetry, owned-child timeout and cleanup outcomes. No parent
allocation action exists in this helper. Fresh node/process inventory and
qualification must precede its use.

Old evidence and failed attempts remain unchanged under icp-public-721392.
New harnesses, packets, logs and results live under icp-public-723744. Every
resolved timed launch must record concrete nodes/UUIDs, inner and outer argv,
image/model/source hashes, protocol, output directory, timeout and cleanup.
