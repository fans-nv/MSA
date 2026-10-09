The 65,536-input/1,024-output sentinel comparison completed one R_fixed→B_fixed pair on hecate0011 GPU 0/1 at 4752 MHz. Each arm completed 72 warmup and 360 measured requests across C8/C12/C16 with zero request errors, zero preemptions and verified child/clock cleanup. The same 432 input/payload identities were independently checked across arms.

Whole-request measurements include prefill, decode and request tails. They are descriptive results from one pair; observer overhead and the remaining five paired repetitions are unqualified.

| Concurrency | R output tok/s | B output tok/s | B vs R | R TTFT p50 ms | B TTFT p50 ms | R request TPOT p50 ms | B request TPOT p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 739.19 | 741.33 | +0.29% | 4547.59 | 4541.39 | 5.924 | 5.927 |
| 12 | 798.55 | 796.23 | -0.29% | 6533.29 | 6528.61 | 8.252 | 8.055 |
| 16 | 833.26 | 825.60 | -0.92% | 8496.90 | 8511.78 | 9.913 | 10.210 |

All 30 original scheduler-window checks in each arm fail because the final 1–2 overlapping CPU envelopes have a reduced request batch. All client windows passed the 0.25 s/256 tokens-per-request thresholds; these facts do not override the scheduler failure. The frozen analyzer emits no qualified pure-decode rate.

The following separately named post-hoc diagnostic ends each original client window at its first terminal reduced-batch CPU scheduling timestamp. All 30 cohorts in each arm remain included and pass the diagnostic checks. The boundary was selected after inspecting the failure, uses CPU/client timestamps, and is not a replacement primary metric.

| Concurrency | R diagnostic tok/s | B diagnostic tok/s | B vs R | R diagnostic TPOT ms | B diagnostic TPOT ms |
|---|---:|---:|---:|---:|---:|
| 8 | 2739.99 | 2751.72 | +0.43% | 2.920 | 2.907 |
| 12 | 3575.46 | 3541.64 | -0.95% | 3.356 | 3.388 |
| 16 | 4347.40 | 4247.49 | -2.30% | 3.680 | 3.767 |

Both actual TP workers enabled verbose JIT monitoring. The three measured populations in each arm contain no overlapping event from the monitored APIs; this is not universal proof of no compilation. Clock telemetry and resets passed for both arms.

Coverage remains incomplete: the three decode sentinel cells have 1/6 collected pairs and 0 accepted pairs; the two sentinel prefill/mixed cells have 0/6. All 30 H/B primary cells have 0/6. The full matrix, exact packet/source/image hashes, request-file hashes and unreduced counts are preserved in result.json. No confidence interval, equivalence or performance-retention claim is made.
