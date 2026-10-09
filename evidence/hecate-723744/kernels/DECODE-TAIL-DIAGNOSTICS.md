The C16/Q1/64K core-graph candidate mean is higher in all five completed pairs. Sample medians and tails below are diagnostics; the predeclared decision still uses complete arithmetic means and requires six pairs. No sample or call was removed.

Each sample averages 10 complete-call intervals after taking the slower rank for each call. There are 100 samples per process.

| Pair | R mean µs | B mean µs | R sample median µs | B sample median µs | R max sample µs | B max sample µs |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 28.31 | 34.05 | 25.12 | 25.18 | 330.72 | 512.43 |
| 1 | 27.40 | 30.04 | 23.76 | 25.11 | 261.55 | 478.10 |
| 2 | 33.11 | 40.69 | 23.50 | 23.51 | 915.38 | 1707.90 |
| 3 | 34.06 | 38.51 | 25.07 | 25.25 | 585.94 | 623.52 |
| 4 | 35.89 | 37.25 | 23.43 | 25.28 | 759.53 | 652.76 |

The separate CUPTI run is shorter and runs on node0011. Its similar scorer/selector spans do not establish the cause of the primary mean difference. Host submission skew, generation polling and rare complete-call stalls remain possible contributors; only the observed intervals are reported.
