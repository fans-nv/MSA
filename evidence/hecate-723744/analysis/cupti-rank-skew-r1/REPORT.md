This offline diagnostic retains all ten calls for each arm’s largest 150K mean-span outlier, its matched controls, and the R C8/Q1 graph counterpart. It normalizes each event as baseTimeNanoseconds + ts × 1,000 using decimal arithmetic. CUDA launch-correlation IDs connect the GPU kernels to their actual host launch calls.

| Arm / case / mode | Mean span µs | CPU entry skew µs | CUDA score-launch skew µs | GPU score-start skew µs | Score-start skew / span | Merge before peer publish starts / span | Merge before peer publish finishes / span |
|---|---:|---:|---:|---:|---:|---:|---:|
| R_fixed b8-q1-l150000 eager | 45.677 | 14.864 | 25.051 | 26.701 | 58.5% | 52.8% | 66.9% |
| B_fixed b8-q1-l150000 eager | 21.178 | 1.503 | 2.081 | 2.215 | 10.5% | 0.0% | 29.2% |
| K_fixed b8-q1-l150000 eager | 20.819 | 1.510 | 2.037 | 2.158 | 10.4% | 0.0% | 27.7% |
| R_fixed b8-q1-l150000 graph | 43.268 | 20.873 | 23.410 | 25.464 | 58.9% | 50.7% | 67.4% |
| B_fixed b8-q1-l150000 graph | 19.776 | 1.333 | 1.399 | 1.641 | 8.3% | 0.0% | 28.2% |
| K_fixed b8-q1-l150000 graph | 19.927 | 1.997 | 1.830 | 2.041 | 10.2% | 1.2% | 28.9% |
| R_fixed b12-q1-l150000 eager | 26.029 | 2.292 | 3.107 | 3.274 | 12.6% | 1.5% | 34.3% |
| B_fixed b12-q1-l150000 eager | 61.028 | 24.470 | 35.882 | 38.217 | 62.6% | 50.5% | 63.8% |
| K_fixed b12-q1-l150000 eager | 24.628 | 1.048 | 1.451 | 1.467 | 6.0% | 0.0% | 29.1% |
| R_fixed b16-q4-l150000 graph | 32.276 | 1.568 | 1.541 | 2.170 | 6.7% | 0.0% | 20.4% |
| B_fixed b16-q4-l150000 graph | 31.389 | 1.168 | 1.271 | 1.354 | 4.3% | 0.0% | 17.9% |
| K_fixed b16-q4-l150000 graph | 54.417 | 24.307 | 24.158 | 24.363 | 44.8% | 40.7% | 52.6% |

Ratios divide sums across all ten calls; no call or outlier was removed. The two merge columns measure the temporal overlap of the slower-span rank’s merge kernel with the period before its peer’s publish kernel starts/finishes. Publication occurs during that kernel, so these columns are not lower/upper bounds on pure spin time or link latency. Local processing, PDL dependencies and generation checks remain inside the merge.

CPU or GPU start-skew ratios are descriptive and do not estimate a percentage causally explained by host scheduling. The profiled maximum per-rank span omits some cross-rank launch skew; the full aligned GPU envelope is recorded separately in the JSON. Ten profiled calls from one process per arm cannot replace the primary paired protocol or establish a bandwidth/performance-retention claim.
