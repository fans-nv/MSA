All three corrected arms passed the separate CUPTI diagnostic at 4,752 MHz: 12 shapes, two modes, both ranks, 20 warmups of the selected mode, and 10 profiled calls. Each arm retains 48 traces, 480 call ranges, and 1,440 GPU kernel events.

These are diagnostic means on node0011. They do not replace the primary paired means on node0010 or qualify performance retention.

The 150K captures include substantial outliers: R C8/Q1 eager 45.677 µs, B C12/Q1 eager 61.028 µs, and K C16/Q4 graph 54.417 µs. Their mean cross-rank GPU score-start skews were 26.701, 38.217 and 24.363 µs, respectively. [Aligned rank timelines and matched controls](../analysis/cupti-rank-skew-r1/REPORT.md) retain every call and show merge overlap before the peer publishes. These are observed temporal relationships, not causal percentages or link-speed measurements.

| Case | Mode | K span µs | R span µs | B span µs | B/R | B/K |
| --- | --- | --- | --- | --- | --- | --- |
| b12-q1-l150000 | eager | 24.628 | 26.029 | 61.028 | 2.345 | 2.478 |
| b12-q1-l150000 | graph | 23.386 | 23.876 | 25.588 | 1.072 | 1.094 |
| b12-q1-l65536 | eager | 22.112 | 18.103 | 20.490 | 1.132 | 0.927 |
| b12-q1-l65536 | graph | 21.220 | 17.386 | 16.855 | 0.969 | 0.794 |
| b12-q4-l150000 | eager | 28.608 | 31.667 | 29.802 | 0.941 | 1.042 |
| b12-q4-l150000 | graph | 28.381 | 28.804 | 28.125 | 0.976 | 0.991 |
| b12-q4-l65536 | eager | 23.331 | 21.690 | 22.074 | 1.018 | 0.946 |
| b12-q4-l65536 | graph | 22.275 | 20.723 | 19.950 | 0.963 | 0.896 |
| b16-q1-l150000 | eager | 32.007 | 31.623 | 31.118 | 0.984 | 0.972 |
| b16-q1-l150000 | graph | 29.757 | 29.876 | 30.285 | 1.014 | 1.018 |
| b16-q1-l65536 | eager | 22.611 | 23.159 | 22.864 | 0.987 | 1.011 |
| b16-q1-l65536 | graph | 21.604 | 22.157 | 22.944 | 1.036 | 1.062 |
| b16-q4-l150000 | eager | 43.569 | 32.074 | 31.956 | 0.996 | 0.733 |
| b16-q4-l150000 | graph | 54.417 | 32.276 | 31.390 | 0.973 | 0.577 |
| b16-q4-l65536 | eager | 23.213 | 24.064 | 23.501 | 0.977 | 1.012 |
| b16-q4-l65536 | graph | 23.757 | 21.927 | 22.042 | 1.005 | 0.928 |
| b8-q1-l150000 | eager | 20.819 | 45.677 | 21.178 | 0.464 | 1.017 |
| b8-q1-l150000 | graph | 19.927 | 43.268 | 19.776 | 0.457 | 0.992 |
| b8-q1-l65536 | eager | 20.278 | 20.067 | 21.168 | 1.055 | 1.044 |
| b8-q1-l65536 | graph | 15.965 | 14.480 | 13.965 | 0.964 | 0.875 |
| b8-q4-l150000 | eager | 23.376 | 24.938 | 23.130 | 0.928 | 0.989 |
| b8-q4-l150000 | graph | 21.139 | 23.328 | 22.170 | 0.950 | 1.049 |
| b8-q4-l65536 | eager | 18.036 | 18.279 | 18.266 | 0.999 | 1.013 |
| b8-q4-l65536 | graph | 18.186 | 15.117 | 15.155 | 1.003 | 0.833 |

PDL overlaps stages, so scorer, select/publish and merge durations must not be added as serial latency. Merge includes polling and reduction. The JSON and CSV artifacts retain both ranks, every profiled call, exact stage names and the source-derived padded publication bytes. No link-bandwidth or inter-node result is inferred.
