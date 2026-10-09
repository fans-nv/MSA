Five of six independent pairs are complete. These are descriptive arithmetic means, with every raw outlier retained; the predeclared six-pair decision remains incomplete.

| Case | Round0 R/B µs | Round1 R/B µs | Round2 R/B µs | Round3 R/B µs | Round4 R/B µs | Mean R/B µs | Mean B−R µs | B/R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b8-q1 | 23.20/19.11 | 21.87/32.10 | 23.85/24.05 | 19.46/23.26 | 26.10/27.71 | 22.90/25.25 | +2.35 | 1.103 |
| b8-q4 | 25.95/29.14 | 38.48/23.94 | 19.74/24.70 | 36.37/25.94 | 23.93/29.91 | 28.90/26.73 | -2.17 | 0.925 |
| b12-q1 | 31.52/34.53 | 23.56/28.08 | 20.24/28.82 | 28.97/26.77 | 33.43/28.71 | 27.55/29.38 | +1.84 | 1.067 |
| b12-q4 | 32.13/25.41 | 24.61/29.26 | 36.62/31.28 | 25.96/30.47 | 37.84/38.30 | 31.43/30.94 | -0.49 | 0.984 |
| b16-q1 | 28.31/34.05 | 27.40/30.04 | 33.11/40.69 | 34.06/38.51 | 35.89/37.25 | 31.75/36.11 | +4.36 | 1.137 |
| b16-q4 | 30.78/37.11 | 32.74/37.45 | 33.04/31.56 | 33.03/39.88 | 40.26/33.71 | 33.97/35.94 | +1.97 | 1.058 |

Direction counts across all five rounds:

- core/eager: {'mixed': 36} (36 cells)
- core/graph: {'mixed': 35, 'slower': 1} (36 cells)
- producer-first/eager: {'mixed': 35, 'faster': 1} (36 cells)
- producer-first/graph: {'mixed': 33, 'faster': 2, 'slower': 1} (36 cells)
- producer-later/eager: {'mixed': 35, 'slower': 1} (36 cells)
- producer-later/graph: {'mixed': 35, 'slower': 1} (36 cells)

Worst persistent slowdowns across all216 cells, sorted by ratio of arithmetic means:

- indexer-decode-b12-q1-l150000 producer-later/graph: R39.91µs, B46.51µs, Δ+6.60µs, ratio1.165.
- indexer-decode-b16-q1-l65536 core/graph: R31.75µs, B36.11µs, Δ+4.36µs, ratio1.137.
- indexer-decode-b16-q4-l2048 producer-first/graph: R34.13µs, B35.81µs, Δ+1.68µs, ratio1.049.
- indexer-decode-b12-q1-l65536 producer-later/eager: R179.91µs, B185.56µs, Δ+5.65µs, ratio1.031.

Warmup qualification: the frozen primary harness performs 20 uncaptured complete calls, then graph capture and one untimed validation replay. It does not perform 20 graph replay warmups. Both arms use this same procedure; all original measured samples remain unchanged.
