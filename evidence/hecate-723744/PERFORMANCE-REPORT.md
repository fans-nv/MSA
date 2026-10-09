Kernel and communication-path performance have been measured. Performance retention is **not established**: the repeated complete-indexer measurements contain persistent adverse cells, only five of six required pairs completed, and the whole-model pure-decode windows failed their declared scheduler check. All original samples, outliers and failed checks are retained.

These results use Hecate allocation **723744**, the existing Rubin base image, and continuously verified **4752 MHz memory clocks**. Primary module timing used hecate0010 GPU0/1; kernel profiles and the model sentinel used hecate0011 GPU0/1, after other work on that node had drained. Both are TP2 within one node. The two allocated nodes were independent validation tracks, not one cross-node model. Source/image/packet identities and commands are in [the execution record](EXECUTION.md) and the linked receipts.

`B_fixed` is the public candidate, MSA `cd20e206a3373e8d258da8cd403b4b3399b1f714` with vLLM `19afac40273aa84c2693ca1083995c4156aca4e2`; its MSA base is the requested upstream **dev** revision `968061ab16cb14991a5c7d4c0e8eabf531be0dcf`. `R_fixed` is the corrected consolidation reference and `K_fixed` the corrected original icp-kernels control. All three include the same minimal inherited scorer-race correction; they are not relabelled unchanged historical baselines. The [source amendment](kernels/SCORER-FIX-AMENDMENT.md) preserves original failures and exact identities.

The complete indexer includes scoring, local selection/publication, peer exchange and final merge/polling. The measured matrix also includes actual checkpoint-loaded QKV production and the native cache writer, separately for first- and later-layer ownership. Five independent R/B pairs completed all **216 cells per arm**: 36 decode shapes, three boundaries, and eager/graph modes. Shapes cover C8/C12/C16, Q1/Q4, 2K/16K/64K/150K history, Q2/Q3 at 64K, and ragged batches. Each cell retains 100 samples of ten complete calls, taking the slower rank per call before arithmetic averaging.

| Complete core, graph mode, 64K history | Reference µs | Candidate µs | Candidate latency change |
| --- | ---: | ---: | ---: |
| C8, Q1 | 22.90 | 25.25 | +10.3% |
| C8, Q4 | 28.90 | 26.73 | −7.5% |
| C12, Q1 | 27.55 | 29.38 | +6.7% |
| C12, Q4 | 31.43 | 30.94 | −1.6% |
| C16, Q1 | 31.75 | 36.11 | +13.7% |
| C16, Q4 | 33.97 | 35.94 | +5.8% |

C16/Q1/64K was slower in every completed pair. The largest persistent slowdown across all 216 cells was **C12/Q1/150K, later-layer producer plus indexer, graph mode: 39.91 → 46.51 µs (+16.5%)**. Four cells were slower in every pair, three faster in every pair, and 209 had mixed directions. These are descriptive results, not the incomplete six-pair statistical decision. See [all five-pair observations](kernels/DECODE-FIVE-PAIR-OBSERVATIONS.md), [full analysis](kernels/decode-analysis-after-pair4.json), and the [pair-by-pair plot](analysis/module-core-five-pairs-64k.png).

The primary harness performs 20 uncaptured complete-call warmups, followed by graph capture and **one** validation replay for graph cases; it does not perform 20 graph replay warmups. Both arms use this same procedure. Untimed TP barriers separate complete-call intervals to order the ACK-free protocol. This warmup qualification and possible effects of inter-call idle gaps remain explicit; neither is a reason to discard the slower measurements or substitute medians.

Separate CUPTI profiles measure the actual scorer, fused selection/publication, and merge/poll kernels. K/R/B each completed 12 shapes (C8/C12/C16 × Q1/Q4 × 64K/150K), eager and graph, with 20 warmups of the selected callable and ten profiled calls. Across all three arms, independently reviewed evidence contains **144 traces and 4,320 constituent GPU kernels**. The stages overlap through PDL, so their durations cannot be summed as serial latency.

| C16/Q1/64K graph diagnostic | R_fixed | B_fixed |
| --- | ---: | ---: |
| Complete GPU span, slower rank | 22.157 µs | 22.944 µs |
| Scorer, rank0 / rank1 means | 12.13 / 12.20 µs | 12.07 / 12.12 µs |
| Fused selection/publication, rank0 / rank1 | 7.789 / 7.789 µs | 7.722 / 7.744 µs |
| Merge/poll, rank0 / rank1 | 8.502 / 6.691 µs | 7.514 / 8.535 µs |

K_fixed's same diagnostic span was 21.604 µs. These short profiles on a different node/process do not explain or override the primary +13.7% result. The 150K profiles also retain large outliers: B C12/Q1 eager averaged 61.028 µs versus R 26.029 µs; K C16/Q4 graph averaged 54.417 µs versus B 31.389 µs. Normalized cross-rank timestamps show these outliers coincide with increased CPU/CUDA-launch/GPU-start skew, while the early rank's merge overlaps the peer's not-yet-completed publication. This is temporal alignment, not a causal attribution to host scheduling or the fabric. See [all K/R/B stage measurements](kernels/CORE-CUPTI-K-R-B-COMPARISON.md), [independent trace review](analysis/cupti-stages-independent-review-r2.json), and [rank-skew analysis](analysis/cupti-rank-skew-r1/REPORT.md).

Communication here is **same-node NVLink peer publication through CUDA symmetric memory**, on the selected GPUs' NV36 connection. Selection is fused with publication; the receiver polls generation tags and merges candidates. The source-derived tagged remote payload per rank per call is 4/16 KiB for C8 Q1/Q4, 8/24 KiB for C12, and 8/32 KiB for C16; both directions double those values. These are payload counts, not measured link utilization. The implementation option named `merge=network` refers to the local sorting/shuffle network. **Standalone fabric bandwidth, inter-node NIC latency/bandwidth, and isolated spin time have not been measured.** Dividing these bytes by a complete kernel span would not yield a valid link-bandwidth result. Prefill/mixed carrier pack/NCCL paths have correctness coverage but no primary performance results.

An additional [binary audit](analysis/device-text-comparison-r1.json) found the eight R/B decode artifacts byte-identical and all 58 compared embedded native GPU text sections byte-identical. That narrows instruction-code differences in this comparison; it does not establish equivalent runtime dispatch, data, addresses, synchronization, clocks or latency.

The whole-model sentinel completed one matched R/B pair at **ISL65536 / OSL1024**, C8/C12/C16, TP2/ICP2 and real Eagle3 k=3. Each fresh server handled all three cells, with 2C warmup and 10C measured requests per cell at offered rate infinity subject to C. Each arm completed 72 warmup and 360 measured requests, with matched input identities, zero request errors and zero preemptions. Ten prompt turnovers are one measured run, not ten independent runs.

| Concurrency | R whole-request output tok/s | B whole-request output tok/s | Candidate change |
| --- | ---: | ---: | ---: |
| 8 | 739.19 | 741.33 | +0.29% |
| 12 | 798.55 | 796.23 | −0.29% |
| 16 | 833.26 | 825.60 | −0.92% |

Those descriptive rates include prefill, decode and completion tails. **Every original pure-decode scheduler window failed** because its terminal overlapping CPU envelopes contained fewer than C requests. Thus these three sentinel cells have **1/6 collected pairs and 0 accepted pairs**. The separately reported post-hoc window analysis does not replace the frozen metric. Observer overhead remains unqualified. Sentinel prefill/mixed and all 30 historical-image/candidate primary model cells remain unmeasured. See [the whole-model report](analysis/model-decode-one-pair-r1/REPORT.md).

The outstanding performance work is a fully declared completion/rerun of the repeated module protocol with the warmup issue addressed, investigation of the persistent adverse cells, primary K/B and prefill/mixed timing, and whole-model decode followed by prefill/mixed with valid scheduling windows and the required independent pairs. Inter-node testing is a separate missing measurement if that is part of the deployment topology. The ordinary ICP-off model failure also remains a compatibility blocker. The [validation status](VALIDATION-STATUS.md) and [kernel evidence report](kernels/KERNEL-VALIDATION-RESULTS.md) distinguish these gaps from completed correctness and accuracy work.
