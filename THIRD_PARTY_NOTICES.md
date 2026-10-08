# Source attribution and license status

Existing MSA files retain their MIT notices and the repository's original
`LICENSE`. This integration does not relicense imported files. Per-file notices
and original rights remain in force.

| Material | Existing notice |
| --- | --- |
| Existing `vllm-project/MSA` dev files at `ed4e40efcb5895aba1a3554d62cd1dcaf77920cb`, extended with ICP scoring from the frozen implementation | Existing per-file notices; repository `LICENSE` and `NOTICE` |
| Decode scan/helpers with vLLM headers, and local planning/candidate transport extracted from vLLM `6795fa319cf6145b16a5818bdda765cf888df847` | Apache-2.0; `licenses/Apache-2.0.txt`; original SPDX/copyright lines retained |
| NVIDIA FMHA/CUTLASS-derived headers | Their existing BSD-3-Clause notices; `licenses/BSD-3-Clause-NVIDIA.txt` |
| Additional ICP selector/merge/transport and host sources imported from `icp-kernels` `4232bce` without a per-file license grant | The donor package was marked Proprietary. No MIT or other grant is inferred here; publication requires the rights holder's clearance. |

The original MSA project metadata remains MIT; it is not a new license grant for
the separately identified imported ICP material. This candidate is prepared for
review and must not be publicly distributed until those rights are resolved.

`tools/stage_cutlass.py` stages selected CUTLASS headers with their actual
`LICENSE*`/`NOTICE*` files and a SHA256 inventory in `cutlass/SOURCE.json`.
The selected dev submodule is CUTLASS `098de2a652cf8f00fd70b2df54051c7eccbb855a`.
Its root `LICENSE.txt` is included with the staged headers and preserved as
`licenses/BSD-3-Clause-CUTLASS.txt`.
Dependencies are installed as separate distributions; their sources and licenses
are not replaced by this package.
