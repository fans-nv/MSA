# Public MiniMax-M3 writer: additive ICP extension

The compatibility baseline is public vLLM `242e4213fc9845ff6fe607af1aee626fd8acc990`.
The qualified ICP producer comes from `53e229bc28df277f0906b94a082293c60a2bc2e7`.

Keep the public 25-argument schema prefix and ordinary arithmetic, including
model-dtype rounding of index Q, unchanged. Append optional ICP arguments only.
The Python wrapper keeps ordinary calls compatible with the original compiled
extension and expands the preallocated ABI-3 device plan only for ICP calls.

Both paths use public alternating head slots: K0, V0, K1, V1. With P128/Hkv2,
each slot occupies 9,216 bytes; data occupies its first 8,192 bytes. The compound
page stride is 45,056 bytes, with index storage following the 36,864 main bytes.
The K/V data offsets are 0/9,216 and their scale offsets are 8,192/17,408, with
head stride 18,432. K scales are linear and V scales use the existing token-quad
swizzle. There is no intermediate repack or region-major reinterpretation.

One writer body serves ordinary and ICP entry points. Compile-time ICP choices
retain the qualified reciprocal-multiply NVFP4 arithmetic, direct FP32-to-FP8
index Q, packed norm-weight loads, rank-owned R64 index writes, 256-thread/eight
block launch bound, and the flat-row specialization at N >= 1024. Main Q can be
emitted in both model dtype and FP8. K/V global calibration tensors are consumed
by the writer and reader; Q scale is one. Live metadata and device-plan CTAs are
in the same non-PDL producer launch, including a zero-token producer with bound
caches and retained metadata. Ordinary step-metadata machinery is not imported.

Test design: the module normalizes/rotates fused projections, writes main and
owned index cache bytes, and publishes live plan state. Its contract is the
public ordinary prefix plus opt-in TP2/Q32/KV2/I4/P128/R64/ABI3. The regressions
to prevent are ordinary numerical/layout drift, an owner writing another
rank's index fragment, compound-page corruption, stale plan bookkeeping after
a failed launch, and changed public call compatibility. Retain the independent
public CUDA oracles; reuse the CPU writer-contract suite for schema/ordering and
add separate ICP numerical/layout coverage in the nearby kernel suite. CPU
checks and native translation-unit compilation are local gates; target GPU
numerical, graph, end-to-end, and performance qualification remain required.

## Implemented and locally checked

The eight owned writer paths are staged. The schema has 55 arguments, with its
original 25-argument prefix preserved verbatim. Startup can distinguish this
public head-slot binary from the older region-major binary by the public
`kv_k_scale`/`kv_v_scale` names together with the ICP tail and plan ABI 3.
ICP admission also checks the alignment required by packed accesses: 8 bytes
for QKV, all four norm weights and model-dtype Q; 4 bytes for FP8 gathers; and
2 bytes for main-cache base/page stride. The index-fragment checks already
require 4-byte alignment. RoPE table accesses remain scalar.

Final local evidence:

- `logs/writer-cpu-r3`: 17 passed, 134 GPU cases skipped; Ruff lint and Python
  formatting pass.
- `logs/writer-format21-r1`: the repository-pinned clang-format 21.1.2 was
  applied to all four C++ files through an isolated `/tmp` uv tool environment;
  its configured `--style=file --dry-run --Werror` check passes. Only comment
  indentation and declaration wrapping changed. The saved comparison preserves
  every literal, comment, identifier/operator token and logical preprocessor
  directive, and binds the before-hashes to the native and CPU checks below.
  The Python/tests remain byte-identical. `writer-final-source-sha256.json`
  records all eight final source hashes, so no native recompilation is needed
  for this formatter-only update.
- `logs/writer-native-r2`: the actual writer translation unit builds for
  SM107a and SM107f with CUDA 13.4.92, release-like flags and the installed
  stable PyTorch headers. Supplemental cuBLAS files are declaration headers
  only; no CUDA 12.5 runtime/device headers are used.
- `logs/writer-binding-syntax-r1`: the complete actual binding translation
  unit passes C++ syntax checking.
- `logs/writer-boxing-r1`: actual schema and C++ declaration register and
  dispatch through stable PyTorch boxing on CPU, including ordinary defaults,
  all appended scalar/tensor arguments and mutation annotations. Its body is
  a CPU probe, not the CUDA writer.
- `writer-source-parity.json`: 21 scoped comparisons pass, including twelve
  public numerical helpers, five qualified ICP helpers, the public slot
  stores, the complete qualified metadata header, the untouched independent
  public CUDA tests and the verbatim ordinary schema prefix.
- `logs/writer-codegen-r2`: all 76 ordinary kernel instruction streams are
  identical to the public baseline on each of SM107a and SM107f. This compares
  SASS opcodes/operands/order, retaining register names while omitting encoded
  hex and instruction addresses.

The new GPU cases separately cover public per-head compound bytes/canaries,
both fragment owners, N=129/1023/1024, distinct nonunit K/V scales, the index-Q
rounding midpoint, NaN class, invalid geometry and misaligned offset views.
The retained plan suite includes zero-token metadata publication. None of
these GPU cases has been executed here. The ICP main-cache byte order changes
to the public layout, so target-image numerical, graph, distributed, model
accuracy and performance qualification must be repeated before claiming the
public candidate retains measured ICP performance.
