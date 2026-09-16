# NVFP4 cache readers

`sparse_atten_nvfp4_kv_func(..., kv_layout="legacy")` preserves the existing
128x4 tiled block scales and existing argument names. `kv_layout="vllm"`
selects page-128 HND payloads and K-linear/V-token-quad block scales. Both
layouts use the existing prefill math and may run in one process. Prefill
accepts BF16 or E4M3 Q; the optimized decoder accepts E4M3 Q.

`fmha_sm100(..., nvfp4_kv=cache)` and the sparse adapter accept this dictionary:

```python
cache = dict(
    k_data=k_data, v_data=v_data,       # uint8 [P,H,128,64]
    k_scale=k_sf, v_scale=v_sf,         # uint8 E4M3 bits [P,H,128,8]
    layout="vllm",
    k_global_scale=alpha_k,            # one CUDA float32 value
    v_global_scale=alpha_v,            # one CUDA float32 value
)
```

The allocation is `[P,2*H,128,72]` bytes. Each physical page contains
`K_data[8192H]`, `K_scale[1024H]`, `V_data[8192H]`, `V_scale[1024H]`.
The actual page stride may exceed `18432H` bytes. All four views retain that
stride and their storage offsets. Payload head/token strides are 8192/64;
scale head/token strides are 1024/8. Scale storage indexing is physical:
K `(t,s)` is at `t*8+s`; V is at `(t//4)*32+4*s+t%4` within one head.
Views must be 16-byte aligned. Address products use 64-bit arithmetic.
No full-cache copy or scale-layout conversion occurs in either reader.

Global scales may have shape `[]` or `[1]`; both readers preserve their
storage and accept them during graph replay. Global scales reconstruct
`x = E2M1(code)*E4M3(sf)*alpha`. The owner validates
positive finite scales before writing pages and keeps them fixed while those
pages are live. Readers consume device buffers without host synchronization.
The decoder stages `E4M3(E2M1(code)*sf/6)` and compensates by `6*alpha`.
For example K code=2, sf=3, alpha=0.5 stages 1 and reconstructs 3; V code=1.5,
sf=8, alpha=2 stages 2 and reconstructs 24. The staging factor is independent
of alpha. Split-KV uses the identical effective logit scale in its reduction.
The public dequantizer uses FP16 intermediates for scale/gamma and code
multiplication before converting to E4M3. FP8 staging adds rounding. The
existing decoder also rounds softmax probabilities to E4M3 before PV. Tests
require bit-exact agreement with the existing FP8 reader on independently
staged K/V and report error against float attention with both staged K/V and
direct FP4 reconstruction. The float reference does not emulate probability
rounding and is not treated as an exact oracle for that arithmetic.

Existing logical metadata conventions remain unchanged: sparse block indices
name logical pages, and `kv_indices` maps them to physical pages. Provide
current used lengths and current selected pages; unused page capacity is not
valid context. JIT/planning occur outside capture. Device metadata and scale
buffer addresses remain stable during replay. The descriptor overrides the
ordinary positional K/V arguments, so callers may pass its packed payloads
without allocating shape-only full-width tensors.

TP2 `(Hq,Hkv)=(32,2)` and TP4 `(16,1)` are required correctness geometries;
actual hardware and test evidence must be recorded independently. The decoder
uses the existing planner and its query/head limits. BF16 short chunks use
the prefill reader. Other NVFP4 variants and page geometries are rejected.

The optimized decode implementation is adapted from the source-only reader
change at `5ddd03a605672e0a1f8ac23785204ec36d32ad09`; it contains no ICP writer
or planning changes. The required dequantization header is a regular packaged
file. NVFP4 JIT variants use separate names and hidden visibility to prevent
weak C++ symbols from interposing between FP8 and NVFP4 libraries. CUDA and
CuTe DSL versions follow vLLM's supported build configuration.

Validation uses `tests/regression/test_nvfp4_readers.py`. The source was built
and exercised on one GB300 (SM103), using vLLM's CUDA 13.0.3 components
(NVCC 13.0.88, CUDART 13.0.96), CuTe DSL 4.7.1, quack-kernels 0.6.5, and
MSA's unchanged CUTLASS gitlink `eb61c911471867a5fd2466bfd8f29306cea6ebf8`.
The public compiler built SM100a/SM103a/SM100f targets; only SM103 was run.

Coverage includes Hkv=1/2 (TP4/TP2 per-rank geometry), decode batches
1/2/15/16/32, Q lengths 1/4/32, one and two KV splits, unequal device global
scales, nonunit Q/output multipliers, both FP8/NVFP4 load orders, and graph
replay with changed Q, physical pages, and scale buffers. Prefill covers
Q=1/33, K=127/128/129/257, ragged batches with changing used lengths, BF16
and FP8 Q, legacy/new layout equivalence, scalar scale buffers under graph
capture, storage offsets, and padded page strides. CPU tests also render both AOT templates for all 350 ordinary
variants without supplying NVFP4 options. A wheel build and installation
check verifies public modules, embedded CUTLASS, and regular reader headers.

These are reader regression checks, not a model accuracy acceptance gate.
In the fixed-seed decode cases, NVFP4 and independently staged ordinary FP8
outputs agree bit for bit. Their shared float-reference relative L2 error is
about 2.6%; direct FP4 reconstruction differs by about 3.0%. The Q=32 case
has four of 262144 elements beyond the initial 0.025 + 0.025*abs(reference)
criterion in both readers, with maximum absolute error 0.03045. This existing
FP8 arithmetic difference is retained and reported; the reader comparison
uses zero tolerance. BF16 ragged prefill has approximately 0.29% relative
L2 error against the float reference. Actual multi-GPU TP2/TP4 serving,
full-model quality, and other GPU architectures remain separate validation.
