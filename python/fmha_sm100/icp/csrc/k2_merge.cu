// K2: the reference ICP merge. The C4 carrier -> this rank's Top-16.
//
// Input  `cand`  [W, Qchunk, H_local, 16, 2] **int32**, the refined-icp-v1 C4
//                candidate carrier in its RECEIVE form: axis 0 is the SOURCE
//                rank and axis 2 holds only THIS rank's own heads (C4).
//                `[...,0]` is the raw FP32 score bits and `[...,1]` the int32
//                global block id -- both BITCAST, never converted.
//                Produced by `icp_kernels.carrier.exchange_carrier`: either an
//                `all_to_all_single` with the fixed per-peer split, or the
//                all-gather + head-slice reference under the same API.
// Output `out`   [Qchunk, H_local, 16] int32, global block ids ascending with a
//                -1 tail (CONTRACT C6 -- exactly today's sparse_topk_select
//                contract, so the attend side is untouched).
//
// One warp per (query, local head) row. The merge itself lives in
// merge_topk.cuh so K5 can reuse it byte for byte.
//
// ---------------------------------------------------------------------------
// CARRIER `refined-icp-v1.C4`. What changed, and what deliberately did not.
//
// WAS: fp32 `[C, T, H_group, 16, 2]` -- every rank receiving every rank's ALL
// FOUR heads, with the kernel selecting its own window by `h = head_offset+hl`.
// That ships W times the bytes the merge reads, and makes the C7 head window a
// runtime assertion rather than a property of the buffer.
//
// NOW: int32 `[W, Qchunk, H_local, 16, 2]`, source-major, carrying only the
// destination's owned heads. `head_offset` stays in the signature and is still
// checked, but it no longer ADDRESSES anything: a source-major carrier cannot
// reach a peer's heads. The total element count is unchanged
// (`W * H_local == H_group == 4` for this model), which is why K5's slot sizing
// in `k5_exchange.cu::k5_slot_floats` needs no change at all.
//
// The WIRE CONTENT is unchanged: the fp32 carrier already held
// `(fp32 score, int32 gid bitcast to fp32)`. This is a container dtype and an
// indexing change, NOT a numeric conversion.
//
// ARITY IS DELIBERATELY UNTOUCHED. The carrier move changed only the dtype,
// the shape of `cand` and the resulting call into merge_topk.cuh. No argument
// is added or removed, the forced-slot semantics and the NaN failure path are
// unmodified by it, and `forced` /
// `n_ordinary` remain per-TOKEN-ROW `[Qchunk]` planes -- the C4 carrier changes
// where a row's CANDIDATES come from, never what a row IS.
//
// ---------------------------------------------------------------------------
// HOST ABI `refined-icp-v1.k2.2`. Three arguments were added to the five of
// `refined-icp-v1.k2.1`:
//
//   forced      int32 [T] or None -- this row's forced block f = p//128; -1
//                                    marks an INACTIVE row (zero valid ids).
//   n_ordinary  int32 [T] or None -- how many ordinary winners the caller wants
//                                    for this row: C3's min(15, M-1).
//   status      int32 [>=1]       -- REQUIRED. The invocation's failure word.
//
// WHY A PER-ROW PLANE AND NOT A SCALAR. `f` and `M` are functions of the row's
// own query position p, and prefill / chunked / Q>1 verification rows in one
// invocation have different p -- a final sequence length is never a substitute.
// M0's scalar forced-block argument is exactly the thing C3 records as unable
// to express per-token ownership. It is indexed by t only, never by head: forcing depends on the
// query position, so a per-head plane would be able to express a contradiction.
//
// WHY BOTH f AND n_ordinary, WHEN n_ordinary = min(15, f) IS DERIVABLE.
// M = ceil((p+1)/128) = p//128 + 1 = f+1 identically, so the kernel COULD derive
// the count. Carrying it makes the caller state its own arithmetic and lets the
// kernel check it against C3's formula (`kStatusRowMeta`). That check is the
// cheap one that catches a caller deriving M from a batch-wide sequence length
// instead of the row's p -- the single failure mode the design docs warn about
// twice. The redundancy is the point; it is not a second source of truth.
//
// WHY status IS REQUIRED AND NOT OPTIONAL. C5 says a NaN score fails the
// invocation. An optional failure channel is an opt-out from a MUST, and the
// default of any opt-out is today's behaviour: `merge_topk.cuh` sorts NaN ABOVE
// +inf, so the NaN candidate silently WINS. Callers that do not pass a status
// word now fail to compile/bind instead of silently selecting on a NaN.
// The kernel only ORs bits into it; zeroing it before the invocation and
// checking it afterwards belong to the caller (see
// `icp_kernels/merge.py::merge_candidates`), which is what keeps this usable
// under CUDA-graph capture, where a device->host sync is not available.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include "merge_topk.cuh"

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

template <int KPT>
__global__ void k2_merge_kernel(const int32_t* __restrict__ cand,
                                int32_t* __restrict__ out,
                                const int32_t* __restrict__ forced,
                                const int32_t* __restrict__ n_ordinary,
                                int32_t* __restrict__ status, int W, int Qchunk,
                                int K, int Hl) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= Qchunk * Hl) return;

  const int t = row / Hl;
  const int hl = row - t * Hl;
  // No `h = head_offset + hl`: the C4 receive carrier is already this rank's
  // slice, so `hl` IS the address. See merge_topk.cuh::load_candidates_c4.

  // Row metadata. Every value read here is warp-uniform: one warp owns one
  // (token, head) row, and forcing is indexed by the token only.
  int32_t f = icp::kNoForcedBlock;
  int q = icp::kTopK;
  int32_t err = icp::kStatusOk;
  if (forced != nullptr) {
    f = forced[t];
    q = n_ordinary[t];
    // C3: min(15, M-1) with M = ceil((p+1)/128) = f+1, and 0 for an inactive
    // row. A caller that disagrees fails the invocation; it does not get a
    // quietly different number of blocks.
    const int expect = (f < 0) ? 0 : (f < icp::kTopK - 1 ? f : icp::kTopK - 1);
    if (q != expect) err |= icp::kStatusRowMeta;
  }

  uint64_t key[KPT];
  int32_t gid[KPT];
  bool nan_seen = false;
  icp::load_candidates_c4<KPT>(cand, W, Qchunk, Hl, K, t, hl, lane, key, gid,
                               nan_seen);
  // C5: a NaN on any valid record of this row fails the invocation.
  if (__any_sync(icp::kFullMask, nan_seen)) err |= icp::kStatusNaN;

  if (err != icp::kStatusOk) {
    // A failed row publishes NO selection rather than a plausible one: a
    // numerical failure must prevent stale selections reaching attention or
    // token commit. All three arms write the same all--1 row, so a failing run
    // is still bit-exact across backends.
    //
    // Expressed as "no forced block, zero ordinary winners" rather than as an
    // early return with its own -1 store. The merge already renders that as the
    // all--1 row -- `win[i]` needs `rank[i] < 0`, which never holds -- so the
    // failure path needs no second implementation of the output format.
    //
    // It is ALSO what keeps the register count down. Measured compile-only on
    // sm_103a: an early
    // `return` between `load_candidates` and the merge costs 8 registers at
    // KPT=4 (40 vs 32) and 1 at KPT=2, with no spills either way; removing the
    // branch and keeping every other line identical restores it. (Those
    // numbers were taken on the fp32 gathered loader. With the int32 C4 loader
    // the whole merge baselines at 22/30/32; see
    // `_build.MERGE_REGISTER_BASELINE`. The branch result below is unaffected
    // -- it is about the early return, and it still holds.) The
    // branch splits the candidate loads from their consumers and ptxas stops
    // interleaving them.
    f = icp::kNoForcedBlock;
    q = 0;
    if (lane == 0) atomicOr(status, err);
  }

  icp::warp_merge_topk16<KPT>(key, gid, lane, f, q,
                              out + (long)row * icp::kTopK);
}

}  // namespace

void k2_merge(at::Tensor cand, at::Tensor out, int64_t head_offset,
              int64_t world, int64_t rank,
              c10::optional<at::Tensor> forced,
              c10::optional<at::Tensor> n_ordinary, at::Tensor status) {
  TORCH_CHECK(cand.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  // C4: the carrier is int32. A float32 `cand` is the PRE-C4 gathered tensor,
  // whose axis 2 is H_group rather than H_local -- accepting it here would read
  // it with the wrong head stride and merge a peer's heads under this rank's
  // name, silently. This is a refusal, never a reinterpretation.
  TORCH_CHECK(cand.scalar_type() == at::kInt,
              "cand must be int32: refined-icp-v1 C4 replaced the fp32 "
              "[C,T,H_group,16,2] gathered tensor with the int32 "
              "[W,Qchunk,H_local,16,2] carrier. The wire CONTENT is unchanged "
              "(score bits + block id, both bitcast), but the axes are not: "
              "axis 2 of the C4 carrier is H_LOCAL. Pack with "
              "icp_carrier.pack_send_carrier() and exchange with "
              "icp_carrier.exchange_carrier(); got ", cand.scalar_type());
  TORCH_CHECK(out.scalar_type() == at::kInt, "out must be int32");
  TORCH_CHECK(cand.is_contiguous() && out.is_contiguous(), "must be contiguous");
  TORCH_CHECK(cand.dim() == 5, "cand must be [W, Qchunk, H_local, K, 2]");
  TORCH_CHECK(out.dim() == 3, "out must be [Qchunk, H_local, 16]");

  const int C = cand.size(0);          // W, the SOURCE axis of the C4 carrier
  const int T = cand.size(1);          // Qchunk
  const int Hl_cand = cand.size(2);    // H_LOCAL, not H_group
  const int K = cand.size(3);
  TORCH_CHECK(cand.size(4) == 2, "last dim must be 2 (score bits, id)");
  TORCH_CHECK(K == icp::kTopK, "K must be 16");
  TORCH_CHECK(out.size(0) == T, "out query count must match cand");
  TORCH_CHECK(out.size(2) == icp::kTopK, "out last dim must be 16");
  const int Hl = out.size(1);
  // The C7 guards. In the gathered form the first of these read
  // `Hg == world * Hl` and the kernel then sliced with `head_offset`; in the C4
  // form the carrier IS the slice, so the check that replaces it is that the
  // carrier's head axis is exactly the output's. A carrier still carrying
  // H_group heads fails here instead of merging four heads into one.
  TORCH_CHECK(world == C, "world must be the carrier's source axis W; got ",
              world, " for W=", C);
  TORCH_CHECK(rank >= 0 && rank < world, "rank outside [0, world), got ", rank);
  TORCH_CHECK(Hl_cand == Hl,
              "C4 receive carrier axis 2 must be H_local (the destination's "
              "OWN heads, H_local = 4/W), matching out.size(1); got carrier ",
              Hl_cand, " vs out ", Hl,
              ". A carrier whose axis 2 is H_group has not been head-directed: "
              "either pack it destination-major and exchange it, or slice the "
              "all-gather reference path before calling this.");
  // `head_offset` no longer addresses anything, but C7 still pins its value and
  // a caller that cannot name its own rank has no business calling this kernel.
  // The K5 endpoint still needs it to select the slab it publishes.
  TORCH_CHECK(head_offset == rank * Hl,
              "C7 pins head_offset = icp_rank * H_local; got ", head_offset);
  TORCH_CHECK(C * K <= icp::kMaxCandidates, "too many candidates");

  // --- the failure word (C5) ----------------------------------------------
  // Required, never optional: see the ABI note at the top of this file.
  TORCH_CHECK(status.is_cuda(), "status must be a CUDA tensor");
  TORCH_CHECK(status.scalar_type() == at::kInt, "status must be int32");
  TORCH_CHECK(status.is_contiguous() && status.numel() >= 1,
              "status must be a contiguous int32 tensor with >= 1 element");

  // --- the forced-row metadata (C3) ----------------------------------------
  // Both planes or neither. One alone is a caller that thinks it is forcing and
  // is not, which is precisely the bug class C3 replaced.
  TORCH_CHECK(forced.has_value() == n_ordinary.has_value(),
              "forced and n_ordinary must be supplied together (C3): got "
              "forced=", forced.has_value(), " n_ordinary=",
              n_ordinary.has_value());
  const int32_t* fp = nullptr;
  const int32_t* np = nullptr;
  if (forced.has_value()) {
    const at::Tensor& ft = forced.value();
    const at::Tensor& nt = n_ordinary.value();
    TORCH_CHECK(ft.is_cuda() && nt.is_cuda(), "forced/n_ordinary must be CUDA");
    TORCH_CHECK(ft.scalar_type() == at::kInt && nt.scalar_type() == at::kInt,
                "forced/n_ordinary must be int32 (they are block ids and "
                "counts, never floats)");
    TORCH_CHECK(ft.is_contiguous() && nt.is_contiguous(),
                "forced/n_ordinary must be contiguous");
    TORCH_CHECK(ft.numel() == T && nt.numel() == T,
                "forced/n_ordinary are per-TOKEN-ROW planes of length T=", T,
                "; got ", ft.numel(), " and ", nt.numel(),
                ". They are indexed by t only -- forcing is a function of the "
                "row's query position, not of the head.");
    fp = ft.data_ptr<int32_t>();
    np = nt.data_ptr<int32_t>();
  }

  const int rows = T * Hl;
  if (rows == 0) return;
  const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  auto stream = c10::cuda::getCurrentCUDAStream();

  const int32_t* cp = cand.data_ptr<int32_t>();
  int32_t* op = out.data_ptr<int32_t>();
  int32_t* sp = status.data_ptr<int32_t>();
  const int kpt = (C * K + 31) / 32;

  switch (kpt) {
    case 1:
      k2_merge_kernel<1><<<blocks, kThreads, 0, stream>>>(
          cp, op, fp, np, sp, C, T, K, Hl);
      break;
    case 2:
      k2_merge_kernel<2><<<blocks, kThreads, 0, stream>>>(
          cp, op, fp, np, sp, C, T, K, Hl);
      break;
    case 4:
      k2_merge_kernel<4><<<blocks, kThreads, 0, stream>>>(
          cp, op, fp, np, sp, C, T, K, Hl);
      break;
    default:
      TORCH_CHECK(false, "unsupported candidates-per-lane ", kpt);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --- the C5 key, exposed for differential testing --------------------------
//
// `icp::canonical_key` is device code with no other host entry point, so a test
// that wants to compare the *shipped* key against a reference would otherwise
// have to transcribe it -- and a transcription that drifts in lockstep with the
// thing it gates is exactly how the `-0.0` divergence survived for weeks in the
// upstream tree. This probe calls the header's function; it does not restate
// it. `merge_topk.cuh` remains the only definition of the key.
//
// Output is the key **biased so that a signed 64-bit compare reproduces the
// contract's unsigned compare**: `(int64_t)(key ^ 2^63)`. The raw uint64 does
// not survive a round trip through an int64 torch tensor as an *ordered* value
// (keys >= 2^63 come back negative), and every consumer of this probe wants to
// order by it. `icp_kernels.canonical_key_reference` applies the same bias, so
// the two are directly `torch.equal`.
namespace {

__global__ void k2_key_kernel(const float* __restrict__ scores,
                              const int32_t* __restrict__ ids,
                              int64_t* __restrict__ out, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint64_t key = icp::canonical_key(scores[i], ids[i]);
  // Flip the sign bit: an unsigned compare on `key` and a signed compare on the
  // result agree, so the int64 tensor stays ordered. See the note above.
  out[i] = static_cast<int64_t>(key ^ (1ull << 63));
}

}  // namespace

void k2_canonical_key(at::Tensor scores, at::Tensor ids, at::Tensor out) {
  TORCH_CHECK(scores.is_cuda() && ids.is_cuda() && out.is_cuda(),
              "tensors must be CUDA");
  TORCH_CHECK(scores.scalar_type() == at::kFloat, "scores must be float32");
  TORCH_CHECK(ids.scalar_type() == at::kInt, "ids must be int32");
  TORCH_CHECK(out.scalar_type() == at::kLong, "out must be int64");
  TORCH_CHECK(scores.is_contiguous() && ids.is_contiguous() &&
                  out.is_contiguous(),
              "must be contiguous");
  TORCH_CHECK(scores.numel() == ids.numel() && scores.numel() == out.numel(),
              "scores, ids and out must have the same number of elements");
  const long n = scores.numel();
  if (n == 0) return;
  const int threads = 256;
  const long blocks = (n + threads - 1) / threads;
  auto stream = c10::cuda::getCurrentCUDAStream();
  k2_key_kernel<<<blocks, threads, 0, stream>>>(scores.data_ptr<float>(),
                                                ids.data_ptr<int32_t>(),
                                                out.data_ptr<int64_t>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k2_merge", &k2_merge,
        "ICP candidate merge (refined-icp-v1.k2.2 host ABI on the "
        "refined-icp-v1.C4 carrier): [W,Qchunk,Hl,16,2] int32 -> "
        "[Qchunk,Hl,16] int32, with the C3 reserved forced slot and the C5 NaN "
        "failure word",
        // No defaults, for any of the eight: C7 is a statement about the
        // caller's rank, C3 is a statement about the caller's row positions,
        // and C5's failure word is not something a caller may forget. A caller
        // that cannot name them has no business calling this kernel. `forced`
        // and `n_ordinary` accept None -- explicitly, at the call site --
        // which selects the plain ordinary merge with no reservation.
        pybind11::arg("cand"), pybind11::arg("out"),
        pybind11::arg("head_offset"), pybind11::arg("world"),
        pybind11::arg("rank"), pybind11::arg("forced"),
        pybind11::arg("n_ordinary"), pybind11::arg("status"));
  m.def("canonical_key", &k2_canonical_key,
        "CONTRACT C5 key from merge_topk.cuh, biased by 2^63 so a signed "
        "compare is the contract's unsigned compare",
        pybind11::arg("scores"), pybind11::arg("ids"), pybind11::arg("out"));
  m.attr("k2_abi_version") = "refined-icp-v1.k2.2";
  // The CARRIER version is independent of the HOST ABI version: the host ABI
  // moved k2.1 -> k2.2 (arity), and the carrier moved from the M0 gathered
  // fp32 tensor to C4. A caller that checks only `k2_abi_version`
  // would happily hand this build a float32 gathered tensor, so the carrier is
  // advertised too and the dtype check above refuses the old one.
  m.attr("k2_carrier") = "refined-icp-v1.C4";
  m.attr("k2_carrier_dtype") = "int32";
  m.attr("k2_carrier_shape") = "[W, Qchunk, H_local, 16, 2] source-major";
  m.attr("k2_status_nan") = static_cast<int>(icp::kStatusNaN);
  m.attr("k2_status_row_meta") = static_cast<int>(icp::kStatusRowMeta);
}
