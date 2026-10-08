// K5T: tiled symmetric-memory candidate exchange + merge, for EVERY extent.
//
// One launch per sparse layer replaces both routes the caller used to choose
// between by phase: the NCCL route (ATen destination-major pack + all_to_all +
// K2) and K5. The kernel reads this rank's QUERY-major `[T, H_group, 16, 2]`
// candidates, pushes each destination's head slab straight into that peer's
// symmetric receive window (the C4 source-major layout, so no host pack exists),
// polls its own window per word and merges with the SAME `merge_topk.cuh` K2
// uses, with K2's per-row C3/C5 semantics. The output is therefore
// bit-identical to `all_to_all + k2_merge` on every row, failed rows included.
//
// Window layout (per rank, per slot) is K5's OPT_LAMPORT layout:
//   u64 [W_source][T_cap][H_local][16][2], each u64 = (generation << 32) | word
// so the receive window is k5t_slot_words(T_cap, H_group) int32 words per slot.
//
// Why a new kernel rather than a K5 mask: K5 derives each word's Lamport
// generation from a per-CTA counter, which pins the row->CTA map to the
// allocation capacity and therefore the grid to ceil(T / tpb_cap). Prefill
// needs the grid sized by occupancy at every extent. Here the generation is
// owned by a fixed TILE of rows (tile_tok = 8 / H_local tokens, one warp per
// (token, local head) row), and a CTA walks a contiguous range of tiles. The
// tile->word map is a property of the window, so the grid is free per launch:
//   grid = min(ntiles(T), SMs * resident CTAs/SM)
// which is G5's nblocks* = min(ceil(T*H_local/8), SMs*CTAs_per_SM).
//
// Protocol, no ACK: every CTA first publishes ALL its tiles, then polls and
// merges them, so every resident CTA has published before any waits. Slot reuse
// is safe under the caller's discipline (no two consecutive launches on one
// slot, wrap included, slots >= 3) plus the per-layer TP collective between two
// exchanges: by the time a rank reuses a slot its peer has retired the launch
// that read it. A word's tag is at most its tile's current counter, and the
// expected tag is counter + 1, so a stale word can never be accepted.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "merge_topk.cuh"

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;
constexpr int kMaxWorld = 8;
// sm_107 schedules 1024 threads/SM: four 256-thread CTAs is the ceiling, so
// asking ptxas for it caps registers at 64 without costing a CTA anywhere.
constexpr int kMinBlocksPerSm = 4;
constexpr int kWordsPerRecord = 4;  // two u64 tagged words per (score, id)

// Transport failure: a tag never matched within the spin budget, or the
// workspace was already dead. Sticky; distinct from K2's C5/C3 bits.
constexpr int32_t kStatusTransport = 4;
static_assert((kStatusTransport & (icp::kStatusNaN | icp::kStatusRowMeta)) == 0,
              "the transport bit must not alias a K2 status bit");

// Gate-only knobs; production passes 0.
constexpr int kFlagNoPublish = 1;  // negative control: merge whatever is there
constexpr int kFlagNoAcquire = 2;  // negative control: read without the tag check
// Production knob: launch as a programmatic dependent; the kernel waits on the
// selector before any global access and lets the attention prologue launch.
constexpr int kFlagPdl = 4;
// A/B knob: the classic 32-round merge instead of the shuffle-network merge
// (`warp_merge_topk16_net`, same bits) where one candidate fits per lane.
constexpr int kFlagClassicMerge = 8;
constexpr int kFlagsKnown = kFlagNoPublish | kFlagNoAcquire | kFlagPdl | kFlagClassicMerge;

struct PeerBufs {
  int32_t* buf[kMaxWorld];
};

// Strong, .sys-scoped, naturally aligned 64-bit accesses: single-copy atomic
// under the PTX memory model, so a tag and its payload word never tear.
__device__ __forceinline__ uint64_t ld_relaxed_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_relaxed_sys_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.sys.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// Generation 0 is what a zeroed window holds, so it is never issued.
__device__ __forceinline__ uint32_t next_generation(uint32_t prev) {
  const uint32_t next = prev + 1u;
  return next == 0u ? 1u : next;
}

template <int KPT>
__global__ __launch_bounds__(kThreads, kMinBlocksPerSm) void k5t_kernel(
    const int32_t* __restrict__ local_cand, const __grid_constant__ PeerBufs peers,
    int32_t* __restrict__ out, const int32_t* __restrict__ forced,
    const int32_t* __restrict__ n_ordinary, uint32_t* __restrict__ gen,
    int32_t* __restrict__ status, int rank, int world, int T, int T_cap, int Hg,
    int Hl, int tile_tok, int ntiles, long long slot_words, int slot, int flags,
    long long spin_cycles) {
  constexpr int K = icp::kTopK;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // Balanced contiguous tile range; grid <= ntiles, so every CTA owns >= 1.
  const int j0 = (int)(((long long)ntiles * blockIdx.x) / gridDim.x);
  const int j1 = (int)(((long long)ntiles * (blockIdx.x + 1)) / gridDim.x);
  const int per_tok = Hl * K;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (flags & kFlagPdl) {
    // Every CTA is resident before the dependent may launch, so the early
    // trigger cannot starve this grid's own polls of SM slots.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
  __shared__ int s_dead;
  if (tid == 0)
    s_dead = (*reinterpret_cast<volatile int32_t*>(status) & kStatusTransport) != 0;
  __syncthreads();
  if (s_dead) {
    // A dead workspace renders K2's failed row (all -1) and publishes nothing.
    const long r0 = (long)j0 * tile_tok * per_tok;
    const long t1 = (long)j1 * tile_tok < T ? (long)j1 * tile_tok : T;
    for (long i = r0 + tid; i < t1 * per_tok; i += kThreads) out[i] = -1;
    return;
  }

  // 1. Publish every owned tile, tagged with that tile's next generation.
  for (int j = j0; j < j1; ++j) {
    const uint32_t seq = next_generation(gen[j]);
    __syncthreads();  // every thread has read gen[j] before it moves
    if (tid == 0) gen[j] = seq;
    if (flags & kFlagNoPublish) continue;
    const int t0 = j * tile_tok;
    const int ntok = (T - t0) < tile_tok ? (T - t0) : tile_tok;
    const int nunit = ntok * per_tok;  // records per destination
    const uint64_t tag = (uint64_t)seq << 32;
    for (int u = tid; u < world * nunit; u += kThreads) {
      const int p = u / nunit;
      const int i = u - p * nunit;
      const int lt = i / per_tok;
      const int r = i - lt * per_tok;  // hl * K + k
      // Destination p owns global heads [p*Hl, (p+1)*Hl) (C7 rank-major).
      const int2 pr = *reinterpret_cast<const int2*>(
          local_cand + (((long)(t0 + lt) * Hg + (long)p * Hl) * K + r) * 2);
      uint64_t* d = reinterpret_cast<uint64_t*>(peers.buf[p] +
                                                (long)slot * slot_words) +
                    2L * (((long)rank * T_cap + t0) * per_tok + i);
      st_relaxed_sys_u64(d, tag | (uint64_t)(uint32_t)pr.x);
      st_relaxed_sys_u64(d + 1, tag | (uint64_t)(uint32_t)pr.y);
    }
  }
  __syncthreads();  // gen[j] writes by thread 0 are visible to the block

  // 2. Poll + merge, one warp per (token, local head) row.
  const uint64_t* const mybuf = reinterpret_cast<const uint64_t*>(
      peers.buf[rank] + (long)slot * slot_words);
  const int n = world * K;
  bool lam_bad = false;  // warp-uniform and sticky: one timeout per warp
  for (int j = j0; j < j1; ++j) {
    const uint32_t seq = gen[j];
    const int t0 = j * tile_tok;
    const int ntok = (T - t0) < tile_tok ? (T - t0) : tile_tok;
    for (int w = warp; w < ntok * Hl; w += kWarpsPerBlock) {
      const int lt = w / Hl;
      const int hl = w - lt * Hl;
      const int t = t0 + lt;

      // C3 row metadata, K2's check and K2's failure rendering.
      int32_t f = icp::kNoForcedBlock;
      int q = icp::kTopK;
      int32_t err = icp::kStatusOk;
      if (forced != nullptr) {
        f = forced[t];
        q = n_ordinary[t];
        const int expect =
            (f < 0) ? 0 : (f < icp::kTopK - 1 ? f : icp::kTopK - 1);
        if (q != expect) err |= icp::kStatusRowMeta;
      }

      uint64_t key[KPT];
      int32_t gid[KPT];
      bool nan_seen = false;
#pragma unroll
      for (int i = 0; i < KPT; ++i) {
        // Same (lane, i) -> (source, k) map as icp::load_candidates_c4.
        const int idx = lane + i * 32;
        const bool active = idx < n;
        const int c = active ? idx / K : 0;
        const int k = active ? idx - c * K : 0;
        const uint64_t* p2 =
            mybuf + ((((long)c * T_cap + t) * Hl + hl) * K + k) * 2;
        uint32_t sw = 0u;
        uint32_t gw = 0xffffffffu;
        if (!lam_bad && !(flags & kFlagNoAcquire)) {
          const long long q0 = clock64();
          for (;;) {
            bool ready = true;
            if (active) {
              const uint64_t w0 = ld_relaxed_sys_u64(p2);
              const uint64_t w1 = ld_relaxed_sys_u64(p2 + 1);
              // Exact compare: a later generation is refused, not accepted.
              ready = ((uint32_t)(w0 >> 32) == seq) && ((uint32_t)(w1 >> 32) == seq);
              if (ready) {
                sw = (uint32_t)w0;
                gw = (uint32_t)w1;
              }
            }
            if (__all_sync(icp::kFullMask, ready)) break;
            if (__any_sync(icp::kFullMask, (clock64() - q0) > spin_cycles)) {
              lam_bad = true;
              break;
            }
          }
        } else if (!lam_bad && active) {
          sw = (uint32_t)ld_relaxed_sys_u64(p2);
          gw = (uint32_t)ld_relaxed_sys_u64(p2 + 1);
        }
        const int32_t g = (active && !lam_bad) ? (int32_t)gw : -1;
        const float score = __uint_as_float(sw);
        nan_seen = nan_seen || (g >= 0 && score != score);
        key[i] = (active && !lam_bad) ? icp::canonical_key(score, g) : 0ull;
        gid[i] = g;
      }
      if (lam_bad) err |= kStatusTransport;
      if (__any_sync(icp::kFullMask, nan_seen)) err |= icp::kStatusNaN;
      if (err != icp::kStatusOk) {
        f = icp::kNoForcedBlock;
        q = 0;
        if (lane == 0) atomicOr(status, err);
      }
      int32_t* const row_out = out + (long)(t * Hl + hl) * icp::kTopK;
      if constexpr (KPT == 1) {
        if (!(flags & kFlagClassicMerge)) {
          icp::warp_merge_topk16_net(key[0], gid[0], lane, f, q, row_out);
        } else {
          icp::warp_merge_topk16<KPT>(key, gid, lane, f, q, row_out);
        }
      } else {
        icp::warp_merge_topk16<KPT>(key, gid, lane, f, q, row_out);
      }
    }
  }
}

int tile_tokens(int Hl) {
  const int tt = kWarpsPerBlock / Hl;
  return tt < 1 ? 1 : tt;
}

int cdiv(long a, long b) { return (int)((a + b - 1) / b); }

template <int KPT>
int resident_ctas_for() {
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  static std::mutex mu;
  static std::vector<int> cache;
  std::lock_guard<std::mutex> lock(mu);
  if ((int)cache.size() <= dev) cache.resize(dev + 1, 0);
  if (cache[dev] == 0) {
    int sms = 0;
    int per_sm = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, k5t_kernel<KPT>, kThreads, 0));
    TORCH_CHECK(per_sm >= 1, "k5t: kernel cannot be resident on device ", dev);
    cache[dev] = sms * per_sm;
  }
  return cache[dev];
}

int resident_ctas(int kpt) {
  switch (kpt) {
    case 1: return resident_ctas_for<1>();
    case 2: return resident_ctas_for<2>();
    case 4: return resident_ctas_for<4>();
    default: TORCH_CHECK(false, "k5t: unsupported candidates-per-lane ", kpt);
  }
  return 0;
}

int kpt_for(int64_t world) { return (int)((world * icp::kTopK + 31) / 32); }

}  // namespace

int64_t k5t_slot_words(int64_t T_cap, int64_t Hg) {
  return T_cap * Hg * icp::kTopK * kWordsPerRecord;
}

// (tile_tok, ntiles for this extent, ntiles at capacity, grid, resident cap)
std::vector<int64_t> k5t_plan(int64_t T, int64_t T_cap, int64_t Hl,
                              int64_t world, int64_t max_ctas) {
  TORCH_CHECK(Hl >= 1 && T >= 0 && T_cap >= T, "k5t_plan: bad geometry");
  const int tt = tile_tokens((int)Hl);
  const int ntiles = cdiv(T, tt);
  const int ntiles_cap = cdiv(T_cap, tt);
  const int cap = max_ctas > 0 ? (int)max_ctas : resident_ctas(kpt_for(world));
  const int grid = ntiles < cap ? ntiles : cap;
  return {tt, ntiles, ntiles_cap, grid, cap};
}

void k5t_exchange(at::Tensor local_cand, at::Tensor out,
                  std::vector<int64_t> peer_buf_ptrs, at::Tensor gen,
                  at::Tensor status, c10::optional<at::Tensor> forced,
                  c10::optional<at::Tensor> n_ordinary, int64_t rank,
                  int64_t world, int64_t head_offset, int64_t slot,
                  int64_t slots, int64_t tokens_capacity,
                  int64_t slot_capacity_words, int64_t max_ctas,
                  int64_t spin_cycles, int64_t flags) {
  TORCH_CHECK(local_cand.is_cuda() && out.is_cuda() && gen.is_cuda() &&
                  status.is_cuda(),
              "k5t: tensors must be CUDA");
  const auto device = local_cand.device();
  TORCH_CHECK(out.device() == device && gen.device() == device &&
                  status.device() == device,
              "k5t: every tensor must be on the candidates' device");
  TORCH_CHECK(local_cand.scalar_type() == at::kInt,
              "k5t: cand must be int32 (C4 words, bitcast); got ",
              local_cand.scalar_type());
  TORCH_CHECK(out.scalar_type() == at::kInt, "k5t: out must be int32");
  TORCH_CHECK(gen.scalar_type() == at::kInt, "k5t: gen must be int32");
  TORCH_CHECK(status.scalar_type() == at::kInt && status.numel() >= 1,
              "k5t: status must be int32 with >= 1 element");
  TORCH_CHECK(local_cand.is_contiguous() && out.is_contiguous() &&
                  gen.is_contiguous() && status.is_contiguous(),
              "k5t: tensors must be contiguous");
  TORCH_CHECK(local_cand.dim() == 4 && local_cand.size(2) == icp::kTopK &&
                  local_cand.size(3) == 2,
              "k5t: cand must be [T, H_group, 16, 2]");
  TORCH_CHECK(out.dim() == 3 && out.size(2) == icp::kTopK,
              "k5t: out must be [T, H_local, 16]");
  TORCH_CHECK(world >= 1 && world <= kMaxWorld, "k5t: world outside [1, 8]");
  TORCH_CHECK((int64_t)peer_buf_ptrs.size() == world,
              "k5t: need one receive window pointer per rank");
  TORCH_CHECK(rank >= 0 && rank < world, "k5t: rank outside [0, world)");

  const int T = (int)local_cand.size(0);
  const int Hg = (int)local_cand.size(1);
  const int Hl = (int)out.size(1);
  TORCH_CHECK(out.size(0) == T, "k5t: out token count must match cand");
  TORCH_CHECK(Hg == (int)world * Hl, "k5t: H_group must be world * H_local (C7)");
  TORCH_CHECK(head_offset == rank * Hl,
              "k5t: C7 pins head_offset = icp_rank * H_local; got ",
              head_offset, " for rank ", rank, " and H_local ", Hl);
  TORCH_CHECK(Hl >= 1 && Hl <= 16 && (16 % Hl) == 0,
              "k5t: H_local must be in {1, 2, 4, 8, 16}; got ", Hl);
  TORCH_CHECK(slots >= 1 && slot >= 0 && slot < slots, "k5t: slot outside [0, slots)");
  TORCH_CHECK(flags >= 0 && (flags & ~(int64_t)kFlagsKnown) == 0,
              "k5t: unknown flags ", flags);
  TORCH_CHECK(spin_cycles > 0, "k5t: spin_cycles must be positive");
  TORCH_CHECK(tokens_capacity >= 1 && tokens_capacity >= T,
              "k5t: extent ", T, " exceeds the window capacity ", tokens_capacity,
              "; the window cannot be resized, and a clamp would leave stale rows");
  const int T_cap = (int)tokens_capacity;
  const long long slot_words = (long long)k5t_slot_words(T_cap, Hg);
  TORCH_CHECK(slot_capacity_words >= slot_words,
              "k5t: the window holds ", slot_capacity_words, " words per slot but ",
              "T_cap=", T_cap, " H_group=", Hg, " needs ", slot_words,
              "; an undersized window is a silent overrun of a PEER's allocation");
  const int tt = tile_tokens(Hl);
  const int ntiles_cap = cdiv(T_cap, tt);
  TORCH_CHECK(gen.numel() >= (int64_t)slots * ntiles_cap,
              "k5t: gen must hold slots * ntiles(T_cap) = ", slots * ntiles_cap,
              " counters; got ", gen.numel());
  PeerBufs peers{};
  for (int i = 0; i < (int)world; ++i) {
    TORCH_CHECK(peer_buf_ptrs[i] != 0, "k5t: null peer window ", i);
    TORCH_CHECK((peer_buf_ptrs[i] % 16) == 0, "k5t: peer window ", i,
                " is not 16-byte aligned");
    peers.buf[i] = reinterpret_cast<int32_t*>(peer_buf_ptrs[i]);
  }

  // C3 planes: both or neither, per token row.
  TORCH_CHECK(forced.has_value() == n_ordinary.has_value(),
              "k5t: forced and n_ordinary must be supplied together (C3)");
  const int32_t* fp = nullptr;
  const int32_t* np = nullptr;
  if (forced.has_value()) {
    const at::Tensor& ft = forced.value();
    const at::Tensor& nt = n_ordinary.value();
    TORCH_CHECK(ft.device() == device && nt.device() == device,
                "k5t: forced/n_ordinary must be on the candidates' device");
    TORCH_CHECK(ft.scalar_type() == at::kInt && nt.scalar_type() == at::kInt,
                "k5t: forced/n_ordinary must be int32");
    TORCH_CHECK(ft.is_contiguous() && nt.is_contiguous(),
                "k5t: forced/n_ordinary must be contiguous");
    TORCH_CHECK(ft.numel() == T && nt.numel() == T,
                "k5t: forced/n_ordinary are per-token-row planes of length ", T);
    fp = ft.data_ptr<int32_t>();
    np = nt.data_ptr<int32_t>();
  }

  if (T == 0) return;
  const c10::cuda::CUDAGuard guard(device);
  const int kpt = kpt_for(world);
  const int ntiles = cdiv(T, tt);
  const int cap = max_ctas > 0 ? (int)max_ctas : resident_ctas(kpt);
  const int grid = ntiles < cap ? ntiles : cap;
  auto stream = c10::cuda::getCurrentCUDAStream();
  uint32_t* gp = reinterpret_cast<uint32_t*>(gen.data_ptr<int32_t>()) +
                 (long)slot * ntiles_cap;

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = (flags & kFlagPdl) ? 1 : 0;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int32_t* const outp = out.data_ptr<int32_t>();
  const int32_t* const candp = local_cand.data_ptr<int32_t>();
  int32_t* const statp = status.data_ptr<int32_t>();

#define K5T_LAUNCH(KPT)                                                        \
  C10_CUDA_CHECK(cudaLaunchKernelEx(                                           \
      &cfg, k5t_kernel<KPT>, candp, peers, outp, fp, np, gp, statp, (int)rank, \
      (int)world, T, T_cap, Hg, Hl, tt, ntiles, slot_words, (int)slot,        \
      (int)flags, (long long)spin_cycles))

  switch (kpt) {
    case 1: K5T_LAUNCH(1); break;
    case 2: K5T_LAUNCH(2); break;
    case 4: K5T_LAUNCH(4); break;
    default: TORCH_CHECK(false, "k5t: unsupported candidates-per-lane ", kpt);
  }
#undef K5T_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k5t_exchange", &k5t_exchange,
        "Tiled symmetric-memory ICP candidate push + merge (every extent)",
        pybind11::arg("local_cand"), pybind11::arg("out"),
        pybind11::arg("peer_buf_ptrs"), pybind11::arg("gen"),
        pybind11::arg("status"), pybind11::arg("forced"),
        pybind11::arg("n_ordinary"), pybind11::arg("rank"),
        pybind11::arg("world"), pybind11::arg("head_offset"),
        pybind11::arg("slot"), pybind11::arg("slots"),
        pybind11::arg("tokens_capacity"), pybind11::arg("slot_capacity_words"),
        pybind11::arg("max_ctas"), pybind11::arg("spin_cycles"),
        pybind11::arg("flags"));
  m.def("k5t_plan", &k5t_plan,
        "(tile_tok, ntiles, ntiles_cap, grid, resident_cap)",
        pybind11::arg("T"), pybind11::arg("T_cap"), pybind11::arg("H_local"),
        pybind11::arg("world"), pybind11::arg("max_ctas") = 0);
  m.def("k5t_slot_words", &k5t_slot_words,
        "int32 words per slot of the receive window",
        pybind11::arg("T_cap"), pybind11::arg("H_group"));
  m.attr("k5t_abi_version") = "refined-icp-v1.k5t.1";
  m.attr("k5t_window_layout") =
      "u64 (gen<<32 | word) [W_source][T_cap][H_local][16][2] per slot";
  m.attr("k5t_status_nan") = static_cast<int>(icp::kStatusNaN);
  m.attr("k5t_status_row_meta") = static_cast<int>(icp::kStatusRowMeta);
  m.attr("k5t_status_transport") = static_cast<int>(kStatusTransport);
  m.attr("k5t_flag_no_publish") = kFlagNoPublish;
  m.attr("k5t_flag_no_acquire") = kFlagNoAcquire;
  m.attr("k5t_flag_pdl") = kFlagPdl;
  m.attr("k5t_flag_classic_merge") = kFlagClassicMerge;
}
