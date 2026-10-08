// ===========================================================================
// THE refined-icp-v1 C4 CARRIER IN THIS FILE: A DTYPE CHANGE, NOTHING ELSE.
//
// `local_cand` and every peer payload plane are now **int32**. Nothing else in
// this file changed: the wire content was ALREADY the C4 record -- word 0 the
// FP32 score's raw bits, word 1 the int32 block id -- and this kernel already
// moved both as opaque 4-byte words. The float4 staging copies became int4
// (same 16 bytes), `float2` became `int2`, `__float_as_uint(x)` became a plain
// cast of an already-int word, and the merge's `base[0]` is now bitcast with
// `__int_as_float` instead of being loaded as a float. No numeric conversion
// existed before and none exists now.
//
// WHY THE PUBLISH LOOP IS UNTOUCHED. Section 2 below already emits the C4 wire
// layout: for each destination `p` it writes into `peers.buf[p]`'s source plane
// `rank` the head slab `[p*Hlocal, (p+1)*Hlocal)` of every token, giving the
// receiver `[W_source, T, Hlocal, K, 2]` -- C4's SOURCE-major receive carrier.
// It reads `local_cand` QUERY-major (its `src` strides by `Hg` per token) and
// packs in-kernel, which C4 explicitly allows for a direct publish; the
// no-host-pack requirement is scoped to K2. Handing this kernel the C4
// *sender* carrier instead would break that indexing silently for every
// `Q > 1` -- the negative control for it fires at Q=5 and coincides with the
// correct answer at Q=1.
//
// WHY NO SIZE CHANGED. `k5_slot_floats` returns `T * Hg * kTopK * words`. C4's
// carrier is `W * Qchunk * Hlocal * 16 * 2` elements and `W*Hlocal == Hg == 4`
// for this model, so the counts are equal and the symmetric window keeps its
// size. The unit is now int32 words rather than fp32 words -- the same 4 bytes.
// The NAME `k5_slot_floats` is deliberately NOT changed: it is an exported
// pybind symbol that `icp_comm.py` and `icp_kernels/exchange.py` both call, and
// a rename would be an ABI break for a unit that did not change size.
//
// THE C3 RESERVED FORCED SLOT IS DONE HERE. `forced` and `n_ordinary` are
// per-TOKEN-ROW int32 `[T]` planes, exactly as in `k2_merge.cu`, and the merge
// is reached through the SIX-argument `warp_merge_topk16(key, gid, lane,
// forced, n_ordinary, out)`. Omitting both planes still selects the four-
// argument compatibility overload's behaviour -- the plain ordinary merge --
// so the change is additive for every caller that does not force.
//
// Until this landed, K5 forwarded `kNoForcedBlock` on every row. That is not
// an edge case: the refined selector excludes the forced column from ordinary
// ranking unconditionally (`local_candidates.cuh`, CONTRACT C3 "forcing by
// exclusion"), so with nothing reinjecting it the forced block was absent from
// EVERY row K5 produced. A query in logical block 31 got `[0..15]` instead of
// `[0..14, 31]`: a full, ascending, well formed row with the query's own block
// missing, which the consumer reads as if it were correct.
//
// THE C5 NaN STATUS WORD IS STILL NOT DONE HERE. `load_candidates_c4`'s
// `nan_seen` is not consumed on this path and K5 has no status word; it keeps
// the `err` latch, which renders a failed row as block id 0
// (`fill_failed_rows` below) where K2 renders it as all -1. Those two
// renderings are not interchangeable and the divergence is deliberate rather
// than pending: the consumer derives its trip count from `kv_len` and never
// from this tensor, so an all--1 row is dereferenced entry by entry while a
// -1 TAIL on a healthy row is never read. See `docs/INTEGRATION.md` §9 and
// `docs/BOUNDARY.md`.
// ===========================================================================
//
// K5: the fused symmetric-memory candidate exchange + merge. One launch per
// indexer layer replaces `all_gather_into_tensor` + `k2_merge`: publish this
// rank's [T, H_group, 16, 2] candidates into symmetric memory, hand off, read the
// peers' candidates, write this rank's [T, H_local, 16] int32 (CONTRACT C6).
// The (lane, i) -> (source rank, k) mapping below is copied from
// `icp::load_candidates` character for character, because with duplicate
// global ids the merge is order sensitive. Blocks are partitioned by token identically on every
// rank, so block b only ever waits on block b of a peer. Only PUSH (mode 1) is
// built; `PUSH` and `mode` stay so restoring PULL is a `case`, not a rewrite.

#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

#include <cstdint>

#include "merge_topk.cuh"

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;
// One 128 B line per flag: sharing one turns C peer stores into a ping-pong.
constexpr int kFlagStride = 32;  // in uint32
// 0 = publish flag, 1 = ack, 2 = own generation, 3 = OPT_HIER arrival counter.
// `kind` is the outermost axis of `ctl_off`, so growing it moves no live address.
constexpr int kCtlKinds = 4;
constexpr int kMaxWorld = 8;
// Starving the merge costs far more than the extra block-pair handshakes save.
constexpr int kDefaultMaxBlocks = 64;

// Option bits. Values are frozen and reserved forever, so a mask number always
// decodes to the same optimisation; `k5_opts()` exports the table to Python.
enum : int {
  // Skip the `p == rank` iteration of the PUSH store loop and merge straight from
  // `local_cand`. RETIRED from the dispatch; the value stays in this enum so that
  // a request for it BY NAME gets its own refusal, not an anonymous unknown bit.
  OPT_SELF_ELIDE = 1 << 0,
  // 1 << 1 = vec2 -- not implemented here.
  // The `__threadfence_system()` before `st.release.sys` is redundant: the release
  // is itself cumulative and `__syncthreads()` already orders the other threads.
  OPT_NOFENCE = 1 << 2,
  // Hierarchical handshake: one system-scope release per grid, not per block.
  // COSTS the no-residency property, so `k5_exchange` refuses a non-resident grid.
  OPT_HIER = 1 << 3,
  // 1 << 4 backoff, 1 << 5 mcflag, 1 << 6 bcast, 1 << 7 incr -- not implemented.
  // Hoist `i / per_tok` out of both publish loops into one division per thread;
  // correct only when `per_tok` divides `kThreads`, which `k5_exchange` refuses on.
  // DORMANT, not retired: implemented below and dispatched by NOTHING.
  OPT_NODIV = 1 << 8,
  // Data-as-flag: no handshake at all, the generation tag rides in the upper half
  // of every 64-bit payload word. PUSH only, 2x the receive window, excludes HIER.
  OPT_LAMPORT = 1 << 9,
};

// Every bit this revision can run; anything else is a LOUD error in `k5_exchange`,
// never a silent fallback. A supported bit no `case` accepts would die anonymously.
constexpr int kOptSupported = OPT_NOFENCE | OPT_HIER | OPT_LAMPORT;
// `kRetiredOpts` was dispatched and taken out again; `kDormantOpts` never was.
constexpr int kRetiredOpts = OPT_SELF_ELIDE;
constexpr int kDormantOpts = OPT_NODIV;
constexpr int kUndispatchedOpts = kRetiredOpts | kDormantOpts;
static_assert((kRetiredOpts & kDormantOpts) == 0,
              "a bit is either retired (it was dispatched and lost) or dormant "
              "(it was never dispatched here); it cannot be both, and the "
              "refusal message has to pick one story");
static_assert((kUndispatchedOpts & kOptSupported) == 0,
              "an un-dispatched bit must not also be supported: it would pass "
              "the bit check and fail as an anonymous uninstantiated mask");
// Bits that change the SHAPE of the receive window, i.e. the allocation's size.
constexpr int kOptLayout = OPT_LAMPORT;
// `OPT_BCAST` is not landed here, so `k5_slot_floats` refuses to size for it.
constexpr int kOptLayoutReserved = (1 << 6);
// A layout bit that is not supported is a shape this file can size but not run.
static_assert((kOptLayout & kOptSupported) == kOptLayout,
              "every kOptLayout bit must also be in kOptSupported");
static_assert((kOptLayout & kOptLayoutReserved) == 0,
              "a bit cannot be both modelled and reserved");
// An un-dispatched layout bit would size a window no dispatch case can consume.
static_assert((kOptLayout & kUndispatchedOpts) == 0,
              "a layout bit cannot be un-dispatched: the window shape would "
              "outlive its dispatch");

// Peer addresses travel BY VALUE: a dependent load before the first peer access is
// pure critical path. `__grid_constant__` or the compiler stack-copies it to index.
struct PeerPtrs {
  // C4: the payload plane is int32. The two words of every 8-byte record are
  // BITCAST -- raw FP32 score bits, then the int32 block id -- so the pointer
  // type says what the memory IS, and nothing on this path converts.
  int32_t* buf[kMaxWorld];
  uint32_t* ctl[kMaxWorld];
};

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_relaxed_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// OPT_LAMPORT's 64-bit tagged word. These are single-copy atomic because they are
// MORALLY STRONG under PTX ISA 8.10.3: strong (not `.weak`), same scope, same
// width, naturally aligned -- ALL FOUR, not just the alignment. A plain or
// `volatile` 64-bit load keeps alignment and width, silently leaves 8.10.3 and
// reintroduces tearing: THE SCOPE QUALIFIER IS LOAD-BEARING. Inline PTX, not a
// `uint64_t` store, so two cannot fuse into one non-atomic `STG.128`.
__device__ __forceinline__ uint64_t ld_relaxed_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void st_relaxed_sys_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.sys.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ long ctl_off(int kind, int slot, int slots, int b,
                                        int nblocks, int s, int world) {
  return ((((long)kind * slots + slot) * nblocks + b) * world + s) *
         kFlagStride;
}

// Spin until *p is at or past `target`; false on timeout, which the caller turns
// into an error flag -- a hang inside a cudagraph replay is far worse to debug.
__device__ __forceinline__ bool spin_until(const uint32_t* p, uint32_t target,
                                           long long spin_cycles) {
  uint32_t v = ld_acquire_sys(p);
  if ((int32_t)(v - target) >= 0) return true;
  const long long t0 = clock64();
  int it = 0;
  for (;;) {
    v = ld_acquire_sys(p);
    if ((int32_t)(v - target) >= 0) return true;
    if (((++it) & 0xff) == 0 && (clock64() - t0) > spin_cycles) return false;
  }
}

// The derived generation, `own_generation + 1`. Generation 0 is POISON under
// Lamport -- a zero tag is what an unwritten word looks like -- so on wrap this
// goes to 1, not 0. `seq` is never reset (CONTRACT C9).
template <int OPTS>
__device__ __forceinline__ uint32_t derived_seq(uint32_t prev) {
  const uint32_t next = prev + 1u;
  if constexpr ((OPTS & OPT_LAMPORT) != 0) return (next == 0u) ? 1u : next;
  return next;
}

// Every `err` exit fills this block's own output rows with block id 0 -- not -1,
// which a consumer would dereference as `bt_row - 1`. DECODE-safe only: a caller
// that may consume `out` on a PREFILL path must poll `err` before it does.
__device__ __forceinline__ void fill_failed_rows(int32_t* __restrict__ out,
                                                 int t0, int ntok, int Hl) {
  const long base = (long)t0 * Hl * icp::kTopK;
  const long n = (long)ntok * Hl * icp::kTopK;
  for (long i = threadIdx.x; i < n; i += blockDim.x) out[base + i] = 0;
}

// THE EXTENT AND THE CAPACITY ARE TWO NUMBERS, AND ONLY ONE OF THEM MAY MOVE.
//
// `T` is this launch's EXECUTION EXTENT -- how many token rows are published,
// polled and merged. `T_cap` is the ALLOCATION CAPACITY the symmetric window
// was built for, and it is the stride of every address below: the receive
// buffer is `[world][T_cap][Hl][K][2]` and one slot is
// `k5_slot_floats(T_cap, ...)` words. A launch at `T < T_cap` therefore touches
// exactly the PREFIX of the rows a launch at `T_cap` would, at exactly the same
// addresses -- which is what makes a short call bit-identical to a full one on
// the rows they share, and what lets one window serve every captured extent.
//
// `tpb` AND `nblocks` COME FROM THE CAPACITY TOO, NOT FROM THE EXTENT, and that
// is a safety property rather than a convenience:
//
//   * `nblocks` is the control block's stride (`ctl_off`), so if it moved with
//     the extent, block `b` of a short launch and block `b` of a full one would
//     read a DIFFERENT generation counter for the same rows.
//   * `tpb` fixes the row -> block map. Each block derives its generation as
//     `own_generation + 1` from its own counter, so a data word's tag sequence
//     is strictly increasing only while the same block always writes it. Let
//     the partition move with the extent and two independent counters write one
//     word; OPT_LAMPORT's tag compare is exact (`==`), so a stale tag that
//     happens to equal the current `seq` is accepted as arrived -- a silently
//     wrong answer, not a timeout. The handshake masks have the same hazard on
//     the publish flag, where the compare is `>=` and therefore even weaker.
//
// So the grid shrinks with the extent (`grid_blocks = ceil(T / tpb)` blocks are
// launched, and no block covers a row past `T`) while every stride stays where
// the allocation put it. `grid_blocks` is passed separately because OPT_HIER's
// arrival counter counts the blocks that are actually RESIDENT, and that is the
// one place the live grid -- not the capacity -- is the right number.
template <int KPT, bool PUSH, int OPTS>
__global__ __launch_bounds__(kThreads) void k5_kernel(
    const int32_t* __restrict__ local_cand,
    const __grid_constant__ PeerPtrs peers,
    int32_t* __restrict__ out,
    const int32_t* __restrict__ forced,
    const int32_t* __restrict__ n_ordinary,
    int32_t* __restrict__ err, const uint32_t* __restrict__ seq_ptr, int rank,
    int world, int T, int T_cap, int Hg, int Hl, int head_offset, int slot,
    int slots, int nblocks, int grid_blocks, int tpb, long long slot_floats,
    int do_publish, int acquire_on, int use_ack, long long spin_cycles) {
  constexpr int K = icp::kTopK;
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int t0 = b * tpb;
  int ntok = T - t0;
  if (ntok <= 0) return;
  if (ntok > tpb) ntok = tpb;

  uint32_t* const myctl = peers.ctl[rank];
  // OPT_HIER publishes one flag per grid, so every block acquires on flag block 0.
  const int fb = (OPTS & OPT_HIER) ? 0 : b;

  __shared__ uint32_t s_seq;
  __shared__ uint32_t s_prev;
  __shared__ int s_bad;

  if (tid == 0) {
    // A workspace that has already failed is DEAD; without this every remaining
    // exchange of the captured step spins its full `spin_cycles`. `s_bad` is reused
    // rather than adding a `__shared__`: the shared-memory footprint is part of what
    // the codegen custody check compares. This exit does not re-set `err`.
    s_bad = (ld_relaxed_sys(reinterpret_cast<const uint32_t*>(err)) != 0u);
    const uint32_t prev =
        ld_relaxed_sys(myctl + ctl_off(2, slot, slots, b, nblocks, 0, world));
    s_prev = prev;
    s_seq = (seq_ptr != nullptr) ? ld_relaxed_sys(seq_ptr) : derived_seq<OPTS>(prev);
  }
  __syncthreads();
  // Rides the `__syncthreads()` above -- no extra barrier, no extra state.
  if (s_bad) {
    fill_failed_rows(out, t0, ntok, Hl);
    return;
  }
  const uint32_t seq = s_seq;
  const uint32_t prev = s_prev;

  // C3 row metadata, checked against the contract's own formula before this
  // block publishes anything. `n_ordinary` is redundant -- `min(15, f)` is
  // derivable from `f` -- and it is carried so that a caller deriving its count
  // from a batch-wide sequence length instead of the row's own query position
  // is caught, which is the single failure mode the refined design warns about
  // twice.
  //
  // The disagreement is refused rather than clamped because it is not a tuning
  // question: with `forced >= 0` and `n_ordinary = 16`, pass 2 of the merge
  // seeds `pos` at 1 for every winner above the forced id and the sixteenth
  // winner then writes `out[16]` -- one int32 into the NEXT row.
  //
  // K2 reports this through its C5 status word. K5 has none and reports it
  // through the `err` latch it already has, which renders the failed rows as
  // block id 0 rather than as K2's all--1 row. The two renderings diverge
  // deliberately: this kernel's consumer takes its trip count from `kv_len` and
  // never from this tensor, so it dereferences an all--1 row entry by entry.
  //
  // This rides section 1's barrier and section 1's `err` latch -- no extra
  // `__syncthreads()` on a kernel whose whole point is latency. Every write
  // here stores a 1 into a word that is 0 on every thread that reaches this
  // line, so the threads cannot disagree.
  if (forced != nullptr) {
    for (int i = tid; i < ntok; i += kThreads) {
      const int32_t f = forced[t0 + i];
      const int expect =
          (f < 0) ? 0 : (f < icp::kTopK - 1 ? f : icp::kTopK - 1);
      if (n_ordinary[t0 + i] != expect) s_bad = 1;
    }
  }

  // Generation 0 is poison. Unlike the derived path an explicit `seq` can be 0
  // against a FRESHLY ZEROED window, where every tag really is 0. NOT subsumed by
  // the exact tag compare, which only refuses a tag from a LATER generation.
  if ((OPTS & OPT_LAMPORT) && acquire_on && seq == 0u) {
    if (tid == 0) atomicExch(err, 1);
    fill_failed_rows(out, t0, ntok, Hl);
    return;
  }

  // 1. Do not overwrite a generation a peer has not consumed. The ack (kind 1) is
  // NOT deleted by OPT_LAMPORT, which deletes only the publish flag (kind 0).
  // Peers push their ack into OUR control block, so this is a local read.
  if (use_ack && tid < world) {
    const uint32_t* a =
        myctl + ctl_off(1, slot, slots, b, nblocks, tid, world);
    if (!spin_until(a, prev, spin_cycles)) s_bad = 1;
  }
  __syncthreads();
  if (s_bad) {
    if (tid == 0) atomicExch(err, 1);
    fill_failed_rows(out, t0, ntok, Hl);
    return;
  }

  // 2. Publish.
  if (do_publish) {
    if constexpr (PUSH && (OPTS & OPT_LAMPORT) != 0) {
      // Data-as-flag: each payload word rides in its own naturally aligned 64-bit
      // word tagged with `seq`, so a peer's plane is twice the ordinary size.
      const uint64_t tag = (uint64_t)seq << 32;
      const int per_tok = Hl * K;  // candidates per token
      const int nunit = ntok * per_tok;
      // OPT_NODIV folds to (0, 0, 1) at every dispatched mask. The `if` below is a
      // plain `if`, not `if constexpr`, so the dormant arm is still type-checked.
      const int first_tok = (OPTS & OPT_NODIV) ? (tid / per_tok) : 0;
      const int elem_in_tok =
          (OPTS & OPT_NODIV) ? (tid - first_tok * per_tok) : 0;
      const int tok_stride = (OPTS & OPT_NODIV) ? (kThreads / per_tok) : 1;
      for (int p = 0; p < world; ++p) {
        // Our own plane is data the merge can read out of `local_cand`.
        if ((OPTS & OPT_SELF_ELIDE) && p == rank) continue;
        uint64_t* dst = reinterpret_cast<uint64_t*>(
            peers.buf[p] + (long)slot * slot_floats +
            ((long)rank * T_cap + t0) * Hl * K * 4);
        const int32_t* sp = local_cand + (long)p * Hl * K * 2;
        if (OPTS & OPT_NODIV) {
          // Each thread owns one element-within-token and walks the tokens.
          for (int lt = first_tok; lt < ntok; lt += tok_stride) {
            const int2 pr = *reinterpret_cast<const int2*>(
                sp + (long)(t0 + lt) * Hg * K * 2 + elem_in_tok * 2);
            uint64_t* d = dst + 2L * ((long)lt * per_tok + elem_in_tok);
            st_relaxed_sys_u64(d, tag | (uint64_t)(uint32_t)pr.x);
            st_relaxed_sys_u64(d + 1, tag | (uint64_t)(uint32_t)pr.y);
          }
        } else {
          for (int i = tid; i < nunit; i += kThreads) {
            const int lt = i / per_tok;
            const int r = i - lt * per_tok;
            const int2 pr = *reinterpret_cast<const int2*>(
                sp + (long)(t0 + lt) * Hg * K * 2 + r * 2);
            uint64_t* d = dst + 2L * i;
            st_relaxed_sys_u64(d, tag | (uint64_t)(uint32_t)pr.x);
            st_relaxed_sys_u64(d + 1, tag | (uint64_t)(uint32_t)pr.y);
          }
        }
      }
    } else if (PUSH) {
      // Peer p receives [world][T][Hl][K][2]; we fill its [rank] plane for our
      // tokens with our heads [p*Hl, (p+1)*Hl) (C7: the head axis is rank-major).
      const int per_tok = Hl * K * 2 / 4;  // float4 per token
      const int nvec = ntok * per_tok;
      // OPT_NODIV: with stride `kThreads` and `per_tok | kThreads` the remainder is
      // loop-invariant, so one division per thread does the whole loop.
      const int first_tok = (OPTS & OPT_NODIV) ? (tid / per_tok) : 0;
      const int elem_in_tok =
          (OPTS & OPT_NODIV) ? (tid - first_tok * per_tok) : 0;
      const int tok_stride = (OPTS & OPT_NODIV) ? (kThreads / per_tok) : 1;
      for (int p = 0; p < world; ++p) {
        int4* dst = reinterpret_cast<int4*>(
            peers.buf[p] + (long)slot * slot_floats +
            ((long)rank * T_cap + t0) * Hl * K * 2);
        if (OPTS & OPT_NODIV) {
          for (int lt = first_tok; lt < ntok; lt += tok_stride) {
            const int4* src = reinterpret_cast<const int4*>(
                local_cand + (((long)(t0 + lt) * Hg + (long)p * Hl) * K * 2));
            dst[(long)lt * per_tok + elem_in_tok] = src[elem_in_tok];
          }
        } else {
          for (int i = tid; i < nvec; i += kThreads) {
            const int lt = i / per_tok;
            const int r = i - lt * per_tok;
            const int4* src = reinterpret_cast<const int4*>(
                local_cand + (((long)(t0 + lt) * Hg + (long)p * Hl) * K * 2));
            dst[i] = src[r];
          }
        }
      }
    } else {
      const long nvec = (long)ntok * Hg * K * 2 / 4;
      const int4* src = reinterpret_cast<const int4*>(
          local_cand + (long)t0 * Hg * K * 2);
      int4* dst = reinterpret_cast<int4*>(peers.buf[rank] +
                                              (long)slot * slot_floats +
                                              (long)t0 * Hg * K * 2);
      for (long i = tid; i < nvec; i += kThreads) dst[i] = src[i];
    }
  }

  // 3. Release. `__syncthreads()` first, then a system fence on the RELEASING
  // threads only: fences are cumulative, so the barrier orders the others' stores.
  // OPT_LAMPORT skips sections 3 and 4, but the `__syncthreads()` below STAYS --
  // it stops a warp in the merge polling a word another warp has not stored yet.
  __syncthreads();
  if constexpr ((OPTS & OPT_LAMPORT) == 0) {
  if (PUSH) {
    // `if constexpr`, not a plain `if`: a discarded `if` still instantiates its body,
    // so the `__shared__` below would move the footprint the custody check compares.
    if constexpr ((OPTS & OPT_HIER) != 0) {
      // `atomicInc(p, n-1)` wraps at n, so the counter needs no reset; it is kind 3,
      // which is why `kCtlKinds` is 4. A single-block grid degenerates both.
      __shared__ int s_last;
      // `grid_blocks`, not `nblocks`: the counter counts the blocks that are
      // RESIDENT in this launch, and `atomicInc(p, n-1)` wraps at exactly that
      // number so it needs no reset. `nblocks` stays the ctl_off stride -- the
      // capacity geometry -- so the counter's ADDRESS does not move with the
      // extent while its modulus does.
      const bool solo = (grid_blocks == 1);
      if (!solo && tid == 0) {
        __threadfence();  // this block's stores, device-visible, before arrival
        unsigned* ctr = reinterpret_cast<unsigned*>(
            myctl + ctl_off(3, slot, slots, 0, nblocks, 0, world));
        const unsigned old = atomicInc(ctr, (unsigned)(grid_blocks - 1));
        s_last = (old == (unsigned)(grid_blocks - 1)) ? 1 : 0;
      }
      if (!solo) __syncthreads();
      if (solo || s_last) {
        // With more than one block the fence STAYS whatever OPT_NOFENCE says: this
        // thread releases OTHER blocks' writes, which missed this block's barrier.
        const bool need_fence = !(solo && (OPTS & OPT_NOFENCE));
        if (tid < world) {
          if (need_fence) __threadfence_system();
          // Flag block 0: OPT_HIER publishes one flag for the whole grid.
          st_release_sys(
              peers.ctl[tid] + ctl_off(0, slot, slots, 0, nblocks, rank, world),
              seq);
        }
      }
    } else {
      // Flag lands in the consumer's control block, indexed by *our* rank.
      if (tid < world) {
        if (!(OPTS & OPT_NOFENCE)) __threadfence_system();
        st_release_sys(
            peers.ctl[tid] + ctl_off(0, slot, slots, b, nblocks, rank, world), seq);
      }
    }
  } else {
    if (tid == 0) {
      if (!(OPTS & OPT_NOFENCE)) __threadfence_system();
      st_release_sys(myctl + ctl_off(0, slot, slots, b, nblocks, rank, world),
                     seq);
    }
  }

  // 4. Acquire.
  if (acquire_on && tid < world) {
    const uint32_t* f =
        PUSH ? (myctl + ctl_off(0, slot, slots, fb, nblocks, tid, world))
             : (peers.ctl[tid] + ctl_off(0, slot, slots, fb, nblocks, tid, world));
    if (!spin_until(f, seq, spin_cycles)) s_bad = 1;
  }
  __syncthreads();
  if (s_bad) {
    if (tid == 0) atomicExch(err, 1);
    fill_failed_rows(out, t0, ntok, Hl);
    return;
  }
  }  // !OPT_LAMPORT

  // 5. Read + merge.
  const int32_t* const mybuf = peers.buf[rank] + (long)slot * slot_floats;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int n = world * K;
  // Sticky and warp-uniform: later rows skip the poll instead of paying it again.
  bool lam_bad = false;
  for (int w = warp; w < ntok * Hl; w += kWarpsPerBlock) {
    const int lt = w / Hl;
    const int hl = w - lt * Hl;
    const int t = t0 + lt;

    // C3's row metadata, warp-uniform because one warp owns one (token, head)
    // row. Indexed by `t` and never by `hl` or by the warp's own row number
    // `w`: forcing is a function of the query POSITION, so a per-head plane
    // could express a contradiction and `w` would give the second head of one
    // token the metadata of the next token. Already validated against C3's
    // formula at entry, so this is a load and nothing else.
    int32_t forced_block = icp::kNoForcedBlock;
    int n_ord = icp::kTopK;
    if (forced != nullptr) {
      forced_block = forced[t];
      n_ord = n_ordinary[t];
    }

    uint64_t key[KPT];
    int32_t gid[KPT];
    // Identical (lane, i) -> (c, k) mapping to icp::load_candidates.
#pragma unroll
    for (int i = 0; i < KPT; ++i) {
      const int idx = lane + i * 32;
      if constexpr ((OPTS & OPT_LAMPORT) != 0) {
        // Every branch below is WARP-UNIFORM: `__all_sync` with the full mask from
        // a divergent branch is undefined.
        const bool active = (idx < n);
        const int c = active ? (idx / K) : 0;
        const int k = active ? (idx - c * K) : 0;
        const bool self = (OPTS & OPT_SELF_ELIDE) && do_publish && (c == rank);
        const uint64_t* p2 = nullptr;
        uint32_t sw = 0u;
        uint32_t gw = 0xffffffffu;  // gid = -1, contributing nothing
        if (active && self) {
          // This rank's own candidates were never staged, so they carry no tag.
          const int32_t* base =
              local_cand + (((((long)t * Hg + head_offset + hl) * K) + k) * 2);
          sw = (uint32_t)base[0];
          gw = (uint32_t)base[1];
        } else if (active) {
          p2 = reinterpret_cast<const uint64_t*>(mybuf) +
               (((((long)c * T_cap + t) * Hl + hl) * K) + k) * 2;
        }
        const bool poll = active && !self;
        if (acquire_on && !lam_bad) {
          const long long q0 = clock64();
          for (;;) {
            bool ready = true;
            if (poll) {
              const uint64_t w0 = ld_relaxed_sys_u64(p2);
              const uint64_t w1 = ld_relaxed_sys_u64(p2 + 1);
              // The tag compare is EXACT (`==`, not `>= 0`): `>=` would accept a
              // tag from a LATER generation, so a producer a generation ahead of
              // this consumer would be a silently wrong answer; `==` refuses it.
              ready = ((int32_t)((uint32_t)(w0 >> 32) - seq) == 0) &&
                      ((int32_t)((uint32_t)(w1 >> 32) - seq) == 0);
              if (ready) {
                sw = (uint32_t)w0;
                gw = (uint32_t)w1;
              }
            }
            if (__all_sync(icp::kFullMask, ready)) break;
            // The timeout is decided by vote too: `clock64()` can differ across
            // lanes, and a per-lane break would diverge at the next `__all_sync`.
            if (__any_sync(icp::kFullMask, (clock64() - q0) > spin_cycles)) {
              lam_bad = true;
              break;
            }
          }
        } else if (poll && !acquire_on) {
          // `&& !acquire_on` is LOAD-BEARING: `lam_bad` is sticky, so without this
          // clause every row after a warp's first timeout fell through to here --
          // the negative control -- in a production mask with `acquire_on = 1`.
          // `acquire_on = 0` IS the Lamport negative control.
          sw = (uint32_t)ld_relaxed_sys_u64(p2);
          gw = (uint32_t)ld_relaxed_sys_u64(p2 + 1);
        }
        const int32_t g = active ? (int32_t)gw : -1;
        key[i] = active ? icp::canonical_key(__uint_as_float(sw), g) : 0ull;
        gid[i] = g;
      } else if (idx < n) {
        const int c = idx / K;
        const int k = idx - c * K;
        const int32_t* base;
        if (PUSH) {
          base = mybuf + (((((long)c * T_cap + t) * Hl + hl) * K) + k) * 2;
        } else {
          base = peers.buf[c] + (long)slot * slot_floats +
                 (((((long)t * Hg + head_offset + hl) * K) + k) * 2);
        }
        // C4 bitcast, not conversion: word 0 holds the FP32 score's raw bits.
        const float score = __int_as_float(base[0]);
        const int32_t g = base[1];
        key[i] = icp::canonical_key(score, g);
        gid[i] = g;
      } else {
        key[i] = 0ull;
        gid[i] = -1;
      }
    }
    // The poll is inside the merge loop, so a timeout is found after this warp holds
    // garbage keys. `lam_bad` is warp-uniform, so this call is not divergent.
    if (!((OPTS & OPT_LAMPORT) && lam_bad))
      icp::warp_merge_topk16<KPT>(key, gid, lane, forced_block, n_ord,
                                  out + (long)(t * Hl + hl) * icp::kTopK);
  }

  if constexpr ((OPTS & OPT_LAMPORT) != 0) {
    // The Lamport timeout is per warp, so collect it before the block decides.
    if (lam_bad) s_bad = 1;
    __syncthreads();
    if (s_bad) {
      if (tid == 0) atomicExch(err, 1);
      fill_failed_rows(out, t0, ntok, Hl);
      return;
    }
  }

  // 6. Ack + record our generation. NOT skipped under OPT_LAMPORT; see 1.
  __syncthreads();
  if (use_ack && tid < world) {
    // A plain release, never waited on here, so it is off the critical path.
    st_release_sys(
        peers.ctl[tid] + ctl_off(1, slot, slots, b, nblocks, rank, world), seq);
  }
  if (tid == 0)
    st_relaxed_sys(myctl + ctl_off(2, slot, slots, b, nblocks, 0, world), seq);
}

int plan_tpb(int T, int Hl, int max_blocks) {
  int tpb = kWarpsPerBlock / Hl;   // one warp per (token, local head)
  if (tpb < 1) tpb = 1;
  if (max_blocks < 1) max_blocks = 1;
  const int need = (T + max_blocks - 1) / max_blocks;
  if (need > tpb) tpb = need;      // widen the block rather than add blocks
  return tpb;
}

}  // namespace

std::vector<int64_t> k5_plan(int64_t T, int64_t Hl, int64_t max_blocks) {
  const int tpb = plan_tpb((int)T, (int)Hl, (int)max_blocks);
  const int nblocks = (int)((T + tpb - 1) / tpb);
  return {tpb, nblocks};
}

int64_t k5_ctl_words(int64_t slots, int64_t nblocks, int64_t world) {
  return kCtlKinds * slots * nblocks * world * kFlagStride;
}

// The option bits by name: the NAME->VALUE table, not the supported set, so
// retired and dormant bits stay in it and a request for one is refused BY NAME
// instead of dying as an unknown option string. `exchange.py` mirrors this.
std::vector<std::pair<std::string, int64_t>> k5_opts() {
  return {{"self_elide", OPT_SELF_ELIDE},
          {"nofence", OPT_NOFENCE},
          {"hier", OPT_HIER},
          {"nodiv", OPT_NODIV},
          {"lamport", OPT_LAMPORT}};
}

// int32 words per slot the receive buffer needs -- the single source of truth
// for both the kernel's slot stride and the allocator's size. The name says
// "floats" for ABI reasons: C4 made the carrier int32, but the word COUNT and
// the byte size are unchanged and this is an exported pybind symbol, so
// renaming it would be an ABI break for a quantity that did not move.
// `w = 2` ordinarily, `w = 4` under OPT_LAMPORT. A window built at w = 2 and
// exchanged under
// OPT_LAMPORT writes PAST THE END OF A PEER's symmetric window, which no allocator
// sees. Mask and allocation are one decision, fixed at construction.
int64_t k5_slot_floats(int64_t T, int64_t Hg, int64_t world, int64_t opts) {
  TORCH_CHECK(!(opts & kOptLayoutReserved),
              "k5_slot_floats: opts ", opts, " sets a reserved layout bit "
              "(OPT_BCAST = 64 makes a slot `world` times larger). It is not "
              "implemented in this revision, so this function does not know "
              "the size of the window it needs. Land the sizing in here and in "
              "IcpExchange before landing the bit.");
  (void)world;
  const int64_t words = (opts & OPT_LAMPORT) ? 4 : 2;
  return T * Hg * icp::kTopK * words;
}

void k5_exchange(at::Tensor local_cand, at::Tensor out,
                 std::vector<int64_t> peer_buf_ptrs,
                 std::vector<int64_t> peer_ctl_ptrs, at::Tensor err,
                 c10::optional<at::Tensor> seq, int64_t rank, int64_t world,
                 int64_t head_offset, int64_t slot, int64_t slots,
                 int64_t mode, int64_t do_publish, int64_t acquire_on,
                 int64_t use_ack, int64_t spin_cycles, int64_t max_blocks,
                 int64_t opts, int64_t slot_capacity_floats,
                 c10::optional<at::Tensor> forced,
                 c10::optional<at::Tensor> n_ordinary,
                 int64_t tokens_capacity) {
  TORCH_CHECK(local_cand.is_cuda() && out.is_cuda() && err.is_cuda(),
              "tensors must be CUDA");
  // C4: the carrier is int32. The wire CONTENT is unchanged -- this kernel
  // already moved (fp32 score bits, int32 id) as opaque 4-byte words -- but a
  // float32 tensor here is the PRE-C4 producer layout and must be refused, not
  // reinterpreted. See the carrier note at the top of this file.
  TORCH_CHECK(local_cand.scalar_type() == at::kInt,
              "cand must be int32 (refined-icp-v1 C4 carrier words); got ",
              local_cand.scalar_type());
  TORCH_CHECK(out.scalar_type() == at::kInt, "out must be int32");
  TORCH_CHECK(err.scalar_type() == at::kInt && err.numel() >= 1,
              "err must be an int32 tensor");
  TORCH_CHECK(local_cand.is_contiguous() && out.is_contiguous(),
              "must be contiguous");
  TORCH_CHECK(local_cand.dim() == 4, "cand must be [T, H_group, K, 2]");
  TORCH_CHECK(out.dim() == 3, "out must be [T, H_local, 16]");
  TORCH_CHECK((int64_t)peer_buf_ptrs.size() == world &&
                  (int64_t)peer_ctl_ptrs.size() == world,
              "need one buffer and one control pointer per rank");
  TORCH_CHECK(world >= 1 && world <= kMaxWorld, "world outside [1, 8]");
  PeerPtrs peers{};
  for (int i = 0; i < (int)world; ++i) {
    TORCH_CHECK(peer_buf_ptrs[i] && peer_ctl_ptrs[i], "null peer pointer ", i);
    peers.buf[i] = reinterpret_cast<int32_t*>(peer_buf_ptrs[i]);
    peers.ctl[i] = reinterpret_cast<uint32_t*>(peer_ctl_ptrs[i]);
  }

  const int T = (int)local_cand.size(0);
  const int Hg = (int)local_cand.size(1);
  const int K = (int)local_cand.size(2);
  TORCH_CHECK(local_cand.size(3) == 2, "last dim must be 2 (score, id)");
  TORCH_CHECK(K == icp::kTopK, "K must be 16");
  const int Hl = (int)out.size(1);
  TORCH_CHECK(out.size(0) == T, "out token count must match cand");
  TORCH_CHECK(out.size(2) == icp::kTopK, "out last dim must be 16");
  TORCH_CHECK(Hg == (int)world * Hl, "H_group must be world * H_local (C7)");
  TORCH_CHECK(head_offset == rank * Hl,
              "C7 pins head_offset = icp_rank * H_local; got ", head_offset);
  TORCH_CHECK(slot >= 0 && slot < slots, "slot outside [0, slots)");
  TORCH_CHECK((int)world * K <= icp::kMaxCandidates, "too many candidates");
  // An unsupported bit is a LOUD error, never a silent fall back to the control: a
  // fallback would time the baseline under the optimisation's name. Sign first, so
  // a negative mask cannot reach the by-name refusals below.
  TORCH_CHECK(opts >= 0, "k5: opts must be non-negative; got ", opts);
  // Un-dispatched bits are refused BY NAME, ahead of the generic bit check.
  TORCH_CHECK(!(opts & OPT_SELF_ELIDE),
              "k5: opts ", opts, " sets OPT_SELF_ELIDE = 1, which is RETIRED "
              "from the dispatch together with mask 513 "
              "(OPT_LAMPORT | OPT_SELF_ELIDE), its only instantiation. It was "
              "not removed for being wrong: it was measured, and the "
              "difference was inside the noise of the arm it was measured "
              "against, for the cost of one more instantiation and one more "
              "arm on every gate. The BIT VALUE 1 stays reserved forever and "
              "the device code below still implements it, so an old mask "
              "number still means the same optimisation. To bring it back: add "
              "OPT_SELF_ELIDE to kOptSupported, take it out of kRetiredOpts, "
              "add `case OPT_LAMPORT | OPT_SELF_ELIDE` to the PUSH dispatch "
              "switch, delete this refusal, and add 513 to "
              "icp_kernels/exchange.py's INSTANTIATED_MASKS (upstream: "
              "icp_comm.py's K5_OPTS_INSTANTIATED). Re-price it on the harness "
              "kernel, which is not in this repository but still builds it, "
              "before you do.");
  // OPT_NODIV is DORMANT rather than retired, and the message has to say which:
  // nothing here has priced it against the arms this file does dispatch.
  TORCH_CHECK(!(opts & OPT_NODIV),
              "k5: opts ", opts, " sets OPT_NODIV = 256, which is implemented "
              "in the device code below and instantiated by no dispatch case. "
              "It is not a removed optimisation: it hoists the per-element "
              "`i / per_tok` out of both publish loops into one division per "
              "thread, and what it buys under OPT_LAMPORT is register pressure "
              "rather than division. On the handshake path it is a loss, "
              "because plan_tpb widens the block until the grid reaches "
              "kDefaultMaxBlocks and there is then no per-element division "
              "left to remove. It is DORMANT rather than retired: nothing in "
              "this repository has priced it against the masks this file does "
              "dispatch, and it has not been built on the architecture it "
              "would ship on, so instantiating it would ship a mask this "
              "file's own evidence has never covered. To land it: add "
              "OPT_NODIV to "
              "kOptSupported, take it out of kDormantOpts, add `case "
              "OPT_LAMPORT | OPT_NODIV` to the PUSH dispatch switch, delete "
              "this refusal, add 768 to icp_kernels/exchange.py's "
              "INSTANTIATED_MASKS, and gate mask 768 bit-identical against mask "
              "512 on the arch you intend to ship. Note the "
              "precondition below, which this file already enforces: "
              "per_tok | kThreads.");
  TORCH_CHECK(opts >= 0 && (opts & ~(int64_t)kOptSupported) == 0,
              "k5: opts ", opts, " sets a bit this revision does not "
              "implement. Supported: OPT_NOFENCE = 4, OPT_HIER = 8, "
              "OPT_LAMPORT = 512 (mask ",
              (int64_t)kOptSupported, "); OPT_SELF_ELIDE = 1 is implemented "
              "but retired and OPT_NODIV = 256 is implemented but dormant "
              "(together, mask ", (int64_t)kUndispatchedOpts, ") -- both are "
              "refused above with their own reasons. The other values are "
              "reserved for the harness bits of the same numbering and are not "
              "silently ignored.");
  // Refused here rather than at the dispatch, ahead of any mask-vs-mode reasoning.
  TORCH_CHECK(mode != 0,
              "k5: mode = 0 is PULL, which was REMOVED from this kernel's "
              "dispatch. Only PUSH (mode = 1) is instantiated. PULL's "
              "critical path is two peer round trips against PUSH's one, and "
              "the harness measured PUSH ahead at EVERY point, which is why "
              "its own PULL arms are "
              "marked `superseded`. Nothing "
              "reached it in either tree: IcpExchange here hardcodes MODE_PUSH "
              "and upstream's IcpSymmExchange takes `mode: int = K5_MODE_PUSH` "
              "with no caller overriding it, and no gate drives PULL. It was "
              "also the file's only spilling "
              "instantiation. The `mode` argument and the `PUSH` "
              "template parameter are both still here, so a restoration is "
              "this check plus one mask-0 dispatch case at PUSH = false, in an "
              "`else` branch -- but re-measure it on the harness kernel first, "
              "which still builds every PULL mask. (Spelling the macro "
              "call out here would make the codegen gate's dispatch parser "
              "read this string literal as a declared instantiation, so it is "
              "deliberately described rather than quoted.)");
  // The two PUSH-only checks below cannot fire while `mode != 0` is refused above.
  // They stay because they are exactly the guards a PULL restoration has to keep.
  if (opts & OPT_HIER) {
    TORCH_CHECK(mode != 0,
                "OPT_HIER is PUSH-only: in PULL the flag is released into the "
                "producer's own control block by thread 0 of each block, so "
                "there is no per-block system store to collapse");
  }
  if (opts & OPT_LAMPORT) {
    TORCH_CHECK(mode != 0,
                "OPT_LAMPORT is PUSH-only: the tags live in the receive "
                "buffer, which PULL does not have");
    TORCH_CHECK(do_publish,
                "OPT_LAMPORT has no zero-copy form: the producer has to write "
                "the tag alongside every payload word");
    TORCH_CHECK(!(opts & OPT_HIER),
                "OPT_LAMPORT removes the handshake entirely; OPT_HIER "
                "optimises a handshake that is no longer there and would be "
                "silently dead code in a measured mask");
    // Really OPT_NODIV's precondition, asserted here anyway: refusing an unmeasured
    // shape is the right direction for a mask whose failure mode is a silent write
    // into a peer.
    TORCH_CHECK(Hl >= 1 && Hl <= 16 && (16 % Hl) == 0,
                "OPT_LAMPORT is gated to H_local in {1, 2, 4, 8, 16} (got ",
                Hl, "). `per_tok = H_local * K = ", (int64_t)(Hl * K),
                "` must divide kThreads = ", (int64_t)kThreads,
                " for the OPT_NODIV follow-on to be equivalent to the division "
                "it removes. This is a refusal, not a fallback.");
    // The tagged store is two `st.relaxed.sys.u64` at `dst + 2*i`, so every plane
    // must start 8-byte aligned; K is the only thing that makes that true.
    TORCH_CHECK((K % 2) == 0,
                "OPT_LAMPORT needs an even K so every plane starts 8-byte "
                "aligned; K=", K);
  }
  // OPT_NODIV's precondition, and the ONE thing making the affine recurrence
  // equivalent to the division. Against `per_tok`: the two publish loops differ.
  if (opts & OPT_NODIV) {
    const int per_tok = (opts & OPT_LAMPORT) ? (Hl * K) : (Hl * K * 2 / 4);
    TORCH_CHECK(per_tok > 0 && per_tok <= kThreads &&
                    (kThreads % per_tok) == 0,
                "OPT_NODIV needs per_tok | kThreads (", (int64_t)kThreads,
                "); H_local=", Hl, " gives per_tok=", (int64_t)per_tok,
                ". This is a refusal, not a fallback: a silently divergent "
                "fallback would report the control's behaviour under this "
                "bit's name.");
  }

  // The ALLOCATION capacity, which every stride below is taken from. `0` means
  // "the caller did not say", i.e. the window was allocated for exactly the
  // rows it is being handed -- the behaviour of every call that predates this
  // argument, and identical to passing `T` explicitly.
  //
  // A larger extent than the window holds is REFUSED, never clamped: a clamp
  // would silently merge fewer rows than the caller asked for and leave the
  // tail of `out` holding whatever was there, which on a block-table load is a
  // wild page rather than a visible error. The refusal is also the only place
  // that can catch it -- the extent is a host integer and the window's size is
  // a host integer, so nothing downstream ever compares them again.
  TORCH_CHECK(tokens_capacity >= 0,
              "k5: tokens_capacity must be non-negative; got ", tokens_capacity);
  const int T_cap = tokens_capacity > 0 ? (int)tokens_capacity : T;
  TORCH_CHECK(T_cap >= T,
              "k5: the execution extent is ", T, " token rows but the "
              "symmetric window was allocated for ", T_cap,
              ". Symmetric memory cannot be resized and cannot be allocated "
              "during a cudagraph capture, so a longer extent is an error, not "
              "a re-allocation and not a clamp.");
  if (T == 0) return;
  // Capacity, not extent: `tpb` fixes the row -> block map and `nblocks` is the
  // control block's stride, and both must be properties of the WORKSPACE rather
  // than of the call. See the note above `k5_kernel` for what moving them would
  // cost -- an exact Lamport tag compare against a counter that changed hands.
  const int tpb = plan_tpb(T_cap, Hl, (int)max_blocks);
  const int nblocks = (T_cap + tpb - 1) / tpb;
  // Only the GRID shrinks with the extent. No block covers a row past `T`, so
  // rows in `[T, T_cap)` are neither published nor polled nor merged.
  const int grid_blocks = (T + tpb - 1) / tpb;
  // From the kernel's own layout helper, never a constant.
  const long long slot_floats =
      (long long)k5_slot_floats(T_cap, Hg, world, opts);
  // A receive layout larger than the window it was handed writes past the end of a
  // PEER's symmetric allocation, silently. Under OPT_LAMPORT the capacity is NOT
  // optional: the pybind default of 0 means "caller did not say" and skips this.
  TORCH_CHECK(!(opts & OPT_LAMPORT) || slot_capacity_floats > 0,
              "OPT_LAMPORT requires an explicit slot_capacity_floats > 0; got ",
              slot_capacity_floats,
              ". The Lamport receive window is 2x the ordinary layout, and "
              "without the capacity this call cannot tell whether the "
              "allocation it was handed was made for this mask. Allocate with "
              "the same mask you exchange with and pass its capacity.");
  TORCH_CHECK(slot_capacity_floats <= 0 || slot_capacity_floats >= slot_floats,
              "the symmetric window holds ", slot_capacity_floats,
              " fp32 per slot but opts ", opts, " needs ", slot_floats,
              "; the layout bits in this mask are ", (opts & kOptLayout),
              " (OPT_LAMPORT = 512 is 2x). Allocate with the same mask you "
              "exchange with.");
  if (opts & OPT_HIER) {
    // Every block must arrive before any block spins, so a bigger grid deadlocks.
    // The LIVE grid is what has to be resident; the capacity geometry bounds it.
    TORCH_CHECK(grid_blocks <= kDefaultMaxBlocks,
                "OPT_HIER requires a resident grid; nblocks=", grid_blocks);
    // OPT_HIER IS INCOMPATIBLE WITH A SHORT EXTENT, and this is a refusal
    // rather than a fallback because the failure is a race in both directions.
    //
    // The bit publishes ONE flag for the whole grid, carrying the generation of
    // whichever block arrives last, and every block acquires on it. That is
    // sound only while all the grid's blocks are on the SAME generation, which
    // holds only while every block runs on every launch: each block derives its
    // own `own_generation + 1` from its own counter, so a block skipped by a
    // short launch stays a generation behind. A later full launch on the same
    // slot then has blocks expecting different values from one flag -- and the
    // acquire compares `>=`, so the blocks below the published generation pass
    // instantly on somebody else's flag while the ones above it spin to their
    // timeout. Measured: this is exactly what mask 12 does under an 8-row call
    // followed by a 64-row one on one slot, and which of the two it does
    // depends on the block arrival order.
    //
    // The per-block handshake (mask 0) and the data-as-flag path (mask 512)
    // have no such coupling: their flag, or their tag, belongs to the same
    // block that derived the generation. Both are gated at short extents.
    TORCH_CHECK(T == T_cap,
                "OPT_HIER cannot run a short extent: the extent is ", T,
                " token rows against an allocation of ", T_cap,
                ". OPT_HIER publishes ONE release for the whole grid, so every "
                "block must be on the same generation, and a block that a "
                "short launch skips is left a generation behind. Use "
                "MaskPreset.LAMPORT (512) or the control mask 0 for a "
                "workspace that serves several extents, or launch this "
                "workspace at its full capacity.");
  }

  // --- the forced-row metadata (C3) ----------------------------------------
  // Both planes or neither, exactly as `k2_merge` requires: one alone is a
  // caller that thinks it is forcing and is not, which is the bug class C3
  // replaced. Omitting both selects the plain ordinary merge -- which is what
  // this kernel did unconditionally before these planes existed.
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

  const uint32_t* seq_ptr = nullptr;
  if (seq.has_value()) {
    TORCH_CHECK(seq->is_cuda() && seq->numel() >= 1, "seq must be CUDA");
    TORCH_CHECK(seq->scalar_type() == at::kInt || seq->scalar_type() == at::kUInt32,
                "seq must be int32/uint32");
    seq_ptr = reinterpret_cast<const uint32_t*>(seq->data_ptr());
  }

  auto stream = c10::cuda::getCurrentCUDAStream();
  const int32_t* cp = local_cand.data_ptr<int32_t>();
  int32_t* op = out.data_ptr<int32_t>();
  int32_t* ep = err.data_ptr<int32_t>();
  const int kpt = ((int)world * K + 31) / 32;

#define K5_LAUNCH(KPT, PUSH, OPTS)                                            \
  k5_kernel<KPT, PUSH, OPTS><<<grid_blocks, kThreads, 0, stream>>>(           \
      cp, peers, op, fp, np, ep, seq_ptr, (int)rank, (int)world, T, T_cap,    \
      Hg, Hl, (int)head_offset, (int)slot, (int)slots, nblocks, grid_blocks,  \
      tpb, slot_floats, (int)do_publish, (int)acquire_on, (int)use_ack,       \
      (long long)spin_cycles)

#define K5_KPT(PUSH, OPTS)                                                    \
  switch (kpt) {                                                              \
    case 1: K5_LAUNCH(1, PUSH, OPTS); break;                                  \
    case 2: K5_LAUNCH(2, PUSH, OPTS); break;                                  \
    case 4: K5_LAUNCH(4, PUSH, OPTS); break;                                  \
    default: TORCH_CHECK(false, "unsupported candidates-per-lane ", kpt);     \
  }

  // Three masks, PUSH only, nine specialisations. `exchange.py`'s
  // `INSTANTIATED_MASKS` is this table's Python mirror and must move with it.
  // An unlisted mask is a LOUD error, never a silent fall back to the control.
  switch (opts) {
    case 0: K5_KPT(true, 0); break;
    case OPT_NOFENCE | OPT_HIER: K5_KPT(true, OPT_NOFENCE | OPT_HIER); break;
    case OPT_LAMPORT: K5_KPT(true, OPT_LAMPORT); break;
    default:
      TORCH_CHECK(false,
                  "k5: opts mask ", opts, " is not instantiated. This revision "
                  "builds exactly 0 (control), ",
                  (int64_t)(OPT_NOFENCE | OPT_HIER),
                  " (OPT_NOFENCE | OPT_HIER) and ", (int64_t)OPT_LAMPORT,
                  " (OPT_LAMPORT). If you asked for 4 "
                  "(OPT_NOFENCE) or 8 (OPT_HIER) alone: both bits are still "
                  "implemented and both still ship -- inside mask 12. Their "
                  "single-bit instantiations were ABLATION arms, retired "
                  "because neither is best at every T on "
                  "its own while mask 12 is, and because mask 12 "
                  "is in turn beaten by mask 512 everywhere it was measured. "
                  "An arm that loses to an arm that loses is a "
                  "measurement, not a production option, and the measurement "
                  "instrument is the harness kernel, "
                  "which is not in this repository. "
                  "Re-price there; if it wins, add the `case` here and the mask "
                  "to icp_kernels/exchange.py's INSTANTIATED_MASKS. If you "
                  "asked for 768 (OPT_LAMPORT | OPT_NODIV): see the OPT_NODIV "
                  "refusal above -- the bit is implemented here and dormant.");
  }
#undef K5_KPT
#undef K5_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k5_exchange", &k5_exchange,
        "Fused symmetric-memory ICP candidate exchange + merge",
        pybind11::arg("local_cand"), pybind11::arg("out"),
        pybind11::arg("peer_buf_ptrs"), pybind11::arg("peer_ctl_ptrs"),
        pybind11::arg("err"), pybind11::arg("seq"), pybind11::arg("rank"),
        pybind11::arg("world"), pybind11::arg("head_offset"),
        pybind11::arg("slot") = 0, pybind11::arg("slots") = 2,
        pybind11::arg("mode") = 1, pybind11::arg("do_publish") = 1,
        pybind11::arg("acquire_on") = 1, pybind11::arg("use_ack") = 1,
        pybind11::arg("spin_cycles") = 2000000000LL,
        pybind11::arg("max_blocks") = kDefaultMaxBlocks,
        // `opts = 0` is the control arm, legal precisely because `case 0` is
        // instantiated above; `slot_capacity_floats = 0` is refused under LAMPORT.
        pybind11::arg("opts") = 0,
        pybind11::arg("slot_capacity_floats") = 0,
        // C3's planes go on the END, defaulted, rather than beside `out` where
        // `k2_merge` keeps them. This entry point is called POSITIONALLY --
        // `icp_kernels/exchange.py` passes every argument that way and so does
        // the upstream carrier -- so inserting into the middle would rebind
        // nineteen existing arguments to new meanings at every call site that
        // was not rebuilt against this header. Appending cannot: a caller that
        // does not force keeps compiling and keeps its exact behaviour, which
        // is the same additive rule the merge overloads in `merge_topk.cuh`
        // follow.
        pybind11::arg("forced") = c10::nullopt,
        pybind11::arg("n_ordinary") = c10::nullopt,
        // The ALLOCATION capacity, appended for the same reason C3's planes
        // were: this entry point is called positionally by both trees, so the
        // twenty-one arguments above must not move. `0` means "capacity ==
        // extent", which is what every caller written before this argument
        // existed meant, so their behaviour is unchanged bit for bit.
        pybind11::arg("tokens_capacity") = 0);
  m.def("k5_plan", &k5_plan, "(tpb, nblocks) for a given (T, H_local)",
        pybind11::arg("T"), pybind11::arg("H_local"),
        pybind11::arg("max_blocks") = kDefaultMaxBlocks);
  m.def("k5_ctl_words", &k5_ctl_words, "uint32 words of control block needed",
        pybind11::arg("slots"), pybind11::arg("nblocks"),
        pybind11::arg("world"));
  m.def("k5_slot_floats", &k5_slot_floats,
        "fp32 words per slot the receive buffer needs for a given opts mask",
        pybind11::arg("T"), pybind11::arg("H_group"), pybind11::arg("world"),
        pybind11::arg("opts") = 0);
  m.def("k5_opts", &k5_opts, "the option bit names and values");
}
