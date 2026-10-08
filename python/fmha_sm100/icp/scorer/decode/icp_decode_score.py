# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Prepared, graph-safe FP8 decode scoring for ``refined-icp-v1``.

The supported profile is TP=ICP2, physical/ranking page128, compact index fragment64,
four query heads and head dimension128. It accepts uniform Q1 through Q4 decode
and Eagle3 verification; prefill and mixed batches keep their existing route.

Call :func:`get_icp_decode_scorer` before CUDA graph capture. The returned object
launches the native scan once. One native compile serves a
``(query_len, rank, world_size, split_k)`` profile: the token window and request
range are runtime launch scalars (launch ABI 2), passed per call or bound once
with :meth:`_PreparedIcpDecodeScorer.bind`. A captured graph records the window
values of its capture, so a caller that captures should pass the graph's padded
extent. ABI version 2 writes only the live fragment prefix
of each active query row; it does not initialize the score or validity plane.
For request i and active row t, let U=max(0,min(seq_lens[i],positions[t]+1)) and
n=U//128 + (U%128 > rank*64). All four heads in columns [0,n) receive their
scores and valid=1. Every other cell is undefined and may retain stale values.
The selector owns these exact bounds and must not read outside them, including
the validity plane. ABI version 1's full rectangular completion is superseded.

Invocation performs only tensor metadata checks and the prepared scan launch.
It does not allocate CUDA tensors, compile, inspect device values, synchronize,
or import vLLM. No initializer is compiled or exposed by the factory.

Input K is already rebased to the local index component of each compound page.
Its outer stride is the whole compound-page pitch, not ``64 * 128``. Rank affects
logical positions only. Query offsets, lengths, positions and activity remain
live device tensors on every replay. The caller owns their consistency and must
provide a fixed request range covering the token window, including collapsed
zero-length padding requests. Output columns cover the ordinary table capacity.

``positions`` and ``active_rows`` must have **exactly** ``index_q``'s row
count. The compiled signature binds all three to one symbol, so a longer
retained plane is rejected at bind time; slice it, e.g. ``buf[:num_tokens]``.
A prefix view preserves ``data_ptr`` and is safe to build across graph capture.
The token window may extend past ``index_q``'s rows: the kernel bounds every row
by ``t < index_q.shape[0]``, so rows past the invocation are never read or
written.

``active_rows`` is ``uint8``, not ``bool``: any nonzero byte is live. A torch
``bool`` tensor is one byte per element, so ``tensor.view(torch.uint8)`` is the
zero-copy, same-``data_ptr`` way to supply it and is safe to build once and
retain across CUDA graph capture. The byte type is what the plane has ever had
-- CUTLASS lowers a 1-bit element type to an ``i8`` pointer regardless -- and
declaring it avoids a ``divisibility * width // 8`` byte-alignment computation
that underflows to zero for 1-bit dtypes. See ``_compile_profile``.
"""

from dataclasses import dataclass
from functools import cache


ICP_DECODE_SCORE_ABI_VERSION = 2
# 1: the window and request range were compile-time profile constants.
# 2: they are runtime launch scalars; one compile per (q, rank, world, split_k).
# 3: split_k is a runtime launch scalar too, chosen per request count; the
#    scan is a PDL launch with a K pre-issue ahead of its wait.
ICP_DECODE_SCORE_LAUNCH_ABI_VERSION = 3

#: Fallback resident CTAs per SM, used only when the device limits are unknown.
AUTO_SPLIT_CTAS_PER_SM = 6
#: Floor and ceiling of an automatic split (grid.y).
AUTO_SPLIT_MIN = 64
AUTO_SPLIT_MAX = 512
#: Fraction of one resident wave to target, and the split rounding step. The
#: v9 150K sweep's best split per batch (b1 512, b4 256, b8 128, b12 96, b16 64)
#: is ~1024 CTAs = 0.8 x 6 x 212; a full wave loses to its tail.
AUTO_SPLIT_WAVE_FRACTION = 0.8
AUTO_SPLIT_GRANULE = 32
#: K fragments in flight per CTA.
DEFAULT_NUM_STAGES = 4
#: Shared memory the driver reserves per CTA on SM100-class parts.
_SMEM_RESERVED_PER_CTA = 1024
#: IndexDecodeScoreKernel geometry: K page fragment, heads, head dim, compute warps.
_BLOCK_K, _HEADS, _HEAD_DIM, _COMPUTE_WARPS = 64, 4, 128, 2


def _align(value, alignment):
    return -(-value // alignment) * alignment


def decode_smem_bytes(query_len, num_stages):
    """Dynamic shared memory of one decode scorer CTA (its SmemAllocator order)."""
    max_dql = 2 if query_len <= 2 else 4
    block_q = _HEADS * max_dql
    size = num_stages * _BLOCK_K * _HEAD_DIM  # FP8 K stages
    size = _align(size, 1024) + block_q * _HEAD_DIM  # own Q tile
    size = _align(size, 8) + _align(block_q, 8) * _COMPUTE_WARPS * 4  # epilogue
    size += 8 * (2 * num_stages + 1)  # mbarriers
    # Every allocation may be padded to the 1024 B swizzle atom.
    return _align(size, 1024) + 1024


def decode_ctas_per_sm(query_len, num_stages, *, smem_per_sm, max_threads_per_sm,
                       max_blocks_per_sm=32):
    """Resident decode scorer CTAs per SM: the smem, thread and block limits."""
    threads = 32 * (_COMPUTE_WARPS + 1)
    per_cta = decode_smem_bytes(query_len, num_stages) + _SMEM_RESERVED_PER_CTA
    return max(1, min(smem_per_sm // per_cta, max_threads_per_sm // threads,
                      max_blocks_per_sm))


def auto_split_k(request_count, sm_count, ctas_per_sm=AUTO_SPLIT_CTAS_PER_SM):
    """split_k for ``request_count`` requests: ~0.8 of one resident wave of CTAs."""
    wave = ctas_per_sm * sm_count
    requests = max(1, request_count)
    granule = AUTO_SPLIT_GRANULE
    split = -(-int(AUTO_SPLIT_WAVE_FRACTION * wave) // requests // granule) * granule
    if split * requests > wave:
        split -= granule
    return max(AUTO_SPLIT_MIN, min(AUTO_SPLIT_MAX, split))
_SUPPORTED_ARCHITECTURES = ((10, 0), (10, 3), (10, 7))

__all__ = [
    "ICP_DECODE_SCORE_ABI_VERSION",
    "ICP_DECODE_SCORE_LAUNCH_ABI_VERSION",
    "supports_icp_decode_score",
    "get_icp_decode_scorer",
    "icp_decode_score",
]


def _integer(name, value, minimum):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class _DecodeProfile:
    """The compile key: everything the native scan specializes on."""

    query_len: int
    rank: int
    world_size: int
    split_k: int  # 0: chosen per request count at launch
    num_stages: int = DEFAULT_NUM_STAGES
    use_pdl: bool = True

    def __post_init__(self):
        _integer("query_len", self.query_len, 1)
        if self.query_len > 4:
            raise ValueError("query_len must be one of 1, 2, 3, 4")
        _integer("rank", self.rank, 0)
        _integer("world_size", self.world_size, 1)
        _integer("split_k", self.split_k, 0)
        _integer("num_stages", self.num_stages, 1)
        if self.num_stages > 8:
            raise ValueError("num_stages must be <= 8")
        if type(self.use_pdl) is not bool:
            raise TypeError("use_pdl must be a bool")
        if not self.use_pdl:
            # The scan is PDL-only: its non-PDL form scored wrong cells in v9.
            raise ValueError(
                "use_pdl=False is refused: the decode scorer is PDL-only (the "
                "non-PDL path scored wrong cells at b8 rank0 with 4 stages and "
                "was removed); use the PDL scorer (VLLM_MINIMAX_ICP_DECODE_PDL=1)"
            )
        if self.world_size != 2:
            raise ValueError("ICP decode ABI v2 supports only TP=ICP2/P128/R64")
        if self.rank >= self.world_size:
            raise ValueError("rank must be 0 or 1 for TP=ICP2")
        if self.split_k > 65535:
            raise ValueError("split_k must fit CUDA grid.y (<= 65535)")

    @property
    def max_decode_query_len(self):
        # A Q1/H4 tile still needs eight initialized columns for ldmatrix/MMA.
        return 2 if self.query_len <= 2 else 4


@dataclass(frozen=True)
class _DecodeWindow:
    """Runtime launch scalars: rows ``[token_begin, token_begin + token_count)``
    of the invocation, owned by requests ``[request_begin, request_begin +
    request_count)``. ``request_count`` is the grid's x extent."""

    token_begin: int
    token_count: int
    request_begin: int
    request_count: int

    def __post_init__(self):
        for name in ("token_begin", "request_begin"):
            _integer(name, getattr(self, name), 0)
        for name in ("token_count", "request_count"):
            _integer(name, getattr(self, name), 1)
        if self.token_begin + self.token_count > _INT32_MAX:
            raise ValueError("token window must fit int32")
        if self.request_begin + self.request_count > _INT32_MAX:
            raise ValueError("request range must fit int32")
        if self.request_count > _INT32_MAX:
            raise ValueError("request_count must fit CUDA grid.x")

    @property
    def request_end(self):
        return self.request_begin + self.request_count

    @property
    def scalars(self):
        return (self.token_begin, self.token_count, self.request_begin,
                self.request_count)


def _cuda_device_info(device):
    """Resolve a startup device without reading any tensor's contents."""
    import torch

    if isinstance(device, int) and not isinstance(device, bool):
        resolved = torch.device("cuda", device)
    else:
        resolved = torch.device("cuda" if device is None else device)
    if resolved.type != "cuda":
        raise ValueError("ICP decode scoring requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("ICP decode scoring requires an available CUDA device")
    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
    return index, tuple(torch.cuda.get_device_capability(index))


def supports_icp_decode_score(
    *,
    query_len,
    world_size=2,
    page_size=128,
    num_heads=4,
    head_dim=128,
    dtype=None,
    device=None,
):
    """Return startup admission for the initial Blackwell decode profile.

    ``dtype=None`` denotes FP8 E4M3FN. This checks the fixed shape profile and
    CUDA architecture; compilation/dependency errors are reported by the factory.
    It is not a test of live request lengths and must not run during capture.
    """
    try:
        _DecodeProfile(query_len, 0, world_size, 256)
        if page_size != 128 or num_heads != 4 or head_dim != 128:
            return False
        if dtype is not None and str(dtype) not in (
            "float8_e4m3fn", "torch.float8_e4m3fn"
        ):
            return False
        _, architecture = _cuda_device_info(device)
        return architecture in _SUPPORTED_ARCHITECTURES
    except (ImportError, RuntimeError, TypeError, ValueError, AssertionError):
        # Unsupported/unavailable startup configurations select the existing
        # scorer; the explicit factory reports the underlying failure instead.
        return False


class _PreparedIcpDecodeScorer:
    """Retains one compiled scan over caller-owned tensors.

    Calling the object and ``scan(*nine_tensors)`` validate host metadata and
    launch once; :meth:`launch` launches without validation, for a caller that
    validated the same bindings once. None initializes outputs. The consumer
    must establish exact per-row live prefixes from request geometry before
    reading score or validity cells.

    The window is a runtime launch scalar: pass ``window=`` per call, or
    :meth:`bind` it once to get a scorer that shares this compiled scan.
    """

    def __init__(self, profile, device_index, architecture, scan, torch,
                 window=None, sm_count=None, ctas_per_sm=None):
        self.profile = profile
        self.device_index = device_index
        self.architecture = architecture
        self.window = window
        self._scan = scan
        self._torch = torch
        self._sm_count = sm_count
        self._ctas_per_sm = ctas_per_sm
        self._split = None if window is None else self.split_for(window)

    def _resolve_occupancy(self):
        props = self._torch.cuda.get_device_properties(self.device_index)
        self._sm_count = props.multi_processor_count
        smem = getattr(props, "shared_memory_per_multiprocessor", 0)
        threads = getattr(props, "max_threads_per_multi_processor", 0)
        if smem and threads:
            self._ctas_per_sm = decode_ctas_per_sm(
                self.profile.query_len, self.profile.num_stages,
                smem_per_sm=smem, max_threads_per_sm=threads,
            )
        else:
            self._ctas_per_sm = AUTO_SPLIT_CTAS_PER_SM

    @property
    def sm_count(self):
        if self._sm_count is None:
            self._resolve_occupancy()
        return self._sm_count

    @property
    def ctas_per_sm(self):
        if self._ctas_per_sm is None:
            self._resolve_occupancy()
        return self._ctas_per_sm

    def split_for(self, window):
        """grid.y for ``window``: the profile's fixed split, else automatic."""
        if self.profile.split_k:
            return self.profile.split_k
        return auto_split_k(window.request_count, self.sm_count, self.ctas_per_sm)

    def bind(self, *, token_begin, token_count, request_begin, request_count):
        """This compiled scan with a default window; compiles nothing."""
        window = _DecodeWindow(token_begin, token_count, request_begin,
                               request_count)
        return _PreparedIcpDecodeScorer(
            self.profile, self.device_index, self.architecture, self._scan,
            self._torch, window, self._sm_count, self._ctas_per_sm,
        )

    def _window(self, window):
        window = self.window if window is None else window
        if window is None:
            raise ValueError(
                "no token window: pass window= or bind() one; launch ABI 2 "
                "takes the window at launch, not at compile time"
            )
        return window

    def _tensor(self, tensor, name, dimensions, dtype):
        if not isinstance(tensor, self._torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != dimensions or tensor.dtype != dtype:
            raise ValueError(f"{name} must be a {dimensions}D {dtype} tensor")
        if tensor.device.type != "cuda" or tensor.device.index != self.device_index:
            raise ValueError(f"{name} must be on cuda:{self.device_index}")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name}'s last dimension must have unit stride")
        for axis in range(dimensions - 1):
            if tensor.stride(axis) <= 0:
                raise ValueError(f"{name} must have positive outer strides")

    @staticmethod
    def _nonoverlapping_3d(tensor, name):
        # Accept both THK and HTK physical orders, including retained padding.
        rows, heads, columns = tensor.shape
        row_stride, head_stride = tensor.stride(0), tensor.stride(1)
        if row_stride <= head_stride:
            overlaps = row_stride < columns or (
                heads > 1 and head_stride < (rows - 1) * row_stride + columns
            )
        else:
            overlaps = head_stride < columns or (
                rows > 1 and row_stride < (heads - 1) * head_stride + columns
            )
        if overlaps:
            raise ValueError(f"{name} must not have overlapping rows or heads")

    def _outputs(self, score_out, valid_out, window):
        self._tensor(score_out, "score_out", 3, self._torch.float32)
        self._tensor(valid_out, "valid_out", 3, self._torch.uint8)
        if score_out.shape != valid_out.shape:
            raise ValueError("score_out and valid_out must have matching THK shapes")
        if score_out.shape[0] < window.token_count or score_out.shape[1] != 4:
            raise ValueError("outputs must have shape [rows >= token_count, 4, Pwave]")
        if score_out.shape[2] < 1:
            raise ValueError("Pwave must be positive")
        self._nonoverlapping_3d(score_out, "score_out")
        self._nonoverlapping_3d(valid_out, "valid_out")

    def validate(
        self, index_q, index_k, block_table, query_start_loc, seq_lens, positions,
        active_rows, score_out, valid_out, *, window=None,
    ):
        """Check shape, dtype and stride metadata without device-value access.

        Live offsets must be nondecreasing, genuine intervals must have Q rows,
        and lengths/positions and physical IDs must remain within their buffers.
        These device-value invariants belong to the caller and are not read back.
        Output capacity checks are allocation bounds, not a full-plane write
        contract: inactive rows and columns outside live prefixes are undefined.
        The token window may extend past ``index_q``'s rows; the kernel never
        touches a row at or past ``index_q.shape[0]``.
        """
        torch = self._torch
        window = self._window(window)
        self._outputs(score_out, valid_out, window)
        self._tensor(index_q, "index_q", 3, torch.float8_e4m3fn)
        self._tensor(index_k, "index_k", 3, torch.float8_e4m3fn)
        self._tensor(block_table, "block_table", 2, torch.int32)
        self._tensor(query_start_loc, "query_start_loc", 1, torch.int32)
        self._tensor(seq_lens, "seq_lens", 1, torch.int32)
        self._tensor(positions, "positions", 1, torch.int64)
        self._tensor(active_rows, "active_rows", 1, torch.uint8)
        if index_q.shape[1:] != (4, 128):
            raise ValueError("index_q must have shape [T, 4, 128]")
        if index_k.shape[1:] != (64, 128) or index_k.shape[0] < 1:
            raise ValueError("index_k must have shape [physical_pages > 0, 64, 128]")
        self._nonoverlapping_3d(index_q, "index_q")
        self._nonoverlapping_3d(index_k, "index_k")
        for tensor, name in ((index_q, "index_q"), (index_k, "index_k")):
            if tensor.data_ptr() % 16:
                raise ValueError(f"{name} must have a 16-byte-aligned TMA base")
            if tensor.stride(0) % 16 or tensor.stride(1) % 16:
                raise ValueError(f"{name}'s TMA outer byte strides must be multiples of 16")
        request_end = window.request_end
        if block_table.shape[0] < request_end or seq_lens.shape[0] < request_end:
            raise ValueError("block_table and seq_lens must contain the request range")
        if query_start_loc.shape[0] < request_end + 1:
            raise ValueError("query_start_loc must include the request range's end offset")
        if block_table.shape[1] < 1 or block_table.stride(0) < block_table.shape[1]:
            raise ValueError("block_table must have nonoverlapping, nonempty rows")
        if score_out.shape[2] < block_table.shape[1]:
            raise ValueError("Pwave must cover the full block_table column capacity")
        # EXACTLY index_q's row count, not "at least". `_compile_profile` binds
        # index_q, positions and active_rows to ONE symbol (`total_tokens`), so
        # the compiled signature requires equality and tvm-ffi enforces it at
        # bind time; checking here names the argument and the numbers.
        if (
            positions.shape[0] != index_q.shape[0]
            or active_rows.shape[0] != index_q.shape[0]
        ):
            raise ValueError(
                "positions and active_rows must have exactly index_q's row count: "
                f"index_q {index_q.shape[0]}, positions {positions.shape[0]}, "
                f"active_rows {active_rows.shape[0]}. Slice the caller's retained "
                "planes to the invocation, e.g. buf[:num_tokens] -- a prefix view "
                "preserves data_ptr and is CUDA-graph safe."
            )

    def launch(
        self, index_q, index_k, block_table, query_start_loc, seq_lens, positions,
        active_rows, score_out, valid_out, *, window=None,
    ):
        """One prepared launch WITHOUT host validation.

        For callers that validated these bindings once (see :meth:`validate`);
        tvm-ffi still checks dtypes and the compiled shape symbols at bind time.
        """
        if window is None:
            window, split = self.window, self._split
        else:
            split = self.split_for(window)
        self._scan(
            index_q, index_k, block_table, score_out, seq_lens, query_start_loc,
            positions, active_rows, valid_out, *window.scalars, split,
        )

    def scan(
        self, index_q, index_k, block_table, query_start_loc, seq_lens, positions,
        active_rows, score_out, valid_out, *, window=None,
    ):
        """Write only live-prefix scores and valid=1, in one prepared launch.

        No initialization is needed or performed. Consumers must not inspect
        either output plane outside the exact active-row fragment prefixes.
        """
        window = self._window(window)
        self.validate(
            index_q, index_k, block_table, query_start_loc, seq_lens, positions,
            active_rows, score_out, valid_out, window=window,
        )
        self.launch(
            index_q, index_k, block_table, query_start_loc, seq_lens, positions,
            active_rows, score_out, valid_out, window=window,
        )

    __call__ = scan


def _compile_profile(profile, device_index, architecture):
    # split_k is a launch scalar: every split shares one compile.
    scan, torch = _compile_scan(
        profile.query_len, profile.rank, profile.num_stages, profile.use_pdl,
        device_index, architecture,
    )
    return _PreparedIcpDecodeScorer(
        profile, device_index, architecture, scan, torch
    )


@cache
def _compile_scan(query_len, rank, num_stages, use_pdl, device_index,
                  architecture):
    # The prewarmed AOT object first; a JIT compile is the fallback, and the
    # serving policy decides whether that fallback is allowed.
    import torch

    scan = _load_aot_scan(query_len, rank, num_stages, use_pdl, architecture)
    if scan is None:
        from fmha_sm100.icp import _jit_guard

        _jit_guard.on_compile(
            "decode", aot_function_name(query_len, rank, num_stages, use_pdl),
            arch=_arch_key(architecture),
        )
        scan = _jit_compile_scan(query_len, rank, num_stages, use_pdl, device_index)
    return scan, torch


def _trace_scan(query_len, rank, num_stages, use_pdl):
    """The native scan's fake signature and kernel object, ready for cute.compile."""
    profile = _DecodeProfile(query_len, rank, 2, 0, num_stages, use_pdl)
    from cutlass import Float8E4M3FN, Float32, Int32, Int64, Uint8, cute
    from quack.compile_utils import make_fake_tensor

    from ._icp_decode_score_kernel import IndexDecodeScoreKernel

    total_tokens, output_rows, output_columns = (
        cute.sym_int64(), cute.sym_int64(), cute.sym_int64()
    )
    # quack's helper keeps all non-leading strides dynamic int64 values. Only
    # the innermost stride is fixed at one; compound pitch and THK strides are
    # never inferred from shape or frozen to contiguous storage.
    q = make_fake_tensor(
        Float8E4M3FN, (total_tokens, 4, 128), divisibility=16, leading_dim=-1
    )
    k = make_fake_tensor(
        Float8E4M3FN, (cute.sym_int64(), 64, 128), divisibility=16, leading_dim=-1
    )
    table = make_fake_tensor(
        Int32, (cute.sym_int64(), cute.sym_int64()), divisibility=1, leading_dim=-1
    )
    shape = (output_rows, 4, output_columns)
    score = make_fake_tensor(Float32, shape, divisibility=1, leading_dim=-1)
    valid = make_fake_tensor(Uint8, shape, divisibility=1, leading_dim=-1)
    lengths = make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=-1)
    offsets = make_fake_tensor(Int32, (cute.sym_int64(),), divisibility=1, leading_dim=-1)
    positions = make_fake_tensor(Int64, (total_tokens,), divisibility=1, leading_dim=-1)
    # Uint8, not Boolean, and the byte width is the whole reason. quack's
    # `make_fake_tensor` computes `assumed_align = divisibility * dtype.width // 8`
    # (quack 0.6.1 `compile_utils.py:28`); `Boolean.width` is 1 *bit*, so a
    # 1-element divisibility floors to alignment 0, which CUTLASS DSL 4.8.0a0
    # rejects in `cute/typing.py:739` ("expects alignment >= 1, got 0"). quack
    # 0.6.3 added a `max(..., 1)` floor, so this only bites on <= 0.6.2 -- and
    # the shipped serving image is 4.8.0a0 + quack 0.6.1. Declaring the byte
    # type the plane actually has removes the dependency on either fix: CUTLASS
    # already lowers a Boolean tensor to an i8 pointer in memory
    # (`cute/typing.py:738`), so this changes no addressing, no stride and no
    # bit pattern. A torch `bool` tensor is one byte per element and is accepted
    # here as a zero-copy `uint8` view; the kernel tests `!= 0`.
    active = make_fake_tensor(Uint8, (total_tokens,), divisibility=1, leading_dim=-1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = IndexDecodeScoreKernel(
        Float8E4M3FN, 4, profile.max_decode_query_len, 0, 128,
        query_len=profile.query_len,
        rank=profile.rank,
        num_stages=profile.num_stages,
    )
    # Dynamic Int32 launch scalars: token_begin, token_count, request_begin,
    # request_count, split_k. Their compile-time values are placeholders only.
    window = (Int32(0), Int32(1), Int32(0), Int32(1), Int32(1))
    return kernel, (q, k, table, score, lengths, offsets, positions, active, valid,
                    *window, stream)


def _jit_compile_scan(query_len, rank, num_stages, use_pdl, device_index):
    """Compile in-process for the current device (no cache)."""
    import torch
    from cutlass import cute

    kernel, args = _trace_scan(query_len, rank, num_stages, use_pdl)
    with torch.cuda.device(device_index):
        return cute.compile(kernel, *args, options="--enable-tvm-ffi")


# ---------------------------------------------------------------------------
# AOT: prewarmed objects, keyed by source digest + arch + profile
# ---------------------------------------------------------------------------

#: Every compile profile vLLM can request: query_len 1-4 x rank 0-1, PDL only
#: (use_pdl=False is refused), at the default stage count.
AOT_PROFILES = tuple(
    (query_len, rank, DEFAULT_NUM_STAGES, True)
    for query_len in (1, 2, 3, 4) for rank in (0, 1)
)


def _arch_key(architecture):
    from fmha_sm100.icp import _cache

    return _cache.normalize_arch(architecture)


def aot_function_name(query_len, rank, num_stages, use_pdl):
    return (f"icp_decode_scan_q{int(query_len)}_r{int(rank)}_s{int(num_stages)}"
            f"_{'pdl' if use_pdl else 'nopdl'}")


def aot_path(query_len, rank, num_stages, use_pdl, arch):
    from fmha_sm100.icp import _cache

    name = aot_function_name(query_len, rank, num_stages, use_pdl)
    return _cache.component_dir("decode", _cache.normalize_arch(arch)) / f"{name}.so"


def _load_aot_scan(query_len, rank, num_stages, use_pdl, architecture):
    import os

    if os.environ.get("ICP_DECODE_AOT", "1") == "0":
        return None
    path = aot_path(query_len, rank, num_stages, use_pdl, _arch_key(architecture))
    if not path.is_file():
        return None
    from cutlass import runtime

    module = runtime.load_module(str(path), enable_tvm_ffi=True)
    return getattr(module, aot_function_name(query_len, rank, num_stages, use_pdl))


def export_aot_scan(query_len, rank, num_stages, use_pdl, arch):
    """Compile one profile for ``arch`` with no device and write its ``.so``.

    The DSL target is ``CUTE_DSL_ARCH``, read when CUTLASS DSL initialises, so the
    caller (``fmha_sm100.icp.prewarm``) sets it before the first ``import cutlass``;
    this refuses a mismatch rather than writing a wrong-arch object under this
    arch's key. Returns the artifact path.
    """
    import os
    import subprocess
    import tempfile

    from fmha_sm100.icp import _cache

    arch = _cache.normalize_arch(arch)
    dsl_arch = os.environ.get("CUTE_DSL_ARCH", "")
    if _cache.normalize_arch(dsl_arch or "0") != arch:
        raise RuntimeError(
            f"CUTE_DSL_ARCH={dsl_arch!r} but the artifact is keyed sm_{arch}; set "
            f"CUTE_DSL_ARCH=sm_{arch} before CUTLASS DSL is imported")
    from cutlass import cute, runtime

    target = aot_path(query_len, rank, num_stages, use_pdl, arch)
    target.parent.mkdir(parents=True, exist_ok=True)
    name = aot_function_name(query_len, rank, num_stages, use_pdl)
    kernel, args = _trace_scan(query_len, rank, num_stages, use_pdl)
    scan = cute.compile(kernel, *args, options="--enable-tvm-ffi")
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        obj = os.path.join(tmp, f"{name}.o")
        lib = os.path.join(tmp, f"{name}.so")
        scan.export_to_c(obj, function_name=name)
        cc = os.environ.get("CC", "cc")
        subprocess.run([cc, "-shared", "-o", lib, obj,
                        *runtime.find_runtime_libraries(enable_tvm_ffi=True)],
                       check=True, capture_output=True, text=True)
        os.replace(lib, target)
    return target


def get_icp_decode_scorer(
    *, query_len, rank, world_size=2, split_k=0, device=None,
    token_begin=None, token_count=None, request_begin=None, request_count=None,
    num_stages=DEFAULT_NUM_STAGES, use_pdl=True,
):
    """Compile/cache the native scan before graph capture and return a callable.

    One compile per ``(query_len, rank, num_stages, use_pdl)`` and device;
    ``split_k`` (grid.y) is a launch scalar, ``0`` choosing it from the
    window's request count (:func:`auto_split_k`). The scan is a programmatic
    dependent (``use_pdl`` must be True): fragments of blocks that end before the
    step's new tokens are issued before its wait, everything else follows it. The
    window arguments are optional: when all four are given the returned scorer
    has them bound as its default window (sharing the cached compile), otherwise
    pass ``window=`` per call or :meth:`~_PreparedIcpDecodeScorer.bind` one.
    All layouts retain dynamic outer strides, permitting both FP8 and NVFP4
    compound-page pitches. ABI v2 leaves inactive/tail cells undefined; the
    selector must own live bounds. The factory compiles no initializer, and the
    callable performs no output fill.
    """
    profile = _DecodeProfile(query_len, rank, world_size, split_k, num_stages,
                             use_pdl)
    window_args = (token_begin, token_count, request_begin, request_count)
    window = None
    if any(value is not None for value in window_args):
        window = _DecodeWindow(*window_args)
    device_index, architecture = _cuda_device_info(device)
    if architecture not in _SUPPORTED_ARCHITECTURES:
        raise ValueError("ICP decode ABI v2 requires SM100, SM103 or SM107")
    import torch

    with torch.cuda.device(device_index):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare the ICP decode scorer before CUDA graph capture")
    scorer = _compile_profile(profile, device_index, architecture)
    if window is None:
        return scorer
    return scorer.bind(**{
        name: getattr(window, name)
        for name in ("token_begin", "token_count", "request_begin", "request_count")
    })


def icp_decode_score(
    index_q, index_k, block_table, query_start_loc, seq_lens, positions, active_rows,
    score_out, valid_out, *, scorer, window=None,
):
    """Execute a scorer obtained earlier from :func:`get_icp_decode_scorer`."""
    scorer(
        index_q, index_k, block_table, query_start_loc, seq_lens, positions,
        active_rows, score_out, valid_out, window=window,
    )
