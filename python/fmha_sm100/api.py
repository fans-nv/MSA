# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""FMHA varlen attention API: plan + run for SM100.

fmha_sm100_plan and fmha_sm100.

Doc is REPO/docs/fmha_sm100_api.md
"""

import math
from typing import Optional, Tuple, Union

__all__ = [
    "fmha_sm100_plan", "fmha_sm100", "sparse_topk_select", "nvfp4_head_slot_views",
    "_fmha_sm100_plan", "_fmha_sm100",
    "PlanWorkspace", "PlanWorkspacePool", "PlanWorkspaceError",
    "icp_plan_workspace_pool",
    "IcpDevicePlanStore", "IcpDevicePlanSlot", "ICP_DEVICE_PLAN_ABI_VERSION",
    "prewarm_icp_scorer",
]

import numpy as np
from pathlib import Path

import torch

from .jit import _dlpack_dtype_code, _PACK_FACTORS, get_fmha_variant, get_reduction_module, get_plan_fn, get_sparse_topk_module
from .nvfp4_kv import nvfp4_head_slot_views
from . import q8kv4_decode_adapter
from . import q8kv4_prefill_adapter
def sparse_fmha(*args, **kwargs):
    from .sparse_fmha_adapter import sparse_fmha as run

    return run(*args, **kwargs)


def sparse_fmha_plan(*args, **kwargs):
    from .sparse_fmha_adapter import sparse_fmha_plan as plan

    return plan(*args, **kwargs)


_np_staging = np.empty(4096 * 1024, dtype=np.int32)
_np_staging_offset = 0

def _reset_np_staging():
    global _np_staging_offset
    _np_staging_offset = 0

def _plan_buf_from_list(data, device, ws=None, tag=None):
    """Stage a host int list to an int32 device buffer.

    ``ws`` is a caller-owned :class:`PlanWorkspace`.  When it is supplied the
    destination is that workspace's STABLE region for ``tag``, written in place
    with no device allocation, and the returned tensor is a prefix VIEW of it --
    same ``data_ptr()`` on every build, exact ``shape[0] == len(data)`` so every
    downstream shape guard
    (``batch_size = qo_segment_lens.shape[0]``, the ``kv_page_indptr`` length
    check in ``_fmha_sm100``) still reads the real element count.

    When ``ws`` is None the historical per-call ``torch.empty`` is kept, so every
    caller that has not been migrated behaves exactly as before.
    """
    global _np_staging, _np_staging_offset
    n = len(data)
    end = _np_staging_offset + n
    if end > _np_staging.shape[0]:
        _np_staging = np.empty(max(end, _np_staging.shape[0] * 2), dtype=np.int32)
        _np_staging_offset = 0
        end = n
    _np_staging[_np_staging_offset:end] = data
    if ws is None:
        buf = torch.empty(n, dtype=torch.int32, device=device)
    else:
        buf = ws.bind(tag, n, torch.int32, device)
    buf.copy_(torch.from_numpy(_np_staging[_np_staging_offset:end]), non_blocking=True)
    _np_staging_offset = end
    return buf

from enum import IntEnum, auto

class _BuffTag(IntEnum):
    fmha_sm100_cutlass_workspace = auto()

    packed_work_range = auto()
    packed_work_info = auto()
    kv_tile_begin_indices = auto()
    kv_tile_end_indices = auto()
    kv_split_indices = auto()
    plan_cost = auto()
    num_kv_splits_per_row = auto()
    workspace_lse = auto()

    workspace_o = auto()

    sparse_topk_workspace = auto()

    # ---- per-plan int32 metadata staged by `_plan_buf_from_list` ------------
    # These exist so that every per-plan device buffer has a NAME, which is what
    # lets a `PlanWorkspace` give each of them its own region.  They are never
    # used with `_alloc_workspace_buf` (the process-global, tag-keyed cache).
    plan_qo_segment_offsets = auto()
    plan_kv_segment_offsets = auto()
    plan_qo_offset = auto()
    plan_kv_segment_lens = auto()
    plan_kv_page_indptr = auto()
    plan_qo_segment_lens = auto()
    plan_kv_lens = auto()

    Total = auto()


_workspace_cache = [[None] * _BuffTag.Total for _ in range(16)]
# _workspace_cache_per_plan = []

def _new_ws_cache():
    pass
    # global _workspace_cache
    # if len(_workspace_cache) == 0:
    #     print("new")
    #     _workspace_cache_per_plan.append([[None] * _BuffTag.Total for _ in range(16)])

def _alloc_workspace_buf(tag, size, device, dtype):
    global _workspace_cache
    device_id = torch.device(device).index
    buf = _workspace_cache[device_id][tag]
    if buf is not None and buf.shape[0] >= size:
        return buf
    buf = torch.empty(size, dtype=dtype, device=device)
    _workspace_cache[device_id][tag] = buf
    return buf

def _get_workspace_buf(tag, device):
    global _workspace_cache
    device_id = torch.device(device).index
    return _workspace_cache[device_id][tag]

def _alloc_perplan_buf(tag, size, device, dtype, ws=None):
    """Storage for ONE per-plan buffer.

    The commented-out body this replaced cached on ``tag`` alone, in a
    process-global table.  That is exactly the hazard it must not have: the
    indexer builds one plan per query chunk and every chunk plan of a forward is
    ALIVE AT THE SAME TIME, so a tag-keyed global cache would hand two live
    plans the same `packed_work_info` and each would overwrite the other's work
    list -- silently, with fluent wrong output.  It stays disabled.

    ``ws`` is the supported answer: a caller-owned :class:`PlanWorkspace`, one
    per concurrently-live plan, whose regions are stable across builds.  Without
    one the historical per-call allocation is kept.
    """
    if ws is not None:
        return ws.bind(tag, size, dtype, device)
    buf = torch.empty(size, dtype=dtype, device=device)
    return buf


# ---------------------------------------------------------------------------
# Caller-owned plan workspaces
# ---------------------------------------------------------------------------
# WHY THIS EXISTS.  Every per-plan buffer above is a fresh `torch.empty` per
# build.  The indexer rebuilds one plan PER QUERY CHUNK on every forward, so a
# steady-state step allocates and frees nine device buffers per chunk -- around
# a megabyte each for `packed_work_info` -- purely to write the same metadata
# into different addresses.  That is allocator churn on the submission path, and
# it is the thing this removes.
#
# It is NOT a fix for a capture-correctness defect.  Plan builds happen in the
# metadata builder, which runs before replay rather than inside capture, and the
# eager prefill route is the one that dominates plan construction today.  What a
# stable workspace additionally buys is that the plan's device pointers no
# longer move between builds, so a future profile that does capture a plan's
# consumers is not exposed to the pointers changing underneath it.
#
# The shape required by `refined-icp-design/runtime/MODEL_OWNED_INTEGRATION_
# BOUNDARY.md` ("bind stable model/package workspace views for each supported
# profile and update exact device metadata in place"; MSA owns "reusable bounded
# plans and page metadata") is:
#
#   * the CALLER allocates once, at startup, from bounds it already knows;
#   * plan construction writes IN PLACE into that storage;
#   * concurrently-live plans get DISJOINT storage, and disjointness is a
#     property of the construction, not of a calling convention.
#
# Nothing here reads a device value, allocates on the submission path, or
# synchronises: the only per-build work is host integer bookkeeping
# (`GPU_EXECUTION_NONBLOCKING_REVIEW.md`).

class PlanWorkspaceError(RuntimeError):
    """A caller-owned plan workspace cannot serve a plan.

    Always raised, never worked around: every condition that reaches it is one
    whose silent alternative is a plan that reads the wrong bytes.
    """


#: Work-item capacity the planner is handed per unit of split-KV degree.  This
#: is the literal `_fmha_sm100_plan_impl` has always passed as
#: `packed_work_info`'s length (and, through `packed_work_info_size`, as the
#: kernel's own bounds-check extent).  It is named here so a workspace is sized
#: to EXACTLY the number the planner asks for rather than to a second,
#: independently-guessed one -- a workspace smaller than the request would be
#: refused, and a request larger than the kernel's bound would be an OOB write.
_PLAN_MAX_WORK_ITEMS_PER_SPLIT = 131072

#: Byte alignment of every region inside a workspace arena.  256 is the CUDA
#: allocator's own base alignment, and being a multiple of 8 it keeps every
#: `uint8 -> int32/int64/float32` retype legal at a non-zero storage offset.
_PLAN_WS_ALIGN = 256

#: The seven int32 metadata buffers `_plan_buf_from_list` produces per plan, in
#: build order.  Each holds at most `batch_size + 1` entries (the prefix sums);
#: the four length/offset vectors hold `batch_size`.
_PLAN_LIST_TAGS = (
    _BuffTag.plan_qo_segment_offsets,
    _BuffTag.plan_kv_segment_offsets,
    _BuffTag.plan_qo_offset,
    _BuffTag.plan_kv_segment_lens,
    _BuffTag.plan_kv_page_indptr,
    _BuffTag.plan_qo_segment_lens,
    _BuffTag.plan_kv_lens,
)


def _norm_device(device):
    """One canonical `torch.device`, so region/plan device checks compare equal."""
    if isinstance(device, int):
        return torch.device("cuda", device)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


class PlanWorkspace:
    """Stable device storage for ONE concurrently-live plan.

    Allocate once, at startup, from startup bounds; hand it to
    ``_fmha_sm100_plan``/``_fmha_sm100_plan_impl`` as ``plan_workspace=``.  Every
    plan built against it writes into the same bytes: no device allocation on
    the submission path, and the plan's device pointers do not move between
    builds.

    One workspace serves ONE plan at a time.  Two plans that must be alive
    together -- the indexer's per-chunk ICP plans are the motivating case -- need
    two workspaces; :class:`PlanWorkspacePool` supplies them and proves they do
    not overlap.  Rebinding a workspace while an earlier plan is still in use is
    not silently permitted: ``begin_plan`` bumps :attr:`epoch`, the plan records
    the epoch it was built at, and ``_fmha_sm100`` refuses a plan whose epoch has
    moved on.

    Parameters
    ----------
    device
        Where the arena lives.  All regions are on this device and a plan built
        for a different device is refused.
    max_segments
        Upper bound on a plan's ``batch_size`` (the number of Q/KV segments).
        Sizes the seven int32 metadata regions at ``max_segments + 1``.
    max_kv_splits
        Largest split-KV degree this workspace admits.  At the default 1 the
        split-KV regions are not allocated at all, and a plan that asks for them
        is refused rather than served from nowhere.  ICP is always 1 (split-KV is
        asserted unsupported in ``_fmha_sm100``).
    max_total_qo_len, max_num_qo_heads
        Only needed when ``max_kv_splits > 1``; they size
        ``num_kv_splits_per_row`` and ``workspace_lse``.
    num_ctas
        Length of ``packed_work_range``; the planner uses
        ``min(usable_SM_count, multi_processor_count)``, so the device's SM count
        is the bound.  Defaults to the device's SM count, which requires CUDA;
        pass it explicitly to build a workspace without touching the driver.
    max_work_items
        Length of ``packed_work_info`` (and of the split-KV index regions).
        Defaults to ``_PLAN_MAX_WORK_ITEMS_PER_SPLIT * max_kv_splits``, which is
        what the planner asks for today.
    label
        Free-form name used in error messages.
    """

    def __init__(self, *, device, max_segments, max_kv_splits=1,
                 max_total_qo_len=0, max_num_qo_heads=0, num_ctas=None,
                 max_work_items=None, label=None):
        if int(max_segments) < 1:
            raise PlanWorkspaceError(
                f"max_segments must be >= 1, got {max_segments}")
        if int(max_kv_splits) < 1:
            raise PlanWorkspaceError(
                f"max_kv_splits must be >= 1, got {max_kv_splits}")
        self.device = _norm_device(device)
        self.label = label
        self.max_segments = int(max_segments)
        self.max_kv_splits = int(max_kv_splits)
        self.max_total_qo_len = int(max_total_qo_len)
        self.max_num_qo_heads = int(max_num_qo_heads)
        self.num_ctas = int(_get_num_cta(self.device) if num_ctas is None
                            else num_ctas)
        self.max_work_items = int(
            _PLAN_MAX_WORK_ITEMS_PER_SPLIT * self.max_kv_splits
            if max_work_items is None else max_work_items)

        regions = [(tag, torch.int32, self.max_segments + 1)
                   for tag in _PLAN_LIST_TAGS]
        regions.append((_BuffTag.packed_work_range, torch.int64, self.num_ctas))
        regions.append((_BuffTag.packed_work_info, torch.int64,
                        self.max_work_items))
        if self.max_kv_splits > 1:
            if self.max_total_qo_len < 1 or self.max_num_qo_heads < 1:
                raise PlanWorkspaceError(
                    "max_kv_splits > 1 needs max_total_qo_len and "
                    "max_num_qo_heads to size num_kv_splits_per_row and "
                    f"workspace_lse; got {self.max_total_qo_len} and "
                    f"{self.max_num_qo_heads}")
            for tag in (_BuffTag.kv_tile_begin_indices,
                        _BuffTag.kv_tile_end_indices,
                        _BuffTag.kv_split_indices):
                regions.append((tag, torch.int32, self.max_work_items))
            regions.append((_BuffTag.num_kv_splits_per_row, torch.int32,
                            self.max_total_qo_len))
            regions.append((_BuffTag.workspace_lse, torch.float32,
                            self.max_kv_splits * self.max_total_qo_len
                            * self.max_num_qo_heads))

        # ---- one arena, carved at explicit byte offsets ---------------------
        # Disjointness inside a workspace is ARITHMETIC, not an allocator
        # promise: the offsets are a running sum of aligned region sizes, and
        # the loop below asserts the carved intervals are ordered, in bounds and
        # non-overlapping before a single plan can touch them.
        layout, offset = [], 0
        for tag, dtype, count in regions:
            if count < 0:
                raise PlanWorkspaceError(
                    f"region {_BuffTag(tag).name} has negative size {count}")
            nbytes = count * torch.empty(0, dtype=dtype).element_size()
            layout.append((tag, dtype, count, offset, nbytes))
            offset += -(-nbytes // _PLAN_WS_ALIGN) * _PLAN_WS_ALIGN
        self._arena = torch.empty(max(offset, _PLAN_WS_ALIGN),
                                  dtype=torch.uint8, device=self.device)
        self._regions = {}
        prev_end = 0
        for tag, dtype, count, off, nbytes in layout:
            if off < prev_end or off + nbytes > self._arena.numel():
                raise PlanWorkspaceError(
                    f"internal layout error at {_BuffTag(tag).name}: "
                    f"[{off}, {off + nbytes}) vs prev_end={prev_end}, "
                    f"arena={self._arena.numel()}")
            prev_end = off + nbytes
            self._regions[int(tag)] = (
                self._arena[off:off + nbytes].view(dtype) if nbytes
                else torch.empty(0, dtype=dtype, device=self.device))
        self._epoch = 0
        self._bound = set()

    # -- identity -----------------------------------------------------------
    @property
    def epoch(self):
        """Bumped by every :meth:`begin_plan`; a plan records the value it saw."""
        return self._epoch

    def arena_span(self):
        """``(first_byte, last_byte_exclusive)`` of this workspace's storage."""
        base = self._arena.data_ptr()
        return (base, base + self._arena.numel())

    def nbytes(self):
        return self._arena.numel()

    def region(self, tag):
        """The full stable region for ``tag``, or None if not sized for it."""
        return self._regions.get(int(tag))

    # -- per-build protocol -------------------------------------------------
    def begin_plan(self):
        """Open a new plan build; returns the epoch it must be validated at."""
        self._epoch += 1
        self._bound.clear()
        return self._epoch

    def bind(self, tag, size, dtype, device=None):
        """The stable region for ``tag``, viewed as exactly ``size`` elements."""
        if tag is None:
            raise PlanWorkspaceError(
                "a plan buffer reached a workspace without a tag; every "
                "per-plan buffer must name its own region")
        key = int(tag)
        region = self._regions.get(key)
        name = _BuffTag(key).name
        if region is None:
            raise PlanWorkspaceError(
                f"{self}: not sized for {name}.  This workspace admits "
                f"max_kv_splits={self.max_kv_splits}; a plan needing "
                "split-KV storage must be given a workspace built with the "
                "matching max_kv_splits / max_total_qo_len / max_num_qo_heads.")
        if region.dtype != dtype:
            raise PlanWorkspaceError(
                f"{self}: {name} is {region.dtype}, plan asked for {dtype}")
        if device is not None and _norm_device(device) != self.device:
            raise PlanWorkspaceError(
                f"{self}: lives on {self.device}, plan is building on "
                f"{_norm_device(device)}")
        if size > region.shape[0]:
            raise PlanWorkspaceError(
                f"{self}: {name} needs {size} elements, workspace holds "
                f"{region.shape[0]}.  Raise the bound this workspace was sized "
                "from (max_segments / max_kv_splits / max_work_items); the "
                "plan is NOT truncated to fit.")
        if key in self._bound:
            raise PlanWorkspaceError(
                f"{self}: {name} bound twice in one plan build -- the second "
                "binding would alias the first.  Each per-plan buffer owns one "
                "region; this is a planner bug, not a capacity problem.")
        self._bound.add(key)
        return region[:size]

    def __repr__(self):
        tail = f" {self.label!r}" if self.label else ""
        return (f"<PlanWorkspace{tail} device={self.device} "
                f"max_segments={self.max_segments} "
                f"max_kv_splits={self.max_kv_splits} "
                f"max_work_items={self.max_work_items} "
                f"bytes={self._arena.numel()}>")

    __str__ = __repr__


class PlanWorkspacePool:
    """A fixed number of :class:`PlanWorkspace` slots that do not overlap.

    ``num_slots`` bounds how many plans may be alive at once.  Slot ``i`` is a
    separate workspace with a separate arena, and the constructor VERIFIES that
    the arenas are pairwise disjoint rather than assuming it -- one sort and one
    scan over ``num_slots`` host integers, at startup.

    Asking for a slot outside ``[0, num_slots)`` raises: wrapping a chunk index
    modulo the pool size is the one way two live plans could still share storage,
    so it is refused instead of silently aliasing.
    """

    def __init__(self, num_slots, **workspace_kwargs):
        num_slots = int(num_slots)
        if num_slots < 1:
            raise PlanWorkspaceError(f"num_slots must be >= 1, got {num_slots}")
        label = workspace_kwargs.pop("label", "pool")
        self._slots = [PlanWorkspace(label=f"{label}[{i}]", **workspace_kwargs)
                       for i in range(num_slots)]
        spans = sorted(s.arena_span() for s in self._slots)
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            if a1 > b0:
                raise PlanWorkspaceError(
                    f"plan workspace arenas overlap: [{a0}, {a1}) and "
                    f"[{b0}, {b1}).  Two live plans would corrupt each other.")

    def __len__(self):
        return len(self._slots)

    def __iter__(self):
        return iter(self._slots)

    def slot(self, index):
        """Slot ``index``; raises rather than wrapping."""
        index = int(index)
        if not 0 <= index < len(self._slots):
            raise PlanWorkspaceError(
                f"plan workspace slot {index} is outside [0, "
                f"{len(self._slots)}).  Size the pool from the bound on "
                "concurrently-live plans; do not reuse a slot that a live plan "
                "still owns.")
        return self._slots[index]

    __getitem__ = slot

    def nbytes(self):
        return sum(s.nbytes() for s in self._slots)


def _assert_plan_workspace_current(plan_info):
    """The plan's storage must still BE the plan's.

    A plan built into a caller-owned :class:`PlanWorkspace` records that
    workspace's epoch.  If the workspace has since been rebound -- a second plan
    built on the same slot while this one was still alive -- then every device
    buffer this plan names now holds the OTHER plan's metadata.  That is the
    exact silent corruption a tag-keyed shared cache would have produced, so it
    is refused here rather than executed.

    One host integer compare on a dict the caller already owns: no device read,
    no synchronization, nothing that can interrupt GPU submission
    (``GPU_EXECUTION_NONBLOCKING_REVIEW.md``).  Plans built without a workspace
    carry ``None`` and skip it entirely.
    """
    ws = plan_info.get("_plan_workspace")
    if ws is None:
        return
    built_at = plan_info.get("_plan_workspace_epoch")
    if built_at != ws.epoch:
        raise PlanWorkspaceError(
            f"this plan was built at {ws} epoch {built_at}, but the workspace "
            f"is now at epoch {ws.epoch}: its buffers were rebound by a later "
            "plan and no longer hold this plan's metadata.  Give each "
            "concurrently-live plan its own workspace slot.")


def icp_plan_workspace_pool(*, device, max_num_seqs, max_num_batched_tokens,
                            query_chunk_tokens=128, num_ctas=None):
    """A pool sized for the indexer's per-chunk ICP plans, from startup bounds.

    Capacity comes only from constants the caller has at startup:

    ``num_slots``
        ``ceil(max_num_batched_tokens / query_chunk_tokens)`` -- the chunk loop
        walks ``range(0, num_tokens, query_chunk_tokens)`` with
        ``num_tokens <= max_num_batched_tokens``, and every chunk's plan is alive
        until the forward ends.  Pass the PADDED token capacity here if the
        caller rounds ``max_num_batched_tokens`` up (the indexer aligns it to
        ``QCAPACITY_ALIGNMENT``), because that padded value is what its own
        ``num_tokens <= cap`` assert admits.
    ``max_segments``
        ``min(max_num_seqs, query_chunk_tokens)`` -- a chunk's plan has one
        segment per request whose query range intersects the chunk, each such
        request contributes at least one of the chunk's ``<= query_chunk_tokens``
        tokens, and there are never more than ``max_num_seqs`` requests.
    ``max_kv_splits = 1``
        ``_fmha_sm100`` asserts ICP x split-KV unsupported, and the indexer
        passes ``num_kv_splits=1``.  The split-KV regions are therefore not
        allocated, and a split-KV plan built against this pool is REFUSED.

    ``max_model_len``, the 128-token block size and the ICP degree W are
    deliberately absent: they size the page table, the score wave and
    ``max_k_tiles``, all of which the caller already owns, and none of them
    reaches a buffer allocated here.  Taking them would suggest this pool grows
    with context length, which it does not.
    """
    if int(query_chunk_tokens) < 1:
        raise PlanWorkspaceError(
            f"query_chunk_tokens must be >= 1, got {query_chunk_tokens}")
    if int(max_num_seqs) < 1 or int(max_num_batched_tokens) < 1:
        raise PlanWorkspaceError(
            f"max_num_seqs={max_num_seqs} and "
            f"max_num_batched_tokens={max_num_batched_tokens} must both be >= 1")
    num_slots = -(-int(max_num_batched_tokens) // int(query_chunk_tokens))
    return PlanWorkspacePool(
        num_slots,
        device=device,
        max_segments=min(int(max_num_seqs), int(query_chunk_tokens)),
        max_kv_splits=1,
        num_ctas=num_ctas,
        label="icp-plan",
    )

# ---------------------------------------------------------------------------
# Device-derived ICP prefill plans (ICP_DEVICE_PLAN_ABI 1)
# ---------------------------------------------------------------------------
#: Must equal the fused writer's ``ICP_DEVICE_PLAN_ABI``; the writer wrapper
#: refuses a store whose version differs.
ICP_DEVICE_PLAN_ABI_VERSION = 3

#: Row order of ``IcpDevicePlanStore.segments`` (the writer's plan CTAs use it).
_ICP_PLAN_ROWS = ("qo_segment_offsets", "kv_segment_offsets",
                  "qo_segment_lens", "kv_segment_lens", "qo_offset")
_ICP_PLAN_QO_TILE = 128
#: Row order of ``IcpDevicePlanStore.ranges``; columns are [final | scratch].
_ICP_PLAN_RANGE_ROWS = ("kv_tile_begin_indices", "kv_tile_end_indices",
                        "kv_split_indices")
#: kv_tile_end the writer gives the (only) piece of a tile: open-ended.
ICP_PLAN_OPEN_END = 0x7FFFFFFF
#: Writer-ABI argument (``icp_plan_min_split_tiles``); inert at max_splits 1.
_WRITER_MIN_SPLIT_TILES = 16


def _icp_tile_pages():
    """Local pages per ICP KV compute tile, read from the JIT variant axes.

    The writer's per-item tile ranges are in the scorer's own trip unit
    (``get<1>(TileShape)`` = the page axis' ``tile_kv``), which binds tile_kv to
    twice the fragment page so the K-split (thread_shape ``_1, _2, _1``) gives
    one page per softmax stage.
    """
    from .jit import _FMHA_SM100_DISPATCH

    axes = dict(_FMHA_SM100_DISPATCH)
    pages = {int(p["tile_kv"].lstrip("_")) // v
             for v, p in axes["int page_size"] if v in (32, 64)}
    q128 = dict(axes["int qo_tile_size"])[128]
    if len(pages) != 1 or q128["thread_shape"] != "_1, _2, _1":
        raise RuntimeError(
            f"ICP tile/page binding changed ({pages}, {q128}); the device plan's "
            "tile units must be re-derived")
    return pages.pop()


ICP_PLAN_TILE_PAGES = _icp_tile_pages()


class IcpDevicePlanSlot:
    """One chunk slot of an :class:`IcpDevicePlanStore`, at one chunk width.

    The slot's window is ``[max(index * chunk_width, row_begin),
    (index + 1) * chunk_width)``; ``row_begin`` is the step's FMHA window origin.
    """

    __slots__ = ("chunk_width", "index", "row_begin", "store")

    def __init__(self, store, index, chunk_width, row_begin=0):
        self.store = store
        self.index = index
        self.chunk_width = chunk_width
        self.row_begin = row_begin

    @property
    def window_begin(self):
        return max(self.index * self.chunk_width, self.row_begin)

    @property
    def window_tokens(self):
        return (self.index + 1) * self.chunk_width - self.window_begin

    def views(self, num_segments):
        """Exact-length views of this slot's rows; no device work."""
        store, i, n = self.store, self.index, int(num_segments)
        seg = store.segments[i]
        work = store.work[i]
        return {
            "qo_segment_offsets": seg[0, :n + 1],
            "kv_segment_offsets": seg[1, :n + 1],
            "qo_segment_lens": seg[2, :n],
            "kv_segment_lens": seg[3, :n],
            "qo_offset": seg[4, :n],
            "packed_work_range": work[:store.num_ctas],
            "packed_work_info": work[store.num_ctas:
                                     store.num_ctas + store.max_work_items],
            **{name: store.ranges[i, r, :store.max_work_items]
               for r, name in enumerate(_ICP_PLAN_RANGE_ROWS)},
        }

    def assert_current(self, built_launches):
        """Refuse rows not refreshed by a writer launch after the build."""
        store = self.store
        if store.writer_launches <= built_launches:
            raise PlanWorkspaceError(
                f"{self!r} was built after the last fused-writer launch that "
                "refreshed it; its rows still hold an earlier step's plan.  "
                "Pass the store to the first sparse writer (icp_device_plan=) "
                "before the scorer runs.")
        if store.last_chunk_width != self.chunk_width:
            raise PlanWorkspaceError(
                f"{self!r} was built for chunk width {self.chunk_width} but "
                f"the writer last planned chunk width {store.last_chunk_width}")
        if store.last_row_begin != self.row_begin:
            raise PlanWorkspaceError(
                f"{self!r} was built for window origin {self.row_begin} but the "
                f"writer last planned origin {store.last_row_begin}")
        if self.window_begin >= store.last_num_rows:
            raise PlanWorkspaceError(
                f"{self!r} starts at row {self.window_begin}, past "
                f"the writer's planned extent {store.last_num_rows}")

    def __repr__(self):
        return (f"<IcpDevicePlanSlot {self.index} of {self.store!r} "
                f"chunk_width={self.chunk_width} row_begin={self.row_begin}>")


class IcpDevicePlanStore:
    """Stable storage for every chunk's OnlyScoreIcp prefill plan.

    The first sparse layer's fused KV writer fills it: one extra CTA per chunk
    slot derives the plan from the exact device ``query_start_loc`` and
    ``seq_lens`` in the same launch as the live metadata. Host plan builds only
    select sizes and return views, so no plan content is staged from the host.
    The plan is written once per step and read by the scorer of every layer.

    ``max_segments`` bounds the requests of one chunk and ``max_chunk_tokens``
    the widest chunk; together with ``num_heads`` they bound the work items:
    ``num_heads * (ceil(max_chunk_tokens / 128) + max_segments)``.
    Plans are unsplit: ``max_kv_splits`` is accepted only as 1 (the split-KV
    ICP scorer was removed in v13).
    """

    abi_version = ICP_DEVICE_PLAN_ABI_VERSION
    max_kv_splits = 1

    def __init__(self, *, device, num_slots, max_segments, num_heads,
                 max_chunk_tokens, num_ctas=None, max_kv_splits=1,
                 label="icp-device-plan"):
        num_slots, max_segments = int(num_slots), int(max_segments)
        num_heads, max_chunk_tokens = int(num_heads), int(max_chunk_tokens)
        if min(num_slots, max_segments, num_heads, max_chunk_tokens) < 1:
            raise PlanWorkspaceError(
                "IcpDevicePlanStore bounds must all be >= 1; got "
                f"num_slots={num_slots} max_segments={max_segments} "
                f"num_heads={num_heads} max_chunk_tokens={max_chunk_tokens}")
        if max_segments > 0xFFFF or num_heads > 0xFFFF:
            raise PlanWorkspaceError(
                "packed work info carries 16-bit batch and head indices; got "
                f"max_segments={max_segments} num_heads={num_heads}")
        self.device = _norm_device(device)
        self.label = label
        self.num_slots = num_slots
        self.max_segments = max_segments
        self.num_heads = num_heads
        self.max_chunk_tokens = max_chunk_tokens
        self.num_ctas = int(_get_num_cta(self.device) if num_ctas is None
                            else num_ctas)
        if not 1 <= self.num_ctas <= 256:
            raise PlanWorkspaceError(
                f"num_ctas must be in [1, 256] (one plan CTA thread per "
                f"bucket); got {self.num_ctas}")
        if int(max_kv_splits) != 1:
            raise PlanWorkspaceError(
                f"max_kv_splits must be 1; got {max_kv_splits} (the split-KV "
                "ICP scorer was removed in v13)")
        self.max_work_items = num_heads * (
            -(-max_chunk_tokens // _ICP_PLAN_QO_TILE) + max_segments)
        # Work row: [range | info | scratch info | scratch bucket/position].
        self.segments = torch.empty(
            (num_slots, len(_ICP_PLAN_ROWS), max_segments + 1),
            dtype=torch.int32, device=self.device)
        self.work = torch.empty(
            (num_slots, self.num_ctas + 3 * self.max_work_items),
            dtype=torch.int64, device=self.device)
        self.header = torch.empty((num_slots, 4), dtype=torch.int32,
                                  device=self.device)
        self.ranges = torch.empty(
            (num_slots, len(_ICP_PLAN_RANGE_ROWS), 2 * self.max_work_items),
            dtype=torch.int32, device=self.device)
        self.writer_launches = 0
        self.last_chunk_width = None
        self.last_num_rows = 0
        self.last_row_begin = 0

    def slot(self, index, *, chunk_width, row_begin=0):
        """Slot ``index`` for chunks of ``chunk_width`` tokens; never wraps."""
        index, chunk_width, row_begin = int(index), int(chunk_width), int(row_begin)
        if row_begin < 0 or row_begin >= (index + 1) * chunk_width:
            raise PlanWorkspaceError(
                f"window origin {row_begin} leaves slot {index} at chunk width "
                f"{chunk_width} empty")
        if not 0 <= index < self.num_slots:
            raise PlanWorkspaceError(
                f"device plan slot {index} is outside [0, {self.num_slots})")
        if not 1 <= chunk_width <= self.max_chunk_tokens:
            raise PlanWorkspaceError(
                f"chunk_width {chunk_width} is outside [1, "
                f"{self.max_chunk_tokens}] this store was sized for")
        return IcpDevicePlanSlot(self, index, chunk_width, row_begin)

    def writer_kwargs(self):
        """Arguments of the fused writer's device-plan CTAs.

        The writer ABI still carries the ranges plane and the split controls;
        ``icp_plan_max_splits=1`` keeps every plan unsplit.
        """
        return {
            "icp_plan_segments": self.segments,
            "icp_plan_work": self.work,
            "icp_plan_header": self.header,
            "icp_plan_num_ctas": self.num_ctas,
            "icp_plan_num_heads": self.num_heads,
            "icp_plan_ranges": self.ranges,
            "icp_plan_max_splits": 1,
            "icp_plan_tile_pages": ICP_PLAN_TILE_PAGES,
            "icp_plan_min_split_tiles": _WRITER_MIN_SPLIT_TILES,
        }

    def check_writer_launch(self, *, chunk_width, num_rows, row_begin=0):
        """Refuse a writer launch whose chunks exceed the sized bounds."""
        chunk_width, num_rows = int(chunk_width), int(num_rows)
        if not 0 <= int(row_begin) <= num_rows:
            raise PlanWorkspaceError(
                f"writer window origin {row_begin} is outside [0, {num_rows}]")
        if not 1 <= chunk_width <= self.max_chunk_tokens:
            raise PlanWorkspaceError(
                f"writer chunk width {chunk_width} exceeds the "
                f"{self.max_chunk_tokens} tokens {self!r} was sized for")
        if -(-num_rows // chunk_width) > self.num_slots:
            raise PlanWorkspaceError(
                f"writer extent {num_rows} at chunk width {chunk_width} needs "
                f"{-(-num_rows // chunk_width)} slots; {self!r} has "
                f"{self.num_slots}")

    def note_writer_launch(self, *, chunk_width, num_rows, row_begin=0):
        """Record a writer launch that refreshed this store."""
        self.writer_launches += 1
        self.last_chunk_width = int(chunk_width)
        self.last_num_rows = int(num_rows)
        self.last_row_begin = int(row_begin)

    def nbytes(self):
        return sum(t.numel() * t.element_size()
                   for t in (self.segments, self.work, self.header,
                             self.ranges))

    def __repr__(self):
        tail = f" {self.label!r}" if self.label else ""
        return (f"<IcpDevicePlanStore{tail} device={self.device} "
                f"slots={self.num_slots} max_segments={self.max_segments} "
                f"num_heads={self.num_heads} "
                f"max_chunk_tokens={self.max_chunk_tokens} "
                f"num_ctas={self.num_ctas} max_work={self.max_work_items}>")


def icp_scorer_variants(*, page_sizes, dtype=torch.float8_e4m3fn):
    """``(variant_name, get_fmha_variant args)`` of every ICP scorer a device plan
    can select: OnlyScoreIcp, qo tile 128, unsplit, pack 1, fp8 KV, both
    ``single_wg`` values (``_fmha_sm100`` picks ``max_qo_len <= 64``), per page size.
    """
    from .jit import _variant_key_from_runtime

    code = _dlpack_dtype_code(dtype)
    variants = []
    for page_size in page_sizes:
        for single_wg in (True, False):
            args = (code, _ICP_PLAN_QO_TILE, single_wg, 4, int(page_size),
                    False, 1)
            variants.append((_variant_key_from_runtime(*args)[0], args))
    return variants


def prewarm_icp_scorer(*, page_sizes, load=True, max_workers=None):
    """Compile every ICP scorer variant now, in parallel; returns their names.

    Startup and image-build hook so no variant JIT-compiles on the serving path.
    ``load=False`` only builds the cache (no CUDA driver needed).
    """
    from concurrent.futures import ThreadPoolExecutor

    from . import jit

    variants = icp_scorer_variants(page_sizes=page_sizes)
    manager = jit._variant_manager
    manager._load_templates()

    def build(item):
        name, args = item
        params = jit._variant_key_from_runtime(*args)[1]
        manager.compile_locked(name, params)

    if variants:
        with ThreadPoolExecutor(max_workers or len(variants)) as pool:
            list(pool.map(build, variants))
    if load:
        for _, args in variants:
            get_fmha_variant(*args)
    return [name for name, _ in variants]


def _icp_device_plan_info(*, slot, qo_lens, kv_lens, num_qo_heads,
                          num_kv_splits, usable_SM_count, causal,
                          output_maxscore, device,
                          icp_direct_fields):
    """Views of ``slot``; sizes come from host bounds, contents from the writer."""
    store = slot.store
    if not causal or num_kv_splits not in (-1, 0, 1):
        raise PlanWorkspaceError(
            "the ICP device plan is causal and unsplit; got "
            f"causal={causal} num_kv_splits={num_kv_splits}")
    if _norm_device(device) != store.device:
        raise PlanWorkspaceError(
            f"{store!r} lives on {store.device}, plan is building on "
            f"{_norm_device(device)}")
    n = len(qo_lens)
    if not 1 <= n <= store.max_segments:
        raise PlanWorkspaceError(
            f"a chunk plan carries {n} requests; {store!r} admits 1.."
            f"{store.max_segments}")
    if sum(qo_lens) > slot.window_tokens or min(qo_lens) < 0:
        raise PlanWorkspaceError(
            f"chunk query lengths {qo_lens} do not fit the {slot.window_tokens}"
            f"-row window of {slot!r} (chunk width {slot.chunk_width})")
    num_ctas = _get_num_cta(device)
    if usable_SM_count > 0:
        num_ctas = min(usable_SM_count, num_ctas)
    if num_ctas != store.num_ctas:
        raise PlanWorkspaceError(
            f"the scorer runs {num_ctas} CTAs but {store!r} plans "
            f"{store.num_ctas} buckets")
    if num_qo_heads != store.num_heads:
        raise PlanWorkspaceError(
            f"the plan has {num_qo_heads} query heads but {store!r} plans "
            f"{store.num_heads}")
    work = num_qo_heads * sum(-(-q // _ICP_PLAN_QO_TILE) for q in qo_lens)
    if work > store.max_work_items:
        raise PlanWorkspaceError(
            f"a chunk plan needs {work} work items; {store!r} holds "
            f"{store.max_work_items}")

    max_qo_len = max(qo_lens)
    total_qo_len = sum(qo_lens)
    max_kv_len = max(kv_lens)
    page_tokens = (icp_direct_fields["index_rows_per_rank"]
                   * icp_direct_fields["index_world_size"])
    pitch = icp_direct_fields["block_table_row_stride"]
    if -(-max_kv_len // page_tokens) > pitch:
        raise PlanWorkspaceError(
            f"block_table_row_stride={pitch} cannot hold the "
            f"{-(-max_kv_len // page_tokens)} compound pages of a "
            f"{max_kv_len}-token sequence; size the row from max_model_len")
    max_k_tiles = (math.ceil(math.ceil(max_kv_len / 128) / 128) * 128
                   if output_maxscore else -1)
    if output_maxscore and num_qo_heads * max_k_tiles * total_qo_len > (1 << 31):
        max_k_tiles = -1
    views = slot.views(n)
    info = _make_plan_info(
        packed_work_range=views["packed_work_range"],
        packed_work_info=views["packed_work_info"],
        **dict.fromkeys(_ICP_PLAN_RANGE_ROWS),
        num_kv_splits=1, workspace_o=None,
        workspace_lse=None, max_qo_len=max_qo_len, predicted_speedup=1.0,
        num_kv_splits_per_row=None,
        qo_segment_offsets=views["qo_segment_offsets"],
        kv_segment_offsets=views["kv_segment_offsets"],
        kv_page_indptr=None, max_k_tiles=max_k_tiles,
        qo_segment_lens=views["qo_segment_lens"],
        kv_segment_lens=views["kv_segment_lens"],
        qo_offset=views["qo_offset"],
        pack_factor=1, orig_num_qo_heads=num_qo_heads,
        qo_len_uniform=min(qo_lens) == max_qo_len,
        cute_workspace_buffer=_alloc_workspace_buf(
            _BuffTag.fmha_sm100_cutlass_workspace, 32 * 1024 * 1024, device,
            torch.uint8),
        kv_page_indptr_local_last=None,
        **icp_direct_fields,
)
    info["_device_plan"] = slot
    info["_device_plan_launches"] = store.writer_launches
    return info

_NUM_CTA = None
def _get_num_cta(device):
    global _NUM_CTA
    if _NUM_CTA is None:
        _NUM_CTA = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_CTA

def _compute_pack_factor(max_qo_len, num_qo_heads, num_kv_heads):
    if num_kv_heads == -1:
        return 1
    h_r = num_qo_heads // num_kv_heads
    if h_r <= 1 or max_qo_len <= 0 or max_qo_len > 32:
        return 1
    max_pf = 128 // max_qo_len
    for pf in reversed(_PACK_FACTORS):
        if pf <= max_pf and pf <= h_r and h_r % pf == 0:
            return pf
    return 1

def _to_device(t, d):
    return t.to(d, non_blocking=True) if t is not None else None

def _prefill_qlen_threshold(sparse):
    return 32 if sparse else 128

def _validate_fmha_inputs(
    q, k, v, qo_segment_lens, kv_segment_lens,
    num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
    batch_size, qo_total_len, is_paged,
    kv_indices, kv_block_indexes, qo_offset, page_size,
    pack_factor=1,
    packed_work_range=None, packed_work_info=None,
    num_kv_splits=1,
    kv_tile_begin_indices=None, kv_tile_end_indices=None, kv_split_indices=None,
    qo_segment_offsets=None, kv_page_indptr=None,
):
    assert q.dim() == 3, f"q must be [total_qo_len, num_qo_heads, head_dim], got {q.shape}"
    assert num_qo_heads % num_kv_heads == 0, (
        f"num_qo_heads ({num_qo_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    assert qo_segment_lens.dim() == 1, f"qo_segment_lens must be 1D, got {qo_segment_lens.shape}"
    assert kv_segment_lens.dim() == 1, f"kv_segment_lens must be 1D, got {kv_segment_lens.shape}"
    assert qo_segment_lens.shape[0] == batch_size
    assert kv_segment_lens.shape[0] == batch_size

    qo_lens_cpu = qo_segment_lens.cpu()
    kv_lens_cpu = kv_segment_lens.cpu()

    expected_qo_total = qo_total_len * pack_factor if pack_factor > 1 else qo_total_len
    assert qo_lens_cpu.sum().item() == expected_qo_total, (
        f"sum(qo_segment_lens)={qo_lens_cpu.sum().item()} != expected={expected_qo_total} "
        f"(q.shape[0]={qo_total_len}, pack_factor={pack_factor})"
    )

    if is_paged:
        assert k.dim() == 4, f"paged K must be [total_pages, H_kv, page_size, D], got {k.shape}"
        assert v.dim() == 4, f"paged V must be [total_pages, H_kv, page_size, D], got {v.shape}"
        total_pages = k.shape[0]
        assert k.shape[1] == num_kv_heads
        assert k.shape[2] == page_size
        assert v.shape[0] == total_pages
        assert v.shape[1] == num_kv_heads
        assert v.shape[2] == page_size
    else:
        total_kv_len = k.shape[0]
        assert kv_lens_cpu.sum().item() == total_kv_len, (
            f"sum(kv_segment_lens)={kv_lens_cpu.sum().item()} != k.shape[0]={total_kv_len}"
        )

    if qo_offset is not None:
        assert qo_offset.dim() == 1 and qo_offset.shape[0] == batch_size, (
            f"qo_offset must be [batch_size={batch_size}], got {qo_offset.shape}"
        )
        # off_cpu = qo_offset.cpu()
        # assert (off_cpu >= 0).all(), (
        #     f"qo_offset must be non-negative, got min={off_cpu.min().item()}"
        # )
        # unpacked_qo_lens = qo_lens_cpu // pack_factor if pack_factor > 1 else qo_lens_cpu
        # max_offsets = kv_lens_cpu - unpacked_qo_lens
        # violations = off_cpu > max_offsets
        # if violations.any():
        #     b = violations.nonzero()[0].item()
        #     assert False, (
        #         f"qo_offset[{b}]={off_cpu[b].item()} > kv_len-qo_len="
        #         f"{kv_lens_cpu[b].item()}-{qo_lens_cpu[b].item()}={max_offsets[b].item()}"
        #     )

    if kv_indices is not None and kv_block_indexes is None:
        assert is_paged, "kv_indices provided but K/V are not paged (4D)"
        kvi_cpu = kv_indices.cpu()
        total_pages_in_table = kvi_cpu.shape[0]
        total_pages_needed = sum(
            (kv_lens_cpu[b].item() + page_size - 1) // page_size for b in range(batch_size)
        )
        assert total_pages_in_table == total_pages_needed, (
            f"kv_indices length {total_pages_in_table} != "
            f"sum(ceil(kv_segment_lens/page_size))={total_pages_needed}"
        )
        total_pages = k.shape[0]
        assert (kvi_cpu >= 0).all() and (kvi_cpu < total_pages).all(), (
            f"kv_indices has values outside [0, {total_pages}): "
            f"min={kvi_cpu.min().item()}, max={kvi_cpu.max().item()}"
        )

    if kv_block_indexes is not None:
        assert is_paged, "sparse mode requires paged KV (kv_indices)"
        assert kv_indices is not None, "sparse mode requires kv_indices"
        assert kv_block_indexes.dim() == 3, (
            f"kv_block_indexes must be [total_qo_len, H_kv, KVBlockNum], got {kv_block_indexes.shape}"
        )
        assert kv_block_indexes.shape[0] == qo_total_len, (
            f"kv_block_indexes must be [total_qo_len={qo_total_len}, H_kv, KVBlockNum], got {kv_block_indexes.shape}"
        )
        assert kv_block_indexes.shape[1] == num_kv_heads

        bi_cpu = kv_block_indexes.cpu()
        valid_mask = (bi_cpu != -1)
        assert ((bi_cpu >= 0) | (bi_cpu == -1)).all(), (
            "kv_block_indexes must contain only non-negative indices or -1"
        )
        assert (valid_mask[:, :, :-1] >= valid_mask[:, :, 1:]).all(), (
            "kv_block_indexes: -1 padding must be at the end"
        )
        unpacked_qo_lens = qo_lens_cpu // pack_factor if pack_factor > 1 else qo_lens_cpu
        batch_of_qtoken = torch.repeat_interleave(
            torch.arange(batch_size), unpacked_qo_lens)
        kv_lens_per_qtoken = kv_lens_cpu[batch_of_qtoken]
        num_pages = ((kv_lens_per_qtoken + page_size - 1) // page_size).view(-1, 1, 1)
        assert (bi_cpu[valid_mask] < num_pages.expand_as(bi_cpu)[valid_mask]).all(), (
            "kv_block_indexes: index out of range for batch page count"
        )
        both_valid = valid_mask[:, :, :-1] & valid_mask[:, :, 1:]
        if both_valid.any():
            assert (bi_cpu[:, :, 1:][both_valid] > bi_cpu[:, :, :-1][both_valid]).all(), (
                "kv_block_indexes: must be strictly ascending"
            )

    if qo_segment_offsets is not None:
        off_cpu = qo_segment_offsets.cpu()
        assert off_cpu.shape[0] == batch_size + 1, (
            f"qo_segment_offsets must have {batch_size + 1} elements, got {off_cpu.shape[0]}"
        )
        assert off_cpu[0].item() == 0, f"qo_segment_offsets[0]={off_cpu[0].item()} != 0"
        assert off_cpu[-1].item() == expected_qo_total, (
            f"qo_segment_offsets[-1]={off_cpu[-1].item()} != expected={expected_qo_total}"
        )
        diffs = off_cpu[1:] - off_cpu[:-1]
        assert (diffs >= 0).all(), "qo_segment_offsets must be non-decreasing"
        assert (diffs == qo_lens_cpu).all(), "qo_segment_offsets must be cumsum of qo_segment_lens"

    if kv_page_indptr is not None:
        ip_cpu = kv_page_indptr.cpu()
        assert ip_cpu.shape[0] == batch_size + 1, (
            f"kv_page_indptr must have {batch_size + 1} elements, got {ip_cpu.shape[0]}"
        )
        assert ip_cpu[0].item() == 0, f"kv_page_indptr[0]={ip_cpu[0].item()} != 0"
        assert (ip_cpu[1:] >= ip_cpu[:-1]).all(), "kv_page_indptr must be non-decreasing"
        if kv_indices is not None:
            assert ip_cpu[-1].item() <= kv_indices.shape[0], (
                f"kv_page_indptr[-1]={ip_cpu[-1].item()} > kv_indices.size={kv_indices.shape[0]}"
            )

    if packed_work_range is not None and packed_work_info is not None:
        import math
        pwr_cpu = packed_work_range.cpu()
        pwi_cpu = packed_work_info.cpu()
        num_ctas = pwr_cpu.shape[0]
        max_work_idx = pwi_cpu.shape[0]

        packed_num_heads = num_qo_heads // pack_factor if pack_factor > 1 else num_qo_heads
        qo_tile_size = 128 if int(qo_lens_cpu.max()) <= 128 else 256
        max_tiles_per_batch = [(int(l) + qo_tile_size - 1) // qo_tile_size for l in qo_lens_cpu]

        for cta in range(num_ctas):
            r = int(pwr_cpu[cta].item())
            start = r & 0xFFFFFFFF
            end = (r >> 32) & 0xFFFFFFFF
            assert start <= end, (
                f"packed_work_range[{cta}]: start={start} > end={end}"
            )
            assert end <= max_work_idx, (
                f"packed_work_range[{cta}]: end={end} > packed_work_info.size={max_work_idx}"
            )
            for wi in range(start, end):
                packed = int(pwi_cpu[wi].item())
                bi = packed & 0xFFFF
                hi = (packed >> 16) & 0xFFFF
                qo_tile = (packed >> 32) & 0xFFFFFFFF
                assert bi < batch_size, (
                    f"packed_work_info[{wi}]: batch_idx={bi} >= batch_size={batch_size}"
                )
                assert hi < packed_num_heads, (
                    f"packed_work_info[{wi}]: head_idx={hi} >= packed_num_heads={packed_num_heads}"
                )
                assert qo_tile < max_tiles_per_batch[bi], (
                    f"packed_work_info[{wi}]: qo_tile={qo_tile} >= max_tiles={max_tiles_per_batch[bi]} "
                    f"for batch {bi} (qo_len={int(qo_lens_cpu[bi])})"
                )

        # Verify completeness: every (batch, head, qo_tile) must appear exactly once
        # (for num_kv_splits=1) or exactly num_splits times (for split-KV, each with
        # a different split index covering the full KV range).
        expected = set()
        for bi in range(batch_size):
            for hi in range(packed_num_heads):
                for qt in range(max_tiles_per_batch[bi]):
                    expected.add((bi, hi, qt))

        seen = {}
        for cta in range(num_ctas):
            r = int(pwr_cpu[cta].item())
            start = r & 0xFFFFFFFF
            end = (r >> 32) & 0xFFFFFFFF
            for wi in range(start, end):
                packed = int(pwi_cpu[wi].item())
                bi = packed & 0xFFFF
                hi = (packed >> 16) & 0xFFFF
                qt = (packed >> 32) & 0xFFFFFFFF
                key = (bi, hi, qt)
                seen[key] = seen.get(key, 0) + 1

        missing = expected - set(seen.keys())
        assert not missing, (
            f"plan missing {len(missing)} work items, e.g. {list(missing)[:5]}"
        )

        if num_kv_splits <= 1:
            duplicates = {k: v for k, v in seen.items() if v > 1}
            assert not duplicates, (
                f"plan has {len(duplicates)} duplicated work items (nosplit), e.g. {list(duplicates.items())[:5]}"
            )

        if num_kv_splits > 1 and kv_tile_begin_indices is not None:
            tb_cpu = kv_tile_begin_indices.cpu()
            te_cpu = kv_tile_end_indices.cpu()
            sp_cpu = kv_split_indices.cpu()

            from collections import defaultdict
            tile_splits = defaultdict(list)

            for cta in range(num_ctas):
                r = int(pwr_cpu[cta].item())
                start = r & 0xFFFFFFFF
                end = (r >> 32) & 0xFFFFFFFF
                for wi in range(start, end):
                    tb, te = int(tb_cpu[wi].item()), int(te_cpu[wi].item())
                    assert tb <= te, (
                        f"kv_tile_begin[{wi}]={tb} > kv_tile_end[{wi}]={te}"
                    )
                    sp = int(sp_cpu[wi].item())
                    assert 0 <= sp < num_kv_splits, (
                        f"kv_split_indices[{wi}]={sp} out of range [0, {num_kv_splits})"
                    )
                    packed = int(pwi_cpu[wi].item())
                    key = (packed & 0xFFFF, (packed >> 16) & 0xFFFF,
                           (packed >> 32) & 0xFFFFFFFF)
                    tile_splits[key].append((tb, te, sp))

            is_sparse = kv_block_indexes is not None
            kv_tile_size = 256 if qo_tile_size == 128 else 128
            for key, splits in tile_splits.items():
                bi, hi, qt = key
                if is_sparse:
                    kv_block_num = kv_block_indexes.shape[2]
                    kl = kv_block_num * page_size
                    off_q = kl - int(qo_lens_cpu[bi])
                else:
                    kl = int(kv_lens_cpu[bi])
                    off_q = int(qo_offset[bi].cpu()) if qo_offset is not None else (kl - int(qo_lens_cpu[bi]))
                packed_q_end = (qt + 1) * qo_tile_size
                q_end = (packed_q_end - 1) // pack_factor + 1 if pack_factor > 1 else packed_q_end
                eff_kv = min(q_end + off_q, kl)
                expected_iters = max(0, (eff_kv + kv_tile_size - 1) // kv_tile_size)
                splits_sorted = sorted(splits, key=lambda x: x[0])
                if expected_iters > 0:
                    assert splits_sorted[0][0] == 0, (
                        f"tile {key}: first split begins at {splits_sorted[0][0]}, expected 0"
                    )
                    assert splits_sorted[-1][1] == expected_iters, (
                        f"tile {key}: last split ends at {splits_sorted[-1][1]}, expected {expected_iters}"
                    )
                for i in range(1, len(splits_sorted)):
                    prev_end = splits_sorted[i - 1][1]
                    curr_begin = splits_sorted[i][0]
                    assert prev_end == curr_begin, (
                        f"tile {key}: gap/overlap at split boundary: "
                        f"prev_end={prev_end}, curr_begin={curr_begin}"
                    )
                sub_ids = sorted(s[2] for s in splits_sorted)
                expected_ids = list(range(len(splits_sorted)))
                assert sub_ids == expected_ids, (
                    f"tile {key}: sub_ids {sub_ids} not contiguous 0..{len(splits_sorted)-1}"
                )


def _expand_for_per_token_sparse(qo_lens, kv_lens, qo_offset, page_size, pack_factor=1):
    B = len(qo_lens)
    total_q = sum(qo_lens)

    if qo_offset is None:
        qo_offset = [kv_lens[i] - qo_lens[i] for i in range(B)]

    kv_page_indptr = [0]
    acc = 0
    for k in kv_lens:
        acc += (k + page_size - 1) // page_size
        kv_page_indptr.append(acc)

    expanded_qo_lens = [pack_factor] * total_q
    expanded_kv_lens = [0] * total_q
    expanded_qo_offset = [0] * total_q
    expanded_kv_page_indptr = [0] * (total_q + 1)

    idx = 0
    for b in range(B):
        kv_len_b = kv_lens[b]
        qo_offset_b = qo_offset[b]
        kv_page_indptr_b = kv_page_indptr[b]
        for j in range(qo_lens[b]):
            expanded_kv_lens[idx] = kv_len_b
            expanded_qo_offset[idx] = qo_offset_b + j
            expanded_kv_page_indptr[idx] = kv_page_indptr_b
            idx += 1

    expanded_kv_page_indptr[total_q] = kv_page_indptr[B]

    return expanded_qo_lens, expanded_kv_lens, expanded_qo_offset, expanded_kv_page_indptr

class PlanInfo(dict):
    """Execution plan returned by the internal dense FMHA planner.

    Users normally receive the public tuple returned by ``fmha_sm100_plan`` and
    pass it unchanged to ``fmha_sm100``.  The dictionary stores CUDA worklists,
    sequence metadata, split-KV workspaces, and cached buffers owned by the
    plan.
    """

    def __del__(self):
        pass
        # print("del")
        # _workspace_cache_per_plan.append(self["_ws_cache"])

def _make_plan_info(
    packed_work_range, packed_work_info,
    kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
    num_kv_splits, workspace_o, workspace_lse,
    max_qo_len, predicted_speedup, num_kv_splits_per_row,
    qo_segment_offsets, kv_segment_offsets, kv_page_indptr, max_k_tiles,
    qo_segment_lens, kv_segment_lens, qo_offset,
    pack_factor, orig_num_qo_heads,
    qo_len_uniform, cute_workspace_buffer,
    kv_page_indptr_local_last=None,
    block_table=None, block_table_row_stride=0, block_table_row_begin=0,
    index_rank=0, index_world_size=1, index_rows_per_rank=0,
    plan_workspace=None, plan_workspace_epoch=None
):
    # ws = _workspace_cache_per_plan.pop()
    # print("pop")
    return PlanInfo({
        # "_ws_cache" : ws,
        "packed_work_range": packed_work_range,
        "packed_work_info": packed_work_info,
        "kv_tile_begin_indices": kv_tile_begin_indices,
        "kv_tile_end_indices": kv_tile_end_indices,
        "kv_split_indices": kv_split_indices,
        "num_kv_splits": num_kv_splits,
        "workspace_o": workspace_o,
        "workspace_lse": workspace_lse,
        "max_qo_len": max_qo_len,
        "predicted_speedup": predicted_speedup,
        "num_kv_splits_per_row": num_kv_splits_per_row,
        "qo_segment_offsets": qo_segment_offsets,
        "kv_segment_offsets": kv_segment_offsets,
        "kv_page_indptr": kv_page_indptr,
        "max_k_tiles": max_k_tiles,
        "qo_segment_lens": qo_segment_lens,
        "kv_segment_lens": kv_segment_lens,
        "qo_offset": qo_offset,
        "pack_factor": pack_factor,
        "orig_num_qo_heads": orig_num_qo_heads,
        "qo_len_uniform": qo_len_uniform,
        "cute_workspace_buffer": cute_workspace_buffer,
        # Last entry of `kv_page_indptr`, i.e. this rank's total local page
        # count, carried as a HOST int taken from the CPU list the tensor was
        # built from.  The execution-time shape guard in `_fmha_sm100` reads
        # this instead of `kv_page_indptr[-1].item()`, so no plan build does a
        # D->H round trip on the per-step submission path.
        "kv_page_indptr_local_last": kv_page_indptr_local_last,
        # ---- refined-icp-v1: THE DIRECT COMPOUND-PAGE TABLE ---------------------
        # `DIRECT_TABLE_CONTRACT.md` §5.1, verbatim.  In direct-table mode the two
        # entries ABOVE are both None -- `kv_page_indptr` because the table plus a
        # device-derived bound replaces the CSR, and `kv_page_indptr_local_last`
        # because it exists only to let the execution-time guard read the CSR's
        # last entry without a D->H sync, and there is no CSR to read.
        #
        # `block_table` is the ORDINARY rectangular table's stable base: offset 0,
        # so `data_ptr()` is invariant to the live request count (I-6).
        # `block_table_row_stride` is an ADDRESS PITCH -- `max_blocks_per_req`, a
        # startup constant -- and is NEVER a live page count (I-1): the tail of a
        # row holds physical page IDs left by evicted requests, which read cleanly
        # and answer plausibly.  The live bound is `kv_segment_lens`, turned into a
        # fragment count ON THE DEVICE by the loader.
        #
        # `block_table_row_begin` is the row this plan's batch index 0 maps to.
        # "Batch index IS table row" is true for the FIRST query chunk and false
        # for every later one: the indexer chunks an invocation by query rows
        # (`icp_outer_chunk_tokens` is 1024 at 8k but 128 at 1M), so a 512-token
        # decode graph at 1M is four plans and the fourth starts at row 384.  A
        # HOST int, not a mapping tensor, and I-3 admits it because uniform decode
        # query length makes `t0 // query_len` a PER-GRAPH constant that does not
        # move with the live request count -- unlike the base pointer, which is
        # why I-6 pins that at storage offset 0 and not this.
        #
        # The three index_* fields are host startup constants.  They are the same
        # quantities as `icp_rank` / `icp_c` / `128 // icp_c` under the contract's
        # names; both spellings are stored so neither the contract's reader nor
        # the existing ICP code has to translate, and `_fmha_sm100` asserts the
        # pair agree rather than trusting that they were filled in consistently.
        "block_table": block_table,
        "block_table_row_stride": block_table_row_stride,
        "block_table_row_begin": block_table_row_begin,
        "index_rank": index_rank,
        "index_world_size": index_world_size,
        "index_rows_per_rank": index_rows_per_rank,
        # The caller-owned workspace this plan's device buffers live in, and the
        # epoch it was built at.  None when the plan owns freshly allocated
        # buffers (the unmigrated path).  `_fmha_sm100` compares the epoch
        # against the workspace's current one, so a plan whose storage has since
        # been rebound to a newer plan FAILS instead of reading the newer plan's
        # metadata.  It is a host int compare: no device read, no sync.
        "_plan_workspace": plan_workspace,
        "_plan_workspace_epoch": plan_workspace_epoch,
        "MM-SA-Nv":False
    })


def _call_plan(qo_segment_offsets, qo_segment_lens, kv_segment_lens,
               packed_work_range, packed_work_info,
               qo_tile_size, kv_tile_size, num_qo_heads, num_ctas, causal,
               qo_offset, num_kv_splits,
               kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
               chunk_size, out_max_sm_cost, num_kv_splits_per_row, workspace_lse, lse_total_size,
               pack_factor, cuda_stream=None,
               ):

    plan_module = get_plan_fn()

    if cuda_stream is None:
        cuda_stream = torch.cuda.current_stream().cuda_stream
    plan_module.plan(
        qo_segment_offsets, qo_segment_lens, kv_segment_lens,
        packed_work_range, packed_work_info,
        qo_tile_size, kv_tile_size, num_qo_heads, num_ctas, causal,
        qo_offset, num_kv_splits,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        chunk_size, out_max_sm_cost, num_kv_splits_per_row,
        cuda_stream,
        workspace_lse, lse_total_size,
        pack_factor,
    )

def _kernel_has(symbol: str) -> bool:
    """Whether the kernel driver header carries `symbol`."""
    hdr = (
        Path(__file__).resolve().parent
        / "csrc/include/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
    )
    try:
        return symbol in hdr.read_text()
    except OSError:
        return False


# Derived, not declared: an overlay whose Python half is patched and whose
# kernel half is not takes a CUDA illegal memory access on any short prompt.
_FMHA_ICP_EMPTY_WORK_SKIP = _kernel_has("is_empty_work")


def _overlay_file_has(relpath: str, symbol: str) -> bool:
    """Whether the overlay file `relpath` carries `symbol`."""
    f = Path(__file__).resolve().parent / relpath
    try:
        return symbol in f.read_text()
    except OSError:
        return False


# ---- refined-icp-v1: the OUT-OF-BAND VALIDITY PLANE --------------------------------
# OUTPUT-ABI VERSION of the ICP score wave.  This is the explicit version, not a
# comment:
#   1 = score plane only; emptiness signalled IN BAND with -inf.  SUPERSEDED.  -inf
#       is a legal representable score (ABI refined-icp-v1.abi.1 W12), so it cannot
#       encode invalidity, and NaN is a contract failure rather than a spare value.
#       There is no compatibility mode; see the assert in the icp_c > 1 branch below.
#   2 = score plane + out-of-band uint8 validity plane, shape-matched and
#       caller-owned with explicit strides (W4/W5/W8/W9).  Every advertised cell of
#       BOTH planes is written on every invocation, so no blanket initialisation pass
#       is required (W7).
_FMHA_ICP_SCORE_ABI = "refined-icp-v1.abi.1/icp-score-wave.v2"
_FMHA_ICP_SCORE_ABI_VERSION = 2
# DERIVED, not declared -- the same discipline as _FMHA_ICP_EMPTY_WORK_SKIP above, and
# for a sharper reason: a Python half that advertises the plane over a kernel half
# that ignores it hands the consumer a buffer the kernel never wrote, and the consumer
# cannot tell, because an unwritten byte is a perfectly good uint8.
_FMHA_HAS_ICP_VALIDITY_PLANE = (
    _overlay_file_has(
        "csrc/include/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp", "ptr_ValidScore")
    and _overlay_file_has(
        "csrc/include/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp", "ptr_ValidScore")
    and _overlay_file_has("csrc/fmha_sm100_variant_run.cu.jinja", "maybe_valid_score")
    and _overlay_file_has("csrc/fmha_sm100_inst.jinja", "valid_score_ptr")
)


# ---- refined-icp-v1: the DIRECT COMPOUND-PAGE TABLE ---------------------------------
# INPUT-ABI version of the direct-table entry (`DIRECT_TABLE_CONTRACT.md` §5).  It is
# SEPARATE from `_FMHA_ICP_SCORE_ABI_VERSION`, which versions the OUTPUT wave: the two
# move independently and conflating them would make one of them unversionable.
#   1 = block table base + row pitch + PER-CHUNK ROW ORIGIN + device KV lengths; no
#       `kv_page_indptr`, no host page count.  The packed-list entry is NOT retired --
#       the `icp_c == 1` production route uses the same CSR -- so this is an added mode,
#       not a cutover.
# The row origin is part of v1, not a later addition: without it "batch index is table
# row" silently holds for the first query chunk only, so `icp_block_table_row_begin` is
# one of the symbols the capability check below requires.
_FMHA_ICP_DIRECT_TABLE_ABI = "refined-icp-v1.direct-table.v1"
_FMHA_ICP_DIRECT_TABLE_ABI_VERSION = 1
# DERIVED, not declared, for the same reason `_FMHA_HAS_ICP_VALIDITY_PLANE` is -- and it
# is ASSERTED at every use, which `_FMHA_ICP_EMPTY_WORK_SKIP` above is not.  The failure
# this prevents is specific and silent: a kernel half without these symbols still ACCEPTS
# the call (the two tail arguments would simply be absent from its signature, or present
# and unread), leaves `icp_block_table` null, and therefore takes the PACKED path -- on a
# plan that carries no packed list, i.e. with `kv_page_indptr == nullptr`.  Every file
# below is one whose omission produces that outcome, so all five are required rather than
# one representative.
_FMHA_HAS_ICP_DIRECT_TABLE = (
    _overlay_file_has(
        "csrc/include/sm100_fmha_load_tma_warpspecialized.hpp", "icp_local_blocks_exact")
    and _overlay_file_has(
        "csrc/include/sm100_fmha_load_tma_warpspecialized.hpp",
        "icp_block_table_row_begin")
    and _overlay_file_has("csrc/include/fmha_cutlass_sm100.cuh", "icp_install_direct_table")
    and _overlay_file_has("csrc/fmha_sm100_params.h", "icp_block_table_ptr")
    and _overlay_file_has("csrc/fmha_sm100_variant_run.cu.jinja", "maybe_icp_block_table")
    and _overlay_file_has("csrc/fmha_sm100_inst.jinja", "icp_block_table_ptr")
)


def _assert_icp_direct_table_supported():
    """Refuse a direct-table call whose kernel half cannot consume one."""
    assert _FMHA_HAS_ICP_DIRECT_TABLE, (
        "this overlay's Python half advertises the direct compound-page table "
        f"({_FMHA_ICP_DIRECT_TABLE_ABI}) but its kernel half does not carry "
        "`icp_block_table_row_begin` / `icp_local_blocks_exact`; refusing to run "
        "rather than silently fall back to the packed-list path on a plan that has "
        "no packed list, or address every query chunk as if it were the first"
    )


def _qo_tile_size_for(max_qo_len: int, icp_c: int) -> int:
    """The Q tile of the dispatched variant.  UNDER ICP IT IS PINNED TO 128.

    This is ONE function because the value is chosen twice -- once in
    `_fmha_sm100_plan_impl` (it feeds `_call_plan`'s work decomposition) and once
    in `_fmha_sm100` (it selects the variant) -- and the two MUST agree.

    WHY ICP PINS IT.  `jit.py`'s qo_tile axis binds qo_tile 128 -> thread_shape
    `_1, _2, _1` and qo_tile 256 -> `_2, _1, _1`, and the KV extent the mainloop
    actually computes on is `TileShapeQK = shape_div(TileShape, ThreadShape)`, i.e.
    tile_kv / thread_shape[1] (sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:226).
    ICP's page axis sets tile_kv `_128` at page 64 and `_64` at page 32, which gives
    effective_tile_kv == KVPageSize ONLY while thread_shape[1] == 2.  At qo_tile 256
    the KV split disappears, effective_tile_kv becomes 2 x KVPageSize, and the
    loader's `logical_page = tile_idx / tiles_per_page` with `tiles_per_page == 1`
    (sm100_fmha_load_tma_warpspecialized.hpp:725-728,770) then feeds ONE page to a
    tile that spans TWO -- while `get_full_trip_count`'s ICP arm
    (:266-282) divides the SAME local fragment count by the doubled tile, so half
    (W=2) of this rank's pages are never visited at all.

    MEASURED, not argued (tests/integration/test_icp_fragment_maxscore.py,
    kv_len=1024, GB300, fp8; `qo_tile` is the tile the ICP arm ran on WITHOUT the
    pin below, and every one of these points is exact WITH it):

        Q      qo_tile   W=2                       W=4
        1,8    128       exact                     exact
        128    128       exact                     exact
        129    256       2060/66048 cells wrong    2060/66048 wrong
        192    256       2894/98304 wrong          2894/98304 wrong
        256    256       3844/131072 wrong         3844/131072 wrong

    and the failure is SILENT, not loud: the W5 completion pass still marks those
    columns VALID (it derives validity from the row predicate V2/V3, which is
    correct and tile-independent), while the main loop never writes their score.
    The harness pre-fills the score plane with NaN and the NaNs survive; a
    production caller pre-fills -inf, or reuses a buffer, and reads a plausible
    number that the validity plane vouches for.  Some columns are ALSO
    finite-and-wrong (max |diff| 19.6 at Q=256), i.e. this is not only a
    missing-write bug.

    The cost of pinning is a smaller Q tile on ICP prefill, not a wrong answer, and
    ICP is OnlyScore-only -- no output tensor, no split-KV -- so the 256-wide tile
    buys it nothing that is proven correct.
    """
    if icp_c > 1:
        return 128
    return 128 if max_qo_len <= 128 else 256


def _make_icp_geometry_descriptor(qo_segment_lens, icp_c: int, icp_rank: int):
    """Encode ICP geometry in a retained tensor's shape without device work.

    The FFI reads only ``size(2)`` for ``OnlyScoreIcp``. Its data pointer is
    never dereferenced in that mode, so a zero-stride view of the first
    retained sequence length carries the same rank/world encoding as an
    allocated dummy tensor. This allocates no CUDA storage and launches no
    fill kernel, including when a plan is built in caller-owned workspace.
    """
    assert qo_segment_lens.numel() > 0, "ICP requires a nonempty request batch"
    return qo_segment_lens.as_strided(
        (1, 1, icp_rank * 16 + icp_c), (0, 0, 0))


def _assert_device_plan_current(plan_info):
    """A device plan must have been refreshed by a fused-writer launch since it
    was built (one host compare: no device read, no synchronization)."""
    device_plan = plan_info.get("_device_plan")
    if device_plan is not None:
        device_plan.assert_current(plan_info["_device_plan_launches"])


def _fmha_sm100_plan(*args, icp_c: int = 1, icp_rank: int = 0, **kwargs):
    """Stamp the ICP geometry on the plan; inert at ``icp_c == 1``."""
    info = _fmha_sm100_plan_impl(*args, icp_c=icp_c, icp_rank=icp_rank, **kwargs)
    if icp_c == 1:
        # The C == 1 production path pays ONE extra Python frame and nothing
        # else: no dict stores, and in particular no `.item()` D->H sync. The
        # `.get()` defaults in `_fmha_sm100` cover the missing keys.
        return info
    info["icp_c"] = icp_c
    info["icp_rank"] = icp_rank
    if "qo_segment_lens" in info:
        info["icp_geometry_descriptor"] = _make_icp_geometry_descriptor(
            info["qo_segment_lens"], icp_c, icp_rank)
    # N4 (GPU_EXECUTION_NONBLOCKING_REVIEW): this used to be
    # `int(info["kv_page_indptr"][-1].item())`, a D->H sync on EVERY chunk plan
    # build -- prefill, decode and Eagle3 verification alike.  The planner now
    # carries the identical host-side value (`kv_page_indptr_list[-1]`, the
    # final prefix-sum entry the device tensor is copied from), so the
    # execution-time shape guard in `_fmha_sm100` is unchanged while nothing on
    # the submission path reads a CUDA tensor.  `setdefault` covers the
    # `sparse_fmha_plan` early return, whose plain dict has no `kv_page_indptr`
    # and therefore scored None under the old code too.
    info.setdefault("kv_page_indptr_local_last", None)
    return info


def _fmha_sm100_plan_impl(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int = -1,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    num_kv_splits: int = -1,
    page_size: int = -1,
    output_maxscore: bool = False,
    kv_block_num: int = -1,
    usable_SM_count: int = -1,
    causal: bool = True,
    sparse_kernel_mode: str = 'auto',
    use_fp8_kvcache: bool = False,
    device = None,
    stream = None,
    *,
    icp_c: int = 1,
    icp_rank: int = 0,
    icp_block_table: Optional[torch.Tensor] = None,
    icp_block_table_row_stride: int = 0,
    icp_block_table_row_begin: int = 0,
    plan_workspace: Optional["PlanWorkspace"] = None,
    device_plan: Optional["IcpDevicePlanSlot"] = None,
):
    device = torch.cuda.current_device() if device is None else device
    _reset_np_staging()

    # Open the build BEFORE any buffer is bound: `begin_plan` is what makes the
    # workspace's previous plan detectably stale and clears the one-region-per-
    # buffer check.  `_ws` is threaded to every per-plan allocation below; the
    # process-global `_alloc_workspace_buf` buffers (cutlass workspace,
    # plan_cost, workspace_o) are deliberately NOT routed here -- they are
    # already stable-pointer by construction and are execution scratch rather
    # than plan metadata.
    _ws = plan_workspace
    _ws_epoch = None if _ws is None else _ws.begin_plan()

    # ---- refined-icp-v1: THE DIRECT COMPOUND-PAGE TABLE ---------------------------
    # An ADDED plan mode selected by the presence of the table, checked here once so
    # that every later branch can test one boolean.  Everything below is host metadata
    # -- dtype, ndim, strides, storage offset -- so none of it reads device memory,
    # allocates, or synchronises.
    icp_direct_table = icp_block_table is not None
    if icp_direct_table:
        _assert_icp_direct_table_supported()
        assert icp_c >= 2 and 0 <= icp_rank < icp_c, (
            "the direct compound-page table is an ICP input; got "
            f"icp_c={icp_c}, icp_rank={icp_rank}")
        assert icp_block_table.dtype == torch.int32, (
            f"block_table must be int32, got {icp_block_table.dtype}")
        assert icp_block_table.dim() == 2, (
            "block_table must be 2D [num_requests, max_blocks_per_req]; got "
            f"{tuple(icp_block_table.shape)}")
        assert icp_block_table.is_cuda, "block_table must be a CUDA tensor"
        # I-6, ASSERTED rather than documented.  A view that begins at the first live
        # request has a non-zero storage offset, so its `data_ptr()` MOVES with the
        # batch -- which is precisely what a graph capture freezes wrong, and what
        # `block_table[num_decodes:]` (the prefill band) would be.  This one host
        # integer is the entire check.
        #
        # It is the BASE that is pinned here, not the row origin.  A per-chunk row
        # origin is passed separately as `icp_block_table_row_begin` and is legal
        # because it is a per-graph constant; folding it into the pointer instead
        # would be exactly the moving base this refuses.
        assert icp_block_table.storage_offset() == 0, (
            "block_table must be passed as its STABLE BASE (storage_offset == 0, i.e. "
            f"`block_table[:num_decodes]`), got offset {icp_block_table.storage_offset()}."
            "  A view beginning at the first live request moves with the live request "
            "count and reintroduces the request mapping this entry exists to delete "
            "(contract I-6); the prefill band `block_table[num_decodes:]` is for the "
            "same reason NOT capture-eligible and not accepted here.  A per-chunk row "
            "origin belongs in icp_block_table_row_begin, not in the pointer.")
        icp_block_table_row_begin = int(icp_block_table_row_begin)
        _bt_batch = int(qo_segment_lens.shape[0])
        assert icp_block_table_row_begin >= 0, (
            "icp_block_table_row_begin is the table row this plan's batch index 0 maps "
            f"to and must be non-negative; got {icp_block_table_row_begin}")
        # The rows this plan addresses are [row_begin, row_begin + batch).  A short
        # table does not fault: the row it lands on is another request's or another
        # chunk's, which is in bounds and reads cleanly.
        assert icp_block_table_row_begin + _bt_batch <= int(icp_block_table.shape[0]), (
            f"this plan addresses block_table rows [{icp_block_table_row_begin}, "
            f"{icp_block_table_row_begin + _bt_batch}) but the table has "
            f"{int(icp_block_table.shape[0])} rows")
        icp_block_table_row_stride = int(icp_block_table_row_stride)
        assert icp_block_table_row_stride > 0, (
            "block_table_row_stride is the row ADDRESS PITCH (max_blocks_per_req) and "
            f"must be positive; got {icp_block_table_row_stride}")
        assert icp_block_table.stride(0) == icp_block_table_row_stride, (
            f"block_table_row_stride={icp_block_table_row_stride} != the tensor's own "
            f"row stride {icp_block_table.stride(0)}.  The pitch is an ADDRESS, not a "
            "column count: a padded table whose logical width differs from its stride "
            "would address every row but the first at the wrong offset -- in bounds, "
            "wrong tenant.")
        assert icp_block_table.stride(1) == 1, (
            f"block_table must be row-contiguous, got stride(1)={icp_block_table.stride(1)}")
        assert kv_block_num <= 0, (
            "the direct compound-page table and the sparse top-k block map are two "
            "different page directories; they cannot both drive one call")
        assert page_size > 0 and page_size * icp_c == 128, (
            f"direct-table ICP needs page_size * icp_c == 128 (ABI G13/P3); got "
            f"page_size={page_size}, icp_c={icp_c}")
    elif icp_block_table_row_stride or icp_block_table_row_begin:
        raise ValueError(
            "icp_block_table_row_stride / icp_block_table_row_begin were given without "
            "icp_block_table; they select nothing on their own and silently doing the "
            "packed thing would hide the caller's mistake")

    if icp_direct_table:
        # Plan contents come only from the fused writer's plan CTAs; the host
        # staging copy below is not reachable for this mode.
        if device_plan is None:
            raise PlanWorkspaceError(
                "direct-table ICP plans are device-derived: pass device_plan="
                "IcpDevicePlanStore.slot(...) and hand the store to the first "
                "sparse writer.  Host-staged plan contents are not supported.")
        if qo_segment_lens.is_cuda or kv_segment_lens.is_cuda:
            raise PlanWorkspaceError(
                "device-plan sizing reads host tensors only; got CUDA lengths")
        return _icp_device_plan_info(
            slot=device_plan, qo_lens=qo_segment_lens.tolist(),
            kv_lens=kv_segment_lens.tolist(), num_qo_heads=num_qo_heads,
            num_kv_splits=num_kv_splits, usable_SM_count=usable_SM_count,
            causal=causal, output_maxscore=output_maxscore, device=device,
            icp_direct_fields={
                "block_table": icp_block_table,
                "block_table_row_stride": icp_block_table_row_stride,
                "block_table_row_begin": icp_block_table_row_begin,
                "index_rank": icp_rank,
                "index_world_size": icp_c,
                "index_rows_per_rank": 128 // icp_c,
            })
    if device_plan is not None:
        raise ValueError(
            "device_plan is the direct-table ICP plan; it requires "
            "icp_block_table")

    qo_lens = qo_segment_lens.tolist()
    max_qo_len_orig = max(qo_lens) if qo_lens else 0

    if kv_block_num > 0 and (sparse_kernel_mode == 'prefill' or (sparse_kernel_mode == 'auto' and max_qo_len_orig > _prefill_qlen_threshold(True))):
        # print("Nv-Prefill")
        if _ws is not None:
            # Refused, not ignored.  `sparse_fmha_plan` is a different planner
            # with its own buffers; accepting a workspace here and dropping it
            # would leave the caller believing its plan pointers are stable when
            # they are not -- the same silent class the workspace exists to end.
            raise PlanWorkspaceError(
                "plan_workspace is not supported on the sparse-prefill route "
                "(`MM-SA-Nv`): that plan is built by `sparse_fmha_plan`, which "
                "does not allocate from this workspace.  Omit plan_workspace "
                "for this route rather than assuming its buffers are stable.")
        if icp_direct_table:
            # `sparse_fmha_plan` is a DIFFERENT planner with a different page
            # directory (`kv_block_indexes`), and `sparse_fmha` raises on ICP
            # arguments anyway.  Accepting the table here and dropping it is the
            # one failure mode that is finite, plausible and silent.
            raise ValueError(
                "the direct compound-page table is not supported on the "
                "sparse-prefill route (`MM-SA-Nv`): that plan is built by "
                "`sparse_fmha_plan`, which has its own page directory.")
        qo_segment_lens = qo_segment_lens.to(device)
        kv_segment_lens = kv_segment_lens.to(device)
        qo_offset = qo_offset.to(device)
        return sparse_fmha_plan(qo_segment_lens=qo_segment_lens, kv_segment_lens=kv_segment_lens,
                num_qo_heads=num_qo_heads, causal=causal, qo_offset=qo_offset,
                num_kv_splits=num_kv_splits, page_size=page_size, output_maxscore=output_maxscore,
                kv_block_num=kv_block_num, num_kv_heads=num_kv_heads, usable_SM_count=usable_SM_count,
                use_fp8_kvcache=use_fp8_kvcache)

    _new_ws_cache()

    cute_workspace_buffer = _alloc_workspace_buf(_BuffTag.fmha_sm100_cutlass_workspace, 32 * 1024 * 1024, device, torch.uint8)

    cuda_stream = stream

    num_ctas = _get_num_cta(device)
    if usable_SM_count > 0:
        num_ctas = min(usable_SM_count, num_ctas)

    orig_num_qo_heads = num_qo_heads
    # `SparseAttnMode::OnlyScoreIcp` is pack_factor == 1 only.
    if icp_c > 1:
        pack_factor = 1
    else:
        pack_factor = _compute_pack_factor(max_qo_len_orig, num_qo_heads, num_kv_heads)
    qo_len_uniform = len(qo_lens) > 0 and min(qo_lens) == max_qo_len_orig
    if pack_factor > 1:
        num_qo_heads = num_qo_heads // pack_factor

    kv_lens = kv_segment_lens.tolist()
    kv_page_indptr_list = None
    if kv_block_num > 0 and page_size > 0:
        total_q = sum(qo_lens)
        total_q_packed = total_q * pack_factor if pack_factor > 1 else total_q
        assert total_q_packed * num_qo_heads <= 65536
        qo_offset_in = qo_offset.tolist() if qo_offset is not None else None
        qo_lens, kv_lens, qo_offset, kv_page_indptr_list = \
            _expand_for_per_token_sparse(qo_lens, kv_lens, qo_offset_in, page_size, pack_factor)
    elif kv_block_num > 0 and page_size <= 0:
        print("[Error] Sparse mode must be used together with paged kv!")
    else:
        if pack_factor > 1:
            qo_lens = [q * pack_factor for q in qo_lens]
        if page_size > 0 and not icp_direct_table:
            acc = 0
            kv_page_indptr_list = [0]
            for k in kv_lens:
                gb = (k + page_size - 1) // page_size
                # Under ICP the page table handed to the kernel is THIS rank's
                # local one, so count only the block-cyclic pages this rank
                # owns: ceil((gb - icp_rank) / icp_c), floored at zero for a
                # request too short to reach this rank.
                if icp_c == 1:
                    acc += gb
                else:
                    _nl = -(-(gb - icp_rank) // icp_c)
                    acc += _nl if _nl > 0 else 0
                kv_page_indptr_list.append(acc)
        elif icp_direct_table:
            # ---- refined-icp-v1: the CSR is GONE, not merely unused ------------
            # `kv_page_indptr_list` stays None, so the plan carries neither the
            # device indptr nor `kv_page_indptr_local_last`.  The loop above is
            # exactly the host-side rank-local page arithmetic the contract
            # deletes (§7): the SAME function as the kernel's
            # `local_blocks(L, r)`, but evaluated on the host from lengths that
            # are a capture-time snapshot.  Computing it here and then ignoring
            # it would leave a second, staler page count alive in the plan.
            #
            # ONE host check survives, and only because it is free: the row pitch
            # must be able to hold the blocks the lengths THIS PLAN WAS BUILT
            # FROM imply.  `kv_lens` is already a host list at this point, so
            # this adds no device read.  It is NOT a replay guarantee -- under
            # capture these lengths are the capture-time ones (4, not 150k), and
            # sizing the row from `max_model_len` remains the caller's job (I-2).
            # `_blk` is the compound page extent.  It is re-derived from `page_size`
            # only because the assert above pins `page_size * icp_c == 128`, so this
            # is that same constant and not a second authority.  NO BOUND is derived
            # from `page_size` in this mode: the CSR it used to feed is gone, and the
            # kernel takes R from the paged tensor's mode-0 extent.
            _blk = page_size * icp_c
            _need = -(-max(kv_lens) // _blk) if kv_lens else 0
            assert _need <= icp_block_table_row_stride, (
                f"block_table_row_stride={icp_block_table_row_stride} cannot hold the "
                f"{_need} compound pages the longest sequence in this plan "
                f"({max(kv_lens)} tokens at {_blk} tokens/page) needs.  Size the row "
                "from max_model_len at startup; it is an address pitch and the kernel "
                "will not truncate the scan to fit it.")

    acc = 0
    qo_offsets = [0]
    for q in qo_lens:
        acc += q
        qo_offsets.append(acc)
    acc = 0
    kv_offsets = [0]
    for k in kv_lens:
        acc += k
        kv_offsets.append(acc)

    max_kv_len = max(kv_lens)
    qo_offset_list = qo_offset if isinstance(qo_offset, list) else qo_offset.tolist()
    if not causal:
        qo_offset_list = [max_kv_len] * len(kv_lens)

    qo_segment_offsets = _plan_buf_from_list(
        qo_offsets, device, _ws, _BuffTag.plan_qo_segment_offsets)
    kv_segment_offsets = _plan_buf_from_list(
        kv_offsets, device, _ws, _BuffTag.plan_kv_segment_offsets)
    qo_offset = _plan_buf_from_list(
        qo_offset_list, device, _ws, _BuffTag.plan_qo_offset)
    # ---- refined-icp-v1: THIS BUFFER IS THE LIVE BOUND, AND IT MAY BE TIGHTENED ---
    # The direct-table entry makes the DEVICE contents of this buffer the sole
    # authority for the fragment bound, and the caller is expected to overwrite them
    # IN PLACE with exact lengths after the plan is built.  It must: vLLM's
    # `seq_lens_cpu_upper_bound` -- which is what `kv_lens` above comes from -- still
    # counts REJECTED Eagle3 drafts, so as a bound it is optimistic, and at every
    # compound-page boundary a rejected draft crosses, the kernel would buy one column
    # past the request's live end (I-1: reads cleanly, answers plausibly).
    #
    # Nothing in this planner or in the loader may assume the device contents still
    # equal `kv_lens`.  Audited, and the asymmetry is safe in one direction only --
    # device <= host:
    #   * `kv_segment_offsets` (prefix sum of the host list) is read by the kernel ONLY
    #     on the non-paged arm; ICP is always paged.
    #   * `max_k_tiles` / the score-plane width are sized from the host list, so the
    #     advertised wave is WIDER than the live one and the surplus columns are marked
    #     invalid by the W5 pass.  Over-advertising is the safe direction (it is the P2
    #     residual, not a correctness bug); under-advertising would not be.
    #   * the work decomposition (`_call_plan`) enumerates (batch, head, qo_tile) and is
    #     KV-length-independent at num_kv_splits == 1, which ICP asserts.  A longer host
    #     length can only produce work items that turn out empty, and `is_empty_work`
    #     already retires those from device data.
    #   * every kernel-side consumer of the length -- the loader's page clamp and trip
    #     count, the mainloop's trip count and `icp_local_blocks`, `is_empty_work`, the
    #     W5 validity pass -- reads `get<1>(problem_shape)`, i.e. THIS buffer, so they
    #     all tighten together and stay step-locked.
    # The one thing that does NOT self-correct is `qo_offset` below: it is staged from
    # a host list that the caller may have derived as `kv_len - qo_len` from the SAME
    # optimistic bound.  A caller that tightens the lengths must also supply exact
    # causal offsets (`q_offset_override`, or by overwriting `plan["qo_offset"]`),
    # exactly as the bound/exact split works for query positions.  Leaving the stale
    # offsets in place shifts the causal origin; there is no device-free way to detect
    # it here, so it is a stated precondition of the direct-table entry.
    kv_segment_lens = _plan_buf_from_list(
        kv_lens, device, _ws, _BuffTag.plan_kv_segment_lens)
    kv_page_indptr = _plan_buf_from_list(
        kv_page_indptr_list, device, _ws, _BuffTag.plan_kv_page_indptr
    ) if kv_page_indptr_list is not None else None
    # Same quantity as `kv_page_indptr[-1]` -- `_plan_buf_from_list` is an
    # element-wise int32 copy of exactly this list -- but read on the HOST,
    # before the list is staged to the device.  Sourcing the guard's scalar
    # here is what removes the per-plan-build `.item()` D->H sync.
    kv_page_indptr_local_last = (
        int(kv_page_indptr_list[-1]) if kv_page_indptr_list is not None else None)

    # ---- refined-icp-v1: the direct-table half of the plan, built ONCE ------------
    # Both `_make_plan_info` call sites below (split-KV and not) have to carry these,
    # and a plan that carried the table on one path and not the other would take the
    # packed path in the kernel with no packed list to take.  One dict, splatted at
    # both, is what makes that unrepresentable.  `index_rows_per_rank` is R = 128 // W
    # and equals the dispatched `page_size` by the assert above.
    _icp_direct_fields = dict(
        block_table=icp_block_table,
        block_table_row_stride=icp_block_table_row_stride if icp_direct_table else 0,
        block_table_row_begin=icp_block_table_row_begin if icp_direct_table else 0,
        index_rank=icp_rank,
        index_world_size=icp_c,
        index_rows_per_rank=(128 // icp_c) if icp_direct_table else 0,
    )

    plan_kv_lens_list = kv_lens
    plan_qo_offset = qo_offset
    if kv_block_num > 0 and page_size > 0:
        plan_kv_lens_list = [kv_block_num * page_size] * len(kv_lens)
        plan_qo_offset = None

    max_qo_len = max(qo_lens)
    qo_tile_size = _qo_tile_size_for(max_qo_len, icp_c)
    kv_tile_size = 256 if qo_tile_size == 128 else 128

    total_qo_len = qo_offsets[-1]

    max_k_tiles = math.ceil(math.ceil(max_kv_len / 128) / 128) * 128 if output_maxscore else -1
    maxscore_elems = num_qo_heads * max_k_tiles * total_qo_len
    if output_maxscore and maxscore_elems > (1 << 31):
        print(f"Too huge setting to output maxscore!")
        max_k_tiles = -1

    # `and icp_c == 1` is the OTHER half of pinning qo_tile above, not a separate
    # tidy-up: this branch AUTO-SELECTS split-KV whenever the Q tile is 128, and
    # `_fmha_sm100` asserts ICP x split-KV UNSUPPORTED (the W5 completion pass
    # assumes one work item covers the whole advertised wave).  Before the pin it
    # was reachable under ICP only at max_qo_len <= 128; after it, at every Q.
    # Without this guard the pin would trade a silent tile bug for a hard assert.
    # Falling through lands on `elif num_kv_splits < 1: num_kv_splits = 1`, which
    # is exactly the num_kv_splits the ICP path requires.
    if num_kv_splits < 1 and qo_tile_size == 128 and icp_c == 1:
        group_iters = []
        batch_size = len(qo_lens)
        for b_idx in range(batch_size):
            ql_b, kl_b = qo_lens[b_idx], plan_kv_lens_list[b_idx]
            off_q = kl_b - ql_b
            for t in range(math.ceil(ql_b / qo_tile_size)):
                packed_q_end = (t + 1) * qo_tile_size
                q_end = (packed_q_end - 1) // pack_factor + 1 if pack_factor > 1 else packed_q_end
                if causal and q_end + off_q <= 0:
                    continue
                eff_kv = min(q_end + off_q, kl_b) if causal else kl_b
                group_iters.append(math.ceil(max(eff_kv, 0) / kv_tile_size))

        if not group_iters:
            num_kv_splits = 1
        elif len(group_iters) * num_qo_heads > 4096:
            # Too many tiles for split-KV smem — fall back to nosplit greedy
            num_kv_splits = 1
        else:
            total_iters = sum(group_iters) * num_qo_heads
            avg_iters = max(2, (total_iters + num_ctas - 1) // num_ctas + 3)
            chunk_size = avg_iters
            max_pieces = max(math.ceil(g / chunk_size) for g in group_iters)
            max_kv_splits = min(2 * max(max_pieces, 1), 64)

            max_work_items = _PLAN_MAX_WORK_ITEMS_PER_SPLIT * max_kv_splits
            
            packed_work_range = _alloc_perplan_buf(_BuffTag.packed_work_range, num_ctas, device, torch.int64, _ws)
            packed_work_info = _alloc_perplan_buf(_BuffTag.packed_work_info, max_work_items, device, torch.int64, _ws)
            kv_tile_begin_indices = _alloc_perplan_buf(_BuffTag.kv_tile_begin_indices, max_work_items, device, torch.int32, _ws)
            kv_tile_end_indices = _alloc_perplan_buf(_BuffTag.kv_tile_end_indices, max_work_items, device, torch.int32, _ws)
            kv_split_indices = _alloc_perplan_buf(_BuffTag.kv_split_indices, max_work_items, device, torch.int32, _ws)
            num_kv_splits_per_row = _alloc_perplan_buf(_BuffTag.num_kv_splits_per_row, total_qo_len, device, torch.int32, _ws)
            plan_cost = _alloc_workspace_buf(_BuffTag.plan_cost, 2, device, dtype=torch.float32)

            lse_total_size = max_kv_splits * total_qo_len * num_qo_heads
            workspace_lse = _alloc_perplan_buf(_BuffTag.workspace_lse, lse_total_size, device, torch.float32, _ws)

            qo_segment_lens_gpu = _plan_buf_from_list(
                qo_lens, device, _ws, _BuffTag.plan_qo_segment_lens)
            plan_kv_lens_gpu = _plan_buf_from_list(
                plan_kv_lens_list, device, _ws, _BuffTag.plan_kv_lens)
            _call_plan(qo_segment_offsets,
                qo_segment_lens_gpu, plan_kv_lens_gpu,
                packed_work_range, packed_work_info,
                qo_tile_size, kv_tile_size,
                num_qo_heads, num_ctas, causal, plan_qo_offset,
                -max_kv_splits,
                kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
                chunk_size, plan_cost, num_kv_splits_per_row, workspace_lse, lse_total_size,
                pack_factor, cuda_stream)

            _cost = plan_cost.tolist()
            do_split = _cost[0] > 0
            predicted_speedup = _cost[1]

            if do_split:
                workspace_o = _alloc_workspace_buf(_BuffTag.workspace_o, total_qo_len * max_kv_splits * num_qo_heads * 128, device, dtype=torch.bfloat16)
            else:
                kv_tile_begin_indices = None
                kv_tile_end_indices = None
                kv_split_indices = None
                num_kv_splits_per_row = None
                workspace_o = None
                workspace_lse = None
                max_kv_splits = 1

            return _make_plan_info(
                packed_work_range=packed_work_range, packed_work_info=packed_work_info,
                kv_tile_begin_indices=kv_tile_begin_indices, kv_tile_end_indices=kv_tile_end_indices, 
                kv_split_indices=kv_split_indices, num_kv_splits=max_kv_splits, workspace_o=workspace_o, 
                workspace_lse=workspace_lse, max_qo_len=max_qo_len, predicted_speedup=predicted_speedup, 
                num_kv_splits_per_row=num_kv_splits_per_row, qo_segment_offsets=qo_segment_offsets, 
                kv_segment_offsets=kv_segment_offsets, kv_page_indptr=kv_page_indptr, max_k_tiles=max_k_tiles,
                qo_segment_lens=qo_segment_lens_gpu, kv_segment_lens=kv_segment_lens, qo_offset=qo_offset,
                pack_factor=pack_factor, orig_num_qo_heads=orig_num_qo_heads,
                qo_len_uniform=qo_len_uniform, cute_workspace_buffer=cute_workspace_buffer,
                kv_page_indptr_local_last=kv_page_indptr_local_last,
                **_icp_direct_fields,
                plan_workspace=_ws, plan_workspace_epoch=_ws_epoch)

        num_kv_splits = 1
    elif num_kv_splits < 1:
        num_kv_splits = 1
    elif qo_tile_size == 256:
        num_kv_splits = 1

    packed_work_range = _alloc_perplan_buf(_BuffTag.packed_work_range, num_ctas, device, torch.int64, _ws)
    max_work_items = _PLAN_MAX_WORK_ITEMS_PER_SPLIT * max(num_kv_splits, 1)
    packed_work_info = _alloc_perplan_buf(_BuffTag.packed_work_info, max_work_items, device, torch.int64, _ws)
    if num_kv_splits > 1:
        kv_tile_begin_indices = _alloc_perplan_buf(_BuffTag.kv_tile_begin_indices, max_work_items, device, torch.int32, _ws)
        kv_tile_end_indices = _alloc_perplan_buf(_BuffTag.kv_tile_end_indices, max_work_items, device, torch.int32, _ws)
        kv_split_indices = _alloc_perplan_buf(_BuffTag.kv_split_indices, max_work_items, device, torch.int32, _ws)
        num_kv_splits_per_row = _alloc_perplan_buf(_BuffTag.num_kv_splits_per_row, total_qo_len, device, torch.int32, _ws)
        workspace_o = _alloc_workspace_buf(_BuffTag.workspace_o, total_qo_len * num_kv_splits * num_qo_heads * 128, device, dtype=torch.bfloat16)

        lse_total_size = num_kv_splits * total_qo_len * num_qo_heads
        workspace_lse = _alloc_perplan_buf(_BuffTag.workspace_lse, lse_total_size, device, torch.float32, _ws)
    else:
        kv_tile_begin_indices = None
        kv_tile_end_indices = None
        kv_split_indices = None
        num_kv_splits_per_row = None
        workspace_o = None
        
        lse_total_size = 0
        workspace_lse = None

    qo_segment_lens_gpu = _plan_buf_from_list(
        qo_lens, device, _ws, _BuffTag.plan_qo_segment_lens)
    plan_kv_lens_gpu = _plan_buf_from_list(
        plan_kv_lens_list, device, _ws, _BuffTag.plan_kv_lens)
    _call_plan(qo_segment_offsets, qo_segment_lens_gpu, plan_kv_lens_gpu,
        packed_work_range, packed_work_info,
        qo_tile_size, kv_tile_size,
        num_qo_heads, num_ctas, causal, plan_qo_offset,
        num_kv_splits,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        0, None, num_kv_splits_per_row, workspace_lse, lse_total_size,
        pack_factor, cuda_stream)

    return _make_plan_info(
        packed_work_range=packed_work_range, packed_work_info=packed_work_info,
        kv_tile_begin_indices=kv_tile_begin_indices, kv_tile_end_indices=kv_tile_end_indices,
        kv_split_indices=kv_split_indices, num_kv_splits=num_kv_splits, workspace_o=workspace_o,
        workspace_lse=workspace_lse, max_qo_len=max_qo_len, predicted_speedup=1.0,
        num_kv_splits_per_row=num_kv_splits_per_row, qo_segment_offsets=qo_segment_offsets,
        kv_segment_offsets=kv_segment_offsets, kv_page_indptr=kv_page_indptr, max_k_tiles=max_k_tiles,
        qo_segment_lens=qo_segment_lens_gpu, kv_segment_lens=kv_segment_lens, qo_offset=qo_offset,
        pack_factor=pack_factor, orig_num_qo_heads=orig_num_qo_heads,
        qo_len_uniform=qo_len_uniform, cute_workspace_buffer=cute_workspace_buffer,
        kv_page_indptr_local_last=kv_page_indptr_local_last,
        **_icp_direct_fields,
        plan_workspace=_ws, plan_workspace_epoch=_ws_epoch)



def _fmha_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    k_scale: Optional[Union[float, torch.Tensor]] = None,
    v_scale: Optional[Union[float, torch.Tensor]] = None,
    o_scale: Optional[float] = None,
    output_maxscore: bool = True,
    output_o: bool = True,
    check_input_valid: bool = False,
    *,
    valid_score: Optional[torch.Tensor] = None,
    icp_c: int = 1,
    icp_rank: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run the ordinary dev variants or an explicit ICP score-only plan."""
    _assert_plan_workspace_current(plan_info)
    _assert_device_plan_current(plan_info)

    if plan_info["MM-SA-Nv"]:
        if valid_score is not None or icp_c != 1 or icp_rank != 0:
            raise ValueError("ICP score planes require an OnlyScoreIcp plan")
        return sparse_fmha(q=q, k=k, v=v, plan_info=plan_info, out=out, max_score=max_score, 
                sm_scale=sm_scale, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale, o_scale=o_scale,
                kv_indices=kv_indices, output_maxscore=output_maxscore, output_o=output_o, q_offset_override=q_offset_override,
                kv_block_indexes=kv_block_indexes, check_input_valid=check_input_valid)

    is_nvfp4 = k.dtype == torch.uint8
    if is_nvfp4 and (valid_score is not None or icp_c != 1 or icp_rank != 0):
        raise ValueError("ICP scoring requires FP8 index fragments, not NVFP4 attention caches")
    if is_nvfp4:
        k, k_sf, v, v_sf = nvfp4_head_slot_views(k, v)
        k_global_scale, v_global_scale = k_scale, v_scale
        k_scale = v_scale = 1.0
        # Sparse NVFP4 decode runs on the Q8KV4 kernel when the plan carries one and the call
        # fits it; otherwise the kv_mode 3 kernel below serves it.
        q8kv4 = plan_info.get(q8kv4_decode_adapter.PLAN_KEY)
        if q8kv4 is not None:
            blocker = q8kv4_decode_adapter.run_blocker(
                q8kv4, q=q, kv_indices=kv_indices, kv_block_indexes=kv_block_indexes,
                max_score=max_score, output_o=output_o, q_offset_override=q_offset_override,
                o_scale=o_scale, k_global_scale=k_global_scale, v_global_scale=v_global_scale)
            if blocker is None:
                out = q8kv4_decode_adapter.run(
                    q8kv4, q, k, v, k_sf, v_sf, kv_indices=kv_indices,
                    kv_block_indexes=kv_block_indexes, k_global_scale=k_global_scale,
                    v_global_scale=v_global_scale, sm_scale=sm_scale, q_scale=q_scale, out=out)
                return out, None
            if q8kv4["backend"] == "q8kv4":
                raise ValueError(f"decode_backend='q8kv4' cannot serve this call: {blocker}")

    nnz_qo, num_qo_heads, head_dim_qk = q.shape
    # refined-icp-v1: a direct-table plan is PAGED while passing no `kv_indices` --
    # the block table IS the page directory.  Deriving pagedness from `kv_indices`
    # alone would unpack `v.shape` as a dense 3-tuple and fail, or worse, dispatch a
    # dense variant against a paged pool.
    icp_block_table = plan_info.get("block_table")
    if kv_indices is None and icp_block_table is None:
        nnz_kv, num_kv_heads, head_dim_vo = v.shape
        is_paged = False
    else:
        nnz_kv, num_kv_heads, page_size, head_dim_vo = v.shape
        is_paged = True

    if is_nvfp4:
        head_dim_vo = 128
    qo_total_len = nnz_qo

    packed_work_range = plan_info["packed_work_range"]
    packed_work_info = plan_info["packed_work_info"]
    kv_tile_begin_indices = plan_info["kv_tile_begin_indices"]
    kv_tile_end_indices = plan_info["kv_tile_end_indices"]
    kv_split_indices = plan_info["kv_split_indices"]
    num_kv_splits = plan_info["num_kv_splits"]
    workspace_o = plan_info["workspace_o"]
    workspace_lse = plan_info["workspace_lse"]
    plan_max_qo_len = plan_info["max_qo_len"]
    num_kv_splits_per_row = plan_info["num_kv_splits_per_row"]
    qo_segment_offsets = plan_info["qo_segment_offsets"]
    kv_segment_offsets = plan_info["kv_segment_offsets"]
    kv_page_indptr = plan_info["kv_page_indptr"]
    max_k_tiles = plan_info["max_k_tiles"]
    qo_segment_lens = plan_info["qo_segment_lens"]
    kv_segment_lens = plan_info["kv_segment_lens"]
    qo_offset = plan_info["qo_offset"] if q_offset_override is None else q_offset_override
    pack_factor = plan_info["pack_factor"]
    orig_num_qo_heads = plan_info["orig_num_qo_heads"]
    qo_len_uniform = plan_info["qo_len_uniform"]
    workspace_buffer = plan_info["cute_workspace_buffer"]

    batch_size = qo_segment_lens.shape[0]

    if isinstance(qo_offset, int):
        qo_offset = torch.full_like(qo_segment_lens, qo_offset)
    elif qo_offset is not None:
        assert qo_offset.device == qo_segment_lens.device

    if check_input_valid:
        # refined-icp-v1: in direct-table mode `kv_indices` IS the rectangular block
        # table (the caller passes the same object so that `is_paged` resolves), and
        # the packed-list checks in `_validate_fmha_inputs` -- length ==
        # sum(ceil(kv_len/page_size)), every entry < total_pages -- are false for it:
        # the table's tail is capacity, deliberately holding stale page IDs.  Hand it
        # None rather than teach that validator a second page directory.
        _vfi_kv_indices = None if icp_block_table is not None else kv_indices
        _validate_fmha_inputs(
            q, k, v, qo_segment_lens, kv_segment_lens,
            num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
            batch_size, qo_total_len, is_paged,
            _vfi_kv_indices, kv_block_indexes, qo_offset,
            page_size if is_paged else 0,
            pack_factor=pack_factor,
            packed_work_range=packed_work_range,
            packed_work_info=packed_work_info,
            num_kv_splits=num_kv_splits,
            kv_tile_begin_indices=kv_tile_begin_indices,
            kv_tile_end_indices=kv_tile_end_indices,
            kv_split_indices=kv_split_indices,
            qo_segment_offsets=qo_segment_offsets,
            kv_page_indptr=kv_page_indptr,
        )

    if pack_factor > 1 and orig_num_qo_heads is not None:
        num_qo_heads = orig_num_qo_heads // pack_factor
        qo_total_len = nnz_qo * pack_factor

    max_qo_len = plan_max_qo_len
    # Same function as the planner used, and for the same reason it exists: this
    # value selects the VARIANT, the planner's selected the WORK DECOMPOSITION, and
    # a disagreement between them is not checkable downstream.
    qo_tile_size = _qo_tile_size_for(max_qo_len, icp_c)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim_qk)
    if q_scale is None:
        q_scale = 1.0
    if k_scale is None:
        k_scale = 1.0
    if v_scale is None:
        v_scale = 1.0
    if o_scale is None:
        o_scale = 1.0

    assert output_o or output_maxscore

    use_split_kv = (num_kv_splits > 1 and workspace_o is not None)

    if not output_maxscore or max_k_tiles == -1:
        max_score = None
        output_o = True
    elif max_score is None and max_k_tiles > 0:
        max_score = torch.full(
            (orig_num_qo_heads, max_k_tiles, nnz_qo),
            -float("inf"), dtype=torch.float32, device=q.device)
    elif max_score is not None and max_k_tiles > 0:
        unpacked_t = nnz_qo
        unpacked_h = orig_num_qo_heads
        packed_t = qo_total_len
        packed_h = num_qo_heads
        valid_max_score_shapes = {
            (unpacked_h, max_k_tiles, unpacked_t),  # legacy [H, K, T]
            (unpacked_t, unpacked_h, max_k_tiles),  # row-contiguous [T, H, K]
            (packed_h, max_k_tiles, packed_t),
            (packed_t, packed_h, max_k_tiles),
        }
        assert max_score.dtype == torch.float32, (
            f"max_score must be float32, got {max_score.dtype}"
        )
        assert max_score.device == q.device, (
            f"max_score must be on {q.device}, got {max_score.device}"
        )
        assert tuple(max_score.shape) in valid_max_score_shapes, (
            "max_score must have shape [H,K,T] or [T,H,K]; "
            f"got {tuple(max_score.shape)}, expected one of "
            f"{sorted(valid_max_score_shapes)}"
        )

    # ---- refined-icp-v1: the out-of-band validity plane, ABI W4/W8/W9 -------------
    if max_score is None:
        valid_score = None
    elif valid_score is not None:
        assert valid_score.dtype == torch.uint8, (
            f"valid_score must be uint8 (ABI {_FMHA_ICP_SCORE_ABI} W4/W9), "
            f"got {valid_score.dtype}"
        )
        assert valid_score.device == q.device, (
            f"valid_score must be on {q.device}, got {valid_score.device}"
        )
        assert tuple(valid_score.shape) == tuple(max_score.shape), (
            "valid_score must be SHAPE-MATCHED to max_score (ABI W4); got "
            f"{tuple(valid_score.shape)} vs {tuple(max_score.shape)}"
        )

    if not output_o:
        out = None
    elif out is None:
        out_dtype = torch.bfloat16 if q.dtype.itemsize == 1 else q.dtype
        out = torch.empty(
            nnz_qo,
            orig_num_qo_heads, head_dim_vo,
            device=q.device, dtype=out_dtype,
        )

    # Determine variant dispatch parameters
    dtype_code = _dlpack_dtype_code(q.dtype)

    kv_block_indexes_ptr_exists = kv_block_indexes is not None
    max_score_exists = max_score is not None
    o_exists = out is not None
    # refined-icp-v1: the direct table is an `OnlyScoreIcp` input.  At icp_c == 1 the
    # dispatch below picks a non-ICP variant, whose run binding refuses the pointer --
    # catch it here, where the message can say which side is wrong.
    assert icp_block_table is None or icp_c > 1, (
        "this plan carries a direct compound-page table, which is an ICP input, but "
        f"the call passed icp_c={icp_c}")
    if icp_c > 1:
        # ICP-aware OnlyScore.
        #  * kv_indices is THIS RANK's block-cyclic page list (local ordinals);
        #  * the plan's kv_segment_lens stays GLOBAL, because
        #    CausalMask::apply_mask (fmha_fusion.hpp) compares the GLOBAL cS
        #    coordinate against get<1>(problem_shape) and derives the causal
        #    offset from CausalMask::get_qo_offset. Sparse mode does exactly the
        #    same -- see _expand_for_per_token_sparse, which replicates the
        #    global kv_len per expanded token;
        #  * the trip count comes from icp_c/icp_rank, not from kv_len;
        #  * kv_page_indptr must be LOCAL or the loader's
        #    min(logical_page, num_pages_batch-1) clamp
        #    (`min(page_for_lookup, num_pages_batch - 1)` in the loader) walks
        #    off the end of this rank's shorter page table.
        # In DIRECT-TABLE mode the third and fourth bullets change source but not
        # meaning: kv_indices/kv_page_indptr are both absent, the loader addresses
        # the parent row as `request * row_stride` in the ordinary block table, and
        # the same clamp is fed by `local_blocks(L, r)` computed on the device from
        # the exact KV length.  The first two bullets are unchanged -- in
        # particular kv_segment_lens stays GLOBAL, which is what makes that
        # device-derived bound correct.
        assert max_score_exists and not o_exists, "ICP mode is OnlyScore-only"
        # ---- refined-icp-v1: OUTPUT-ABI VERSION 2 IS MANDATORY HERE ----------------
        # An unmigrated caller must fail LOUDLY, not read an in-band -inf as "empty".
        # There is no compatibility mode: ABI version 1 is not merely deprecated, it
        # is unimplementable under this contract: there is no spare FP32 value left
        # to mean "invalid", so an in-band encoding cannot exist.
        assert _FMHA_HAS_ICP_VALIDITY_PLANE, (
            "this overlay's Python half advertises "
            f"{_FMHA_ICP_SCORE_ABI} but its kernel half does not carry "
            "ptr_ValidScore; refusing to run rather than return an unwritten "
            "validity buffer"
        )
        assert valid_score is not None, (
            f"SparseAttnMode::OnlyScoreIcp output ABI version "
            f"{_FMHA_ICP_SCORE_ABI_VERSION} ({_FMHA_ICP_SCORE_ABI}) REQUIRES the "
            "out-of-band uint8 validity plane, shape-matched to max_score.  "
            "ABI version 1 (emptiness signalled in band with -inf) is SUPERSEDED: "
            "-inf is a legal representable score (W12), so a consumer reading "
            "max_score alone cannot tell 'scored, maximum is -inf' from 'past the "
            "end'.  Pass valid_score=<uint8 tensor shaped like max_score>."
        )
        # P1/P2/P3 hold only when the compound page IS this rank's fragment, i.e.
        # KVPageSize == R == 128 // C (ABI G13).  The kernel's validity formula is
        # P3 verbatim; at any other page granularity it would be silently wrong.
        assert is_paged and int(k.shape[2]) * icp_c == 128, (
            f"ICP page granularity: k.shape[2]={int(k.shape[2]) if is_paged else None} "
            f"with icp_c={icp_c} does not satisfy R*C == 128 (ABI G13/P3).  "
            "Bounded fragment loading requires page_size == 128 // icp_c."
        )
        assert not kv_block_indexes_ptr_exists, "ICP mode supplies its own map"
        assert pack_factor == 1, "ICP prototype requires pack_factor == 1"
        # The page-granularity assert above pins R * C == 128; this pins the OTHER
        # factor of the same 1-tile-per-page property, the one that is NOT visible in
        # any tensor shape.  `_qo_tile_size_for` already returns 128 under ICP, so
        # reaching this is either a plan built by an older/other planner or an edit
        # that unpinned it -- and the consequence is silent (columns marked valid by
        # the W5 row predicate whose score the main loop never wrote), so it has to
        # be caught here rather than downstream.
        assert qo_tile_size == 128, (
            f"ICP requires qo_tile_size == 128, got {qo_tile_size}.  At qo_tile 256 "
            "jit.py binds thread_shape '_2,_1,_1', so effective_tile_kv = tile_kv / "
            "thread_shape[1] becomes 2 x KVPageSize and one compute tile spans TWO "
            "compound pages, which the loader (tiles_per_page == 1) does not gather "
            "and the ICP trip count does not account for.  MEASURED: 2060/66048 "
            "score cells wrong at Q=129 W=2, 3844/131072 at Q=256, most of them "
            "never written at all while still marked valid."
        )
        # refined-icp-v1: the W5 completion pass writes columns
        # [0, max_k_tiles) for the row it owns, which assumes the work item
        # covers the whole wave, i.e. kv_tile_begin == 0.  Split-KV would give a
        # work item a sub-range and the pass would then overwrite another
        # split's columns with -inf.  Split-KV is unreachable here today
        # (output_o is False under ICP, so `use_split_kv` is False), but this
        # asserts it rather than relying on that staying true.
        assert not use_split_kv and num_kv_splits == 1, (
            "ICP x split-KV is not supported: the refined-icp-v1 W5 completion "
            "pass assumes one work item covers the whole advertised wave "
            f"(kv_tile_begin == 0); got num_kv_splits={num_kv_splits}"
        )
        assert 2 <= icp_c <= 15 and 0 <= icp_rank < icp_c
        # The plan must have been built for THIS rank: its kv_page_indptr is
        # local and block-cyclic, so a plan built for another (C, rank) would
        # point the loader at a different page run. batch_size > 1 is supported
        # -- kv_page_indptr carries one entry per request plus the terminator,
        # which is what the next two asserts pin.
        assert plan_info.get("icp_c", 1) == icp_c and \
            plan_info.get("icp_rank", 0) == icp_rank, \
            "plan was not built with this (icp_c, icp_rank)"
        if icp_block_table is not None:
            # ---- refined-icp-v1: THE DIRECT COMPOUND-PAGE TABLE -----------------
            # Everything here is host metadata the plan already holds: no device
            # read, no `.item()`, no stream sync on the submission path.
            _assert_icp_direct_table_supported()
            # `kv_indices` is ALSO how this function decides a call is paged, so the
            # caller passes the SAME TENSOR OBJECT as both the plan field and
            # `kv_indices` and the two cannot diverge.  Accept that -- including a
            # rectangular 2-D tensor, which a flat packed list never is -- and refuse
            # only a genuinely DIFFERENT second page directory.  The identity is the
            # check; presence is not.
            assert kv_indices is None or kv_indices.data_ptr() == \
                icp_block_table.data_ptr(), (
                    "a direct-table plan received a different kv_indices buffer: two "
                    "page directories in one call, and which one the loader used "
                    "would be decided by a pointer test rather than by the call.  "
                    "Pass the table itself as kv_indices, or nothing.")
            # This is what actually separates the modes, and it is the guard that
            # replaces the packed arm's `kv_page_indptr_local_last == kv_indices.numel()`
            # -- which is meaningless here (the table's numel is a capacity, not a page
            # count) and which the direct arm must therefore never reach.
            assert kv_page_indptr is None, (
                "a direct-table plan must not carry kv_page_indptr: the row "
                "origin/pitch and the device KV lengths replace the CSR entirely")
            assert plan_info["kv_page_indptr_local_last"] is None, (
                "a direct-table plan must not carry a host page count: it exists "
                "only to let the packed-list guard read the CSR's last entry "
                "without a D->H sync, and a stale one is a second page authority")
            # The contract's names and the prototype's must agree.  They are
            # written by one dict in the planner, so a disagreement means the plan
            # was hand-assembled or edited -- and the consequence is a fragment
            # bound taken at the wrong rank, which is finite, plausible and silent.
            assert plan_info["index_world_size"] == icp_c and \
                plan_info["index_rank"] == icp_rank, (
                    f"plan index_(rank,world_size)=({plan_info['index_rank']},"
                    f"{plan_info['index_world_size']}) disagrees with "
                    f"(icp_rank,icp_c)=({icp_rank},{icp_c})")
            _R = plan_info["index_rows_per_rank"]
            assert _R * icp_c == 128 and _R == int(k.shape[2]), (
                f"index_rows_per_rank={_R} must be 128 // icp_c ({128 // icp_c}) "
                f"AND the dispatched page extent k.shape[2]={int(k.shape[2])}: the "
                "kernel takes R from the paged tensor's mode-0 extent, so a "
                "disagreement shifts every fragment bound (ABI G13/P3)")
            assert int(plan_info["block_table_row_stride"]) > 0, (
                "block_table_row_stride is the row address pitch and must be "
                "positive")
            # Re-checked at RUN time, not only at plan time: the plan holds the
            # tensor, and a caller that re-sliced or reallocated it between the two
            # would move the base the kernel was promised is stable (I-6).
            assert icp_block_table.storage_offset() == 0, (
                "block_table must be passed as its stable base (storage_offset == "
                f"0); got {icp_block_table.storage_offset()}")
            assert icp_block_table.stride(0) == int(
                plan_info["block_table_row_stride"]), (
                    "block_table's row stride moved between plan and run: "
                    f"{icp_block_table.stride(0)} vs "
                    f"{plan_info['block_table_row_stride']}")
            _row0 = int(plan_info["block_table_row_begin"])
            assert _row0 >= 0 and _row0 + batch_size <= int(icp_block_table.shape[0]), (
                f"this call addresses block_table rows [{_row0}, "
                f"{_row0 + batch_size}) but the table has "
                f"{int(icp_block_table.shape[0])} rows.  Row origin + batch index is "
                "the whole request->row mapping; a short table lands on another "
                "request's row, which is in bounds and reads cleanly.")
        else:
            assert kv_page_indptr is not None, "ICP mode requires a paged plan"
            assert int(kv_page_indptr.shape[0]) == batch_size + 1, \
                "kv_page_indptr must have batch_size + 1 entries"
            assert plan_info["kv_page_indptr_local_last"] == int(kv_indices.numel()), (
                "kv_page_indptr[-1]="
                f"{plan_info['kv_page_indptr_local_last']} != len(kv_indices)="
                f"{int(kv_indices.numel())}; the plan's indptr is not this rank's")
        # PROTOTYPE encoding: params.kv_block_num is derived in the run
        # template from kv_block_indexes.size(2), and is already plumbed to
        # params.load.kv_block_num. Carrying (rank, C) in it keeps the whole
        # change inside two headers instead of threading two ints through
        # run_fmha_fwd, FMHACutlassSM100Params and both jinja templates.
        # The pointer is never dereferenced: every read of kv_block_indexes is
        # under `if constexpr (kNeedSparse)`, which is false in this mode.
        kv_block_indexes = plan_info.get("icp_geometry_descriptor")
        if kv_block_indexes is None:
            # Older caller-supplied plans remain usable. This fallback creates
            # only a tensor view, so it adds no CUDA allocation or GPU kernel.
            kv_block_indexes = _make_icp_geometry_descriptor(
                qo_segment_lens, icp_c, icp_rank)
            plan_info["icp_geometry_descriptor"] = kv_block_indexes
        assert tuple(kv_block_indexes.shape) == (1, 1, icp_rank * 16 + icp_c), (
            "plan's ICP geometry descriptor does not match its rank/world size")
        # kv_page_indptr comes from the plan and is already LOCAL to this rank.
        sparse_mode = 4
    elif kv_block_indexes_ptr_exists:
        sparse_mode = 0
    elif max_score_exists and o_exists:
        sparse_mode = 1
    elif max_score_exists:
        sparse_mode = 2
    else:
        sparse_mode = 3

    variant_page_size = (k.shape[2] if is_paged else -1)

    # Keep the ordinary variant's FFI signature/cache identity unchanged.
    nvfp4_args = ()
    _kv_dtype = None
    _nv_g_k = 1.0
    if is_nvfp4:
        _kv_dtype = "nvfp4"
        _nv_g_k = 6.0
        nvfp4_args = (
            k, k_sf, v, v_sf,
            k.stride(0), k.stride(1), k_sf.stride(1),
            6.0, 6.0, k_global_scale, v_global_scale,
        )

    icp_args = ()
    if sparse_mode == 4:
        icp_args = (
            valid_score, icp_block_table,
            int(plan_info.get("block_table_row_stride", 0) or 0),
            int(plan_info.get("block_table_row_begin", 0) or 0),
        )

    variant_module = get_fmha_variant(
        dtype_code, qo_tile_size, (max_qo_len <= 64),
        sparse_mode, variant_page_size, use_split_kv, pack_factor,
        kv_dtype=_kv_dtype)

    variant_module.run(
        workspace_buffer,
        q, k, v,
        qo_segment_lens, kv_segment_lens,
        qo_segment_offsets, kv_segment_offsets,
        packed_work_range, packed_work_info,
        out,
        sm_scale, q_scale, k_scale, v_scale, o_scale,
        max_qo_len,
        qo_offset,
        num_kv_splits,
        kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
        workspace_o, workspace_lse,
        num_kv_splits_per_row,
        qo_tile_size,
        kv_indices, kv_page_indptr,
        max_score,
        max_k_tiles,
        kv_block_indexes,
        pack_factor,
        bool(qo_len_uniform),
        torch.cuda.current_stream().cuda_stream,
        *nvfp4_args,
        *icp_args,
    )

    # Split-KV reduction
    if use_split_kv and out is not None:
        log2_e = math.log2(math.exp(1.0))
        # Gamma compensates raw code*block_scale/gamma staging. The kernel
        # multiplies this by the same device alpha_k read by the mainloop.
        scale_softmax_log2 = float(q_scale * k_scale * _nv_g_k * sm_scale) * log2_e
        inv_scale_o = float(o_scale)

        reduction_module = get_reduction_module(nvfp4=is_nvfp4)
        reduction = (reduction_module.reduction if not is_nvfp4
                     else reduction_module.reduction_nvfp4)
        reduction(
            workspace_o,
            out,
            workspace_lse,
            num_kv_splits_per_row,
            scale_softmax_log2, inv_scale_o,
            num_kv_splits, qo_total_len, num_qo_heads, head_dim_vo,
            num_qo_heads * head_dim_vo, head_dim_vo,
            num_qo_heads * head_dim_vo, head_dim_vo,
            orig_num_qo_heads,
            num_kv_heads,
            pack_factor,
            torch.cuda.current_stream().cuda_stream,
            *((k_global_scale,) if is_nvfp4 else ()),
        )

    # NOTE: the validity plane is caller-owned (ABI W8/W9) and is written
    # in place, exactly like `max_score` when the caller supplies it.  The return
    # tuple is deliberately NOT widened: widening it would break every existing
    # `out, ms = _fmha_sm100(...)` call site, and the plane is already in the
    # caller's hands.
    return out, max_score


def fmha_sm100_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    *args,
    qo_offset: Optional[Union[int, torch.Tensor]] = None,
    split_prefill_decode = True,
    **kwargs
):
    """Build a reusable execution plan for ``fmha_sm100``.

    The plan is shape-dependent and can be reused across layers or repeated
    calls that share the same sequence lengths, head counts, page size, sparse
    mode, and output mode.  Planning may run CUDA kernels and allocates
    workspaces, so it should be done outside tight per-layer loops when
    possible.

    Parameters
    ----------
    qo_segment_lens : torch.Tensor
        Shape ``[batch_size]``, dtype int32/int64.  Per-request Q/O lengths.
        CPU tensors are accepted for the dense planner; sparse prefill planning
        moves them to CUDA internally.
    kv_segment_lens : torch.Tensor
        Shape ``[batch_size]``, dtype int32/int64.  Per-request KV lengths.
    *args
        Positional arguments forwarded to the internal planner.  In normal
        usage this is ``num_qo_heads`` and optionally ``num_kv_heads``.
    qo_offset : int or torch.Tensor, optional
        Per-request causal offset.  If omitted, defaults to
        ``kv_segment_lens - qo_segment_lens`` for bottom-right causal masking.
        A tensor must have shape ``[batch_size]``.
    decode_backend : str, optional
        ``"auto"`` (default) plans sparse NVFP4 decode on the Q8KV4 kernel when the batch fits
        it (page size 128, 8 or 16 Q heads per KV head, uniform query lengths, at most 64
        blocks, causal, no max-score output) and keeps the kv_mode 3 kernel otherwise;
        ``"q8kv4"`` requires the Q8KV4 kernel and raises when the batch or a later call does
        not fit; ``"kv_mode3"`` never plans it.
    prefill_backend : str, optional
        ``"auto"`` (default) plans sparse NVFP4 prefill on the Q8KV4 kernel when the batch fits
        it (page size 128, 16 Q heads per KV head, 4/8/16/32 blocks, causal, no max-score
        output, SM100/SM103/SM107 with a CUDA 13.4+ toolkit) and keeps the CuTe-DSL NVFP4
        kernel otherwise; ``"q8kv4"`` requires the Q8KV4 kernel and raises when the batch or a
        later call does not fit; ``"cute_dsl"`` never plans it.
    kv_dtype : str, optional
        ``"fp8"`` skips the Q8KV4 plans (they only serve NVFP4 caches); ``"nvfp4"`` or
        ``None`` allows them.
    block_scale_shift : int, optional
        Block-scale staging of the Q8KV4 kernels; 3 (default) for caches whose E4M3 block
        scales use the full range next to a global scale (TransformerEngine convention), 0 for
        caches whose ``code * block_scale`` products already fit E4M3.
    split_prefill_decode : bool, optional
        If True, a mixed batch ordered as decode requests followed by prefill
        requests is split into two sub-plans.  The original order must already
        group short decode sequences before long prefill sequences.
    **kwargs
        Planner options forwarded to ``_fmha_sm100_plan``.  Common options are
        ``num_kv_heads``, ``num_kv_splits``, ``page_size``,
        ``output_maxscore``, ``kv_block_num``, ``usable_SM_count``, ``causal``,
        ``sparse_kernel_mode``, ``use_fp8_kvcache``, ``device``, and ``stream``.

        ``icp_block_table`` / ``icp_block_table_row_stride`` /
        ``icp_block_table_row_begin`` select the ICP DIRECT COMPOUND-PAGE TABLE
        (``docs/INTEGRATION.md`` §4.6): the plan then carries the ordinary
        rectangular table's stable base, row pitch and row origin instead of
        ``kv_page_indptr``.  It is an added mode, not a replacement -- the packed
        page list is unchanged and is what ``icp_c == 1`` uses.  ``row_begin`` is
        the table row this plan's batch index 0 maps to, which is 0 only for the
        first query chunk of an invocation.  A split mixed prefill/decode plan is
        refused in that mode, because splitting re-bases the prefill half's
        pointer rather than its origin.

    Returns
    -------
    tuple
        ``(has_mixed_prefill, split, batch_size, decode_plan, prefill_plan)``.
        Pass this tuple unchanged as ``plan_info`` to ``fmha_sm100``.
    """

    # assert qo_segment_lens.device.type == 'cpu' \
    #         and kv_segment_lens.device.type == 'cpu'
    # assert qo_offset is None or isinstance(qo_offset, int) or qo_offset.device.type == 'cpu'

    decode_backend, kv_dtype, block_scale_shift = q8kv4_decode_adapter.plan_options(kwargs)
    prefill_backend = q8kv4_prefill_adapter.plan_options(kwargs)

    def attach_q8kv4(plan, decode_qo_lens, decode_kv_lens):
        q8kv4_decode_adapter.attach_plan(
            plan, qo_segment_lens=decode_qo_lens, kv_segment_lens=decode_kv_lens,
            num_qo_heads=args[0] if args else kwargs["num_qo_heads"],
            num_kv_heads=args[1] if len(args) > 1 else kwargs.get("num_kv_heads", -1),
            page_size=kwargs.get("page_size", -1), kv_block_num=kwargs.get("kv_block_num", -1),
            causal=kwargs.get("causal", True), output_maxscore=kwargs.get("output_maxscore", False),
            usable_sm_count=kwargs.get("usable_SM_count", -1), device=kwargs.get("device"),
            backend=decode_backend, kv_dtype=kv_dtype, block_scale_shift=block_scale_shift)

    def attach_q8kv4_prefill(plan, prefill_kv_lens):
        q8kv4_prefill_adapter.attach_plan(
            plan, kv_segment_lens=prefill_kv_lens,
            num_qo_heads=args[0] if args else kwargs["num_qo_heads"],
            num_kv_heads=args[1] if len(args) > 1 else kwargs.get("num_kv_heads", -1),
            page_size=kwargs.get("page_size", -1), kv_block_num=kwargs.get("kv_block_num", -1),
            causal=kwargs.get("causal", True), output_maxscore=kwargs.get("output_maxscore", False),
            device=kwargs.get("device"), backend=prefill_backend, kv_dtype=kv_dtype,
            block_scale_shift=block_scale_shift)

    if qo_offset is None:
        qo_offset = kv_segment_lens - qo_segment_lens
    elif isinstance(qo_offset, int):
        qo_offset = torch.full_like(qo_segment_lens, qo_offset)
    
    batch_size = qo_segment_lens.shape[0]
    has_mixed_prefill = False
    qmax = qo_segment_lens.max().item()
    sparse = kwargs.get("kv_block_num", -1) > 0
    split_threshold = _prefill_qlen_threshold(sparse)
    if split_prefill_decode and qmax > split_threshold:
        split = (qo_segment_lens > split_threshold).nonzero(as_tuple=False)[0, 0].item()
        has_mixed_prefill = split > 0
    if has_mixed_prefill and kwargs.get("plan_workspace") is not None:
        # Both sub-plans below are returned together and are therefore ALIVE
        # TOGETHER; one workspace cannot hold both, and the existing `.clone()`
        # of the decode half would defeat the stable pointers the workspace was
        # passed for.  Refuse instead of silently giving back one stable plan and
        # one freshly allocated one.
        raise PlanWorkspaceError(
            "plan_workspace cannot be used with a split mixed prefill/decode "
            "plan: `fmha_sm100_plan` returns two sub-plans that are live at the "
            "same time, so they need two workspaces.  Either pass "
            "split_prefill_decode=False, or call `_fmha_sm100_plan` once per "
            "half with its own workspace slot.")
    if has_mixed_prefill and kwargs.get("icp_block_table") is not None:
        # refined-icp-v1: REFUSED, not split.  Two independent reasons, either
        # sufficient: (1) the decode half below is `.clone()`d field by field, which
        # would clone the block table and hand the kernel a pointer that is neither
        # stable nor the caller's; (2) the prefill half's rows begin at the first
        # prefill request, i.e. the table would have to be re-based to a non-zero
        # offset -- exactly the slicing contract I-6 forbids, and the reason the
        # prefill band is not capture-eligible (DIRECT_TABLE_CONTRACT §5.2).
        raise ValueError(
            "the direct compound-page table does not admit a split mixed "
            "prefill/decode plan: the batch index must BE the table row from row 0 "
            "(contract I-6), and splitting re-bases the prefill half.  Plan the "
            "decode band on its own (split_prefill_decode=False over the decode "
            "requests) and route prefill through the existing phase admission.")
    if has_mixed_prefill:
        # print(f"Split into 2 parts at index {split}")
        decode_qo_segment_lens = qo_segment_lens[:split]
        decode_kv_segment_lens = kv_segment_lens[:split]
        decode_qo_offset = qo_offset[:split]
        decode = _fmha_sm100_plan(decode_qo_segment_lens, decode_kv_segment_lens, *args,
                                    qo_offset=decode_qo_offset, **kwargs)
        attach_q8kv4(decode, decode_qo_segment_lens, decode_kv_segment_lens)
        decode = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in decode.items()}
        prefill_qo_segment_lens = qo_segment_lens[split:]
        prefill_kv_segment_lens = kv_segment_lens[split:]
        prefill_qo_offset = qo_offset[split:]
        prefill = _fmha_sm100_plan(prefill_qo_segment_lens, prefill_kv_segment_lens, *args,
                                    qo_offset=prefill_qo_offset, **kwargs)
        attach_q8kv4_prefill(prefill, prefill_kv_segment_lens)
        return (True, split, batch_size, decode, prefill)
    else:
        plan = _fmha_sm100_plan(qo_segment_lens, kv_segment_lens, *args, 
                                    qo_offset=qo_offset, **kwargs)
        attach_q8kv4(plan, qo_segment_lens, kv_segment_lens)
        attach_q8kv4_prefill(plan, kv_segment_lens)
        return (False, 0, batch_size, plan, None)

def fmha_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan_info,
    kv_indices: Optional[torch.Tensor] = None,
    kv_block_indexes: Optional[torch.Tensor] = None,
    q_offset_override: Optional[Union[int, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
    **kwargs
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run dense, paged, or sparse SM100 FMHA using a precomputed plan.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[total_qo_len, num_qo_heads, head_dim]``.  Supported dtypes are
        ``torch.bfloat16`` and ``torch.float8_e4m3fn``.  ``head_dim`` must be
        128.
    k : torch.Tensor
        Dense layout ``[total_kv_len, num_kv_heads, head_dim]`` or paged layout
        ``[total_pages, num_kv_heads, page_size, head_dim]``.
    v : torch.Tensor
        Same layout as ``k``. The output head dimension is 128 for NVFP4,
        otherwise it follows ``v.shape[-1]``.
    plan_info : tuple
        Return value from ``fmha_sm100_plan`` for the same lengths, head layout,
        page size, and sparse/output mode.
    kv_indices : torch.Tensor, optional
        Paged-KV physical page table, flattened across the batch.  Required when
        ``k`` and ``v`` use paged layout.  Shape is ``[sum_pages]`` and dtype is
        int32.  When the plan carries the ICP direct compound-page table, pass
        either ``None`` or the table itself -- it is also how this function
        decides a call is paged, so the table doubles as it and a rectangular 2-D
        tensor is accepted here.  A *different* second page directory is refused
        rather than silently resolved by a pointer test.
    kv_block_indexes : torch.Tensor, optional
        Sparse KV block indices from ``sparse_topk_select``.  Shape
        ``[total_qo_len, num_kv_heads or num_qo_heads, kv_block_num]``, dtype
        int32, ascending per row with ``-1`` padding at the tail.
    q_offset_override : int or torch.Tensor, optional
        Runtime causal-offset override.  Tensor form must have shape
        ``[batch_size]`` on the same CUDA device as the plan metadata.  The
        override must stay within the causal range visible to the original
        plan.
    out : torch.Tensor, optional
        Preallocated output buffer with shape
        ``[total_qo_len, num_qo_heads, head_dim_v]``.
    max_score : torch.Tensor, optional
        Preallocated per-KV-tile score buffer with dtype float32.  Accepted
        layouts are legacy ``[num_qo_heads, max_k_tiles, total_qo_len]`` and
        row-contiguous ``[total_qo_len, num_qo_heads, max_k_tiles]``.
    **kwargs
        Runtime options forwarded to the kernel runner.  Common options are
        ``sm_scale``, ``q_scale``, ``k_scale``, ``v_scale``, ``o_scale``,
        ``output_maxscore``, ``output_o``, and ``check_input_valid``.

    NVFP4 cache layout
    ------------------
    The cache is one uint8 ``[pages, 2*Hkv, 128, 72]`` allocation of per-head
    K/V slots: slot ``2*h`` is head ``h``'s K and slot ``2*h+1`` its V, so every
    head is one contiguous run of the page. A slot stores the head's packed
    data first (8192 bytes), followed by its E4M3 block scales (1024 bytes);
    the last dimension describes storage size, not interleaved token rows.
    Pass K and V as the slot views ``cache[:, 0::2]`` and ``cache[:, 1::2]``
    (see ``nvfp4_head_slot_views``); other layouts are rejected. Each byte
    packs two E2M1 values, and each block scale covers 16 values. K scales use
    ``token*8+group``; V scales use ``(token//4)*32+group*4+token%4`` within
    each head.
    ``k_scale`` and ``v_scale`` are CUDA float32 scalar tensors for NVFP4:
    ``value = E2M1(code) * E4M3(block_scale) * global_scale``.
    Use BF16 or E4M3 Q for prefill and E4M3 Q for decode. No cache conversion is needed.
    Sparse NVFP4 decode runs on the Q8KV4 kernel when ``fmha_sm100_plan`` planned it
    (``decode_backend``); calls it cannot serve (max-score output, ``q_offset_override``,
    ``o_scale`` other than 1, BF16 Q) fall back to the kv_mode 3 kernel unless the backend
    was forced. Sparse NVFP4 prefill likewise runs on the Q8KV4 prefill kernel when planned
    (``prefill_backend``) and Q is E4M3 (pass its dequant factor as ``q_scale``); BF16 Q keeps
    the CuTe-DSL NVFP4 kernel.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        ``(out, max_score)``.  Either item may be ``None`` if the corresponding
        output was disabled.  When both decode and prefill sub-plans are used,
        outputs are concatenated back into the original batch order.
    """
    has_mixed_prefill, split, batch_size, decode, prefill = plan_info
    if not has_mixed_prefill:
        return _fmha_sm100(q, k, v, decode, out=out, max_score=max_score, kv_indices=kv_indices,kv_block_indexes=kv_block_indexes, q_offset_override=q_offset_override, **kwargs)
    else:
        # refined-icp-v1: unreachable by construction -- `fmha_sm100_plan` refuses to
        # BUILD a split plan with a direct table -- but asserted here too, because the
        # split path below slices `kv_indices` by a page count the direct-table plan
        # does not have, and the KeyError it would raise names nothing useful.
        assert isinstance(decode, dict) and decode.get("block_table") is None, (
            "a direct compound-page table cannot be used with a split mixed "
            "prefill/decode plan; see fmha_sm100_plan")

        decode_pack = decode.get("pack_factor", 1)
        decode_nnz = decode["qo_segment_offsets"][-1].item() // decode_pack
        is_paged = kv_indices is not None
        nnz_qo = q.shape[0]
        num_qo_heads = q.shape[1]

        q_decode = q[:decode_nnz]
        q_prefill = q[decode_nnz:]

        if is_paged:
            k_decode, v_decode = k, v
            k_prefill, v_prefill = k, v
            if "kv_page_indptr" in decode:
                kv_page_split = decode["kv_page_indptr"][-1].item()
            else:
                kv_page_split = decode["total_rows"]
            decode_kv_indices = kv_indices[:kv_page_split]
            prefill_kv_indices = kv_indices[kv_page_split:]
        else:
            if "kv_segment_offsets" in decode:
                decode_kv_nnz = decode["kv_segment_offsets"][-1].item()
            else:
                decode_kv_nnz = decode["cu_seqlens_k"][-1].item()
            k_decode, k_prefill = k[:decode_kv_nnz], k[decode_kv_nnz:]
            v_decode, v_prefill = v[:decode_kv_nnz], v[decode_kv_nnz:]
            decode_kv_indices = None
            prefill_kv_indices = None

        decode_block_idx = kv_block_indexes[:decode_nnz] if kv_block_indexes is not None else None
        prefill_block_idx = kv_block_indexes[decode_nnz:] if kv_block_indexes is not None else None
        if isinstance(q_offset_override, int):
            decode_qo_offset = torch.full((split,), q_offset_override, dtype=torch.int32, device=q.device)
            prefill_qo_offset = torch.full((batch_size - split,), q_offset_override, dtype=torch.int32, device=q.device)
        elif q_offset_override is not None:
            decode_qo_offset = q_offset_override[:split]
            prefill_qo_offset = q_offset_override[split:]
        else:
            decode_qo_offset = None
            prefill_qo_offset = None

        # ---- Run kernels ----
        decode_out, decode_ms = _fmha_sm100(
            q_decode, k_decode, v_decode, decode,
            out=None, max_score=None,
            kv_indices=decode_kv_indices, kv_block_indexes=decode_block_idx,
            q_offset_override=decode_qo_offset,
            **kwargs)
        prefill_out, prefill_ms = _fmha_sm100(
            q_prefill, k_prefill, v_prefill, prefill,
            out=None, max_score=None,
            kv_indices=prefill_kv_indices, kv_block_indexes=prefill_block_idx,
            q_offset_override=prefill_qo_offset,
            **kwargs)

        # ---- Merge out ----
        if decode_out is not None and prefill_out is not None:
            combined_out = torch.cat([decode_out, prefill_out], dim=0)
        else:
            combined_out = None
        
        if out is not None and combined_out is not None:
            out.copy_(combined_out)

        # ---- Merge max_score ----
        if decode_ms is not None and prefill_ms is not None:
            d_kt, p_kt = decode_ms.shape[1], prefill_ms.shape[1]
            max_kt = max(d_kt, p_kt)
            combined_ms = torch.full(
                (num_qo_heads, max_kt, nnz_qo),
                -float("inf"), dtype=torch.float32, device=q.device)
            combined_ms[:, :d_kt, :decode_nnz] = decode_ms
            combined_ms[:, :p_kt, decode_nnz:] = prefill_ms
        else:
            combined_ms = decode_ms if decode_ms is not None else prefill_ms

        if max_score is not None and combined_ms is not None:
            if max_score.shape == combined_ms.shape:
                max_score.copy_(combined_ms)
            elif max_score.shape == (nnz_qo, num_qo_heads, combined_ms.shape[1]):
                max_score.copy_(combined_ms.permute(2, 0, 1).contiguous())
            else:
                raise ValueError(
                    f"max_score shape {tuple(max_score.shape)} is incompatible "
                    f"with combined max_score shape {tuple(combined_ms.shape)}"
                )

        return (out if out is not None else combined_out,
                max_score if max_score is not None else combined_ms)


# ============================================================================
# Sparse TopK Select
# ============================================================================


def sparse_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[Union[int, torch.Tensor]] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    max_score_layout: str = "HKT",
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Select top-k KV-tile indices per (qo_head, token) row from the FMHA max-score tensor.

    Designed for the MQA proxy-KV sparse attention path where the dense pass uses
    ``num_kv_heads_dense=1``, so ``max_score.shape[0] == num_kv_heads_real`` and
    each head row is processed independently (no GQA reduction inside this call).

    Parameters
    ----------
    max_score : torch.Tensor
        Contiguous float32 max-score tensor.  ``max_score_layout="HKT"`` expects
        shape ``(num_qo_heads, max_k_tiles, total_qo_len)``.  ``"THK"`` expects
        shape ``(total_qo_len, num_qo_heads, max_k_tiles)`` and skips the
        internal transpose before top-k.  Slots beyond the actual KV tile count
        must be pre-filled with ``-inf`` (fmha_sm100 does this automatically via
        ``torch.full``).
    topk : int
        Must be exactly 16.
    num_valid_pages : int or torch.Tensor, optional
        Actual number of KV pages in the page table, i.e. ``ceil(kv_len / page_size)``.
        ``max_k_tiles`` is round-up-aligned and always >= ``num_valid_pages``.
        The kernel may select tile indices in ``[num_valid_pages, max_k_tiles-1]``
        (all-``-inf`` padding tiles). Passing ``num_valid_pages`` replaces those
        out-of-range indices with ``-1`` and sorts them to the tail, matching the
        sparse FMHA kernel's kv_block_indexes contract.
        Tensor form must be CUDA int32/int64 with shape ``[total_qo_len]`` and
        provides a per-query-token page count for mixed-length batches.
        **Strongly recommended**: omitting this allows OOB page-table accesses in
        the sparse attention pass.
    force_begin_blocks : int
        Number of KV blocks at the beginning of the sequence (indices 0..N-1) to
        always include in the top-k result, regardless of their scores.  Useful
        for sink tokens.  Default 0.
    force_end_blocks : int
        Number of KV blocks at the end of the valid sequence (indices
        nvp-N..nvp-1, closest to the current query) to always include.  Useful
        for local-window attention.  Default 0.
    block_table : torch.Tensor, optional
        Optional int32 tensor with shape ``(total_qo_len, num_qo_heads, max_k_tiles)``.
        The kernel still selects and sorts logical tile indices, but after sorting
        each logical index ``idx`` is replaced with ``block_table[t, h, idx]`` in
        the output.  Use this for per-token/per-head physical page-table gathers.

    Returns
    -------
    torch.Tensor
        Shape ``(total_qo_len, num_qo_heads, topk)``, int32.  Without
        ``block_table``, values are logical tile indices in ascending tile order.
        With ``block_table``, values are gathered block-table entries after that
        logical ascending sort.  Out-of-range entries (if any) are ``-1`` at the tail.
    """

    assert max_score.dtype == torch.float32, f"max_score must be float32, got {max_score.dtype}"
    assert max_score.dim() == 3, f"max_score must be 3D, got {max_score.shape}"
    assert max_score.is_contiguous(), "max_score must be contiguous"
    assert topk == 16, f"topk must be 16, got {topk}"

    layout = max_score_layout.upper()
    assert layout in {"HKT", "THK"}, (
        f"max_score_layout must be 'HKT' or 'THK', got {max_score_layout!r}"
    )
    if layout == "HKT":
        num_qo_heads, max_k_tiles, total_qo_len = max_score.shape
        layout_arg = 0
    else:
        total_qo_len, num_qo_heads, max_k_tiles = max_score.shape
        layout_arg = 1

    if block_table is not None:
        assert block_table.dtype == torch.int32, (
            f"block_table must be int32, got {block_table.dtype}"
        )
        assert block_table.device == max_score.device, (
            f"block_table must be on {max_score.device}, got {block_table.device}"
        )
        assert block_table.dim() == 3, (
            f"block_table must be 3D [total_qo_len, num_qo_heads, max_k_tiles], "
            f"got {tuple(block_table.shape)}"
        )
        assert tuple(block_table.shape) == (total_qo_len, num_qo_heads, max_k_tiles), (
            f"block_table shape must be {(total_qo_len, num_qo_heads, max_k_tiles)}, "
            f"got {tuple(block_table.shape)}"
        )
        assert all(s >= 0 for s in block_table.stride()), (
            f"block_table must have non-negative strides, got {block_table.stride()}"
        )

    # v2.3 kernel only supports the insertion-sort path (K < 12288).
    assert max_k_tiles < 12288, (
        f"max_k_tiles={max_k_tiles} >= 12288: v2.3 kernel only supports K < 12288 "
        f"(radix-sort path not yet implemented). kv_len must be < {12288 * 128} tokens."
    )

    nvp_tensor = None
    if isinstance(num_valid_pages, torch.Tensor):
        assert num_valid_pages.dim() == 1, (
            f"num_valid_pages tensor must be 1D [total_qo_len], got {tuple(num_valid_pages.shape)}"
        )
        assert num_valid_pages.shape[0] == total_qo_len, (
            f"num_valid_pages tensor length {num_valid_pages.shape[0]} must match "
            f"total_qo_len={total_qo_len}"
        )
        assert num_valid_pages.device == max_score.device, (
            f"num_valid_pages tensor must be on {max_score.device}, got {num_valid_pages.device}"
        )
        assert num_valid_pages.dtype in (torch.int32, torch.int64), (
            f"num_valid_pages tensor must be int32 or int64, got {num_valid_pages.dtype}"
        )
        nvp_tensor = num_valid_pages.to(dtype=torch.int32).contiguous()
        nvp_arg = int(max_k_tiles)
    elif num_valid_pages is not None:
        nvp_arg = int(num_valid_pages)
        assert 0 < nvp_arg <= max_k_tiles, (
            f"num_valid_pages={nvp_arg} must be in (0, max_k_tiles={max_k_tiles}]"
        )
    else:
        # v2.5_oob_clamp_in_kernel: kernel takes a unified num_valid_pages arg.
        # When the caller doesn't supply one, pass max_k_tiles so the in-kernel
        # `idx >= num_valid_pages` check never triggers (idx is always in
        # [0, max_k_tiles)).
        nvp_arg = int(max_k_tiles)

    assert force_begin_blocks >= 0 and force_end_blocks >= 0, (
        f"force_begin_blocks={force_begin_blocks} and force_end_blocks={force_end_blocks} "
        f"must be non-negative"
    )
    assert force_begin_blocks + force_end_blocks <= topk, (
        f"force_begin_blocks({force_begin_blocks}) + force_end_blocks({force_end_blocks}) "
        f"= {force_begin_blocks + force_end_blocks} exceeds topk={topk}"
    )

    # HKT needs a transpose buffer; THK is already row-contiguous over K and
    # should not allocate or pass a dummy workspace, especially under CUDA graph
    # capture.
    workspace_size = 0 if layout == "THK" else num_qo_heads * max_k_tiles * total_qo_len
    workspace_buffer = None
    if workspace_size:
        workspace_buffer = _alloc_workspace_buf(
            _BuffTag.sparse_topk_workspace, workspace_size, max_score.device, torch.int32
        )
    
    if output is not None:
        assert output.dtype == torch.int32, f"output must be int32, got {output.dtype}"
        assert output.device == max_score.device, (
            f"output must be on {max_score.device}, got {output.device}"
        )
        assert output.dim() == 3, f"output must be 3D, got {tuple(output.shape)}"
        assert tuple(output.shape) == (total_qo_len, num_qo_heads, topk), (
            f"output shape must be {(total_qo_len, num_qo_heads, topk)}, "
            f"got {tuple(output.shape)}"
        )
        output_indices = output
    else:
        output_indices = torch.empty(
            total_qo_len, num_qo_heads, topk,
            dtype=torch.int32, device=max_score.device,
        )

    module = get_sparse_topk_module()
    # MQA dense pass: num_kv_heads_dense=1, so h_r=num_qo_heads.
    # The kernel only uses num_qo_heads = h_r * num_kv_heads as a product;
    # passing num_kv_heads=1 is equivalent to any other valid factorisation.
    #
    # v2.5_oob_clamp_in_kernel: OOB clamp is folded into the kernel — the prior
    # post-process torch.where + sort + torch.where chain (~84-101 us / call)
    # is replaced by passing num_valid_pages directly to the kernel.
    module.sparse_topk_select(
        max_score, output_indices, workspace_buffer, block_table,
        topk,
        nvp_arg,
        nvp_tensor,
        int(force_begin_blocks),
        int(force_end_blocks),
        layout_arg,
        torch.cuda.current_stream().cuda_stream,
    )

    return output_indices
