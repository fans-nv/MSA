"""The ICP local-candidate selector: dense block scores -> this rank's Top-16.

This is the producer that feeds :func:`fmha_sm100.icp.merge_candidates` and
:class:`fmha_sm100.icp.IcpExchange`. Each ICP rank scores only the blocks it owns,
selects its own best 16 per (token, head), and emits them as CONTRACT C4
candidate tuples; the merge or the exchange then reduces the ``C`` per-rank
lists to the global Top-16.

Shapes
------

``scores``  ``[T, H_group, N]`` fp32, N = ``max_local_blocks``. Padding is
masked by the runtime counts in :class:`CandidateGeometry`, never by the shape.

``out``     ``[T, H_group, 16, 2]`` fp32 (C4): ``[..., 0]`` the score,
``[..., 1]`` the int32 **global** block id *bitcast* to fp32. Invalid entries
are ``(-inf, -1)``. ``torch.equal`` on the fp32 view of this tensor is broken --
``int32(-1)`` bitcasts to a negative quiet NaN, so it reports a mismatch on
every invalid slot of two byte-identical buffers. Compare
``.view(torch.int32)``.

Bands
-----

Prefill and decode are **different types**, **different entry points** and
**different launchers**: :class:`PrefillPlan` / :func:`select_prefill_candidates`
and :class:`DecodePlan` / :func:`select_decode_candidates`. They are not
interchangeable and a mismatch does not type-check and does not run. The
distinction is not naming hygiene -- **decode is cudagraph-captured and prefill
always runs eager** -- and every difference below follows from it.

Decode, captured
    The launch geometry may depend on nothing the device knows, so the rung and
    the partition count come from ``max_local_blocks``, i.e. from **capacity**.
    ``out`` is **required**, because a tensor allocated inside a capture comes
    from the graph-private pool. ``T`` is pinned at :attr:`token_capacity`,
    because a capture bakes the shape in. :class:`SelectArm` keeps every
    measurement arm reachable, since the A/B has to be interleavable inside one
    process and one timing call.

Prefill, eager
    Eager because it allocates and because ``T`` is a per-call prefix, NOT
    because its launch geometry is per-call data. The rung and the partition
    count come from :attr:`PrefillPlan.scan_extent`, which is
    ``max_local_blocks`` -- ``ceil(max_model_len / 128)``, fixed at engine
    startup. The bounded radix arm remains the default; the whole-row arm is
    an explicit opt-in. There is no decode workspace or device readback.
    See :func:`select_prefill_candidates`.

    **Changed 2026-09-13 (nonblocking execution, N2).** The geometry used to be
    sized from a per-call ``live_blocks`` that was validated against the device
    counts it claimed to bound -- a CUDA reduction read into a Python integer on
    every prefill layer and query chunk. The accepted policy forbids that on the
    submission path whether or not the call is captured. It is not replaced by a
    weaker check: the extent is now the startup bound, so the metadata cannot
    under-report it and there is nothing left to validate. See
    :func:`select_prefill_candidates` for what that costs and what it does not.

:func:`ladder_items` mirrors the decode launcher's capacity ladder and
:func:`prefill_launch` mirrors the prefill launcher's whole geometry, both in
pure Python, so a rung can be gated on a host with no GPU.

``docs/CONTRACT.md`` records which kernel is valid in which band and why, and
``docs/REMOVED.md`` records the four capture blockers that killed a shared
score kernel.

No torch at import
------------------

This module imports torch inside the functions that need it, so the plan
validation, the arm table and :func:`ladder_items` are usable -- and gateable --
on a host with no torch. That is deliberate: the ladder is the part most likely
to be edited, and requiring a GPU box to reason about it is what stopped an
earlier attempt.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _build

if TYPE_CHECKING:  # torch is a runtime dependency of the calls, not the import
    import torch

CAND_K = 16

# Additive selector mapping capability; independent of the frozen P128 ABI.
CANDIDATE_MAPPING_ABI_VERSION = 2
SUPPORTED_GLOBAL_BLOCK_STRIDES = (1, 2)

#: Must match ``icp::kLocalPartition`` in ``csrc/local_candidates.cuh``: the
#: host sizes ``partials`` from it and the kernel indexes with it.
PARTITION_BLOCKS = 4096

#: Must match ``icp::kLocalThreads``. One CTA covers ``LOCAL_THREADS * Items``.
LOCAL_THREADS = 128

#: The rungs the launcher instantiates, ascending. Mirrors the ladder in
#: ``csrc/local_candidates.cu``; a rung this tuple does not name is not built.
LADDER_RUNGS = (1, 2, 4, 8, 16, 32)

#: The only rung a partitioned CTA may take: it strides by
#: :data:`PARTITION_BLOCKS` and must span one. ``csrc/local_candidates.cuh``
#: makes any other choice a compile error.
PARTITIONED_ITEMS = PARTITION_BLOCKS // LOCAL_THREADS

#: The largest partition count the prefill entry point accepts. Beyond it the
#: combine kernel's streaming loop would take a second iteration, which nothing
#: has ever executed; refusing is honest where reaching it is not.
MAX_PREFILL_PARTITIONS = LOCAL_THREADS * PARTITIONED_ITEMS // CAND_K - 1

_INT32_MAX = (1 << 31) - 1

#: The refined-icp-v1 attribute contract for :meth:`CandidateGeometry.from_metadata`'s
#: duck-typed binding. Versioned on purpose: the selector's meaning of the
#: diagonal changed without its dtype or its shape changing, so only a rename
#: can make an unmigrated builder fail loudly rather than mis-select silently.
_V1_REQUIRED = (
    "topk_num_valid_pages",
    "icp_forced_column_v1",
    "icp_scan_block_begin_v1",
    "icp_active_rows",
)


class SelectArm(enum.IntEnum):
    """The selector's arms. The value IS the extension's ``cap`` code.

    ``RADIX_FULL_ROW``
        One CTA per row with device-bounded work and an exact canonical-key
        finish. Capacities up to 8192 avoid partition scratch and the
        combine launch. Larger capacities use the bounded partitioned arm.
    ``RADIX_BOUNDED``
        Keys-only radix select with ``wanted`` bounded by the live element
        count. :data:`SHIPPED_ARM`. Byte-identical to ``RADIX`` wherever a row
        holds at least 16 live blocks, because the bound is a no-op there.
    ``RADIX``
        The same radix select with the bound off.
    ``SORT``
        ``cub::BlockRadixSort`` over every slot. The older arm, and the only one
        that carries the fp32 score as a value payload rather than
        reconstructing it from the key.
    ``CAPPED2`` .. ``CAPPED16``
        Radix select with the register depth pinned at the cap and the row
        streamed in chunks, so depth and shared footprint stop tracking
        capacity. No ladder: the cap *is* the depth.

    All arms are bit-exact against each other under CONTRACT C4/C5, with one
    stated exception: ``SORT`` preserves ``-0.0`` in the emitted score and every
    radix arm flushes it to ``+0.0``, which C5 mandates. A gate that
    bit-compares them must flush the ``SORT`` side.
    """

    RADIX_FULL_ROW = -3
    RADIX_BOUNDED = -2
    RADIX = -1
    SORT = 0
    CAPPED2 = 2
    CAPPED4 = 4
    CAPPED8 = 8
    CAPPED16 = 16


#: The arm the selector ships with, and the default of both entry points.
SHIPPED_ARM = SelectArm.RADIX_BOUNDED

#: Everything else. Reachable by name so an A/B can run, not because a caller
#: should choose one -- the same split :data:`fmha_sm100.icp.MaskPreset` draws.
MEASUREMENT_ARMS = frozenset(a for a in SelectArm if a is not SHIPPED_ARM)

#: Prototype spellings kept resolvable so older harness invocations and result
#: files still name an arm that exists.
ARM_ALIASES = {"radixsr": SelectArm.RADIX_BOUNDED}

#: Overrides the default arm for an A/B without touching a call site.
SELECTOR_ENV = "ICP_SELECTOR"


def selector_arm(value: SelectArm | str | int | None = None) -> SelectArm:
    """Resolve an arm from a member, a name, an alias or a raw ``cap`` code.

    ``None`` reads :data:`SELECTOR_ENV`, falling back to :data:`SHIPPED_ARM`.
    An unknown spelling is refused by name rather than silently falling back to
    the default: a silent fallback runs and times one arm under another's name,
    which no output comparison can catch because the arms are bit-exact.
    """
    if value is None:
        value = os.environ.get(SELECTOR_ENV) or SHIPPED_ARM
    if isinstance(value, SelectArm):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ARM_ALIASES:
            return ARM_ALIASES[key]
        try:
            return SelectArm[key.upper()]
        except KeyError:
            raise ValueError(
                f"unknown selector arm {value!r}; known: "
                f"{sorted(a.name.lower() for a in SelectArm)} "
                f"plus aliases {sorted(ARM_ALIASES)}"
            ) from None
    try:
        return SelectArm(int(value))
    except ValueError:
        raise ValueError(
            f"{value!r} is not a cap code the extension builds; known: "
            f"{sorted(int(a) for a in SelectArm)}"
        ) from None


def ladder_items(extent: int) -> int:
    """The compile-time ``Items`` the DECODE launcher's ladder picks.

    The Python mirror of the capacity ladder in ``csrc/local_candidates.cu``, so
    a rung can be gated on a host with no GPU and no torch. The launcher feeds
    it ``scores.shape[-1]``, i.e. :attr:`DecodePlan.ladder_extent`, which is
    capacity. It has no effect on the capped arms, which have no ladder.
    """
    extent = int(extent)
    if extent < 0:
        raise ValueError(f"extent must be non-negative; got {extent}")
    for items in LADDER_RUNGS:
        if extent <= LOCAL_THREADS * items:
            return items
    return LADDER_RUNGS[-1]


@dataclass(frozen=True, slots=True)
class PrefillLaunch:
    """The eager prefill path's whole launch geometry.

    ``partitions`` is 0 when one CTA covers the row, and ``items`` is the
    compile-time register depth. Both come from the live extent; capacity is
    only ever the row stride.
    """

    partitions: int
    items: int


def prefill_launch(live_blocks: int) -> PrefillLaunch:
    """The geometry the prefill launcher picks for this live extent.

    The Python mirror of ``icp::prefill_launch`` in
    ``csrc/local_candidates.cuh``, pure and torch-free for the same reason
    :func:`ladder_items` is. A partitioned CTA strides by
    :data:`PARTITION_BLOCKS` and must span one, so the partitioned rung is
    :data:`PARTITIONED_ITEMS` and nothing else.
    """
    live = int(live_blocks)
    if live < 0:
        raise ValueError(f"live_blocks must be non-negative; got {live}")
    if live > PARTITION_BLOCKS:
        partitions = -(-live // PARTITION_BLOCKS)
        return PrefillLaunch(partitions=partitions, items=PARTITIONED_ITEMS)
    return PrefillLaunch(partitions=0, items=ladder_items(live))


def prefill_scan_coverage(extent: int) -> int:
    """How many columns of a row a launch at ``extent`` actually scans.

    The host mirror of the only thing that can silently truncate a row: a CTA
    covers ``LOCAL_THREADS * items`` columns and starts at
    ``partition * PARTITION_BLOCKS``, so the launch reads column ``j`` of a row
    only while ``j`` is inside one of the partitions it launched. The kernel
    states the same bound as two device assertions (``local_candidates_rs.cuh``:
    the rung must span its partition, and the partition count must span the
    row); this is the pure-Python copy, so the property can be gated with no GPU.

    ``prefill_scan_coverage(extent) >= extent`` for every extent -- that is what makes
    :attr:`PrefillPlan.scan_extent` a safe launch size, since the kernel clamps
    every row to ``max_local_blocks``. It is the whole reason the selector no
    longer reads the device counts: a bound the *plan* already carries cannot be
    under-reported by per-step metadata.
    """
    launch = prefill_launch(extent)
    partitions = max(int(launch.partitions), 1)
    return partitions * min(PARTITION_BLOCKS, LOCAL_THREADS * launch.items)


def live_blocks(max_kv_len: int, *, page_size: int) -> int:
    """Upper bound, over every row of a launch, on the reachable block columns.

    The causal block count evaluated once at the batch's longest sequence. It is
    monotone in the sequence length, so the value at the maximum bounds every
    token in the batch. Taking a host int keeps the derivation off the device: a
    prefill builder already holds the CPU-side KV lengths.

    **It no longer sizes the selector's launch** (N2, 2026-09-13): the extent is
    :attr:`PrefillPlan.scan_extent`, a startup constant, so no per-step value --
    right, wrong or per-row -- can shorten the scan. Evaluated at
    ``max_model_len`` this function *is* ``max_local_blocks``, which is how the
    startup bound is built; evaluated per step it remains a legitimate
    description of how much of the row is live, and the benchmarks still use it
    to name a working point.

    **refined-icp-v1.** There is no rank term. Under fragment placement every
    rank holds ``R = 128 / W`` rows of *every* logical block and scores the full
    global block domain, so the reachable column count is ``ceil(len /
    page_size)`` on every rank -- ``W`` times what block-cyclic placement gave a
    single rank, and identical across ranks rather than differing by one.
    """
    if max_kv_len < 0:
        raise ValueError("max_kv_len must be non-negative")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    return -(-int(max_kv_len) // int(page_size))


def live_local_blocks(*_args: object, **_kwargs: object) -> int:
    """Removed by refined-icp-v1. Use :func:`live_blocks`.

    Kept as a raising stub rather than deleted because the replacement takes
    *fewer* arguments: a caller that dropped ``icp_degree``/``icp_rank`` but
    kept the name would otherwise bind to nothing and fail somewhere less
    obvious, and a caller that kept them would silently compute a block-cyclic
    count that is now ``W`` times too small.
    """
    raise RuntimeError(
        "live_local_blocks() is block-cyclic and was removed by "
        "refined-icp-v1: every rank now scores the full global block domain, so "
        "the count no longer depends on icp_degree or icp_rank. Call "
        "live_blocks(max_kv_len, page_size=...) instead."
    )


@dataclass(frozen=True, slots=True)
class SelectorPlan:
    """Immutable launch bounds. Not constructible: use a band's subclass.

    Parameters
    ----------
    icp_degree
        ``C``, the number of ICP ranks sharing the block axis.
    icp_rank
        This rank's index in ``[0, C)``. It pins the **head** window
        (:attr:`head_offset`), where a rotation is invisible at H_local == 1.
        Global block IDs use the geometry's explicit origin and stride rather
        than inferring physical-page ownership from this plan's rank.
    num_heads_local
        ``H_local``. ``H_group == C * H_local`` and the gathered head axis is
        rank-major, which is invisible at ``H_local == 1``.
    max_local_blocks
        ``N``, the capacity of the score row: both the row stride and, today,
        the ladder input. P128 uses ``ceil(max_model_len / 128)`` columns;
        TP2/P256 uses ``ceil(max_model_len / 256)`` compact columns. This is not a
        share of it -- ``W`` times wider than the same field meant under
        block-cyclic placement, which is what moves the 262144-token production
        point from the ``Items=4`` ladder rung to ``Items=16``.
    token_capacity
        The largest ``T`` this plan will be launched at.
    use_pdl
        Programmatic dependent launch. Refused by the extension below compute
        capability 9.0.
    """

    icp_degree: int
    icp_rank: int
    num_heads_local: int
    max_local_blocks: int
    token_capacity: int
    use_pdl: bool = False

    #: ``"prefill"`` or ``"decode"``. Set by the subclass, never by a caller.
    band = "unbanded"

    def __post_init__(self) -> None:
        if type(self) is SelectorPlan:
            raise TypeError(
                "SelectorPlan carries no band; construct a PrefillPlan or a "
                "DecodePlan. The two are not interchangeable -- decode is "
                "cudagraph-captured and prefill is not."
            )
        for name in (
            "icp_degree",
            "icp_rank",
            "num_heads_local",
            "max_local_blocks",
            "token_capacity",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an int")
        if not 2 <= self.icp_degree <= 8:
            raise ValueError("ICP candidate selection requires 2 <= icp_degree <= 8")
        if not 0 <= self.icp_rank < self.icp_degree:
            raise ValueError("icp_rank must lie in [0, icp_degree)")
        for name in ("num_heads_local", "max_local_blocks", "token_capacity"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if type(self.use_pdl) is not bool:
            raise TypeError("use_pdl must be a bool")
        # The id is `scan_block_begin + global_block_stride * column`, so the plan
        # bound the window's own width. The full bound, which needs the launch's
        # scan origin, is re-checked host-side in `local_candidates.cu`
        # (`scan_block_begin + global_block_stride * (blocks - 1) <= INT32_MAX`).
        if self.max_local_blocks - 1 > _INT32_MAX:
            raise ValueError("global block IDs exceed int32 capacity")
        rows = self.token_capacity * self.num_heads_group
        if rows * max(1, self._capacity_partitions) > _INT32_MAX:
            raise ValueError("selector launch exceeds int32 grid capacity")

    @property
    def num_heads_group(self) -> int:
        return self.icp_degree * self.num_heads_local

    @property
    def head_offset(self) -> int:
        # C7 pins this; the merge and the exchange re-check it.
        return self.icp_rank * self.num_heads_local

    @property
    def _capacity_partitions(self) -> int:
        """How many CTAs capacity alone would put on a row. Both bands are
        bounded by it: prefill's live extent cannot exceed capacity."""
        if self.max_local_blocks <= PARTITION_BLOCKS:
            return 0
        return (self.max_local_blocks + PARTITION_BLOCKS - 1) // PARTITION_BLOCKS

    @property
    def candidate_shape(self) -> tuple[int, int, int, int]:
        return (self.token_capacity, self.num_heads_group, CAND_K, 2)


@dataclass(frozen=True, slots=True)
class PrefillPlan(SelectorPlan):
    """A plan for the prefill band. Always eager, never captured.

    ``T`` may be any positive prefix of :attr:`token_capacity`. There is no
    workspace and no retained scratch: prefill allocates its own ``out`` and
    scratch per call, which is legal precisely because it is never captured.

    The launch geometry is :attr:`scan_extent`, a **startup constant**, not a
    per-call quantity. Both bounds a serving runtime has -- the longest query
    (``token_capacity``) and the longest context (``max_local_blocks``) -- are
    fixed when the engine is built, so the selector never has to ask the device
    how long this step's rows are.
    """

    band = "prefill"

    @property
    def scan_extent(self) -> int:
        """The extent the launch is sized from: capacity, and nothing else.

        ``max_local_blocks`` is the compact score-column capacity: at TP2,
        ``ceil(max_model_len / P)`` for physical P128 or P256. The kernel clamps every
        row to it (``valid = min(nvalid[token], blocks)``), so a launch sized
        from it covers every row that can ever arrive --
        ``prefill_scan_coverage(scan_extent) >= scan_extent`` -- and no per-step value
        can shrink it. That is the N2 fix: the failure mode the old
        synchronising check existed to catch (an under-reported live extent
        silently truncating every longer row) is not detected differently, it is
        **unreachable**.

        It does NOT mean every row is read to the end. The columns actually read
        are still ``valid`` per row, which is device metadata and stays on the
        device; the extent only sets the register depth and the partition count,
        which are launch geometry and must be host-known either way.
        """
        return self.max_local_blocks

    def launch(self, live_blocks: int) -> PrefillLaunch:
        """The geometry a launch at this extent takes, per :func:`prefill_launch`.

        The mirror of ``icp::prefill_launch``, kept parameterised because it is
        what ``planned_prefill_kernel`` and the ladder gates compare against.
        The entry point feeds it :attr:`scan_extent` and nothing else.
        """
        live = int(live_blocks)
        if not 0 <= live <= self.max_local_blocks:
            raise ValueError(
                f"live_blocks must lie in [0, max_local_blocks="
                f"{self.max_local_blocks}]; got {live}"
            )
        planned = prefill_launch(live)
        if planned.partitions > MAX_PREFILL_PARTITIONS:
            raise ValueError(
                f"prefill supports at most {MAX_PREFILL_PARTITIONS} partitions; "
                f"live_blocks={live} needs {planned.partitions}"
            )
        return planned

    def partial_shape(self, tokens: int,
                      live_blocks: int) -> tuple[int, int, int, int, int]:
        """The scratch this call needs, or a zero-width tensor if none."""
        return (int(tokens), self.num_heads_group,
                self.launch(live_blocks).partitions, CAND_K, 2)


@dataclass(frozen=True, slots=True)
class DecodePlan(SelectorPlan):
    """A plan for the decode band. Cudagraph-captured.

    ``T`` is pinned at :attr:`token_capacity` and ``out`` must be a buffer the
    caller allocated before the capture. Every quantity here is capacity, which
    is the only thing a capture may depend on.
    """

    band = "decode"

    @property
    def partial_count(self) -> int:
        """``S``: how many CTAs share a row, or 0 when one CTA covers it."""
        return self._capacity_partitions

    @property
    def partitioned(self) -> bool:
        return self.partial_count > 0

    @property
    def ladder_extent(self) -> int:
        """What the decode launcher's ladder is fed. Capacity, not live data."""
        return self.max_local_blocks

    @property
    def planned_items(self) -> int:
        """The rung :attr:`ladder_extent` selects, per :func:`ladder_items`."""
        return ladder_items(self.ladder_extent)

    @property
    def partial_shape(self) -> tuple[int, int, int, int, int]:
        return (self.token_capacity, self.num_heads_group, self.partial_count,
                CAND_K, 2)

    @property
    def workspace_bytes(self) -> int:
        return (
            self.token_capacity
            * self.num_heads_group
            * CAND_K
            * 2
            * 4
            * (1 + self.partial_count)
        )


@dataclass(frozen=True, slots=True)
class CandidateGeometry:
    """This launch's per-token device metadata. Aliases, never copies.

    ``local_valid_blocks``
        int32 ``[T]``: live blocks per token in this scan window, clamped by the
        kernel to ``max_local_blocks``.
    ``forced_column``
        int32 ``[T]``: the column of the forced block ``f = p // 128`` inside
        **this** scan window, i.e. the guarded inverse of the affine ID map.
        A missing parity or a block outside the live window maps to -1.
        Any negative value
        means "the forced block is not in this window" and is legal (bounded
        waves). The column is EXCLUDED from ordinary ranking on every rank; it
        is not scored ``+inf``, and the receiver reserves its output slot and
        injects it once after duplicate reduction (refined C3).
    ``active_rows``
        bool ``[T]``: false rows select nothing and emit ``(-inf, -1)``.
    ``scan_block_begin``
        host int: the absolute logical block id of column 0. The emitted global
        id is ``scan_block_begin + global_block_stride * column``.
    ``global_block_stride``
        host int, 1 by default. P128's columns are consecutive logical B128
        blocks. TP2/P256 uses stride 2 and origin ``2 * compact_begin + rank``.
        Its forced column is the guarded inverse of this mapping; the caller
        supplies that local column, while merge-side forcing stays global.

    These are builder preconditions, checked by device assertions rather than
    by synchronising host reads: a valid row needs
    ``0 <= local_valid_blocks <= max_local_blocks``, a non-negative
    ``forced_column`` must name a live column, and no valid block's score may be
    NaN.
    """

    local_valid_blocks: torch.Tensor
    forced_column: torch.Tensor
    active_rows: torch.Tensor
    scan_block_begin: int = 0
    global_block_stride: int = 1

    def __post_init__(self) -> None:
        # Reject tensors before any int() conversion can read device contents.
        for name in ("scan_block_begin", "global_block_stride"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be a host int")
        if not 0 <= self.scan_block_begin <= _INT32_MAX:
            raise ValueError("scan_block_begin must be a non-negative int32")
        if self.global_block_stride not in SUPPORTED_GLOBAL_BLOCK_STRIDES:
            raise ValueError("global_block_stride must be 1 or 2")

    @classmethod
    def from_metadata(cls, metadata: Any) -> CandidateGeometry:
        """Read this launch's geometry off an attention-metadata object by name.

        Duck-typed on purpose: this package has no serving-stack dependency, so
        the integration point is four required attribute names plus optional
        ``icp_global_block_stride_v1`` (default 1). See ``docs/INTEGRATION.md``.

        The names are **versioned** (``_v1``) because the meaning of the
        diagonal changed without its dtype or its shape changing. Keeping the
        old names would let an unmigrated metadata builder bind cleanly and feed
        block-cyclic local ordinals into a fragment-placement kernel: wrong
        selections, no error anywhere. A rename makes that a named failure on
        the first call.
        """
        missing = [n for n in _V1_REQUIRED if not hasattr(metadata, n)]
        if missing:
            raise AttributeError(
                "metadata is not refined-icp-v1: missing "
                + ", ".join(missing)
                + ". The selector ABI changed -- global ids are "
                "scan_block_begin + global_block_stride * column, forcing is "
                "exclusion rather than "
                "+inf, and there is no diagonal owner. icp_owns_diagonal was "
                "removed and icp_diagonal_local was renamed to "
                "icp_forced_column_v1 so that an unmigrated builder fails here "
                "instead of silently selecting the wrong blocks."
            )
        if hasattr(metadata, "icp_owns_diagonal"):
            raise AttributeError(
                "metadata still exposes icp_owns_diagonal, which refined-icp-v1 "
                "removes. Mixing the two generations is the exact silent "
                "mis-selection this check exists to prevent."
            )
        return cls(
            local_valid_blocks=metadata.topk_num_valid_pages,
            forced_column=metadata.icp_forced_column_v1,
            active_rows=metadata.icp_active_rows,
            scan_block_begin=metadata.icp_scan_block_begin_v1,
            global_block_stride=getattr(metadata, "icp_global_block_stride_v1", 1),
        )


# `eq=False`: the fields are tensors, and a generated `__eq__` would return a
# tensor rather than a bool. Identity is the only comparison that means anything.
@dataclass(frozen=True, slots=True, eq=False)
class CandidateWorkspace:
    """DECODE ONLY. Retained allocations plus the extension, both obtained
    before capture.

    ``candidates`` is a pre-allocated ``[T_cap, H_group, 16, 2]`` output buffer
    the decode caller passes as ``out``, which is what keeps allocation out of
    the captured region. ``partials`` is the partitioned arms' scratch and is
    empty when one CTA covers a row.

    There is no prefill equivalent: prefill runs eager, so it allocates what
    that call needs from the live extent, and a buffer sized from capacity would
    be sized wrong.

    Keep the workspace alive while any captured graph references it. Calls
    sharing one workspace must run on one ordered stream; concurrent callers
    need separate workspaces.
    """

    plan: DecodePlan
    candidates: torch.Tensor
    partials: torch.Tensor
    _extension: Any = field(repr=False, compare=False)


def allocate_workspace(plan: DecodePlan,
                       device: torch.device | str) -> CandidateWorkspace:
    """Load the extension and allocate the decode plan's retained capacity.

    Eager on purpose, and it refuses to run during a capture: the JIT build is
    an nvcc invocation and the allocation would be reclaimed when the capture's
    private pool is reset. Call it at model construction.
    """
    if not isinstance(plan, DecodePlan):
        raise TypeError(
            "the candidate workspace is decode's: it exists to keep allocation "
            "out of a capture, and prefill is never captured. Prefill needs no "
            "workspace -- see select_prefill_candidates()."
        )
    import torch  # noqa: PLC0415

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("the candidate workspace requires a CUDA device")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "allocate the candidate workspace before cudagraph capture: "
                "the JIT build is an nvcc invocation and a tensor allocated "
                "inside a capture is freed when its pool is reset"
            )
        return CandidateWorkspace(
            plan=plan,
            candidates=torch.empty(plan.candidate_shape, dtype=torch.float32,
                                   device=device),
            partials=torch.empty(plan.partial_shape, dtype=torch.float32,
                                 device=device),
            _extension=_build._select_ext(),
        )


def select_prefill_candidates(scores: torch.Tensor,
                              geometry: CandidateGeometry, *,
                              plan: PrefillPlan,
                              live_blocks: int,
                              out: torch.Tensor | None = None,
                              partials: torch.Tensor | None = None,
                              arm: SelectArm | str | int | None = SHIPPED_ARM,
                              full_row_threads: int = 512,
                              full_row_cached_items: int = 3,
                              full_row_four_warp_finish: bool = False,
                              validate: bool = True,
                              ) -> torch.Tensor:
    """Select this rank's local Top-16 on the eager prefill path.

    ``scores`` is ``[T, H_group, N]`` with ``0 < T <= plan.token_capacity``; the
    trailing extent must equal ``plan.max_local_blocks`` even when far fewer
    blocks are live, because it is the **row stride** and nothing else.

    The launch geometry -- the rung and the partition count -- is
    ``plan.scan_extent``, i.e. ``max_local_blocks``, a **startup constant**.
    Nothing per-step sizes it.

    ``live_blocks`` **no longer sizes anything** and is retained only so an
    existing caller keeps binding. It is still range-checked against capacity on
    the host, because a value outside ``[0, max_local_blocks]`` says the
    caller's metadata is wrong and that is worth naming; it can no longer make
    the selector read fewer columns.

    *Why (N2, nonblocking execution, 2026-09-13).* The extent used to be this
    argument, and because under-reporting it truncates every longer row with no
    other symptom, the entry point re-derived the true maximum from
    ``local_valid_blocks`` -- ``int(...max())``, a CUDA reduction turned into a
    Python integer, on every prefill layer and every query chunk. The accepted
    policy forbids a device read on the submission path, and being eager is not
    an exemption. The fix is not to drop the check and trust the caller: a
    serving runtime's max query length and max model length are fixed at
    startup, so the extent is taken from the bound the plan already carries.
    ``prefill_scan_coverage(plan.scan_extent) >= plan.scan_extent >= valid`` for every
    row the kernel will ever see (it clamps ``valid`` to ``max_local_blocks``
    itself), so truncation is unreachable rather than undetected.

    *What it costs, stated plainly.* The rung is now the capacity rung on every
    call, so a short first chunk pays the register depth and, above
    ``PARTITION_BLOCKS`` of capacity, the partition CTAs and the combine pass
    that its own live extent would not have needed. It does **not** cost a
    longer scan: the columns a row reads are ``min(local_valid_blocks, N)``,
    device metadata the kernel already holds, and the extent only bounds them.
    The emitted candidates are unchanged -- every rung is bit-exact, which
    ``test_the_live_extent_does_not_change_the_answer`` gates.

    ``out`` is ``[T, H_group, 16, 2]`` fp32, allocated here if omitted.

    ``partials`` is the partitioned arm's scratch,
    ``plan.partial_shape(T, plan.scan_extent)`` fp32, allocated here if omitted.
    It is **pure scratch**: the kernel writes every used slot before it reads
    one, so a retained buffer and a fresh one give the same answer, and a caller
    that runs many chunks and many layers can hold one and pass a prefix view of
    it -- ``retained[:T]`` is contiguous, which is what the shape check demands.

    Its shape depends on ``T`` and on :attr:`PrefillPlan.scan_extent`; the
    latter is a startup constant, so the only thing that varies is the row
    count, and above ``PARTITION_BLOCKS`` of capacity the partition count is
    fixed too. On every capacity that fits one CTA per row the tensor is
    zero-width and there is nothing to retain.

    Passing it changes **nothing** about what the launch reads: the extent is
    still :attr:`PrefillPlan.scan_extent` and no host value is derived from a
    device tensor here. This is an allocation, and only an allocation.

    Returns ``out``. Every used slot is overwritten; nothing is accumulated.

    ``arm`` accepts only ``RADIX_BOUNDED`` (the unchanged default) or
    ``RADIX_FULL_ROW``. ``None`` also means bounded; ``ICP_SELECTOR`` does not
    override prefill. The whole-row arm uses one CTA per (token, head) at
    capacities up to 8192, and the existing bounded partitioned path above
    8192. It retains the same output and scratch shapes, although its fast
    path does not read or write ``partials``. Both arms check activity, exact
    per-row prefix and forced-column exclusion before reading scores. Choosing
    an arm changes neither the prefill producer's complete-write contract nor
    this entry point's eager-only policy.

    ``full_row_threads`` and ``full_row_cached_items`` are explicit prefill
    tuning controls. Their unchanged default is ``(512, 3)``; supported
    alternatives are ``(128, 0)``, ``(128, 12)``, ``(256, 0)`` and ``(256, 6)``.
    Zero cached items streams the exact prefix; the cached alternatives retain
    at most 1536 scores per CTA. Both quantities are host ints, never device
    counts. Non-default controls require ``RADIX_FULL_ROW``. The same retained
    buffers, bounds, canonical output, and fallback above 8192 apply to all.

    ``full_row_four_warp_finish=True`` opts the 128-thread specializations
    into an exact four-warp merge for boundaries containing 65--128 candidates.
    The default keeps the existing finish. This host bool changes no output
    contract, capacity guard or device prefix; it is not a device-dependent
    dispatch decision and is not available on the unchanged 512-thread kernel.

    When the actual capacity is at most 32, either explicit 128-thread choice
    uses one warp per score row, with four rows per CTA. This static admission
    does not consult ``live_blocks`` or read device counts on the host. Both
    finish settings use this exact tiny-row kernel; the recorded grid is
    ``ceil(T * H_group / 4)`` and :func:`planned_prefill_kernel` reports its
    symbol. Other capacities, the 512-thread default and bounded radix retain
    their existing dispatch.

    ``validate=False`` is the pre-validated hot form: the caller has already
    made the same call with ``validate=True`` for this step's bindings (same
    plan, geometry, buffers and controls; only ``scores``' storage differs), so
    every host check, the arm/control resolution and any allocation are
    skipped and the call is the native launch alone. ``out`` and ``partials``
    are then required and ``arm`` must already be a resolved
    :class:`SelectArm`.
    """
    if not validate:
        if out is None or partials is None:
            raise ValueError("validate=False requires out and partials")
        _build._select_ext().select_prefill_candidates(
            scores,
            geometry.local_valid_blocks,
            geometry.forced_column,
            geometry.active_rows,
            out,
            partials,
            geometry.scan_block_begin,
            plan.use_pdl,
            plan.scan_extent,
            global_block_stride=geometry.global_block_stride,
            arm=int(arm),
            full_row_threads=full_row_threads,
            full_row_cached_items=full_row_cached_items,
            full_row_four_warp_finish=full_row_four_warp_finish,
        )
        return out
    if not isinstance(plan, PrefillPlan):
        raise TypeError(
            f"select_prefill_candidates needs a PrefillPlan, got "
            f"{type(plan).__name__}"
        )
    resolved = _prefill_arm(arm)
    _prefill_full_row_config(resolved, full_row_threads, full_row_cached_items,
                             full_row_four_warp_finish)
    import torch  # noqa: PLC0415

    if torch.cuda.is_current_stream_capturing():
        # The allocation and the per-call `T` are legal only because this band
        # is eager. (It no longer reads a device value on the host at all --
        # that was N2, and it is gone from both bands.)
        raise RuntimeError(
            "prefill is not capturable: it allocates and takes T per call. "
            "Capture select_decode_candidates(), which does neither."
        )
    if isinstance(live_blocks, torch.Tensor):
        raise TypeError(
            "live_blocks must be a host int. Reading it off a device tensor "
            "here is exactly the submission-path synchronisation N2 removed, "
            "and it would reappear inside this int() call."
        )
    # Range-checked, then DISCARDED: see the docstring. Nothing below reads it.
    plan.launch(int(live_blocks))
    extent = plan.scan_extent
    tokens = int(scores.shape[0])
    partial_shape = plan.partial_shape(tokens, extent)
    if out is None:
        out = torch.empty((tokens, plan.num_heads_group, CAND_K, 2),
                          dtype=torch.float32, device=scores.device)
    if partials is None:
        partials = torch.empty(partial_shape, dtype=torch.float32,
                               device=scores.device)
    # A supplied buffer is validated by `_validate` below, against the SAME
    # `partial_shape` the allocation would have used -- shape, dtype, device and
    # contiguity. Nothing is reshaped or re-strided to fit.
    _validate(scores, geometry, plan=plan, out=out, partials=partials,
              partial_shape=partial_shape)
    _build._select_ext().select_prefill_candidates(
        scores,
        geometry.local_valid_blocks,
        geometry.forced_column,
        geometry.active_rows,
        out,
        partials,
        geometry.scan_block_begin,
        plan.use_pdl,
        extent,
        global_block_stride=geometry.global_block_stride,
        arm=int(resolved),
        full_row_threads=full_row_threads,
        full_row_cached_items=full_row_cached_items,
        full_row_four_warp_finish=full_row_four_warp_finish,
    )
    return out


def select_decode_candidates(scores: torch.Tensor,
                             geometry: CandidateGeometry, *,
                             plan: DecodePlan,
                             workspace: CandidateWorkspace,
                             out: torch.Tensor,
                             arm: SelectArm | str | int | None = None,
                             validate: bool = True,
                             ) -> torch.Tensor:
    """Select this rank's local Top-16 on the decode path, under capture.

    ``scores`` is ``[T, H_group, N]`` with ``T == plan.token_capacity``: a
    capture bakes the shape in, so a shorter prefix is a different graph, and
    rows past the real batch carry ``(-inf, -1)`` and are not read.

    ``out`` is **required** -- ``workspace.candidates`` is the buffer to pass.
    There is no allocating form on this path: a tensor allocated inside a
    capture is freed when the capture's pool is reset, so the graph would replay
    against memory that has been handed to something else.

    Nothing here allocates, reads a device value on the host, or branches on
    one, so the call is capturable. Returns ``out``.

    ``validate=False`` skips the plan/workspace/shape checks for a binding the
    caller already validated this step; only the native launch remains.
    """
    if not validate:
        resolved = arm if isinstance(arm, SelectArm) else selector_arm(arm)
        return _launch_decode(scores, geometry, plan=plan, workspace=workspace,
                              out=out, resolved=resolved)
    if not isinstance(plan, DecodePlan):
        raise TypeError(
            f"select_decode_candidates needs a DecodePlan, got "
            f"{type(plan).__name__}"
        )
    if int(scores.shape[0]) != plan.token_capacity:
        raise ValueError(
            f"decode is captured at a fixed shape: T must equal "
            f"token_capacity ({plan.token_capacity}); got {scores.shape[0]}. "
            "Pad the batch and let the padded rows emit (-inf, -1)."
        )
    return _select(scores, geometry, plan=plan, workspace=workspace, out=out,
                   arm=arm)


def select_with_control(scores: torch.Tensor, geometry: CandidateGeometry, *,
                        plan: DecodePlan, workspace: CandidateWorkspace,
                        out: torch.Tensor, control: int,
                        arm: SelectArm | str | int | None = None) -> torch.Tensor:
    """TEST ONLY: run the selector with one site deliberately perturbed.

    ``control`` selects the perturbation; they are listed at the
    ``ICP_RS_NEGATIVE_CONTROL`` macro in ``csrc/local_candidates_rs.cuh``.
    ``control = 0`` perturbs nothing, so a gate that cannot tell 0 from a real
    control is not gating. This runs a **different extension**, built with the
    macro defined; the shipping one contains none of this code.

    It drives the capacity-fed launcher, so it takes a :class:`DecodePlan`.
    The prefill launcher has no controls in this repository; the standalone
    suite that gates it lives in the campaign workspace, not here.

    ``arm`` reaches the control extension as the **same** ``cap`` code the
    shipping launcher dispatches on -- ``0`` sort, ``-1``/``-2`` radix, ``> 0``
    capped -- so the control and the thing it controls take the same branch.
    ``RADIX_FULL_ROW`` has no mutation-control implementation and is rejected
    before tensor validation or building the control extension.
    """
    resolved = selector_arm(arm)
    if resolved is SelectArm.RADIX_FULL_ROW:
        raise ValueError(
            "RADIX_FULL_ROW has no mutation-control implementation; "
            "select_with_control supports only the legacy selector arms."
        )
    _validate(scores, geometry, plan=plan, out=out,
              partials=workspace.partials, partial_shape=plan.partial_shape)
    _build._select_control_ext().select_local_candidates_control(
        scores,
        geometry.local_valid_blocks,
        geometry.forced_column,
        geometry.active_rows,
        out,
        workspace.partials[: scores.shape[0]],
        geometry.scan_block_begin,
        plan.use_pdl,
        int(control),
        # NOT `max(int(resolved), 0)`. That was correct while the control
        # extension treated every cap <= 0 as the radix arm; it now dispatches
        # cap == 0 to the SORT arm, exactly as the shipping launcher does, so
        # clamping would silently run the sort arm's controls under the radix
        # arm's name.
        int(resolved),
        global_block_stride=geometry.global_block_stride,
    )
    return out


def planned_kernel(arm: SelectArm | str | int, blocks: int) -> dict[str, int]:
    """``[symbol, regs, shared]`` of the kernel this ``(arm, blocks)`` launches.

    A second, independent copy of the ladder, living in the extension. The gate
    compares it against :func:`last_launch` so a dispatch that quietly took a
    different arm is caught -- which no output comparison can do, because the
    arms are bit-exact.
    """
    symbol, regs, shared = _build._select_ext().selector_kernel_attributes(
        int(selector_arm(arm)), int(blocks)
    )
    return {"symbol": symbol, "regs": regs, "shared": shared}


def _prefill_arm(value: SelectArm | str | int | None) -> SelectArm:
    """Resolve prefill's explicit choices without consulting ICP_SELECTOR."""
    resolved = selector_arm(SHIPPED_ARM if value is None else value)
    if resolved not in (SelectArm.RADIX_BOUNDED, SelectArm.RADIX_FULL_ROW):
        raise ValueError(
            "prefill arm must be RADIX_BOUNDED or RADIX_FULL_ROW; "
            f"got {resolved.name}"
        )
    return resolved


def _prefill_full_row_config(
    arm: SelectArm, threads: int, cached_items: int,
    four_warp_finish: bool = False,
) -> None:
    """Reject unsupported tuning controls without a device conversion or JIT."""
    if type(threads) is not int or type(cached_items) is not int:
        raise TypeError("prefill full-row threads and cached items must be host ints")
    if type(four_warp_finish) is not bool:
        raise TypeError("prefill full-row four-warp finish must be a host bool")
    supported = ((512, 3), (128, 0), (128, 12), (256, 0), (256, 6))
    if (threads, cached_items) not in supported:
        raise ValueError(f"unsupported prefill full-row configuration; use {supported}")
    if arm is not SelectArm.RADIX_FULL_ROW and (threads, cached_items) != (512, 3):
        raise ValueError("non-default full-row controls require RADIX_FULL_ROW")
    if four_warp_finish and (arm is not SelectArm.RADIX_FULL_ROW or threads != 128):
        raise ValueError("four-warp finish requires a 128-thread RADIX_FULL_ROW")


def planned_prefill_kernel(
    live_blocks: int, *, arm: SelectArm | str | int | None = SHIPPED_ARM,
    full_row_threads: int = 512, full_row_cached_items: int = 3,
    full_row_four_warp_finish: bool = False,
) -> dict[str, int]:
    """``[symbol, regs, shared]`` of the kernel prefill launches at this extent.

    Pass ``plan.scan_extent``, the capacity supplied by the public entry point.
    This independently checks the selected arm and rung: exact output equality
    alone cannot detect an unintended fallback to a slower implementation.
    """
    resolved = _prefill_arm(arm)
    _prefill_full_row_config(resolved, full_row_threads, full_row_cached_items,
                             full_row_four_warp_finish)
    symbol, regs, shared = _build._select_ext().prefill_kernel_attributes(
        int(live_blocks), int(resolved), full_row_threads, full_row_cached_items,
        full_row_four_warp_finish
    )
    return {"symbol": symbol, "regs": regs, "shared": shared}


def record_launches(enabled: bool) -> None:
    """Record each select launch's identity. Off by default: it is a host call."""
    _build._select_ext().set_launch_recording(bool(enabled))


def last_launch() -> dict[str, int]:
    """The last recorded select launch. ``count`` is 0 if nothing was recorded."""
    symbol, regs, shared, grid, count = _build._select_ext().last_launch_info()
    return {"symbol": symbol, "regs": regs, "shared": shared, "grid": grid,
            "count": count}


def last_launch_threads() -> int:
    """Actual blockDim.x of the recorded selector; -1 before any recording."""
    return int(_build._select_ext().last_launch_threads())


def _select(scores: torch.Tensor, geometry: CandidateGeometry, *,
            plan: DecodePlan, workspace: CandidateWorkspace,
            out: torch.Tensor,
            arm: SelectArm | str | int | None) -> torch.Tensor:
    resolved = selector_arm(arm)
    if workspace.plan != plan:
        # Dataclass equality is class-sensitive, so this also rejects a
        # workspace allocated for the other band.
        raise ValueError(
            f"this workspace was allocated for {workspace.plan!r}, not {plan!r}"
        )
    _validate(scores, geometry, plan=plan, out=out,
              partials=workspace.partials, partial_shape=plan.partial_shape)
    return _launch_decode(scores, geometry, plan=plan, workspace=workspace,
                          out=out, resolved=resolved)


def _launch_decode(scores: torch.Tensor, geometry: CandidateGeometry, *,
                   plan: DecodePlan, workspace: CandidateWorkspace,
                   out: torch.Tensor, resolved: SelectArm) -> torch.Tensor:
    tokens = int(scores.shape[0])
    partials = workspace.partials
    if int(partials.shape[0]) != tokens:
        partials = partials[:tokens]
    extension = workspace._extension
    if resolved > 0:
        entry = extension.select_local_candidates_capped
        extra = {"cap": int(resolved)}
    elif resolved is SelectArm.RADIX_FULL_ROW:
        entry, extra = extension.select_local_candidates_full_row, {}
    elif resolved is SelectArm.RADIX_BOUNDED:
        entry, extra = extension.select_local_candidates_radix_sr, {}
    elif resolved is SelectArm.RADIX:
        entry, extra = extension.select_local_candidates_radix, {}
    else:
        entry, extra = extension.select_local_candidates, {}
    entry(
        scores,
        geometry.local_valid_blocks,
        geometry.forced_column,
        geometry.active_rows,
        out,
        partials,
        geometry.scan_block_begin,
        plan.use_pdl,
        global_block_stride=geometry.global_block_stride,
        **extra,
    )
    return out


def _validate(scores: torch.Tensor, geometry: CandidateGeometry, *,
              plan: SelectorPlan, out: torch.Tensor, partials: torch.Tensor,
              partial_shape: tuple[int, ...]) -> None:
    import torch  # noqa: PLC0415

    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise ValueError("scores must be a [T, H_group, N] tensor")
    if (geometry.scan_block_begin
            + geometry.global_block_stride * (plan.max_local_blocks - 1)
            > _INT32_MAX):
        raise ValueError("global block IDs exceed int32 capacity")
    tokens = int(scores.shape[0])
    if not 0 < tokens <= plan.token_capacity:
        raise ValueError(
            f"T must be in (0, token_capacity={plan.token_capacity}]; got "
            f"{tokens}"
        )
    device = scores.device
    if device.type != "cuda":
        raise ValueError("candidate selection requires CUDA scores")
    _check("scores", scores,
           shape=(tokens, plan.num_heads_group, plan.max_local_blocks),
           dtype=torch.float32, device=device)
    for name, dtype in (
        ("local_valid_blocks", torch.int32),
        ("forced_column", torch.int32),
        ("active_rows", torch.bool),
    ):
        _check(name, getattr(geometry, name), shape=(tokens,), dtype=dtype,
               device=device)
    _check("out", out, shape=(tokens, plan.num_heads_group, CAND_K, 2),
           dtype=torch.float32, device=device)
    _check("partials", partials, shape=partial_shape, dtype=torch.float32,
           device=device)


def _check(name: str, tensor: torch.Tensor, *, shape: tuple[int, ...],
           dtype: torch.dtype, device: torch.device) -> None:
    import torch  # noqa: PLC0415

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} shape must be {shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} must have dtype {dtype} on {device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
