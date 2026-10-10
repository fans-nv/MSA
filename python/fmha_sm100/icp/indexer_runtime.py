# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-local ICP indexer runtime: retained capacity, step planning, layer launch.

A serving framework constructs one :class:`IcpIndexerRuntime` per indexer
metadata builder, calls :meth:`IcpIndexerRuntime.plan_step` once per forward and
:meth:`IcpIndexerRuntime.run_layer` once per sparse layer. The framework keeps
the KV writer, cache binding, metadata types and the candidate exchange's
lifetime; this module owns scorer/selector capacity, the per-step windows and
the launch loop.

Production profile only: P128/R64 at W=2, CuTe decode scorer (required),
full-row decode and prefill selectors, D3 fused decode publication, derived
prefill wave ladder. Allocation order and lifetimes are those of the vLLM
builder this was moved from, so captured graph addresses are unchanged.
"""

from __future__ import annotations

import functools
import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .exchange_plan import (
    ICP_PREFILL_EXTENT_STEPS_PER_OCTAVE,
    admitted_exchange_extents,
    exchange_token_capacities,
    prefill_exchange_token_capacities,
    select_token_capacity,
)
from .local_indexer import (
    bind_cute_launch,
    bind_fmha_launch,
    icp_cute_decode_window,
    icp_decode_chunk_tokens,
    icp_prefill_arena_elems,
    icp_prefill_column_rungs,
    icp_prefill_rung,
    icp_prefill_wave_ladder,
    icp_row_windows,
    prefill_full_row_controls,
    prefill_selector_kwargs,
)

__all__ = [
    "IcpChunk",
    "IcpIndexerConfig",
    "IcpIndexerRuntime",
    "IcpLiveViews",
    "IcpStep",
]

_logger = logging.getLogger(__name__)

# Page size == sparse block size == index-K block; fmha tile id == M3 block id.
PAGE_SIZE = 128
# Inner MSA tile; `_qo_tile_size_for` pins it to 128 whenever icp_c > 1.
ICP_MSA_QUERY_TILE_TOKENS = 128
# Decode fp32 score + uint8 validity scratch budget (r13 sizing).
ICP_SCORE_SCRATCH_BUDGET_BYTES = 32 * 1024 * 1024
# Qualified per-chunk query-length envelope.
ICP_MAX_OUTER_CHUNK_TOKENS = 1024
ICP_PREFILL_COLUMN_FLOOR = 4
ICP_PREFILL_COLUMN_STEPS_PER_OCTAVE = 2
# Retained prefill score + validity arena, once per runtime (not per layer).
ICP_PREFILL_SCORE_SCRATCH_BUDGET_BYTES = 1024 * 1024 * 1024
# Bounded by the wave-invariance gate and the token budget (r13).
ICP_MAX_PREFILL_WAVE_TOKENS = 16384
# r13's fixed CuTe decode split; MSA's `auto_split_k` (0) is never selected.
ICP_CUTE_SPLIT_K = 128
# Above this the package dispatches full-row work to the bounded selector
# silently, so admission is refused instead.
ICP_FULL_ROW_MAX_COLUMNS = 8192
# Measured prefill full-row recipe (not the package defaults (512, 3, False)).
ICP_PREFILL_FULL_ROW_THREADS = 128
ICP_PREFILL_FULL_ROW_CACHED_ITEMS = 0
ICP_PREFILL_FOUR_WARP_FINISH_MAX_COLUMNS = 128
# The fused KV producer must be a non-PDL launch: it is the barrier that makes
# the CuTe scorer's pre-wait reads of GPU-produced step metadata safe.
ICP_PRODUCER_ENABLE_PDL = False
# Scorer input ABI: ordinary block table + row pitch + row origin + exact lengths.
DIRECT_TABLE_ABI_VERSION = 1
DIRECT_TABLE_PLAN_ARGS = frozenset(
    {"icp_block_table", "icp_block_table_row_stride", "icp_block_table_row_begin"}
)
_EXPECTED_ABIS = (
    "refined-icp-v1.abi.1",
    "refined-icp-v1.k2.2",
    "refined-icp-v1.C4",
    2,
)

# Greppable markers; external capture harnesses count their occurrences.
ICP_PREFILL_WAVE_MARKER = "ICP_PREFILL_WAVE_SCHEMA"
ICP_PREFILL_EXTENT_MARKER = "ICP_PREFILL_EXTENT_SCHEMA"
ICP_PREFILL_PATH_MARKER = "ICP_PREFILL_PATH"
ICP_DECODE_PROFILE_MARKER = "ICP_DECODE_PROFILE"
ICP_CUTE_STATUS_MARKER = "ICP_CUTE_DECODE_STATUS"
ICP_CUTE_ENGAGED_MARKER = "ICP_CUTE_DECODE_ENGAGED"
ICP_DECODE_SELECTOR_MARKER = "ICP_DECODE_SELECTOR"
ICP_PREFILL_SELECTOR_MARKER = "ICP_PREFILL_SELECTOR"

_MSA_API_MODULE = "fmha_sm100.icp.scorer.prefill.api"
_CUTE_DECODE_MODULE = "fmha_sm100.icp.scorer.decode.icp_decode_score"

# Once per process, as in the vLLM builder this replaces.
_prewarm_checked = False
_decode_profile_logged = False
# Recorded (not replayed) CuTe invocations per rank.
_ICP_CUTE_ENGAGEMENTS: dict[int, int] = {}


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _msa_api() -> Any:
    """The canonical ICP prefill scorer API."""
    import importlib

    try:
        return importlib.import_module(_MSA_API_MODULE)
    except ImportError as exc:
        raise ImportError(
            f"msa_icp requires {_MSA_API_MODULE}; install MSA with ICP support: {exc}"
        ) from exc


def _msa_cute_decode_module() -> Any:
    """The CuTe decode scorer with live-prefix ABI 2 and launch ABI 3, or None."""
    import importlib

    try:
        module = importlib.import_module(_CUTE_DECODE_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == _CUTE_DECODE_MODULE:
            return None
        raise
    if module.ICP_DECODE_SCORE_ABI_VERSION != 2:
        raise RuntimeError("MiniMax ICP decode requires MSA live-prefix ABI 2")
    if getattr(module, "ICP_DECODE_SCORE_LAUNCH_ABI_VERSION", 1) != 3:
        raise RuntimeError(
            "MiniMax ICP decode requires MSA's per-batch-split launch ABI 3"
        )
    return module


def _msa_has_direct_table() -> bool:
    """Whether the planner takes the ordinary block table AND the kernel half carries it."""
    import inspect

    api = _msa_api()
    impl = getattr(api, "_fmha_sm100_plan_impl", None)
    if impl is None:
        return False
    try:
        params = inspect.signature(impl).parameters
    except (TypeError, ValueError):
        return False
    if not params.keys() >= DIRECT_TABLE_PLAN_ARGS:
        return False
    return bool(getattr(api, "_FMHA_HAS_ICP_DIRECT_TABLE", False))


def _device_arch_str(device) -> str:
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:  # noqa: BLE001 - diagnostics only
        return "unknown"
    return f"{major}{minor}"


def _uniform_decode_query_len(query_start_loc_cpu, has_prefill: bool) -> int:
    if has_prefill or int(query_start_loc_cpu[0]) != 0:
        return 0
    lengths = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    live = lengths[lengths > 0]
    if not live.numel():
        return 0
    # V2 packs live requests before zero-length capture padding.
    if not torch.equal(lengths[: live.numel()], live):
        return 0
    query_len = int(live[0])
    return query_len if query_len <= 4 and bool((live == query_len).all()) else 0


def _icp_fused_publish(scores, geometry, token_offset, exchange, layer_idx):
    """D3: one decode window's selector, publishing into the owners' windows."""
    exchange.publish(scores, geometry, layer_idx=layer_idx, token_offset=token_offset)


def icp_cute_rows(*, use_cute_decode: bool, has_prefill: bool, num_tokens: int) -> int:
    """Leading rows the CuTe scorer takes: a pure-decode step only (r13)."""
    return num_tokens if use_cute_decode and not has_prefill else 0


def _allocate_score_planes(*, elems, device, budget_bytes, heads_group, columns):
    """The retained fp32 score arena and uint8 validity arena; failure is a startup error."""
    want = elems * (4 + 1)
    try:
        scores = torch.empty(elems, dtype=torch.float32, device=device)
        valid = torch.empty(elems, dtype=torch.uint8, device=device)
    except (RuntimeError, MemoryError) as exc:
        raise RuntimeError(
            "MiniMax-M3 ICP indexer: the prefill score scratch could not be "
            f"allocated. It needs {want} bytes ({want / 2**30:.2f} GiB) on "
            f"{device}: one fp32 score arena and one uint8 validity arena, each "
            f"{elems} cells, holding the widest of the prefill wave ladder's "
            f"[wave, {heads_group}, <= {columns}] planes under a "
            f"{budget_bytes // 2**20} MiB budget, once per rank and NOT per "
            "sparse layer. Lower gpu memory utilization, max-num-batched-tokens "
            "or max-model-len, or lower IcpIndexerConfig.prefill_score_budget_bytes "
            "(fewer one-launch columns)."
        ) from exc
    return scores, valid


@dataclass(frozen=True)
class IcpIndexerConfig:
    """Startup constants of one ICP indexer; identical on every rank but ``rank``."""

    world_size: int
    rank: int
    num_index_heads: int  # H_local; H_group = world_size * num_index_heads
    index_head_dim: int
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    max_decode_query_len: int
    page_tokens: int = PAGE_SIZE
    cudagraph_capture_sizes: tuple[int, ...] = ()
    fused_decode: bool = True
    decode_pdl: bool = True
    debug_checks: bool = False
    prefill_score_budget_bytes: int = ICP_PREFILL_SCORE_SCRATCH_BUDGET_BYTES


@dataclass
class IcpChunk:
    """One scorer + selector window of a step; launches are bound by plan_step."""

    plan: dict | None  # None for the CuTe decode route
    token_begin: int
    num_live_tokens: int
    # The ordinary block table, whole and at offset 0 so captured graphs keep its address.
    block_table: torch.Tensor
    block_table_row_stride: int
    block_table_row_begin: int
    qo_offset: torch.Tensor
    live_blocks: int
    score_extent: int
    select_extent: int
    topk_num_valid_pages: torch.Tensor
    icp_forced_column_v1: torch.Tensor
    icp_active_rows: torch.Tensor
    icp_scan_block_begin_v1: int = 0
    icp_global_block_stride_v1: int = 1
    cute_scorer: Callable[..., None] | None = None
    # The no-init scorer's write bound equals the selector's read bound.
    rank_aware_nvalid: bool = False
    geometry: object | None = None
    launch_score: Callable[..., None] | None = None
    select: Callable[..., object] | None = None
    fused_select: Callable[[Any, int], None] | None = None
    cute_check_args: tuple | None = None
    checked: bool = False


@dataclass(frozen=True)
class IcpLiveViews:
    """Retained views the framework's first sparse KV writer fills each step."""

    positions: torch.Tensor
    active: torch.Tensor
    local_nvalid: torch.Tensor
    local_forced: torch.Tensor
    global_nvalid: torch.Tensor
    forced: torch.Tensor
    n_ordinary: torch.Tensor
    candidates: torch.Tensor
    qo_offsets: torch.Tensor
    chunk_width: int
    device_plan: object | None


@dataclass
class IcpStep:
    """One forward's plan, shared by every sparse layer; buffers are the runtime's."""

    chunks: list[IcpChunk]
    scores: torch.Tensor
    valid: torch.Tensor
    candidates: torch.Tensor
    forced: torch.Tensor
    n_ordinary: torch.Tensor
    prefill_plan: object
    prefill_partials: torch.Tensor
    decode_plans: dict[int, object]
    decode_workspaces: dict[int, object]
    has_prefill: bool
    token_capacity: int
    chunk_width: int
    decode_arm: object
    prefill_arm: object
    prefill_controls: tuple[int, int, bool]
    num_tokens: int
    live: IcpLiveViews
    step_n_valid: torch.Tensor
    topk_num_valid_pages: torch.Tensor
    forced_column: torch.Tensor
    active_rows_bool: torch.Tensor
    query_start_loc: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    # uint8 retype of the active plane for the CuTe scorer's ABI.
    active_rows: torch.Tensor | None = None
    cute_rows: int = 0
    decode_chunk: int = 0
    fused_decode: bool = False
    checked: bool = False
    exchange_candidates: torch.Tensor | None = None
    exchange_buf: torch.Tensor | None = None
    exchange_out: torch.Tensor | None = None
    fused_ready: bool = False
    fused_padding: tuple = ()


class IcpIndexerRuntime:
    """Retained ICP indexer capacity plus per-step planning and the layer launch loop."""

    def __init__(
        self,
        cfg: IcpIndexerConfig,
        *,
        device: torch.device,
        logger: logging.Logger | None = None,
        prewarm_check: str = "fail",
    ) -> None:
        self.cfg = cfg
        self.logger = logger or _logger
        self.icp_c = cfg.world_size
        self.icp_rank = cfg.rank
        self.icp_physical_page_tokens = cfg.page_tokens
        self.num_index_heads = cfg.num_index_heads
        self.index_head_dim = cfg.index_head_dim
        self.max_num_reqs = cfg.max_num_seqs
        self.max_decode_query_len = cfg.max_decode_query_len
        self.max_blocks_per_req = _cdiv(cfg.max_model_len, cfg.page_tokens)
        if self.icp_c != 2:
            raise ValueError("MiniMax M3 ICP requires TP2")
        if prewarm_check not in ("fail", "warn"):
            raise ValueError(f"prewarm_check={prewarm_check!r} must be fail or warn")
        self._prewarm_check = prewarm_check
        self._closed = False
        self._init_icp(device)

    # -- startup ----------------------------------------------------------

    @staticmethod
    def check_installed() -> None:
        """Refuse an MSA build whose ABIs differ from the ones this runtime emits."""
        from . import abi
        from .candidates import CANDIDATE_MAPPING_ABI_VERSION

        actual_abis = (
            abi.ABI_VERSION,
            abi.K2_ABI_VERSION,
            abi.K2_CARRIER_VERSION,
            CANDIDATE_MAPPING_ABI_VERSION,
        )
        if actual_abis != _EXPECTED_ABIS:
            raise RuntimeError(f"ICP package ABI mismatch: {actual_abis!r}")
        abi.assert_compatible()
        if not _msa_has_direct_table():
            raise RuntimeError(
                "ICP needs an fmha_sm100 whose planner takes "
                f"{sorted(DIRECT_TABLE_PLAN_ARGS)} and whose kernel half "
                "derives the rank fragment bound from exact device lengths."
            )
        api = _msa_api()
        direct_version = getattr(api, "_FMHA_ICP_DIRECT_TABLE_ABI_VERSION", None)
        if direct_version != DIRECT_TABLE_ABI_VERSION:
            raise RuntimeError(
                f"fmha_sm100 advertises direct-table input ABI version "
                f"{direct_version}; this runtime emits version "
                f"{DIRECT_TABLE_ABI_VERSION}."
            )
        if api._FMHA_ICP_SCORE_ABI_VERSION != 2:
            raise RuntimeError(
                f"fmha_sm100 advertises ICP score ABI version "
                f"{api._FMHA_ICP_SCORE_ABI_VERSION}; refined-icp-v1 requires "
                "version 2, the out-of-band uint8 validity plane."
            )
        if not api._FMHA_HAS_ICP_VALIDITY_PLANE:
            raise RuntimeError(
                "fmha_sm100's kernel half does not carry ptr_ValidScore, so the "
                "ICP score wave's validity plane would come back unwritten."
            )

    def _check_prewarm(self) -> None:
        """Once per process: every ICP kernel is in the image's keyed cache."""
        global _prewarm_checked
        if _prewarm_checked:
            return
        mode = self._prewarm_check
        try:
            from . import prewarm
        except ImportError as exc:
            message = f"ICP prewarm verification is unavailable: {exc}"
            if mode == "fail":
                raise RuntimeError(message) from exc
            self.logger.warning(
                "ICP_PREWARM_PROBLEM rank=%d %s", self.icp_rank, message
            )
            _prewarm_checked = True
            return
        problems = prewarm.check()
        if problems and mode == "fail":
            raise RuntimeError(
                "ICP kernels are not fully prewarmed for this image; serving would "
                f"JIT-compile (prewarm_check='warn' to continue): {problems}"
            )
        for problem in problems:
            self.logger.warning(
                "ICP_PREWARM_PROBLEM rank=%d %s", self.icp_rank, problem
            )
        if not problems:
            self.logger.info("ICP_PREWARM_VERIFIED rank=%d", self.icp_rank)
        _prewarm_checked = True

    def _init_icp(self, device: torch.device) -> None:
        """Allocate retained capacity once; nothing here is sized from a step."""
        cfg, log = self.cfg, self.logger
        from .abi import QCAPACITY_ALIGNMENT
        from .candidates import (
            CAND_K,
            DecodePlan,
            PrefillPlan,
            SelectArm,
            allocate_workspace,
        )

        self.check_installed()

        physical_page_tokens = self.icp_physical_page_tokens
        if physical_page_tokens != 128:
            raise ValueError(
                "ICP supports P128 compound pages only; got "
                f"P{physical_page_tokens}. P256/R128 is not integrated."
            )
        self.icp_rows = physical_page_tokens // self.icp_c
        # Identity at P128: local score column `c` IS global B128 block `c`.
        self.icp_global_block_stride = 1
        self.icp_scan_block_begin = 0
        self.icp_heads_group = self.num_index_heads * self.icp_c
        self.icp_index_head_dim = self.index_head_dim

        # ---- CuTe decode scorer admission (strict at W=2) ----
        self.icp_cute_module = _msa_cute_decode_module()
        self.icp_cute_query_lens: set[int] = set()
        self.icp_cute_scorers: dict[int, Any] = {}
        self.icp_cute_factory_has_page_size = False
        self.icp_cute_factory_has_pdl = False
        self._icp_cute_dtypes_checked = False
        if self.icp_cute_module is None:
            self.icp_cute_status = (
                "unavailable: fmha_sm100.icp_decode_score is not installed "
                "(this image predates the MSA CuTe decode scorer)"
            )
        else:
            import inspect

            factory_params = inspect.signature(
                self.icp_cute_module.get_icp_decode_scorer
            ).parameters
            self.icp_cute_factory_has_page_size = "page_size" in factory_params
            self.icp_cute_factory_has_pdl = "use_pdl" in factory_params
            self.icp_cute_query_lens = {
                q
                for q in range(1, min(4, self.max_decode_query_len) + 1)
                if self.icp_cute_module.supports_icp_decode_score(
                    query_len=q,
                    world_size=self.icp_c,
                    page_size=physical_page_tokens,
                    num_heads=self.icp_heads_group,
                    head_dim=self.icp_index_head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
            }
            if self.icp_cute_query_lens:
                self.icp_cute_status = (
                    f"engaged: query_lens={sorted(self.icp_cute_query_lens)} "
                    f"split_k={ICP_CUTE_SPLIT_K}"
                )
            else:
                self.icp_cute_status = (
                    "unavailable: MSA refused every query_len "
                    f"1..{min(4, self.max_decode_query_len)} for "
                    f"world_size={self.icp_c} page_size={physical_page_tokens} "
                    f"num_heads={self.icp_heads_group} "
                    f"head_dim={self.icp_index_head_dim} "
                    f"arch=sm_{_device_arch_str(device)}"
                )
        _log = log.info if self.icp_cute_query_lens else log.warning
        _log(
            "%s rank=%d %s", ICP_CUTE_STATUS_MARKER, self.icp_rank, self.icp_cute_status
        )
        if not self.icp_cute_query_lens:
            raise RuntimeError(
                "MiniMax ICP CuTe decode scorer is required at ICP2 but is "
                f"not usable: {self.icp_cute_status}."
            )
        self.icp_max_blocks = _cdiv(cfg.max_model_len, physical_page_tokens)

        # ---- selector arms: full_row on both bands, fixed before capture ----
        if self.icp_max_blocks > ICP_FULL_ROW_MAX_COLUMNS:
            raise RuntimeError(
                f"the full_row ICP selectors admit up to {ICP_FULL_ROW_MAX_COLUMNS} "
                f"B128 columns, but this deployment's capacity is "
                f"{self.icp_max_blocks} (max_model_len={cfg.max_model_len}, "
                f"page={physical_page_tokens}); the package would silently "
                "dispatch to the bounded partitioned selector."
            )
        self.icp_decode_arm = SelectArm.RADIX_FULL_ROW
        self.icp_prefill_arm = SelectArm.RADIX_FULL_ROW
        log.info(
            "%s rank=%d arm=%s mode=%s capacity=%d columns",
            ICP_DECODE_SELECTOR_MARKER,
            self.icp_rank,
            self.icp_decode_arm.name,
            "full_row",
            self.icp_max_blocks,
        )
        self.icp_fused_decode = bool(cfg.fused_decode)
        log.info(
            "%s rank=%d arm=%s mode=%s capacity=%d columns %s",
            ICP_PREFILL_SELECTOR_MARKER,
            self.icp_rank,
            self.icp_prefill_arm.name,
            "full_row",
            self.icp_max_blocks,
            f"threads={ICP_PREFILL_FULL_ROW_THREADS} "
            f"cached_items={ICP_PREFILL_FULL_ROW_CACHED_ITEMS} "
            f"four_warp_finish=columns<={ICP_PREFILL_FOUR_WARP_FINISH_MAX_COLUMNS}",
        )

        # ---- exchange extents: the same calls the exchange factory makes ----
        capture_sizes = tuple(cfg.cudagraph_capture_sizes or ())
        max_num_batched_tokens = cfg.max_num_batched_tokens
        self.icp_exchange_capacities = exchange_token_capacities(
            max_token_capacity=max_num_batched_tokens,
            cudagraph_capture_sizes=capture_sizes,
        )
        self.icp_token_capacity = self.icp_exchange_capacities[-1]
        assert self.icp_token_capacity == (
            _cdiv(max_num_batched_tokens, QCAPACITY_ALIGNMENT) * QCAPACITY_ALIGNMENT
        )
        # Eager prefill gets its own geometric extent ladder; decode's is untouched.
        self.icp_prefill_exchange_capacities = prefill_exchange_token_capacities(
            max_token_capacity=max_num_batched_tokens,
        )
        assert self.icp_prefill_exchange_capacities[-1] == self.icp_token_capacity
        self.icp_admitted_exchange_extents = admitted_exchange_extents(
            max_token_capacity=max_num_batched_tokens,
            cudagraph_capture_sizes=capture_sizes,
        )
        assert set(self.icp_exchange_capacities) <= set(
            self.icp_admitted_exchange_extents
        )
        assert set(self.icp_prefill_exchange_capacities) <= set(
            self.icp_admitted_exchange_extents
        )

        # ---- prefill wave ladder, one rung per admitted column extent ----
        self.icp_prefill_budget_bytes = int(cfg.prefill_score_budget_bytes)
        if self.icp_prefill_budget_bytes < 1:
            raise ValueError("prefill_score_budget_bytes must be positive")
        self.icp_decode_pdl = bool(cfg.decode_pdl)
        self.icp_prefill_columns = icp_prefill_column_rungs(
            self.icp_max_blocks,
            floor=ICP_PREFILL_COLUMN_FLOOR,
            steps_per_octave=ICP_PREFILL_COLUMN_STEPS_PER_OCTAVE,
        )
        self.icp_prefill_waves = icp_prefill_wave_ladder(
            self.icp_prefill_columns,
            heads_group=self.icp_heads_group,
            token_capacity=self.icp_token_capacity,
            budget_bytes=self.icp_prefill_budget_bytes,
            max_wave=ICP_MAX_PREFILL_WAVE_TOKENS,
            query_tile_tokens=ICP_MSA_QUERY_TILE_TOKENS,
        )
        self.icp_prefill_max_wave = max(self.icp_prefill_waves.values())
        self.icp_prefill_min_wave = min(self.icp_prefill_waves.values())

        # ---- the decode outer chunk: a startup constant of every decode graph ----
        prefill_elems = icp_prefill_arena_elems(
            self.icp_prefill_waves, self.icp_heads_group
        )
        self.icp_chunk = icp_decode_chunk_tokens(
            max_model_len=cfg.max_model_len,
            heads_group=self.icp_heads_group,
            token_capacity=self.icp_token_capacity,
            physical_page_tokens=physical_page_tokens,
            budget_bytes=ICP_SCORE_SCRATCH_BUDGET_BYTES,
            max_chunk=ICP_MAX_OUTER_CHUNK_TOKENS,
            query_tile_tokens=ICP_MSA_QUERY_TILE_TOKENS,
        )
        split_sizes = sorted(
            size
            for size in capture_sizes
            if min(size, self.icp_token_capacity) > self.icp_chunk
        )
        if split_sizes:
            log.warning(
                "MiniMax-M3 ICP indexer (W=%d rank=%d): the %d-row decode chunk "
                "is smaller than captured decode sizes %s, which run several "
                "scorer + selector launches per sparse layer.",
                self.icp_c,
                self.icp_rank,
                self.icp_chunk,
                split_sizes,
            )

        # Rounded so the last chunk's candidate slice is whole for every stride.
        cap_padded = max(
            _cdiv(self.icp_token_capacity, width) * width
            for width in {self.icp_chunk, *self.icp_prefill_waves.values()}
        )

        # One device plan slot per chunk position either band can reach.
        self.icp_chunk_slots = max(
            _cdiv(self.icp_token_capacity, self.icp_chunk),
            _cdiv(self.icp_token_capacity, self.icp_prefill_min_wave),
        )
        self.icp_plan_slots = self.icp_chunk_slots
        pool_chunk = max(self.icp_chunk, self.icp_prefill_max_wave)
        # Unsplit direct_greedy plans, derived on device by the KV writer.
        self.icp_plan_store = _msa_api().IcpDevicePlanStore(
            device=device,
            num_slots=self.icp_plan_slots,
            max_segments=min(pool_chunk, self.max_num_reqs),
            num_heads=self.icp_heads_group,
            max_chunk_tokens=pool_chunk,
            max_kv_splits=1,
        )
        log.info(
            "%s rank=%d r13 prefill/mix: device_plan num_kv_splits=%d plan_slots=%d "
            "prefill_selector=%s mixed_steps=fmha_only rungs=geometric "
            "max_wave=%d exchange=native_pack+all_to_all+k2 | decode: "
            "decode chunk %d rows (fixed) cute_decode=strict debug_checks=%d "
            "decode_pdl=%d fused_decode=%d",
            ICP_PREFILL_PATH_MARKER,
            self.icp_rank,
            self.icp_plan_store.max_kv_splits,
            self.icp_plan_store.num_slots,
            "full_row",
            self.icp_prefill_max_wave,
            self.icp_chunk,
            int(cfg.debug_checks),
            int(self.icp_decode_pdl),
            int(self.icp_fused_decode),
        )
        self._check_prewarm()
        if self.icp_decode_pdl and ICP_PRODUCER_ENABLE_PDL:
            raise RuntimeError(
                "ICP decode PDL requires the fused producer to be a non-PDL "
                "launch: the CuTe scorer reads step metadata before its wait"
            )

        # ---- one score/validity arena; the decode view sits at offset 0 ----
        decode_plane = (self.icp_chunk, self.icp_heads_group, self.icp_max_blocks)
        decode_elems = self.icp_chunk * self.icp_heads_group * self.icp_max_blocks
        self.icp_score_arena, self.icp_valid_arena = _allocate_score_planes(
            elems=max(decode_elems, prefill_elems),
            device=device,
            budget_bytes=self.icp_prefill_budget_bytes,
            heads_group=self.icp_heads_group,
            columns=self.icp_max_blocks,
        )
        # CuTe ABI2 leaves tails undefined; selection reads exact live prefixes.
        self.icp_scores = self.icp_score_arena[:decode_elems].view(decode_plane)
        self.icp_valid = self.icp_valid_arena[:decode_elems].view(decode_plane)
        self.icp_candidates = torch.empty(
            (cap_padded, self.icp_heads_group, 16, 2),
            dtype=torch.float32,
            device=device,
        )
        self.icp_positions = torch.zeros(cap_padded, dtype=torch.int64, device=device)
        self.icp_active = torch.zeros(cap_padded, dtype=torch.bool, device=device)
        # Same bytes retyped once: the CuTe ABI takes uint8, the selector bool.
        self.icp_active_bytes = self.icp_active.view(torch.uint8)
        self.icp_nvalid = torch.zeros(cap_padded, dtype=torch.int32, device=device)
        self.icp_global_nvalid = torch.zeros(
            cap_padded, dtype=torch.int32, device=device
        )
        self.icp_forced_column = torch.full(
            (cap_padded,), -1, dtype=torch.int32, device=device
        )
        self.icp_forced = torch.full(
            (cap_padded,), -1, dtype=torch.int32, device=device
        )
        self.icp_n_ordinary = torch.zeros(cap_padded, dtype=torch.int32, device=device)

        # No page pool: the ordinary block table is read directly.
        assert self.icp_max_blocks == self.max_blocks_per_req
        # Slot-major causal offsets: row `slot` is at a startup-constant address.
        self.icp_reqs_per_slot = min(pool_chunk, self.max_num_reqs)
        self.icp_qo_offset = torch.zeros(
            (self.icp_chunk_slots, self.icp_reqs_per_slot),
            dtype=torch.int32,
            device=device,
        )
        # `(data_ptr, row_stride)` of the block table, latched on the first step.
        self._icp_table_identity: tuple[int, int] | None = None

        plan_kwargs = {
            "icp_degree": self.icp_c,
            "icp_rank": self.icp_rank,
            "num_heads_local": self.num_index_heads,
            "max_local_blocks": self.icp_max_blocks,
        }
        # One prefill plan per column rung; any prefix of its wave is admitted.
        self.icp_prefill_plans = {
            columns: PrefillPlan(
                token_capacity=wave,
                **{**plan_kwargs, "max_local_blocks": columns},
            )
            for columns, wave in self.icp_prefill_waves.items()
        }
        self.icp_prefill_plan = self.icp_prefill_plans[self.icp_max_blocks]
        # Decode selection is pinned to the graph bucket: one plan per extent.
        self.icp_select_extents = sorted(
            {
                min(self.icp_chunk, capacity - t0)
                for capacity in self.icp_exchange_capacities
                for t0 in range(0, capacity, self.icp_chunk)
            }
        )
        self.icp_decode_plans = {
            extent: DecodePlan(
                token_capacity=extent, use_pdl=self.icp_decode_pdl, **plan_kwargs
            )
            for extent in self.icp_select_extents
        }
        self.icp_decode_workspaces = {
            extent: allocate_workspace(plan, device)
            for extent, plan in self.icp_decode_plans.items()
        }
        # Retained prefill selector scratch, one arena carved per rung.
        prefill_partial_shapes = {
            columns: plan.partial_shape(
                self.icp_prefill_waves[columns], plan.scan_extent
            )
            for columns, plan in self.icp_prefill_plans.items()
        }
        self.icp_prefill_partials_arena = torch.empty(
            max(_numel(shape) for shape in prefill_partial_shapes.values()),
            dtype=torch.float32,
            device=device,
        )
        self.icp_prefill_views = {
            columns: (
                self.icp_score_arena[
                    : self.icp_prefill_waves[columns] * self.icp_heads_group * columns
                ].view(self.icp_prefill_waves[columns], self.icp_heads_group, columns),
                self.icp_valid_arena[
                    : self.icp_prefill_waves[columns] * self.icp_heads_group * columns
                ].view(self.icp_prefill_waves[columns], self.icp_heads_group, columns),
                self.icp_prefill_partials_arena[
                    : _numel(prefill_partial_shapes[columns])
                ].view(prefill_partial_shapes[columns]),
            )
            for columns in self.icp_prefill_columns
        }
        self.icp_prefill_partials = self.icp_prefill_views[self.icp_max_blocks][2]

        self._log_capacity(physical_page_tokens, QCAPACITY_ALIGNMENT, CAND_K)
        self._prewarm_cute_decode(device)

    def _log_capacity(self, physical_page_tokens, alignment, cand_k) -> None:
        log = self.logger
        log.info(
            "MiniMax-M3 ICP indexer retained capacity (W=%d rank=%d): decode "
            "chunk %d rows, prefill wave <= %d rows, %d plan slots (inner MSA "
            "tile pinned at %d), score+validity arena %.1f MiB, page directory "
            "NONE (direct-table input ABI v%d: "
            "the ordinary block table is consumed directly, %d columns per "
            "request), slot-major "
            "causal offsets %d x %d, exchange extents %s of a %d-row capacity, "
            "decode selector extents %s, retained prefill selector partials "
            "%.1f MiB. Every one of these is a startup constant; nothing "
            "here is sized from a step.",
            self.icp_c,
            self.icp_rank,
            self.icp_chunk,
            self.icp_prefill_max_wave,
            self.icp_plan_store.num_slots,
            ICP_MSA_QUERY_TILE_TOKENS,
            (self.icp_score_arena.numel() * 4 + self.icp_valid_arena.numel()) / 2**20,
            DIRECT_TABLE_ABI_VERSION,
            self.icp_max_blocks,
            self.icp_qo_offset.shape[0],
            self.icp_reqs_per_slot,
            self.icp_exchange_capacities,
            self.icp_token_capacity,
            self.icp_select_extents,
            self.icp_prefill_partials_arena.numel() * 4 / 2**20,
        )
        column_ratio = max(
            (hi / lo for lo, hi in itertools.pairwise(self.icp_prefill_columns)),
            default=1.0,
        )
        ladder, previous = [], None
        for columns in self.icp_prefill_columns:
            wave = self.icp_prefill_waves[columns]
            if wave != previous:
                ladder.append(
                    f"<={columns} cols -> wave {wave} "
                    f"({_cdiv(self.icp_token_capacity, wave)} calls/layer)"
                )
                previous = wave
        one_launch = (
            _cdiv(self.icp_token_capacity, ICP_MSA_QUERY_TILE_TOKENS)
            * ICP_MSA_QUERY_TILE_TOKENS
        )
        self.icp_prefill_one_launch_columns = max(
            (
                columns
                for columns in self.icp_prefill_columns
                if self.icp_prefill_waves[columns] >= one_launch
            ),
            default=0,
        )
        arena_bytes = self.icp_score_arena.numel() * 4 + self.icp_valid_arena.numel()
        log.info(
            "%s (W=%d rank=%d): %s, %d MiB score budget, %d admitted column "
            "rungs, %d bytes (%.4f GiB) of score+validity arena actually "
            "retained. Column ladder %s: floor %d columns == %d tokens, %d "
            "steps/octave, worst-case column over-run %.2fx. Resolved ladder "
            "at a %d-row invocation: %s. One "
            "OnlyScoreIcp call per sparse layer holds up to %d B128 columns == "
            "%d tokens of live history.",
            ICP_PREFILL_WAVE_MARKER,
            self.icp_c,
            self.icp_rank,
            "wave derived per invocation from the admitted live column extent",
            self.icp_prefill_budget_bytes // 2**20,
            len(self.icp_prefill_columns),
            arena_bytes,
            arena_bytes / 2**30,
            list(self.icp_prefill_columns),
            self.icp_prefill_columns[0],
            self.icp_prefill_columns[0] * physical_page_tokens,
            ICP_PREFILL_COLUMN_STEPS_PER_OCTAVE,
            column_ratio,
            self.icp_token_capacity,
            "; ".join(ladder),
            self.icp_prefill_one_launch_columns,
            self.icp_prefill_one_launch_columns * physical_page_tokens,
        )
        if self.icp_prefill_one_launch_columns < self.icp_max_blocks:
            log.warning(
                "%s (W=%d rank=%d): a %d-row invocation whose live history "
                "exceeds %d B128 columns cannot be scored in one wave at a %d "
                "MiB budget; the %d-column capacity rung degrades to %d waves "
                "per sparse layer. This is bounded and LOGGED, not silent. One "
                "launch at capacity would need %d MiB.",
                ICP_PREFILL_WAVE_MARKER,
                self.icp_c,
                self.icp_rank,
                self.icp_token_capacity,
                self.icp_prefill_one_launch_columns,
                self.icp_prefill_budget_bytes // 2**20,
                self.icp_max_blocks,
                _cdiv(self.icp_token_capacity, self.icp_prefill_min_wave),
                _cdiv(
                    one_launch * self.icp_heads_group * self.icp_max_blocks * 5,
                    2**20,
                ),
            )
        bytes_per_row = self.icp_heads_group * cand_k * 2 * 4
        prefill_extents = self.icp_prefill_exchange_capacities
        worst_ratio = max(
            (hi / (lo + 1) for lo, hi in itertools.pairwise(prefill_extents)),
            default=1.0,
        )
        decode_only = set(self.icp_exchange_capacities)
        log.info(
            "%s (W=%d rank=%d): %d admitted eager-prefill exchange extents, "
            "%d steps/octave, %d B alignment: %s. Carrier payload %d B per row "
            "per rank per sparse layer, so those extents span %d B .. %d B; "
            "the shared exchange retains one carrier workspace per extent of "
            "the UNION of both ladders (%d extents, %d rows, %.1f MiB of "
            "send+recv carriers) against %d extents / %d rows / %.1f MiB for "
            "the capture ladder alone. Worst-case extent over-run %.2fx. The "
            "DECODE ladder is unchanged and still %s; a decode invocation "
            "never selects out of the prefill list.",
            ICP_PREFILL_EXTENT_MARKER,
            self.icp_c,
            self.icp_rank,
            len(prefill_extents),
            ICP_PREFILL_EXTENT_STEPS_PER_OCTAVE,
            alignment,
            prefill_extents,
            bytes_per_row,
            prefill_extents[0] * bytes_per_row,
            prefill_extents[-1] * bytes_per_row,
            len(self.icp_admitted_exchange_extents),
            sum(self.icp_admitted_exchange_extents),
            2 * sum(self.icp_admitted_exchange_extents) * bytes_per_row / 2**20,
            len(decode_only),
            sum(decode_only),
            2 * sum(decode_only) * bytes_per_row / 2**20,
            worst_ratio,
            self.icp_exchange_capacities,
        )

    def _prewarm_cute_decode(self, device: torch.device) -> None:
        """Compile the CuTe scorer for every admitted query length at startup."""
        module = self.icp_cute_module
        start = time.monotonic()
        for query_len in sorted(self.icp_cute_query_lens):
            assert module is not None
            kwargs = {
                "query_len": query_len,
                "rank": self.icp_rank,
                "world_size": self.icp_c,
                "split_k": ICP_CUTE_SPLIT_K,
                "device": device,
            }
            if self.icp_cute_factory_has_page_size:
                kwargs["page_size"] = self.icp_physical_page_tokens
            if self.icp_cute_factory_has_pdl:
                kwargs["use_pdl"] = self.icp_decode_pdl
            self.icp_cute_scorers[query_len] = module.get_icp_decode_scorer(**kwargs)
        if self.icp_cute_scorers:
            self.logger.info(
                "%s rank=%d prewarmed %d CuTe scorers (query_lens=%s) in %.1f s",
                ICP_CUTE_STATUS_MARKER,
                self.icp_rank,
                len(self.icp_cute_scorers),
                sorted(self.icp_cute_scorers),
                time.monotonic() - start,
            )

    # -- public read-only geometry ----------------------------------------

    @property
    def token_capacity(self) -> int:
        return self.icp_token_capacity

    @property
    def admitted_exchange_extents(self) -> list[int]:
        return self.icp_admitted_exchange_extents

    @property
    def heads_group(self) -> int:
        return self.icp_heads_group

    @property
    def decode_chunk(self) -> int:
        return self.icp_chunk

    @property
    def max_blocks(self) -> int:
        return self.icp_max_blocks

    @property
    def plan_store(self) -> object:
        return self.icp_plan_store

    # -- per step ---------------------------------------------------------

    def plan_step(
        self,
        *,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu_upper_bound: torch.Tensor | None,
        is_prefilling: torch.Tensor | None,
        num_reqs: int,
        num_tokens: int,
    ) -> IcpStep:
        """Plan one invocation; reads no device value and allocates no device memory.

        Extents come from ``seq_lens_cpu_upper_bound`` and startup constants;
        causality and live bounds come from the exact device ``seq_lens``.
        """
        from .candidates import (
            CandidateGeometry,
            live_blocks,
            select_decode_candidates,
            select_prefill_candidates,
        )

        api = _msa_api()
        _fmha_sm100_plan = api._fmha_sm100_plan

        c, rank, rows = self.icp_c, self.icp_rank, self.icp_rows
        assert num_tokens <= self.icp_token_capacity

        qsl_cpu = query_start_loc_cpu[: num_reqs + 1].to(torch.int64)
        q_begin, q_end = qsl_cpu[:-1], qsl_cpu[1:]
        # Live count from host offsets; padded rows stay inactive.
        live_tokens = int(q_end[-1]) if num_reqs > 0 else 0
        assert 0 <= live_tokens <= num_tokens

        # The phase from actual prefill state, identical on every rank.
        if is_prefilling is None:
            raise RuntimeError(
                "refined-icp-v1 needs is_prefilling to route the candidate "
                "exchange: the phase decides the transport and must be "
                "identical on every rank."
            )
        has_prefill = bool(is_prefilling[:num_reqs].any().item())

        cap = select_token_capacity(
            self.icp_prefill_exchange_capacities
            if has_prefill
            else self.icp_exchange_capacities,
            num_tokens,
        )

        decode_query_len = _uniform_decode_query_len(qsl_cpu, has_prefill)
        if seq_lens_cpu_upper_bound is None:
            raise RuntimeError(
                "refined-icp-v1 sizes the scorer's work decomposition from "
                "seq_lens_cpu_upper_bound; a device->host copy of seq_lens is "
                "forbidden on the submission path."
            )
        kv_ub = seq_lens_cpu_upper_bound[:num_reqs].to(torch.int64)
        use_cute_decode = decode_query_len in self.icp_cute_query_lens
        cute_rows = icp_cute_rows(
            use_cute_decode=use_cute_decode,
            has_prefill=has_prefill,
            num_tokens=num_tokens,
        )
        use_cute_decode = cute_rows > 0
        if use_cute_decode and not self._icp_cute_dtypes_checked:
            for _name, _tensor, _want in (
                ("query_start_loc", query_start_loc, torch.int32),
                ("seq_lens", seq_lens, torch.int32),
            ):
                if _tensor.dtype != _want:
                    raise RuntimeError(
                        f"MiniMax ICP CuTe decode needs {_name} as {_want}; "
                        f"the caller supplied {_tensor.dtype}."
                    )
            self._icp_cute_dtypes_checked = True

        forced = self.icp_forced[:cap]
        n_ordinary = self.icp_n_ordinary[:cap]

        # The ordinary block table, read directly: offset 0, unit columns, wide rows.
        assert block_table.dim() == 2, block_table.shape
        assert block_table.storage_offset() == 0, (
            "ICP: the block table must be passed at storage offset 0, got "
            f"{block_table.storage_offset()}; a captured graph would replay "
            "against another step's first request."
        )
        table_row_stride = block_table.stride(0)
        assert block_table.stride(1) == 1, (
            f"ICP: the block table's column stride must be 1; got {block_table.stride(1)}."
        )
        assert table_row_stride >= self.icp_max_blocks, (
            f"ICP: the block table's row pitch {table_row_stride} is narrower "
            f"than the {self.icp_max_blocks} pages a request can name."
        )
        table_identity = (block_table.data_ptr(), table_row_stride)
        if self._icp_table_identity is None:
            self._icp_table_identity = table_identity
        elif self._icp_table_identity != table_identity:
            raise RuntimeError(
                "ICP: the block table moved between steps "
                f"({self._icp_table_identity} -> {table_identity}); every "
                "captured graph would be stale."
            )

        # Prefill takes the rung covering the host-bounded live extent; decode the capacity plane.
        if has_prefill:
            invocation_live_blocks = min(
                live_blocks(
                    max(1, int(kv_ub.max()) if num_reqs else 1), page_size=PAGE_SIZE
                ),
                self.icp_max_blocks,
            )
            score_columns = icp_prefill_rung(
                self.icp_prefill_columns, invocation_live_blocks
            )
            width = self.icp_prefill_waves[score_columns]
            score_plane, valid_plane, prefill_partials = self.icp_prefill_views[
                score_columns
            ]
            prefill_plan = self.icp_prefill_plans[score_columns]
        else:
            score_columns = self.icp_max_blocks
            width = self.icp_chunk
            score_plane, valid_plane = self.icp_scores, self.icp_valid
            prefill_partials = self.icp_prefill_partials
            prefill_plan = self.icp_prefill_plan
        prefill_controls = prefill_full_row_controls(
            score_columns,
            threads=ICP_PREFILL_FULL_ROW_THREADS,
            cached_items=ICP_PREFILL_FULL_ROW_CACHED_ITEMS,
            four_warp_finish_max_columns=ICP_PREFILL_FOUR_WARP_FINISH_MAX_COLUMNS,
        )

        fmha = api._fmha_sm100
        chunks: list[IcpChunk] = []
        for window in icp_row_windows(
            num_tokens=num_tokens,
            cap=cap,
            cute_rows=cute_rows,
            decode_chunk=self.icp_chunk,
            prefill_width=width,
            has_prefill=has_prefill,
        ):
            t0, begin, t1 = window.grid_begin, window.row_begin, window.row_end
            extent = window.select_rows
            geometry = CandidateGeometry(
                local_valid_blocks=self.icp_nvalid[begin : begin + extent],
                forced_column=self.icp_forced_column[begin : begin + extent],
                active_rows=self.icp_active[begin : begin + extent],
                scan_block_begin=self.icp_scan_block_begin,
                global_block_stride=self.icp_global_block_stride,
            )
            out = self.icp_candidates[begin : begin + extent]
            if window.cute:
                launch_window = icp_cute_decode_window(
                    query_len=decode_query_len,
                    token_begin=t0,
                    rows=cute_rows - t0,
                    chunk=self.icp_chunk,
                    num_reqs=num_reqs,
                )
                if launch_window is None:
                    # Capture padding only: stays invalid candidates.
                    continue
                window_rows, request_begin = launch_window[1], launch_window[2]
                scorer = self.icp_cute_scorers[decode_query_len].bind(
                    token_begin=launch_window[0],
                    token_count=window_rows,
                    request_begin=request_begin,
                    request_count=launch_window[3],
                )
                table = block_table
                cute_scores = self.icp_scores[:window_rows]
                cute_valid = self.icp_valid[:window_rows]
                positions = self.icp_positions[:num_tokens]
                active_rows = self.icp_active_bytes[:num_tokens]
                fused_select = None
                if window.decode_selector:
                    select = functools.partial(
                        select_decode_candidates,
                        self.icp_scores[:extent],
                        geometry,
                        plan=self.icp_decode_plans[extent],
                        workspace=self.icp_decode_workspaces[extent],
                        out=out,
                        arm=self.icp_decode_arm,
                    )
                    if self.icp_fused_decode:
                        fused_select = functools.partial(
                            _icp_fused_publish,
                            self.icp_scores[:extent],
                            geometry,
                            begin,
                        )
                chunks.append(
                    IcpChunk(
                        plan=None,
                        token_begin=t0,
                        num_live_tokens=t1 - t0,
                        block_table=block_table,
                        block_table_row_stride=table_row_stride,
                        block_table_row_begin=request_begin,
                        qo_offset=self.icp_qo_offset[0, :0],
                        live_blocks=self.icp_max_blocks,
                        score_extent=self.icp_max_blocks,
                        select_extent=extent,
                        topk_num_valid_pages=geometry.local_valid_blocks,
                        icp_forced_column_v1=geometry.forced_column,
                        icp_active_rows=geometry.active_rows,
                        icp_scan_block_begin_v1=self.icp_scan_block_begin,
                        icp_global_block_stride_v1=self.icp_global_block_stride,
                        cute_scorer=scorer,
                        rank_aware_nvalid=True,
                        geometry=geometry,
                        launch_score=bind_cute_launch(
                            scorer,
                            table,
                            query_start_loc,
                            seq_lens,
                            positions,
                            active_rows,
                            cute_scores,
                            cute_valid,
                        ),
                        select=select,
                        fused_select=fused_select,
                        cute_check_args=(
                            scorer,
                            table,
                            query_start_loc,
                            seq_lens,
                            positions,
                            active_rows,
                            cute_scores,
                            cute_valid,
                        ),
                    )
                )
                continue
            # FMHA OnlyScoreIcp window (r13), unsplit.
            lo = q_begin.clamp(t0, t1)
            hi = q_end.clamp(t0, t1)
            sel = (hi - lo > 0).nonzero(as_tuple=True)[0]
            n_sel = int(sel.numel())
            if n_sel == 0:
                continue
            slot_index = window.slot
            first, last = int(sel[0]), int(sel[-1])
            assert last - first + 1 == n_sel
            if n_sel > self.icp_reqs_per_slot:
                raise RuntimeError(
                    f"ICP: chunk {slot_index} covers {n_sel} requests but a "
                    f"slot row holds {self.icp_reqs_per_slot}."
                )
            qo_offset = self.icp_qo_offset[slot_index, :n_sel]
            kv_sel = kv_ub[sel]
            qo_offset_bound = kv_sel - q_end[sel] + lo[sel]
            last_position_bound = int((kv_sel - q_end[sel] + hi[sel] - 1).max())
            chunk_live_blocks = min(
                live_blocks(last_position_bound + 1, page_size=PAGE_SIZE),
                self.icp_max_blocks,
            )
            plan = _fmha_sm100_plan(
                (hi[sel] - lo[sel]).to(torch.int32),
                kv_sel.to(torch.int32),  # global kv lengths; the mask needs them
                self.icp_heads_group,
                num_kv_heads=1,
                qo_offset=qo_offset_bound.to(torch.int32),
                page_size=rows,  # R: one page IS one rank fragment (ABI G13)
                output_maxscore=True,
                causal=True,
                num_kv_splits=1,
                icp_c=c,
                icp_rank=rank,
                icp_block_table=block_table,
                icp_block_table_row_stride=table_row_stride,
                icp_block_table_row_begin=first,
                device_plan=self.icp_plan_store.slot(slot_index, chunk_width=width),
            )
            score_extent = chunk_live_blocks if has_prefill else self.icp_max_blocks
            assert 1 <= score_extent <= self.icp_max_blocks
            assert score_extent <= score_columns, (
                f"chunk {slot_index} advertises {score_extent} score columns "
                f"but the invocation's plane is {score_columns} wide; the "
                "wave rung must cover every chunk's live extent."
            )
            plan["max_k_tiles"] = score_extent
            live = min(t1, live_tokens) - t0
            fused_select = None
            if window.decode_selector:
                select = functools.partial(
                    select_decode_candidates,
                    score_plane[:extent],
                    geometry,
                    plan=self.icp_decode_plans[extent],
                    workspace=self.icp_decode_workspaces[extent],
                    out=out,
                    arm=self.icp_decode_arm,
                )
                if self.icp_fused_decode:
                    fused_select = functools.partial(
                        _icp_fused_publish,
                        score_plane[:extent],
                        geometry,
                        begin,
                    )
            else:
                select = functools.partial(
                    select_prefill_candidates,
                    score_plane[:extent],
                    geometry,
                    plan=prefill_plan,
                    live_blocks=chunk_live_blocks,
                    out=out,
                    partials=prefill_partials[:extent],
                    arm=self.icp_prefill_arm,
                    **prefill_selector_kwargs(prefill_controls),
                )
            chunks.append(
                IcpChunk(
                    plan=plan,
                    token_begin=t0,
                    num_live_tokens=live,
                    block_table=block_table,
                    block_table_row_stride=table_row_stride,
                    block_table_row_begin=first,
                    qo_offset=qo_offset,
                    live_blocks=chunk_live_blocks,
                    score_extent=score_extent,
                    select_extent=extent,
                    topk_num_valid_pages=geometry.local_valid_blocks,
                    icp_forced_column_v1=geometry.forced_column,
                    icp_active_rows=geometry.active_rows,
                    geometry=geometry,
                    launch_score=bind_fmha_launch(
                        fmha,
                        plan,
                        t0,
                        live,
                        kv_indices=block_table,
                        max_score=score_plane[:live, :, :score_extent],
                        valid_score=valid_plane[:live, :, :score_extent],
                        q_offset_override=qo_offset,
                        icp_c=c,
                        icp_rank=rank,
                    ),
                    select=select,
                    fused_select=fused_select,
                )
            )

        fused_ready = False
        fused_padding: list = []
        if self.icp_fused_decode and not has_prefill:
            fused_ready = all(chunk.fused_select is not None for chunk in chunks)
            covered = sorted(
                (chunk.token_begin, chunk.token_begin + chunk.select_extent)
                for chunk in chunks
            )
            gaps: list[tuple[int, int]] = []
            at = 0
            for cover_begin, cover_end in covered:
                if cover_begin > at:
                    gaps.append((at, cover_begin))
                at = max(at, cover_end)
            if at < cap:
                gaps.append((at, cap))
            for gap_begin, gap_end in gaps:
                for p0 in range(gap_begin, gap_end, self.icp_chunk):
                    pad_rows = min(self.icp_chunk, gap_end - p0)
                    # Unselected rows are inactive; they publish (-inf, -1) records.
                    pad_geometry = CandidateGeometry(
                        local_valid_blocks=self.icp_nvalid[p0 : p0 + pad_rows],
                        forced_column=self.icp_forced_column[p0 : p0 + pad_rows],
                        active_rows=self.icp_active[p0 : p0 + pad_rows],
                        scan_block_begin=self.icp_scan_block_begin,
                        global_block_stride=self.icp_global_block_stride,
                    )
                    fused_padding.append(
                        functools.partial(
                            _icp_fused_publish,
                            self.icp_scores[:pad_rows],
                            pad_geometry,
                            p0,
                        )
                    )

        live_views = IcpLiveViews(
            positions=self.icp_positions[:cap],
            active=self.icp_active[:cap],
            local_nvalid=self.icp_nvalid[:cap],
            local_forced=self.icp_forced_column[:cap],
            global_nvalid=self.icp_global_nvalid[:cap],
            forced=forced,
            n_ordinary=n_ordinary,
            candidates=self.icp_candidates[:cap],
            qo_offsets=self.icp_qo_offset[: _cdiv(cap, width)],
            chunk_width=width,
            device_plan=(
                self.icp_plan_store
                if any(chunk.plan is not None for chunk in chunks)
                else None
            ),
        )
        return IcpStep(
            chunks=chunks,
            scores=score_plane,
            valid=valid_plane,
            candidates=self.icp_candidates,
            forced=forced,
            n_ordinary=n_ordinary,
            prefill_plan=prefill_plan,
            prefill_partials=prefill_partials,
            decode_plans=self.icp_decode_plans,
            decode_workspaces=self.icp_decode_workspaces,
            has_prefill=has_prefill,
            token_capacity=cap,
            chunk_width=width,
            decode_arm=self.icp_decode_arm,
            prefill_arm=self.icp_prefill_arm,
            prefill_controls=prefill_controls,
            num_tokens=num_tokens,
            live=live_views,
            step_n_valid=self.icp_global_nvalid[:num_tokens],
            topk_num_valid_pages=self.icp_nvalid[:num_tokens],
            forced_column=self.icp_forced_column[:num_tokens],
            active_rows_bool=self.icp_active[:num_tokens],
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            positions=self.icp_positions[:num_tokens],
            active_rows=self.icp_active_bytes[:num_tokens],
            cute_rows=cute_rows,
            # `view`, not a conversion: both carrier words are bitcast (ABI C1/C2).
            exchange_candidates=self.icp_candidates[:cap].view(torch.int32),
            fused_ready=fused_ready,
            fused_padding=tuple(fused_padding),
            decode_chunk=self.icp_chunk,
            fused_decode=self.icp_fused_decode,
        )

    # -- per layer --------------------------------------------------------

    def _log_decode_profile(self, exchange: object, chunk: int, d3: bool) -> None:
        """Once per process: the decode profile of the bound exchange."""
        global _decode_profile_logged
        if _decode_profile_logged:
            return
        profile = getattr(exchange, "decode_profile", None)
        if profile is None:
            raise RuntimeError(
                "ICP: the bound candidate exchange has no decode_profile"
            )
        if profile["d3"] != int(d3):
            raise RuntimeError(
                f"ICP: the bound exchange has d3={profile['d3']} but the indexer "
                f"runtime was built with d3={int(d3)}; configure fused_decode "
                "identically for every process."
            )
        self.logger.info(
            "%s rank=%d split_k=%d chunk=%d exchange=%s merge=%s d3=%d pdl=%d",
            ICP_DECODE_PROFILE_MARKER,
            self.icp_rank,
            ICP_CUTE_SPLIT_K,
            chunk,
            profile["exchange"],
            profile["merge"],
            profile["d3"],
            profile["pdl"],
        )
        _decode_profile_logged = True

    def _note_cute_engagement(self, chunk: IcpChunk) -> None:
        seen = _ICP_CUTE_ENGAGEMENTS.get(self.icp_rank, 0)
        _ICP_CUTE_ENGAGEMENTS[self.icp_rank] = seen + 1
        if seen == 0:
            self.logger.info(
                "%s rank=%d token_begin=%d num_live_tokens=%d score_extent=%d scorer=%r",
                ICP_CUTE_ENGAGED_MARKER,
                self.icp_rank,
                chunk.token_begin,
                chunk.num_live_tokens,
                chunk.score_extent,
                chunk.cute_scorer,
            )

    def _check_invocation(self, index_query, step: IcpStep, buf) -> None:
        assert buf is not None
        assert step.token_capacity <= buf.shape[0], (
            f"ICP: the shared top-k buffer holds {buf.shape[0]} rows but this "
            f"invocation's exchange extent is {step.token_capacity}."
        )
        heads_group = step.candidates.shape[1]
        head_dim = self.index_head_dim
        if index_query.shape[-1] != heads_group * head_dim:
            raise RuntimeError(
                f"ICP scores ALL {heads_group} of the group's index-query heads "
                f"on every rank and routes head h to rank h // "
                f"{self.num_index_heads} (ABI G5/C9), but this rank's index_query "
                f"carries {index_query.shape[-1] // head_dim} head(s); the index-Q "
                "projection must replicate the group across the W ICP ranks."
            )

    def _check_chunk(self, chunk: IcpChunk, index_q, index_kv) -> None:
        if chunk.launch_score is None or chunk.select is None:
            raise RuntimeError(
                "ICP: a chunk reached the forward without its bound launches."
            )
        if chunk.cute_scorer is None:
            return
        assert chunk.rank_aware_nvalid, (
            "CuTe decode chunk carries a non-rank-aware topk_num_valid_pages; "
            "the no-init scorer leaves tails undefined."
        )
        scorer, *tensors = chunk.cute_check_args
        scorer.validate(index_q, index_kv, *tensors)

    def run_layer(
        self,
        step: IcpStep,
        index_query: torch.Tensor,
        index_kv: torch.Tensor,
        *,
        layer_idx: int,
        exchange: Any,
        topk_buffer: torch.Tensor,
        scale: float,
    ) -> None:
        """Score this rank's fragment, select locally, exchange exactly once.

        Validators run on the first sparse layer of a step (every layer with
        ``debug_checks``); later layers only launch the pre-bound closures.
        """
        if exchange is None:
            raise RuntimeError(
                f"ICP: at W={self.icp_c} local candidates cover only this rank's "
                "index ownership; bind the candidate exchange before serving."
            )
        buf = topk_buffer
        assert buf is not None
        debug = self.cfg.debug_checks
        if debug or not step.checked:
            self._check_invocation(index_query, step, buf)
            step.checked = True

        heads_group = step.candidates.shape[1]
        index_q = index_query[: step.num_tokens].view(
            -1, heads_group, self.index_head_dim
        )
        # `unsqueeze`, not `view`: the index region carries the compound page stride.
        k_pages = index_kv.unsqueeze(1)

        self._log_decode_profile(exchange, step.decode_chunk, step.fused_decode)
        fused = not step.has_prefill and getattr(exchange, "fused_decode", False)
        if fused and not step.fused_ready:
            raise RuntimeError(
                "ICP: the D3 exchange is bound but this decode step has a window "
                "without a fused selector binding; configure fused_decode "
                "identically on every rank."
            )
        for chunk in step.chunks:
            validate = debug or not chunk.checked
            if validate:
                self._check_chunk(chunk, index_q, index_kv)
                chunk.checked = True
            if chunk.cute_scorer is not None:
                self._note_cute_engagement(chunk)
            assert chunk.launch_score is not None and chunk.select is not None
            chunk.launch_score(index_q, index_kv, k_pages, scale)
            if fused:
                assert chunk.fused_select is not None
                chunk.fused_select(exchange, layer_idx)
            else:
                chunk.select(validate=validate)
        if fused:
            for publish in step.fused_padding:
                publish(exchange, layer_idx)

        # ONE exchange per layer, after every window; views bound once per step.
        if step.exchange_buf is not buf:
            step.exchange_buf = buf
            step.exchange_out = buf[: step.token_capacity]
        exchange(
            step.exchange_candidates,
            layer_idx=layer_idx,
            has_prefill=step.has_prefill,
            forced=step.forced,
            n_ordinary=step.n_ordinary,
            out=step.exchange_out,
        )

    def shutdown(self) -> None:
        """Release retained device buffers; the exchange drain is the exchange's ``close``."""
        if self._closed:
            return
        self._closed = True
        for name in (
            "icp_cute_scorers",
            "icp_decode_workspaces",
            "icp_decode_plans",
            "icp_prefill_plans",
            "icp_prefill_views",
        ):
            setattr(self, name, {})
        for name in (
            "icp_plan_store",
            "icp_score_arena",
            "icp_valid_arena",
            "icp_scores",
            "icp_valid",
            "icp_candidates",
            "icp_positions",
            "icp_active",
            "icp_active_bytes",
            "icp_nvalid",
            "icp_global_nvalid",
            "icp_forced_column",
            "icp_forced",
            "icp_n_ordinary",
            "icp_qo_offset",
            "icp_prefill_partials_arena",
            "icp_prefill_partials",
            "icp_prefill_plan",
        ):
            setattr(self, name, None)
