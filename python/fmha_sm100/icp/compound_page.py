# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compound KV page layout for MiniMax-M3 indexer context parallelism.

One physical page of ``P`` tokens holds this rank's main K/V and its compact
index-key rows under one allocation, block-table entry, write slot and lifetime.
Ranking always uses logical ``B=128`` blocks. The integrated profile is
``P128/R64`` at TP2 with plain NVFP4 main K/V.
``P256/R128`` is not integrated in this path and is gated off at construction:
it needs MSA subpage readers and a selector route that emits the
non-contiguous ranking ids ``2*column + rank``.

With ``W`` equal to TP and indexer CP, each rank stores ``R=P//W`` index rows,
``Hlocal=4//W`` main KV heads, and head dimension 128. Physical-page token
``t`` belongs to index rank ``t//R`` at local row ``t%R``. The compact local
index view starts at row zero on every rank; rank never offsets its base.
At P128 every rank holds a fragment of every logical block, so its local score
column ``c`` maps to global logical block ``c`` -- the identity.

Page interior, main region first at the layer's shared-storage offset::

    [ main K/V .................. | index keys ....... ]
    0                    main_span   index_offset   compound_span

The main region reproduces the layout the main attention backend already
expects.  For NVFP4 that is the byte-budget carrier
``[2*Hlocal, P, nvfp4_kv_cache_full_dim(D)]`` whose physical content is per-head
``[K_data | K_scale | V_data | V_scale]`` slots, matching public vLLM;
for FP8 E4M3 it is
``[Hlocal, P, 2*D]``.  The index region is ``[R, D]`` of scale-free FP8 E4M3.

Pure layout math shared by the serving framework's cache binding and the
native writer/readers; the framework owns the tensors and their binding.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

# Frozen ABI marker.  Bump with any incompatible layout change; the value is
# recorded alongside GPU evidence so a result can be tied to a layout.
COMPOUND_PAGE_ABI_VERSION = "refined-icp-v1.compound-page.2"

LOGICAL_BLOCK_TOKENS = 128
INDEX_HEAD_DIM = 128
MAIN_HEAD_DIM = 128
TOTAL_MAIN_KV_HEADS = 4


def nvfp4_kv_cache_full_dim(head_size: int) -> int:
    """Packed last dim for NVFP4 KV cache: fp4 data + fp8 block scales."""
    return head_size // 2 + head_size // 16


class SparseMainFormat(str, enum.Enum):
    """Main K/V storage format of one sparse layer, resolved once at startup."""

    NVFP4 = "nvfp4"
    FP8_E4M3 = "fp8_e4m3"


class CompoundPageError(RuntimeError):
    """Raised when a compound page or its binding violates the layout contract.

    Deliberately not ``AssertionError``: these checks must survive ``python -O``
    and must not be confused with a test assertion.
    """


def icp_local_read_bounds(
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    physical_page_tokens: int,
    world_size: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive readable score prefixes and local forced columns on device.

    ``nvalid`` is the EXACT number of score columns the ABI2 decode scorer
    writes for each row, not an upper bound on it. With
    ``U = max(0, min(seq_lens, positions + 1))`` the scorer's per-row store
    guard admits column ``b`` iff ``P*b + R*rank < U``, which counts to
    ``U//P + (U%P > rank*R)`` -- the expression below, term for term. The
    scorer leaves every other cell of the score plane undefined, so this must
    stay equal to that guard; a bound that merely dominates it would let the
    selector rank uninitialised memory.

    Args:
        positions: int64 absolute token positions, on device.
        seq_lens: int32 per-token materialised history length, on device.
        physical_page_tokens: ``P``; 128 is the only integrated value.
        world_size: ``W``, the indexer context-parallel degree.
        rank: this rank's index in ``[0, W)``.

    Returns:
        ``(nvalid, forced_column)``, both int32 and on ``positions``' device.

    Raises:
        CompoundPageError: if ``physical_page_tokens`` is not 128.

    """
    if physical_page_tokens != LOGICAL_BLOCK_TOKENS:
        raise CompoundPageError(
            f"icp_local_read_bounds is P{LOGICAL_BLOCK_TOKENS} only; got "
            f"P{physical_page_tokens}. At P256 a local column would denote "
            "global ranking id `2*column + rank`, which the selector cannot "
            "emit -- it produces a contiguous prefix."
        )
    if world_size != 2 or rank not in (0, 1):
        raise CompoundPageError("ICP read bounds require TP2 and rank 0 or 1")
    rows = physical_page_tokens // world_size
    visible = torch.minimum(seq_lens, positions + 1).clamp_min(0)
    nvalid = visible // physical_page_tokens + (
        visible % physical_page_tokens > rank * rows
    )
    # P128 identity: local score column `c` IS global ranking block `c`, so
    # the forced column is just the position's own block when this rank holds
    # a readable fragment of it.
    column = positions // LOGICAL_BLOCK_TOKENS
    owns = (column >= 0) & (column < nvalid)
    forced = torch.where(owns, column, -1)
    return nvalid.to(torch.int32), forced.to(torch.int32)


@dataclass(frozen=True, slots=True)
class CompoundPageLayout:
    """Immutable, rank-invariant description of one sparse layer's page.

    Rank-invariant on purpose: every rank of a TP group has a byte-identical
    layout and differs only in *which* index rows and main heads it holds.  The
    worker rank lives in the binding (:class:`BoundCompoundCache`), never here,
    so this object can be compared, hashed and logged across ranks.
    """

    abi_version: str
    physical_page_tokens: int
    tp_size: int
    main_format: SparseMainFormat
    main_num_kv_heads: int
    main_head_dim: int
    index_rows_per_rank: int
    index_head_dim: int

    @classmethod
    def build(
        cls,
        *,
        tp_size: int,
        main_format: SparseMainFormat,
        physical_page_tokens: int = LOGICAL_BLOCK_TOKENS,
        ranking_block_tokens: int = LOGICAL_BLOCK_TOKENS,
        total_main_kv_heads: int = TOTAL_MAIN_KV_HEADS,
        main_head_dim: int = MAIN_HEAD_DIM,
        index_head_dim: int = INDEX_HEAD_DIM,
    ) -> CompoundPageLayout:
        if ranking_block_tokens != LOGICAL_BLOCK_TOKENS:
            raise CompoundPageError("ranking blocks must remain 128 tokens")
        # P128 is asserted by the layout itself, not only by its caller:
        # `nvidia/model.py` refuses a non-128 `--block-size` earlier, but this
        # constructor is public and a layout that exists at P256 is a P256
        # configuration.
        if physical_page_tokens != LOGICAL_BLOCK_TOKENS:
            raise CompoundPageError(
                f"compound pages are P{LOGICAL_BLOCK_TOKENS} only; got "
                f"P{physical_page_tokens}. P256/R128 is not integrated: it "
                "needs MSA subpage readers and a selector able to emit the "
                "non-contiguous ranking ids `2*column + rank`."
            )
        if tp_size != 2:
            raise CompoundPageError(f"compound pages require TP2, got TP{tp_size}")
        if SparseMainFormat(main_format) is not SparseMainFormat.NVFP4:
            raise CompoundPageError("compound pages require plain NVFP4 main K/V")
        if (
            total_main_kv_heads != TOTAL_MAIN_KV_HEADS
            or main_head_dim != MAIN_HEAD_DIM
            or index_head_dim != INDEX_HEAD_DIM
        ):
            raise CompoundPageError("compound pages require global KV4 and D128")
        if physical_page_tokens % tp_size:
            raise CompoundPageError(
                f"index rows do not partition: block tokens "
                f"{physical_page_tokens} % W {tp_size} != 0"
            )
        if total_main_kv_heads % tp_size:
            raise CompoundPageError(
                f"main KV heads do not shard without replication: "
                f"{total_main_kv_heads} % W {tp_size} != 0"
            )
        return cls(
            abi_version=COMPOUND_PAGE_ABI_VERSION,
            physical_page_tokens=physical_page_tokens,
            tp_size=tp_size,
            main_format=SparseMainFormat(main_format),
            main_num_kv_heads=total_main_kv_heads // tp_size,
            main_head_dim=main_head_dim,
            index_rows_per_rank=physical_page_tokens // tp_size,
            index_head_dim=index_head_dim,
        )

    # -- derived geometry -------------------------------------------------

    @property
    def ranking_block_tokens(self) -> int:
        return LOGICAL_BLOCK_TOKENS

    @property
    def main_element_dim(self) -> int:
        """Trailing bytes per (head, token) of the main region.

        NVFP4 packs fp4 data plus its inline fp8 block scales into
        ``nvfp4_kv_cache_full_dim(D)`` bytes per side; FP8 E4M3 is one byte per
        element and carries **no** scale payload for this model.
        """
        if self.main_format is SparseMainFormat.NVFP4:
            return nvfp4_kv_cache_full_dim(self.main_head_dim)
        return self.main_head_dim

    @property
    def main_shape(self) -> tuple[int, int, int]:
        """Per-page main shape, matching the backend's existing carrier."""
        if self.main_format is SparseMainFormat.NVFP4:
            # Slot 2*h is head h's K; slot 2*h+1 is its V.
            return (
                2 * self.main_num_kv_heads,
                self.physical_page_tokens,
                self.main_element_dim,
            )
        # K and V packed into the content dim.
        return (
            self.main_num_kv_heads,
            self.physical_page_tokens,
            2 * self.main_element_dim,
        )

    @property
    def main_span_bytes(self) -> int:
        span = 1
        for dim in self.main_shape:
            span *= dim
        return span

    @property
    def index_shape(self) -> tuple[int, int]:
        return (self.index_rows_per_rank, self.index_head_dim)

    @property
    def index_span_bytes(self) -> int:
        # Scale-free FP8 E4M3: one byte per element, no scale payload.
        return self.index_rows_per_rank * self.index_head_dim

    @property
    def main_offset_bytes(self) -> int:
        return 0

    @property
    def index_offset_bytes(self) -> int:
        # The index region starts where main ends and ``compound_span_bytes``
        # is their sum, so the regions are disjoint and page-covering by
        # construction. The load-bearing guards are the page-stride and
        # allocation-size checks in ``bind_compound_cache``.
        return self.main_span_bytes

    @property
    def compound_span_bytes(self) -> int:
        return self.main_span_bytes + self.index_span_bytes

    def describe(self) -> str:
        return (
            f"{self.abi_version} W={self.tp_size} "
            f"{self.main_format.value} main={self.main_span_bytes}B "
            f"index={self.index_span_bytes}B(R={self.index_rows_per_rank}) "
            f"page={self.compound_span_bytes}B"
        )
