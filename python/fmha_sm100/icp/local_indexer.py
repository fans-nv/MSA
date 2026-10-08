# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host planning and launch bindings for the rank-local ICP indexer.

The caller supplies resolved sizing and selector policies. Importing this
module needs neither torch nor the scorers; binding a launch only captures
the supplied callable and views, and never builds or launches a kernel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _icp_wave_from_budget(
    *,
    columns: int,
    heads_group: int,
    token_capacity: int,
    budget_bytes: int,
    max_wave: int,
    query_tile_tokens: int,
) -> int:
    """Query rows that fit ``budget_bytes`` of ``[wave, H_group, columns]``.

    The fp32 score plane and the uint8 validity plane cost
    ``heads_group * columns * (4 + 1)`` bytes per query row, and the wave is the
    largest whole number of inner ``query_tile_tokens`` tiles that
    fits, never wider than the invocation can be and never wider than
    ``max_wave``. Both bands share this arithmetic but pass their own budget and
    ceiling, so raising the prefill budget cannot perturb the decode chunk.

    Args:
        columns: ``N``, the score plane's column count.
        heads_group: ``H_group``, the plane's head stride.
        token_capacity: The exchange capacity; a wave wider than the whole
            invocation is pure waste.
        budget_bytes: Device bytes the score + validity planes may occupy.
        max_wave: Ceiling on the result.
        query_tile_tokens: Inner scorer tile width.

    Returns:
        The wave in query rows: a multiple of the inner tile, at least one tile.
    """
    tile = query_tile_tokens
    bytes_per_row = heads_group * columns * (4 + 1)
    tiles = max(1, (budget_bytes // bytes_per_row) // tile)
    wave = min(tiles * tile, max_wave, _cdiv(token_capacity, tile) * tile)
    return max(tile, wave)


def icp_prefill_column_rungs(
    max_blocks: int,
    *,
    floor: int,
    steps_per_octave: int,
) -> tuple[int, ...]:
    """The admitted score-plane column widths, ascending, ending at capacity.

    A GEOMETRIC fill: ``steps_per_octave`` equally spaced rungs in every octave
    from ``floor`` up to ``max_blocks``, plus ``max_blocks`` itself so selection
    never fails to find a rung. At the defaults and the deployed 8192-column
    capacity that is ``[4, 6, 8, 12, 16, 24, ..., 4096, 6144, 8192]`` -- 23
    rungs.

    A rung is a startup constant, so each one owns a ``PrefillPlan``, a score /
    validity view and a selector-scratch view built in ``__init__``; a build
    picks the smallest rung that covers its live extent and allocates nothing.

    **Why geometric and not a fixed step.** The rung sets the score/validity
    plane's row pitch and ``PrefillPlan.max_local_blocks``, i.e. the selector's
    launch geometry, so what a request pays is the RATIO of its rung to its live
    column count. A fixed column step is a fixed residual, which is unbounded as
    a ratio at short context; a geometric fill bounds it at
    ``1 + 1/steps_per_octave`` everywhere above ``floor``. This is the same
    derivation ``exchange_plan.py``'s
    :func:`prefill_exchange_token_capacities` makes for the exchange extent.

    The trade is at the top end: a looser rung can cost an extra scorer launch
    in bounded bands of live column counts near the wave transitions. The
    column-rung report quantifies them per configuration.

    Adding rungs is cheap in memory: every profile's planes are VIEWS into
    shared arenas, so peak retained memory is the MAXIMUM over profiles, not
    their sum. Only the per-profile object overhead tracks the rung count.

    The floor is where the ratio stops being the interesting quantity: below it
    a plane is a few KiB per wave. 4 columns == 512 tokens of history, well
    below the narrowest selector arm the package could grow, so the vLLM side
    is not what would block such an arm.

    Args:
        max_blocks: ``cdiv(max_model_len, physical_page_tokens)``, the capacity.
        floor: Smallest admitted rung, resolved by the caller.
        steps_per_octave: Rungs per octave. Raising it lowers the
            worst-case over-run to ``1 + 1/steps`` and costs rungs.

    Returns:
        Ascending column counts, the last of which is ``max_blocks``.

    Raises:
        ValueError: If ``max_blocks``, ``floor`` or ``steps_per_octave`` is not
            positive.
    """
    max_blocks, floor = int(max_blocks), int(floor)
    steps_per_octave = int(steps_per_octave)
    if max_blocks < 1:
        raise ValueError(f"max_blocks={max_blocks} must be at least 1")
    if floor < 1:
        raise ValueError(f"floor={floor} must be at least 1")
    if steps_per_octave < 1:
        raise ValueError(
            f"steps_per_octave={steps_per_octave} must be at least 1; one step "
            "per octave is the pure power-of-two ladder."
        )
    rungs = {max_blocks}
    octave = floor
    while octave <= max_blocks:
        for step in range(steps_per_octave):
            rung = octave + (octave * step) // steps_per_octave
            if floor <= rung <= max_blocks:
                rungs.add(rung)
        octave *= 2
    return tuple(sorted(rungs))


def icp_outer_chunk_tokens(
    *,
    max_model_len: int,
    heads_group: int,
    token_capacity: int,
    physical_page_tokens: int,
    budget_bytes: int,
    max_chunk: int,
    query_tile_tokens: int,
) -> int:
    """The outer query chunk, from startup bounds only.

    Every input is a config constant, so the result is identical on every rank
    -- which it must be: the chunk sets the number of ``OnlyScoreIcp`` calls and
    the plan-workspace pool's slot count, and a rank-dependent value would give
    two ranks different exchange geometry.

    The chunk is a multiple of the inner tile ``query_tile_tokens``
    so that raising it only adds whole tiles to each plan.

    Args:
        max_model_len: Sets ``N = cdiv(max_model_len, 128)``, the wave's
            column stride.
        heads_group: ``H_group``, the wave's head stride.
        token_capacity: The exchange capacity; the chunk never exceeds it,
            because a chunk wider than the whole invocation is pure waste.
        physical_page_tokens: Tokens per compound page.
        budget_bytes: Device bytes the score + validity scratch may occupy.
        max_chunk: Ceiling on the result.
        query_tile_tokens: Inner scorer tile width.

    Returns:
        The outer chunk in query rows, a multiple of the inner tile and at
        least one tile wide.
    """
    return _icp_wave_from_budget(
        columns=_cdiv(max_model_len, physical_page_tokens),
        heads_group=heads_group,
        token_capacity=token_capacity,
        budget_bytes=budget_bytes,
        max_wave=max_chunk,
        query_tile_tokens=query_tile_tokens,
    )


def icp_decode_chunk_tokens(
    *,
    max_model_len: int,
    heads_group: int,
    token_capacity: int,
    physical_page_tokens: int,
    budget_bytes: int,
    max_chunk: int,
    query_tile_tokens: int,
) -> int:
    """The decode outer chunk, from the fixed decode scratch budget (r13).

    Args:
        max_model_len: Sets ``N``, the decode plane's column count.
        heads_group: ``H_group``.
        token_capacity: The exchange capacity.
        physical_page_tokens: Tokens per compound page.
        budget_bytes: Resolved decode score + validity scratch budget.
        max_chunk: Decode chunk ceiling.
        query_tile_tokens: Inner scorer tile width.

    Returns:
        The decode chunk in query rows, a multiple of the inner tile.
    """
    return icp_outer_chunk_tokens(
        max_model_len=max_model_len,
        heads_group=heads_group,
        token_capacity=token_capacity,
        physical_page_tokens=physical_page_tokens,
        budget_bytes=budget_bytes,
        max_chunk=max_chunk,
        query_tile_tokens=query_tile_tokens,
    )


def icp_prefill_wave_ladder(
    columns: tuple[int, ...],
    *,
    heads_group: int,
    token_capacity: int,
    budget_bytes: int,
    max_wave: int,
    query_tile_tokens: int,
) -> dict[int, int]:
    """The derived prefill wave, in query rows, per admitted column rung."""
    return {
        rung: _icp_wave_from_budget(
            columns=rung,
            heads_group=heads_group,
            token_capacity=token_capacity,
            budget_bytes=budget_bytes,
            max_wave=max_wave,
            query_tile_tokens=query_tile_tokens,
        )
        for rung in columns
    }


def icp_prefill_rung(columns: tuple[int, ...], live_blocks: int) -> int:
    """The smallest admitted rung covering ``live_blocks`` columns."""
    return next(rung for rung in columns if rung >= live_blocks)


def icp_prefill_arena_elems(waves: dict[int, int], heads_group: int) -> int:
    """Cells the widest ``[wave, H_group, columns]`` prefill plane needs."""
    return max(wave * heads_group * columns for columns, wave in waves.items())


def icp_cute_decode_window(
    *, query_len: int, token_begin: int, rows: int, chunk: int, num_reqs: int
) -> tuple[int, int, int, int] | None:
    """``(token_begin, token_count, request_begin, request_count)``.

    ``rows`` is the band's row count from ``token_begin``. Returns None when no
    request of the metadata reaches the window (capture padding only).
    """
    request_begin = token_begin // query_len
    if request_begin >= num_reqs:
        return None
    token_count = min(rows, chunk)
    request_end = min(_cdiv(token_begin + token_count, query_len), num_reqs)
    return (token_begin, token_count, request_begin, request_end - request_begin)


@dataclass(frozen=True)
class ICPRowWindow:
    """One scorer launch + one selector launch of a ``W > 1`` invocation.

    Rows ``[row_begin, row_end)`` are scored and rows ``[row_begin, row_begin +
    select_rows)`` are selected into the same candidate rows, so windows of
    both bands write one candidate layout and the single exchange downstream is
    unchanged. ``slot`` is the position on the band's own grid
    (``grid_begin // grid_width``), which is what the native writer's slot-major
    causal offsets and the plan-workspace pool are keyed by.
    """

    cute: bool
    slot: int
    grid_begin: int
    row_begin: int
    row_end: int
    select_rows: int
    decode_selector: bool


def icp_row_windows(
    *,
    num_tokens: int,
    cap: int,
    cute_rows: int,
    decode_chunk: int,
    prefill_width: int,
    has_prefill: bool,
) -> list[ICPRowWindow]:
    """Split one invocation into scorer + selector windows.

    Without prefill the whole invocation is one band on the decode grid, and
    each window's selector extent is pinned to a captured decode plan
    (``min(chunk, cap - t0)``). With prefill (r13: a mixed step is all FMHA)
    every row is scored by FMHA on the prefill-wave grid.

    Args:
        num_tokens: Rows of the invocation (padded under a FULL graph).
        cap: The invocation's exchange extent.
        cute_rows: Rows scored by CuTe: 0 or ``num_tokens`` without prefill,
            0 with prefill.
        decode_chunk: The decode plane's row count.
        prefill_width: The prefill wave (ignored without prefill).
        has_prefill: The invocation's phase.

    Returns:
        Windows in row order.
    """
    if not has_prefill:
        assert cute_rows in (0, num_tokens), (cute_rows, num_tokens)
        return [
            ICPRowWindow(
                cute=cute_rows > 0,
                slot=slot,
                grid_begin=t0,
                row_begin=t0,
                row_end=min(t0 + decode_chunk, num_tokens),
                select_rows=min(decode_chunk, cap - t0),
                decode_selector=True,
            )
            for slot, t0 in enumerate(range(0, num_tokens, decode_chunk))
        ]
    assert cute_rows == 0, cute_rows
    windows = []
    for slot, t0 in enumerate(range(0, num_tokens, prefill_width)):
        windows.append(
            ICPRowWindow(
                cute=False,
                slot=slot,
                grid_begin=t0,
                row_begin=t0,
                row_end=min(t0 + prefill_width, num_tokens),
                select_rows=min(t0 + prefill_width, cap) - t0,
                decode_selector=False,
            )
        )
    return windows


def bind_cute_launch(
    scorer,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    active_rows: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
) -> Callable[..., None]:
    """The per-layer CuTe launch: MSA's unchecked ``launch`` on a scorer with
    this window bound, every per-step argument bound here."""
    launch = scorer.launch

    def run(index_q, index_kv, k_pages, scale) -> None:
        launch(
            index_q,
            index_kv,
            block_table,
            query_start_loc,
            seq_lens,
            positions,
            active_rows,
            scores,
            valid,
        )

    return run


def bind_fmha_launch(
    fmha: Callable[..., None],
    plan: dict,
    row_begin: int,
    rows: int,
    **kwargs,
) -> Callable[..., None]:
    """The per-layer FMHA ``OnlyScoreIcp`` launch with its step bound."""

    def launch(index_q, index_kv, k_pages, scale) -> None:
        fmha(
            index_q[row_begin : row_begin + rows],
            k_pages,
            k_pages,  # V placeholder; never read in OnlyScore
            plan,
            output_o=False,
            output_maxscore=True,
            sm_scale=scale,
            **kwargs,
        )

    return launch


def prefill_full_row_controls(
    columns: int,
    *,
    threads: int,
    cached_items: int,
    four_warp_finish_max_columns: int,
) -> tuple[int, int, bool]:
    """The measured controls for a rung of ``columns`` score columns."""
    return (
        threads,
        cached_items,
        columns <= four_warp_finish_max_columns,
    )


def prefill_selector_kwargs(controls: tuple[int, int, bool]) -> dict:
    """``select_prefill_candidates`` control keywords."""
    return dict(
        full_row_threads=controls[0],
        full_row_cached_items=controls[1],
        full_row_four_warp_finish=controls[2],
    )
