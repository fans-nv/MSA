# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-invariant candidate extents and ACK-free layer-slot admission.

Pure host planning: importing this module does not import torch or initialize
CUDA. The caller supplies resolved scheduler and capture limits on every rank.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .abi import QCAPACITY_ALIGNMENT

ICP_PREFILL_EXTENT_STEPS_PER_OCTAVE = 2
ICP_EXCHANGE_SLOTS = 3
ACK_FREE_SLOTS = 3


def _align_capacity(n: int) -> int:
    """Round a token count up to the exchange's row alignment."""
    return -(-int(n) // QCAPACITY_ALIGNMENT) * QCAPACITY_ALIGNMENT


def exchange_token_capacities(
    *,
    max_token_capacity: int,
    cudagraph_capture_sizes: Iterable[int] = (),
) -> list[int]:
    """The fixed, ascending set of exchange extents the decode band may bind.

    Derived from resolved serving configuration only -- the scheduler's token budget and
    the compiled cudagraph capture sizes -- so the list is identical on every
    rank. Nothing a step discovers may enter it (refined-icp-v1 C4).

    Every aligned capture size is kept, plus powers of two for eager decode
    and the global capacity. This includes 1024-row Q4 verification beyond
    the 512-row capture ceiling. The global capacity is what an eager batch
    can reach; it is
    always the last entry, so :func:`select_token_capacity` never fails to find
    a rung. The ladder exists for C4, not for the kernel, which accepts any
    extent up to the capacity.

    Args:
        max_token_capacity: ``max_num_batched_tokens``; the returned list's
            last entry is this value aligned up.
        cudagraph_capture_sizes: The compiled capture ladder, if any.

    Returns:
        Ascending, deduplicated, alignment-respecting token capacities.
    """
    cap = _align_capacity(max_token_capacity)
    if cap < QCAPACITY_ALIGNMENT:
        raise ValueError(
            f"max_token_capacity={max_token_capacity} does not admit a single "
            f"{QCAPACITY_ALIGNMENT}-row exchange."
        )
    captured = {
        _align_capacity(int(size))
        for size in cudagraph_capture_sizes
        if 0 < int(size) <= cap
    }
    eager = QCAPACITY_ALIGNMENT
    while eager < cap:
        captured.add(eager)
        eager *= 2
    captured.add(cap)
    return sorted(captured)


def prefill_exchange_token_capacities(
    *,
    max_token_capacity: int,
    steps_per_octave: int | None = None,
) -> list[int]:
    """The fixed, ascending set of extents the eager-prefill band may bind.

    A geometric fill: ``steps_per_octave`` equally spaced rungs in every octave
    from the row alignment up to the global capacity, each rounded up to
    :data:`QCAPACITY_ALIGNMENT`, plus the capacity itself so selection never
    fails to find a rung. At the default two steps that is
    ``[4, 8, 12, 16, 24, 32, 48, ...]``.

    A fill is admissible here and not in :func:`exchange_token_capacities`
    because that ladder serves FULL cudagraph replays, where a rung no capture
    size names can never be selected and the extent must additionally be
    constant across replays of one graph. ``has_prefill`` implies this
    invocation is eager (the converse does not hold -- a decode invocation can
    be eager too), so there is no graph to be constant across and every rung
    here is reachable, nothing else quantising an eager batch's row count.

    What C4 still requires is that both ranks resolve the same extent, and they
    do: every input below is a startup constant and the caller's selection
    input is ``num_actual_tokens``, which is rank-identical scheduler metadata.

    Args:
        max_token_capacity: ``max_num_batched_tokens``; the returned list's
            last entry is this value aligned up.
        steps_per_octave: Rungs per octave; defaults to
            :data:`ICP_PREFILL_EXTENT_STEPS_PER_OCTAVE`. Raising it lowers the
            worst-case over-run to ``1 + 1/steps`` and raises the retained
            carrier bytes, which are proportional to the sum of the rungs.

    Returns:
        Ascending, deduplicated, alignment-respecting token capacities.
    """
    cap = _align_capacity(max_token_capacity)
    if cap < QCAPACITY_ALIGNMENT:
        raise ValueError(
            f"max_token_capacity={max_token_capacity} does not admit a single "
            f"{QCAPACITY_ALIGNMENT}-row exchange."
        )
    if steps_per_octave is None:
        steps_per_octave = ICP_PREFILL_EXTENT_STEPS_PER_OCTAVE
    steps_per_octave = int(steps_per_octave)
    if steps_per_octave < 1:
        raise ValueError(
            f"steps_per_octave={steps_per_octave} must be at least 1; one step "
            "per octave is the pure power-of-two ladder."
        )
    rungs = {cap}
    octave = QCAPACITY_ALIGNMENT
    while octave <= cap:
        for step in range(steps_per_octave):
            rung = _align_capacity(octave + (octave * step) // steps_per_octave)
            if QCAPACITY_ALIGNMENT <= rung <= cap:
                rungs.add(rung)
        octave *= 2
    return sorted(rungs)


def admitted_exchange_extents(
    *,
    max_token_capacity: int,
    cudagraph_capture_sizes: Iterable[int] = (),
    steps_per_octave: int | None = None,
) -> list[int]:
    """Every extent the one exchange must admit: both bands' ladders, merged.

    The bands share one :class:`CandidateExchange`, which validates the
    presented extent against a single admitted set and retains one carrier
    workspace per member, so the instance is built from the union. Merging
    cannot move either band's own selection: ``select_token_capacity`` is
    evaluated per band over that band's list, never over this one.

    Returns:
        Ascending, deduplicated token capacities; the last entry is the global
        capacity, which both ladders end at.
    """
    decode = exchange_token_capacities(
        max_token_capacity=max_token_capacity,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
    )
    prefill = prefill_exchange_token_capacities(
        max_token_capacity=max_token_capacity,
        steps_per_octave=steps_per_octave,
    )
    return sorted(set(decode) | set(prefill))


def select_token_capacity(capacities: Sequence[int], num_tokens: int) -> int:
    """The smallest admitted extent that holds ``num_tokens`` rows.

    Under a FULL cudagraph ``num_tokens`` must be the invocation's padded
    extent, never a live count: the padded extent is a constant of the graph,
    so the extent this returns is too, which is what C4 requires. On the eager
    band there is no graph and ``num_actual_tokens`` already is the padded
    extent; what remains of C4 there is rank-equality, which that quantity has.

    Pass the band's own ``capacities`` (:func:`exchange_token_capacities` for
    decode, :func:`prefill_exchange_token_capacities` for eager prefill), never
    the merged :func:`admitted_exchange_extents`, which exists only so the
    shared instance admits both.
    """
    for capacity in capacities:
        if num_tokens <= capacity:
            return capacity
    raise ValueError(
        f"ICP: {num_tokens} token rows exceed every admitted exchange extent "
        f"{list(capacities)}; the largest is the scheduler's own token budget, "
        "so a batch above it was not built under the configured bounds."
    )


def check_slot_discipline(num_sparse_layers: int, slots: int) -> None:
    """Check the ACK-free slot-reuse obligation at startup.

    The native transport requires at least three slots. The serving owner
    must also avoid repeating a slot across the wrap of its layer sweep.
    """
    if slots < ACK_FREE_SLOTS:
        raise ValueError(
            f"ICP: slots={slots} is below the ACK-free bound "
            f"{ACK_FREE_SLOTS}; K5T has no acknowledgement, so slot rotation "
            f"is its only reuse protection."
        )
    if num_sparse_layers % slots == 1:
        raise ValueError(
            f"ICP: {num_sparse_layers} sparse layers rotating through {slots} "
            f"slots repeats a slot across the wrap of a layer sweep "
            f"({num_sparse_layers} % {slots} == 1), so two consecutive "
            f"launches share a slot with no acknowledgement between them. "
            f"Under K5T's per-word generation tags that is a transport "
            f"timeout."
        )
