"""Exact C4 and dispatch gates for prefill's capacity <=32 row warps.

Only explicit 128-thread FULL_ROW controls admit this specialization. The
oracle sorts ordinary CPU (score, affine ID) records, independently of CUDA
keys or any other selector. Graph tests capture the qualified native entry;
the typed public prefill API remains eager-only.
"""

from __future__ import annotations

import math
import random
import struct

import pytest

from fmha_sm100.icp import candidates as c

_POISON = 0x5A5A5A5A
_INT_MAX = (1 << 31) - 1
_CONTROLS = [(0, False), (0, True), (12, False), (12, True)]


def _controls(cached_items, merge4):
    return dict(
        full_row_threads=128,
        full_row_cached_items=cached_items,
        full_row_four_warp_finish=merge4,
    )


def _word(value):
    return struct.unpack("<i", struct.pack("<f", 0.0 if value == 0.0 else value))[0]


def _scores_and_oracle(
    columns, heads, counts, excluded, active, *, begin, stride, phase=0
):
    scores = [[[math.nan] * columns for _ in range(heads)] for _ in counts]
    expected = [
        [[[_word(-math.inf), -1] for _ in range(16)] for _ in range(heads)]
        for _ in counts
    ]
    for token, enabled in enumerate(active):
        if not enabled:
            continue
        for head in range(heads):
            kind = (token + head + phase) % 4
            records = []
            for column in range(counts[token]):
                if column == excluded[token]:
                    continue
                if kind == 0:
                    score = (-math.inf, -3.0, 0.0, -0.0, 1.0, math.inf, 1.0)[
                        (column + token) % 7
                    ]
                elif kind == 1:
                    score = -math.inf
                elif kind == 2:
                    score = 0.0 if column % 2 else -0.0
                else:
                    score = struct.unpack(
                        "<f",
                        struct.pack("<I", 0x3F800000 + ((column * 37 + token) % 11)),
                    )[0]
                scores[token][head][column] = score
                records.append((score, begin + stride * column))
            records.sort(key=lambda record: (-record[0], record[1]))
            for slot, (score, gid) in enumerate(records[:16]):
                expected[token][head][slot] = [_word(score), gid]
    return scores, expected


def _buffers(device, plan, counts, excluded, active, *, begin=0, stride=1, phase=0):
    import torch

    source, expected = _scores_and_oracle(
        plan.scan_extent,
        plan.num_heads_group,
        counts,
        excluded,
        active,
        begin=begin,
        stride=stride,
        phase=phase,
    )
    scores = torch.tensor(source, dtype=torch.float32, device=device)
    geometry = c.CandidateGeometry(
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(excluded, dtype=torch.int32, device=device),
        torch.tensor(active, dtype=torch.bool, device=device),
        scan_block_begin=begin,
        global_block_stride=stride,
    )
    rows = len(counts) * plan.num_heads_group
    # Absent warps in the final CTA must not write beyond the advertised rows.
    storage = torch.full(((rows + 4) * 32,), _POISON, dtype=torch.int32, device=device)
    out = (
        storage[: rows * 32]
        .view(len(counts), plan.num_heads_group, 16, 2)
        .view(torch.float32)
    )
    partials = torch.empty(
        plan.partial_shape(len(counts), plan.scan_extent),
        dtype=torch.float32,
        device=device,
    )
    return (
        scores,
        geometry,
        out,
        partials,
        storage,
        torch.tensor(expected, dtype=torch.int32),
    )


def _check_output(out, storage, expected):
    import torch

    torch.testing.assert_close(out.view(torch.int32).cpu(), expected, rtol=0, atol=0)
    assert (storage[out.numel() :].cpu() == _POISON).all()


def _launch_and_record(scores, geometry, out, partials, plan, **kwargs):
    c.record_launches(True)
    try:
        c.select_prefill_candidates(
            scores,
            geometry,
            plan=plan,
            live_blocks=0,
            out=out,
            partials=partials,
            **kwargs,
        )
        launched, threads = c.last_launch(), c.last_launch_threads()
    finally:
        c.record_launches(False)
    planned = c.planned_prefill_kernel(plan.scan_extent, **kwargs)
    assert launched["symbol"] == planned["symbol"]
    assert launched["regs"] == planned["regs"]
    assert launched["shared"] == planned["shared"]
    return launched, threads


def test_tiny_warp_shuffle_network_against_independent_global_sort():
    rng = random.Random(2026091503)
    for count in range(33):
        for _ in range(16):
            values = [
                key | ((index & 1) << 63)
                for index, key in enumerate(rng.sample(range(1, 1 << 63), count))
            ] + [0] * (32 - count)
            rng.shuffle(values)
            wanted = sorted(values, reverse=True)
            width = 2
            while width <= 32:
                distance = width >> 1
                while distance:
                    before = values[:]
                    for lane in range(32):
                        pair = (before[lane], before[lane ^ distance])
                        values[lane] = (
                            max(pair)
                            if ((lane & width) == 0) == ((lane & distance) == 0)
                            else min(pair)
                        )
                    distance >>= 1
                width <<= 1
            assert values == wanted


@pytest.mark.gpu
@pytest.mark.parametrize("cached_items, merge4", _CONTROLS)
def test_tiny_warp_exact_poison_affine_and_partial_cta(
    cuda_device, cached_items, merge4
):
    tiny_symbol = None
    for columns in (1, 8, 16, 17, 31, 32):
        for rank, stride in ((0, 1), (1, 1), (0, 2), (1, 2)):
            # P128 uses identity on both ranks. The stride-two cases also
            # prove that huge even/odd affine IDs survive the C4 bitcast.
            begin = 0 if stride == 1 else _INT_MAX - (1 - rank) - stride * (columns - 1)
            counts = [0, 1, columns, min(17, columns), columns, columns, _INT_MAX]
            excluded = [-7, 0, columns // 2, -1, columns - 1, -1, _INT_MAX]
            active = [True] * 6 + [False]
            # H2 and odd T leave two absent row warps in the final CTA;
            # H4 is the measured TP2 profile and has a complete final CTA.
            for heads_local in (1, 2):
                plan = c.PrefillPlan(
                    icp_degree=2,
                    icp_rank=rank,
                    num_heads_local=heads_local,
                    token_capacity=len(counts),
                    max_local_blocks=columns,
                )
                scores, geometry, out, partials, storage, expected = _buffers(
                    cuda_device,
                    plan,
                    counts,
                    excluded,
                    active,
                    begin=begin,
                    stride=stride,
                )
                launched, threads = _launch_and_record(
                    scores,
                    geometry,
                    out,
                    partials,
                    plan,
                    arm=c.SelectArm.RADIX_FULL_ROW,
                    **_controls(cached_items, merge4),
                )
                assert threads == 128
                assert launched["grid"] == (len(counts) * plan.num_heads_group + 3) // 4
                assert launched["shared"] == 0
                tiny_symbol = launched["symbol"] if tiny_symbol is None else tiny_symbol
                assert launched["symbol"] == tiny_symbol
                _check_output(out, storage, expected)


@pytest.mark.gpu
def test_tiny_warp_static_capacity_admission_and_retained_controls(cuda_device):
    tiny = c.planned_prefill_kernel(
        32, arm=c.SelectArm.RADIX_FULL_ROW, **_controls(0, False)
    )["symbol"]
    variants = [
        (c.SelectArm.RADIX_BOUNDED, {}),
        (c.SelectArm.RADIX_FULL_ROW, {}),
        (
            c.SelectArm.RADIX_FULL_ROW,
            dict(full_row_threads=256, full_row_cached_items=0),
        ),
        (
            c.SelectArm.RADIX_FULL_ROW,
            dict(full_row_threads=256, full_row_cached_items=6),
        ),
    ]
    variants.extend(
        (c.SelectArm.RADIX_FULL_ROW, _controls(cache, merge4))
        for cache, merge4 in _CONTROLS
    )
    for columns in (8, 32, 33):
        plan = c.PrefillPlan(
            icp_degree=2,
            icp_rank=0,
            num_heads_local=1,
            token_capacity=3,
            max_local_blocks=columns,
        )
        for arm, controls in variants:
            # N33 has a short device prefix, which must not admit the tiny
            # kernel: launch policy may only depend on the advertised shape.
            counts = [min(columns, 17), 1, 0]
            scores, geometry, out, partials, storage, expected = _buffers(
                cuda_device, plan, counts, [7, 0, -1], [True] * 3
            )
            launched, threads = _launch_and_record(
                scores, geometry, out, partials, plan, arm=arm, **controls
            )
            admitted = columns <= 32 and controls.get("full_row_threads") == 128
            assert (launched["symbol"] == tiny) == admitted
            assert launched["grid"] == (2 if admitted else 6)
            expected_threads = (
                128
                if arm is c.SelectArm.RADIX_BOUNDED
                else controls.get("full_row_threads", 512)
            )
            assert threads == expected_threads
            _check_output(out, storage, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("cached_items, merge4", _CONTROLS)
@pytest.mark.parametrize("use_pdl", [False, True])
def test_tiny_warp_native_replay_mutates_prefix_and_active_rows(
    cuda_device,
    cached_items,
    merge4,
    use_pdl,
):
    import torch

    if use_pdl and torch.cuda.get_device_capability(cuda_device)[0] < 9:
        pytest.skip("PDL needs compute capability >=9; the target is GB300")
    plan = c.PrefillPlan(
        icp_degree=2,
        icp_rank=1,
        num_heads_local=1,
        token_capacity=3,
        max_local_blocks=32,
        use_pdl=use_pdl,
    )
    metadata = [
        ([32, 17, 0], [7, -1, -7], [True] * 3),
        ([1, 31, _INT_MAX], [0, 30, _INT_MAX], [True, True, False]),
        ([_INT_MAX, 16, 32], [_INT_MAX, 15, -1], [False, True, True]),
        ([_INT_MAX] * 3, [_INT_MAX] * 3, [False] * 3),
        ([0, 1, 15], [-9, -1, 7], [True] * 3),
    ]
    begin = _INT_MAX - 62
    states = [
        _buffers(cuda_device, plan, *state, begin=begin, stride=2, phase=i)
        for i, state in enumerate(metadata)
    ]
    scores, geometry, out, partials, storage, expected = _buffers(
        cuda_device, plan, *metadata[0], begin=begin, stride=2
    )
    controls = _controls(cached_items, merge4)
    _launch_and_record(
        scores,
        geometry,
        out,
        partials,
        plan,
        arm=c.SelectArm.RADIX_FULL_ROW,
        **controls,
    )
    _check_output(out, storage, expected)
    extension = c._build._select_ext()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        extension.select_prefill_candidates(
            scores,
            geometry.local_valid_blocks,
            geometry.forced_column,
            geometry.active_rows,
            out,
            partials,
            begin,
            use_pdl,
            plan.scan_extent,
            global_block_stride=2,
            arm=int(c.SelectArm.RADIX_FULL_ROW),
            **controls,
        )
    # Return to the first full prefix after empty/inactive replays. Only
    # device buffers change; launch geometry and pointers remain frozen.
    for next_scores, next_geometry, _, _, _, expected in states + states[:1]:
        scores.copy_(next_scores)
        geometry.local_valid_blocks.copy_(next_geometry.local_valid_blocks)
        geometry.forced_column.copy_(next_geometry.forced_column)
        geometry.active_rows.copy_(next_geometry.active_rows)
        storage.fill_(_POISON)
        graph.replay()
        _check_output(out, storage, expected)
