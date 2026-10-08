"""P128 prefill gates for explicit bounded and whole-row selection.

The CPU oracle sorts the same immutable FP32 scores by score descending and
logical B128 ID ascending. It uses neither CUDA keys nor another selector.
Each rank owns a fragment of every logical block, including with a cached
prefix; query positions remain absolute across inner Q128 tiles.

Only the private, already-qualified native entry is captured for the replay
gate. The public PrefillPlan API deliberately remains eager-only.
"""

from __future__ import annotations

import bisect
import math
import random
import struct
import sys
from types import SimpleNamespace

import pytest

from fmha_sm100.icp import candidates as c

_ARMS = (c.SelectArm.RADIX_BOUNDED, c.SelectArm.RADIX_FULL_ROW)
_VARIANTS = (
    pytest.param(c.SelectArm.RADIX_BOUNDED, 512, 3, id="bounded"),
    pytest.param(c.SelectArm.RADIX_FULL_ROW, 512, 3, id="full-row-512-cache3"),
    pytest.param(c.SelectArm.RADIX_FULL_ROW, 128, 0, id="full-row-128-stream"),
    pytest.param(c.SelectArm.RADIX_FULL_ROW, 128, 12, id="full-row-128-cache12"),
    pytest.param(c.SelectArm.RADIX_FULL_ROW, 256, 0, id="full-row-256-stream"),
    pytest.param(c.SelectArm.RADIX_FULL_ROW, 256, 6, id="full-row-256-cache6"),
)
_POISON = 0x5A5A5A5A
_CAPACITY = 8192


def _plan(*, tokens=1024, rank=0, columns=_CAPACITY):
    return c.PrefillPlan(
        icp_degree=2,
        icp_rank=rank,
        num_heads_local=2,
        token_capacity=tokens,
        max_local_blocks=columns,
    )


def _word(score):
    score = 0.0 if score == 0.0 else score
    return struct.unpack("<i", struct.pack("<f", score))[0]


def _oracle(scores, counts, excluded, active, *, begin=0):
    import torch

    out = torch.empty((*scores.shape[:2], 16, 2), dtype=torch.int32)
    out[..., 0], out[..., 1] = _word(-math.inf), -1
    for token, enabled in enumerate(active):
        if not enabled:
            continue
        for head in range(scores.shape[1]):
            ordinary = []
            for column, score in enumerate(
                scores[token, head, : counts[token]].tolist()
            ):
                if column == excluded[token]:
                    continue
                assert not math.isnan(score), "live ordinary score is NaN"
                ordinary.append((score, begin + column))
            ordinary.sort(key=lambda pair: (-pair[0], pair[1]))
            for slot, (score, gid) in enumerate(ordinary[:16]):
                out[token, head, slot, 0] = _word(score)
                out[token, head, slot, 1] = gid
    return out


def _scores(counts, excluded, active):
    import torch

    width = max((n for n, enabled in zip(counts, active) if enabled), default=0)
    scores = torch.full((len(counts), 4, width), math.nan, dtype=torch.float32)
    for token, enabled in enumerate(active):
        if not enabled:
            continue
        live = counts[token]
        index = torch.arange(live, dtype=torch.int32)
        scores[token, 0, :live] = ((index * 37 + token * 11) % 67).float() - 33
        if token % 5 == 0:
            scores[token, 0, :live:19] = math.inf
        scores[token, 1, :live] = -math.inf
        scores[token, 2, :live] = torch.where(index % 2 == 0, 0.0, -0.0)
        # One coarse bin, distinct FP32 mantissas and tied cutoff IDs.
        words = 0x3F800000 + ((index * 73 + token * 17) % 257)
        scores[token, 3, :live] = words.view(torch.float32)
        if excluded[token] >= 0:
            scores[token, :, excluded[token]] = math.nan
    return scores


def _prefill_rows(tokens, history, rank):
    # Enumerate the rank's fragment starts independently of the production
    # division/remainder formula. Future rows in this materialized invocation
    # do not become visible to an earlier query.
    starts = range(rank * 64, history + tokens, 128)
    counts = [sum(start <= history + i for start in starts) for i in range(tokens)]
    excluded = [
        ((history + i) // 128 if (history + i) // 128 < counts[i] else -1)
        for i in range(tokens)
    ]
    active = [True] * tokens
    # Padding may retain arbitrary metadata and scores. Keep the early causal
    # boundary rows active, including rank one's genuinely empty prefix.
    active[-1] = False
    counts[-1] = excluded[-1] = (1 << 31) - 1
    return counts, excluded, active


def _buffers(device, plan, scores, counts, excluded, active):
    import torch

    tokens = len(counts)
    gpu_scores = torch.full(
        (tokens, 4, plan.max_local_blocks), math.nan, dtype=torch.float32, device=device
    )
    gpu_scores[..., : scores.shape[-1]].copy_(scores)
    geometry = c.CandidateGeometry(
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(excluded, dtype=torch.int32, device=device),
        torch.tensor(active, dtype=torch.bool, device=device),
    )
    out = torch.full(
        (tokens, 4, 16, 2), _POISON, dtype=torch.int32, device=device
    ).view(torch.float32)
    partials = torch.full(
        plan.partial_shape(tokens, plan.scan_extent),
        _POISON,
        dtype=torch.int32,
        device=device,
    ).view(torch.float32)
    return gpu_scores, geometry, out, partials


@pytest.mark.parametrize("arm", [a for a in c.SelectArm if a not in _ARMS])
def test_unsupported_prefill_arms_fail_before_validation_or_build(monkeypatch, arm):
    def forbidden(*args, **kwargs):
        pytest.fail("an unsupported prefill arm reached validation or JIT")

    monkeypatch.setattr(c, "_validate", forbidden)
    monkeypatch.setattr(c._build, "_select_ext", forbidden)
    with pytest.raises(
        ValueError, match="prefill arm must be RADIX_BOUNDED or RADIX_FULL_ROW"
    ):
        c.select_prefill_candidates(None, None, plan=_plan(), live_blocks=0, arm=arm)


@pytest.mark.parametrize(
    "arm, expected",
    [
        (None, -2),
        (c.SelectArm.RADIX_BOUNDED, -2),
        ("radix_bounded", -2),
        (c.SelectArm.RADIX_FULL_ROW, -3),
        ("radix_full_row", -3),
    ],
)
def test_prefill_dispatch_is_explicit_and_capacity_bound(monkeypatch, arm, expected):
    calls = []
    plan = _plan(tokens=1024)
    geometry = c.CandidateGeometry("counts", "forced", "active")
    out, partials = object(), object()
    monkeypatch.setenv(c.SELECTOR_ENV, "capped16")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Tensor=type("Tensor", (), {}),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
        ),
    )
    monkeypatch.setattr(c, "_validate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        c._build,
        "_select_ext",
        lambda: SimpleNamespace(
            select_prefill_candidates=lambda *args, **kwargs: calls.append(
                (args, kwargs)
            )
        ),
    )
    result = c.select_prefill_candidates(
        SimpleNamespace(shape=(257, 4, _CAPACITY)),
        geometry,
        plan=plan,
        live_blocks=1,
        out=out,
        partials=partials,
        arm=arm,
    )
    assert result is out
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1:6] == ("counts", "forced", "active", out, partials)
    assert args[8] == _CAPACITY
    assert kwargs == {
        "global_block_stride": 1,
        "arm": expected,
        "full_row_threads": 512,
        "full_row_cached_items": 3,
        "full_row_four_warp_finish": False,
    }


@pytest.mark.parametrize(
    "threads, cached_items, merge4",
    [
        (128, 0, False),
        (128, 12, False),
        (256, 0, False),
        (256, 6, False),
        (128, 0, True),
        (128, 12, True),
    ],
)
def test_prefill_cta_controls_reach_native_and_attribute_queries(
    monkeypatch,
    threads,
    cached_items,
    merge4,
):
    calls = []
    attrs = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Tensor=type("Tensor", (), {}),
            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
        ),
    )
    monkeypatch.setattr(c, "_validate", lambda *args, **kwargs: None)

    def query(*args):
        attrs.append(args)
        return 17, 56, 5328

    monkeypatch.setattr(
        c._build,
        "_select_ext",
        lambda: SimpleNamespace(
            select_prefill_candidates=lambda *args, **kwargs: calls.append(kwargs),
            prefill_kernel_attributes=query,
            last_launch_threads=lambda: threads,
        ),
    )
    controls = dict(
        full_row_threads=threads,
        full_row_cached_items=cached_items,
        full_row_four_warp_finish=merge4,
    )
    c.select_prefill_candidates(
        SimpleNamespace(shape=(1024, 4, _CAPACITY)),
        c.CandidateGeometry("counts", "forced", "active"),
        plan=_plan(),
        live_blocks=8,
        out=object(),
        partials=object(),
        arm=c.SelectArm.RADIX_FULL_ROW,
        **controls,
    )
    assert calls == [{"global_block_stride": 1, "arm": -3, **controls}]
    assert (
        c.planned_prefill_kernel(_CAPACITY, arm=c.SelectArm.RADIX_FULL_ROW, **controls)[
            "symbol"
        ]
        == 17
    )
    assert attrs == [(_CAPACITY, -3, threads, cached_items, merge4)]
    assert c.last_launch_threads() == threads


@pytest.mark.parametrize(
    "arm, threads, cached_items, error",
    [
        (c.SelectArm.RADIX_FULL_ROW, 64, 0, ValueError),
        (c.SelectArm.RADIX_FULL_ROW, 128, 3, ValueError),
        (c.SelectArm.RADIX_FULL_ROW, 256, 3, ValueError),
        (c.SelectArm.RADIX_FULL_ROW, 512, 0, ValueError),
        (c.SelectArm.RADIX_FULL_ROW, 128.0, 0, TypeError),
        (c.SelectArm.RADIX_FULL_ROW, True, 0, TypeError),
        (c.SelectArm.RADIX_FULL_ROW, object(), 0, TypeError),
        (c.SelectArm.RADIX_FULL_ROW, 128, object(), TypeError),
        (c.SelectArm.RADIX_BOUNDED, 128, 0, ValueError),
    ],
)
def test_bad_prefill_controls_fail_before_validation_or_build(
    monkeypatch,
    arm,
    threads,
    cached_items,
    error,
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid prefill controls reached tensor validation or JIT")

    monkeypatch.setattr(c, "_validate", forbidden)
    monkeypatch.setattr(c._build, "_select_ext", forbidden)
    with pytest.raises(error, match="prefill full-row|full-row controls"):
        c.select_prefill_candidates(
            None,
            None,
            plan=_plan(),
            live_blocks=0,
            arm=arm,
            full_row_threads=threads,
            full_row_cached_items=cached_items,
        )


@pytest.mark.parametrize(
    "arm, threads, cached_items, merge4, error",
    [
        (c.SelectArm.RADIX_FULL_ROW, 128, 0, 1, TypeError),
        (c.SelectArm.RADIX_FULL_ROW, 128, 0, object(), TypeError),
        (c.SelectArm.RADIX_FULL_ROW, 256, 0, True, ValueError),
        (c.SelectArm.RADIX_FULL_ROW, 512, 3, True, ValueError),
        (c.SelectArm.RADIX_BOUNDED, 512, 3, True, ValueError),
    ],
)
def test_four_warp_controls_fail_before_validation_or_build(
    monkeypatch,
    arm,
    threads,
    cached_items,
    merge4,
    error,
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid four-warp controls reached tensor validation or JIT")

    monkeypatch.setattr(c, "_validate", forbidden)
    monkeypatch.setattr(c._build, "_select_ext", forbidden)
    with pytest.raises(error, match="four-warp finish"):
        c.select_prefill_candidates(
            None,
            None,
            plan=_plan(),
            live_blocks=0,
            arm=arm,
            full_row_threads=threads,
            full_row_cached_items=cached_items,
            full_row_four_warp_finish=merge4,
        )


def test_four_warp_rank_model_against_independent_global_sort():
    # The unchanged canonical-key helper already supplies positive, unique
    # 64-bit keys. This models the new merge proof, with a separate global
    # sort as oracle; zero is padding and never an ordinary candidate.
    rng = random.Random(2026091501)
    for count in range(65, 129):
        for trial in range(32):
            keys = [
                ((1 if trial % 2 else rng.getrandbits(32)) << 32) | gid
                for gid in rng.sample(range(1, 100000), count)
            ]
            rng.shuffle(keys)
            padded = keys + [0] * (128 - count)
            warps = [
                sorted(padded[i : i + 32], reverse=True) for i in range(0, 128, 32)
            ]
            result = [None] * 16
            for warp, local in enumerate(warps):
                for lane, key in enumerate(local):
                    if key == 0:
                        continue
                    rank = lane
                    for other_warp, remote in enumerate(warps):
                        if other_warp != warp:
                            rank += bisect.bisect_left(
                                [-value for value in remote], -key
                            )
                    if rank < 16:
                        assert result[rank] is None
                        result[rank] = key
            assert result == sorted(keys, reverse=True)[:16]


def test_static_tiny_loop_covers_every_admitted_prefix_and_forced_hole():
    for threads in (128, 256):
        items = (128 + threads) // threads
        for valid in range(130):
            for excluded in [-1, *range(valid)]:
                ordinary = valid - (excluded >= 0)
                if ordinary > 128:
                    continue
                loaded = {
                    item * threads + lane
                    for item in range(items)
                    if item * threads < valid
                    for lane in range(threads)
                    if item * threads + lane < valid
                    and item * threads + lane != excluded
                }
                assert loaded == set(range(valid)) - {excluded}
        # This column is lost by a single 128-thread iteration but is ordinary
        # when the forced hole is in the interior of a 129-column prefix.
        assert 128 in {
            item * threads + lane
            for item in range(items)
            for lane in range(threads)
            if item * threads + lane < 129 and item * threads + lane != 64
        }


def test_full_row_does_not_make_public_prefill_capturable(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_current_stream_capturing=lambda: True)),
    )
    with pytest.raises(RuntimeError, match="prefill is not capturable"):
        c.select_prefill_candidates(
            None, None, plan=_plan(), live_blocks=0, arm=c.SelectArm.RADIX_FULL_ROW
        )


@pytest.mark.gpu
@pytest.mark.parametrize("arm, threads, cached_items", _VARIANTS)
@pytest.mark.parametrize("history", [0, 150000])
@pytest.mark.parametrize("rank", [0, 1])
def test_prefill_full_q1024_exact_c4_and_dispatch(
    cuda_device,
    arm,
    threads,
    cached_items,
    history,
    rank,
):
    import torch

    plan = _plan(rank=rank)
    counts, excluded, active = _prefill_rows(plan.token_capacity, history, rank)
    scores = _scores(counts, excluded, active)
    expected = _oracle(scores, counts, excluded, active)
    gpu_scores, geometry, out, partials = _buffers(
        cuda_device, plan, scores, counts, excluded, active
    )
    controls = dict(full_row_threads=threads, full_row_cached_items=cached_items)
    c.record_launches(True)
    try:
        # Intentionally under-report the obsolete hint. The public entry point
        # still launches from capacity and reads every device-valid ordinary
        # score across this entire Q1024 invocation in one selector call.
        c.select_prefill_candidates(
            gpu_scores,
            geometry,
            plan=plan,
            live_blocks=1,
            out=out,
            partials=partials,
            arm=arm,
            **controls,
        )
        launched = c.last_launch()
        actual_threads = c.last_launch_threads()
    finally:
        c.record_launches(False)
    assert (
        launched["symbol"]
        == c.planned_prefill_kernel(plan.scan_extent, arm=arm, **controls)["symbol"]
    )
    partitions = 1 if arm is c.SelectArm.RADIX_FULL_ROW else 2
    assert launched["grid"] == plan.token_capacity * 4 * partitions
    assert actual_threads == (threads if arm is c.SelectArm.RADIX_FULL_ROW else 128)
    torch.testing.assert_close(out.view(torch.int32).cpu(), expected, rtol=0, atol=0)
    if arm is c.SelectArm.RADIX_FULL_ROW:
        assert (partials.view(torch.int32).cpu() == _POISON).all()


@pytest.mark.gpu
@pytest.mark.parametrize("arm, threads, cached_items", _VARIANTS)
def test_prefill_above_full_row_capacity_preserves_exact_fallback(
    cuda_device,
    arm,
    threads,
    cached_items,
):
    import torch

    counts = [8193, 8192, 4097, 17, 16, 1, 0]
    excluded = [7, 8191, -1, 8, -1, 0, -1]
    active = [True] * len(counts)
    plan = _plan(tokens=len(counts), columns=8193)
    scores = _scores(counts, excluded, active)
    expected = _oracle(scores, counts, excluded, active)
    gpu_scores, geometry, out, partials = _buffers(
        cuda_device, plan, scores, counts, excluded, active
    )
    controls = dict(full_row_threads=threads, full_row_cached_items=cached_items)
    c.select_prefill_candidates(
        gpu_scores,
        geometry,
        plan=plan,
        live_blocks=8193,
        out=out,
        partials=partials,
        arm=arm,
        **controls,
    )
    torch.testing.assert_close(out.view(torch.int32).cpu(), expected, rtol=0, atol=0)
    assert (
        c.planned_prefill_kernel(8193, arm=arm, **controls)["symbol"]
        == (c.planned_prefill_kernel(8193, arm=c.SelectArm.RADIX_BOUNDED)["symbol"])
    )


@pytest.mark.gpu
@pytest.mark.parametrize("arm, threads, cached_items", _VARIANTS)
def test_validated_native_prefill_graph_reloads_device_prefix(
    cuda_device,
    arm,
    threads,
    cached_items,
):
    _check_native_prefill_graph(cuda_device, arm, threads, cached_items)


def _check_native_prefill_graph(
    cuda_device,
    arm,
    threads,
    cached_items,
    *,
    four_warp_finish=False,
):
    import torch

    plan = _plan(tokens=129, rank=1)
    states = []
    # Cross cached -> 500K streaming -> cached with unchanged buffers and
    # launch arguments, then retain the short/1536-boundary/maximum prefixes.
    for history in (
        150000,
        500000,
        150000,
        0,
        8256,
        16256,
        163777,
        180161,
        196416,
        196545,
        1048447,
    ):
        counts, excluded, active = _prefill_rows(plan.token_capacity, history, 1)
        scores = _scores(counts, excluded, active)
        expected = _oracle(scores, counts, excluded, active)
        buffers = _buffers(cuda_device, plan, scores, counts, excluded, active)
        states.append((buffers, expected))
    (gpu_scores, geometry, out, partials), _ = states[0]
    controls = dict(
        full_row_threads=threads,
        full_row_cached_items=cached_items,
        full_row_four_warp_finish=four_warp_finish,
    )
    c.select_prefill_candidates(
        gpu_scores,
        geometry,
        plan=plan,
        live_blocks=plan.scan_extent,
        out=out,
        partials=partials,
        arm=arm,
        **controls,
    )
    torch.cuda.synchronize()
    extension = c._build._select_ext()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        extension.select_prefill_candidates(
            gpu_scores,
            geometry.local_valid_blocks,
            geometry.forced_column,
            geometry.active_rows,
            out,
            partials,
            0,
            plan.use_pdl,
            plan.scan_extent,
            global_block_stride=1,
            arm=int(arm),
            **controls,
        )
    for (next_scores, next_geometry, _, _), expected in states:
        gpu_scores.copy_(next_scores)
        geometry.local_valid_blocks.copy_(next_geometry.local_valid_blocks)
        geometry.forced_column.copy_(next_geometry.forced_column)
        geometry.active_rows.copy_(next_geometry.active_rows)
        out.view(torch.int32).fill_(_POISON)
        partials.view(torch.int32).fill_(_POISON)
        graph.replay()
        torch.testing.assert_close(
            out.view(torch.int32).cpu(), expected, rtol=0, atol=0
        )


@pytest.mark.gpu
@pytest.mark.parametrize("cached_items", [0, 12])
@pytest.mark.parametrize("merge4", [False, True])
def test_four_warp_finish_tiny_candidate_boundaries(cuda_device, cached_items, merge4):
    import torch

    ordinary = [15, 16, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128]
    counts = [count + 1 for count in ordinary] + [(1 << 31) - 1]
    excluded = [
        count // 2 if row % 2 == 0 else count for row, count in enumerate(ordinary)
    ] + [(1 << 31) - 1]
    active = [True] * len(ordinary) + [False]
    plan = _plan(tokens=len(counts))
    scores = _scores(counts, excluded, active)
    # The final active row has valid=129 and an interior forced hole. Make
    # column128 a unique winner so dropping the second 128-thread iteration
    # changes the answer, rather than merely omitting a losing candidate.
    scores[len(ordinary) - 1, :, 128] = torch.tensor([math.inf, math.inf, 1.0, 2.0])
    expected = _oracle(scores, counts, excluded, active)
    gpu_scores, geometry, out, partials = _buffers(
        cuda_device, plan, scores, counts, excluded, active
    )
    controls = dict(
        full_row_threads=128,
        full_row_cached_items=cached_items,
        full_row_four_warp_finish=merge4,
    )
    c.record_launches(True)
    try:
        c.select_prefill_candidates(
            gpu_scores,
            geometry,
            plan=plan,
            live_blocks=1,
            out=out,
            partials=partials,
            arm=c.SelectArm.RADIX_FULL_ROW,
            **controls,
        )
        launched = c.last_launch()
        assert c.last_launch_threads() == 128
    finally:
        c.record_launches(False)
    selected = c.planned_prefill_kernel(
        _CAPACITY, arm=c.SelectArm.RADIX_FULL_ROW, **controls
    )["symbol"]
    other = c.planned_prefill_kernel(
        _CAPACITY,
        arm=c.SelectArm.RADIX_FULL_ROW,
        full_row_threads=128,
        full_row_cached_items=cached_items,
        full_row_four_warp_finish=not merge4,
    )["symbol"]
    assert selected != other and launched["symbol"] == selected
    torch.testing.assert_close(out.view(torch.int32).cpu(), expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("cached_items", [0, 12])
@pytest.mark.parametrize("merge4", [False, True])
def test_four_warp_finish_coarse_and_exact_boundaries(
    cuda_device, cached_items, merge4
):
    import torch

    boundary_sizes = [64, 65, 95, 96, 127, 128]
    counts = [1172] * len(boundary_sizes)
    excluded, active = [7] * len(counts), [True] * len(counts)
    scores = torch.full((len(counts), 4, 1172), -math.inf, dtype=torch.float32)
    ordinary_columns = torch.tensor([column for column in range(1172) if column != 7])
    for row, count in enumerate(boundary_sizes):
        indices = torch.arange(count)
        winners = ordinary_columns[(indices * 73 + row * 17) % len(ordinary_columns)]
        words = (0x3F800000 + ((indices * 37) % 127)).to(torch.int32)
        scores[row, 0, winners] = words.view(torch.float32)
        scores[row, 1, winners] = math.inf
        scores[row, 2, winners] = torch.where(indices % 2 == 0, 0.0, -0.0)
        # Head3 remains all -inf. Its tie crosses the coarse histogram and
        # refines the ID bytes; the non-aligned high ID origin creates a
        # 95-key first exact boundary after forced-column exclusion.
        scores[row, :, 7] = math.nan
    begin = (1 << 25) + 160
    expected = _oracle(scores, counts, excluded, active, begin=begin)
    plan = _plan(tokens=len(counts))
    gpu_scores, geometry, out, partials = _buffers(
        cuda_device, plan, scores, counts, excluded, active
    )
    geometry = c.CandidateGeometry(
        geometry.local_valid_blocks,
        geometry.forced_column,
        geometry.active_rows,
        scan_block_begin=begin,
    )
    c.select_prefill_candidates(
        gpu_scores,
        geometry,
        plan=plan,
        live_blocks=1172,
        out=out,
        partials=partials,
        arm=c.SelectArm.RADIX_FULL_ROW,
        full_row_threads=128,
        full_row_cached_items=cached_items,
        full_row_four_warp_finish=merge4,
    )
    torch.testing.assert_close(out.view(torch.int32).cpu(), expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("cached_items", [0, 12])
def test_four_warp_finish_native_graph_replay(cuda_device, cached_items):
    _check_native_prefill_graph(
        cuda_device,
        c.SelectArm.RADIX_FULL_ROW,
        128,
        cached_items,
        four_warp_finish=True,
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "threads, cached_items, merge4",
    [
        (128, 12, False),
        (128, 12, True),
        (256, 6, False),
    ],
)
def test_cached_item_initialization_at_each_late_slot_boundary(
    cuda_device,
    threads,
    cached_items,
    merge4,
):
    import torch

    counts = [1279, 1280, 1281, 1407, 1408, 1409, 1535, 1536, 1537]
    excluded = [
        live // 2 if row % 2 == 0 else live - 1 for row, live in enumerate(counts)
    ]
    active = [True] * len(counts)
    scores = _scores(counts, excluded, active)
    for row, live in enumerate(counts):
        scores[row, 0, :live] = torch.arange(live, dtype=torch.float32)
        if excluded[row] != live - 1:
            # The newly readable last slot must affect exact selection.
            scores[row, :, live - 1] = torch.tensor([math.inf, math.inf, 1.0, 2.0])
        scores[row, :, excluded[row]] = math.nan
    expected = _oracle(scores, counts, excluded, active)
    plan = _plan(tokens=len(counts))
    gpu_scores, geometry, out, partials = _buffers(
        cuda_device, plan, scores, counts, excluded, active
    )
    c.select_prefill_candidates(
        gpu_scores,
        geometry,
        plan=plan,
        live_blocks=1,
        out=out,
        partials=partials,
        arm=c.SelectArm.RADIX_FULL_ROW,
        full_row_threads=threads,
        full_row_cached_items=cached_items,
        full_row_four_warp_finish=merge4,
    )
    torch.testing.assert_close(out.view(torch.int32).cpu(), expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_native_prefill_rejects_unsupported_arm_and_noncapacity_full_row(cuda_device):
    import torch

    extension = c._build._select_ext()
    # Host-side failures precede even CUDA tensor validation, so no malformed
    # launch is enqueued and neither check poisons the CUDA context.
    cpu = torch.empty((1, 4, 8192))
    with pytest.raises(RuntimeError, match="prefill arm must be"):
        extension.select_prefill_candidates(
            cpu, cpu, cpu, cpu, cpu, cpu, 0, False, 8192, arm=0
        )
    with pytest.raises(RuntimeError, match="requires live_blocks == N"):
        extension.select_prefill_candidates(
            cpu, cpu, cpu, cpu, cpu, cpu, 0, False, 1172, arm=-3
        )
    with pytest.raises(
        RuntimeError, match="unsupported prefill full-row configuration"
    ):
        extension.select_prefill_candidates(
            cpu,
            cpu,
            cpu,
            cpu,
            cpu,
            cpu,
            0,
            False,
            8192,
            arm=-3,
            full_row_threads=128,
            full_row_cached_items=3,
        )
    with pytest.raises(RuntimeError, match="four-warp finish requires a 128-thread"):
        extension.select_prefill_candidates(
            cpu,
            cpu,
            cpu,
            cpu,
            cpu,
            cpu,
            0,
            False,
            8192,
            arm=-3,
            full_row_threads=256,
            full_row_cached_items=0,
            full_row_four_warp_finish=True,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "threads,cached_items,merge4",
    [
        (128, 12, False),
        (128, 12, True),
        (256, 6, False),
        (128, 0, False),
        (128, 0, True),
        (256, 0, False),
    ],
)
def test_coarse_candidates_preserve_sparse_exact_scores(
    cuda_device,
    threads,
    cached_items,
    merge4,
):
    """Place exact winners throughout streaming/cache tiles and poison unread cells.

    Exactly 16 scores exceed the background, forcing the coarse fast finish.
    Their magnitudes cover infinities, canonical zeros, distinct FP32 values
    inside one coarse bin, word/tile boundaries and the final live column.
    """
    import torch

    counts = [
        129,
        130,
        255,
        256,
        257,
        383,
        384,
        385,
        767,
        768,
        769,
        1151,
        1152,
        1153,
        1299,
        1300,
        1535,
        1536,
        1537,
        2047,
        2048,
        2049,
        3905,
        3906,
        3907,
        4035,
        4095,
        4096,
        4097,
        8191,
        8192,
    ]
    excluded = [count // 2 for count in counts]
    active = [True] * len(counts)
    counts.append((1 << 31) - 1)
    excluded.append((1 << 31) - 1)
    active.append(False)
    scores = torch.full((len(counts), 4, 8192), math.nan, dtype=torch.float32)
    for token, enabled in enumerate(active):
        if not enabled:
            continue
        valid, forced = counts[token], excluded[token]
        scores[token, :, :valid] = -1.0
        probes = [
            valid - 1,
            *[
                item * threads + item % 32
                for item in range((valid + threads - 1) // threads)
            ],
            *range(32),
        ]
        winners = list(
            dict.fromkeys(
                column for column in probes if column < valid and column != forced
            )
        )[:16]
        assert len(winners) == 16 and valid - 1 in winners
        scores[token, 0, winners] = torch.arange(16, dtype=torch.float32) + 1.0
        scores[token, 1, winners] = torch.arange(16, dtype=torch.float32) + 1.0
        scores[token, 1, winners[:4]] = math.inf
        scores[token, 2, winners] = torch.tensor([0.0, -0.0] * 8)
        scores[token, 3, winners] = (
            0x3F800001 + torch.arange(16, dtype=torch.int32)
        ).view(torch.float32)
        scores[token, :, forced] = math.nan
    expected = _oracle(scores, counts, excluded, active)
    plan = _plan(tokens=len(counts))
    gpu_scores, geometry, out, partials = _buffers(
        cuda_device, plan, scores, counts, excluded, active
    )
    c.select_prefill_candidates(
        gpu_scores,
        geometry,
        plan=plan,
        live_blocks=1,
        out=out,
        partials=partials,
        arm=c.SelectArm.RADIX_FULL_ROW,
        full_row_threads=threads,
        full_row_cached_items=cached_items,
        full_row_four_warp_finish=merge4,
    )
    torch.testing.assert_close(out.view(torch.int32).cpu(), expected, rtol=0, atol=0)
    assert (partials.view(torch.int32).cpu() == _POISON).all()
