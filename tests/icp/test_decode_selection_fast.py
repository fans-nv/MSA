"""Exact C4 gates for the optimized, whole-row decode selector.

The oracle sorts ordinary CPU FP32 values by score descending and global ID
ascending. It does not use another selector or the CUDA key implementation.
Every score outside the readable prefix, every inactive row, and every forced
score is NaN. Outputs and retained scratch are poisoned before reuse.

These are correctness tests, not timings: CUDA copies and synchronization are
intentional outside the selector's submission/capture path.
"""

from __future__ import annotations

import math
import pathlib
import struct
import subprocess
import sys

import pytest

from fmha_sm100.icp import candidates as c

_POISON = 0x5A5A5A5A
_INT32_MAX = (1 << 31) - 1


def _profile(page, rank):
    columns = 1048576 // page
    stride = page // 128
    begin = rank if stride == 2 else 0
    # Global history is 150000 for both profiles. These values are the number
    # of rank-local score columns, not an independent 75000-token request.
    live = 1172 if page == 128 else 586
    forced = 1171 if page == 128 else (585 if rank == 1 else -1)
    return columns, stride, begin, live, forced


def _score_word(value):
    # C5 requires either signed zero to be emitted as +0. Conversion of the
    # ID through float is deliberately absent, including IDs above 2**24.
    value = 0.0 if value == 0.0 else value
    return struct.unpack("<i", struct.pack("<f", value))[0]


def _expected_words(scores, counts, excluded, active, *, begin, stride):
    import torch

    expected = torch.empty((*scores.shape[:2], 16, 2), dtype=torch.int32)
    expected[..., 0] = _score_word(-math.inf)
    expected[..., 1] = -1
    for token, rows in enumerate(scores.tolist()):
        if not active[token]:
            continue
        for head, values in enumerate(rows):
            ordinary = []
            for column in range(counts[token]):
                if column == excluded[token]:
                    continue
                score = values[column]
                assert not math.isnan(score), "the oracle received a live NaN"
                ordinary.append((score, begin + stride * column))
            ordinary.sort(key=lambda pair: (-pair[0], pair[1]))
            for slot, (score, gid) in enumerate(ordinary[:16]):
                expected[token, head, slot, 0] = _score_word(score)
                expected[token, head, slot, 1] = gid
    return expected


def _case(device, *, columns, rank, begin, stride, counts, excluded, active, scores):
    import torch

    plan = c.DecodePlan(
        icp_degree=2,
        icp_rank=rank,
        num_heads_local=2,
        max_local_blocks=columns,
        token_capacity=len(counts),
    )
    workspace = c.allocate_workspace(plan, device)
    geometry = c.CandidateGeometry(
        torch.tensor(counts, dtype=torch.int32, device=device),
        torch.tensor(excluded, dtype=torch.int32, device=device),
        torch.tensor(active, dtype=torch.bool, device=device),
        begin,
        stride,
    )
    expected = _expected_words(
        scores, counts, excluded, active, begin=begin, stride=stride
    )
    return plan, workspace, geometry, scores.to(device), expected


def _poison(workspace):
    import torch

    workspace.candidates.view(torch.int32).fill_(_POISON)
    workspace.partials.view(torch.int32).fill_(_POISON)


def _run(plan, workspace, geometry, scores):
    return c.select_decode_candidates(
        scores,
        geometry,
        plan=plan,
        workspace=workspace,
        out=workspace.candidates,
        arm=c.SelectArm.RADIX_FULL_ROW,
    )


def _check(workspace, expected):
    import torch

    actual = workspace.candidates.view(torch.int32).cpu()
    assert not (actual == _POISON).any(), "an output word was not overwritten"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _adversarial_scores(columns, counts, excluded, active):
    import torch

    scores = torch.full((len(counts), 4, columns), math.nan, dtype=torch.float32)
    for token, live in enumerate(counts):
        if not active[token]:
            continue
        index = torch.arange(live, dtype=torch.int32)
        if token % 2 == 0:
            scores[token, 0, :live] = 1.0
            scores[token, 1, :live] = -math.inf
            # Alternating raw +0/-0 verifies both canonical score bytes and
            # the smaller-ID tie break, including a tied cutoff.
            scores[token, 2, :live] = torch.where(index % 2 == 0, 0.0, -0.0)
            scores[token, 3, :live] = math.inf
        else:
            scores[token, 0, :live] = ((index * 37 + token) % 29).float() - 14
            scores[token, 1, :live] = index.float()
            scores[token, 2, :live] = -index.float()
            # All values fall into the same coarse FP16 bin, but their FP32
            # mantissas still decide the exact Top16 before the ID tie break.
            words = 0x3F800000 + ((index * 73 + token) % 257)
            scores[token, 3, :live] = words.view(torch.float32)
        if excluded[token] >= 0:
            scores[token, :, excluded[token]] = math.nan
    return scores


@pytest.mark.gpu
@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("batch", [8, 16])
def test_fast_selector_150k_eagle3_shape(cuda_device, page, rank, batch):
    import torch

    columns, stride, begin, live, forced = _profile(page, rank)
    tokens = batch * 4
    counts, excluded, active = [live] * tokens, [forced] * tokens, [True] * tokens
    generator = torch.Generator().manual_seed(20260914 + batch + rank)
    scores = torch.full((tokens, 4, columns), math.nan, dtype=torch.float32)
    scores[..., :live] = torch.randn((tokens, 4, live), generator=generator)
    if forced >= 0:
        scores[..., forced] = math.nan
    plan, workspace, geometry, gpu_scores, expected = _case(
        cuda_device,
        columns=columns,
        rank=rank,
        begin=begin,
        stride=stride,
        counts=counts,
        excluded=excluded,
        active=active,
        scores=scores,
    )
    _poison(workspace)
    _run(plan, workspace, geometry, gpu_scores)
    _check(workspace, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("extent", ["150k", "capacity"])
def test_fast_selector_adversarial_c4_bytes(cuda_device, page, rank, extent):
    columns, stride, _, live, _ = _profile(page, rank)
    live = columns if extent == "capacity" else live
    # Exercise the int32 limit while retaining P256's owner residue. This also
    # catches a score/ID pair packed with a lossy numeric float conversion.
    begin = _INT32_MAX - stride * (columns - 1)
    if stride == 2 and rank == 0:
        begin -= 1
    counts = [live, live, 17, 16, 15, 1, 0, _INT32_MAX]
    excluded = [7, live // 2, 8, -1, 0, 0, -1, _INT32_MAX]
    active = [True] * 7 + [False]
    scores = _adversarial_scores(columns, counts, excluded, active)
    plan, workspace, geometry, gpu_scores, expected = _case(
        cuda_device,
        columns=columns,
        rank=rank,
        begin=begin,
        stride=stride,
        counts=counts,
        excluded=excluded,
        active=active,
        scores=scores,
    )
    _poison(workspace)
    _run(plan, workspace, geometry, gpu_scores)
    _check(workspace, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("candidate_count", [31, 32, 33, 63, 64, 65])
def test_fast_selector_coarse_finish_around_one_warp(
    cuda_device,
    page,
    candidate_count,
):
    import torch

    columns, stride, begin, live, _ = _profile(page, 1)
    # A small group at one exactly representable score sits above a large
    # -inf background. Thus the whole boundary group must survive the coarse
    # pass, and its exact canonical Top16 crosses both the 31/32/33 and
    # 63/64/65 finish boundaries.
    # Signed powers of two sweep all 16 coarse-histogram warp chunks; this
    # also checks threshold locations that all-equal fallback rows cannot.
    thresholds = (
        -(2.0**15),
        -(2.0**12),
        -(2.0**8),
        -(2.0**4),
        -1.0,
        -(2.0**-4),
        -(2.0**-8),
        -(2.0**-12),
        0.0,
        2.0**-8,
        2.0**-4,
        1.0,
        2.0**4,
        2.0**8,
        2.0**12,
        2.0**15,
    )
    tokens = len(thresholds)
    counts = [live] * tokens
    excluded = [live // 2] * tokens
    active = [True] * tokens
    scores = torch.full((tokens, 4, columns), math.nan, dtype=torch.float32)
    scores[..., :live] = -math.inf
    for token, threshold in enumerate(thresholds):
        for head in range(4):
            # 37 is coprime to both 586 and 1172: spread the tied candidates
            # across input warps/chunks in different orders for every head.
            selected = [
                (i * 37 + 11 * token + 53 * head) % live
                for i in range(live)
                if (i * 37 + 11 * token + 53 * head) % live != excluded[token]
            ][:candidate_count]
            assert len(set(selected)) == candidate_count
            scores[token, head, selected] = threshold
            if threshold == 0.0:
                scores[token, head, selected[::2]] = -0.0
        scores[token, :, excluded[token]] = math.nan
    plan, workspace, geometry, gpu_scores, expected = _case(
        cuda_device,
        columns=columns,
        rank=1,
        begin=begin,
        stride=stride,
        counts=counts,
        excluded=excluded,
        active=active,
        scores=scores,
    )
    _poison(workspace)
    _run(plan, workspace, geometry, gpu_scores)
    _check(workspace, expected)


def _prefix_boundary_fixture(columns, counts):
    import torch

    excluded = [counts[0] // 2, counts[1] - 1]
    scores = torch.full((2, 4, columns), math.nan, dtype=torch.float32)
    for token, live in enumerate(counts):
        index = torch.arange(live, dtype=torch.int32)
        for head in range(4):
            scores[token, head, :live] = (
                ((index * (head + 1) + token) % 13) - 6
            ).float()
            last_ordinary = live - 1 if token == 0 else live - 2
            # The last ordinary column must win, with observable FP32 low
            # mantissa bits. A dropped column or a cached rounded score fails.
            scores[token, head, last_ordinary] = 16.0 + (head + 1) * (2.0**-19)
        scores[token, :, excluded[token]] = math.nan
    return scores, excluded


@pytest.mark.gpu
@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("live", [1535, 1536, 1537])
def test_fast_selector_prefix_1536_boundary(cuda_device, page, live):
    columns, stride, begin, _, _ = _profile(page, 1)
    counts, active = [live, live], [True, True]
    scores, excluded = _prefix_boundary_fixture(columns, counts)
    plan, workspace, geometry, gpu_scores, expected = _case(
        cuda_device,
        columns=columns,
        rank=1,
        begin=begin,
        stride=stride,
        counts=counts,
        excluded=excluded,
        active=active,
        scores=scores,
    )
    _poison(workspace)
    _run(plan, workspace, geometry, gpu_scores)
    _check(workspace, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("page", [128, 256])
def test_fast_selector_graph_replays_across_prefix_1536(cuda_device, page):
    import torch

    columns, stride, begin, _, _ = _profile(page, 1)
    stages = ([1535, 1537], [1536, 1535], [1537, 1536], [1535, 1537])
    active = [True, True]
    scores, excluded = _prefix_boundary_fixture(columns, stages[0])
    plan, workspace, geometry, gpu_scores, _ = _case(
        cuda_device,
        columns=columns,
        rank=1,
        begin=begin,
        stride=stride,
        counts=stages[0],
        excluded=excluded,
        active=active,
        scores=scores,
    )
    _run(plan, workspace, geometry, gpu_scores)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(plan, workspace, geometry, gpu_scores)
    for counts in stages:
        scores, excluded = _prefix_boundary_fixture(columns, counts)
        expected = _expected_words(
            scores, counts, excluded, active, begin=begin, stride=stride
        )
        gpu_scores.copy_(scores)
        geometry.local_valid_blocks.copy_(torch.tensor(counts, dtype=torch.int32))
        geometry.forced_column.copy_(torch.tensor(excluded, dtype=torch.int32))
        _poison(workspace)
        graph.replay()
        _check(workspace, expected)


def _forbid_host_reads(monkeypatch):
    import torch

    def refuse(*_args, **_kwargs):
        raise AssertionError("selector submission attempted a device readback")

    for attribute in (
        "item",
        "tolist",
        "cpu",
        "numpy",
        "__int__",
        "__float__",
        "__bool__",
        "__index__",
    ):
        monkeypatch.setattr(torch.Tensor, attribute, refuse, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", refuse)
    monkeypatch.setattr(torch.cuda.Stream, "synchronize", refuse, raising=False)
    monkeypatch.setattr(torch.cuda.Event, "synchronize", refuse, raising=False)


@pytest.mark.gpu
@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("rank", [0, 1])
def test_fast_selector_replay_changes_device_prefix_without_host_reads(
    cuda_device,
    monkeypatch,
    page,
    rank,
):
    import torch

    columns, stride, begin, live, _ = _profile(page, rank)
    stages = (
        ([0, 1, 15, 16, 17, 128, 129, -123], [True] * 7 + [False]),
        ([live, live - 1, 512, 513, live, 7, live, live], [True] * 8),
        (
            [columns, columns - 1, min(4097, columns), 2049, 17, 1, 0, columns],
            [True] * 8,
        ),
        (
            [17, -123, 0, 1, 15, 16, 129, 128],
            [True, False, True, True, True, True, True, True],
        ),
    )

    def fixture(counts, active):
        excluded = [
            count // 2 if enabled and count else -1
            for count, enabled in zip(counts, active)
        ]
        scores = _adversarial_scores(columns, counts, excluded, active)
        expected = _expected_words(
            scores, counts, excluded, active, begin=begin, stride=stride
        )
        return scores, excluded, expected

    counts, active = stages[0]
    scores, excluded, _ = fixture(counts, active)
    plan, workspace, geometry, gpu_scores, _ = _case(
        cuda_device,
        columns=columns,
        rank=rank,
        begin=begin,
        stride=stride,
        counts=counts,
        excluded=excluded,
        active=active,
        scores=scores,
    )
    _run(plan, workspace, geometry, gpu_scores)  # JIT and first load are startup.
    torch.cuda.synchronize()

    with monkeypatch.context() as blocked:
        _forbid_host_reads(blocked)
        with pytest.raises(AssertionError, match="device readback"):
            int(geometry.local_valid_blocks[0])
        _run(plan, workspace, geometry, gpu_scores)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Only submission is under this guard; CUDA graph setup is allowed to
        # synchronize its streams before capture begins.
        with monkeypatch.context() as blocked:
            _forbid_host_reads(blocked)
            _run(plan, workspace, geometry, gpu_scores)

    for counts, active in stages:
        scores, excluded, expected = fixture(counts, active)
        gpu_scores.copy_(scores)
        geometry.local_valid_blocks.copy_(torch.tensor(counts, dtype=torch.int32))
        geometry.forced_column.copy_(torch.tensor(excluded, dtype=torch.int32))
        geometry.active_rows.copy_(torch.tensor(active, dtype=torch.bool))
        _poison(workspace)
        graph.replay()
        _check(workspace, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("columns", [1, 17, 127, 129, 513, 2049, 4097, 8193])
def test_fast_selector_capacity_edges_and_partitioned_fallback(cuda_device, columns):
    counts, excluded, active = [columns, columns, -123], [-1, -1, -1], [True] * 2
    active.append(False)
    scores = _adversarial_scores(columns, counts, excluded, active)
    # The final column must win, including the first column of a third
    # partition at N=8193. A truncated fast path cannot pass this check.
    scores[1, :, columns - 1] = 1e30
    plan, workspace, geometry, gpu_scores, expected = _case(
        cuda_device,
        columns=columns,
        rank=1,
        begin=257,
        stride=2,
        counts=counts,
        excluded=excluded,
        active=active,
        scores=scores,
    )
    _poison(workspace)
    _run(plan, workspace, geometry, gpu_scores)
    _check(workspace, expected)


_NAN_TRAP = """
import sys
sys.path.insert(0, {root!r})
import torch
from fmha_sm100.icp import candidates as c

device = torch.device("cuda", {device})
torch.cuda.set_device(device)
plan = c.DecodePlan(icp_degree=2, icp_rank=1, num_heads_local=2,
                    max_local_blocks={columns}, token_capacity=1)
workspace = c.allocate_workspace(plan, device)
scores = torch.full((1, 4, {columns}), float("nan"), device=device)
scores[..., :{live}] = 1.0
geometry = c.CandidateGeometry(
    torch.full((1,), {live}, dtype=torch.int32, device=device),
    torch.zeros((1,), dtype=torch.int32, device=device),
    torch.ones((1,), dtype=torch.bool, device=device), 1, 2)
scores[..., 0] = float("nan")  # Excluded NaNs must remain unread.

def run():
    c.select_decode_candidates(scores, geometry, plan=plan, workspace=workspace,
                               out=workspace.candidates,
                               arm=c.SelectArm.RADIX_FULL_ROW)

run()
torch.cuda.synchronize()
print("NUMERIC_TRAP_READY", flush=True)
scores[0, 3, {live} - 1] = float("nan")
run()
torch.cuda.synchronize()
print("NO TRAP: a NaN ordinary score was accepted", flush=True)
"""


@pytest.mark.gpu
@pytest.mark.parametrize("page", [128, 256])
def test_fast_selector_rejects_nan_inside_ordinary_prefix(cuda_device, page):
    columns, _, _, live, _ = _profile(page, 1)
    root = str(pathlib.Path(c.__file__).resolve().parents[2])
    finished = subprocess.run(
        [
            sys.executable,
            "-c",
            _NAN_TRAP.format(
                root=root, device=cuda_device.index, columns=columns, live=live
            ),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = finished.stdout + finished.stderr
    assert "NUMERIC_TRAP_READY" in output, output[-4000:]
    assert "NO TRAP" not in output, output[-4000:]
    assert finished.returncode != 0, output[-4000:]
    assert "device-side assert" in output.lower(), output[-4000:]
