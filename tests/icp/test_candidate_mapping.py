"""Affine block IDs for both TP2 physical-page profiles.

CPU tests cover metadata/dispatch and independent partition geometry. GPU tests
exercise actual selectors with undefined score tails and remain separately
marked; CPU success is not a claim about CUDA execution.
"""

from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

import pytest

import fmha_sm100.icp
from fmha_sm100.icp import candidates as c


def _plan(plan_type=c.DecodePlan, *, columns=4096, rank=1):
    return plan_type(
        icp_degree=2,
        icp_rank=rank,
        num_heads_local=2,
        max_local_blocks=columns,
        token_capacity=4,
    )


def _geometry(*, begin=1, stride=2):
    return c.CandidateGeometry(
        "counts", "forced", "active", scan_block_begin=begin, global_block_stride=stride
    )


def test_mapping_capability_is_available_without_loading_a_kernel():
    assert fmha_sm100.icp.CANDIDATE_MAPPING_ABI_VERSION == 2
    assert fmha_sm100.icp.SUPPORTED_GLOBAL_BLOCK_STRIDES == (1, 2)


@pytest.mark.parametrize("stride", [1, 2])
def test_metadata_binds_the_mapping_and_preserves_legacy_default(stride):
    metadata = SimpleNamespace(
        topk_num_valid_pages="counts",
        icp_forced_column_v1="forced",
        icp_active_rows="active",
        icp_scan_block_begin_v1=17,
    )
    if stride == 2:
        metadata.icp_global_block_stride_v1 = stride
    geometry = c.CandidateGeometry.from_metadata(metadata)
    assert geometry.global_block_stride == stride
    assert geometry.scan_block_begin == 17
    assert geometry.local_valid_blocks is metadata.topk_num_valid_pages


class _DeviceScalar:
    def __int__(self):
        raise AssertionError("device scalar was read on the host")


@pytest.mark.parametrize("name", ["scan_block_begin", "global_block_stride"])
@pytest.mark.parametrize("value", [True, 1.0, _DeviceScalar()])
def test_mapping_rejects_non_host_integers_without_converting_them(name, value):
    with pytest.raises(TypeError, match="host int"):
        dataclasses.replace(_geometry(), **{name: value})


@pytest.mark.parametrize("stride", [-1, 0, 3, 4])
def test_unsupported_strides_fail_explicitly(stride):
    with pytest.raises(ValueError, match="1 or 2"):
        _geometry(stride=stride)


@pytest.mark.parametrize("begin", [-1, 2**31])
def test_invalid_logical_origins_fail_explicitly(begin):
    with pytest.raises(ValueError, match="non-negative int32"):
        _geometry(begin=begin)


def test_stride_is_included_in_int32_capacity_validation():
    torch = pytest.importorskip("torch")
    scores = torch.empty((4, 4, 2))
    with pytest.raises(ValueError, match="int32 capacity"):
        c._validate(
            scores,
            _geometry(begin=2**31 - 2),
            plan=_plan(columns=2),
            out=None,
            partials=None,
            partial_shape=(),
        )


@pytest.mark.parametrize("arm", list(c.SelectArm))
@pytest.mark.parametrize("stride", [1, 2])
def test_every_decode_arm_passes_the_same_mapping(monkeypatch, arm, stride):
    calls = []

    def entry(*args, **kwargs):
        calls.append((args, kwargs))

    extension = SimpleNamespace(
        **{
            name: entry
            for name in (
                "select_local_candidates",
                "select_local_candidates_radix",
                "select_local_candidates_radix_sr",
                "select_local_candidates_capped",
                "select_local_candidates_full_row",
            )
        }
    )
    plan = _plan()
    # Retained partials already sized to the token count are passed unsliced.
    partials = SimpleNamespace(shape=(4, 4, 2))
    workspace = SimpleNamespace(plan=plan, _extension=extension, partials=partials)
    scores = SimpleNamespace(shape=(4, 4, 4096))
    geometry = _geometry(begin=19, stride=stride)
    monkeypatch.setattr(c, "_validate", lambda *args, **kwargs: None)
    out = object()
    assert (
        c.select_decode_candidates(
            scores, geometry, plan=plan, workspace=workspace, out=out, arm=arm
        )
        is out
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1:4] == ("counts", "forced", "active")
    assert args[4] is out and args[5] is partials
    assert args[6] == 19
    assert kwargs["global_block_stride"] == stride


@pytest.mark.parametrize("stride", [1, 2])
def test_prefill_passes_mapping_without_allocating_outputs(monkeypatch, stride):
    torch = pytest.importorskip("torch")
    calls = []
    extension = SimpleNamespace(
        select_prefill_candidates=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setattr(c._build, "_select_ext", lambda: extension)
    monkeypatch.setattr(c, "_validate", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    out, partials = object(), object()
    c.select_prefill_candidates(
        SimpleNamespace(shape=(4, 4, 4096)),
        _geometry(begin=23, stride=stride),
        plan=_plan(c.PrefillPlan),
        live_blocks=19,
        out=out,
        partials=partials,
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[4] is out and args[5] is partials
    assert args[6] == 23 and args[8] == 4096
    assert kwargs["global_block_stride"] == stride


def test_compact_1m_capacity_does_not_need_selector_partials():
    assert _plan(columns=1048576 // 128).partial_count == 2
    compact = _plan(columns=1048576 // 256)
    assert compact.partial_count == 0
    assert compact.partial_shape[2] == 0
    assert c.prefill_launch(4096).partitions == 0


def _visible_blocks(page_tokens, rank, query, length):
    # Independent token enumeration, without using the prefix formula.
    rows = page_tokens // 2
    return {
        token // 128
        for token in range(max(0, min(query + 1, length)))
        if (token % page_tokens) // rows == rank
    }


@pytest.mark.parametrize("page_tokens", [128, 256])
@pytest.mark.parametrize("rank", [0, 1])
def test_prefix_and_forced_inverse_match_token_ownership(page_tokens, rank):
    stride = page_tokens // 128
    rows = page_tokens // 2
    for query in range(513):
        for length in (0, max(0, query - 3), query + 1, 600):
            visible = _visible_blocks(page_tokens, rank, query, length)
            upper = max(0, min(length, query + 1))
            nlocal = upper // page_tokens + (upper % page_tokens > rank * rows)
            for compact_begin in (0, 1, 3):
                begin = compact_begin * stride + (rank if stride == 2 else 0)
                nvalid = max(0, nlocal - compact_begin)
                generated = {begin + stride * col for col in range(nvalid)}
                assert generated == {b for b in visible if b >= begin}
                forced = query // 128
                delta = forced - begin
                column = (
                    delta // stride
                    if delta >= 0 and delta % stride == 0 and delta // stride < nvalid
                    else -1
                )
                assert (column >= 0) == (forced in generated)
                if column >= 0:
                    assert begin + stride * column == forced


@pytest.mark.parametrize("page_tokens", [128, 256])
def test_local_top16_union_equals_global_ordinary_top15(page_tokens):
    rng = random.Random(2718)
    stride = page_tokens // 128
    for blocks in (1, 2, 15, 16, 17, 33, 129):
        for tied in (False, True):
            per_rank = []
            for rank in (0, 1):
                ids = range(rank, blocks, 2) if stride == 2 else range(blocks)
                per_rank.append(
                    {b: (1.0 if tied else rng.randrange(-9, 10)) for b in ids}
                )
            forced = blocks - 1
            global_scores = {
                b: max(row[b] for row in per_rank if b in row) for b in range(blocks)
            }

            def order(item):
                return (-item[1], item[0])

            expected = sorted(
                ((b, s) for b, s in global_scores.items() if b != forced), key=order
            )[:15]
            survivors = {}
            for row in per_rank:
                local = sorted(
                    ((b, s) for b, s in row.items() if b != forced), key=order
                )[:16]
                for block, score in local:
                    survivors[block] = max(survivors.get(block, -float("inf")), score)
            assert sorted(survivors.items(), key=order)[:15] == expected


@pytest.mark.gpu
@pytest.mark.parametrize("arm", list(c.SelectArm))
@pytest.mark.parametrize("columns", [4096, 8192])
def test_affine_ids_and_uninitialized_tail_isolation(cuda_device, arm, columns):
    import torch

    plan = _plan(columns=columns)
    workspace = c.allocate_workspace(plan, cuda_device)
    # Undefined tails and an excluded forced score must never be loaded.
    scores = torch.full((4, 4, columns), float("nan"), device=cuda_device)
    counts = [23, 0, 7, 41]
    excluded = [11, -1, -1, 40]
    active = [True, True, False, True]
    for token, count in enumerate(counts):
        if not active[token]:
            continue
        for head in range(4):
            scores[token, head, :count] = torch.arange(count, device=cuda_device) % 5
            if excluded[token] >= 0:
                scores[token, head, excluded[token]] = float("nan")
    geometry = c.CandidateGeometry(
        torch.tensor(counts, dtype=torch.int32, device=cuda_device),
        torch.tensor(excluded, dtype=torch.int32, device=cuda_device),
        torch.tensor(active, dtype=torch.bool, device=cuda_device),
        scan_block_begin=19,
        global_block_stride=2,
    )
    c.select_decode_candidates(
        scores,
        geometry,
        plan=plan,
        workspace=workspace,
        out=workspace.candidates,
        arm=arm,
    )
    result = workspace.candidates.cpu()
    ids = result.view(torch.int32)[..., 1]
    for token, count in enumerate(counts):
        ordinary = (
            [col for col in range(count) if col != excluded[token]]
            if active[token]
            else []
        )
        ordinary.sort(key=lambda col: (-(col % 5), 19 + 2 * col))
        expected = [19 + 2 * col for col in ordinary[:16]]
        expected += [-1] * (16 - len(expected))
        for head in range(4):
            assert ids[token, head].tolist() == expected
            for slot, col in enumerate(ordinary[:16]):
                assert result[token, head, slot, 0].item() == col % 5


@pytest.mark.gpu
def test_prefill_affine_mapping_matches_decode(cuda_device):
    import torch

    decode = _plan(columns=4096)
    prefill = _plan(c.PrefillPlan, columns=4096)
    workspace = c.allocate_workspace(decode, cuda_device)
    scores = torch.full((4, 4, 4096), float("nan"), device=cuda_device)
    scores[:, :, :31] = torch.arange(31, device=cuda_device)
    geometry = c.CandidateGeometry(
        torch.full((4,), 31, dtype=torch.int32, device=cuda_device),
        torch.full((4,), 17, dtype=torch.int32, device=cuda_device),
        torch.ones(4, dtype=torch.bool, device=cuda_device),
        scan_block_begin=64,
        global_block_stride=2,
    )
    c.select_decode_candidates(
        scores, geometry, plan=decode, workspace=workspace, out=workspace.candidates
    )
    out = torch.empty_like(workspace.candidates)
    partials = torch.empty(prefill.partial_shape(4, 4096), device=cuda_device)
    c.select_prefill_candidates(
        scores, geometry, plan=prefill, live_blocks=31, out=out, partials=partials
    )
    assert torch.equal(out.view(torch.int32), workspace.candidates.view(torch.int32))


@pytest.mark.gpu
def test_affine_full_replay_uses_updated_prefix_without_reading_tail(cuda_device):
    import torch

    plan = _plan(columns=4096)
    workspace = c.allocate_workspace(plan, cuda_device)
    assert workspace._extension.candidate_mapping_abi_version == 2
    scores = torch.full((4, 4, 4096), float("nan"), device=cuda_device)
    scores[:, :, :31] = torch.arange(31, device=cuda_device)
    counts = torch.full((4,), 31, dtype=torch.int32, device=cuda_device)
    forced = torch.full((4,), -1, dtype=torch.int32, device=cuda_device)
    active = torch.ones(4, dtype=torch.bool, device=cuda_device)
    geometry = c.CandidateGeometry(counts, forced, active, 1, 2)

    def run():
        c.select_decode_candidates(
            scores, geometry, plan=plan, workspace=workspace, out=workspace.candidates
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for live in (7, 0, 19):
        scores.fill_(float("nan"))
        scores[:, :, :live] = torch.arange(live, device=cuda_device)
        counts.fill_(live)
        graph.replay()
        ids = workspace.candidates.view(torch.int32)[..., 1].cpu()
        expected = [1 + 2 * col for col in reversed(range(live))][:16]
        expected += [-1] * (16 - len(expected))
        assert ids.tolist() == [[expected] * 4] * 4
