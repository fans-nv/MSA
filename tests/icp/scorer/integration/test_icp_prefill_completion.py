# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Prefill W5 completion: row ownership, vector boundaries and strided fallback."""

import pytest
import torch

from icp_device_plan_harness import build_device_plan
from icp_prefill_fixtures import make_prefill_fixture


CASES = [
    pytest.param(64, 0, 128, "THK", id="single-wg-fallback"),
    pytest.param(65, 0, 128, "THK", id="q65-partial-warp"),
    pytest.param(129, 63, 128, "THK", id="q129-offset63"),
    pytest.param(257, 127, 128, "THK", id="prefix-three"),
    pytest.param(513, 112, 128, "THK", id="prefix-five"),
    pytest.param(65, 0, 5, "THK", id="narrow-last-group"),
    pytest.param(129, 63, 127, "THK", id="vector-tail-three"),
    pytest.param(129, 63, 129, "THK", id="vector-tail-one"),
    pytest.param(129, 63, 130, "THK", id="retained-parent-pitch"),
    pytest.param(129, 63, 131, "THK", id="vector-tail-three-above128"),
    pytest.param(129, 63, 511, "THK", id="below-cooperative-threshold"),
    pytest.param(129, 63, 512, "THK", id="cooperative-threshold"),
    pytest.param(129, 63, 513, "THK", id="cooperative-tail-one"),
    pytest.param(129, 63, 514, "THK", id="cooperative-tail-two"),
    pytest.param(129, 63, 515, "THK", id="cooperative-tail-three"),
    pytest.param(129, 63, 512, "HKT", id="cooperative-fallback-legacy-strides"),
    pytest.param(129, 63, 512, "THK-step2", id="cooperative-fallback-column-stride"),
    pytest.param(129, 63, 512, "THK-offset1", id="cooperative-fallback-row-base"),
    pytest.param(129, 63, 512, "THK-row-gap", id="cooperative-fallback-row-pitch"),
    pytest.param(129, 63, 128, "HKT", id="legacy-strides"),
    pytest.param(129, 63, 128, "THK-step2", id="nonunit-column-stride"),
    pytest.param(129, 63, 128, "THK-offset1", id="misaligned-row-base"),
    pytest.param(129, 63, 128, "THK-row-gap", id="misaligned-row-pitch"),
    pytest.param(1024, 150000, 1280, "THK", id="long-prefix-boundary"),
]


@pytest.mark.parametrize("query_len,history,columns,layout", CASES)
@pytest.mark.parametrize("rank", [0, 1])
def test_prefill_completion(query_len, history, columns, layout, rank):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires the SM100/SM103 FMHA target")
    from fmha_sm100.icp.scorer.prefill.api import _fmha_sm100

    device = torch.device("cuda", torch.cuda.current_device())
    fixture = make_prefill_fixture(query_len=query_len, history=history, seed=47)
    data = fixture.materialize(rank, device)
    plan, _store, _ = build_device_plan(
        query_lens=[query_len],
        seq_lens=[fixture.total_len],
        table=data["table"],
        rank=rank,
        device=device,
    )
    plan["max_k_tiles"] = columns
    capacity = ((columns + 31) // 32) * 32 + 32
    if layout == "THK-row-gap":
        row_pitch = 4 * capacity + 1
        shape = ((query_len + 2) * row_pitch,)
        slices = None
    elif layout == "HKT":
        shape = (4, capacity, query_len + 2)
        slices = (slice(None), slice(0, columns), slice(0, query_len))
    elif layout == "THK-step2":
        shape = (query_len + 2, 4, capacity * 2)
        slices = (slice(0, query_len), slice(None), slice(0, columns * 2, 2))
    elif layout == "THK-offset1":
        shape = (query_len + 2, 4, capacity)
        slices = (slice(0, query_len), slice(None), slice(1, columns + 1))
    else:
        shape = (query_len + 2, 4, capacity)
        slices = (slice(0, query_len), slice(None), slice(0, columns))

    def output_view(storage):
        if layout == "THK-row-gap":
            return storage.as_strided((query_len, 4, columns), (row_pitch, capacity, 1))
        return storage[slices]

    score_storage = torch.full(shape, float("nan"), dtype=torch.float32, device=device)
    valid_storage = torch.full(shape, 9, dtype=torch.uint8, device=device)
    guard = torch.ones(shape, dtype=torch.bool, device=device)
    output_view(guard).fill_(False)
    score_before = score_storage.view(torch.int32).clone()
    score, valid = output_view(score_storage), output_view(valid_storage)

    _fmha_sm100(
        data["q"],
        data["local_keys"],
        data["local_keys"],
        plan,
        kv_indices=data["table"],
        output_o=False,
        output_maxscore=True,
        max_score=score,
        valid_score=valid,
        icp_c=2,
        icp_rank=rank,
    )
    torch.cuda.synchronize()
    if layout == "HKT":
        score, valid = score.permute(2, 0, 1), valid.permute(2, 0, 1)
    col = torch.arange(columns, device=device)
    expected_valid = col[None, None] < data["geometry"]["prefix"][:, None, None]
    expected_valid = expected_valid.expand_as(valid)
    assert torch.equal(valid, expected_valid.to(torch.uint8))
    assert not bool(torch.isnan(score).any())
    assert bool(torch.isneginf(score[~expected_valid]).all())
    assert torch.equal(score_storage.view(torch.int32)[guard], score_before[guard])
    assert bool((valid_storage[guard] == 9).all())

    # Independent encoded-input dots check valid scores on both sides of the
    # last Q tile; producer repeatability alone would miss a shared overwrite.
    rows = sorted({0, min(128, query_len - 1), query_len - 1})
    _, expected, reference_valid = fixture.numerical_reference(rank, query_rows=rows)
    actual = score[rows, :, : fixture.live_blocks].cpu()
    readable = reference_valid[:, None].expand_as(expected)
    torch.testing.assert_close(
        actual[readable], expected[readable], rtol=2e-5, atol=1e-4
    )


@pytest.mark.parametrize(
    "query_len,history,row_begin",
    [
        pytest.param(129, 63, 1, id="q129-origin1"),
        pytest.param(257, 127, 37, id="q257-origin37"),
        pytest.param(513, 112, 128, id="q513-origin128"),
        pytest.param(1024, 150000, 300, id="long-prefix-origin300"),
    ],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_prefill_completion_mid_slot_window(query_len, history, row_begin, rank):
    """ABI 3: the FMHA window starts at row_begin inside the chunk slot."""
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires the SM100/SM103 FMHA target")
    from fmha_sm100.icp.scorer.prefill.api import _fmha_sm100

    device = torch.device("cuda", torch.cuda.current_device())
    fixture = make_prefill_fixture(query_len=query_len, history=history, seed=47)
    data = fixture.materialize(rank, device)
    plan, _store, model = build_device_plan(
        query_lens=[query_len],
        seq_lens=[fixture.total_len],
        table=data["table"],
        rank=rank,
        device=device,
        row_begin=row_begin,
    )
    rows = query_len - row_begin
    assert model["header"][3] == row_begin
    assert plan["qo_segment_lens"].cpu().tolist() == [rows]
    assert plan["qo_offset"].cpu().tolist() == [history + row_begin]
    columns = fixture.live_blocks + 3
    plan["max_k_tiles"] = columns
    score_storage = torch.full((rows + 2, 4, columns + 8), float("nan"), device=device)
    valid_storage = torch.full(
        (rows + 2, 4, columns + 8), 9, dtype=torch.uint8, device=device
    )
    score, valid = score_storage[:rows, :, :columns], valid_storage[:rows, :, :columns]
    _fmha_sm100(
        data["q"][row_begin:],
        data["local_keys"],
        data["local_keys"],
        plan,
        kv_indices=data["table"],
        output_o=False,
        output_maxscore=True,
        max_score=score,
        valid_score=valid,
        icp_c=2,
        icp_rank=rank,
    )
    torch.cuda.synchronize()
    col = torch.arange(columns, device=device)
    prefix = data["geometry"]["prefix"][row_begin:]
    expected_valid = (col[None, None] < prefix[:, None, None]).expand_as(valid)
    assert torch.equal(valid, expected_valid.to(torch.uint8))
    assert not bool(torch.isnan(score).any())
    assert bool(torch.isneginf(score[~expected_valid]).all())
    assert bool(torch.isnan(score_storage[rows:]).all())
    assert bool((valid_storage[:, :, columns:] == 9).all())
    check = sorted({row_begin, min(row_begin + 128, query_len - 1), query_len - 1})
    _, expected, reference_valid = fixture.numerical_reference(rank, query_rows=check)
    actual = score[[r - row_begin for r in check], :, : fixture.live_blocks].cpu()
    readable = reference_valid[:, None].expand_as(expected)
    torch.testing.assert_close(
        actual[readable], expected[readable], rtol=2e-5, atol=1e-4
    )
