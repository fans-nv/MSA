# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Persistent scorer pipeline epochs, ragged work and repeated graph reuse.

The large Q points give a CTA multiple work items. The 500K origin also changes
rank-specific trip counts at alternating Q128 boundaries. Poisoned output
replay checks exercise the final per-work signal, W5 and TMEM lifetime together.
"""

import pytest
import torch

from icp_device_plan_harness import build_device_plan
from icp_prefill_fixtures import make_prefill_fixture


@pytest.mark.parametrize(
    "query_len,history",
    [
        pytest.param(65, 0, id="partial-active-warps"),
        pytest.param(65, 500000, id="unequal-rank-trip-count"),
        pytest.param(8193, 150000, id="persistent-ragged-150k"),
        pytest.param(16385, 500000, id="persistent-ragged-500k"),
    ],
)
@pytest.mark.parametrize("rank", [0, 1])
def test_prefill_pipeline_repeated_graph(query_len, history, rank):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires the SM100/SM103 FMHA target")
    from fmha_sm100.icp.scorer.prefill.api import _fmha_sm100

    device = torch.device("cuda", torch.cuda.current_device())
    fixture = make_prefill_fixture(query_len=query_len, history=history, seed=59)
    data = fixture.materialize(rank, device)
    plan, _store, _ = build_device_plan(
        query_lens=[query_len],
        seq_lens=[fixture.total_len],
        table=data["table"],
        rank=rank,
        device=device,
    )
    assert plan["num_kv_splits"] == 1
    if query_len >= 8193:
        ranges = plan["packed_work_range"].cpu().tolist()
        assert max((value >> 32) - (value & 0xFFFFFFFF) for value in ranges) > 1
    columns = max(2, 2 * ((fixture.live_blocks + 1) // 2))
    plan["max_k_tiles"] = columns

    # Retain parent columns and two extra query rows to catch the final ragged
    # work item overwriting either boundary while pipeline phases are reused.
    shape = (query_len + 2, 4, columns + 4)
    score_storage = torch.full(shape, float("nan"), dtype=torch.float32, device=device)
    valid_storage = torch.full(shape, 201, dtype=torch.uint8, device=device)
    score = score_storage[:query_len, :, :columns]
    valid = valid_storage[:query_len, :, :columns]

    def run():
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

    run()
    torch.cuda.synchronize()
    reference_score = score.clone()
    expected_valid = (
        torch.arange(columns, device=device)[None, None]
        < data["geometry"]["prefix"][:, None, None]
    ).expand_as(valid)
    assert not bool(torch.isnan(reference_score).any())
    assert torch.equal(valid, expected_valid.to(torch.uint8))
    assert bool(torch.isneginf(reference_score[~expected_valid]).all())

    rows = sorted({0, min(128, query_len - 1), query_len - 1})
    _, expected, reference_valid = fixture.numerical_reference(rank, query_rows=rows)
    actual = reference_score[rows, :, : fixture.live_blocks].cpu()
    readable = reference_valid[:, None].expand_as(expected)
    torch.testing.assert_close(
        actual[readable], expected[readable], rtol=2e-5, atol=1e-4
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()  # Exactly one whole-Q scorer in the graph.
    for _ in range(3):
        score_storage.fill_(float("nan"))
        valid_storage.fill_(201)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(score.view(torch.int32), reference_score.view(torch.int32))
        assert torch.equal(valid, expected_valid.to(torch.uint8))
        assert bool(torch.isnan(score_storage[query_len:]).all())
        assert bool(torch.isnan(score_storage[:query_len, :, columns:]).all())
        assert bool((valid_storage[query_len:] == 201).all())
        assert bool((valid_storage[:query_len, :, columns:] == 201).all())
