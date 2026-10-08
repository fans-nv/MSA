# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""One full-prefill scorer launch, including the Python/FFI dispatch path.

Run on the GB300 with the integration fixtures on pytest's import path. The
profiler scope contains exactly one invocation and excludes planning, JIT,
input construction and output poisoning. These are launch/ABI regressions;
the prefill campaign separately checks numerical correctness and latency.
"""

import json

import pytest
import torch

from icp_device_plan_harness import build_device_plan
from icp_prefill_fixtures import make_prefill_fixture


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("legacy_plan", [False, True])
def test_full_prefill_one_scorer_without_metadata_fill(rank, legacy_plan, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires the SM100/SM103 FMHA target")

    from fmha_sm100.icp.scorer.prefill.api import _fmha_sm100

    device = torch.device("cuda", torch.cuda.current_device())
    query_len, history = 1024, 0
    fixture = make_prefill_fixture(query_len=query_len, history=history, seed=23)
    data = fixture.materialize(rank, device)
    # The device plan store (ICP_DEVICE_PLAN_ABI 3); its rows are loaded
    # before the profiled scope, as the writer refreshes them in production.
    plan, _store, _ = build_device_plan(
        query_lens=[query_len],
        seq_lens=[history + query_len],
        table=data["table"],
        rank=rank,
        device=device,
    )
    assert plan["max_qo_len"] == query_len
    assert plan["qo_segment_lens"].cpu().tolist() == [query_len]
    assert plan["num_kv_splits"] == 1

    # A single plan contains every Q128 tile and all four local query heads.
    ranges = plan["packed_work_range"].cpu().tolist()
    work_end = max(value >> 32 for value in ranges)
    work_info = plan["packed_work_info"][:work_end].cpu().tolist()
    work = [
        work_info[i] for value in ranges for i in range(value & 0xFFFFFFFF, value >> 32)
    ]
    assert {
        (value >> 32, (value >> 16) & 0xFFFF, value & 0xFFFF) for value in work
    } == {(tile, head, 0) for tile in range(query_len // 128) for head in range(4)}
    assert len(work) == 4 * (query_len // 128)

    descriptor = plan["icp_geometry_descriptor"]
    assert descriptor.shape == (1, 1, rank * 16 + 2)
    assert descriptor.stride() == (0, 0, 0)
    assert descriptor.data_ptr() == plan["qo_segment_lens"].data_ptr()
    score = torch.empty(
        (query_len, 4, plan["max_k_tiles"]), dtype=torch.float32, device=device
    )
    valid = torch.empty_like(score, dtype=torch.uint8)

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

    run()  # Compile and initialize the native dispatch before profiling.
    torch.cuda.synchronize()
    if legacy_plan:
        # The first invocation of a pre-existing plan also needs no device work
        # to recover its descriptor; subsequent calls retain the recovered view.
        del plan["icp_geometry_descriptor"]
    score.fill_(float("nan"))
    valid.fill_(9)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        run()
        torch.cuda.synchronize()
    trace_path = tmp_path / f"prefill-r{rank}-legacy{int(legacy_plan)}.json"
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    assert len(kernels) == 1, [event.get("name") for event in kernels]
    assert "fmha" in kernels[0]["name"].lower(), kernels[0]["name"]
    assert plan["icp_geometry_descriptor"].data_ptr() == descriptor.data_ptr()

    # W5 must still replace every advertised poison cell in the same launch.
    columns = torch.arange(plan["max_k_tiles"], device=device)
    expected = columns[None, None] < data["geometry"]["prefix"][:, None, None]
    expected = expected.expand_as(valid)
    assert torch.equal(valid, expected.to(torch.uint8))
    assert not bool(torch.isnan(score).any())
    assert bool(torch.isneginf(score[~expected]).all())
