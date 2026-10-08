# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Qualify the enabled direct scorer across retained empty/nonempty work.

Run against the isolated v17 source with MSA_ICP_DIRECT_SCORE_PIPELINE=1 and a
fresh JIT cache. This test deliberately refuses the disabled source or a
single-warpgroup/split-KV dispatch: either would miss the changed protocol.
No planner work list is rewritten. On the 152-SM GB300 the ordinary planner
retains nonempty -> empty -> nonempty items on the same persistent CTA.
"""

import json
from pathlib import Path
import re

import pytest
import torch

from icp_device_plan_harness import build_device_plan
from icp_prefill_fixtures import (
    CAPACITY,
    DIM,
    FRAGMENT,
    HEADS,
    INDEX_OFFSET,
    PAGE_BYTES,
    causal_geometry,
    make_prefill_fixture,
)


def _materialize_batch(fixtures, rank, device):
    """Disjoint physical pages retain each fixture's independent CPU oracle."""
    tables = []
    page_offset = 0
    for fixture in fixtures:
        table = fixture.block_table.clone()
        table[:, : fixture.live_blocks] += page_offset
        tables.append(table)
        page_offset += fixture.key_bytes.shape[0]
    table = torch.cat(tables).to(device)
    q = (
        torch.cat([fixture.q_bytes for fixture in fixtures])
        .view(torch.float8_e4m3fn)
        .to(device)
    )
    local_bytes = torch.cat(
        [
            fixture.key_bytes[:, rank * FRAGMENT : (rank + 1) * FRAGMENT]
            for fixture in fixtures
        ]
    ).to(device)
    storage = torch.full(
        (page_offset * PAGE_BYTES,), 0x7F, dtype=torch.uint8, device=device
    )
    keys = storage.view(torch.float8_e4m3fn).as_strided(
        (page_offset, 1, FRAGMENT, DIM),
        (PAGE_BYTES, FRAGMENT * DIM, DIM, 1),
        INDEX_OFFSET,
    )
    keys.view(torch.uint8)[:, 0].copy_(local_bytes)
    assert table.shape == (len(fixtures), CAPACITY)
    assert table.storage_offset() == 0 and table.stride(1) == 1
    return q, keys, storage, table


def _inspect_retained_work(plan, fixtures, empty_batches):
    """Read the actual GPU plan once, before any captured scorer invocation."""
    packed_ranges = plan["packed_work_range"].cpu().tolist()
    work_end = max(value >> 32 for value in packed_ranges)
    packed_items = plan["packed_work_info"][:work_end].cpu().tolist()
    work = []
    by_cta = []
    for cta, packed_range in enumerate(packed_ranges):
        begin, end = packed_range & 0xFFFFFFFF, packed_range >> 32
        items = [
            (value >> 32, (value >> 16) & 0xFFFF, value & 0xFFFF)
            for value in packed_items[begin:end]
        ]
        work.extend(items)
        by_cta.append(
            {
                "cta": cta,
                "items": items,
                "empty": [batch in empty_batches for _, _, batch in items],
            }
        )
    expected = {
        (tile, head, batch)
        for batch, fixture in enumerate(fixtures)
        for tile in range((fixture.query_len + 127) // 128)
        for head in range(HEADS)
    }
    assert len(work) == len(expected) and set(work) == expected
    transitions = {
        (a, b) for row in by_cta for a, b in zip(row["empty"], row["empty"][1:])
    }
    assert (False, True) in transitions, "No retained nonempty -> empty transition"
    assert (True, False) in transitions, "No retained empty -> nonempty transition"
    sandwich = [
        row
        for row in by_cta
        if any(
            row["empty"][i - 1 : i + 2] == [False, True, False]
            for i in range(1, len(row["empty"]) - 1)
        )
    ]
    assert sandwich, "No CTA retains a nonempty -> empty -> nonempty sequence"
    return by_cta, sandwich


@pytest.mark.parametrize("history", [150000, 500000])
def test_direct_prefill_mixed_empty_persistent_replay(history, monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires the SM100/SM103 FMHA target")
    from fmha_sm100.icp.scorer.prefill import api

    header = (
        Path(api.__file__).resolve().parent
        / "csrc/include"
        / "sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
    )
    assert re.search(
        r"^\s*#define\s+MSA_ICP_DIRECT_SCORE_PIPELINE\s+1\s*$",
        header.read_text(),
        re.MULTILINE,
    ), (
        "This qualification must use the enabled direct-pipeline source and a fresh JIT cache"
    )

    rank = 1
    device = torch.device("cuda", torch.cuda.current_device())
    # Q64 has no rank-1 fragment at all. Q65 has exactly one visible rank-1
    # token, so its last query supplies a nonvacuous numerical boundary check.
    # The long request keeps the entire invocation in the full-prefill variant.
    requests = [(8193, history)] + [(64, 0), (65, 0)] * 8
    fixtures = [
        make_prefill_fixture(query_len=q, history=h, seed=71 + batch)
        for batch, (q, h) in enumerate(requests)
    ]
    empty_batches = {
        batch for batch, fixture in enumerate(fixtures) if fixture.total_len <= FRAGMENT
    }
    assert empty_batches == set(range(1, len(fixtures), 2))
    # Hold the compound allocation for the lifetime of the strided key view.
    q, keys, _storage, table = _materialize_batch(fixtures, rank, device)
    # Unsplit on purpose: this qualifies the direct-score pipeline, which the
    # split-KV variant does not use. The retained-transition assertions were
    # derived on the 152-SM GB300, so the scorer grid is pinned to 152 CTAs
    # (212-SM VR200 would assign the 150K batch without an empty -> nonempty
    # transition).
    plan, _store, _ = build_device_plan(
        query_lens=[f.query_len for f in fixtures],
        seq_lens=[f.total_len for f in fixtures],
        table=table,
        rank=rank,
        device=device,
        num_ctas=152,
    )
    assert plan["max_qo_len"] == 8193 and plan["num_kv_splits"] == 1
    assert plan["pack_factor"] == 1
    assert plan["kv_segment_lens"].cpu().tolist() == [f.total_len for f in fixtures]
    by_cta, sandwich = _inspect_retained_work(plan, fixtures, empty_batches)
    work_path = tmp_path / f"mixed-empty-h{history}-work.json"
    work_record = {
        "rank": rank,
        "requests": requests,
        "empty_batches": sorted(empty_batches),
        "all_cta_work": by_cta,
        "sandwich_cta_work": sandwich,
    }
    # Preserve the actual plan even if later GPU qualification fails or hangs.
    work_path.write_text(json.dumps(work_record, indent=2) + "\n")

    total_q = sum(f.query_len for f in fixtures)
    live_blocks = max(f.live_blocks for f in fixtures)
    columns = max(2, 2 * ((live_blocks + 1) // 2))
    plan["max_k_tiles"] = columns
    parent_shape = (total_q + 2, HEADS, columns + 4)
    score_parent = torch.full(
        parent_shape, float("nan"), dtype=torch.float32, device=device
    )
    valid_parent = torch.full(parent_shape, 201, dtype=torch.uint8, device=device)
    score = score_parent[1 : total_q + 1, :, :columns]
    valid = valid_parent[1 : total_q + 1, :, :columns]
    prefix = torch.cat(
        [causal_geometry(f.query_len, f.history, rank)["prefix"] for f in fixtures]
    ).to(device)
    expected_valid = (
        torch.arange(columns, device=device)[None, None] < prefix[:, None, None]
    ).expand_as(valid)

    dispatches = []
    real_get_variant = api.get_fmha_variant

    def record_variant(
        dtype_code,
        qo_tile_size,
        single_wg,
        sparse_mode,
        page_size,
        split_kv,
        pack_factor,
        **kwargs,
    ):
        dispatches.append(
            {
                "q_tile": qo_tile_size,
                "single_wg": single_wg,
                "sparse_mode": sparse_mode,
                "page_size": page_size,
                "split_kv": split_kv,
                "pack_factor": pack_factor,
            }
        )
        assert (
            qo_tile_size,
            single_wg,
            sparse_mode,
            page_size,
            split_kv,
            pack_factor,
        ) == (128, False, 4, 64, False, 1), (
            "Mixed batch escaped the enabled direct route"
        )
        return real_get_variant(
            dtype_code,
            qo_tile_size,
            single_wg,
            sparse_mode,
            page_size,
            split_kv,
            pack_factor,
            **kwargs,
        )

    monkeypatch.setattr(api, "get_fmha_variant", record_variant)

    def run():
        before = len(dispatches)
        api._fmha_sm100(
            q,
            keys,
            keys,
            plan,
            kv_indices=table,
            output_o=False,
            output_maxscore=True,
            max_score=score,
            valid_score=valid,
            icp_c=2,
            icp_rank=rank,
        )
        assert len(dispatches) == before + 1

    def assert_completion():
        assert not bool(torch.isnan(score).any())
        assert torch.equal(valid, expected_valid.to(torch.uint8))
        assert bool(torch.isneginf(score[~expected_valid]).all())
        assert bool(torch.isnan(score_parent[0]).all())
        assert bool(torch.isnan(score_parent[-1]).all())
        assert bool(torch.isnan(score_parent[1:-1, :, columns:]).all())
        assert bool((valid_parent[0] == 201).all())
        assert bool((valid_parent[-1] == 201).all())
        assert bool((valid_parent[1:-1, :, columns:] == 201).all())

    run()
    torch.cuda.synchronize()
    assert_completion()
    reference_score = score.clone()
    offset = 0
    for batch, fixture in enumerate(fixtures):
        # CPU FP32 dots use the original per-request encoded Q/K and original
        # physical table, independently of the assembled GPU cache or planner.
        rows = (
            [0, 127, 128, fixture.query_len - 1]
            if batch == 0
            else list(range(fixture.query_len))
        )
        _, expected, reference_valid = fixture.numerical_reference(
            rank, query_rows=rows
        )
        actual = reference_score[
            [offset + row for row in rows], :, : fixture.live_blocks
        ].cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-4)
        if batch in empty_batches:
            assert not bool(reference_valid.any())
            assert bool(
                torch.isneginf(
                    reference_score[offset : offset + fixture.query_len]
                ).all()
            )
            assert not bool(valid[offset : offset + fixture.query_len].any())
        else:
            assert bool(reference_valid.any()), (
                "Nonempty numerical check became vacuous"
            )
        offset += fixture.query_len
    assert offset == total_q

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()  # One invocation handles every long, short and empty request.

    for replay in range(3):
        score_parent.fill_(float("nan"))
        valid_parent.fill_(201)
        torch.cuda.synchronize()
        if replay == 0:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                graph.replay()
                torch.cuda.synchronize()
            trace = tmp_path / f"mixed-empty-h{history}.json"
            prof.export_chrome_trace(str(trace))
            kernels = [
                event
                for event in json.loads(trace.read_text())["traceEvents"]
                if event.get("cat") == "kernel"
            ]
            assert len(kernels) == 1, [event.get("name") for event in kernels]
            assert "fmha" in kernels[0]["name"].lower(), kernels[0]["name"]
        else:
            graph.replay()
            torch.cuda.synchronize()
        assert_completion()
        assert torch.equal(score.view(torch.int32), reference_score.view(torch.int32))

    work_record.update(
        native_dispatches=dispatches, scorer_kernels_per_replay=len(kernels)
    )
    work_path.write_text(json.dumps(work_record, indent=2) + "\n")
