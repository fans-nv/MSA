# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""TP2/B128/R64 decode correctness, separate from any timed benchmark.

Small-integer fixtures certify exact addresses, geometry, and complete live
prefix writes. In ABI2 inactive/tail cells remain unadvertised and untouched.
Finite random tests qualify numerical behavior independently against
CPU FP32 and a repeated same-arithmetic whole-page CuTe scan. Historical A2/F8
references are not silently admitted as exact oracles.

CPU controls: python -m pytest -q tests/integration/test_icp_decode_score.py -k oracle
GPU suite:   python -m pytest -q tests/integration/test_icp_decode_score.py
150K gate:  MSA_ICP_LONG_CORRECTNESS=1 python -m pytest -q ... -k long_context
"""

import os
from dataclasses import replace

import pytest
import torch

from icp_decode_fixtures import COMPOUND_LAYOUTS, PAD_PAGE, make_decode_fixture


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Requires the SM100/SM103 CuTe target")
    return torch.device("cuda", torch.cuda.current_device())


def assert_outputs(actual_score, actual_valid, expected, *, exact=True):
    """Check only the independently known live prefix, never tail validity."""
    score, valid = actual_score.detach().cpu(), actual_valid.detach().cpu()
    expected_score, expected_valid = expected
    assert score.shape == expected_score.shape
    assert valid.shape == expected_valid.shape
    live = expected_valid.bool()
    assert not bool(torch.isnan(score[live]).any()), (
        "NaN score: unwritten or poisoned live row"
    )
    assert bool((valid[live] == 1).all()), "Unwritten live validity cell"
    torch.testing.assert_close(
        score[live],
        expected_score[live],
        rtol=0 if exact else 2e-5,
        atol=0 if exact else 1e-4,
    )


def run_and_check(fixture, *, scorer=None, exact=True):
    expected = fixture.reference()
    guards = fixture.output_guard_snapshot()
    before_score = fixture.score_out[: fixture.token_count].view(torch.int32).clone()
    before_valid = fixture.valid_out[: fixture.token_count].clone()
    scorer = fixture.prepare_scorer() if scorer is None else scorer
    scorer(*fixture.inputs)
    assert_outputs(
        fixture.score_out[: fixture.token_count],
        fixture.valid_out[: fixture.token_count],
        expected,
        exact=exact,
    )
    invalid = ~expected[1].bool().to(fixture.score_out.device)
    assert torch.equal(
        fixture.score_out[: fixture.token_count].view(torch.int32)[invalid],
        before_score[invalid],
    )
    assert torch.equal(
        fixture.valid_out[: fixture.token_count][invalid], before_valid[invalid]
    )
    for actual, before in zip(fixture.output_guard_snapshot(), guards):
        torch.testing.assert_close(actual, before, rtol=0, atol=0, equal_nan=True)
    return scorer


@pytest.mark.parametrize("rank", [0, 1])
def test_oracle_boundary_geometry_and_compound_bytes(rank):
    fixture = make_decode_fixture(
        query_len=1,
        batch_size=6,
        kv_lens=129,
        rank=rank,
        capacity_blocks=4,
        device="cpu",
        strided_queries=True,
    )
    fixture.positions.copy_(torch.tensor([31, 32, 63, 64, 127, 128]))
    _, valid = fixture.reference()
    if rank == 0:
        first = [1, 1, 1, 1, 1, 1]
        second = [0, 0, 0, 0, 0, 1]
    else:
        first = [0, 0, 0, 1, 1, 1]
        second = [0, 0, 0, 0, 0, 0]
    assert valid[:, 0, 0].tolist() == first
    assert valid[:, 0, 1].tolist() == second
    assert not bool(valid[:, :, 2:].any())
    pitch, offset = COMPOUND_LAYOUTS[fixture.main_format]
    assert fixture.index_k.stride() == (pitch, 128, 1)
    assert fixture.index_k.storage_offset() == offset
    pages = fixture.block_table[:, 0].long()
    torch.testing.assert_close(
        fixture.index_k[pages].float(),
        fixture.full_index_keys[pages, rank * 64 : (rank + 1) * 64].float(),
        rtol=0,
        atol=0,
    )
    assert bool((fixture.block_table[:, 2:] == PAD_PAGE).all())


def test_oracle_rejects_wrong_page_rank_causality_and_skipped_live_write():
    fixture = make_decode_fixture(
        query_len=1,
        batch_size=2,
        kv_lens=257,
        rank=0,
        capacity_blocks=5,
        device="cpu",
    )
    fixture.positions.fill_(0)
    expected = fixture.reference()
    original_table = fixture.block_table.clone()
    fixture.block_table[:, 0] = original_table.flip(0)[:, 0]
    wrong_page = fixture.reference()
    fixture.block_table.copy_(original_table)
    fixture.positions.fill_(256)
    wrong_causal = fixture.reference()
    fixture.positions.fill_(0)
    wrong_rank = fixture.reference(rank=1)
    for corrupted in (wrong_page, wrong_causal, wrong_rank):
        with pytest.raises(AssertionError):
            assert_outputs(*corrupted, expected)
    skipped_score, skipped_valid = (tensor.clone() for tensor in expected)
    skipped_score[0, 0, 0] = float("nan")
    skipped_valid[0, 0, 0] = 9
    with pytest.raises(AssertionError, match="NaN score"):
        assert_outputs(skipped_score, skipped_valid, expected)
    stale_score, stale_valid = (tensor.clone() for tensor in expected)
    stale_score[0, 0, -1] = 1e6
    stale_valid[0, 0, -1] = 1
    # ABI2 deliberately assigns no meaning to an unadvertised tail. Actual
    # selector tests separately prove exact bounds prevent consuming this cell.
    assert_outputs(stale_score, stale_valid, expected)


@pytest.mark.parametrize("query_len", [1, 2, 3, 4])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("main_format", ["nvfp4", "fp8"])
def test_exact_boundaries_compound_strides_and_output_guards(
    cuda_device,
    query_len,
    rank,
    main_format,
):
    fixture = make_decode_fixture(
        query_len=query_len,
        batch_size=8,
        rank=rank,
        main_format=main_format,
        kv_lens=[32, 33, 64, 65, 128, 129, 0, 385],
        capacity_blocks=8,
        strided_queries=True,
        strided_outputs=True,
        retained_rows=8 * query_len + 3,
        device=cuda_device,
    )
    run_and_check(fixture)


@pytest.mark.parametrize("rank", [0, 1])
def test_partial_fragment_future_nan_is_masked_before_max(cuda_device, rank):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=8,
        rank=rank,
        kv_lens=[1, 31, 32, 63, 64, 65, 127, 129],
        capacity_blocks=6,
        poison_future=True,
        device=cuda_device,
    )
    run_and_check(fixture)


@pytest.mark.parametrize("rank", [0, 1])
def test_one_prepared_scorer_matches_packed_and_both_compound_layouts(
    cuda_device, rank
):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=8,
        kv_lens=513,
        rank=rank,
        capacity_blocks=8,
        random_seed=317,
        device=cuda_device,
    )
    scorer = run_and_check(fixture, exact=False)
    expected_score = fixture.score_out.clone()
    expected_valid = fixture.valid_out.clone()
    live = fixture.reference()[1].bool().to(cuda_device)
    packed = replace(fixture, index_k=fixture.index_k.contiguous())
    packed.poison_outputs()
    scorer(*packed.inputs)
    torch.testing.assert_close(
        packed.score_out[live], expected_score[live], rtol=0, atol=0
    )
    torch.testing.assert_close(
        packed.valid_out[live], expected_valid[live], rtol=0, atol=0
    )
    # Change K pitch/base without preparing a different callable: these are live
    # tensor strides, not stale compile-time assumptions about compact storage.
    fp8_compound = make_decode_fixture(
        query_len=4,
        batch_size=8,
        kv_lens=513,
        rank=rank,
        main_format="fp8",
        capacity_blocks=8,
        random_seed=317,
        device=cuda_device,
    )
    scorer(*fp8_compound.inputs)
    torch.testing.assert_close(
        fp8_compound.score_out[live], expected_score[live], rtol=0, atol=0
    )
    torch.testing.assert_close(
        fp8_compound.valid_out[live], expected_valid[live], rtol=0, atol=0
    )


@pytest.mark.parametrize("main_format", ["nvfp4", "fp8"])
def test_memory_layout_negative_controls_detect_real_poison_reads(
    cuda_device, main_format
):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=2,
        kv_lens=385,
        rank=1,
        main_format=main_format,
        capacity_blocks=8,
        device=cuda_device,
    )
    scorer = run_and_check(fixture)
    expected = fixture.reference()
    # The extra guard page makes both deliberately wrong TMA descriptors fully
    # in-bounds. These mutations model address bugs, not illegal-access tests.
    storage = torch.full(
        (fixture.compound_storage.numel() + fixture.compound_page_bytes,),
        0x7F,
        dtype=torch.uint8,
        device=cuda_device,
    )
    storage[: fixture.compound_storage.numel()].copy_(fixture.compound_storage)
    fp8_storage = storage.view(torch.float8_e4m3fn)
    extra_rank_offset = fp8_storage.as_strided(
        fixture.index_k.shape,
        fixture.index_k.stride(),
        fixture.index_component_offset_bytes + 64 * 128,
    )
    narrowed_page_pitch = fp8_storage.as_strided(
        fixture.index_k.shape,
        (64 * 128, 128, 1),
        fixture.index_component_offset_bytes,
    )
    for wrong_keys in (extra_rank_offset, narrowed_page_pitch):
        fixture.poison_outputs()
        scorer(fixture.full_q, wrong_keys, *fixture.inputs[2:])
        # The SAME gate that admits the true producer must reject actual GPU
        # outputs from each faulty memory interpretation.
        with pytest.raises(AssertionError):
            assert_outputs(fixture.score_out, fixture.valid_out, expected)


@pytest.mark.parametrize("query_len", [1, 2, 3, 4])
@pytest.mark.parametrize("rank", [0, 1])
def test_zero_length_request_intervals_and_inactive_rows(cuda_device, query_len, rank):
    fixture = make_decode_fixture(
        query_len=query_len,
        batch_size=5,
        rank=rank,
        kv_lens=[193, 0, 65, 0, 129],
        request_query_lens=[query_len, 0, query_len, 0, query_len],
        capacity_blocks=6,
        device=cuda_device,
    )
    fixture.active_rows[0] = False
    fixture.positions[0] = torch.iinfo(torch.int64).max
    assert fixture.query_start_loc.cpu().tolist() == [
        0,
        query_len,
        query_len,
        2 * query_len,
        2 * query_len,
        3 * query_len,
    ]
    run_and_check(fixture)


@pytest.mark.parametrize("rank", [0, 1])
def test_q3_request_split_across_128_row_windows(cuda_device, rank):
    fixture = make_decode_fixture(
        query_len=3,
        batch_size=45,
        kv_lens=385,
        rank=rank,
        capacity_blocks=8,
        device=cuda_device,
    )
    expected_score, expected_valid = fixture.reference()
    first = fixture.window(0, 128)
    second = fixture.window(128, 7)
    run_and_check(first)
    run_and_check(second)
    assert_outputs(
        torch.cat([first.score_out, second.score_out]),
        torch.cat([first.valid_out, second.valid_out]),
        (expected_score, expected_valid),
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_long_to_short_reuse_changes_only_live_prefix_in_8192_columns(
    cuda_device, rank
):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=8,
        kv_lens=641,
        rank=rank,
        capacity_blocks=8192,
        device=cuda_device,
    )
    scorer = run_and_check(fixture)
    # Leave the first invocation's high/live values in place; no external fill.
    fixture.seq_lens.copy_(
        torch.tensor([0, 1, 32, 64, 65, 128, 129, 193], device=cuda_device)
    )
    fixture.positions.copy_(
        torch.tensor(
            [
                max(length - 4 + slot, -1)
                for length in [0, 1, 32, 64, 65, 128, 129, 193]
                for slot in range(4)
            ],
            dtype=torch.int64,
            device=cuda_device,
        )
    )
    fixture.active_rows[::3] = False
    run_and_check(fixture, scorer=scorer)


@pytest.mark.parametrize("rank", [0, 1])
def test_full_capture_replay_reads_changed_gpu_metadata(cuda_device, rank):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=4,
        kv_lens=385,
        rank=rank,
        capacity_blocks=16,
        device=cuda_device,
    )
    scorer = run_and_check(fixture)
    pointers = tuple(tensor.data_ptr() for tensor in fixture.inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scorer(*fixture.inputs)
    original_table = fixture.block_table.clone()
    fixture.query_start_loc.copy_(
        torch.tensor([0, 4, 4, 8, 12], dtype=torch.int32, device=cuda_device)
    )
    fixture.seq_lens.copy_(
        torch.tensor([65, 0, 129, 257], dtype=torch.int32, device=cuda_device)
    )
    fixture.positions.copy_(
        torch.tensor(
            [64, 63, 32, 0, 128, 127, 65, 64, 256, 255, 192, 191, -1, -1, -1, -1],
            dtype=torch.int64,
            device=cuda_device,
        )
    )
    fixture.active_rows.copy_(
        torch.tensor(
            [
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            device=cuda_device,
        )
    )
    fixture.block_table[:, :3].copy_(original_table.flip(0)[:, :3])
    fixture.poison_outputs()
    graph.replay()
    assert tuple(tensor.data_ptr() for tensor in fixture.inputs) == pointers
    assert_outputs(fixture.score_out, fixture.valid_out, fixture.reference())
    # Empty -> nonempty on the same graph and pointers exercises skipped-CTA reuse.
    fixture.query_start_loc.copy_(
        torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device=cuda_device)
    )
    fixture.seq_lens.fill_(385)
    fixture.positions.copy_(
        torch.tensor([381, 382, 383, 384] * 4, dtype=torch.int64, device=cuda_device)
    )
    fixture.active_rows.fill_(True)
    fixture.block_table.copy_(original_table)
    graph.replay()
    assert tuple(tensor.data_ptr() for tensor in fixture.inputs) == pointers
    assert_outputs(fixture.score_out, fixture.valid_out, fixture.reference())


@pytest.mark.parametrize("rank", [0, 1])
def test_random_fp8_fp32_differential_and_repeatability(cuda_device, rank):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=8,
        rank=rank,
        kv_lens=[129, 193, 257, 319, 384, 385, 449, 513],
        capacity_blocks=12,
        random_seed=103,
        device=cuda_device,
    )
    scorer = run_and_check(fixture, exact=False)
    first_score, first_valid = fixture.score_out.clone(), fixture.valid_out.clone()
    live = fixture.reference()[1].bool().to(cuda_device)
    for _ in range(2):
        fixture.poison_outputs()
        scorer(*fixture.inputs)
        torch.testing.assert_close(
            fixture.valid_out[live], first_valid[live], rtol=0, atol=0
        )
        torch.testing.assert_close(
            fixture.score_out[live], first_score[live], rtol=0, atol=0
        )


@pytest.mark.parametrize("main_format", ["nvfp4", "fp8"])
def test_rank_max_matches_repeatable_whole_cute(cuda_device, main_format):
    from _icp_decode_whole_reference import minimax_m3_index_decode_score_cutedsl

    rank0 = make_decode_fixture(
        query_len=4,
        batch_size=8,
        kv_lens=[129, 193, 257, 319, 384, 385, 449, 513],
        main_format=main_format,
        capacity_blocks=12,
        random_seed=103,
        device=cuda_device,
    )
    rank1 = rank0.for_rank(1)
    run_and_check(rank0, exact=False)
    run_and_check(rank1, exact=False)
    whole = torch.full((4, 32, 12), float("-inf"), device=cuda_device)
    whole_table = rank0.block_table.contiguous()
    repeats = []
    for _ in range(3):
        whole.fill_(float("-inf"))
        minimax_m3_index_decode_score_cutedsl(
            rank0.full_q,
            rank0.full_index_keys,
            whole_table,
            rank0.seq_lens,
            513,
            0,
            0,
            1,
            4,
            4,
            whole,
        )
        repeats.append(whole.transpose(0, 1).clone())
    # Reconstruct only after masking from exact external geometry. ABI2's
    # out-of-prefix validity bytes are undefined and are not read as a mask.
    rank0_live = rank0.reference()[1].bool().to(cuda_device)
    rank1_live = rank1.reference()[1].bool().to(cuda_device)
    live = rank0_live | rank1_live
    # A self-disagreeing reference cannot certify the adapted scorer (F8 gate).
    assert not bool(torch.isnan(repeats[0][live]).any())
    for repeated in repeats[1:]:
        torch.testing.assert_close(repeated[live], repeats[0][live], rtol=0, atol=0)
    combined = torch.maximum(
        torch.where(rank0_live, rank0.score_out, -float("inf")),
        torch.where(rank1_live, rank1.score_out, -float("inf")),
    )
    torch.testing.assert_close(combined[live], repeats[0][live], rtol=0, atol=0)


@pytest.mark.skipif(
    os.environ.get("MSA_ICP_LONG_CORRECTNESS") != "1",
    reason="Set MSA_ICP_LONG_CORRECTNESS=1 for explicit BS8/16 Q4 150K correctness",
)
@pytest.mark.parametrize("batch_size", [8, 16])
@pytest.mark.parametrize("rank", [0, 1])
def test_long_context_150k_with_1m_capacity(cuda_device, batch_size, rank):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=batch_size,
        kv_lens=150000,
        rank=rank,
        capacity_blocks=8192,
        device=cuda_device,
    )
    run_and_check(fixture)


@pytest.mark.parametrize("rank", [0, 1])
def test_one_compile_serves_every_runtime_window(cuda_device, rank):
    """Launch ABI 2: the window is a launch scalar, so windows of one
    (query_len, rank, split_k) share one compiled scan and each still writes
    only its own rows."""
    from fmha_sm100.icp.scorer.decode import icp_decode_score as api

    fixture = make_decode_fixture(
        query_len=4,
        batch_size=12,
        kv_lens=385,
        rank=rank,
        capacity_blocks=16,
        device=cuda_device,
    )
    misses = api._compile_scan.cache_info().misses
    scorers = []
    for begin, count in ((0, 16), (16, 32), (8, 4)):
        window = fixture.window(begin, count)
        scorers.append(run_and_check(window))
    assert len({scorer._scan for scorer in scorers}) == 1
    assert api._compile_scan.cache_info().misses - misses <= 1


@pytest.mark.parametrize("rank", [0, 1])
def test_window_past_the_invocation_rows_scores_only_existing_rows(cuda_device, rank):
    """A rung window wider than index_q (a padded-extent caller) is legal: rows
    at or past index_q.shape[0] are never read or written."""
    fixture = make_decode_fixture(
        query_len=1,
        batch_size=5,
        kv_lens=257,
        rank=rank,
        capacity_blocks=16,
        device=cuda_device,
    )
    expected = fixture.reference()
    wide = fixture.window(0, 5)
    scores, valid, *_ = _wide_outputs(fixture, rows=8)
    scorer = fixture.prepare_scorer().bind(
        token_begin=0,
        token_count=8,
        request_begin=0,
        request_count=5,
    )
    before = scores[5:].view(torch.int32).clone()
    scorer(*wide.inputs[:7], scores, valid)
    assert_outputs(scores[:5], valid[:5], expected)
    assert torch.equal(scores[5:].view(torch.int32), before)


def _wide_outputs(fixture, *, rows):
    scores = torch.full(
        (rows, 4, fixture.score_out.shape[2]),
        float("nan"),
        device=fixture.score_out.device,
    )
    valid = torch.full_like(scores, 9, dtype=torch.uint8)
    return scores, valid
