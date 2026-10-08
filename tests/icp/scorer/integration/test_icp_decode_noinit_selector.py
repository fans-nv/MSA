# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Decode ABI2: score only live prefixes, then run the actual ICP selector.

Unadvertised score/validity cells may contain NaNs or stale large scores. The
selector must exclude them before loads and NaN checks using GPU-authoritative
per-row geometry. The initialized comparison is test-owned storage, not an
initialization path in the MSA API. Prefill is outside this contract.

Requires installed MSA and fmha_sm100.icp for GPU tests. CPU metadata checks:
python -m pytest -q tests/integration/test_icp_decode_noinit_selector.py -k metadata_oracle
"""

from dataclasses import replace

import pytest
import torch

from icp_decode_fixtures import make_decode_fixture


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Requires the SM100/SM103/SM107 CuTe scorer")
    return torch.device("cuda", torch.cuda.current_device())


def metadata_tensors(fixture):
    """Test metadata refresh, outside capture; values stay in GPU tensors.

    V2's prepare_pos_seq_lens writes L=computed+Q and p=computed+u together,
    so its legal active rows have p<L. The general scorer bound below also
    honors its explicit materialized length if a synthetic position is later.
    Expected geometry is checked independently against the full-page oracle.
    """
    rows = torch.arange(
        fixture.token_begin,
        fixture.token_begin + fixture.token_count,
        dtype=torch.int64,
        device=fixture.full_q.device,
    )
    request = torch.searchsorted(fixture.query_start_loc[1:], rows, right=True)
    safe_request = request.clamp(max=fixture.seq_lens.numel() - 1)
    covered = (
        (request >= fixture.request_begin)
        & (request < fixture.request_begin + fixture.request_count)
        & (rows < fixture.query_start_loc[-1])
    )
    # `fixture.active_rows` is the scorer's uint8 ABI plane; the selector's
    # CandidateGeometry.active_rows is torch bool, so convert explicitly.
    active = fixture.active_rows[rows].bool() & covered
    positions = fixture.positions[rows]
    lengths = fixture.seq_lens[safe_request].to(torch.int64).clamp(min=0)
    visible = torch.minimum((positions + 1).clamp(min=0), lengths)
    prefix = visible // 128 + ((visible % 128) > fixture.rank * 64)
    nvalid = torch.where(active, prefix, 0).to(torch.int32)
    forced = positions // 128
    forced = torch.where(active & (positions >= 0) & (forced < nvalid), forced, -1).to(
        torch.int32
    )
    return nvalid, forced, active


class SelectorSession:
    """Retained existing ICP selector state; allocation/JIT precedes capture."""

    def __init__(self, fixture):
        from fmha_sm100.icp.candidates import (
            CandidateGeometry,
            DecodePlan,
            SelectArm,
            allocate_workspace,
            select_decode_candidates,
        )

        self.plan = DecodePlan(
            icp_degree=2,
            icp_rank=fixture.rank,
            num_heads_local=2,
            max_local_blocks=fixture.score_out.shape[2],
            token_capacity=fixture.token_count,
            use_pdl=False,
        )
        self.workspace = allocate_workspace(self.plan, fixture.full_q.device)
        self.geometry = CandidateGeometry(
            *metadata_tensors(fixture), scan_block_begin=0
        )
        self.entry = select_decode_candidates
        self.arm = SelectArm.RADIX_BOUNDED

    def refresh(self, fixture):
        for target, value in zip(self.metadata, metadata_tensors(fixture)):
            target.copy_(value)

    @property
    def metadata(self):
        return (
            self.geometry.local_valid_blocks,
            self.geometry.forced_column,
            self.geometry.active_rows,
        )

    @property
    def output(self):
        return self.workspace.candidates

    def select(self, scores):
        return self.entry(
            scores,
            self.geometry,
            plan=self.plan,
            workspace=self.workspace,
            out=self.output,
            arm=self.arm,
        )


def poison(fixture):
    fixture.score_out[:, :, ::2].fill_(float("nan"))
    fixture.score_out[:, :, 1::2].fill_(1e30)
    fixture.valid_out.fill_(173)


def initialized_candidates(fixture, scorer, session):
    reference = replace(
        fixture,
        score_out=torch.full_like(fixture.score_out, -float("inf")),
        valid_out=torch.zeros_like(fixture.valid_out),
    )
    scorer.scan(*reference.inputs)
    return session.select(reference.score_out).view(torch.int32).clone()


def check_scan(fixture, session, before_score, before_valid):
    expected_score, expected_valid = fixture.reference()
    nvalid = session.geometry.local_valid_blocks.cpu()
    torch.testing.assert_close(
        nvalid, expected_valid[:, 0].sum(-1).to(torch.int32), rtol=0, atol=0
    )
    live = expected_valid.bool().to(fixture.score_out.device)
    torch.testing.assert_close(
        fixture.score_out[live].cpu(),
        expected_score[expected_valid.bool()],
        rtol=0,
        atol=0,
    )
    assert bool((fixture.valid_out[live] == 1).all())
    # Bitwise comparison keeps a NaN poison unchanged instead of treating two
    # identical NaN encodings as unequal. This catches hidden initializer work.
    assert torch.equal(fixture.score_out.view(torch.int32)[~live], before_score[~live])
    assert torch.equal(fixture.valid_out[~live], before_valid[~live])


def run_pair(fixture, scorer, session):
    session.refresh(fixture)
    expected = initialized_candidates(fixture, scorer, session)
    before_score = fixture.score_out.view(torch.int32).clone()
    before_valid = fixture.valid_out.clone()
    scorer.scan(*fixture.inputs)
    check_scan(fixture, session, before_score, before_valid)
    actual = session.select(fixture.score_out).view(torch.int32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return expected


@pytest.mark.parametrize("rank", [0, 1])
def test_metadata_oracle_matches_full_page_visibility(rank):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=5,
        kv_lens=[1, 0, 65, 128, 257],
        rank=rank,
        request_query_lens=[4, 0, 4, 4, 4],
        capacity_blocks=8,
        device="cpu",
    )
    fixture.active_rows[5] = False
    # Materialized length still bounds this deliberately later synthetic row.
    fixture.positions[0] = 4096
    nvalid, forced, active = metadata_tensors(fixture)
    _, valid = fixture.reference()
    torch.testing.assert_close(
        nvalid, valid[:, 0].sum(-1).to(torch.int32), rtol=0, atol=0
    )
    assert not bool(active[16:].any())
    assert bool((forced[~active] == -1).all())
    assert bool(((forced < 0) | (forced < nvalid)).all())


@pytest.mark.parametrize("query_len", [1, 2, 3, 4])
@pytest.mark.parametrize("rank", [0, 1])
def test_noinit_selector_ignores_poisoned_tails_padding_and_empty_fragments(
    cuda_device, query_len, rank
):
    fixture = make_decode_fixture(
        query_len=query_len,
        batch_size=8,
        rank=rank,
        kv_lens=[1, 0, 64, 65, 128, 129, 257, 4097],
        request_query_lens=[
            query_len,
            0,
            query_len,
            query_len,
            query_len,
            query_len,
            query_len,
            query_len,
        ],
        capacity_blocks=8192,
        device=cuda_device,
    )
    fixture.active_rows[query_len] = False
    poison(fixture)
    scorer = fixture.prepare_scorer()
    session = SelectorSession(fixture)
    run_pair(fixture, scorer, session)


@pytest.mark.parametrize("rank", [0, 1])
def test_noinit_selector_excludes_forced_nan_before_score_load(cuda_device, rank):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=2,
        kv_lens=4097,
        rank=rank,
        capacity_blocks=8192,
        device=cuda_device,
    )
    poison(fixture)
    scorer = fixture.prepare_scorer()
    session = SelectorSession(fixture)
    expected = run_pair(fixture, scorer, session)
    columns = torch.arange(8192, device=cuda_device)[None, None, :]
    excluded = (columns == session.geometry.forced_column[:, None, None]).expand_as(
        fixture.score_out
    )
    assert bool(excluded.any())
    fixture.score_out.masked_fill_(excluded, float("nan"))
    torch.testing.assert_close(
        session.select(fixture.score_out).view(torch.int32), expected, rtol=0, atol=0
    )


def test_noinit_selector_overstated_prefix_negative_control(cuda_device):
    fixture = make_decode_fixture(
        query_len=1,
        batch_size=1,
        kv_lens=32,
        rank=1,
        capacity_blocks=8192,
        device=cuda_device,
    )
    fixture.score_out.fill_(1e30)
    scorer = fixture.prepare_scorer()
    session = SelectorSession(fixture)
    expected = run_pair(fixture, scorer, session)
    assert bool((expected[..., 1] == -1).all())
    # Deliberately admit one unwritten, dominant finite cell on an empty rank.
    # The actual selector must now differ; an insensitive fixture is not a gate.
    session.geometry.local_valid_blocks.fill_(1)
    wrong = session.select(fixture.score_out).view(torch.int32)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong, expected, rtol=0, atol=0)
    assert bool((wrong[:, :, 0, 1] == 0).all())


@pytest.mark.parametrize("rank", [0, 1])
def test_noinit_selector_reuses_scratch_across_q3_windows(cuda_device, rank):
    fixture = make_decode_fixture(
        query_len=3,
        batch_size=86,
        kv_lens=4097,
        rank=rank,
        capacity_blocks=8192,
        device=cuda_device,
    )
    first = fixture.window(0, 128)
    second = replace(
        fixture.window(128, 128),
        score_out=first.score_out,
        valid_out=first.valid_out,
        score_storage=first.score_storage,
        valid_storage=first.valid_storage,
    )
    poison(first)
    first_scorer, second_scorer = first.prepare_scorer(), second.prepare_scorer()
    session = SelectorSession(first)
    run_pair(first, first_scorer, session)
    # Request42 straddles row128. The same scratch now serves the next window
    # with a shorter history and stale high/NaN cells from the previous call.
    fixture.seq_lens.fill_(129)
    fixture.positions.copy_(
        torch.tensor([126, 127, 128] * 86, dtype=torch.int64, device=cuda_device)
    )
    fixture.active_rows[::7] = False
    run_pair(second, second_scorer, session)
    run_pair(first, first_scorer, session)


@pytest.mark.parametrize("rank", [0, 1])
def test_noinit_selector_full_graph_replay_shrinks_and_reactivates(cuda_device, rank):
    fixture = make_decode_fixture(
        query_len=4,
        batch_size=8,
        kv_lens=4097,
        rank=rank,
        capacity_blocks=8192,
        device=cuda_device,
    )
    poison(fixture)
    scorer = fixture.prepare_scorer()
    session = SelectorSession(fixture)
    run_pair(fixture, scorer, session)
    original = {
        name: getattr(fixture, name).clone()
        for name in (
            "query_start_loc",
            "seq_lens",
            "positions",
            "active_rows",
            "block_table",
        )
    }

    def retained_pointers():
        return tuple(
            tensor.data_ptr()
            for tensor in (
                *fixture.inputs,
                *session.metadata,
                session.output,
                session.workspace.partials,
            )
        )

    pointers = retained_pointers()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scorer.scan(*fixture.inputs)
        session.select(fixture.score_out)

    lengths = [65, 0, 129, 193, 0, 1025, 4097, 32]
    fixture.query_start_loc.copy_(
        torch.tensor(
            [0, 4, 4, 8, 12, 12, 16, 20, 24], dtype=torch.int32, device=cuda_device
        )
    )
    fixture.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32, device=cuda_device))
    positions = [
        length - 4 + slot for length in lengths if length for slot in range(4)
    ] + [-1] * 8
    fixture.positions.copy_(
        torch.tensor(positions, dtype=torch.int64, device=cuda_device)
    )
    fixture.active_rows[:24].fill_(True)
    fixture.active_rows[24:].fill_(False)
    fixture.active_rows[2::5] = False
    fixture.block_table.copy_(original["block_table"].flip(0))

    for state in ("short_with_padding", "all_inactive", "restored"):
        if state == "all_inactive":
            fixture.query_start_loc.zero_()
            fixture.seq_lens.zero_()
            fixture.active_rows.fill_(False)
        elif state == "restored":
            for name, value in original.items():
                getattr(fixture, name).copy_(value)
        session.refresh(fixture)
        expected = initialized_candidates(fixture, scorer, session)
        before_score = fixture.score_out.view(torch.int32).clone()
        before_valid = fixture.valid_out.clone()
        graph.replay()
        check_scan(fixture, session, before_score, before_valid)
        torch.testing.assert_close(
            session.output.view(torch.int32), expected, rtol=0, atol=0
        )
        assert retained_pointers() == pointers
