"""Public MSA ABI2 scan with baseline and optimized ICP selectors, on one GPU.

The serving scan uses physical B128 pages; generic selector metadata still
covers both affine profiles on CPU. Tests deliberately keep
undefined outputs, NaN cache tails and poisoned unused page-table cells. Only
test setup/oracles read device results; the captured composition has two public
calls over retained buffers and no initializer or host readback.
"""

from __future__ import annotations

import pytest

from fmha_sm100.icp import candidates as c


def _value(token, head, request):
    # Small integers are exact in E4M3 and in this one-hot D128 dot product.
    # The final warp wins in a complete B128 block. Scores also have ties and
    # different head orderings, so dropping a warp or swapping heads is visible.
    return ((token // 128 * (head + 1) + request) % 7) - 3 + token % 128 // 32


def _oracle(page_size, rank, position, length, head, request, active=True):
    """Enumerate logical tokens; never use the producer's count/address helper."""
    scores = {}
    if active:
        for token in range(max(0, min(length, position + 1))):
            if token % page_size // (page_size // 2) == rank:
                block = token // 128
                scores[block] = max(
                    scores.get(block, -float("inf")), _value(token, head, request)
                )
    forced = position // 128
    ordinary = sorted(
        ((block, score) for block, score in scores.items() if block != forced),
        key=lambda item: (-item[1], item[0]),
    )[:16]
    return scores, ordinary


def _geometry_values(page_size, rank, position, length, active):
    upper = max(0, min(length, position + 1)) if active else 0
    count = upper // page_size + (upper % page_size > rank * (page_size // 2))
    origin, stride = (rank, 2) if page_size == 256 else (0, 1)
    delta = position // 128 - origin
    forced = (
        delta // stride
        if active and delta >= 0 and delta % stride == 0 and delta // stride < count
        else -1
    )
    return count, forced


@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("rank", [0, 1])
def test_composed_fixture_metadata_matches_independent_token_oracle(page_size, rank):
    origin, stride = (rank, 2) if page_size == 256 else (0, 1)
    for length in (0, 1, 33, 64, 65, 127, 128, 129, 257, 385, 8577):
        for position in (-1, 0, 63, 64, 127, 128, 255, 256, length - 1, length + 2):
            for active in (False, True):
                scores, ordinary = _oracle(
                    page_size, rank, position, length, 0, 0, active
                )
                count, forced = _geometry_values(
                    page_size, rank, position, length, active
                )
                assert set(scores) == {origin + stride * col for col in range(count)}
                expected_forced = position // 128
                assert (forced >= 0) == (expected_forced in scores)
                if forced >= 0:
                    assert origin + stride * forced == expected_forced
                assert all(block != expected_forced for block, _ in ordinary)
                assert ordinary == sorted(
                    ordinary, key=lambda item: (-item[1], item[0])
                )


@pytest.fixture
def msa_api(cuda_device):
    import torch

    if torch.cuda.get_device_capability(cuda_device) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("Public MSA CuTe scorer requires SM100, SM103 or SM107")
    api = pytest.importorskip(
        "fmha_sm100.icp_decode_score",
        reason="Install the directly consumed MSA package",
    )
    # An installed incompatible package must fail, not silently skip this seam.
    assert api.ICP_DECODE_SCORE_ABI_VERSION == 2
    assert api.ICP_DECODE_SCORE_LAUNCH_ABI_VERSION == 3
    assert not api.supports_icp_decode_score(query_len=4, page_size=256)
    return api


class _Composition:
    def __init__(self, torch, api, device, page_size, rank, query_len, arm):
        self.torch, self.device = torch, device
        self.page_size, self.rank, self.query_len = page_size, rank, query_len
        self.arm = arm
        self.tokens = 4 * query_len
        self.columns = 1048576 // page_size
        self.pages_per_request = (8577 + page_size - 1) // page_size
        self.rows = page_size // 2
        nphysical = 4 * self.pages_per_request + 1
        pitch = 45056 * (page_size // 128)
        offset = 36864 * (page_size // 128)
        self.backing = torch.full(
            (nphysical * pitch,), 0x7F, dtype=torch.uint8, device=device
        )
        self.keys = self.backing.view(torch.float8_e4m3fn).as_strided(
            (nphysical, self.rows, 128), (pitch, 128, 1), offset
        )
        # Nonidentity physical IDs; the last physical page remains all poison.
        self.physical = list(reversed(range(nphysical - 1)))
        self.q = torch.zeros((self.tokens, 4, 128), dtype=torch.float32)
        for head in range(4):
            self.q[:, head, head] = 1
        self.q = self.q.to(device=device, dtype=torch.float8_e4m3fn)
        self.table = torch.empty((4, self.columns), dtype=torch.int32, device=device)
        self.offsets = torch.empty(5, dtype=torch.int32, device=device)
        self.lengths = torch.empty(4, dtype=torch.int32, device=device)
        self.positions = torch.empty(self.tokens, dtype=torch.int64, device=device)
        self.active = torch.empty(self.tokens, dtype=torch.bool, device=device)
        self.scan_active = self.active.view(torch.uint8)
        self.scores = torch.empty((self.tokens, 4, self.columns), device=device)
        self.valid = torch.empty_like(self.scores, dtype=torch.uint8)
        self.geometry = c.CandidateGeometry(
            torch.empty(self.tokens, dtype=torch.int32, device=device),
            torch.empty(self.tokens, dtype=torch.int32, device=device),
            self.active,
            scan_block_begin=rank if page_size == 256 else 0,
            global_block_stride=page_size // 128,
        )
        self.plan = c.DecodePlan(
            icp_degree=2,
            icp_rank=rank,
            num_heads_local=2,
            max_local_blocks=self.columns,
            token_capacity=self.tokens,
            use_pdl=False,
        )
        self.workspace = c.allocate_workspace(self.plan, device)
        assert self.workspace._extension.candidate_mapping_abi_version == 2
        self.scorer = api.get_icp_decode_scorer(
            query_len=query_len,
            token_begin=0,
            token_count=self.tokens,
            request_begin=0,
            request_count=4,
            rank=rank,
            world_size=2,
            split_k=256,
            device=device,
        )

    def poison_outputs(self):
        self.scores[:, :, ::2].fill_(float("nan"))
        self.scores[:, :, 1::2].fill_(1e30)
        self.valid.fill_(173)

    def phase(self, short=False):
        """Test-owned metadata/cache setup, outside capture and replay."""
        torch = self.torch
        lengths = [0, 33, 257, 385] if short else [8577, 7999, 65, 0]
        qcounts = (
            [0, self.query_len, self.query_len, self.query_len]
            if short
            else [self.query_len, self.query_len, self.query_len, 0]
        )
        starts, records = [0], []
        for request, (length, count) in enumerate(zip(lengths, qcounts)):
            records.extend(
                (request, length - count + u, length, True) for u in range(count)
            )
            starts.append(starts[-1] + count)
        records.extend((0, -1, 0, False) for _ in range(self.tokens - len(records)))
        self.records = records
        self.offsets.copy_(torch.tensor(starts, dtype=torch.int32))
        self.lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
        self.positions.copy_(torch.tensor([r[1] for r in records], dtype=torch.int64))
        self.active.copy_(torch.tensor([r[3] for r in records], dtype=torch.bool))
        metadata = [
            _geometry_values(self.page_size, self.rank, pos, length, active)
            for _, pos, length, active in records
        ]
        self.geometry.local_valid_blocks.copy_(
            torch.tensor([x[0] for x in metadata], dtype=torch.int32)
        )
        self.geometry.forced_column.copy_(
            torch.tensor([x[1] for x in metadata], dtype=torch.int32)
        )
        table = torch.full((4, self.columns), -2147480000, dtype=torch.int32)
        keys = torch.full(tuple(self.keys.shape), float("nan"), dtype=torch.float32)
        for request, length in enumerate(lengths):
            for column in range((length + self.page_size - 1) // self.page_size):
                physical = self.physical[request * self.pages_per_request + column]
                table[request, column] = physical
                for row in range(self.rows):
                    token = column * self.page_size + self.rank * self.rows + row
                    if token < length:
                        keys[physical, row].zero_()
                        for head in range(4):
                            keys[physical, row, head] = _value(token, head, request)
        self.table.copy_(table)
        self.keys.copy_(keys.to(dtype=torch.float8_e4m3fn))

    def run(self):
        self.scorer(
            self.q,
            self.keys,
            self.table,
            self.offsets,
            self.lengths,
            self.positions,
            self.scan_active,
            self.scores,
            self.valid,
        )
        c.select_decode_candidates(
            self.scores,
            self.geometry,
            plan=self.plan,
            workspace=self.workspace,
            out=self.workspace.candidates,
            arm=self.arm,
        )

    def snapshot(self):
        # Assertion-only readback, outside the captured execution path.
        return self.scores.view(
            self.torch.int32
        ).cpu().clone(), self.valid.cpu().clone()

    def check(self, before):
        torch = self.torch
        scores, valid = self.scores.cpu(), self.valid.cpu()
        candidates = self.workspace.candidates.cpu()
        ids = candidates.view(torch.int32)[..., 1]
        live = torch.zeros_like(valid, dtype=torch.bool)
        origin, stride = (
            self.geometry.scan_block_begin,
            self.geometry.global_block_stride,
        )
        for row, (request, pos, length, active) in enumerate(self.records):
            for head in range(4):
                expected, ordinary = _oracle(
                    self.page_size, self.rank, pos, length, head, request, active
                )
                for block, value in expected.items():
                    column = (block - origin) // stride
                    live[row, head, column] = True
                    assert scores[row, head, column].item() == value
                    assert valid[row, head, column].item() == 1
                expected_ids = [block for block, _ in ordinary] + [-1] * (
                    16 - len(ordinary)
                )
                expected_scores = [value for _, value in ordinary] + [-float("inf")] * (
                    16 - len(ordinary)
                )
                assert ids[row, head].tolist() == expected_ids
                assert candidates[row, head, :, 0].tolist() == expected_scores
        assert torch.equal(scores.view(torch.int32)[~live], before[0][~live])
        assert torch.equal(valid[~live], before[1][~live])


@pytest.mark.gpu
@pytest.mark.parametrize("page_size", [128])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("query_len", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "arm",
    [c.SelectArm.RADIX_BOUNDED, c.SelectArm.RADIX_FULL_ROW],
    ids=lambda arm: arm.name.lower(),
)
def test_public_msa_noinit_scan_and_affine_selector_full_replay(
    cuda_device,
    msa_api,
    page_size,
    rank,
    query_len,
    arm,
):
    import torch

    session = _Composition(torch, msa_api, cuda_device, page_size, rank, query_len, arm)
    session.phase()
    session.poison_outputs()
    before = session.snapshot()
    session.run()
    session.check(before)
    pointers = tuple(
        t.data_ptr()
        for t in (
            session.q,
            session.keys,
            session.table,
            session.offsets,
            session.lengths,
            session.positions,
            session.active,
            session.scores,
            session.valid,
            session.geometry.local_valid_blocks,
            session.geometry.forced_column,
            session.workspace.candidates,
        )
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        session.run()
    # Long -> short retains real prior scores in now-undefined tails. The
    # collapsed first request moves Q offsets; Q3/Q4 cross global B128 parity.
    for short in (True, False):
        session.phase(short=short)
        if not short:
            session.poison_outputs()
        before = session.snapshot()
        graph.replay()
        session.check(before)
    assert (
        tuple(
            t.data_ptr()
            for t in (
                session.q,
                session.keys,
                session.table,
                session.offsets,
                session.lengths,
                session.positions,
                session.active,
                session.scores,
                session.valid,
                session.geometry.local_valid_blocks,
                session.geometry.forced_column,
                session.workspace.candidates,
            )
        )
        == pointers
    )
