# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Independent FP8 fixtures for the refined-icp-v1 TP2/B128/R64 scorer.

The oracle owns full logical B128 pages. Packing a rank's compact component is
test input construction only: the reference never reads that compact view or
the producer's address helper. The default small-integer data has exactly
representable FP8 values and exact FP32 D128 dot products. ``random_seed`` opts
into finite normal data for a separate numerical qualification.

Construction performs no scoring, timing, or reference computation. Profilers
can reuse the same full_q/full_index_keys bytes with ``fixture.for_rank(1)``.
"""

from dataclasses import dataclass, replace
from typing import Sequence

import torch


FIXTURE_VERSION = "refined-icp-v1.cute-decode-fixtures.1"
COMPOUND_LAYOUTS = {"nvfp4": (45056, 36864), "fp8": (73728, 65536)}
PAD_PAGE = -2147480000
HEADS = 4
DIM = 128
PAGE = 128
FRAGMENT = 64


def _compound_view(full_keys: torch.Tensor, rank: int, main_format: str):
    page_bytes, index_offset = COMPOUND_LAYOUTS[main_format]
    # 0x7f is an E4M3 NaN: a narrowed page pitch or extra rank offset is poison.
    backing = torch.full(
        (full_keys.shape[0] * page_bytes,),
        0x7F,
        dtype=torch.uint8,
        device=full_keys.device,
    )
    view = backing.view(torch.float8_e4m3fn).as_strided(
        (full_keys.shape[0], FRAGMENT, DIM),
        (page_bytes, DIM, 1),
        index_offset,
    )
    # Split the logical row axis before selecting rank; rank is not a byte offset
    # within the destination's already rebased local component.
    view.copy_(full_keys.reshape(-1, 2, FRAGMENT, DIM)[:, rank])
    return backing, view


def _outputs(rows: int, blocks: int, device, strided: bool):
    if strided:
        head_stride = blocks + 8
        row_stride = HEADS * head_stride + 11
        score_storage = torch.full(
            (max(rows, 1) * row_stride,),
            123456.0,
            dtype=torch.float32,
            device=device,
        )
        valid_storage = torch.full_like(score_storage, 77, dtype=torch.uint8)
        layout = ((rows, HEADS, blocks), (row_stride, head_stride, 1))
        scores = score_storage.as_strided(*layout)
        valid = valid_storage.as_strided(*layout)
    else:
        scores = torch.empty((rows, HEADS, blocks), dtype=torch.float32, device=device)
        valid = torch.empty_like(scores, dtype=torch.uint8)
        score_storage, valid_storage = scores, valid
    scores.fill_(float("nan"))
    valid.fill_(9)
    return scores, valid, score_storage, valid_storage


@dataclass
class DecodeFixture:
    query_len: int
    rank: int
    main_format: str
    token_begin: int
    token_count: int
    request_begin: int
    request_count: int
    split_k: int
    full_q: torch.Tensor
    full_index_keys: torch.Tensor
    index_k: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    active_rows: torch.Tensor
    score_out: torch.Tensor
    valid_out: torch.Tensor
    compound_storage: torch.Tensor
    score_storage: torch.Tensor
    valid_storage: torch.Tensor
    host_full_index_keys: torch.Tensor
    strided_outputs: bool = False

    @property
    def index_q(self):
        return self.full_q

    @property
    def compound_page_bytes(self):
        return COMPOUND_LAYOUTS[self.main_format][0]

    @property
    def index_component_offset_bytes(self):
        return COMPOUND_LAYOUTS[self.main_format][1]

    @property
    def inputs(self):
        return (
            self.full_q,
            self.index_k,
            self.block_table,
            self.query_start_loc,
            self.seq_lens,
            self.positions,
            self.active_rows,
            self.score_out,
            self.valid_out,
        )

    def prepare_scorer(self):
        from fmha_sm100.icp.scorer.decode.icp_decode_score import get_icp_decode_scorer

        window = dict(
            token_begin=self.token_begin,
            token_count=self.token_count,
            request_begin=self.request_begin,
            request_count=self.request_count,
        )
        return get_icp_decode_scorer(
            query_len=self.query_len,
            rank=self.rank,
            world_size=2,
            split_k=self.split_k,
            device=self.full_q.device,
            **window,
        )

    def for_rank(self, rank: int):
        if rank not in (0, 1):
            raise ValueError("This fixture implements TP2 only")
        backing, keys = _compound_view(self.full_index_keys, rank, self.main_format)
        scores, valid, score_storage, valid_storage = _outputs(
            self.score_out.shape[0],
            self.score_out.shape[2],
            self.full_q.device,
            self.strided_outputs,
        )
        return replace(
            self,
            rank=rank,
            index_k=keys,
            compound_storage=backing,
            score_out=scores,
            valid_out=valid,
            score_storage=score_storage,
            valid_storage=valid_storage,
        )

    def window(self, token_begin: int, token_count: int):
        if token_begin < 0 or token_begin + token_count > self.full_q.shape[0]:
            raise ValueError("Window must be inside the full invocation")
        scores, valid, score_storage, valid_storage = _outputs(
            token_count,
            self.score_out.shape[2],
            self.full_q.device,
            self.strided_outputs,
        )
        return replace(
            self,
            token_begin=token_begin,
            token_count=token_count,
            score_out=scores,
            valid_out=valid,
            score_storage=score_storage,
            valid_storage=valid_storage,
        )

    def poison_outputs(self):
        self.score_out[: self.token_count].fill_(float("nan"))
        self.valid_out[: self.token_count].fill_(9)

    def output_guard_snapshot(self):
        """Only unadvertised storage: gaps and retained rows past token_count."""
        mask = torch.ones(self.score_storage.numel(), dtype=torch.bool)
        row = torch.arange(self.token_count)[:, None, None]
        head = torch.arange(HEADS)[None, :, None]
        block = torch.arange(self.score_out.shape[2])[None, None, :]
        offsets = (
            row * self.score_out.stride(0)
            + head * self.score_out.stride(1)
            + block * self.score_out.stride(2)
        )
        mask[offsets.flatten()] = False
        return (
            self.score_storage.detach().cpu().flatten()[mask].clone(),
            self.valid_storage.detach().cpu().flatten()[mask].clone(),
        )

    def reference(self, *, rank=None):
        """CPU FP32 oracle over full pages, exact GPU geometry, and actual Q bytes.

        Metadata readbacks here are deliberate test operations outside execution
        and capture. Future/absent rows are masked *before* block max, so NaNs in
        allocated but unwritten key rows cannot contaminate a visible score.
        """
        rank = self.rank if rank is None else rank
        blocks = self.score_out.shape[2]
        scores = torch.full((self.token_count, HEADS, blocks), float("-inf"))
        validity = torch.zeros_like(scores, dtype=torch.uint8)
        table = self.block_table.cpu().to(torch.int64)
        starts = self.query_start_loc.cpu().tolist()
        lengths = self.seq_lens.cpu().tolist()
        positions = self.positions.cpu().tolist()
        active = self.active_rows.cpu().tolist()
        queries = self.full_q.cpu().float()
        owned_rows = torch.arange(PAGE).div(FRAGMENT, rounding_mode="floor") == rank
        for request in range(
            self.request_begin, self.request_begin + self.request_count
        ):
            first = max(starts[request], self.token_begin)
            end = min(starts[request + 1], self.token_begin + self.token_count)
            length = max(lengths[request], 0)
            count = min((length + PAGE - 1) // PAGE, blocks)
            if count == 0 or first >= end:
                continue
            pages = table[request, :count]
            if bool(
                ((pages < 0) | (pages >= self.host_full_index_keys.shape[0])).any()
            ):
                raise AssertionError(
                    "Oracle attempted to read a poisoned page-table tail"
                )
            keys = self.host_full_index_keys[pages].float()
            key_positions = torch.arange(count * PAGE).reshape(count, PAGE)
            for token in range(first, end):
                if not active[token]:
                    continue
                visible = (
                    (key_positions < length)
                    & (key_positions <= positions[token])
                    & owned_rows[None, :]
                )
                # Full D128 dot products and one max per logical B128 block.
                dots = (queries[token] @ keys.flatten(0, 1).T).reshape(
                    HEADS, count, PAGE
                )
                masked = torch.where(visible[None, :, :], dots, float("-inf"))
                row = token - self.token_begin
                scores[row, :, :count] = masked.amax(dim=-1)
                validity[row, :, :count] = visible.any(dim=-1).to(torch.uint8)[None, :]
        return scores, validity


def make_decode_fixture(
    *,
    query_len: int,
    batch_size: int,
    kv_lens: int | Sequence[int] = 385,
    rank: int = 0,
    main_format: str = "nvfp4",
    capacity_blocks: int | None = None,
    device="cuda",
    random_seed: int | None = None,
    request_query_lens: Sequence[int] | None = None,
    token_begin: int = 0,
    token_count: int | None = None,
    retained_rows: int | None = None,
    strided_outputs: bool = False,
    strided_queries: bool = False,
    poison_future: bool = False,
    split_k: int = 256,
):
    """Build inputs with nonidentity pages and poisoned unmaterialized table cells.

    Every genuine request has Q rows; zero-length request intervals represent
    graph padding and collapse query_start_loc. The full Q allocation remains
    batch_size*Q rows. K and table capacities depend on materialized history and
    advertised block capacity independently (e.g. 150000 live, 8192 advertised).
    """
    if query_len not in (1, 2, 3, 4) or rank not in (0, 1):
        raise ValueError("Fixture supports Q1..4, TP2")
    lengths = [kv_lens] * batch_size if isinstance(kv_lens, int) else list(kv_lens)
    query_lengths = (
        [query_len] * batch_size
        if request_query_lens is None
        else list(request_query_lens)
    )
    if len(lengths) != batch_size or len(query_lengths) != batch_size:
        raise ValueError("Expected one length per request slot")
    if any(n < 0 for n in lengths) or any(
        q not in (0, query_len) for q in query_lengths
    ):
        raise ValueError(
            "Lengths must be nonnegative and query intervals uniform or empty"
        )
    pages_per_request = [(length + PAGE - 1) // PAGE for length in lengths]
    required_blocks = max(pages_per_request, default=0)
    capacity_blocks = (
        max(required_blocks + 3, 1) if capacity_blocks is None else capacity_blocks
    )
    if capacity_blocks < required_blocks:
        raise ValueError("Output capacity cannot truncate the fixture's live history")
    total_tokens = batch_size * query_len
    token_count = total_tokens - token_begin if token_count is None else token_count
    retained_rows = token_count if retained_rows is None else retained_rows
    if not 0 <= token_begin <= token_begin + token_count <= total_tokens:
        raise ValueError("Invalid token window")
    if retained_rows < token_count:
        raise ValueError("Retained output has fewer rows than the advertised window")
    seed = 20260914 if random_seed is None else random_seed
    generator = torch.Generator(device="cpu").manual_seed(seed)
    nphysical = sum(pages_per_request) + 8
    physical_order = torch.randperm(nphysical, generator=generator)
    host_keys = torch.full(
        (nphysical, PAGE, DIM),
        float("nan"),
        dtype=torch.float8_e4m3fn,
    )
    # Allocate only a small FP32 generation tile even for BS16/150K histories.
    live_pages = physical_order[: sum(pages_per_request)]
    for begin in range(0, live_pages.numel(), 64):
        target = live_pages[begin : begin + 64]
        shape = (target.numel(), PAGE, DIM)
        raw = (
            torch.randint(-4, 5, shape, generator=generator).float()
            if random_seed is None
            else torch.randn(shape, generator=generator) * 0.7
        )
        host_keys[target] = raw.to(torch.float8_e4m3fn)
    table_host = torch.full(
        (batch_size, capacity_blocks + 5), PAD_PAGE, dtype=torch.int32
    )
    cursor = 0
    for request, count in enumerate(pages_per_request):
        table_host[request, :count] = live_pages[cursor : cursor + count].to(
            torch.int32
        )
        if poison_future and count and lengths[request] % PAGE:
            last_page = int(live_pages[cursor + count - 1])
            host_keys[last_page, lengths[request] % PAGE :] = float("nan")
        cursor += count
    table = table_host.to(device)[:, :capacity_blocks]
    q_shape = (total_tokens, HEADS, DIM)
    q_raw = (
        torch.randint(-3, 4, q_shape, generator=generator).float()
        if random_seed is None
        else torch.randn(q_shape, generator=generator) * 0.7
    )
    q_encoded = q_raw.to(torch.float8_e4m3fn).to(device)
    if strided_queries:
        q_storage = torch.empty(
            total_tokens * 608, dtype=torch.float8_e4m3fn, device=device
        )
        q = q_storage.as_strided(q_shape, (608, 144, 1))
        q.copy_(q_encoded)
    else:
        q = q_encoded
    full_keys = host_keys.to(device)
    backing, index_k = _compound_view(full_keys, rank, main_format)
    starts = [0]
    positions = [-1] * total_tokens
    activity = [False] * total_tokens
    for length, query_count in zip(lengths, query_lengths):
        begin = starts[-1]
        for slot in range(query_count):
            positions[begin + slot] = length - query_count + slot
            activity[begin + slot] = True
        starts.append(begin + query_count)
    scores, valid, score_storage, valid_storage = _outputs(
        retained_rows,
        capacity_blocks,
        device,
        strided_outputs,
    )
    return DecodeFixture(
        query_len=query_len,
        rank=rank,
        main_format=main_format,
        token_begin=token_begin,
        token_count=token_count,
        request_begin=0,
        request_count=batch_size,
        split_k=split_k,
        full_q=q,
        full_index_keys=full_keys,
        index_k=index_k,
        block_table=table,
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
        # Built as torch `bool` and handed over as a `uint8` VIEW, because that
        # is exactly what the vLLM caller does: the liveness plane it owns is a
        # bool buffer and the scorer's ABI takes the byte type. `view` shares
        # `data_ptr` and allocates nothing, so this exercises the graph-safe
        # form of the conversion rather than a copy.
        active_rows=torch.tensor(
            activity,
            dtype=torch.bool,
            device=device,
        ).view(torch.uint8),
        score_out=scores,
        valid_out=valid,
        compound_storage=backing,
        score_storage=score_storage,
        valid_storage=valid_storage,
        host_full_index_keys=host_keys,
        strided_outputs=strided_outputs,
    )
