"""GPU: the ICP prefill selector, OnlyScoreIcp and the CuTe decode scan against torch oracles."""

import math
import os

import pytest
import torch

from fmha_sm100 import api
from fmha_sm100.icp import _build
from fmha_sm100.icp.scorer.decode import icp_decode_score as scorer

pytestmark = pytest.mark.gpu
PAGE, FRAG, DIM, HEADS = 128, 64, 128, 4
PITCH, INDEX_OFFSET = 45056, 36864  # M3 TP2 compound page and its index region
NEG_INF = -8388608  # float32 -inf bits


def _word(x):
    return torch.tensor([0.0 if x == 0 else x]).view(torch.int32).item()


def _select_oracle(scores, counts, forced, active):
    """C4 rows: per head the 16 best (score desc, id asc) live non-forced columns."""
    out = torch.empty((*scores.shape[:2], 16, 2), dtype=torch.int32)
    out[..., 0], out[..., 1] = NEG_INF, -1
    for t in (t for t, on in enumerate(active) if on):
        for h in range(scores.shape[1]):
            row = [(s, c) for c, s in enumerate(scores[t, h, : counts[t]].tolist()) if c != forced[t]]
            for k, (s, c) in enumerate(sorted(row, key=lambda p: (-p[0], p[1]))[:16]):
                out[t, h, k, 0], out[t, h, k, 1] = _word(s), c
    return out


@pytest.mark.parametrize("columns", [1, 32, 33, 128, 129, 8192])
def test_prefill_selector_every_kernel_width(cuda_device, columns):
    """Both sides of the tiny-warp (32), four-warp (128) and wide (8192) dispatch edges."""
    counts = [columns, columns, columns - 1, min(17, columns), 1, 0, (1 << 31) - 1]
    forced = [-1, columns // 2, -1, 0, -1, -1, (1 << 31) - 1]
    forced = [f if f < n else -1 for f, n in zip(forced, counts)]
    active = [True] * 6 + [False]  # padding keeps arbitrary metadata
    col = torch.arange(columns)
    scores = torch.full((len(counts), HEADS, columns), math.nan)
    for t in range(6):
        live = slice(0, counts[t])
        scores[t, 0, live] = (((col * 37 + t * 11) % 67).float() - 33)[live]
        scores[t, 1, live] = -math.inf
        scores[t, 2, live] = torch.where(col % 2 == 0, 0.0, -0.0)[live]  # signed-zero ties
        scores[t, 3, live] = (0x3F800000 + (col * 73 + t * 17) % 257).int().view(torch.float32)[live]
        if forced[t] >= 0:
            scores[t, :, forced[t]] = math.nan  # never read
    if columns > 1:
        scores[0, :, columns - 1] = 1e30  # a truncated row cannot pass
    expected = _select_oracle(scores, counts, forced, active)
    dev = cuda_device
    out = torch.full((len(counts), HEADS, 16, 2), 0x5A5A5A5A, dtype=torch.int32, device=dev)
    _build._select_ext().select_prefill_candidates(
        scores.to(dev), torch.tensor(counts, dtype=torch.int32, device=dev),
        torch.tensor(forced, dtype=torch.int32, device=dev), torch.tensor(active, device=dev),
        out.view(torch.float32), 0, False, global_block_stride=1)
    assert torch.equal(out.cpu(), expected)


def _requests(query_lens, histories, rank, seed):
    """Fp8 Q, this rank's index fragments in compound pages, a shuffled table, the oracle."""
    g = torch.Generator().manual_seed(seed)
    seqs = [h + q for q, h in zip(query_lens, histories)]
    blocks = [-(-s // PAGE) for s in seqs]
    cols = max(blocks)
    pages = sum(blocks) + 3
    keys = (torch.randn(pages, PAGE, DIM, generator=g) * 0.7).to(torch.float8_e4m3fn)
    perm = torch.randperm(pages, generator=g).int()
    table = torch.full((len(seqs), cols + 2), -(1 << 30), dtype=torch.int32)  # poisoned tail
    for i, (b, start) in enumerate(zip(blocks, [0, *torch.tensor(blocks).cumsum(0).tolist()])):
        table[i, :b] = perm[start:start + b]
    q = (torch.randn(sum(query_lens), HEADS, DIM, generator=g) * 0.7).to(torch.float8_e4m3fn)
    pos = torch.cat([torch.arange(q_) + h for q_, h in zip(query_lens, histories)])
    req = torch.repeat_interleave(torch.arange(len(seqs)), torch.tensor(query_lens))
    seq = torch.tensor(seqs)[req]
    p = torch.arange(cols * PAGE).view(cols, PAGE)
    visible = (p[None] <= pos[:, None, None]) & (p[None] < seq[:, None, None])
    visible &= (torch.arange(PAGE) // FRAG == rank)[None, None]
    k = keys[table[req, :cols].clamp(min=0).long()].float()  # [T, cols, PAGE, DIM]
    dots = torch.einsum("thd,tcpd->thcp", q.float(), k)
    score = torch.where(visible[:, None], dots, -math.inf).amax(-1)
    storage = torch.full((pages * PITCH,), 0x7F, dtype=torch.uint8)
    local = storage.view(torch.float8_e4m3fn).as_strided((pages, FRAG, DIM), (PITCH, DIM, 1), INDEX_OFFSET)
    local.copy_(keys[:, rank * FRAG:(rank + 1) * FRAG])
    return q, storage, table, pos, seqs, score, visible.any(-1)[:, None].expand(-1, HEADS, -1).contiguous()


def _fragment_view(storage, device):
    storage = storage.to(device)
    pages = storage.numel() // PITCH
    return storage, storage.view(torch.float8_e4m3fn).as_strided(
        (pages, FRAG, DIM), (PITCH, DIM, 1), INDEX_OFFSET)


def _write_device_plan(store, query_lens, seqs):
    """Host model of the writer's plan CTAs: every (tile, head, request) once, cheapest CTA."""
    n, ctas = len(query_lens), store.num_ctas
    qo_off = [0, *torch.tensor(query_lens).cumsum(0).tolist()]
    kv_off = [0, *torch.tensor(seqs).cumsum(0).tolist()]
    causal = [s - q for q, s in zip(query_lens, seqs)]
    rows = [qo_off, kv_off, query_lens, seqs, causal]
    for r, values in enumerate(rows):
        store.segments[0, r, : len(values)] = torch.tensor(values, dtype=torch.int32)
    cost, buckets = [165] * ctas, [[] for _ in range(ctas)]
    for qt in range(max(-(-q // 128) for q in query_lens) - 1, -1, -1):
        for b in (b for b in range(n) if qt < -(-query_lens[b] // 128)):
            iters = -(-max(0, min((qt + 1) * 128 + causal[b], seqs[b])) // 256)
            for h in range(store.num_heads):
                cta = min(range(ctas), key=lambda c: (cost[c], c))
                buckets[cta].append((qt << 32) | (h << 16) | b)
                cost[cta] += 43 * iters + 110
    ranges, info = [], []
    for items in buckets:
        ranges.append(len(info) | ((len(info) + len(items)) << 32))
        info += items
    store.work[0, :ctas] = torch.tensor(ranges)
    store.work[0, ctas:ctas + len(info)] = torch.tensor(info)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("query_lens,histories", [
    ([300, 64, 65, 1, 129], [1000, 0, 0, 5000, 200]),  # mixed; rank 1 has no Q64 fragment
    ([1, 2, 3, 1], [70000, 129, 63, 0]),  # non-uniform decode: the single-warpgroup variant
])
def test_onlyscoreicp_on_a_writer_model_device_plan(cuda_device, rank, query_lens, histories):
    q, storage, table, _, seqs, score_ref, valid_ref = _requests(query_lens, histories, rank, 7)
    cols, dev = score_ref.shape[-1], cuda_device
    store = api._IcpDevicePlanStore(device=dev, num_slots=1, max_segments=len(seqs),
                                    num_heads=HEADS, max_chunk_tokens=q.shape[0])
    _write_device_plan(store, query_lens, seqs)
    table = table.to(dev)
    plan = api._fmha_sm100_icp_plan(
        torch.tensor(query_lens, dtype=torch.int32), HEADS, icp_rank=rank, block_table=table,
        block_table_row_stride=table.stride(0), block_table_row_begin=0,
        device_plan=store.slot(0), max_k_tiles=cols)
    _storage, keys = _fragment_view(storage, dev)
    score = torch.full((q.shape[0], HEADS, cols + 3), math.nan, device=dev)
    valid = torch.full_like(score, 201, dtype=torch.uint8)
    api._fmha_sm100_icp(q.to(dev), keys.unsqueeze(1), plan, max_score=score[..., :cols],
                        valid_score=valid[..., :cols], sm_scale=DIM ** -0.5,
                        qo_offset=store.segments[0, 4, : len(seqs)])
    score, valid = score.cpu(), valid.cpu()
    assert torch.equal(valid[..., :cols].bool(), valid_ref)
    assert bool(torch.isneginf(score[..., :cols][~valid_ref]).all())
    torch.testing.assert_close(score[..., :cols][valid_ref], score_ref[valid_ref], rtol=2e-5, atol=1e-4)
    assert bool(torch.isnan(score[..., cols:]).all()) and bool((valid[..., cols:] == 201).all())


@pytest.fixture(scope="module")
def decode_scan(cuda_device):
    arch = torch.cuda.get_device_capability(cuda_device)
    # The export targets the device arch; the DSL reads it at first import.
    os.environ.setdefault("CUTE_DSL_ARCH", f"sm_{arch[0]}{arch[1]}a")

    def get(query_len, rank):
        if not scorer._aot_path(query_len, rank, arch).is_file():
            scorer._export_aot_scan(query_len, rank, arch)
        return scorer._compile_scan(query_len, rank, cuda_device.index, arch)

    return get


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("query_len", [1, 2, 3, 4])
def test_cute_decode_scan_matches_torch(cuda_device, decode_scan, query_len, rank):
    seqs = [1, 31, 64, 65, 128, 129, 385, 4000]
    query_lens = [query_len] * len(seqs)
    histories = [max(0, s - query_len) for s in seqs]
    seqs = [h + query_len for h in histories]
    q, storage, table, pos, _, score_ref, valid_ref = _requests(query_lens, histories, rank, 3)
    dev, tokens, cols = cuda_device, q.shape[0], score_ref.shape[-1]
    _storage, keys = _fragment_view(storage, dev)
    qsl = torch.arange(0, tokens + 1, query_len, dtype=torch.int32, device=dev)
    active = torch.ones(tokens, dtype=torch.uint8, device=dev)
    active[-1] = 0  # an inactive row: undefined output, never read
    split = (len(seqs) // 2) * query_len  # two windows, as two chunks of one step
    for t0, t1 in ((0, split), (split, tokens)):
        score = torch.full((t1 - t0, HEADS, cols), math.nan, device=dev)
        valid = torch.zeros_like(score, dtype=torch.uint8)
        decode_scan(query_len, rank)(
            q.to(dev), keys, table.to(dev), score, torch.tensor(seqs, dtype=torch.int32, device=dev),
            qsl, pos.to(dev), active, valid, t0, t1 - t0, t0 // query_len,
            (t1 - t0) // query_len, 128)
        live = valid_ref[t0:t1].clone()
        live[(active[t0:t1] == 0).cpu()] = False
        score, valid = score.cpu(), valid.cpu()
        assert bool((valid[live] == 1).all())
        torch.testing.assert_close(score[live], score_ref[t0:t1][live], rtol=2e-5, atol=1e-4)
