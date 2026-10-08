# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""ICP_DEVICE_PLAN_ABI 3: the OnlyScoreIcp prefill plan is derived on device.

The fused writer's plan CTAs replace the host staging copies of
``_plan_buf_from_list`` plus the separate plan-kernel launch for direct-table
ICP chunks.  These CPU tests compare, over many varlen batches:

* ``_host_reference``: the SHIPPED host list builder (the packed-list ICP route
  of ``_fmha_sm100_plan_impl`` shares the list derivation the direct-table
  route used before this change), run with the vLLM builder's chunking and
  bound lengths, followed by a Python port of ``plan.cuh::direct_greedy``;
* ``device_plan_model``: a line-by-line Python mirror of
  ``icpDevicePlanBlock`` in ``csrc/fused_indexer_nvfp4_kv_write.cu``.

``test_native_planner_matches_model`` runs the actual CUDA planner through a
small ctypes shim when ``nvcc`` and a GPU are present (skipped otherwise; it is
also the first step of the GPU gate).
"""

from __future__ import annotations

import contextlib
import importlib
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[4] / "python"
sys.path.insert(0, str(_REPO))

api = importlib.import_module("fmha_sm100.icp.scorer.prefill.api")

QO_TILE, KV_TILE = 128, 256
HEADS = 4
ICP_C, ICP_RANK, ROWS = 2, 1, 64


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------
def _kv_iters(qt, kv_len, offset):
    eff = min((qt + 1) * QO_TILE + offset, kv_len)
    return 0 if eff <= 0 else -(-eff // KV_TILE)


def _tile_cost(iters):
    return int(np.float32(43.0) * np.float32(iters) + np.float32(110.0))


def direct_greedy_reference(qo_lens, kv_lens, offsets, num_heads, nb):
    """plan.cuh direct_greedy, nosplit/causal/pack 1, both passes."""
    n = len(qo_lens)
    max_tiles = max((-(-q // QO_TILE) for q in qo_lens), default=0)

    def assignments():
        cost = [165] * nb
        for qt in range(max_tiles - 1, -1, -1):
            for b in range(n):
                if qt >= -(-qo_lens[b] // QO_TILE):
                    continue
                ki = _kv_iters(qt, kv_lens[b], offsets[b])
                if ki <= 0:
                    continue
                tc = _tile_cost(ki)
                h_off = 0
                while h_off < num_heads:
                    bc = min(num_heads - h_off, nb)
                    order = sorted(range(nb), key=lambda i: (cost[i], i))[:bc]
                    for rank, bucket in enumerate(order):
                        yield bucket, (qt, h_off + rank, b)
                    for bucket in order:
                        cost[bucket] += tc
                    h_off += bc

    counts = [0] * nb
    for bucket, _ in assignments():
        counts[bucket] += 1
    offs = [0] * (nb + 1)
    for i in range(nb):
        offs[i + 1] = offs[i] + counts[i]
    info = [None] * offs[nb]
    fill = [0] * nb
    for bucket, item in assignments():
        info[offs[bucket] + fill[bucket]] = item
        fill[bucket] += 1
    ranges = [(offs[i], offs[i] + counts[i]) for i in range(nb)]
    return ranges, info


def _request_for_row(qsl, num_reqs, row):
    lo, hi = 0, num_reqs
    while lo < hi:
        mid = (lo + hi) >> 1
        if qsl[mid + 1] <= row:
            lo = mid + 1
        else:
            hi = mid
    return lo


OPEN_END = 0x7FFFFFFF
MIN_SPLIT_TILES = api._WRITER_MIN_SPLIT_TILES
TILE_PAGES = api.ICP_PLAN_TILE_PAGES


def local_trip(qt, kv_len, offset, *, rank=ICP_RANK, world=ICP_C, rows=ROWS):
    """The scorer loader's OnlyScoreIcp compute_effective_end, in KV tiles."""
    gb_avail = -(-kv_len // rows)
    gb_causal = -(-((qt + 1) * QO_TILE + offset) // rows)
    lb = max(-(-(min(gb_avail, gb_causal) - rank) // world), 1)
    return -(-lb // TILE_PAGES)


def device_plan_model(
    qsl,
    seq_lens,
    num_reqs,
    num_rows,
    width,
    slot,
    num_heads,
    nb,
    *,
    max_splits=1,
    row_begin=0,
    min_split_tiles=MIN_SPLIT_TILES,
    rank=ICP_RANK,
):
    """Mirror of icpDevicePlanBlock (split pre-pass, single rank pass, scatter)."""
    t0 = max(slot * width, row_begin)
    t1 = min((slot + 1) * width, num_rows)
    live_end = qsl[num_reqs] if num_reqs > 0 else 0
    end = min(t1, live_end)
    first = n = 0
    if t0 < end:
        first = _request_for_row(qsl, num_reqs, t0)
        n = _request_for_row(qsl, num_reqs, end - 1) - first + 1
    qo_off, kv_off = [0] * (n + 1), [0] * (n + 1)
    qo_len, kv_len, causal = [0] * n, [0] * n, [0] * n
    for b in range(n):
        req = first + b
        begin, stop, seq = qsl[req], qsl[req + 1], seq_lens[req]
        lo, hi = max(begin, t0), min(stop, t1)
        qo_off[b + 1] = hi - t0
        qo_len[b] = hi - lo
        kv_len[b] = seq
        causal[b] = seq - (stop - lo)
        kv_off[b + 1] = kv_off[b] + seq
    max_tiles = max((-(-q // QO_TILE) for q in qo_len), default=0)
    tiles = [
        (qt, b)
        for qt in range(max_tiles - 1, -1, -1)
        for b in range(n)
        if qt < -(-qo_len[b] // QO_TILE)
    ]
    split = max_splits > 1
    chunk = 0
    if split:
        total = sum(local_trip(qt, kv_len[b], causal[b], rank=rank) for qt, b in tiles)
        split = 0 < len(tiles) * num_heads < nb
        if split:
            chunk = max(min_split_tiles, -(-(total * num_heads) // nb))
    cost, count = [165] * nb, [0] * nb
    scratch = []
    for qt, b in tiles:
        ki = _kv_iters(qt, kv_len[b], causal[b])
        if ki <= 0:
            continue
        trip, piece, pieces = 0, 0, 1
        if split:
            trip = local_trip(qt, kv_len[b], causal[b], rank=rank)
            pieces = min(max_splits, max(1, trip // chunk))
            piece = -(-trip // pieces)
            pieces = -(-trip // piece)
        for sp in range(pieces):
            kb = sp * piece
            ke = OPEN_END if sp == pieces - 1 else kb + piece
            tc = _tile_cost(min(kb + piece, trip) - kb) if split else _tile_cost(ki)
            h_off = 0
            while h_off < num_heads:
                bc = min(num_heads - h_off, nb)
                ranks = {}
                for tid in range(nb):
                    ranks[tid] = sum(
                        1
                        for i in range(nb)
                        if cost[i] < cost[tid] or (cost[i] == cost[tid] and i < tid)
                    )
                took = [None] * bc
                for tid in range(nb):
                    if ranks[tid] < bc:
                        took[ranks[tid]] = (tid, count[tid])
                for rk, (tid, pos) in enumerate(took):
                    scratch.append(((qt, h_off + rk, b), (kb, ke, sp), tid, pos))
                    count[tid] += 1
                    cost[tid] += tc
                h_off += bc
    offs = [0] * nb
    for i in range(1, nb):
        offs[i] = offs[i - 1] + count[i - 1]
    info = [None] * len(scratch)
    kv_ranges = [None] * len(scratch)
    for item, kv_range, bucket, pos in scratch:
        info[offs[bucket] + pos] = item
        kv_ranges[offs[bucket] + pos] = kv_range
    return {
        "header": (n, first, len(scratch), t0),
        "qo_segment_offsets": qo_off,
        "kv_segment_offsets": kv_off,
        "qo_segment_lens": qo_len,
        "kv_segment_lens": kv_len,
        "qo_offset": causal,
        "ranges": [(offs[i], offs[i] + count[i]) for i in range(nb)],
        "info": info,
        "kv_ranges": kv_ranges,
    }


@contextlib.contextmanager
def _cpu_planner(nb, spy=None):
    """Run the shipped planner on CPU: stub CUDA-only helpers."""
    saved = {
        k: getattr(api, k)
        for k in ("_call_plan", "_alloc_workspace_buf", "_get_num_cta")
    }
    cache = {}

    def _buf(tag, size, device, dtype):
        key = (int(tag), dtype)
        if key not in cache or cache[key].numel() < size:
            cache[key] = torch.empty(size, dtype=dtype)
        return cache[key]

    api._call_plan = spy if spy is not None else (lambda *a, **k: None)
    api._alloc_workspace_buf = _buf
    api._get_num_cta = lambda device: nb
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(api, k, v)


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        qo_segment_offsets,
        qo_segment_lens,
        kv_segment_lens,
        packed_work_range,
        packed_work_info,
        qo_tile_size,
        kv_tile_size,
        num_qo_heads,
        num_ctas,
        causal,
        qo_offset,
        *args,
        **kwargs,
    ):
        self.calls.append(
            {
                "qo_segment_offsets": qo_segment_offsets.tolist(),
                "qo_segment_lens": qo_segment_lens.tolist(),
                "kv_segment_lens": kv_segment_lens.tolist(),
                "qo_offset": qo_offset.tolist(),
                "qo_tile_size": qo_tile_size,
                "kv_tile_size": kv_tile_size,
                "num_qo_heads": num_qo_heads,
                "num_ctas": num_ctas,
                "causal": causal,
            }
        )


def _host_reference(qsl, kv_bound, num_tokens, width, nb, row_begin=0):
    """vLLM `_build_icp` chunking + the shipped host list builder + greedy."""
    qsl_t = torch.tensor(qsl, dtype=torch.int64)
    q_begin, q_end = qsl_t[:-1], qsl_t[1:]
    kv_ub = torch.tensor(kv_bound, dtype=torch.int64)
    out = {}
    for slot in range(-(-num_tokens // width)):
        # lane D's FMHA window: slot rows at or past the CuTe band end.
        t0 = max(slot * width, row_begin)
        t1 = min((slot + 1) * width, num_tokens)
        if t0 >= t1:
            continue
        lo, hi = q_begin.clamp(t0, t1), q_end.clamp(t0, t1)
        sel = (hi - lo > 0).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        first, last = int(sel[0]), int(sel[-1])
        assert last - first + 1 == sel.numel()
        kv_sel = kv_ub[sel]
        spy = _Spy()
        with _cpu_planner(nb, spy):
            plan = api._fmha_sm100_plan_impl(
                (hi[sel] - lo[sel]).to(torch.int32),
                kv_sel.to(torch.int32),
                HEADS,
                num_kv_heads=1,
                qo_offset=(kv_sel - q_end[sel] + lo[sel]).to(torch.int32),
                page_size=ROWS,
                output_maxscore=True,
                causal=True,
                num_kv_splits=1,
                icp_c=ICP_C,
                icp_rank=ICP_RANK,
                device="cpu",
            )
        (call,) = spy.calls
        assert (call["qo_tile_size"], call["kv_tile_size"]) == (QO_TILE, KV_TILE)
        assert call["num_ctas"] == nb and call["causal"]
        ranges, info = direct_greedy_reference(
            call["qo_segment_lens"],
            call["kv_segment_lens"],
            call["qo_offset"],
            call["num_qo_heads"],
            nb,
        )
        out[slot] = {
            "header": (sel.numel(), first),
            "qo_segment_offsets": call["qo_segment_offsets"],
            "kv_segment_offsets": plan["kv_segment_offsets"].tolist(),
            "qo_segment_lens": call["qo_segment_lens"],
            "kv_segment_lens": call["kv_segment_lens"],
            "qo_offset": call["qo_offset"],
            "ranges": ranges,
            "info": info,
        }
    return out


def _random_batch(rng, *, exact):
    num_reqs = rng.randint(1, 12)
    q, ctx, drafts = [], [], []
    for _ in range(num_reqs):
        kind = rng.random()
        if kind < 0.35:
            q.append(rng.randint(1, 4))  # decode / verify
            ctx.append(rng.randint(0, 300000))
            drafts.append(0 if exact else rng.randint(0, 3))
        else:
            q.append(
                rng.choice(
                    [1, 63, 64, 127, 128, 129, 255, 256, 257, rng.randint(1, 3000)]
                )
            )
            ctx.append(rng.choice([0, 0, 1, 127, 128, rng.randint(0, 200000)]))
            drafts.append(0)
    qsl = [0]
    for x in q:
        qsl.append(qsl[-1] + x)
    seq = [c + x for c, x in zip(ctx, q)]
    bound = [s + d for s, d in zip(seq, drafts)]
    pad = rng.choice([0, 0, 1, 7, 64])
    return qsl, seq, bound, num_reqs, qsl[-1] + pad


def _as_items(info):
    return sorted(info)


def _check_complete(plan, nb):
    ranges, info = plan["ranges"], plan["info"]
    assert ranges[0][0] == 0
    for (a0, a1), (b0, _) in itertools.pairwise(ranges):
        assert a0 <= a1 == b0
    assert ranges[-1][1] == len(info)
    expected = {
        (qt, h, b)
        for b, q in enumerate(plan["qo_segment_lens"])
        for qt in range(-(-q // QO_TILE))
        for h in range(HEADS)
    }
    assert len(info) == len(set(info)) == len(expected)
    assert set(info) == expected


# ---------------------------------------------------------------------------
# tests: device plan vs host builder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [128, 256, 1024, 2048])
@pytest.mark.parametrize("nb", [3, 16, 148])
def test_device_plan_equals_host_plan_when_bounds_are_exact(width, nb):
    rng = random.Random(width * 1000 + nb)
    for _ in range(40):
        qsl, seq, bound, num_reqs, num_tokens = _random_batch(rng, exact=True)
        host = _host_reference(qsl, bound, num_tokens, width, nb)
        extent = num_tokens + rng.choice([0, 5, width])
        for slot in range(-(-extent // width)):
            dev = device_plan_model(qsl, seq, num_reqs, extent, width, slot, HEADS, nb)
            if slot not in host:
                assert dev["header"][0] == 0 and dev["info"] == []
                continue
            ref = host[slot]
            assert dev["header"][:2] == ref["header"]
            for key in (
                "qo_segment_offsets",
                "kv_segment_offsets",
                "qo_segment_lens",
                "kv_segment_lens",
                "qo_offset",
                "ranges",
                "info",
            ):
                assert dev[key] == ref[key], key
            _check_complete(dev, nb)


@pytest.mark.parametrize("width", [256, 1024])
def test_device_plan_uses_exact_lengths_under_rejected_drafts(width):
    nb = 148
    rng = random.Random(7 + width)
    checked = 0
    for _ in range(60):
        qsl, seq, bound, num_reqs, num_tokens = _random_batch(rng, exact=False)
        host = _host_reference(qsl, bound, num_tokens, width, nb)
        for slot, ref in host.items():
            dev = device_plan_model(
                qsl, seq, num_reqs, num_tokens, width, slot, HEADS, nb
            )
            n, first = ref["header"]
            assert dev["header"][:2] == (n, first)
            assert dev["qo_segment_offsets"] == ref["qo_segment_offsets"]
            assert dev["qo_segment_lens"] == ref["qo_segment_lens"]
            assert dev["kv_segment_lens"] == seq[first : first + n]
            t0 = slot * width
            exact_offsets = [
                seq[first + b] - (qsl[first + b + 1] - max(qsl[first + b], t0))
                for b in range(n)
            ]
            assert dev["qo_offset"] == exact_offsets
            assert all(o >= 0 for o in exact_offsets)
            # Same work set as the bound-sized host plan; only balance moves.
            assert _as_items(dev["info"]) == _as_items(ref["info"])
            ranges, info = direct_greedy_reference(
                dev["qo_segment_lens"],
                dev["kv_segment_lens"],
                dev["qo_offset"],
                HEADS,
                nb,
            )
            assert (dev["ranges"], dev["info"]) == (ranges, info)
            _check_complete(dev, nb)
            checked += 1
    assert checked > 50


def test_device_causal_offsets_match_live_metadata_formula():
    rng = random.Random(3)
    width = 512
    for _ in range(50):
        qsl, seq, _, num_reqs, num_tokens = _random_batch(rng, exact=True)
        for slot in range(-(-num_tokens // width)):
            dev = device_plan_model(
                qsl, seq, num_reqs, num_tokens, width, slot, HEADS, 16
            )
            n, first = dev["header"][:2]
            for b in range(n):
                req = first + b
                row = max(qsl[req], slot * width)
                # icpLiveMetadataBlock's icp_qo_offsets cell for (slot, b).
                assert dev["qo_offset"][b] == seq[req] - (qsl[req + 1] - row)


# ---------------------------------------------------------------------------
# tests: the host API builds views only
# ---------------------------------------------------------------------------
class _FakeTable:
    """Duck-typed CUDA block table for the planner's host-side checks."""

    dtype = torch.int32
    is_cuda = True

    def __init__(self, rows, pitch):
        self.shape = (rows, pitch)
        self._pitch = pitch

    def dim(self):
        return 2

    def storage_offset(self):
        return 0

    def stride(self, d):
        return (self._pitch, 1)[d]

    def data_ptr(self):
        return 4096


def _store(nb=16, **kw):
    args = {
        "device": "cpu",
        "num_slots": 4,
        "max_segments": 8,
        "num_heads": HEADS,
        "max_chunk_tokens": 1024,
        "num_ctas": nb,
    }
    args.update(kw)
    return api.IcpDevicePlanStore(**args)


def _plan(store, qo, kv, *, slot=0, width=1024, **kw):
    args = {
        "num_kv_heads": 1,
        "page_size": ROWS,
        "output_maxscore": True,
        "causal": True,
        "num_kv_splits": 1,
        "icp_c": ICP_C,
        "icp_rank": ICP_RANK,
        "icp_block_table": _FakeTable(64, 64),
        "icp_block_table_row_stride": 64,
        "icp_block_table_row_begin": 0,
        "device": "cpu",
        "device_plan": store.slot(slot, chunk_width=width),
    }
    args.update(kw)
    return api._fmha_sm100_plan(
        torch.tensor(qo, dtype=torch.int32),
        torch.tensor(kv, dtype=torch.int32),
        HEADS,
        **args,
    )


@contextlib.contextmanager
def _trap_host_staging():
    def _boom(*a, **k):
        raise AssertionError("host plan staging reached")

    saved = (api._plan_buf_from_list, torch.from_numpy, torch.Tensor.copy_)
    api._plan_buf_from_list = _boom
    torch.from_numpy = _boom
    torch.Tensor.copy_ = _boom
    try:
        yield
    finally:
        api._plan_buf_from_list, torch.from_numpy, torch.Tensor.copy_ = saved


def test_direct_table_plan_is_views_of_the_store():
    store = _store()
    with (
        _cpu_planner(16, spy=lambda *a, **k: pytest.fail("plan kernel")),
        _trap_host_staging(),
    ):
        plan = _plan(store, [100, 900], [5000, 900], slot=2)
    expect = {
        "qo_segment_offsets": (store.segments[2, 0], 3),
        "kv_segment_offsets": (store.segments[2, 1], 3),
        "qo_segment_lens": (store.segments[2, 2], 2),
        "kv_segment_lens": (store.segments[2, 3], 2),
        "qo_offset": (store.segments[2, 4], 2),
        "packed_work_range": (store.work[2], 16),
        "packed_work_info": (store.work[2, 16:], store.max_work_items),
    }
    for key, (base, length) in expect.items():
        assert plan[key].data_ptr() == base.data_ptr(), key
        assert plan[key].shape[0] == length, key
    assert plan["kv_page_indptr"] is None
    assert plan["num_kv_splits"] == 1 and plan["max_qo_len"] == 900
    assert plan["max_k_tiles"] == 128
    assert plan["icp_c"] == ICP_C and plan["icp_rank"] == ICP_RANK
    assert plan["index_rows_per_rank"] == ROWS


def test_direct_table_without_device_plan_is_refused():
    with (
        _cpu_planner(16),
        pytest.raises(api.PlanWorkspaceError, match="device-derived"),
    ):
        _plan(_store(), [8], [100], device_plan=None)


def test_device_plan_refuses_mismatched_sizing():
    store = _store()
    with _cpu_planner(16):
        with pytest.raises(api.PlanWorkspaceError, match="requests"):
            _plan(store, [1] * 9, [10] * 9)
        with pytest.raises(api.PlanWorkspaceError, match="chunk width"):
            _plan(store, [600, 600], [600, 600])
        with pytest.raises(api.PlanWorkspaceError, match="query heads"):
            api._fmha_sm100_plan(
                torch.tensor([8], dtype=torch.int32),
                torch.tensor([100], dtype=torch.int32),
                HEADS + 1,
                num_kv_heads=1,
                page_size=ROWS,
                output_maxscore=True,
                causal=True,
                num_kv_splits=1,
                icp_c=ICP_C,
                icp_rank=ICP_RANK,
                icp_block_table=_FakeTable(64, 64),
                icp_block_table_row_stride=64,
                device="cpu",
                device_plan=store.slot(0, chunk_width=1024),
            )
        with pytest.raises(api.PlanWorkspaceError, match="max_model_len"):
            _plan(store, [8], [64 * 128 + 1])
    with _cpu_planner(17), pytest.raises(api.PlanWorkspaceError, match="CTAs"):
        _plan(store, [8], [100])
    with pytest.raises(api.PlanWorkspaceError, match="outside"):
        store.slot(4, chunk_width=128)
    with pytest.raises(api.PlanWorkspaceError, match="chunk_width"):
        store.slot(0, chunk_width=1025)
    with pytest.raises(ValueError, match="icp_block_table"), _cpu_planner(16):
        api._fmha_sm100_plan_impl(
            torch.tensor([8], dtype=torch.int32),
            torch.tensor([100], dtype=torch.int32),
            HEADS,
            page_size=ROWS,
            icp_c=ICP_C,
            icp_rank=ICP_RANK,
            device="cpu",
            device_plan=store.slot(0, chunk_width=128),
        )


def test_worst_chunk_fits_the_work_capacity():
    store = _store(max_segments=8, max_chunk_tokens=1024)
    worst = max(
        HEADS * sum(-(-q // QO_TILE) for q in qs)
        for qs in ([1024], [1] * 7 + [1017], [129] * 7 + [121], [1] * 8, [128] * 8)
    )
    assert worst <= store.max_work_items == HEADS * (8 + 8)


def test_stale_device_plan_is_refused():
    store = _store()
    with _cpu_planner(16):
        plan = _plan(store, [8, 8], [100, 200], slot=1, width=256)
    with pytest.raises(api.PlanWorkspaceError, match="earlier step"):
        api._assert_device_plan_current(plan)
    store.note_writer_launch(chunk_width=128, num_rows=1024)
    with pytest.raises(api.PlanWorkspaceError, match="chunk width"):
        api._assert_device_plan_current(plan)
    store.note_writer_launch(chunk_width=256, num_rows=256)
    with pytest.raises(api.PlanWorkspaceError, match="extent"):
        api._assert_device_plan_current(plan)
    store.note_writer_launch(chunk_width=256, num_rows=300)
    api._assert_device_plan_current(plan)
    # A re-run of the same step (warmup, then capture) stays current.
    store.note_writer_launch(chunk_width=256, num_rows=300)
    api._assert_device_plan_current(plan)


def test_writer_launch_bounds_are_checked():
    store = _store(num_slots=4, max_chunk_tokens=1024)
    store.check_writer_launch(chunk_width=1024, num_rows=4096)
    with pytest.raises(api.PlanWorkspaceError, match="slots"):
        store.check_writer_launch(chunk_width=1024, num_rows=4097)
    with pytest.raises(api.PlanWorkspaceError, match="sized for"):
        store.check_writer_launch(chunk_width=2048, num_rows=2048)
    kw = store.writer_kwargs()
    assert kw["icp_plan_segments"].shape == (4, 5, 9)
    assert kw["icp_plan_work"].shape == (4, 16 + 3 * store.max_work_items)
    assert kw["icp_plan_header"].shape == (4, 4)
    assert (kw["icp_plan_num_ctas"], kw["icp_plan_num_heads"]) == (16, HEADS)


def test_store_refuses_degenerate_bounds():
    for bad in (
        {"num_slots": 0},
        {"max_segments": 0},
        {"num_heads": 0},
        {"max_chunk_tokens": 0},
        {"num_ctas": 257},
        {"max_segments": 0x10000},
    ):
        with pytest.raises(api.PlanWorkspaceError):
            _store(**bad)


def test_abi_markers_agree():
    assert api.ICP_DEVICE_PLAN_ABI_VERSION == 3
    assert api.IcpDevicePlanStore.abi_version == 3
    assert api._ICP_PLAN_ROWS == (
        "qo_segment_offsets",
        "kv_segment_offsets",
        "qo_segment_lens",
        "kv_segment_lens",
        "qo_offset",
    )


# ---------------------------------------------------------------------------
# Native writer/planner execution is tested by its owner, vLLM.
# ---------------------------------------------------------------------------


def _many_segment_batch(rng, num_reqs):
    q = [rng.randint(1, 3) for _ in range(num_reqs)]
    qsl = [0]
    for x in q:
        qsl.append(qsl[-1] + x)
    seq = [rng.randint(0, 5000) + x for x in q]
    return qsl, seq, num_reqs, qsl[-1]


def test_many_segments_per_chunk_match_host():
    # More than one 256-thread scan pass per chunk.
    rng = random.Random(11)
    nb, width = 16, 1024
    qsl, seq, num_reqs, num_tokens = _many_segment_batch(rng, 700)
    host = _host_reference(qsl, seq, num_tokens, width, nb)
    assert max(ref["header"][0] for ref in host.values()) > 256
    for slot, ref in host.items():
        dev = device_plan_model(qsl, seq, num_reqs, num_tokens, width, slot, HEADS, nb)
        assert dev["header"][:2] == ref["header"]
        for key in (
            "qo_segment_offsets",
            "kv_segment_offsets",
            "qo_segment_lens",
            "kv_segment_lens",
            "qo_offset",
            "ranges",
            "info",
        ):
            assert dev[key] == ref[key], key


# ---------------------------------------------------------------------------
# split-KV (ICP_DEVICE_PLAN_ABI 2)
# ---------------------------------------------------------------------------
def split_cells(plan):
    """Check a plan's KV ranges and return {(qt, h, b): covered local pages}.

    Per (qt, h, b): ranges are disjoint, contiguous from 0, cover exactly
    [0, trip) of the scorer's own trip count, and exactly one item is
    open-ended (the tail owner, which completes columns [trip, pwave)).
    """
    trips = {}
    for b in range(len(plan["qo_segment_lens"])):
        for qt in range(-(-plan["qo_segment_lens"][b] // QO_TILE)):
            trips[qt, b] = local_trip(
                qt, plan["kv_segment_lens"][b], plan["qo_offset"][b]
            )
    by_item = {}
    for item, kv in zip(plan["info"], plan["kv_ranges"], strict=True):
        by_item.setdefault(item, []).append(kv)
    cells = {}
    for (qt, h, b), pieces in by_item.items():
        trip = trips[qt, b]
        pieces = sorted(pieces)
        assert sorted(sp for _, _, sp in pieces) == list(range(len(pieces)))
        tails = [kb for kb, ke, _ in pieces if ke == OPEN_END]
        assert len(tails) == 1, f"{(qt, h, b)}: {len(tails)} tail owners"
        assert pieces[-1][1] == OPEN_END, "the tail owner must be the last range"
        covered, cursor = set(), 0
        for kb, ke, _ in pieces:
            assert kb == cursor, f"{(qt, h, b)}: gap or overlap at {kb} vs {cursor}"
            stop = trip if ke == OPEN_END else ke
            assert kb < stop <= trip, f"{(qt, h, b)}: empty or overlong [{kb},{ke})"
            covered |= set(range(kb, stop))
            cursor = stop
        assert covered == set(range(trip))
        cells[qt, h, b] = covered
    assert set(cells) == {(qt, h, b) for (qt, b) in trips for h in range(HEADS)}
    return cells


def _split_batches(rng):
    yield [0, 128], [1_000_000], 128  # 1M, one wave
    yield [0, 1024], [8192], 1024
    yield [0, 100, 228], [150_000, 128], 256
    for _ in range(40):
        qsl, seq, _, _, _ = _random_batch(rng, exact=True)
        yield qsl, seq, rng.choice([128, 256, 1024])


@pytest.mark.parametrize("nb", [16, 148, 256])
def test_split_plan_covers_the_unsplit_cells_exactly_once(nb):
    rng = random.Random(100 + nb)
    split_seen = 0
    for qsl, seq, width in _split_batches(rng):
        num_reqs, rows = len(seq), qsl[-1]
        for slot in range(-(-rows // width)):
            base = device_plan_model(qsl, seq, num_reqs, rows, width, slot, HEADS, nb)
            plan = device_plan_model(
                qsl, seq, num_reqs, rows, width, slot, HEADS, nb, max_splits=64
            )
            if base["header"][0] == 0:
                continue
            assert all(kv == (0, OPEN_END, 0) for kv in base["kv_ranges"])
            assert split_cells(plan) == split_cells(base)
            items = plan["header"][2]
            unsplit = base["header"][2]
            assert items >= unsplit
            assert items <= unsplit + nb, "split exceeded the num_ctas budget"
            if unsplit >= nb:
                assert plan["kv_ranges"] == base["kv_ranges"]
                assert plan["info"] == base["info"]
            split_seen += items > unsplit
            r = plan["ranges"]
            assert r[0][0] == 0 and r[-1][1] == items
            assert all(a[1] == b[0] for a, b in itertools.pairwise(r))
    assert split_seen > 5


def test_split_shortens_the_long_context_critical_path():
    nb = 148
    qsl, seq = [0, 128], [1_000_000]
    worst = {}
    for splits in (1, 64):
        plan = device_plan_model(qsl, seq, 1, 128, 128, 0, HEADS, nb, max_splits=splits)
        trip = local_trip(0, seq[0], plan["qo_offset"][0])
        load = [0] * nb
        for bucket, (lo, hi) in enumerate(plan["ranges"]):
            for kb, ke, _ in plan["kv_ranges"][lo:hi]:
                load[bucket] += (trip if ke == OPEN_END else ke) - kb
        worst[splits] = max(load)
    assert worst[1] == local_trip(0, seq[0], seq[0] - 128)
    assert worst[64] * 10 < worst[1], worst


def _mutate(plan, fn):
    plan = {**plan, "kv_ranges": list(plan["kv_ranges"]), "info": list(plan["info"])}
    fn(plan)
    return plan


def _split_example():
    plan = device_plan_model(
        [0, 128], [1_000_000], 1, 128, 128, 0, HEADS, 148, max_splits=64
    )
    assert len(plan["info"]) > 2 * HEADS
    return plan


def _first_pieces(plan):
    target = plan["info"][0]
    return [i for i, item in enumerate(plan["info"]) if item == target]


def test_negative_control_overlapping_split_fails():
    def overlap(plan):
        i = min(_first_pieces(plan), key=lambda k: plan["kv_ranges"][k])
        kb, ke, sp = plan["kv_ranges"][i]
        plan["kv_ranges"][i] = (kb, ke + 1, sp)

    with pytest.raises(AssertionError, match="gap or overlap"):
        split_cells(_mutate(_split_example(), overlap))


def test_negative_control_missing_tail_fails():
    def no_tail(plan):
        for k in _first_pieces(plan):
            kb, ke, sp = plan["kv_ranges"][k]
            if ke == OPEN_END:
                plan["kv_ranges"][k] = (kb, kb + 1, sp)

    with pytest.raises(AssertionError, match="0 tail owners"):
        split_cells(_mutate(_split_example(), no_tail))


def test_negative_control_double_tail_and_gap_fail():
    def two_tails(plan):
        k = min(_first_pieces(plan), key=lambda k: plan["kv_ranges"][k])
        kb, _, sp = plan["kv_ranges"][k]
        plan["kv_ranges"][k] = (kb, OPEN_END, sp)

    def gap(plan):
        k = sorted(_first_pieces(plan), key=lambda k: plan["kv_ranges"][k])[1]
        kb, ke, sp = plan["kv_ranges"][k]
        plan["kv_ranges"][k] = (kb + 1, ke, sp)

    with pytest.raises(AssertionError, match="2 tail owners"):
        split_cells(_mutate(_split_example(), two_tails))
    with pytest.raises(AssertionError, match="gap or overlap"):
        split_cells(_mutate(_split_example(), gap))


def test_store_is_unsplit():
    # v13 removed the split-KV ICP scorer: max_kv_splits=1 is still accepted
    # (vLLM passes it) and the writer is always asked for unsplit plans.
    store = _store(max_kv_splits=1)
    assert store.max_kv_splits == 1
    assert store.max_work_items == HEADS * (8 + 8)
    assert store.ranges.shape == (4, 3, 2 * store.max_work_items)
    kw = store.writer_kwargs()
    assert kw["icp_plan_max_splits"] == 1
    assert kw["icp_plan_ranges"] is store.ranges
    with _cpu_planner(16):
        plan = _plan(store, [100], [5000])
    assert plan["num_kv_splits"] == 1
    for name in api._ICP_PLAN_RANGE_ROWS:
        assert plan[name] is None
    assert plan["workspace_o"] is None and plan["workspace_lse"] is None
    for bad in (0, 2, 16, 64):
        with pytest.raises(api.PlanWorkspaceError, match="max_kv_splits"):
            _store(max_kv_splits=bad)
    with pytest.raises(TypeError):
        store.slot(0, chunk_width=1024, kv_splits=1)


def test_scorer_variants_and_prewarm_enumerate_every_selectable_scorer(
    monkeypatch, tmp_path
):
    jit = importlib.import_module("fmha_sm100.icp.scorer.prefill.jit")
    names = [n for n, _ in api.icp_scorer_variants(page_sizes=(64,))]
    # single_wg at W=2, unsplit only: the pair the v6 image baked.
    assert names == ["1_0_0_4_2_0_0", "1_0_1_4_2_0_0"]
    built = []
    monkeypatch.setattr(jit, "CACHE_BASE", tmp_path)
    monkeypatch.setattr(jit._variant_manager, "_load_templates", lambda: None)
    monkeypatch.setattr(
        jit._variant_manager,
        "compile_locked",
        lambda name, params: built.append(params["func_name"]),
    )
    got = api.prewarm_icp_scorer(page_sizes=(64, 32), load=False)
    assert len(got) == 4 and sorted(built) == sorted(f"fmha_sm100_{n}" for n in got)
    assert set(got) == {
        "1_0_0_4_2_0_0",
        "1_0_1_4_2_0_0",
        "1_0_0_4_3_0_0",
        "1_0_1_4_3_0_0",
    }


def test_split_kv_icp_scorer_is_gone():
    jit = importlib.import_module("fmha_sm100.icp.scorer.prefill.jit")
    fp8 = jit._FLOAT8_E4M3FN_CODE
    for page in (64, 32):
        with pytest.raises(ValueError, match="Impossible"):
            jit._variant_key_from_runtime(fp8, 128, False, 4, page, True, 1)
    assert not hasattr(api, "_FMHA_HAS_ICP_SPLIT_KV")
    root = _REPO / "fmha_sm100"
    for rel in (
        "csrc/include/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp",
        "csrc/include/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp",
        "csrc/include/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp",
        "csrc/fmha_sm100_variant_run.cu.jinja",
    ):
        text = (root / rel).read_text()
        assert "kIcpSplitKV" not in text and "icp_split" not in text, rel


def test_split_units_are_the_scorers_compute_tiles():
    # page 64 -> tile_kv _128 and page 32 -> _64: two fragment pages per tile.
    assert TILE_PAGES == 2
    assert (
        local_trip(0, 128, 0) == 1
    )  # one global 128-token page, rank 1 owns rows 64..127
    assert local_trip(7, 1_000_000, 1_000_000 - 1024) == -(-7812 // 2)


def test_split_pieces_respect_the_floor():
    qsl, seq = [0, 128], [1_000_000]
    for nb in (148, 212, 256):
        plan = device_plan_model(qsl, seq, 1, 128, 128, 0, HEADS, nb, max_splits=64)
        for kb, ke, _ in plan["kv_ranges"]:
            if ke != OPEN_END:
                assert ke - kb >= MIN_SPLIT_TILES
        assert plan["header"][2] <= nb
    short = device_plan_model(
        [0, 128], [20_000], 1, 128, 128, 0, HEADS, 212, max_splits=64
    )
    trip = local_trip(0, 20_000, 20_000 - 128)
    assert trip == 78  # the floor, not num_ctas, bounds the piece count
    pieces = len(short["kv_ranges"]) // HEADS
    assert pieces == max(1, trip // MIN_SPLIT_TILES)


# ---------------------------------------------------------------------------
# ABI 3: FMHA windows that start mid-slot (mixed steps, CuTe band end nd)
# ---------------------------------------------------------------------------
def _mixed_batch(rng):
    """Decode rows first (the CuTe band), then prefill rows."""
    nd_reqs = rng.randint(1, 20)
    q = [rng.randint(1, 4) for _ in range(nd_reqs)]
    q += [
        rng.choice([1, 63, 128, 129, rng.randint(1, 2500)])
        for _ in range(rng.randint(1, 5))
    ]
    ctx = [rng.randint(0, 200000) for _ in q]
    qsl = [0]
    for x in q:
        qsl.append(qsl[-1] + x)
    seq = [c + x for c, x in zip(ctx, q, strict=True)]
    return qsl, seq, len(q), qsl[-1] + rng.choice([0, 3]), qsl[nd_reqs]


@pytest.mark.parametrize("width", [128, 256, 1024])
def test_mid_slot_window_matches_host(width):
    rng = random.Random(500 + width)
    mid = 0
    for _ in range(60):
        qsl, seq, num_reqs, num_tokens, nd = _mixed_batch(rng)
        mid += nd % width != 0
        host = _host_reference(qsl, seq, num_tokens, width, 16, row_begin=nd)
        for slot in range(-(-num_tokens // width)):
            dev = device_plan_model(
                qsl, seq, num_reqs, num_tokens, width, slot, HEADS, 16, row_begin=nd
            )
            t0 = max(slot * width, nd)
            assert dev["header"][3] == t0
            if slot not in host:
                assert dev["header"][0] == 0 and dev["info"] == []
                continue
            ref = host[slot]
            assert dev["header"][:2] == ref["header"]
            assert qsl[dev["header"][1] + 1] > t0 >= qsl[dev["header"][1]]
            for key in (
                "qo_segment_offsets",
                "kv_segment_offsets",
                "qo_segment_lens",
                "kv_segment_lens",
                "qo_offset",
                "ranges",
                "info",
            ):
                assert dev[key] == ref[key], key
            # No row below the window origin is planned.
            assert (
                dev["qo_segment_offsets"][-1]
                == min((slot + 1) * width, qsl[num_reqs]) - t0
            )
            n, first = dev["header"][:2]
            for b in range(n):
                req = first + b
                # icpLiveMetadataBlock's icp_qo_offsets cell with the same origin.
                row = max(qsl[req], t0)
                assert dev["qo_offset"][b] == seq[req] - (qsl[req + 1] - row)
    assert mid > 20


def test_row_begin_zero_is_the_abi2_plan():
    rng = random.Random(9)
    for _ in range(20):
        qsl, seq, _, num_reqs, num_tokens = _random_batch(rng, exact=True)
        for slot in range(-(-num_tokens // 256)):
            a = device_plan_model(qsl, seq, num_reqs, num_tokens, 256, slot, HEADS, 16)
            b = device_plan_model(
                qsl, seq, num_reqs, num_tokens, 256, slot, HEADS, 16, row_begin=0
            )
            assert a == b and a["header"][3] == slot * 256


def test_negative_control_window_origin_ignored_fails():
    qsl, seq, num_reqs, num_tokens, nd = [0, 1, 2, 300], [50, 60, 1000], 3, 300, 2
    host = _host_reference(qsl, seq, num_tokens, 256, 16, row_begin=nd)
    wrong = device_plan_model(qsl, seq, num_reqs, num_tokens, 256, 0, HEADS, 16)
    assert wrong["header"][:2] != host[0]["header"]


def test_slot_row_begin_guard():
    store = _store()
    slot = store.slot(1, chunk_width=256, row_begin=300)
    assert (slot.window_begin, slot.window_tokens) == (300, 212)
    with pytest.raises(api.PlanWorkspaceError, match="empty"):
        store.slot(0, chunk_width=256, row_begin=256)
    with _cpu_planner(16):
        with pytest.raises(api.PlanWorkspaceError, match="window"):
            _plan(store, [213], [5000], slot=1, width=256, device_plan=slot)
        plan = _plan(store, [212], [5000], slot=1, width=256, device_plan=slot)
    store.note_writer_launch(chunk_width=256, num_rows=1024, row_begin=0)
    with pytest.raises(api.PlanWorkspaceError, match="origin"):
        api._assert_device_plan_current(plan)
    store.note_writer_launch(chunk_width=256, num_rows=1024, row_begin=300)
    api._assert_device_plan_current(plan)
    with pytest.raises(api.PlanWorkspaceError, match="outside"):
        store.check_writer_launch(chunk_width=256, num_rows=1024, row_begin=1025)


# ---------------------------------------------------------------------------
# check_input_valid: the validator must know the ICP Q-tile pin and the
# open-ended tail owner
# ---------------------------------------------------------------------------
