# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Shared inputs for the ICP device-plan scorer tests (GPU).

Plans are the CPU model of the writer's plan CTAs (``device_plan_model``),
copied into an ``IcpDevicePlanStore`` by the test harness; the native planner
itself is gated against the same model by ``test_icp_device_plan.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "regression"))

import test_icp_device_plan as M

api = M.api
PAGE, FRAGMENT, DIM, HEADS = 128, 64, 128, 4
PAGE_BYTES, INDEX_OFFSET = 45056, 36864


@dataclass
class Batch:
    qsl: list
    seq: list
    table: torch.Tensor  # [reqs, pitch] int32, CPU
    q: torch.Tensor  # [tokens, 4, 128] fp8, CPU
    keys: torch.Tensor  # [pages, 128, 128] fp8, CPU


def make_batch(requests, *, seed=20260929, pitch=None):
    """``requests`` is [(history, query_len)]; pages are shuffled per request."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    blocks = [-(-(h + q) // PAGE) for h, q in requests]
    total_pages = sum(blocks) + 5
    pitch = pitch or max(blocks) + 3
    order = torch.randperm(total_pages, generator=gen).to(torch.int32)
    keys = torch.empty(total_pages, PAGE, DIM, dtype=torch.float8_e4m3fn)
    for begin in range(0, total_pages, 256):
        stop = min(begin + 256, total_pages)
        keys[begin:stop] = (
            torch.randn(stop - begin, PAGE, DIM, generator=gen) * 0.7
        ).to(torch.float8_e4m3fn)
    table = torch.full((len(requests), pitch), total_pages - 1, dtype=torch.int32)
    cursor, qsl, seq = 0, [0], []
    for r, ((h, q), nb) in enumerate(zip(requests, blocks, strict=True)):
        table[r, :nb] = order[cursor : cursor + nb]
        cursor += nb
        qsl.append(qsl[-1] + q)
        seq.append(h + q)
    q = (torch.randn(qsl[-1], HEADS, DIM, generator=gen) * 0.7).to(torch.float8_e4m3fn)
    return Batch(qsl, seq, table, q, keys)


def materialize(batch, rank, device):
    """Compound-page backing with this rank's 64 index rows at INDEX_OFFSET."""
    pages = batch.keys.shape[0]
    backing = torch.full((pages * PAGE_BYTES,), 0x7F, dtype=torch.uint8, device=device)
    local = backing.view(torch.float8_e4m3fn).as_strided(
        (pages, 1, FRAGMENT, DIM), (PAGE_BYTES, FRAGMENT * DIM, DIM, 1), INDEX_OFFSET
    )
    local[:, 0].copy_(batch.keys.to(device)[:, rank * FRAGMENT : (rank + 1) * FRAGMENT])
    return backing, local, batch.table.to(device), batch.q.to(device)


def load_model_plan(store, slot, model):
    """Copy a model plan into ``store`` rows (test harness only)."""
    n, _, items, _ = model["header"]
    seg = store.segments[slot]
    for r, key in enumerate(api._ICP_PLAN_ROWS):
        vals = model[key]
        seg[r, : len(vals)] = torch.tensor(vals, dtype=torch.int32)
    rng = torch.tensor(
        [lo | (hi << 32) for lo, hi in model["ranges"]], dtype=torch.int64
    )
    store.work[slot, : store.num_ctas] = rng
    if items:
        info = torch.tensor(
            [(qt << 32) | (h << 16) | b for qt, h, b in model["info"]],
            dtype=torch.int64,
        )
        store.work[slot, store.num_ctas : store.num_ctas + items] = info
        for r in range(3):
            store.ranges[slot, r, :items] = torch.tensor(
                [kv[r] for kv in model["kv_ranges"]], dtype=torch.int32
            )
    store.header[slot] = torch.tensor(model["header"], dtype=torch.int32)
    del n
