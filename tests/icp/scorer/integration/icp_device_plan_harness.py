# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Direct-table ICP plans for GPU tests, built the production way.

Production order is: host plan build (views of an ``IcpDevicePlanStore`` slot)
-> the first sparse writer refreshes the slot rows -> scorer. Here the writer's
rows come from the CPU model of its plan CTAs (``device_plan_model``, gated
against the native planner by ``test_icp_device_plan.py``) and are copied in by
``load_model_plan``; the scorer then runs on exactly those device rows.
"""

from __future__ import annotations

import torch
from icp_device_plan_fixtures import M, api, load_model_plan

HEADS, FRAGMENT = 4, 64


def build_device_plan(
    *,
    query_lens,
    seq_lens,
    table,
    rank,
    device,
    row_begin=0,
    num_ctas=None,
):
    """One chunk covering every query row; returns (plan, store, model).

    ``num_ctas`` pins the scorer grid (a legal ``usable_SM_count``) for tests
    whose assertions depend on a specific persistent-CTA assignment.
    """
    qsl = [0]
    for q in query_lens:
        qsl.append(qsl[-1] + int(q))
    total = qsl[-1]
    width = total
    nb = api._get_num_cta(device) if num_ctas is None else int(num_ctas)
    model = M.device_plan_model(
        qsl,
        [int(s) for s in seq_lens],
        len(query_lens),
        total,
        width,
        0,
        HEADS,
        nb,
        max_splits=1,
        row_begin=row_begin,
        rank=rank,
    )
    n, first, _, _ = model["header"]
    store = api.IcpDevicePlanStore(
        device=device,
        num_slots=1,
        max_segments=max(n, 1),
        num_heads=HEADS,
        max_chunk_tokens=width,
        num_ctas=nb,
    )
    slot = store.slot(0, chunk_width=width, row_begin=row_begin)
    plan = api._fmha_sm100_plan(
        torch.tensor(model["qo_segment_lens"], dtype=torch.int32),
        torch.tensor(model["kv_segment_lens"], dtype=torch.int32),
        HEADS,
        num_kv_heads=1,
        page_size=FRAGMENT,
        output_maxscore=True,
        causal=True,
        num_kv_splits=1,
        icp_c=2,
        icp_rank=rank,
        icp_block_table=table,
        icp_block_table_row_stride=table.stride(0),
        icp_block_table_row_begin=first,
        usable_SM_count=nb,
        device=device,
        device_plan=slot,
    )
    load_model_plan(store, 0, model)
    store.note_writer_launch(chunk_width=width, num_rows=total, row_begin=row_begin)
    return plan, store, model
