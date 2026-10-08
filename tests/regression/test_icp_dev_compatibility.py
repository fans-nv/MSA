# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Host controls for ordinary dev behavior alongside the additive ICP path.

The public mixed-batch planner runs on real CPU metadata; only the allocating
planner and adapter boundaries are recorded. Numerical GPU oracles remain in
tests/q8kv4 and tests/q8kv4_prefill.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the host compatibility control initialized CUDA")

    monkeypatch.setenv("MSA_REGISTER_TVM_FFI", "0")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def test_nvfp4_head_slots_keep_parent_page_strides_and_aliases():
    from fmha_sm100.nvfp4_kv import nvfp4_head_slot_views

    page_stride, offset, slot_bytes = 45_056, 16, 128 * 72
    backing = torch.zeros(2 * page_stride + 32, dtype=torch.uint8)
    cache = backing.as_strided(
        (2, 4, 128, 72), (page_stride, slot_bytes, 72, 1), offset
    )
    views = nvfp4_head_slot_views(cache[:, 0::2], cache[:, 1::2])
    expected = backing.clone()
    for value, (view, relative, width) in enumerate(
        zip(views, (0, 8192, slot_bytes, slot_bytes + 8192), (64, 8, 64, 8)), 1
    ):
        assert view.shape == (2, 2, 128, width)
        view[1, 1, 7, 3] = value
        expected[offset + page_stride + 2 * slot_bytes + relative + 7 * width + 3] = (
            value
        )
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


def test_nvfp4_ordinary_api_rejects_side_packed_head_regions():
    from fmha_sm100.nvfp4_kv import nvfp4_head_slot_views

    cache = torch.empty((2, 4, 128, 72), dtype=torch.uint8)
    with pytest.raises(ValueError, match="per-head K/V slots"):
        nvfp4_head_slot_views(cache[:, :2], cache[:, 2:])


def test_q8kv4_prefill_stack_binds_canonical_callables_without_compiling(monkeypatch):
    import importlib
    import sys

    import torch.utils.cpp_extension as extensions

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("obtaining the prefill stack compiled a native extension")

    monkeypatch.setattr(extensions, "load", unexpected_compile)
    sparse = importlib.import_module("fmha_sm100.sparse")
    combine = importlib.import_module("fmha_sm100.cute.src.sm100.fwd.combine").combine
    from fmha_sm100.prefill_q8kv4.interface import _sparse_stack

    builder, reducer = _sparse_stack()
    assert builder is sparse.build_k2q_csr and callable(builder)
    assert reducer is combine and callable(reducer)
    assert "src" not in sys.modules and "interface" not in sys.modules


@pytest.mark.parametrize("phase", ("decode", "prefill"))
@pytest.mark.parametrize("backend", ("auto", "q8kv4"))
def test_q8kv4_unfit_plan_falls_back_only_when_not_forced(phase, backend):
    from fmha_sm100 import q8kv4_decode_adapter, q8kv4_prefill_adapter

    adapter = q8kv4_decode_adapter if phase == "decode" else q8kv4_prefill_adapter
    plan = {"MM-SA-Nv": phase == "prefill"}
    options = dict(
        kv_segment_lens=torch.tensor([256], dtype=torch.int32),
        num_qo_heads=32,
        num_kv_heads=2,
        page_size=128,
        kv_block_num=65,
        causal=True,
        output_maxscore=False,
        device="cuda:0",
        backend=backend,
        kv_dtype="nvfp4",
        block_scale_shift=3,
    )
    if phase == "decode":
        options.update(
            qo_segment_lens=torch.tensor([1], dtype=torch.int32), usable_sm_count=-1
        )
    if backend == "q8kv4":
        with pytest.raises(
            ValueError, match=f"{phase}_backend=.*cannot plan this batch"
        ):
            adapter.attach_plan(plan, **options)
    else:
        adapter.attach_plan(plan, **options)
    assert adapter.PLAN_KEY not in plan


def test_public_mixed_plan_retains_each_q8kv4_route_and_scale_policy(monkeypatch):
    from fmha_sm100 import api

    native_calls, routes = [], []

    def plan(qo, kv, *args, **kwargs):
        native_calls.append((qo.tolist(), kv.tolist(), kwargs["qo_offset"].tolist()))
        assert (
            not {"decode_backend", "prefill_backend", "kv_dtype", "block_scale_shift"}
            & kwargs.keys()
        )
        return {"metadata": qo}

    def attach(phase, output, **kwargs):
        routes.append((phase, kwargs))
        output["route"] = phase

    monkeypatch.setattr(api, "_fmha_sm100_plan", plan)
    monkeypatch.setattr(
        api.q8kv4_decode_adapter,
        "attach_plan",
        lambda output, **kwargs: attach("decode", output, **kwargs),
    )
    monkeypatch.setattr(
        api.q8kv4_prefill_adapter,
        "attach_plan",
        lambda output, **kwargs: attach("prefill", output, **kwargs),
    )
    result = api.fmha_sm100_plan(
        torch.tensor([4, 4, 129, 257], dtype=torch.int32),
        torch.tensor([256, 512, 1024, 4096], dtype=torch.int32),
        32,
        num_kv_heads=2,
        page_size=128,
        kv_block_num=16,
        decode_backend="kv_mode3",
        prefill_backend="cute_dsl",
        kv_dtype="nvfp4",
        block_scale_shift=0,
        device="cuda:0",
    )
    assert result[:3] == (True, 2, 4)
    assert [result[3]["route"], result[4]["route"]] == ["decode", "prefill"]
    assert native_calls == [
        ([4, 4], [256, 512], [252, 508]),
        ([129, 257], [1024, 4096], [895, 3839]),
    ]
    assert [(phase, options["backend"]) for phase, options in routes] == [
        ("decode", "kv_mode3"),
        ("prefill", "cute_dsl"),
    ]
    for _, options in routes:
        assert options["kv_dtype"] == "nvfp4" and options["block_scale_shift"] == 0
        assert options["num_qo_heads"] == 32 and options["num_kv_heads"] == 2
    assert [options["kv_segment_lens"].tolist() for _, options in routes] == [
        [256, 512],
        [1024, 4096],
    ]
