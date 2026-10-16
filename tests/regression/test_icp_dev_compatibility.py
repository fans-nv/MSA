# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Host controls: ordinary dev behaviour is unchanged by the additive ICP path."""

import subprocess
import sys

import pytest
import torch


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the host compatibility control initialized CUDA")

    # quack probes the GPU arch at import unless told; that probe initializes CUDA.
    monkeypatch.setenv("QUACK_ARCH", "sm_100a")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def test_non_icp_variant_keys_are_unchanged():
    from fmha_sm100 import jit

    for dtype in (torch.float8_e4m3fn, torch.bfloat16, torch.float16):
        code = jit._dlpack_dtype_code(dtype)
        for args in [(tile, wg, mode, page, split, pack)
                     for tile in (64, 128, 256) for wg in (True, False)
                     for mode in (None, 0, 1, 2) for page in (-1, 128, 32, 64, 256)
                     for split in (False, True) for pack in (1, 2)]:
            tile, wg, mode, page, split, pack = args
            try:
                ref = jit._variant_key_from_runtime(code, tile, wg, mode, 128 if page != -1 else -1,
                                                    split, pack)[0]
            except Exception:  # noqa: BLE001 - impossible combinations stay impossible
                continue
            # Other page sizes still take the page-128 kernel, never the ICP page-64 one.
            assert jit._variant_key_from_runtime(code, *args)[0] == ref, (dtype, args)
            assert ref.split("_")[3:5] != ["4", "2"]


def test_icp_adds_exactly_the_two_fp8_onlyscoreicp_variants():
    from fmha_sm100 import api

    assert [name for name, _ in api._icp_scorer_variants()] == ["1_0_0_4_2_0_0", "1_0_1_4_2_0_0"]


def test_importing_api_and_icp_loads_no_cute_dsl():
    # The prewarm CLI pins CUTE_DSL_ARCH before the DSL is first imported.
    code = ("import sys, fmha_sm100.api, fmha_sm100.icp, fmha_sm100.icp.prewarm; "
            "assert 'cutlass' not in sys.modules, sorted(sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={"PYTHONPATH": ":".join(sys.path), "QUACK_ARCH": "sm_100a"})


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


def test_q8kv4_prefill_stack_binds_canonical_callables_without_compiling(monkeypatch):
    import importlib

    import torch.utils.cpp_extension as extensions

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("obtaining the prefill stack compiled a native extension")

    monkeypatch.setattr(extensions, "load", unexpected_compile)
    sparse = importlib.import_module("fmha_sm100.sparse")
    combine = importlib.import_module("src.sm100.fwd.combine").combine  # dev's sys.path shim
    from fmha_sm100.prefill_q8kv4.interface import _sparse_stack

    builder, reducer = _sparse_stack()
    assert builder is sparse.build_k2q_csr and callable(builder)
    assert reducer is combine and callable(reducer)
