"""CPU contracts for indexer planning and bound scorer arguments."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fmha_sm100.icp import local_indexer as local


def test_helpers_import_without_framework_or_scorer_dependencies():
    code = """
import builtins, sys
real = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'cutlass', 'quack', 'tvm_ffi', 'vllm'}:
        raise ImportError('blocked: ' + name)
    return real(name, *args, **kwargs)
builtins.__import__ = guarded
from fmha_sm100.icp import local_indexer as local
assert local.icp_prefill_column_rungs(10, floor=4, steps_per_octave=2) == (4, 6, 8, 10)
windows = local.icp_row_windows(
    num_tokens=190, cap=256, cute_rows=190,
    decode_chunk=128, prefill_width=1024, has_prefill=False)
assert windows[-1].row_end == 190
assert windows[-1].select_rows == 128
assert not [name for name in sys.modules
            if name.startswith(('torch', 'vllm', 'fmha_sm100.icp.scorer'))]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


@pytest.mark.parametrize(
    "max_model_len,token_capacity,budget_bytes,expected",
    [
        (1048576, 16384, 32 * 1024**2, 128),
        (131072, 16384, 32 * 1024**2, 768),
        (16384, 16384, 32 * 1024**2, 1024),
        (16384, 129, 32 * 1024**2, 256),
        (16384, 1, 32 * 1024**2, 128),
        (1048576, 16384, 1, 128),
    ],
)
def test_decode_budget_preserves_tile_floor_capacity_and_ceiling(
    max_model_len, token_capacity, budget_bytes, expected
):
    assert (
        local.icp_decode_chunk_tokens(
            max_model_len=max_model_len,
            heads_group=8,
            token_capacity=token_capacity,
            physical_page_tokens=128,
            budget_bytes=budget_bytes,
            max_chunk=1024,
            query_tile_tokens=128,
        )
        == expected
    )


def test_prefill_ladder_selects_a_covering_rung_and_shared_arena():
    columns = local.icp_prefill_column_rungs(10, floor=4, steps_per_octave=2)
    assert columns == (4, 6, 8, 10)
    assert local.icp_prefill_rung(columns, 7) == 8
    waves = local.icp_prefill_wave_ladder(
        columns,
        heads_group=8,
        token_capacity=16384,
        budget_bytes=65536,
        max_wave=16384,
        query_tile_tokens=128,
    )
    assert waves == {4: 384, 6: 256, 8: 128, 10: 128}
    assert local.icp_prefill_arena_elems(waves, 8) == 12288


def test_prefill_windows_keep_all_rows_on_fmha_and_pad_selector_extent():
    windows = local.icp_row_windows(
        num_tokens=350,
        cap=384,
        cute_rows=0,
        decode_chunk=128,
        prefill_width=256,
        has_prefill=True,
    )
    assert windows == [
        local.ICPRowWindow(False, 0, 0, 0, 256, 256, False),
        local.ICPRowWindow(False, 1, 256, 256, 350, 128, False),
    ]


@pytest.mark.parametrize("columns,finish", [(128, True), (129, False)])
def test_selector_controls_preserve_four_warp_finish_boundary(columns, finish):
    controls = local.prefill_full_row_controls(
        columns, threads=128, cached_items=0, four_warp_finish_max_columns=128
    )
    assert local.prefill_selector_kwargs(controls) == {
        "full_row_threads": 128,
        "full_row_cached_items": 0,
        "full_row_four_warp_finish": finish,
    }


def test_cute_binding_captures_launch_and_views_without_launching():
    calls = []
    scorer = SimpleNamespace(launch=lambda *args: calls.append(args))
    views = [object() for _ in range(7)]
    bound = local.bind_cute_launch(scorer, *views)
    assert calls == []
    scorer.launch = None
    index_q, index_kv, k_pages = object(), object(), object()
    bound(index_q, index_kv, k_pages, 0.25)
    assert calls == [(index_q, index_kv, *views)]


def test_fmha_binding_slices_query_rows_and_keeps_only_score_arguments():
    calls = []
    plan, scores, valid = {}, object(), object()
    bound = local.bind_fmha_launch(
        lambda *args, **kwargs: calls.append((args, kwargs)),
        plan,
        2,
        3,
        max_scores=scores,
        max_scores_valid=valid,
    )
    assert calls == []
    index_q = [object() for _ in range(8)]
    k_pages = object()
    bound(index_q, object(), k_pages, 0.25)
    args, kwargs = calls[0]
    assert args == (index_q[2:5], k_pages, k_pages, plan)
    assert kwargs == {
        "output_o": False,
        "output_maxscore": True,
        "sm_scale": 0.25,
        "max_scores": scores,
        "max_scores_valid": valid,
    }
