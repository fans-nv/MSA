"""CPU contracts for the ICP indexer runtime's host planning and launch loop.

Native pieces (CuTe scorer, FMHA planner/launch, selector extension, device
plan store, prewarm manifest) are stubbed; everything the runtime computes on
the host is real.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from fmha_sm100.icp import candidates as cand_mod
from fmha_sm100.icp import indexer_runtime as rt
from fmha_sm100.icp import prewarm as prewarm_mod


class _Scorer:
    def __init__(self, calls, query_len):
        self.calls, self.query_len, self.window = calls, query_len, None

    def bind(self, **window):
        bound = _Scorer(self.calls, self.query_len)
        bound.window = window
        return bound

    def launch(self, *args):
        self.calls.append(("cute", self.query_len, self.window))

    def validate(self, *args):
        self.calls.append(("validate", self.query_len))


class _Store:
    max_kv_splits = 1

    def __init__(self, *, num_slots, **kwargs):
        self.num_slots, self.kwargs = num_slots, kwargs

    def slot(self, index, chunk_width):
        return ("slot", index, chunk_width)


class _Exchange:
    fused_decode = True

    def __init__(self, calls):
        self.calls = calls
        self.decode_profile = {"exchange": "k5t", "merge": "network", "d3": 1, "pdl": 1}

    def publish(self, scores, geometry, *, layer_idx, token_offset):
        self.calls.append(("publish", layer_idx, token_offset, scores.shape[0]))

    def __call__(self, cand, *, layer_idx, has_prefill, forced, n_ordinary, out):
        self.calls.append(("exchange", layer_idx, has_prefill, cand.shape[0]))
        assert cand.dtype is torch.int32 and out.shape[0] == cand.shape[0]
        return out


@pytest.fixture
def harness(monkeypatch):
    calls = []

    def plan(qo_lens, kv_lens, heads, **kwargs):
        calls.append(
            ("plan", qo_lens.tolist(), kv_lens.tolist(), kwargs["device_plan"])
        )
        return {}

    def fmha(q, k, v, plan, **kwargs):
        calls.append(("fmha", q.shape[0], kwargs["max_score"].shape))

    cute = SimpleNamespace(
        ICP_DECODE_SCORE_ABI_VERSION=2,
        ICP_DECODE_SCORE_LAUNCH_ABI_VERSION=3,
        supports_icp_decode_score=lambda *, query_len, **_: query_len <= 4,
    )

    def get_scorer(*, query_len, rank, world_size, split_k, device, page_size, use_pdl):
        assert (split_k, page_size, world_size) == (rt.ICP_CUTE_SPLIT_K, 128, 2)
        return _Scorer(calls, query_len)

    cute.get_icp_decode_scorer = get_scorer
    api = SimpleNamespace(
        IcpDevicePlanStore=_Store, _fmha_sm100_plan=plan, _fmha_sm100=fmha
    )
    monkeypatch.setattr(rt, "_msa_api", lambda: api)
    monkeypatch.setattr(rt, "_msa_cute_decode_module", lambda: cute)
    monkeypatch.setattr(
        rt.IcpIndexerRuntime, "check_installed", staticmethod(lambda: None)
    )
    monkeypatch.setattr(prewarm_mod, "check", list)
    monkeypatch.setattr(rt, "_prewarm_checked", False)
    monkeypatch.setattr(rt, "_decode_profile_logged", False)
    monkeypatch.setattr(
        cand_mod, "allocate_workspace", lambda plan, device: ("ws", plan.token_capacity)
    )

    def select_decode(scores, geometry, *, plan, workspace, out, arm, validate=True):
        calls.append(("select_decode", scores.shape[0], arm.name))

    def select_prefill(
        scores,
        geometry,
        *,
        plan,
        live_blocks,
        out,
        partials,
        arm,
        validate=True,
        **controls,
    ):
        calls.append(("select_prefill", scores.shape[0], live_blocks, controls))

    monkeypatch.setattr(cand_mod, "select_decode_candidates", select_decode)
    monkeypatch.setattr(cand_mod, "select_prefill_candidates", select_prefill)
    cfg = rt.IcpIndexerConfig(
        world_size=2,
        rank=1,
        num_index_heads=4,
        index_head_dim=128,
        max_model_len=16384,
        max_num_batched_tokens=2048,
        max_num_seqs=64,
        max_decode_query_len=4,
        cudagraph_capture_sizes=(1, 2, 4, 8, 16, 32, 64),
        prefill_score_budget_bytes=64 << 20,
    )
    runtime = rt.IcpIndexerRuntime(cfg, device=torch.device("cpu"))
    return runtime, calls


def _table(runtime):
    return torch.zeros((64, runtime.max_blocks + 8), dtype=torch.int32)


def test_startup_geometry_is_derived_from_config_only(harness):
    runtime, _ = harness
    assert runtime.heads_group == 8
    assert runtime.max_blocks == 128
    assert runtime.token_capacity == 2048
    assert runtime.icp_exchange_capacities[0] == 4
    assert runtime.icp_prefill_columns[-1] == 128
    assert runtime.icp_decode_arm is cand_mod.SelectArm.RADIX_FULL_ROW
    assert runtime.icp_prefill_arm is cand_mod.SelectArm.RADIX_FULL_ROW
    assert sorted(runtime.icp_cute_scorers) == [1, 2, 3, 4]
    assert set(runtime.icp_select_extents) <= set(runtime.icp_decode_plans)
    # The decode view is the arena prefix: a captured graph keeps its address.
    assert runtime.icp_scores.data_ptr() == runtime.icp_score_arena.data_ptr()
    assert runtime.icp_active_bytes.data_ptr() == runtime.icp_active.data_ptr()


def test_world_size_other_than_two_is_refused():
    cfg = rt.IcpIndexerConfig(
        world_size=4,
        rank=0,
        num_index_heads=2,
        index_head_dim=128,
        max_model_len=1024,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_decode_query_len=1,
    )
    with pytest.raises(ValueError, match="TP2"):
        rt.IcpIndexerRuntime(cfg, device=torch.device("cpu"))


def test_decode_step_is_one_cute_window_with_fused_publish(harness):
    runtime, calls = harness
    table = _table(runtime)
    qsl = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    seq = torch.tensor([300, 129, 5, 1000], dtype=torch.int32)
    step = runtime.plan_step(
        block_table=table,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=seq,
        seq_lens_cpu_upper_bound=seq,
        is_prefilling=torch.zeros(4, dtype=torch.bool),
        num_reqs=4,
        num_tokens=4,
    )
    assert not step.has_prefill and step.token_capacity == 4
    assert step.cute_rows == 4 and step.fused_ready
    assert [c.cute_scorer is not None for c in step.chunks] == [True]
    assert step.live.device_plan is None
    assert step.exchange_candidates.dtype is torch.int32
    calls.clear()
    index_q = torch.zeros(4, 8 * 128)
    index_kv = torch.zeros(10, 64, 128)
    buf = torch.zeros(2048, 4, 16, dtype=torch.int32)
    ex = _Exchange(calls)
    for layer in range(2):
        runtime.run_layer(
            step,
            index_q,
            index_kv,
            layer_idx=layer,
            exchange=ex,
            topk_buffer=buf,
            scale=1.0,
        )
    kinds = [c[0] for c in calls]
    # Validators run on the first layer only; each layer ends in one exchange.
    assert kinds == [
        "validate",
        "cute",
        "publish",
        "exchange",
        "cute",
        "publish",
        "exchange",
    ]
    assert calls[1][2] == {
        "token_begin": 0,
        "token_count": 4,
        "request_begin": 0,
        "request_count": 4,
    }


def test_prefill_step_plans_fmha_windows_on_the_rung_covering_the_live_extent(harness):
    runtime, calls = harness
    table = _table(runtime)
    qsl = torch.tensor([0, 100, 101], dtype=torch.int32)
    seq = torch.tensor([100, 500], dtype=torch.int32)
    step = runtime.plan_step(
        block_table=table,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=seq,
        seq_lens_cpu_upper_bound=seq,
        is_prefilling=torch.tensor([True, False]),
        num_reqs=2,
        num_tokens=101,
    )
    assert step.has_prefill and step.cute_rows == 0
    assert step.token_capacity in runtime.icp_prefill_exchange_capacities
    assert step.scores.shape[2] == 4  # 500 tokens -> 4 live blocks -> rung 4
    assert step.live.device_plan is runtime.plan_store
    plans = [c for c in calls if c[0] == "plan"]
    assert plans == [("plan", [100, 1], [100, 500], ("slot", 0, step.chunk_width))]
    calls.clear()
    ex = _Exchange(calls)
    runtime.run_layer(
        step,
        torch.zeros(101, 8 * 128),
        torch.zeros(10, 64, 128),
        layer_idx=3,
        exchange=ex,
        topk_buffer=torch.zeros(2048, 4, 16, dtype=torch.int32),
        scale=0.5,
    )
    kinds = [c[0] for c in calls]
    assert kinds == ["fmha", "select_prefill", "exchange"]
    assert calls[1][3] == {
        "full_row_threads": 128,
        "full_row_cached_items": 0,
        "full_row_four_warp_finish": True,
    }
    assert calls[2] == ("exchange", 3, True, step.token_capacity)


def test_block_table_identity_is_latched(harness):
    runtime, _ = harness
    qsl = torch.tensor([0, 1], dtype=torch.int32)
    seq = torch.tensor([10], dtype=torch.int32)
    kwargs = {
        "query_start_loc": qsl,
        "query_start_loc_cpu": qsl,
        "seq_lens": seq,
        "seq_lens_cpu_upper_bound": seq,
        "is_prefilling": torch.zeros(1, dtype=torch.bool),
        "num_reqs": 1,
        "num_tokens": 1,
    }
    first, second = _table(runtime), _table(runtime)
    runtime.plan_step(block_table=first, **kwargs)
    runtime.plan_step(block_table=first, **kwargs)
    with pytest.raises(RuntimeError, match="moved"):
        runtime.plan_step(block_table=second, **kwargs)
