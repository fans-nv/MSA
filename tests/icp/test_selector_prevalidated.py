"""``validate=False``: the pre-validated selector launch is the native call alone.

The first sparse layer of a step calls with ``validate=True``; later layers
reuse the same bindings with ``validate=False``. The fast form must give the
same bits, never reach ``_validate`` or any host check, and dispatch no ATen
op (not even the ``partials[:T]`` slice when the prefix is the whole buffer).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from fmha_sm100.icp import candidates as c  # noqa: E402

FULL_ROW = c.SelectArm.RADIX_FULL_ROW
PREFILL_CONTROLS = dict(
    full_row_threads=128, full_row_cached_items=0, full_row_four_warp_finish=True
)


class _RecordAten(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _geometry(tokens, columns, device, seed):
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, columns + 1, (tokens,), generator=g)
    forced = torch.where(counts > 0, counts - 1, torch.full_like(counts, -1))
    active = torch.rand((tokens,), generator=g) > 0.1
    return c.CandidateGeometry(
        counts.to(torch.int32).to(device),
        forced.to(torch.int32).to(device),
        active.to(device),
        0,
        1,
    )


def _scores(tokens, heads, columns, device, seed):
    g = torch.Generator().manual_seed(seed)
    s = torch.round(torch.randn((tokens, heads, columns), generator=g) * 8) / 8
    return s.to(device)


def _forbid_validate(monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("_validate reached on the validate=False path")

    monkeypatch.setattr(c, "_validate", forbidden)
    monkeypatch.setattr(c, "_prefill_full_row_config", forbidden)


@pytest.mark.gpu
def test_decode_fast_path_is_bit_identical_and_aten_free(monkeypatch):
    device = torch.device("cuda")
    plan = c.DecodePlan(
        icp_degree=2,
        icp_rank=1,
        num_heads_local=2,
        max_local_blocks=96,
        token_capacity=64,
    )
    workspace = c.allocate_workspace(plan, device)
    geometry = _geometry(64, 96, device, seed=1)
    scores = _scores(64, 4, 96, device, seed=2)
    c.select_decode_candidates(
        scores,
        geometry,
        plan=plan,
        workspace=workspace,
        out=workspace.candidates,
        arm=FULL_ROW,
    )
    ref = workspace.candidates.clone()
    workspace.candidates.view(torch.int32).fill_(0x5A5A5A5A)
    torch.cuda.synchronize()
    _forbid_validate(monkeypatch)
    with _RecordAten() as mode:
        c.select_decode_candidates(
            scores,
            geometry,
            plan=plan,
            workspace=workspace,
            out=workspace.candidates,
            arm=FULL_ROW,
            validate=False,
        )
    assert mode.ops == []
    assert torch.equal(workspace.candidates.view(torch.int32), ref.view(torch.int32))


@pytest.mark.gpu
def test_prefill_fast_path_is_bit_identical_and_aten_free(monkeypatch):
    device = torch.device("cuda")
    tokens, columns = 48, 200
    plan = c.PrefillPlan(
        icp_degree=2,
        icp_rank=0,
        num_heads_local=2,
        token_capacity=tokens,
        max_local_blocks=columns,
    )
    geometry = _geometry(tokens, columns, device, seed=3)
    scores = _scores(tokens, 4, columns, device, seed=4)
    out = torch.empty((tokens, 4, 16, 2), dtype=torch.float32, device=device)
    partials = torch.empty(
        plan.partial_shape(tokens, plan.scan_extent), dtype=torch.float32, device=device
    )
    c.select_prefill_candidates(
        scores,
        geometry,
        plan=plan,
        live_blocks=0,
        out=out,
        partials=partials,
        arm=FULL_ROW,
        **PREFILL_CONTROLS,
    )
    ref = out.clone()
    out.view(torch.int32).fill_(0x5A5A5A5A)
    torch.cuda.synchronize()
    _forbid_validate(monkeypatch)
    with _RecordAten() as mode:
        c.select_prefill_candidates(
            scores,
            geometry,
            plan=plan,
            live_blocks=0,
            out=out,
            partials=partials,
            arm=FULL_ROW,
            validate=False,
            **PREFILL_CONTROLS,
        )
    assert mode.ops == []
    assert torch.equal(out.view(torch.int32), ref.view(torch.int32))


def test_prefill_fast_path_refuses_to_allocate():
    plan = c.PrefillPlan(
        icp_degree=2,
        icp_rank=0,
        num_heads_local=2,
        token_capacity=8,
        max_local_blocks=8,
    )
    with pytest.raises(ValueError, match="requires out and partials"):
        c.select_prefill_candidates(
            None, None, plan=plan, live_blocks=0, validate=False
        )
