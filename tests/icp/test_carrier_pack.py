"""The prefill/mixed NCCL route: native pack + all_to_all + direct K2.

* ``icp_pack`` equals the ATen reference ``pack_send_carrier`` bit for bit, for
  int32 and bitcast fp32 producers, and the forbidden view-reshape differs.
* ``collective_exchange_and_merge`` with a retained workspace and status issues
  no ATen op apart from the collective (patched out here and counted).
* Native pack -> emulated all_to_all -> K2 equals K5T on the same inputs,
  failed rows included (both render all -1).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from test_tiled_exchange import (  # noqa: E402
    Emulation,
    _bits,
    _inputs,
    make_rank_candidates,
    oracle_k2,
)
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from fmha_sm100.icp import _build, carrier  # noqa: E402


class _RecordAten(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _native_route(cands, forced, n_ord):
    """Every rank: native pack; the all_to_all is emulated; K2 merges."""
    world = len(cands)
    hl = cands[0].shape[1] // world
    T = cands[0].shape[0]
    sends = []
    for c in cands:
        send = torch.empty(
            carrier.send_carrier_shape(world, T, hl), dtype=torch.int32, device="cuda"
        )
        sends.append(carrier.pack_send_carrier_native(c, send, world))
    outs, statuses = [], []
    for dest in range(world):
        recv = torch.stack([s[dest] for s in sends]).contiguous()
        out = torch.empty((T, hl, 16), dtype=torch.int32, device="cuda")
        status = torch.zeros(1, dtype=torch.int32, device="cuda")
        _build._k2().k2_merge(recv, out, dest * hl, world, dest, forced, n_ord, status)
        outs.append(out)
        statuses.append(int(status.item()))
    return outs, statuses


@pytest.mark.gpu
@pytest.mark.parametrize("world,hl", [(1, 2), (2, 2), (4, 1), (8, 1)])
@pytest.mark.parametrize("q", [1, 5, 1000])
def test_native_pack_equals_the_reference_pack(world, hl, q):
    cand = make_rank_candidates(q, world * hl, 0, seed=q).cuda()
    ref = carrier.pack_send_carrier(cand, world)
    out = torch.full_like(ref, 0x5A5A5A5A)
    carrier.pack_send_carrier_native(cand, out, world)
    assert torch.equal(out, ref)
    as_fp32 = torch.full_like(ref, 0x5A5A5A5A)
    carrier.pack_send_carrier_native(cand.view(torch.float32), as_fp32, world)
    assert torch.equal(as_fp32, ref)
    if q > 1 and world > 1:
        wrong = carrier._wrong_pack_view_negative_control(cand, world)
        assert not torch.equal(out, wrong)


@pytest.mark.gpu
def test_native_pack_zero_rows_is_a_no_op_like_the_reference():
    cand = torch.empty((0, 4, 16, 2), dtype=torch.int32, device="cuda")
    ref = carrier.pack_send_carrier(cand, 2)
    out = torch.empty_like(ref)
    carrier.pack_send_carrier_native(cand, out, 2)
    assert out.shape == ref.shape == (2, 0, 2, 16, 2)


@pytest.mark.gpu
def test_native_pack_refuses_bad_shapes():
    cand = make_rank_candidates(4, 4, 0, seed=0).cuda()
    out = torch.empty((2, 4, 1, 16, 2), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="out must be"):
        carrier.pack_send_carrier_native(cand, out, 2)


@pytest.mark.gpu
def test_collective_route_is_aten_free_apart_from_the_collective(monkeypatch):
    world, hl, T = 2, 2, 64
    cands, forced, n_ord = _inputs(world, hl, T, seed=7)
    ws = carrier.allocate_carrier_workspace(
        world=world, qchunk=T, h_local=hl, device="cuda"
    )
    status = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.empty((T, hl, 16), dtype=torch.int32, device="cuda")
    calls = []
    monkeypatch.setattr(
        carrier.dist,
        "all_to_all_single",
        lambda recv, send, group=None: calls.append((recv.data_ptr(), send.data_ptr())),
    )
    _build._pack()
    _build._k2()
    with _RecordAten() as mode:
        carrier.collective_exchange_and_merge(
            cands[0],
            world=world,
            rank=0,
            out=out,
            forced=forced,
            n_ordinary=n_ord,
            status=status,
            group=None,
            workspace=ws,
        )
    assert mode.ops == []
    assert calls == [(ws.recv.data_ptr(), ws.send.data_ptr())]


@pytest.mark.gpu
@pytest.mark.parametrize("T", [1, 37, 1024])
def test_both_routes_are_bit_identical_including_failed_rows(k5t, T):
    world, hl = 2, 2
    cands, forced, n_ord = _inputs(world, hl, T, seed=T)
    nan = _bits(float("nan"))
    cands[1][T // 2, 0, 0] = torch.tensor([nan, 3], dtype=torch.int32)
    native, s_native = _native_route(cands, forced, n_ord)
    emu = Emulation(k5t, world=world, t_cap=1024, h_local=hl)
    fused = emu.run(cands, forced, n_ord, layer_idx=0)
    for r in range(world):
        ref, s_ref = oracle_k2(cands, r, forced, n_ord)
        assert torch.equal(native[r], ref)
        assert torch.equal(fused[r], ref)
        assert s_native[r] == s_ref == int(emu.status[r].item())
    # Global head 0 is rank 0's local head 0: that row failed C5 on both routes.
    assert s_native[0] & 1
    assert bool((fused[0][T // 2, 0] == -1).all())
    assert bool((native[0][T // 2, 0] == -1).all())


@pytest.fixture(scope="module")
def k5t():
    return _build._k5t()
