"""CPU checks for collective setup/teardown ordering, without a native build."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(params=["d3", "k5t"])
def host_exchange(request, monkeypatch):
    import torch.distributed._symmetric_memory as symm_mem

    from fmha_sm100.icp import _build
    from fmha_sm100.icp.fused_exchange import IcpFusedExchange
    from fmha_sm100.icp.tiled_exchange import IcpTiledExchange

    events = []
    state = SimpleNamespace(
        kind=request.param,
        events=events,
        backend="CUDA",
        pointers=[1000, 2000],
        barrier_error=None,
    )

    class Buffer:
        def __init__(self, shape):
            self.shape = shape

        def data_ptr(self):
            return 1000

        def zero_(self):
            events.append(("zero_window",))

    class Handle:
        def barrier(self):
            events.append(("barrier",))
            if state.barrier_error is not None:
                raise state.barrier_error

        @property
        def buffer_ptrs(self):
            events.append(("peer_pointers",))
            return state.pointers

    def empty(size, *, dtype, device):
        events.append(("empty", size, dtype, device))
        return Buffer((size,))

    def zeros(shape, *, dtype, device):
        shape = (shape,) if isinstance(shape, int) else shape
        events.append(("zeros", shape, dtype, device))
        return Buffer(shape)

    def rendezvous(tensor, *, group):
        events.append(("rendezvous",))
        return Handle()

    def get_backend(device):
        events.append(("backend", device))
        return state.backend

    def plan(*args):
        events.append(("plan", *args))
        return (4, 2, 2, 8) if state.kind == "d3" else (4, 2, 2, 2, 8)

    def slot_words(*args):
        events.append(("slot_words", *args))
        return 2048

    native = SimpleNamespace(
        fused_plan=plan,
        fused_slot_words=slot_words,
        fused_flag_pdl=1,
        fused_flag_early_trigger=2,
        fused_flag_end_wait=4,
        fused_flag_classic_merge=32,
        k5t_plan=plan,
        k5t_slot_words=slot_words,
        k5t_flag_pdl=4,
        k5t_flag_classic_merge=8,
    )

    def load():
        events.append(("load",))
        return native

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append(("sync", device))
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(symm_mem, "empty", empty)
    monkeypatch.setattr(symm_mem, "rendezvous", rendezvous)
    monkeypatch.setattr(symm_mem, "get_backend", get_backend, raising=False)
    monkeypatch.setattr(_build, "_fused" if state.kind == "d3" else "_k5t", load)
    cls = IcpFusedExchange if state.kind == "d3" else IcpTiledExchange

    def create(**overrides):
        kwargs = dict(tokens=8, heads_group=4, heads_local=2, slots=3, device=0)
        kwargs.update(overrides)
        return cls(SimpleNamespace(group_name="icp"), **kwargs)

    state.create = create
    state.symm_mem = symm_mem
    return state


def test_counter_allocation_precedes_collective_window_initialization(host_exchange):
    state = host_exchange
    exchange = state.create()
    shapes = [(3, 8, 4), (3, 8, 2), (1,)] if state.kind == "d3" else [(3, 2), (1,)]
    assert state.events == [
        ("load",),
        ("plan", 8, 8, 2, 2, 0),
        ("slot_words", 8, 4),
        ("empty", 6144, torch.int32, 0),
        ("rendezvous",),
        ("backend", torch.device("cuda", 0)),
        *(("zeros", shape, torch.int32, 0) for shape in shapes),
        ("zero_window",),
        ("sync", 0),
        ("barrier",),
        ("sync", 0),
        ("peer_pointers",),
    ]
    assert exchange.status is exchange._status
    assert exchange.symm_backend == "CUDA"
    assert exchange._buf_ptrs == [1000, 2000]
    if state.kind == "d3":
        assert exchange._pub_gen.shape == (3, 8, 4)
        assert exchange._exp_gen.shape == (3, 8, 2)
    else:
        assert exchange._gen.shape == (3, 2)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"heads_group": 3}, "C7: H_group"),
        ({"tokens": 0}, "tokens must be >= 1"),
        ({"slots": 2}, "needs >= 3 slots"),
        ({"max_ctas": -1}, "max_ctas must be >= 0"),
        ({"device": "cpu"}, "device must be CUDA"),
    ],
)
def test_invalid_geometry_fails_before_build_or_allocation(
    host_exchange, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        host_exchange.create(**kwargs)
    assert host_exchange.events == []


def test_backend_failure_precedes_counter_allocation(host_exchange):
    host_exchange.backend = "NVSHMEM"
    with pytest.raises(RuntimeError, match="CUDA backend only"):
        host_exchange.create()
    assert [e[0] for e in host_exchange.events] == [
        "load",
        "plan",
        "slot_words",
        "empty",
        "rendezvous",
        "backend",
    ]


@pytest.mark.parametrize(
    "pointers, message",
    [
        ([1000], "wrong peer count"),
        ([2000, 1000], "did not map this rank's window"),
    ],
)
def test_invalid_peer_table_is_rejected_after_collective_init(
    host_exchange, pointers, message
):
    host_exchange.pointers = pointers
    with pytest.raises(RuntimeError, match=message):
        host_exchange.create()
    assert [e[0] for e in host_exchange.events[-5:]] == [
        "zero_window",
        "sync",
        "barrier",
        "sync",
        "peer_pointers",
    ]


def test_older_backend_api_is_still_supported(host_exchange, monkeypatch):
    monkeypatch.delattr(host_exchange.symm_mem, "get_backend")
    assert host_exchange.create().symm_backend == "CUDA"


def test_context_manager_drains_collectively_and_close_is_idempotent(host_exchange):
    exchange = host_exchange.create()
    host_exchange.events.clear()
    with exchange as entered:
        assert entered is exchange
    exchange.close()
    assert host_exchange.events == [
        ("sync", exchange.device),
        ("barrier",),
        ("sync", exchange.device),
    ]
    assert exchange._closed
    assert exchange._data is None and exchange._h_data is None
    assert exchange._buf_ptrs == []


def test_teardown_barrier_error_does_not_mask_original_failure(host_exchange):
    exchange = host_exchange.create()
    host_exchange.events.clear()
    host_exchange.barrier_error = RuntimeError("peer is gone")
    with pytest.raises(ValueError, match="original failure"):
        with exchange:
            raise ValueError("original failure")
    assert host_exchange.events == [("sync", exchange.device), ("barrier",)]
    assert exchange._closed and exchange._data is None


def test_initial_drain_failure_is_still_observable(host_exchange, monkeypatch):
    exchange = host_exchange.create()
    data, handle = exchange._data, exchange._h_data

    def fail_drain(device):
        raise RuntimeError("outstanding CUDA failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_drain)
    with pytest.raises(RuntimeError, match="outstanding CUDA failure"):
        exchange.close()
    assert exchange._closed
    assert exchange._data is data and exchange._h_data is handle
