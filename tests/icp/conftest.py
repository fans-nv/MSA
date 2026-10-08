"""Markers, skip rules and the shared fixtures for the ICP kernel tests.

Three tiers, and the boundaries are load-bearing:

* **unmarked** -- pure CPU. Imports the package, exercises the vendored ABI,
  ``canonical_key_reference``, the C4 pack and the arch/flag logic. Must pass on
  a workstation with no GPU and no CUDA toolkit, because a suite that can only
  run on the cluster is a suite that stops being run. ``tests/test_abi.py`` and
  the tier-1 half of ``tests/test_candidates.py`` need no torch **at all**.
* ``@pytest.mark.gpu`` -- needs one CUDA device and a JIT build.
* ``@pytest.mark.distributed`` -- needs an initialised process group (torchrun,
  or a Slurm step with ``MASTER_ADDR`` set). The **fixture** sets the minimum
  width, not the marker: ``icp_group`` refuses fewer than two ranks because a
  transport claim is vacuous with one, while ``icp_group_any`` admits one
  because a per-row merge claim is not. See ``icp_group_any``.

Every skip says *why* in a sentence a reader can act on. "1 skipped" with no
reason is how a gate quietly stops gating: 128 silently skipped drift checks
are exactly how the upstream ``-0.0`` divergence survived for weeks.

refined-icp-v1
--------------

Two fixtures below exist because of the refined contract and did not before:

``c4_carrier``
    Builds the int32 ``[W, Qchunk, H_local, 16, 2]`` **receive** carrier
    directly, from plain Python ``(score, block id)`` records. Under fragment
    placement the merge's input is W partial-maximum lists over the *same*
    global block domain, so nearly every property worth gating -- duplicate
    reduction, the reserved forced slot, the NaN failure path -- is a statement
    about a multi-source carrier and about nothing else. Constructing that
    carrier is a legitimate single-GPU test of the merge: the transport's job is
    to produce exactly this buffer, and ``tests/test_exchange.py`` gates the
    transport separately, where a process group exists.
``k2_module``
    The built K2 extension, so a test can assert the status bits and the ABI
    strings against the object that was loaded rather than against a comment.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: needs one CUDA device")
    config.addinivalue_line(
        "markers",
        "distributed: needs an initialised process group; the fixture sets the "
        "minimum width (icp_group >= 2 ranks, icp_group_any >= 1)",
    )


def _cuda_reason() -> str | None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "torch is not installed"
    if not torch.cuda.is_available():
        return (
            "no CUDA device visible to this process (torch.cuda.is_available() "
            "is False). The GPU-marked tests JIT-build a CUDA extension and "
            "launch it; run them on a GPU node."
        )
    return None


#: Test modules that import torch at module scope, and so cannot be *collected*
#: without it. Skipping them by name keeps the promise the tier list above makes:
#: `pytest` on a torch-free workstation runs the pure-CPU tier and reports it,
#: instead of aborting the whole session on a collection error.
_NEEDS_TORCH_TO_IMPORT = ("test_exchange.py", "test_merge_key.py")


def pytest_ignore_collect(collection_path, config):
    if collection_path.name not in _NEEDS_TORCH_TO_IMPORT:
        return None
    try:
        import torch  # noqa: F401
    except ImportError:
        return True
    return None


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    reason = _cuda_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords or "distributed" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------
# distributed
# --------------------------------------------------------------------------


def _env_world_size() -> int:
    for key in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        v = os.environ.get(key)
        if v:
            return int(v)
    return 1


def _env_rank() -> int:
    for key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        v = os.environ.get(key)
        if v:
            return int(v)
    return 0


def _env_local_rank() -> int:
    for key in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        v = os.environ.get(key)
        if v:
            return int(v)
    return 0


@pytest.fixture(scope="session")
def icp_group_any():
    """An initialised NCCL process group over every rank, **at any world size**.

    ``torch.cuda.set_device()`` happens **before** ``init_process_group()``,
    and no rank masks ``CUDA_VISIBLE_DEVICES``. Both are mandatory for
    symmetric memory: masking makes every rank report CUDA ordinal 0, and the
    symmetric-memory allocator then refuses the rendezvous with "detected
    allocations from overlapping devices from different ranks" -- which is the
    check working, not a bug. The documented escape hatch
    (``TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=1``) is never used here: it
    makes ``CUDASymmetricMemory.cu:853`` skip ``init_multicast_for_block`` and
    silently disables NVLS multicast.

    ``world_size == 1`` is admitted here and refused by :func:`icp_group`,
    because the two serve different claims. A **transport** claim -- who sends
    which head slab to whom, whether a global id can arrive from two sources --
    is vacuous with one rank. A **per-row merge** claim is not: at ``W == 1``
    the exchange still publishes into its own symmetric window, hands off and
    reads itself back, so the row that reaches ``warp_merge_topk16`` is a real
    exchanged row. The C3 reserved forced slot is a property of that row and of
    nothing else, which is why its K5 arm takes this fixture and the bit-identity
    arms take the other one. A single-rank NCCL group and a single-rank
    symmetric-memory rendezvous are both legal, and ``IcpExchange`` reads its
    width from the group (``exchange.py``: ``dist.get_world_size(group)``)
    rather than requiring one.
    """
    import torch
    import torch.distributed as dist

    world = _env_world_size()
    if os.environ.get("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES"):
        pytest.fail(
            "TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES is set. It silently "
            "disables NVLS multicast (CUDASymmetricMemory.cu:853 skips "
            "init_multicast_for_block), so any result measured under it is "
            "not the result this kernel ships. Unset it and fix the real "
            "cause: torch.cuda.set_device() before init_process_group(), and "
            "no per-rank CUDA_VISIBLE_DEVICES masking."
        )
    local = _env_local_rank()
    if torch.cuda.device_count() <= local:
        pytest.skip(
            f"local rank {local} but only {torch.cuda.device_count()} visible "
            "device(s): CUDA_VISIBLE_DEVICES looks masked per rank, which "
            "breaks symmetric-memory rendezvous. Give every rank the whole "
            "device list."
        )
    torch.cuda.set_device(local)  # BEFORE init_process_group, always
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29777")
        dist.init_process_group(backend="nccl", world_size=world, rank=_env_rank())
    group = dist.group.WORLD
    yield group
    dist.barrier()


@pytest.fixture(scope="session")
def icp_group(request):
    """:func:`icp_group_any`, refused below two ranks.

    The skip is asked for **before** the group is built, so a workstation run
    does not pay an initialisation it is about to discard.
    """
    world = _env_world_size()
    if world < 2:
        pytest.skip(
            f"world_size={world}: the exchange needs at least 2 ICP ranks. "
            "Launch with e.g. `torchrun --nproc_per_node=2 -m pytest "
            "tests/test_exchange.py`, giving every rank the full "
            "CUDA_VISIBLE_DEVICES list (per-rank masking breaks symmetric "
            "memory rendezvous)."
        )
    return request.getfixturevalue("icp_group_any")


@pytest.fixture(scope="session")
def cuda_device():
    """The device this rank owns, after ``set_device``."""
    import torch

    torch.cuda.set_device(_env_local_rank())
    return torch.device("cuda", torch.cuda.current_device())


# --------------------------------------------------------------------------
# the C4 carrier, built by hand
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def k2_module():
    """The built K2 extension. GPU-only: this JIT-builds on first use."""
    from fmha_sm100.icp import _build

    return _build._k2()


@pytest.fixture()
def c4_carrier():
    """Build an int32 ``[W, Qchunk, H_local, 16, 2]`` C4 **receive** carrier.

    ``fill(source, token, head)`` returns that source's record list for the row,
    as ``(score, global_block_id)`` pairs, at most 16 of them; the rest of the
    row is padded with C4's invalid record ``(-inf, -1)``.

    Both words are written **bitcast**, never converted -- the score's raw fp32
    bits in word 0 and the int32 block id in word 1 -- because that is what C4
    says is on the wire and because an id above ``2**24`` does not survive a
    float round trip. A test that built this with ``.float()`` would pass at
    small ids and silently corrupt large ones, which is the exact defect the
    dtype exists to prevent.
    """
    import torch

    K = 16

    def build(
        fill, *, world: int, qchunk: int = 1, heads_local: int = 1, device="cuda"
    ) -> torch.Tensor:
        scores = torch.full(
            (world, qchunk, heads_local, K), float("-inf"), dtype=torch.float32
        )
        ids = torch.full((world, qchunk, heads_local, K), -1, dtype=torch.int32)
        for s in range(world):
            for q in range(qchunk):
                for h in range(heads_local):
                    records = list(fill(s, q, h))
                    assert len(records) <= K, (
                        f"a C4 record list holds at most {K} entries; source "
                        f"{s} row ({q}, {h}) supplied {len(records)}"
                    )
                    for k, (score, gid) in enumerate(records):
                        scores[s, q, h, k] = score
                        ids[s, q, h, k] = gid
        carrier = torch.empty((world, qchunk, heads_local, K, 2), dtype=torch.int32)
        carrier[..., 0] = scores.view(torch.int32)
        carrier[..., 1] = ids
        return carrier.to(device).contiguous()

    return build
