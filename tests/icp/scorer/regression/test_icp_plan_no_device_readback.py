# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""N4 gate: an ICP plan build must not read a CUDA tensor into Python.

WHAT IS BEING GATED
-------------------
``GPU_EXECUTION_NONBLOCKING_REVIEW.md`` (accepted design update) forbids, on the
per-step submission path, any device/stream synchronization, any read of a CUDA
tensor value into Python, and any CPU status gate.  N4 is the MSA instance:
``_fmha_sm100_plan`` used to finish an ICP plan with

    info["kv_page_indptr_local_last"] = int(info["kv_page_indptr"][-1].item())

a D->H round trip on EVERY chunk plan build (the indexer builds plans for
prefill, decode and Eagle3 verification alike).  The fix takes the identical
quantity from the host list the device tensor was copied from
(``kv_page_indptr_list[-1]``), and changes nothing else -- in particular the
execution-time shape guard in ``_fmha_sm100`` is still there and still compares
the same two numbers.

HOW IT DISCRIMINATES
--------------------
``test_icp_plan_build_does_no_cuda_readback`` runs a real ICP plan build with
``Tensor.item``/``tolist``/``cpu`` trapped *for CUDA tensors only* (host tensors
are left alone, because the planner legitimately reads its CPU inputs -- e.g.
``_fmha_sm100_plan``'s own ``qo_segment_lens.max().item()`` on a CPU tensor).
Restore the ``.item()`` line above and the build raises instead of returning:
that is the exact byte this test exists to catch.  The trap is a value check,
not a source-text check, so it also catches the operation being reintroduced
anywhere else in the plan path.

``test_execution_shape_guard_still_fires`` is the other half: the gate must not
be satisfiable by deleting the guard.  It hands execution a page table of the
wrong length and requires the guard to reject it.
"""

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

import pytest
import torch

from fmha_sm100.icp.scorer.prefill.api import _fmha_sm100, _fmha_sm100_plan

HEAD_DIM = 128
NUM_QO_HEADS = 4
QO_LEN = 8
KV_LEN = 1024  # 8 logical 128-token blocks
ICP_C = 4
ICP_RANK = 1

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ICP planning allocates CUDA plan buffers"
)


class CudaReadbackError(AssertionError):
    """Raised by the trap below; a distinct type so the test cannot confuse it
    with an ordinary assertion from the planner."""


@contextlib.contextmanager
def trap_cuda_readback():
    """Make D->H value reads FAIL LOUDLY, while leaving host reads working."""
    originals = {
        name: getattr(torch.Tensor, name) for name in ("item", "tolist", "cpu")
    }

    def _make(name, real):
        def _wrapped(self, *args, **kwargs):
            if self.is_cuda:
                raise CudaReadbackError(
                    f"forbidden D->H readback: Tensor.{name}() on a CUDA tensor "
                    f"(shape={tuple(self.shape)}, dtype={self.dtype}) during a "
                    "plan build; see GPU_EXECUTION_NONBLOCKING_REVIEW.md N4"
                )
            return real(self, *args, **kwargs)

        return _wrapped

    for name, real in originals.items():
        setattr(torch.Tensor, name, _make(name, real))
    try:
        yield
    finally:
        for name, real in originals.items():
            setattr(torch.Tensor, name, real)


def _build_icp_plan(icp_c=ICP_C, icp_rank=ICP_RANK):
    """The same plan geometry the ICP scorer uses (see tests/integration/
    test_icp_fragment_maxscore.py): page_size is this rank's 128//C fragment."""
    qo_t = torch.tensor([QO_LEN], dtype=torch.int32)
    kv_t = torch.tensor([KV_LEN], dtype=torch.int32)
    return _fmha_sm100_plan(
        qo_t,
        kv_t,
        NUM_QO_HEADS,
        num_kv_heads=-1,
        qo_offset=kv_t - qo_t,
        page_size=128 // icp_c,
        output_maxscore=True,
        causal=True,
        num_kv_splits=1,
        icp_c=icp_c,
        icp_rank=icp_rank,
        device=torch.cuda.current_device(),
    )


def _local_page_table(icp_c=ICP_C, icp_rank=ICP_RANK, device="cuda"):
    """This rank's block-cyclic local page table: local page l is global
    fragment l*C + r."""
    nblk = KV_LEN // 128
    return torch.arange(nblk, dtype=torch.int32, device=device) * icp_c + icp_rank


@requires_cuda
def test_trap_actually_traps():
    """Vacuity gate: the trap must fire on a CUDA read and stay out of the way
    of a host read.  Without this, a broken trap would make the gate below pass
    unconditionally."""
    with trap_cuda_readback():
        assert torch.tensor([7], dtype=torch.int32)[-1].item() == 7  # host: fine
        with pytest.raises(CudaReadbackError):
            torch.tensor([7], dtype=torch.int32, device="cuda")[-1].item()
    # and it is restored afterwards
    assert torch.tensor([7], dtype=torch.int32, device="cuda")[-1].item() == 7


@requires_cuda
def test_icp_plan_build_does_no_cuda_readback():
    """THE N4 GATE.  Fails if the plan build reads a CUDA tensor into Python."""
    with trap_cuda_readback():
        plan = _build_icp_plan()

    plan_dict = plan
    stored = plan_dict["kv_page_indptr_local_last"]
    assert stored is not None, "the execution-time guard's scalar must be present"
    assert isinstance(stored, int) and not isinstance(stored, bool), type(stored)

    # ...and it must be the SAME QUANTITY the removed `.item()` produced: the
    # final prefix-sum entry of the plan's own device indptr.  Read the device
    # here, OUTSIDE the trap -- the test may sync, production may not.
    device_last = int(plan_dict["kv_page_indptr"][-1].item())
    assert stored == device_last, (stored, device_last)

    # and the quantity the guard compares against: this rank's local page count.
    assert stored == int(_local_page_table().numel()), stored


@requires_cuda
def test_non_icp_plan_build_does_no_cuda_readback():
    """The C == 1 production path must stay clean too."""
    with trap_cuda_readback():
        plan = _build_icp_plan(icp_c=1, icp_rank=0)
    assert plan["kv_page_indptr_local_last"] == KV_LEN // 128


@requires_cuda
def test_execution_shape_guard_still_fires():
    """The guard must NOT have been deleted to satisfy the gate: a page table
    that is not this rank's must still be rejected at execution time."""
    plan = _build_icp_plan()
    device = torch.device("cuda")
    kv_indices = _local_page_table(device=device)[:-1]  # one page short
    keys = torch.zeros(
        KV_LEN // 128 * ICP_C, 1, 128 // ICP_C, HEAD_DIM, device=device
    ).to(torch.float8_e4m3fn)
    q = torch.zeros(QO_LEN, NUM_QO_HEADS, HEAD_DIM, device=device).to(
        torch.float8_e4m3fn
    )
    max_k_tiles = plan["max_k_tiles"]
    ms = torch.full(
        (QO_LEN, NUM_QO_HEADS, max_k_tiles),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    vs = torch.full(
        (QO_LEN, NUM_QO_HEADS, max_k_tiles), 9, dtype=torch.uint8, device=device
    )
    with pytest.raises(AssertionError, match=r"kv_page_indptr\[-1\]"):
        _fmha_sm100(
            q,
            keys,
            keys,
            plan,
            kv_indices=kv_indices,
            max_score=ms,
            valid_score=vs,
            output_o=False,
            output_maxscore=True,
            icp_c=ICP_C,
            icp_rank=ICP_RANK,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
