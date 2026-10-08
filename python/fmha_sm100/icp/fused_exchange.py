"""D3: the decode selector publishes; an early-launched kernel polls + merges.

K5T runs after the selector has finished: it re-reads the selector's C4 rows,
pushes them, then polls and merges, and every one of those steps is on the
critical path. This route splits the same work across the launch boundary:

* ``select_and_publish`` is the full-row decode selector with its epilogue
  writing the tagged records straight into the OWNER's symmetric window (the
  K5T window layout, so the wire format is unchanged). It triggers its
  dependents as soon as it has waited on the scorer.
* ``merge`` polls this rank's own window and merges with ``merge_topk.cuh``
  under K2's per-row C3/C5 rules. It is launched as a programmatic dependent
  and does NOT wait on the selector at entry -- the tags carry the data
  dependency for local and remote rows alike -- so its launch, prologue and
  metadata loads overlap the selector. It triggers the attention's launch
  early and (by default) waits on the selector at exit, so its completion
  still implies the selector's.

Generations are per row (``pub_gen[slots, T_cap, H_group]`` owned by selector
CTAs, ``exp_gen[slots, T_cap, H_local]`` owned by merge warps), which is what
removes every read of a counter the other kernel writes. The output is
bit-identical to ``all_to_all + k2_merge`` and to K5T, failed rows included.

Slot discipline, failure policy and backend are K5T's: slots >= 3, the
caller owes ``num_layers % slots != 1``, a transport timeout is sticky and
fatal, CUDA symmetric-memory backend only. Every token row ``t < T`` of an
invocation must be published by exactly one ``select_and_publish`` launch
before (or while) ``merge`` runs over ``T``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from . import _build
from ._symmetric_window import _SymmetricWindow
from .tiled_exchange import (
    DEFAULT_SPIN_CYCLES,
    MIN_SLOTS,
    STATUS_TRANSPORT,
    SYMM_BACKEND,
    _rendezvous,
)

CAND_K = 16

__all__ = ["IcpFusedExchange", "STATUS_TRANSPORT"]


class IcpFusedExchange(_SymmetricWindow):
    """One D3 workspace over one ICP group. Same constructor as K5T's.

    Args:
        group: The ICP process group; its size is ``W``.
        tokens: Window capacity ``T_cap``.
        heads_group: ``H_group = W * H_local``.
        heads_local: ``H_local``.
        slots: Rotating generations, ``>= 3``.
        device: This rank's CUDA device.
        max_ctas: Merge grid cap; 0 sizes it from residency.
        use_pdl: Launch both kernels as programmatic dependents.
        end_wait: The merge waits on the selector before exiting (keeps
            completion transitive; A/B knob, default on).
        classic_merge: Use the 32-round broadcast merge instead of the
            shuffle-network merge (identical output; A/B only).
    """

    def __init__(self, group: dist.ProcessGroup, *, tokens: int,
                 heads_group: int, heads_local: int, slots: int = MIN_SLOTS,
                 device: torch.device | int | None = None,
                 max_ctas: int = 0, use_pdl: bool = True,
                 end_wait: bool = True, classic_merge: bool = False) -> None:
        import torch.distributed._symmetric_memory as symm_mem  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError("IcpFusedExchange needs CUDA")
        for name, value in (("use_pdl", use_pdl), ("end_wait", end_wait),
                            ("classic_merge", classic_merge)):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.T = int(tokens)
        self.Hg = int(heads_group)
        self.Hl = int(heads_local)
        self.slots = int(slots)
        self.max_ctas = int(max_ctas)
        self.use_pdl = use_pdl
        self.end_wait = end_wait
        self.classic_merge = classic_merge
        self._validate_geometry(MIN_SLOTS)
        index = self._set_device(device)

        self.mod = _build._fused()
        m = self.mod
        pdl = int(m.fused_flag_pdl) if use_pdl else 0
        early = int(m.fused_flag_early_trigger) if use_pdl else 0
        self._select_flags = pdl | early
        self._merge_flags = pdl | early | (
            int(m.fused_flag_end_wait) if end_wait else 0) | (
            int(m.fused_flag_classic_merge) if classic_merge else 0)
        plan = m.fused_plan(self.T, self.T, self.Hl, self.world, self.max_ctas)
        self.tile_tok, _, _, self.resident_cap = (int(v) for v in plan)
        self.slot_words = int(m.fused_slot_words(self.T, self.Hg))

        self._allocate_window(symm_mem, _rendezvous, index,
                              transport="D3", backend=SYMM_BACKEND)
        # Plain device memory: only the owning CTA/warp touches each counter.
        self._pub_gen = torch.zeros((self.slots, self.T, self.Hg),
                                    dtype=torch.int32, device=index)
        self._exp_gen = torch.zeros((self.slots, self.T, self.Hl),
                                    dtype=torch.int32, device=index)
        self._status = torch.zeros(1, dtype=torch.int32, device=index)
        self._initialize_window(index)

    # -- hot path: one launch each, no allocation, capturable --------------

    def select_and_publish(self, scores: torch.Tensor, geometry, *,
                           layer_idx: int, token_offset: int = 0,
                           mirror: torch.Tensor | None = None) -> None:
        """Select rows ``[token_offset, token_offset + T_e)`` and publish them.

        ``scores`` is the decode plane ``[T_e, H_group, N]`` (N <= 8192);
        ``geometry`` a ``candidates.CandidateGeometry`` over the same rows.
        ``mirror`` (gates only) also receives the plain C4 rows.
        """
        if self._closed:
            raise RuntimeError("IcpFusedExchange is closed")
        self.mod.fused_select_publish(
            scores, geometry.local_valid_blocks, geometry.forced_column,
            geometry.active_rows, self._buf_ptrs, self._pub_gen, self._status,
            mirror, self.rank, self.world, self.Hl, int(token_offset),
            int(layer_idx) % self.slots, self.slots, self.T, self.slot_words,
            int(geometry.scan_block_begin), int(geometry.global_block_stride),
            self._select_flags)

    def merge(self, out: torch.Tensor, *, layer_idx: int,
              forced: torch.Tensor | None = None,
              n_ordinary: torch.Tensor | None = None) -> torch.Tensor:
        """Poll + merge rows ``[0, T)`` into ``out`` (int32 ``[T, H_local, 16]``)."""
        if self._closed:
            raise RuntimeError("IcpFusedExchange is closed")
        self.mod.fused_merge(
            out, self._buf_ptrs[self.rank], self._exp_gen, self._status,
            forced, n_ordinary, self.world, int(layer_idx) % self.slots,
            self.slots, self.T, self.slot_words, self.max_ctas,
            DEFAULT_SPIN_CYCLES, self._merge_flags)
        return out

    # -- diagnostics ---------------------------------------------------------

    @property
    def status(self) -> torch.Tensor:
        return self._status

    def check_error(self) -> None:
        """Host-synchronising diagnostic; never on the hot path."""
        torch.cuda.synchronize(self.device)
        code = int(self._status.item())
        if code & STATUS_TRANSPORT:
            raise RuntimeError(
                f"D3 exchange failed (status={code}): a record never arrived. "
                "The ranks did not issue the same call sequence (extent, slot, "
                "order) or a row was never published. This workspace is dead.")

    def provenance(self) -> dict[str, object]:
        return {
            "kernel": "d3_fused",
            "abi": str(self.mod.fused_abi_version),
            "world": self.world,
            "T_cap": self.T,
            "H_group": self.Hg,
            "H_local": self.Hl,
            "slots": self.slots,
            "use_ack": 0,
            "use_pdl": int(self.use_pdl),
            "end_wait": int(self.end_wait),
            "classic_merge": int(self.classic_merge),
            "tile_tok": self.tile_tok,
            "resident_cap": self.resident_cap,
            "max_ctas": self.max_ctas,
            "slot_words": self.slot_words,
            "symm_mib": f"{self._data.numel() * 4 / 2 ** 20:.2f}",
            "symm_backend": self.symm_backend,
            "arch": _build.arch(),
        }

    def __repr__(self) -> str:
        return (f"IcpFusedExchange(rank={self.rank}/{self.world}, T={self.T}, "
                f"Hg={self.Hg}, Hl={self.Hl}, slots={self.slots}, "
                f"pdl={self.use_pdl}, end_wait={self.end_wait})")
