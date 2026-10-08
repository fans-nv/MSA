"""K5T: one symmetric-memory push + merge kernel, the pure-decode route.

``IcpTiledExchange.exchange_and_merge`` is a single native launch per sparse
layer for decode and Eagle3 verification; prefill/mixed steps take the NCCL
route (``carrier.collective_exchange_and_merge``: native pack, all_to_all,
K2). The kernel accepts any extent up to its capacity.

The kernel reads the selector's query-major ``[T, H_group, 16, 2]`` int32
candidates and writes each destination's head slab directly into that peer's
symmetric window, so no host or ATen pack exists. It merges with the header K2
compiles (``merge_topk.cuh``) and K2's per-row C3/C5 rules, so its output is
bit-identical to ``all_to_all + k2_merge`` including failed rows (all ``-1``).

Protocol
--------
* Data-as-flag (K5's Lamport layout): every 32-bit payload word travels in a
  64-bit word tagged with its generation. No flags, no fences, no ACK.
* The generation is owned by a fixed row TILE (``8 // H_local`` tokens), held
  in a per-rank device counter plane ``gen[slots, ntiles(T_cap)]``. The grid is
  therefore free per launch and follows the occupancy rule
  ``min(ceil(T * H_local / 8), SMs * resident CTAs/SM)``.
* Slot reuse: no two consecutive launches may use the same slot (wrap
  included); with ``slots >= 3`` and ``num_layers % slots != 1`` that holds,
  and the per-layer TP collective orders a peer's read of slot ``s`` before
  this rank's next write of it. There is no ACK path, so ``slots < 3`` is
  refused.
* Transport failure (a tag that never matches within ``spin_cycles``) sets
  :data:`STATUS_TRANSPORT` in the status word, renders the rows all ``-1`` and
  makes every later launch on the workspace do the same without publishing.
  Treat it as fatal. Nothing on the hot path reads the status word.

Symmetric memory uses the CUDA backend. ``TORCH_SYMMMEM`` and
``NVSHMEM_MAX_TEAMS`` must stay unset, and the constructor refuses any other
backend.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from . import _build
from ._symmetric_window import _SymmetricWindow

CAND_K = 16

#: Mirrors ``kStatusTransport`` in ``csrc/k5t_exchange.cu``.
STATUS_TRANSPORT = 4

#: ~1 s at 2 GHz: a mismatched call sequence becomes a reported error.
DEFAULT_SPIN_CYCLES = 2_000_000_000

#: No ACK path exists, so reuse safety rests on slot rotation alone.
MIN_SLOTS = 3

#: The only symmetric-memory backend this kernel is qualified on.
SYMM_BACKEND = "CUDA"


def _rendezvous(symm_mem: object, tensor: torch.Tensor,
                group: dist.ProcessGroup | None) -> object:
    try:
        return symm_mem.rendezvous(tensor, group=group)
    except TypeError:  # older signature takes the group name
        group_name = (group.group_name if group is not None
                      else dist.group.WORLD.group_name)
        return symm_mem.rendezvous(tensor, group_name)


class IcpTiledExchange(_SymmetricWindow):
    """One K5T workspace over one ICP group.

    Allocation and rendezvous happen eagerly here (collective, never during a
    capture). :meth:`exchange_and_merge` only validates host metadata and
    launches one kernel, so it is cudagraph-capturable and allocates nothing.

    Args:
        group: The ICP process group; its size is ``W``.
        tokens: The window capacity ``T_cap``: the largest extent any call
            may present. Every stride is taken from it.
        heads_group: ``H_group = W * H_local``.
        heads_local: ``H_local``.
        slots: Rotating generations; must be ``>= 3``. The caller also owes
            ``num_layers % slots != 1``.
        device: This rank's CUDA device (after ``set_device``).
        max_ctas: Grid cap override; ``0`` sizes it from the device's
            residency (SMs x resident CTAs/SM for this kernel).
        use_pdl: Launch as a programmatic dependent of the selector and
            trigger the next kernel's launch once every CTA is resident.
        classic_merge: Merge with the 32-round broadcast loops instead of the
            shuffle-network merge (identical output; A/B only).
    """

    def __init__(self, group: dist.ProcessGroup, *, tokens: int,
                 heads_group: int, heads_local: int, slots: int = MIN_SLOTS,
                 device: torch.device | int | None = None,
                 max_ctas: int = 0, use_pdl: bool = False,
                 classic_merge: bool = False) -> None:
        import torch.distributed._symmetric_memory as symm_mem  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError("IcpTiledExchange needs CUDA")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.T = int(tokens)
        self.Hg = int(heads_group)
        self.Hl = int(heads_local)
        self.slots = int(slots)
        self.max_ctas = int(max_ctas)
        if type(use_pdl) is not bool:
            raise TypeError("use_pdl must be a bool")
        self.use_pdl = use_pdl
        self._validate_geometry(MIN_SLOTS, no_ack_prefix="K5T has ")
        self.head_offset = self.rank * self.Hl
        index = self._set_device(device)

        self.mod = _build._k5t()
        if type(classic_merge) is not bool:
            raise TypeError("classic_merge must be a bool")
        self.classic_merge = classic_merge
        self._flags = (int(self.mod.k5t_flag_pdl) if use_pdl else 0) | (
            int(self.mod.k5t_flag_classic_merge) if classic_merge else 0)
        plan = self.mod.k5t_plan(self.T, self.T, self.Hl, self.world,
                                 self.max_ctas)
        self.tile_tok, _, self.ntiles_cap, self.grid_cap, self.resident_cap = (
            int(v) for v in plan)
        self.slot_words = int(self.mod.k5t_slot_words(self.T, self.Hg))

        self._allocate_window(symm_mem, _rendezvous, index,
                              transport="K5T", backend=SYMM_BACKEND)
        # Plain device memory: only the owning rank reads its counters.
        self._gen = torch.zeros((self.slots, self.ntiles_cap),
                                dtype=torch.int32, device=index)
        self._status = torch.zeros(1, dtype=torch.int32, device=index)
        # Tag 0 is poison and counters start at 0, so the first generation
        # is 1. Every rank must see its peers' zeroes before the first push.
        self._initialize_window(index)

    def exchange_and_merge(self, cand: torch.Tensor, *, out: torch.Tensor,
                           layer_idx: int,
                           forced: torch.Tensor | None = None,
                           n_ordinary: torch.Tensor | None = None,
                           ) -> torch.Tensor:
        """Push ``cand``, merge the peers' records into ``out``. One launch.

        ``cand`` is int32 ``[N, H_group, 16, 2]`` (fp32 score bits, block id;
        both bitcast), ``N <= T_cap`` the call's extent, identical on every
        rank and constant across replays of one graph. ``out`` is int32
        ``[N, H_local, 16]``. ``forced``/``n_ordinary`` are C3's int32 ``[N]``
        planes, both or neither. ``layer_idx % slots`` picks the slot.
        """
        if self._closed:
            raise RuntimeError("IcpTiledExchange is closed")
        self.mod.k5t_exchange(
            cand, out, self._buf_ptrs, self._gen, self._status, forced,
            n_ordinary, self.rank, self.world, self.head_offset,
            int(layer_idx) % self.slots, self.slots, self.T, self.slot_words,
            self.max_ctas, DEFAULT_SPIN_CYCLES, self._flags)
        return out

    def plan(self, num_tokens: int) -> dict[str, int]:
        """Grid geometry for one extent (host only)."""
        tt, ntiles, ntiles_cap, grid, cap = (int(v) for v in self.mod.k5t_plan(
            int(num_tokens), self.T, self.Hl, self.world, self.max_ctas))
        return {"tile_tok": tt, "ntiles": ntiles, "ntiles_cap": ntiles_cap,
                "grid": grid, "resident_cap": cap}

    @property
    def status(self) -> torch.Tensor:
        """The sticky status word (K2 bits | :data:`STATUS_TRANSPORT`)."""
        return self._status

    def check_error(self) -> None:
        """Host-synchronising diagnostic; never on the hot path."""
        torch.cuda.synchronize(self.device)
        code = int(self._status.item())
        if code & STATUS_TRANSPORT:
            raise RuntimeError(
                f"k5t exchange failed (status={code}): a peer's generation "
                "never arrived. The ranks did not issue the same call sequence "
                "(extent, slot, order). This workspace is dead.")

    def provenance(self) -> dict[str, object]:
        return {
            "kernel": "k5t",
            "abi": str(self.mod.k5t_abi_version),
            "world": self.world,
            "T_cap": self.T,
            "H_group": self.Hg,
            "H_local": self.Hl,
            "slots": self.slots,
            "use_ack": 0,
            "tile_tok": self.tile_tok,
            "ntiles_cap": self.ntiles_cap,
            "resident_cap": self.resident_cap,
            "max_ctas": self.max_ctas,
            "use_pdl": int(self.use_pdl),
            "classic_merge": int(self.classic_merge),
            "slot_words": self.slot_words,
            "symm_mib": f"{self._data.numel() * 4 / 2 ** 20:.2f}",
            "symm_backend": self.symm_backend,
            "arch": _build.arch(),
        }

    def __repr__(self) -> str:
        return (f"IcpTiledExchange(rank={self.rank}/{self.world}, T={self.T}, "
                f"Hg={self.Hg}, Hl={self.Hl}, slots={self.slots})")
