# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-owned candidate transport, independent of any serving framework.

The local selector and distributed merge share native implementations. D3
publishes directly from the selector; K5T consumes the C4 carrier; mixed/prefill
uses native packing, NCCL all-to-all and K2. Construction retains every buffer;
the submission path performs no allocation or device readback.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist

from .candidates import CAND_K
from .exchange_plan import ICP_EXCHANGE_SLOTS, check_slot_discipline

BACKEND_NCCL = "nccl"
BACKEND_K5T = "k5t"


class CandidateExchange:
    """Retained candidate transport for a model's sparse indexer layers.

    Call ``publish`` for each D3 decode window, including inactive padding,
    then call this owner once per layer to merge. K5T and NCCL instead consume
    the completed local candidate carrier in that final call. Route selection
    uses only ``has_prefill``; no route falls back to another.

    Args:
        group: The ICP process group -- ``W`` consecutive TP ranks on one node.
        world_size: ``W``, this group's size.
        rank: This rank within the group.
        num_heads_local: ``H_local``, the heads this rank merges for.
        num_index_heads: ``H_group``, the group's indexer heads. C7 pins
            ``H_group == W * H_local`` and a rank-major head order.
        num_sparse_layers: How many layers share the rotating slot window. Used
            only by the slot-discipline assertion.
        token_capacities: The admitted exchange extents, ascending, from
            :func:`admitted_exchange_extents` -- both bands' ladders merged.
            One K5T window is built at the last entry (the global capacity
            ``T``) and every entry retains one NCCL carrier workspace.
        device: The CUDA device this rank's buffers live on.
        fused_decode: Use D3 select/publish plus merge; otherwise use K5T.
        use_pdl: Explicit programmatic dependent launch policy for decode.
        classic_merge: Select the classic merge instead of the shuffle network.
        slots: Rotating symmetric windows; must satisfy the layer reuse rule.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        world_size: int,
        rank: int,
        num_heads_local: int,
        num_index_heads: int,
        num_sparse_layers: int,
        token_capacities: Sequence[int],
        device: torch.device,
        fused_decode: bool,
        use_pdl: bool,
        classic_merge: bool,
        slots: int = ICP_EXCHANGE_SLOTS,
    ) -> None:
        from fmha_sm100.icp.carrier import (  # noqa: PLC0415
            allocate_carrier_workspace,
            collective_exchange_and_merge,
        )
        from fmha_sm100.icp.tiled_exchange import IcpTiledExchange  # noqa: PLC0415

        if num_index_heads != world_size * num_heads_local:
            raise ValueError(
                f"ICP C7: num_index_heads ({num_index_heads}) must be "
                f"world_size ({world_size}) * num_heads_local "
                f"({num_heads_local}); the gathered head axis is rank-major "
                f"and any other split addresses another rank's heads."
            )

        capacities = [int(t) for t in token_capacities]
        if not capacities:
            raise ValueError("ICP: at least one exchange extent is required.")
        if capacities != sorted(set(capacities)) or capacities[0] < 1:
            raise ValueError(
                f"ICP: the exchange extents must be strictly ascending and "
                f"positive; got {capacities}. Both ranks must resolve the same "
                "extent from the same ladder."
            )

        self._closed = False
        self.group = group
        self.world_size = world_size
        self.rank = rank
        self.num_heads_local = num_heads_local
        self.num_index_heads = num_index_heads
        self.num_sparse_layers = num_sparse_layers
        self.token_capacities = capacities
        #: Membership test for the admitted extents. The ladder is a list
        #: because its ORDER is part of the contract (ascending, and
        #: identical on every rank); this is only the lookup.
        self._admitted_extents = frozenset(capacities)
        #: The global capacity: the largest extent any invocation may present,
        #: and the size of the one symmetric window.
        self.token_capacity = capacities[-1]
        self.device = device
        self.slots = slots

        # Checked before anything is allocated; capacity does not enter it.
        check_slot_discipline(num_sparse_layers, self.slots)

        #: D3: the decode selector publishes (`publish`) and `__call__`
        #: only merges. The window, slots and ACK-free discipline are K5T's.
        self.fused_decode = fused_decode
        self._fused = None
        self._exchange = None
        if self.fused_decode:
            from fmha_sm100.icp.fused_exchange import (  # noqa: PLC0415
                IcpFusedExchange,
            )

            self._fused = IcpFusedExchange(
                group,
                tokens=self.token_capacity,
                heads_group=num_index_heads,
                heads_local=num_heads_local,
                slots=self.slots,
                device=device,
                use_pdl=use_pdl,
                end_wait=True,
                classic_merge=classic_merge,
            )
            decode_route = self._fused
        else:
            self._exchange = IcpTiledExchange(
                group,
                tokens=self.token_capacity,
                heads_group=num_index_heads,
                heads_local=num_heads_local,
                slots=self.slots,
                device=device,
                use_pdl=use_pdl,
                classic_merge=classic_merge,
            )
            decode_route = self._exchange
        provenance = decode_route.provenance()
        self.decode_provenance = provenance
        #: The decode profile of the route actually constructed (read by the
        #: indexer's ICP_DECODE_PROFILE line and checked against its metadata).
        self.decode_profile = {
            "exchange": BACKEND_K5T,
            "merge": "classic" if classic_merge else "network",
            "d3": int(self._fused is not None),
            "pdl": int(use_pdl),
        }
        if provenance["use_ack"] != 0 or provenance["symm_backend"] != "CUDA":
            raise RuntimeError(
                f"ICP: K5T resolved use_ack={provenance['use_ack']} backend="
                f"{provenance['symm_backend']!r}; expected no ACK on the CUDA "
                "symmetric-memory backend."
            )

        self._collective = collective_exchange_and_merge
        # C5's failure word for the NCCL route: written by K2, never read here.
        self._status = torch.zeros(1, dtype=torch.int32, device=device)
        # One retained send/receive carrier per admitted extent, allocated on
        # every rank at startup: the collective's element counts are fixed by
        # the extent, and with these the route allocates nothing per layer.
        self._carrier_workspaces = {
            extent: allocate_carrier_workspace(
                world=world_size,
                qchunk=extent,
                h_local=num_heads_local,
                device=device,
                transport="all_to_all",
            )
            for extent in capacities
        }

    def close(self) -> None:
        """Drain/barrier the model-scoped window once on every TP rank."""
        if not self._closed:
            route = self._fused if self._fused is not None else self._exchange
            assert route is not None
            route.close()
            self._closed = True

    def publish(
        self,
        scores: torch.Tensor,
        geometry: object,
        *,
        layer_idx: int,
        token_offset: int,
    ) -> None:
        """D3 only: select rows ``[token_offset, +T_e)`` into the owners' windows.

        Every row below the invocation's extent must be published once per
        layer before (or while) ``__call__`` merges it; a missing row is a
        transport timeout, not a wrong answer.
        """
        if self._fused is None:
            raise RuntimeError(
                "ICP: publish() requires fused_decode=True (the D3 decode route)"
            )
        if self._closed:
            raise RuntimeError("ICP exchange was closed at model shutdown")
        self._fused.select_and_publish(
            scores, geometry, layer_idx=layer_idx, token_offset=token_offset
        )

    def __call__(
        self,
        local_candidates: torch.Tensor,
        *,
        layer_idx: int,
        has_prefill: bool,
        forced: torch.Tensor,
        n_ordinary: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Exchange one layer's candidates and merge them into ``out``.

        ``T`` here is the invocation's extent, not the process's global
        capacity: it is read off ``local_candidates``, must be one of the
        admitted extents, and selects that prefix of the one window. Every
        argument is checked against the same extent using host tensor metadata
        only, before submitting the selected transport.

        Args:
            local_candidates: int32 ``[T, H_group, 16, 2]``, query-major. Each
                record is an fp32 score **bitcast** into int32 at ``[..., 0]``
                and an int32 global block id at ``[..., 1]``; ``-1`` marks an
                invalid id.
            layer_idx: This indexer layer's index; the slot is
                ``layer_idx % slots``.
            has_prefill: Whether this batch contains any prefill row. The only
                input to the route decision.
            forced: int32 ``[T]``, C3's per-row forced block ``p // 128``, with
                ``-1`` for an inactive row.
            n_ordinary: int32 ``[T]``, how many ordinary winners each row wants.
            out: int32 ``[T, H_local, 16]`` destination. Written with this
                rank's global block ids, ascending, ``-1``-padded.

        Returns:
            ``out``.
        """
        backend = BACKEND_NCCL if has_prefill else BACKEND_K5T
        if self._closed:
            raise RuntimeError("ICP exchange was closed at model shutdown")
        capacity = self._check_candidates(local_candidates)
        self._check_plane(forced, "forced", capacity)
        self._check_plane(n_ordinary, "n_ordinary", capacity)
        self._check_out(out, capacity)
        if backend == BACKEND_NCCL:
            self._collective(
                local_candidates,
                world=self.world_size,
                rank=self.rank,
                out=out,
                forced=forced,
                n_ordinary=n_ordinary,
                status=self._status,
                transport="all_to_all",
                group=self.group,
                workspace=self._carrier_workspaces[capacity],
            )
        elif self._fused is not None:
            # The rows were published by this layer's selectors (`publish`);
            # `local_candidates` only fixes the extent.
            self._fused.merge(
                out, layer_idx=layer_idx, forced=forced, n_ordinary=n_ordinary
            )
        else:
            assert self._exchange is not None
            self._exchange.exchange_and_merge(
                local_candidates,
                out=out,
                layer_idx=layer_idx,
                forced=forced,
                n_ordinary=n_ordinary,
            )
        return out

    # -- validation -------------------------------------------------------

    def _check_candidates(self, cand: torch.Tensor) -> int:
        """Validate the carrier and return the extent it selects."""
        if cand.dtype is not torch.int32:
            raise TypeError(
                f"ICP C4 pins the candidate carrier at int32; got "
                f"{cand.dtype}. The score bits and the block id are both "
                f"bitcast, never converted -- pass `cand.view(torch.int32)`, "
                f"not `.int()` or `.to()`, which convert and destroy every "
                f"block id above 2**24."
            )
        if cand.dim() != 4:
            raise ValueError(
                f"ICP: local_candidates must be [T, H_group, {CAND_K}, 2], got "
                f"{tuple(cand.shape)}."
            )
        capacity = int(cand.shape[0])
        # Any prefix of the window is addressable, so this is not a kernel
        # requirement: the ladder is what makes the extent capture-fixed and
        # rank-identical (C4), a caller guarantee the kernel cannot check.
        if capacity not in self._admitted_extents or capacity > self.token_capacity:
            raise ValueError(
                f"ICP: local_candidates carries {capacity} token rows, which "
                f"is not one of the admitted exchange extents "
                f"{self.token_capacities}. The extent must come from "
                "`select_token_capacity` over the same config-derived ladder "
                "this exchange was built from, so that it is fixed across "
                "ranks and across capture replays; a live count would differ "
                "between ranks and deadlock the exchange."
            )
        shape = (capacity, self.num_index_heads, CAND_K, 2)
        if tuple(cand.shape) != shape:
            raise ValueError(
                f"ICP: local_candidates must be {shape}, got {tuple(cand.shape)}."
            )
        if not cand.is_contiguous():
            raise ValueError("ICP: local_candidates must be contiguous.")
        return capacity

    def _check_plane(self, plane: torch.Tensor, name: str, capacity: int) -> None:
        if plane.dtype is not torch.int32:
            raise TypeError(f"ICP C3: {name} must be int32, got {plane.dtype}.")
        if tuple(plane.shape) != (capacity,):
            raise ValueError(
                f"ICP C3: {name} is a per-token-row plane of shape "
                f"({capacity},), got {tuple(plane.shape)}."
            )

    def _check_out(self, out: torch.Tensor, capacity: int) -> None:
        shape = (capacity, self.num_heads_local, CAND_K)
        if out.dtype is not torch.int32:
            raise TypeError(f"ICP: out must be int32, got {out.dtype}.")
        if tuple(out.shape) != shape:
            raise ValueError(f"ICP: out must be {shape}, got {tuple(out.shape)}.")
