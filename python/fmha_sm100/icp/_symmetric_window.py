"""Host lifecycle shared by D3 and K5T; their native state stays separate.

Subclasses bind their own plans, flags and generation tensors. Allocation and
initialization are separate so those tensors are allocated before the window
is zeroed and collectively made visible, in the original order.
"""

from __future__ import annotations

from typing import TypeVar

import torch

_Window = TypeVar("_Window", bound="_SymmetricWindow")


class _SymmetricWindow:
    """Private host-only mixin. No method here runs on a serving launch."""

    def _validate_geometry(self, min_slots: int, *, no_ack_prefix: str = "") -> None:
        if self.Hg != self.world * self.Hl:
            raise ValueError(
                f"C7: H_group ({self.Hg}) must be world ({self.world}) * "
                f"H_local ({self.Hl})")
        if self.T < 1:
            raise ValueError("tokens must be >= 1")
        if self.slots < min_slots:
            raise ValueError(
                f"slots={self.slots}: {no_ack_prefix}no ACK, so slot rotation is "
                f"the only reuse protection and needs >= {min_slots} slots")
        if self.max_ctas < 0:
            raise ValueError("max_ctas must be >= 0")

    def _set_device(self, device: torch.device | int | None) -> int:
        if device is None:
            index = torch.cuda.current_device()
        elif isinstance(device, int):
            index = device
        else:
            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError(f"device must be CUDA, got {device}")
            index = (torch.cuda.current_device() if device.index is None
                     else device.index)
        self.device = torch.device("cuda", index)
        return index

    def _allocate_window(self, symm_mem, rendezvous, index: int, *,
                         transport: str, backend: str) -> None:
        self._data = symm_mem.empty(self.slot_words * self.slots,
                                    dtype=torch.int32, device=index)
        self._h_data = rendezvous(symm_mem, self._data, self.group)
        actual = symm_mem.get_backend(self.device) if hasattr(
            symm_mem, "get_backend") else None
        if actual not in (None, backend):
            raise RuntimeError(
                f"symmetric memory backend is {actual!r}; {transport} is qualified on "
                f"the {backend} backend only. Leave TORCH_SYMMMEM and "
                "NVSHMEM_MAX_TEAMS unset.")
        self.symm_backend = actual or backend

    def _initialize_window(self, index: int) -> None:
        self._data.zero_()
        torch.cuda.synchronize(index)
        self._h_data.barrier()
        torch.cuda.synchronize(index)
        self._buf_ptrs = [int(p) for p in self._h_data.buffer_ptrs]
        if len(self._buf_ptrs) != self.world:
            raise RuntimeError("rendezvous returned the wrong peer count")
        if self._buf_ptrs[self.rank] != self._data.data_ptr():
            raise RuntimeError("rendezvous did not map this rank's window")
        self._closed = False

    def close(self) -> None:
        """Drain, barrier, drain, drop. Collective; every rank must call it."""
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize(self.device)
        try:
            self._h_data.barrier()
            torch.cuda.synchronize(self.device)
        except Exception:  # noqa: BLE001 - teardown must not mask the cause
            pass
        self._h_data = None
        self._data = None
        self._buf_ptrs = []

    def __enter__(self: _Window) -> _Window:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
