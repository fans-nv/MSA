# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT
"""Page-128 HND NVFP4 cache contract shared by both attention readers."""

from typing import TypedDict

import torch

from .cute.nvfp4_cache_contract import validate_nvfp4_views


class Nvfp4KvCache(TypedDict):
    """Zero-copy byte views and independent device dequantization scales.

    Data: [pages, heads, 128, 64]. Scales: [pages, heads, 128, 8].
    K scales are linear; V scales use (token//4)*32 + group*4 + token%4.
    Views retain the allocation's full page stride and their storage offsets.
    Global scales must be positive, finite, and immutable while pages live.
    """

    k_data: torch.Tensor
    v_data: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    layout: str
    k_global_scale: torch.Tensor
    v_global_scale: torch.Tensor


def validate_nvfp4_kv(cache: Nvfp4KvCache, device):
    """Validate the descriptor and return its real byte page stride.

    Scale owners validate positive finite values at initialization, before cache
    use. This function never synchronizes values from device to host.
    """
    missing = Nvfp4KvCache.__required_keys__.difference(cache)
    if missing:
        raise ValueError(f"nvfp4_kv is missing {sorted(missing)}")
    if cache["layout"] != "vllm":
        raise ValueError("nvfp4_kv supports layout='vllm' only")
    stride = validate_nvfp4_views(
        cache["k_data"], cache["v_data"], cache["k_scale"], cache["v_scale"], device
    )
    for name in ("k_global_scale", "v_global_scale"):
        scale = cache[name]
        if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32:
            raise TypeError(f"{name} must be a device float32 scalar tensor")
        if scale.device != device or scale.numel() != 1 or not scale.is_contiguous():
            raise ValueError(f"{name} must be one contiguous float32 value on {device}")
    return stride
