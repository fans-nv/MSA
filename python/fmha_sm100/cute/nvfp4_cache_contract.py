# SPDX-License-Identifier: MIT
import torch


def validate_nvfp4_views(k, v, k_sf, v_sf, device):
    """Validate metadata without reading device values or copying pages."""
    planes = (
        ("k_data", k, 64),
        ("v_data", v, 64),
        ("k_scale", k_sf, 8),
        ("v_scale", v_sf, 8),
    )
    if not isinstance(k, torch.Tensor) or k.ndim != 4:
        raise ValueError("NVFP4 k_data must have shape [pages, heads, 128, 64]")
    pages, heads = k.shape[:2]
    if pages < 1 or heads < 1:
        raise ValueError("NVFP4 cache must contain pages and KV heads")
    page_stride = k.stride(0)
    if page_stride < 18432 * heads or page_stride % 16:
        raise ValueError("NVFP4 page stride must cover the full aligned packed page")
    for name, plane, width in planes:
        if not isinstance(plane, torch.Tensor) or plane.dtype != torch.uint8:
            raise TypeError(
                f"{name} must be uint8; reinterpret E4M3 scales with view(torch.uint8)"
            )
        if plane.device != device:
            raise ValueError(f"{name} must be on {device}")
        if tuple(plane.shape) != (pages, heads, 128, width):
            raise ValueError(f"{name} must have shape {(pages, heads, 128, width)}")
        if plane.stride() != (page_stride, 128 * width, width, 1):
            raise ValueError(
                f"{name} must have page-128 HND byte strides; got {plane.stride()}"
            )
        if plane.data_ptr() % 16:
            raise ValueError(f"{name} must have a 16-byte-aligned storage offset")
    return page_stride
