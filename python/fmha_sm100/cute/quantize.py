# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Transformer Engine NVFP4 quantization helper.

This file is intended as a customer-facing example for preparing KV tensors
for the KVFP4 attention kernel:
  - BF16/FP16 K/V input
  - packed E2M1 FP4 data from Transformer Engine
  - E4M3 block scales in cuBLAS/cuDNN 128x4 tiled layout
  - one FP32 tensor/global scale per tensor
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch


NVFP4_BLOCK_SIZE = 16
NVFP4_FP4_MAX = 6.0
NVFP4_FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class Nvfp4QuantizedTensor:
    """Packed NVFP4 tensor plus dequantization metadata.

    Attributes
    ----------
    data : torch.Tensor
        Packed E2M1 FP4 data from Transformer Engine.  The last dimension is
        half of the original logical last dimension because each byte stores
        two FP4 values.
    scale_128x4 : torch.Tensor
        E4M3 block scales in cuBLAS/cuDNN 128x4 tiled rowwise storage.
    global_scale : torch.Tensor
        FP32 tensor/global dequant scale.
    logical_scale_shape : tuple[int, int]
        Logical 2D scale shape ``(rows, cols)`` before 128x4 swizzling.
    original_shape : tuple[int, ...]
        Original BF16/FP16 tensor shape before quantization.
    """

    data: torch.Tensor
    scale_128x4: torch.Tensor
    global_scale: torch.Tensor
    logical_scale_shape: Tuple[int, int]
    original_shape: Tuple[int, ...]


def _round_up(x: int, multiple: int) -> int:
    return ((int(x) + multiple - 1) // multiple) * multiple


def nvfp4_scale_128x4_offset(
    row: torch.Tensor,
    col: torch.Tensor,
    scale_cols: int,
) -> torch.Tensor:
    """Return flat offsets for cuBLAS/cuDNN 128x4 rowwise scale storage.

    Parameters
    ----------
    row : torch.Tensor
        Logical row indices.
    col : torch.Tensor
        Logical scale-column indices.
    scale_cols : int
        Logical number of scale columns before padding to a multiple of 4.

    Returns
    -------
    torch.Tensor
        Flat offsets into the padded 128x4 tiled storage.
    """

    tiles_n = _round_up(scale_cols, 4) // 4
    tile_m = row // 128
    tile_n = col // 4
    outer = row % 128
    inner = col % 4
    return (
        (tile_m * tiles_n + tile_n) * 512
        + (outer % 32) * 16
        + (outer // 32) * 4
        + inner
    )


def swizzle_nvfp4_scale_to_128x4(
    scale: torch.Tensor,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Convert TE logical rowwise scales to cuBLAS/cuDNN 128x4 tiled layout.

    Parameters
    ----------
    scale : torch.Tensor
        Logical rowwise scale tensor with at least shape ``[rows, cols]``.
    rows : int
        Number of logical rows to convert.
    cols : int
        Number of logical scale columns to convert.

    Returns
    -------
    torch.Tensor
        Scale tensor padded to ``round_up(rows, 128)`` by ``round_up(cols, 4)``
        and swizzled into 128x4 tiled storage.
    """

    if scale.ndim != 2:
        raise ValueError(f"scale must be 2D, got shape {tuple(scale.shape)}")

    rows = int(rows)
    cols = int(cols)
    padded_rows = _round_up(rows, 128)
    padded_cols = _round_up(cols, 4)
    if scale.shape[0] < rows or scale.shape[1] < cols:
        raise ValueError(
            "scale is smaller than the requested logical shape: "
            f"got {tuple(scale.shape)}, need at least {(rows, cols)}"
        )

    logical = scale[:rows, :cols].contiguous()
    if logical.shape != (padded_rows, padded_cols):
        logical = torch.nn.functional.pad(
            logical.to(torch.float32),
            (0, padded_cols - cols, 0, padded_rows - rows),
        ).to(scale.dtype)
    swizzled = torch.empty_like(logical)

    row = torch.arange(padded_rows, device=scale.device, dtype=torch.int64)[:, None]
    col = torch.arange(padded_cols, device=scale.device, dtype=torch.int64)[None, :]
    offset = nvfp4_scale_128x4_offset(row, col, padded_cols).reshape(-1)
    swizzled.reshape(-1)[offset] = logical.reshape(-1)
    return swizzled


def nvfp4_global_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Compute TE NVFP4 tensor/global dequant scale from rowwise amax.

    Parameters
    ----------
    amax : torch.Tensor
        Rowwise absolute maxima returned by Transformer Engine.

    Returns
    -------
    torch.Tensor
        FP32 global scale equal to ``amax / (448 * 6)``.
    """

    return amax.to(torch.float32) / (NVFP4_FP8_E4M3_MAX * NVFP4_FP4_MAX)


def _import_te_nvfp4_quantizer():
    try:
        from transformer_engine.pytorch.tensor import NVFP4Quantizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Transformer Engine NVFP4 quantization is unavailable. Install a "
            "Transformer Engine build with its PyTorch dependencies, including "
            "FlashAttention v3 when required by that TE build."
        ) from exc
    return NVFP4Quantizer


# ---------------------------------------------------------------------------
# Dependency-free (pure PyTorch) NVFP4 quantizer
# ---------------------------------------------------------------------------
#
# This is a drop-in replacement for the Transformer Engine path above.  It
# exists so the NVFP4 accuracy tests can run on machines without a working
# Transformer Engine build; it is an exact round-trip partner of
# ``dequantize_nvfp4_128x4_to_bf16`` below and uses the same physical layout the
# kernel reads (``_scale_128x4_offset`` in
# ``src/sm100/fwd/atten_fwd_nvfp4_kv.py``).

# FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit.  Codes 0..7 hold the
# non-negative magnitudes and bit 3 is the sign.  Same table as the LUT used by
# ``dequantize_nvfp4_128x4_to_bf16``.
_E2M1_MAGNITUDES: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# ``_E2M1_MIDPOINTS[i]`` is the tie point between code ``i`` and code ``i + 1``.
_E2M1_MIDPOINTS: Tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _round_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 values to FP4 E2M1 codes, round-to-nearest-even.

    Ties resolve to the code whose mantissa bit is zero, i.e. the even code,
    matching ``cvt.rn.satfinite.e2m1x2.f32``.  Magnitudes above 6 saturate to
    code 7 (6.0); the sign is preserved in bit 3, including for negative zero.
    """

    xf = torch.nan_to_num(x.to(torch.float32), nan=0.0)
    sign = torch.signbit(xf)
    magnitude = xf.abs().contiguous()
    midpoints = torch.tensor(
        _E2M1_MIDPOINTS, dtype=torch.float32, device=xf.device
    )
    # ``low`` counts midpoints strictly below the magnitude (round down on a
    # tie); ``high`` counts midpoints at or below it (round up on a tie).  They
    # differ only exactly on a tie, where the tie sits at index ``low``.
    low = torch.searchsorted(midpoints, magnitude, right=False)
    high = torch.searchsorted(midpoints, magnitude, right=True)
    code = torch.where(low % 2 == 0, low, high).to(torch.uint8)
    return code | (sign.to(torch.uint8) << 3)


def _pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack pairs of E2M1 codes into bytes; even element in the low nibble."""

    return (codes[..., 0::2] & 0x0F) | (codes[..., 1::2] << 4)


def _quantize_bf16_to_nvfp4_128x4_torch(
    x: torch.Tensor,
    *,
    rows: int,
    scale_cols: int,
) -> "Nvfp4QuantizedTensor":
    """Quantize to NVFP4 with plain PyTorch, no Transformer Engine.

    Follows the same recipe TE uses:
      global_scale = amax(|x|) / (448 * 6)
      block_scale  = to_e4m3(amax(|block|) / (6 * global_scale))    <= 448
      fp4          = to_e2m1(block / (block_scale * global_scale))
    so ``fp4 * block_scale * global_scale`` reconstructs the value exactly in
    FP32, which is what ``dequantize_nvfp4_128x4_to_bf16`` computes.
    """

    flat = x.detach().to(torch.float32).contiguous().reshape(rows, -1)
    amax = flat.abs().amax()
    global_scale = nvfp4_global_scale_from_amax(amax.reshape(1))
    # An all-zero tensor would give a zero (and thus unusable) global scale.
    global_scale = torch.where(
        global_scale > 0, global_scale, torch.ones_like(global_scale)
    )

    blocks = flat.reshape(rows, scale_cols, NVFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)
    encoded = (block_amax / (NVFP4_FP4_MAX * global_scale)).clamp(
        min=0.0, max=NVFP4_FP8_E4M3_MAX
    )
    block_scale = encoded.to(torch.float8_e4m3fn)
    dequant_step = block_scale.to(torch.float32) * global_scale
    nonzero = dequant_step > 0
    safe_step = torch.where(nonzero, dequant_step, torch.ones_like(dequant_step))

    codes = _round_to_e2m1_codes(blocks / safe_step.unsqueeze(-1))
    codes = torch.where(nonzero.unsqueeze(-1), codes, torch.zeros_like(codes))
    data = _pack_e2m1_codes(codes.reshape(rows, -1)).reshape(
        *x.shape[:-1], x.shape[-1] // 2
    )

    scale_128x4 = swizzle_nvfp4_scale_to_128x4(
        block_scale.view(torch.uint8),
        rows=rows,
        cols=scale_cols,
    )

    return Nvfp4QuantizedTensor(
        data=data.contiguous(),
        scale_128x4=scale_128x4.contiguous(),
        global_scale=global_scale.to(torch.float32).contiguous(),
        logical_scale_shape=(rows, scale_cols),
        original_shape=tuple(int(v) for v in x.shape),
    )


def te_nvfp4_quantizer_available() -> bool:
    """Return True when Transformer Engine's NVFP4 quantizer can be imported."""

    try:
        _import_te_nvfp4_quantizer()
    except RuntimeError:
        return False
    return True


def quantize_bf16_to_nvfp4_128x4(
    x: torch.Tensor,
    *,
    backend: str = "auto",
) -> Nvfp4QuantizedTensor:
    """Quantize a BF16/FP16 tensor to NVFP4.

    TE returns rowwise scales in logical padded layout.  This helper returns
    the scales in physical 128x4 tiled storage, so the attention kernel can
    load them with ``nvfp4_scale_128x4_offset``.

    Two backends produce the same layout:
      * ``"te"``    - Transformer Engine's ``NVFP4Quantizer``.
      * ``"torch"`` - a dependency-free PyTorch implementation of the same
        recipe.  Use it when Transformer Engine is not installed.
      * ``"auto"``  - TE if importable, otherwise ``"torch"``.  Override with
        the ``MSA_NVFP4_QUANTIZER_BACKEND`` environment variable.

    Parameters
    ----------
    x : torch.Tensor
        CUDA BF16 or FP16 tensor.  The last dimension must be divisible by 16,
        and the flattened row dimension ``prod(x.shape[:-1])`` must also be
        divisible by 16.
    backend : str, optional
        ``"auto"``, ``"te"`` or ``"torch"``.

    Returns
    -------
    Nvfp4QuantizedTensor
        Packed FP4 data, 128x4-swizzled block scales, global scale, and shape
        metadata needed by the KVFP4 attention kernel or by reference
        dequantization.
    """

    if not x.is_cuda:
        raise ValueError("NVFP4 quantization requires a CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"x must be bf16 or fp16, got {x.dtype}")
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.ndim}")
    if x.shape[-1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"last dimension must be divisible by {NVFP4_BLOCK_SIZE}, got {x.shape[-1]}"
        )

    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if rows % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "flattened row dimension must be divisible by "
            f"{NVFP4_BLOCK_SIZE}, got {rows}"
        )

    scale_cols = int(x.shape[-1]) // NVFP4_BLOCK_SIZE

    # An explicit argument wins; the environment only supplies the default, so
    # a caller that asks for a specific backend always gets it (or an error).
    backend = backend.lower()
    if backend == "auto":
        backend = os.environ.get("MSA_NVFP4_QUANTIZER_BACKEND", "auto").lower()
    if backend not in ("auto", "te", "torch"):
        raise ValueError(
            f"backend must be 'auto', 'te' or 'torch', got {backend!r}"
        )
    if backend == "auto":
        backend = "te" if te_nvfp4_quantizer_available() else "torch"
    if backend == "torch":
        return _quantize_bf16_to_nvfp4_128x4_torch(
            x, rows=rows, scale_cols=scale_cols
        )

    NVFP4Quantizer = _import_te_nvfp4_quantizer()
    quantizer = NVFP4Quantizer(rowwise=True, columnwise=False)
    qx = quantizer.quantize(x.contiguous())
    meta = qx.get_metadata()

    data = meta["rowwise_data"]
    if data.dtype is not torch.uint8:
        data = data.view(torch.uint8)
    logical_scale = meta["rowwise_scale_inv"]
    amax = meta["amax_rowwise"]
    scale_128x4 = swizzle_nvfp4_scale_to_128x4(
        logical_scale,
        rows=rows,
        cols=scale_cols,
    )
    global_scale = nvfp4_global_scale_from_amax(amax).contiguous()

    return Nvfp4QuantizedTensor(
        data=data,
        scale_128x4=scale_128x4,
        global_scale=global_scale,
        logical_scale_shape=(rows, scale_cols),
        original_shape=tuple(int(v) for v in x.shape),
    )


def quantize_kv_bf16_to_nvfp4_128x4(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[Nvfp4QuantizedTensor, Nvfp4QuantizedTensor]:
    """Quantize BF16/FP16 K and V tensors independently for KVFP4 attention.

    Parameters
    ----------
    k : torch.Tensor
        CUDA BF16 or FP16 K tensor.
    v : torch.Tensor
        CUDA BF16 or FP16 V tensor.

    Returns
    -------
    tuple[Nvfp4QuantizedTensor, Nvfp4QuantizedTensor]
        Quantized K and V tensors with independent scales.
    """

    return quantize_bf16_to_nvfp4_128x4(k), quantize_bf16_to_nvfp4_128x4(v)


def dequantize_nvfp4_128x4_to_bf16(
    qx: Nvfp4QuantizedTensor,
    *,
    include_global_scale: bool = True,
) -> torch.Tensor:
    """Reference dequantization for validation.

    This mirrors the kernel contract:
      x = e2m1 * E4M3_block_scale_1x16 * FP32_global_scale

    Parameters
    ----------
    qx : Nvfp4QuantizedTensor
        Quantized tensor returned by ``quantize_bf16_to_nvfp4_128x4``.
    include_global_scale : bool, optional
        If True, multiply by ``qx.global_scale`` after applying per-block
        scales.

    Returns
    -------
    torch.Tensor
        BF16 tensor with shape ``qx.original_shape``.
    """

    data = qx.data if qx.data.dtype is torch.uint8 else qx.data.view(torch.uint8)
    if data.shape[-1] * 2 != qx.original_shape[-1]:
        raise ValueError(
            "packed data last dimension does not match original shape: "
            f"{data.shape[-1]} packed vs {qx.original_shape[-1]} logical"
        )

    rows, scale_cols = qx.logical_scale_shape
    logical_dim = int(qx.original_shape[-1])
    if scale_cols * NVFP4_BLOCK_SIZE != logical_dim:
        raise ValueError(
            "logical scale columns do not match original last dimension: "
            f"{scale_cols} scale cols vs dim {logical_dim}"
        )

    fp4_lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=data.device,
    )
    packed = data.reshape(rows, logical_dim // 2)
    lo = packed & 0x0F
    hi = packed >> 4
    values = torch.empty((rows, logical_dim), dtype=torch.float32, device=data.device)
    values[:, 0::2] = fp4_lut[lo.long()]
    values[:, 1::2] = fp4_lut[hi.long()]

    row = torch.arange(rows, device=data.device, dtype=torch.int64)[:, None]
    col = torch.arange(scale_cols, device=data.device, dtype=torch.int64)[None, :]
    offset = nvfp4_scale_128x4_offset(row, col, scale_cols)
    scale_u8 = qx.scale_128x4.reshape(-1)[offset.reshape(-1)].reshape(rows, scale_cols)
    scale = scale_u8.view(torch.float8_e4m3fn).to(torch.float32)
    scale = scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=1)
    out = values * scale
    if include_global_scale:
        global_scale = qx.global_scale.reshape(-1)[0].to(torch.float32)
        out = out * global_scale
    return out.reshape(qx.original_shape).to(torch.bfloat16)


def _example() -> None:
    device = torch.device("cuda")
    k = torch.randn(128, 2, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_q, v_q = quantize_kv_bf16_to_nvfp4_128x4(k, v)
    print("K FP4 data:", tuple(k_q.data.shape), k_q.data.dtype)
    print("K scale 128x4:", tuple(k_q.scale_128x4.shape), k_q.scale_128x4.dtype)
    print("K global scale:", tuple(k_q.global_scale.shape), k_q.global_scale.dtype)
    print("V FP4 data:", tuple(v_q.data.shape), v_q.data.dtype)
    print("V scale 128x4:", tuple(v_q.scale_128x4.shape), v_q.scale_128x4.dtype)
    print("V global scale:", tuple(v_q.global_scale.shape), v_q.global_scale.dtype)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("quantize.py requires CUDA")
    _example()
