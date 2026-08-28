"""Pure-Torch TopMag and packed-C4 reference operations.

No runtime state here — just the per-row magnitude prune. The hook lives in
ops.py; the patch machinery is in ops.py too.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import config


FP8_DTYPE = torch.float8_e4m3fn


def topmag_keep_mask(latent: torch.Tensor, keep: float = 0.5) -> torch.Tensor:
    """Return the exact keep mask used by the dense-zero accuracy baseline."""
    keep_count = int(round(config.HEAD_DIM * keep))
    if keep_count < 0 or keep_count > config.HEAD_DIM:
        raise ValueError(f"keep must be in [0, 1], got {keep}")
    if keep_count == config.HEAD_DIM:
        return torch.ones_like(latent, dtype=torch.bool)
    if keep_count == 0:
        return torch.zeros_like(latent, dtype=torch.bool)
    prune_count = config.HEAD_DIM - keep_count
    prune_idx = latent.abs().float().topk(
        prune_count, dim=-1, largest=False
    ).indices
    mask = torch.ones_like(latent, dtype=torch.bool)
    mask.scatter_(-1, prune_idx, False)
    return mask


def topmag_zero(latent: torch.Tensor, keep: float) -> torch.Tensor:
    """Zero the smallest (1-keep) fraction of coords per row, in place.

    latent: [n, HEAD_DIM] float (bf16 / fp16 / fp32 — the c4 compressor output).
    The smallest-|·| k coords per row are set to 0. Zeros survive any downstream
    per-row RMSNorm scaling (a scalar multiply per row) and the native fp8 store
    encodes them as exactly 0. keep=1.0 is a no-op.
    """
    if keep >= 1.0:
        return latent
    mask = topmag_keep_mask(latent, keep)
    latent.masked_fill_(~mask, 0.0)
    return latent


def rms_norm_masked(
    latent: torch.Tensor,
    keep_mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """Apply the native RMSNorm to the dense-zero TopMag row."""
    zeroed = latent.float().masked_fill(~keep_mask, 0.0)
    inv_rms = torch.rsqrt(zeroed.square().mean(-1, keepdim=True) + norm_eps)
    return zeroed * inv_rms * norm_weight.float()


def quantize_tiles(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Native-style per-64 ue8m0 scaling with E4M3 payload bytes."""
    n = x.shape[0]
    tiles = x.float().view(n, config.SCALE_TILES, config.TILE_SIZE)
    finfo = torch.finfo(FP8_DTYPE)
    maxabs = tiles.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    exponent = torch.ceil(torch.log2(maxabs / finfo.max)).clamp(-127.0, 128.0)
    scaled = (tiles / torch.exp2(exponent)).clamp(finfo.min, finfo.max)
    fp8 = scaled.to(FP8_DTYPE).reshape(n, config.HEAD_DIM)
    scales = (exponent.squeeze(-1) + 127.0).to(torch.uint8)
    return fp8.view(torch.uint8), scales


def dequantize_tiles(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    n = codes.shape[0]
    fp8 = codes.contiguous().view(FP8_DTYPE).float()
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(
        config.TILE_SIZE, dim=-1
    )
    return fp8.view(n, config.HEAD_DIM) * scale


def bitmap_from_mask(keep_mask: torch.Tensor) -> torch.Tensor:
    """Encode MSB-first 64-bit words, matching upstream Mustafar."""
    chunks = keep_mask.view(-1, config.BITMAP_WORDS, 64).to(torch.int64)
    weights = torch.tensor(
        [-(1 << 63)] + [1 << shift for shift in range(62, -1, -1)],
        dtype=torch.int64,
        device=keep_mask.device,
    )
    return (chunks * weights).sum(-1).view(torch.uint64)


def mask_from_bitmap(bitmap: torch.Tensor) -> torch.Tensor:
    signed = bitmap.contiguous().view(torch.int64)
    shifts = torch.arange(63, -1, -1, device=bitmap.device, dtype=torch.int64)
    bits = (signed.unsqueeze(-1) >> shifts) & 1
    return bits.bool().view(*bitmap.shape[:-1], config.HEAD_DIM)


def _flat_pool_tensors(packed_pool):
    return (
        packed_pool.packed_values.view(-1, config.KEEP_DIM),
        packed_pool.bitmap.view(-1, config.BITMAP_WORDS),
        packed_pool.packed_scales.view(-1, config.SCALE_TILES),
    )


def pack_c4_rows_reference(
    latent: torch.Tensor,
    keep_mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    locations: torch.Tensor,
    packed_pool,
) -> None:
    """Pack normalized TopMag rows into fixed-stride paged tensors."""
    if latent.shape != keep_mask.shape or latent.shape[-1] != config.HEAD_DIM:
        raise ValueError("latent and keep_mask must both be [N, 512]")
    counts = keep_mask.sum(-1)
    if not torch.equal(counts, torch.full_like(counts, config.KEEP_DIM)):
        raise ValueError("packed C4 requires exactly 256 retained coordinates per row")
    normed = rms_norm_masked(latent, keep_mask, norm_weight, norm_eps)
    codes, scales = quantize_tiles(normed)
    coords = torch.arange(config.HEAD_DIM, device=latent.device).expand_as(keep_mask)
    kept_coords = coords[keep_mask].view(-1, config.KEEP_DIM)
    packed_codes = codes.gather(-1, kept_coords)
    values, bitmaps, packed_scales = _flat_pool_tensors(packed_pool)
    loc = locations.long()
    values[loc] = packed_codes
    bitmaps.view(torch.int64)[loc] = bitmap_from_mask(keep_mask).view(torch.int64)
    packed_scales[loc] = scales


def _apply_rope_tail(
    dense: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> None:
    if freqs_cis.is_complex():
        freq = torch.view_as_real(freqs_cis[positions.long()]).float()
    else:
        freq = freqs_cis[positions.long()].float().view(-1, config.ROPE_DIM // 2, 2)
    tail = dense[:, config.NOPE_DIM :].view(-1, config.ROPE_DIM // 2, 2)
    real, imag = tail[..., 0].clone(), tail[..., 1].clone()
    tail[..., 0] = real * freq[..., 0] - imag * freq[..., 1]
    tail[..., 1] = real * freq[..., 1] + imag * freq[..., 0]


def unpack_gather_c4_reference(
    packed_pool,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    freqs_cis: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather packed rows, densify, and apply C4 RoPE at position 4*raw."""
    if physical_indices.shape != raw_indices.shape:
        raise ValueError("physical_indices and raw_indices must have identical shape")
    if output is None:
        output = torch.zeros(
            *physical_indices.shape,
            config.HEAD_DIM,
            dtype=torch.bfloat16,
            device=physical_indices.device,
        )
    output.zero_()
    flat_physical = physical_indices.reshape(-1)
    flat_raw = raw_indices.reshape(-1)
    valid = (flat_physical >= 0) & (flat_raw >= 0)
    if not bool(valid.any()):
        return output
    values, bitmaps, packed_scales = _flat_pool_tensors(packed_pool)
    loc = flat_physical[valid].long()
    selected_bitmap = bitmaps.view(torch.int64)[loc].contiguous().view(torch.uint64)
    masks = mask_from_bitmap(selected_bitmap)
    ranks = masks.to(torch.int64).cumsum(-1) - 1
    gathered = values[loc].gather(-1, ranks.clamp_min(0))
    dense_codes = torch.zeros(
        loc.shape[0], config.HEAD_DIM, dtype=torch.uint8, device=loc.device
    )
    dense_codes[masks] = gathered[masks]
    dense = dequantize_tiles(dense_codes, packed_scales[loc])
    positions = (flat_raw[valid].long() * config.C4_RATIO).clamp(
        0, freqs_cis.shape[0] - 1
    )
    _apply_rope_tail(dense, freqs_cis, positions)
    output.view(-1, config.HEAD_DIM)[valid] = dense.to(torch.bfloat16)
    return output
