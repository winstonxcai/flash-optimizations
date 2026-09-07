"""Pure Torch TopMag pruning and packed row-format references.

No persistent state here. The pruning helpers are shared production primitives;
the Torch pack/unpack functions are numerical references. Production storage and
reconstruction wrappers live in packed.py. SGLang source edits live in patches/
and are applied through patching.py.

The exact-global keep-mask (`topmag_keep_mask`) is the single source of truth.
It MUST be computed once from the unmodified latent and passed explicitly to
both the dense-zero baseline and the Triton packer — re-running `topk` (e.g.
after pruning, or on inputs with natural zeros / cutoff ties) does not
reproduce the same index set.
"""

import torch

from . import config
from .bitmap import bitmap_to_mask, mask_to_bitmap


def topmag_keep_mask(latent: torch.Tensor, keep: float) -> torch.Tensor:
    """Exact-global TopMag keep-mask, bool [n, HEAD_DIM], True = keep.

    keep_k  = round(HEAD_DIM * keep)  retained coords per row
    prune_k = HEAD_DIM - keep_k       dropped coords per row
    The mask is the complement of `mag.topk(prune_k, largest=False).indices`,
    i.e. the largest-|·| keep_k coords. Computed from the UNMODIFIED latent;
    never recompute it after pruning (see module docstring).
    """
    keep_k = int(round(config.HEAD_DIM * keep))
    prune_k = config.HEAD_DIM - keep_k
    mask = torch.ones(latent.shape, dtype=torch.bool, device=latent.device)
    if prune_k <= 0:
        return mask
    mag = latent.abs().float()
    _, idx = mag.topk(prune_k, dim=-1, largest=False)  # [n, prune_k]
    # Advanced-index assignment with the Python scalar ``False`` triggers a
    # CPU-to-CUDA scalar copy, which CUDA graph capture rejects. scatter_
    # emits the same device-side mask update without changing TopK semantics.
    mask.scatter_(dim=-1, index=idx, value=False)
    return mask


def topmag_zero_from_mask(
    latent: torch.Tensor, keep_mask: torch.Tensor
) -> torch.Tensor:
    """Zero every coord where keep_mask is False, in place.

    Zeros survive any downstream per-row RMSNorm scaling (a scalar multiply per
    row) and the native fp8 store encodes them as exactly 0. The mask must come
    from `topmag_keep_mask` on the unmodified latent.
    """
    latent.masked_fill_(~keep_mask, 0.0)
    return latent


def _ue8m0_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for SGLang's ceil-to-UE8M0 FP8 quantization."""
    if x.shape[0] == 0:
        return (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((0, config.BITMAP_WORDS), dtype=torch.uint8, device=x.device),
        )
    tiles = x.float().reshape(-1, config.BITMAP_WORDS, config.TILE_SIZE)
    abs_max = tiles.abs().amax(dim=-1)
    raw_scale = abs_max.clamp_min(1.0e-4) / config.FP8_E4M3_MAX
    exponent = torch.ceil(torch.log2(raw_scale)).to(torch.int32)
    scale_codes = (exponent + 127).to(torch.uint8)
    scales = torch.exp2(exponent.float())
    quantized = torch.clamp(
        tiles / scales.unsqueeze(-1),
        -config.FP8_E4M3_MAX,
        config.FP8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    return quantized.reshape(-1, config.HEAD_DIM), scale_codes


def pack_rows_ref(
    latent: torch.Tensor,
    keep_mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Torch row-format reference, independent of page locations."""
    if keep_mask.dtype != torch.bool or keep_mask.shape != latent.shape:
        raise ValueError("keep_mask must be bool and match latent")
    if latent.shape[-1] != config.HEAD_DIM:
        raise ValueError("packed rows must have 512 coordinates")
    if latent.shape[0] and not bool(
        (keep_mask.sum(-1) == config.PACKED_KEPT_VALUES).all()
    ):
        raise ValueError("every packed row must retain exactly 256 coordinates")
    masked = latent.float().masked_fill(~keep_mask, 0.0)
    inv_rms = torch.rsqrt(masked.square().mean(-1, keepdim=True) + norm_eps)
    normalized = (masked * inv_rms * norm_weight.float()).to(torch.bfloat16)
    quantized, scales = _ue8m0_quantize(normalized)
    columns = torch.nonzero(keep_mask, as_tuple=False)[:, 1].reshape(
        latent.shape[0], config.PACKED_KEPT_VALUES
    )
    codes = quantized.view(torch.uint8).gather(1, columns)
    bitmaps = mask_to_bitmap(keep_mask).view(torch.uint64)
    return codes, bitmaps, scales


def unpack_rows_ref(
    values: torch.Tensor,
    bitmaps: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Reference pre-RoPE BF16 reconstruction for unit tests."""
    bits = bitmap_to_mask(bitmaps)
    rank = bits.cumsum(-1) - 1
    codes = values.gather(1, rank.clamp_min(0))
    q = codes.view(torch.float8_e4m3fn).float()
    scale = torch.exp2(scales.to(torch.int32).float() - 127)
    scale = scale.repeat_interleave(config.TILE_SIZE, dim=-1)
    return torch.where(bits, q * scale, torch.zeros_like(q)).to(torch.bfloat16)
