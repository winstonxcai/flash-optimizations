"""Mustafar-compatible MSB-first 512-bit bitmap helpers."""

from __future__ import annotations

import torch

from . import config


def mask_to_bitmap(mask: torch.Tensor) -> torch.Tensor:
    """Encode bool rows; coordinate ``64*w+lane`` uses bit ``63-lane``."""
    n = mask.shape[0]
    bits = mask.reshape(n, config.BITMAP_WORDS, 64).to(torch.int64)
    shifts = (1 << (63 - torch.arange(64, device=mask.device))).to(torch.int64)
    return (bits * shifts[None, None, :]).sum(dim=2)


def bitmap_to_mask(bitmap: torch.Tensor) -> torch.Tensor:
    """Decode signed or unsigned int64-compatible words to boolean rows."""
    signed = bitmap.view(torch.int64)
    shifts = 63 - torch.arange(64, device=bitmap.device)
    bits = (signed[:, :, None] >> shifts[None, None, :]) & 1
    return bits.reshape(bitmap.shape[0], -1).bool()
