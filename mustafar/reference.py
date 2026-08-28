"""Pure torch TopMag pruning math (native c4-latent scope).

No runtime state here — just the per-row magnitude prune. The hook lives in
ops.py; the patch machinery is in ops.py too.

The exact-global keep-mask (`topmag_keep_mask`) is the single source of truth.
It MUST be computed once from the unmodified latent and passed explicitly to
both the dense-zero baseline and the Triton packer — re-running `topk` (e.g.
after pruning, or on inputs with natural zeros / cutoff ties) does not
reproduce the same index set.
"""
import torch

from . import config


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
    _, idx = mag.topk(prune_k, dim=-1, largest=False)        # [n, prune_k]
    n = latent.shape[0]
    rows = torch.arange(n, device=latent.device)[:, None].expand(n, prune_k)
    mask[rows.reshape(-1), idx.reshape(-1)] = False
    return mask


def topmag_zero_from_mask(latent: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Zero every coord where keep_mask is False, in place.

    Zeros survive any downstream per-row RMSNorm scaling (a scalar multiply per
    row) and the native fp8 store encodes them as exactly 0. The mask must come
    from `topmag_keep_mask` on the unmodified latent.
    """
    latent.masked_fill_(~keep_mask, 0.0)
    return latent


def topmag_zero(latent: torch.Tensor, keep: float) -> torch.Tensor:
    """Backward-compatible dense-zero TopMag: mask once, then zero (same result
    as the original scatter-based implementation). keep=1.0 is a no-op.
    """
    keep_mask = topmag_keep_mask(latent, keep)
    return topmag_zero_from_mask(latent, keep_mask)
