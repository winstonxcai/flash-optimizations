"""Pure torch TopMag pruning math (native c4-latent scope).

No runtime state here — just the per-row magnitude prune. The hook lives in
ops.py; the patch machinery is in ops.py too.
"""
import torch

from . import config


def topmag_zero(latent: torch.Tensor, keep: float) -> torch.Tensor:
    """Zero the smallest (1-keep) fraction of coords per row, in place.

    latent: [n, HEAD_DIM] float (bf16 / fp16 / fp32 — the c4 compressor output).
    The smallest-|·| k coords per row are set to 0. Zeros survive any downstream
    per-row RMSNorm scaling (a scalar multiply per row) and the native fp8 store
    encodes them as exactly 0. keep=1.0 is a no-op.
    """
    if keep >= 1.0:
        return latent
    k = config.HEAD_DIM - int(round(config.HEAD_DIM * keep))
    if k <= 0:
        return latent
    n = latent.shape[0]
    mag = latent.abs().float()
    _, idx = mag.topk(k, dim=-1, largest=False)           # [n, k] smallest-k cols
    rows = torch.arange(n, device=latent.device)[:, None].expand(n, k)
    latent[rows.reshape(-1), idx.reshape(-1)] = 0.0
    return latent
