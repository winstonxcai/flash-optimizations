"""Triton pack/unpack for physically-sparse TopMag on the native c4 latent.

Stage 0 (compression-only, no pool/decode change):

    pack_ccomp(latent, keep_mask, keep_k)  ->  (packed [n, KEEP_K], bitmap [n, 8])
    unpack_ccomp(packed, bitmap)           ->  dense [n, HEAD_DIM]

with the correctness invariant

    unpack(pack(latent, keep_mask, keep_k)) == dense-zero TopMag(latent)

where the dense-zero baseline is `latent.masked_fill(~keep_mask, 0)` and the
packed values keep `latent.dtype` (no FP8 — otherwise TopMag error and a new
quantization error would be confounded). The keep-mask MUST be
`reference.topmag_keep_mask(latent, keep)`, computed once from the unmodified
latent and shared by the dense-zero baseline and the packer; never recompute
`topk` (tie-breaking is arbitrary-but-deterministic and not reproducible after
pruning or on natural-zero inputs).

Contract: every row of `keep_mask` has exactly `keep_k` True entries. The
caller owns this; the unit tests assert it. `KEEP_K` is an explicit host scalar
so output allocation and the `KEEP_K: tl.constexpr` launch never require
reading a GPU popcount back to the host.

Stage 0.5 slots FP8 quantization in as a `QUANTIZE: tl.constexpr` flag inside
the kernels without changing this explicit-mask API.
"""
import torch

from . import config
from .bitmap import bitmap_to_bits as _bitmap_to_bits
from .bitmap import mask_to_bitmap as _mask_to_bitmap
from .reference import topmag_keep_mask
from .triton import _pack_ccomp_kernel, _unpack_ccomp_kernel

__all__ = [
    "topmag_keep_mask", "pack_ccomp", "unpack_ccomp",
    "_keep_count", "_prune_count", "_mask_to_bitmap", "_bitmap_to_bits",
    "pack_ccomp_ref", "unpack_ccomp_ref",
]


# --- keep geometry -----------------------------------------------------------
def _keep_count(keep: float) -> int:
    """Coords retained per row: round(HEAD_DIM * keep) (e.g. 0.375 -> 192)."""
    return int(round(config.HEAD_DIM * keep))


def _prune_count(keep: float) -> int:
    """Coords dropped per row: HEAD_DIM - _keep_count(keep) (e.g. 0.375 -> 320)."""
    return config.HEAD_DIM - _keep_count(keep)


# --- GPU pack / unpack --------------------------------------------------------
def pack_ccomp(latent: torch.Tensor, keep_mask: torch.Tensor,
               keep_k: int) -> tuple:
    """Pack a pruned latent row into its kept coords + bitmap.

    latent:    [n, HEAD_DIM] pre-transform c4 latent, kv_compressed.dtype.
    keep_mask: bool [n, HEAD_DIM] from reference.topmag_keep_mask (unchanged
               latent); every row has popcount keep_k.
    keep_k:    retained coords per row (must equal keep_mask.sum(-1)).

    Returns (packed [n, keep_k] in latent.dtype, bitmap [n, 8] int64).
    """
    n = latent.shape[0]
    if n == 0:
        return (torch.empty((0, keep_k), dtype=latent.dtype, device=latent.device),
                torch.empty((0, config.BITMAP_WORDS), dtype=torch.int64,
                            device=latent.device))
    packed = torch.empty((n, keep_k), dtype=latent.dtype, device=latent.device)
    bitmap = _mask_to_bitmap(keep_mask)
    mask_i8 = keep_mask.to(torch.int8)
    _pack_ccomp_kernel[(n,)](latent, mask_i8, packed, n,
                             HEAD_DIM=config.HEAD_DIM, KEEP_K=keep_k,
                             BLOCK_D=config.HEAD_DIM, num_warps=4)
    return packed, bitmap


def unpack_ccomp(packed: torch.Tensor, bitmap: torch.Tensor,
                 n_rows: int | None = None) -> torch.Tensor:
    """Exact inverse of pack_ccomp -> dense [n_rows, HEAD_DIM].

    Pruned coords land as exactly 0 (== dense-zero TopMag). n_rows defaults to
    packed.shape[0]; pass it when the caller wants a different leading dim.
    """
    n = packed.shape[0] if n_rows is None else n_rows
    keep_k = packed.shape[1]
    out = torch.empty((n, config.HEAD_DIM), dtype=packed.dtype, device=packed.device)
    if n == 0:
        return out
    _unpack_ccomp_kernel[(n,)](packed, bitmap, out, n,
                               HEAD_DIM=config.HEAD_DIM, KEEP_K=keep_k,
                               BITMAP_WORDS=config.BITMAP_WORDS,
                               BLOCK_D=config.HEAD_DIM, num_warps=4)
    return out


# --- pure-torch references (cross-check) -------------------------------------
def pack_ccomp_ref(latent: torch.Tensor, keep_mask: torch.Tensor,
                   keep_k: int) -> tuple:
    """Torch pack: gather kept columns in ascending order (== Triton cumsum order)."""
    cols = torch.nonzero(keep_mask, as_tuple=False)[:, 1].reshape(
        latent.shape[0], keep_k)
    return latent.gather(1, cols), _mask_to_bitmap(keep_mask)


def unpack_ccomp_ref(packed: torch.Tensor, bitmap: torch.Tensor) -> torch.Tensor:
    """Torch unpack: rank = cumsum(bits) - 1, gather, zero the pruned coords."""
    bits = _bitmap_to_bits(bitmap)                 # [n, HEAD_DIM]
    rank = bits.cumsum(1) - 1
    vals = packed.gather(1, rank.clamp(min=0))     # masked-off lanes may be < 0
    return torch.where(bits, vals, torch.zeros_like(vals))
