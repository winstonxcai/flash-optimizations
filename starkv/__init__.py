"""STAR-CSA: adaptive low-rank compression of the V4-Flash CSA cache entry.

The 512-dim C4 latent is stored as ``[z in R^r, c_R in R^64]``: the 448 NoPE
dims go through a per-layer basis ``D_l``, the 64 RoPE dims stay exact. The CSA
compressor, the Lightning Indexer and every non-c4 pool are untouched.
"""

from .config import BYTES_PER_TOKEN, HEAD_DIM, NOPE_DIM, RANK, ROPE_DIM, starkv_enabled

__all__ = [
    "BYTES_PER_TOKEN",
    "HEAD_DIM",
    "NOPE_DIM",
    "RANK",
    "ROPE_DIM",
    "starkv_enabled",
    "capture_decode_attn",
    "capture_enabled",
    "decode_lowrank",
    "dequantize_lowrank_k_cache_paged",
    "patch",
    "set_basis_dir",
    "set_cur_layer",
    "store_lowrank",
    "unpatch",
    "verify",
]

_LAZY = {
    "capture_decode_attn",
    "capture_enabled",
    "decode_lowrank",
    "dequantize_lowrank_k_cache_paged",
    "patch",
    "set_basis_dir",
    "set_cur_layer",
    "store_lowrank",
    "unpatch",
    "verify",
}


def __getattr__(name):
    if name in _LAZY:
        from . import ops

        return getattr(ops, name)
    raise AttributeError(name)
