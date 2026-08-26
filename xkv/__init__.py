"""Minimal W3 rank-192 CSA low-rank KV store."""
from .config import BYTES_PER_TOKEN, COEFF_DIM, HEAD_DIM, lowrank_enabled


def __getattr__(name):
    if name in {
        "decode_lowrank", "dequantize_lowrank_k_cache_paged", "patch",
        "set_basis_dir", "set_cur_layer", "store_compressed_lowrank",
        "unpatch", "verify",
    }:
        from . import ops
        return getattr(ops, name)
    raise AttributeError(name)

__all__ = [
    "BYTES_PER_TOKEN", "COEFF_DIM", "HEAD_DIM", "lowrank_enabled",
    "decode_lowrank", "dequantize_lowrank_k_cache_paged", "patch",
    "set_basis_dir", "set_cur_layer", "store_compressed_lowrank",
    "unpatch", "verify",
]
