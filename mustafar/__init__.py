"""TopMag pruning on the native c4-latent store (Mustafar).

The `_sg_lr` module surface the injected sglang hook calls. config exposes the
geometry constants; the heavy runtime symbols are lazy-loaded from ops so the
package imports without touching torch.
"""
from .config import (
    HEAD_DIM,
    PACKED_BYTES_PER_ROW,
    packed_enabled,
    topmag_enabled,
    topmag_keep,
)

_OPS_SYMBOLS = {
    "decode_packed",
    "get_packed_pool",
    "make_keep_mask",
    "maybe_prune",
    "packed_bytes_per_row",
    "register_packed_pool",
    "set_layer_context",
    "store_packed_c4",
    "unpack_packed_c4",
    "patch",
    "unpatch",
    "verify",
}


def __getattr__(name):
    if name in _OPS_SYMBOLS:
        from . import ops
        return getattr(ops, name)
    raise AttributeError(name)


__all__ = [
    "HEAD_DIM", "PACKED_BYTES_PER_ROW", "packed_enabled", "topmag_enabled",
    "topmag_keep", "decode_packed", "get_packed_pool", "make_keep_mask",
    "maybe_prune", "packed_bytes_per_row", "register_packed_pool",
    "set_layer_context", "store_packed_c4", "unpack_packed_c4", "patch",
    "unpatch", "verify",
]
