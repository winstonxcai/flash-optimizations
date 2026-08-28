"""TopMag pruning on the native c4-latent store (Mustafar).

The `_sg_lr` module surface the injected sglang hook calls. config exposes the
geometry constants; the heavy runtime symbols are lazy-loaded from ops so the
package imports without touching torch.
"""
from .config import HEAD_DIM, topmag_enabled, topmag_keep

_OPS_SYMBOLS = {"maybe_prune", "patch", "unpatch", "verify"}


def __getattr__(name):
    if name in _OPS_SYMBOLS:
        from . import ops
        return getattr(ops, name)
    raise AttributeError(name)


__all__ = [
    "HEAD_DIM", "topmag_enabled", "topmag_keep",
    "maybe_prune", "patch", "unpatch", "verify",
]
