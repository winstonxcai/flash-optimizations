"""TopMag pruning on the native c4-latent store (Mustafar).

The `_sg_lr` module surface the injected sglang hook calls. config exposes the
geometry constants; the heavy runtime symbols are lazy-loaded from ops so the
package imports without touching torch.
"""
from .config import (
    HEAD_DIM,
    PACKED_C4_BYTES,
    packed_c4_enabled,
    stage2a_enabled,
    topmag_enabled,
    topmag_keep,
    validate_packed_static_config,
)

_OPS_SYMBOLS = {
    "patch",
    "unpatch",
    "verify",
    "topmag_keep_mask",
    "topmag_zero_from_mask",
    "pack_c4_rows",
    "unpack_gather_c4_native",
    "unpack_gather_c4_bf16",
    "NativeC4Workspace",
    "unpack_gather_c4_native_stage2a",
    "stage2a_available",
}


def __getattr__(name):
    if name in _OPS_SYMBOLS:
        if name in {"topmag_keep_mask", "topmag_zero_from_mask"}:
            from . import reference
            return getattr(reference, name)
        if name in {
            "pack_c4_rows",
            "unpack_gather_c4_native",
            "unpack_gather_c4_bf16",
            "NativeC4Workspace",
            "unpack_gather_c4_native_stage2a",
        }:
            from . import packed_c4
            return getattr(packed_c4, name)
        if name == "stage2a_available":
            from . import stage2a
            return stage2a.available
        from . import ops
        return getattr(ops, name)
    raise AttributeError(name)


__all__ = [
    "HEAD_DIM", "PACKED_C4_BYTES", "topmag_enabled", "topmag_keep",
    "validate_packed_static_config",
    "packed_c4_enabled", "stage2a_enabled", "topmag_keep_mask", "topmag_zero_from_mask",
    "pack_c4_rows", "unpack_gather_c4_native", "unpack_gather_c4_bf16",
    "NativeC4Workspace", "unpack_gather_c4_native_stage2a", "stage2a_available",
    "patch", "unpatch", "verify",
]
