"""TopMag pruning on the native compressed-latent store (Mustafar).

This is the `_sg_lr` module surface used by the injected SGLang hooks.
Geometry lives in config, runtime wrappers in packed, reference math in
reference, and patch commands in patching. Heavy imports remain lazy.
"""

from importlib import import_module

from .config import (
    HEAD_DIM,
    PACKED_RECORD_BYTES,
    fused_enabled,
    packed_enabled,
    topmag_enabled,
    topmag_keep,
    validate_packed_static_config,
)

_LAZY_EXPORTS = {
    "patch": "patching",
    "unpatch": "patching",
    "verify": "patching",
    "topmag_keep_mask": "reference",
    "topmag_zero_from_mask": "reference",
    "pack_rows": "packed",
    "unpack_gather_native": "packed",
    "unpack_gather_bf16": "packed",
    "NativeWorkspace": "packed",
    "unpack_gather_native_fused": "packed",
    "fused_available": "fused",
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        return getattr(import_module(f".{_LAZY_EXPORTS[name]}", __name__), name)
    raise AttributeError(name)


__all__ = [
    "HEAD_DIM",
    "PACKED_RECORD_BYTES",
    "fused_enabled",
    "packed_enabled",
    "topmag_enabled",
    "topmag_keep",
    "validate_packed_static_config",
    *_LAZY_EXPORTS,
]
