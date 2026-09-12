"""Python binding for the fused reconstruction extension."""

from __future__ import annotations

import importlib
import threading

import torch

_extension = None
_load_error: Exception | None = None
_marker_lock = threading.Lock()
_markers_emitted: set[str] = set()
_validated_devices: set[int] = set()


def _load():
    global _extension, _load_error
    if _extension is not None:
        return _extension
    if _load_error is not None:
        raise RuntimeError(
            "Fused was requested, but the CUDA extension is unavailable"
        ) from _load_error
    try:
        _extension = importlib.import_module("mustafar._fused")
    except Exception as exc:
        _load_error = exc
        raise RuntimeError(
            "Fused was requested, but mustafar._fused could not be loaded"
        ) from exc
    return _extension


def fused_available() -> bool:
    """Whether the extension imports; device support is checked at launch."""
    try:
        _load()
    except RuntimeError:
        return False
    return True


def optimized_fused_available() -> bool:
    """Whether the extension exports the optimized reconstruction entry point."""
    try:
        extension = _load()
    except RuntimeError:
        return False
    return callable(getattr(extension, "packed_to_native_optimized", None))


def early_rope_available() -> bool:
    """Whether the benchmark-only early-RoPE candidate is built."""
    try:
        extension = _load()
    except RuntimeError:
        return False
    return callable(getattr(extension, "packed_to_native_early_rope", None))


def _emit_dispatch_marker(optimized: bool, candidate: str | None = None) -> None:
    marker = candidate or ("packed_to_native_optimized" if optimized else "packed_to_native")
    if marker in _markers_emitted:
        return
    with _marker_lock:
        if marker not in _markers_emitted:
            print(f"MUSTAFAR_FUSED_DISPATCH={marker}", flush=True)
            _markers_emitted.add(marker)


def packed_to_native(
    values: torch.Tensor,
    bitmaps: torch.Tensor,
    scales: torch.Tensor,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    freq_pairs: torch.Tensor,
    native_out: torch.Tensor,
    page_size: int,
    bytes_per_page: int,
    *,
    optimized: bool = False,
    candidate: str | None = None,
) -> None:
    """Mutate ``native_out`` on PyTorch's current stream without allocations."""
    if not torch.cuda.is_available():
        raise RuntimeError("Fused requires CUDA")
    device_index = values.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if device_index not in _validated_devices:
        major, minor = torch.cuda.get_device_capability(device_index)
        if (major, minor) < (8, 0):
            raise RuntimeError(
                f"Fused requires CUDA capability >= 8.0, got {major}.{minor}"
            )
        _validated_devices.add(device_index)
    if candidate not in (None, "early_rope"):
        raise ValueError(f"unknown fused benchmark candidate: {candidate}")
    if candidate is not None and not optimized:
        raise ValueError("the early_rope candidate requires optimized=True")
    _emit_dispatch_marker(optimized, candidate)
    extension = _load()
    launcher = extension.packed_to_native
    if candidate == "early_rope":
        launcher = getattr(extension, "packed_to_native_early_rope", None)
        if launcher is None:
            raise RuntimeError("The early_rope fused candidate is unavailable")
    elif optimized:
        launcher = getattr(extension, "packed_to_native_optimized", None)
        if launcher is None:
            raise RuntimeError("Optimized fused reconstruction is unavailable")
    launcher(
        values,
        bitmaps,
        scales,
        physical_indices,
        raw_indices,
        topk_lengths,
        freq_pairs,
        native_out,
        page_size,
        bytes_per_page,
    )
