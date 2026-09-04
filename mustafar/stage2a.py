"""Python binding for the experimental Stage-2A CUDA adapter."""

from __future__ import annotations

import importlib
import threading

import torch

_extension = None
_load_error: BaseException | None = None
_marker_lock = threading.Lock()
_marker_emitted = False
_validated_devices: set[int] = set()


def _load():
    global _extension, _load_error
    if _extension is not None:
        return _extension
    if _load_error is not None:
        raise RuntimeError(
            "Stage 2A was requested, but the CUDA extension is unavailable"
        ) from _load_error
    try:
        _extension = importlib.import_module("mustafar._stage2a_cuda")
    except BaseException as exc:
        _load_error = exc
        raise RuntimeError(
            "Stage 2A was requested, but mustafar._stage2a_cuda could not be loaded"
        ) from exc
    return _extension


def available() -> bool:
    try:
        _load()
    except RuntimeError:
        return False
    return True


def _emit_dispatch_marker() -> None:
    global _marker_emitted
    if _marker_emitted:
        return
    with _marker_lock:
        if not _marker_emitted:
            print("MUSTAFAR_STAGE2A_DISPATCH=packed_c4_to_native", flush=True)
            _marker_emitted = True


def packed_c4_to_native(
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
) -> None:
    """Mutate ``native_out`` on PyTorch's current stream without allocations."""
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2A requires CUDA")
    device_index = values.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if device_index not in _validated_devices:
        major, minor = torch.cuda.get_device_capability(device_index)
        if (major, minor) < (8, 0):
            raise RuntimeError(
                f"Stage 2A requires CUDA capability >= 8.0, got {major}.{minor}"
            )
        _validated_devices.add(device_index)
    _emit_dispatch_marker()
    _load().packed_c4_to_native(
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
