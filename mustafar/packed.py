"""Persistent 328-byte TopMag50 layout and production wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from . import config


class PackedBuffers(NamedTuple):
    values: torch.Tensor
    bitmaps: torch.Tensor
    scales: torch.Tensor


@dataclass
class NativeWorkspace:
    """Reusable buffers used to preserve the existing FlashMLA consumer."""

    native_bytes: torch.Tensor
    dense_bf16: torch.Tensor | None
    temporary_indices: torch.Tensor
    page_size: int
    bytes_per_page: int

    @property
    def max_queries(self) -> int:
        """Maximum query rows this decode/small-extend workspace can hold."""
        return self.temporary_indices.shape[0]

    @property
    def selected_k(self) -> int:
        return self.temporary_indices.shape[1]

    @classmethod
    def allocate(
        cls,
        max_batch: int,
        selected_k: int,
        page_size: int,
        device: torch.device | str,
        *,
        with_dense: bool | None = None,
    ) -> "NativeWorkspace":
        rows = max_batch * selected_k
        pages = (rows + page_size - 1) // page_size
        bytes_per_page = ((config.NATIVE_RECORD_BYTES * page_size + 575) // 576) * 576
        raw = torch.zeros(pages, bytes_per_page, dtype=torch.uint8, device=device)
        if with_dense is None:
            with_dense = not config.fused_enabled()
        dense = (
            torch.empty(rows, config.HEAD_DIM, dtype=torch.bfloat16, device=device)
            if with_dense
            else None
        )
        temporary_indices = torch.arange(
            rows, dtype=torch.int32, device=device
        ).reshape(max_batch, selected_k)
        return cls(raw, dense, temporary_indices, page_size, bytes_per_page)


def _as_buffers(packed_buffers, layer_id: int | None = None) -> PackedBuffers:
    if isinstance(packed_buffers, PackedBuffers):
        return packed_buffers
    # The injected SGLang pool accessor deliberately exposes the stable
    # value/bitmap/scale ABI as a plain tuple. Normalize that form before
    # CUDA-graph capture reaches the unpack wrapper.
    if isinstance(packed_buffers, (tuple, list)):
        if len(packed_buffers) != 3 or not all(
            isinstance(buffer, torch.Tensor) for buffer in packed_buffers
        ):
            raise TypeError(
                "packed buffer sequences must contain value, bitmap, and scale tensors"
            )
        return PackedBuffers(*packed_buffers)
    if hasattr(packed_buffers, "get_packed_buffers"):
        if layer_id is None:
            raise ValueError("layer_id is required for a packed pool")
        return PackedBuffers(*packed_buffers.get_packed_buffers(layer_id))
    raise TypeError(
        "packed_buffers must be PackedBuffers, a three-tensor sequence, "
        "or a packed pool"
    )


def _plan_rows(compressor_plan: object) -> torch.Tensor:
    return compressor_plan[1].view(torch.int32)


def pack_rows(
    latent: torch.Tensor,
    keep_mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    compressor_plan: object,
    locations: torch.Tensor,
    packed_buffers,
    *,
    layer_id: int | None = None,
) -> None:
    """Pack valid compressor rows directly into persistent paged storage."""
    config.validate_packed_static_config()
    if latent.shape[-1] != config.HEAD_DIM:
        raise ValueError(f"packed expects dim 512, got {latent.shape}")
    if keep_mask.shape != latent.shape or keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must be bool and match latent")
    buffers = _as_buffers(packed_buffers, layer_id)
    if latent.numel() == 0:
        return
    from .triton import _pack_fp8_kernel

    plan_rows = _plan_rows(compressor_plan)
    _pack_fp8_kernel[(latent.shape[0],)](
        latent,
        keep_mask,
        norm_weight,
        plan_rows,
        locations,
        buffers.values,
        buffers.bitmaps,
        buffers.scales,
        latent.shape[0],
        norm_eps=norm_eps,
        HEAD_DIM=config.HEAD_DIM,
        KEEP_K=config.PACKED_KEPT_VALUES,
        TILE_SIZE=config.TILE_SIZE,
        BITMAP_WORDS=config.BITMAP_WORDS,
        FP8_MAX=config.FP8_E4M3_MAX,
        IS_DECODE=bool(compressor_plan.is_decode),
        BLOCK_D=config.HEAD_DIM,
        num_warps=8,
    )


def unpack_gather_bf16(
    packed_buffers,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    freqs_cis,
    output: torch.Tensor,
    *,
    layer_id: int | None = None,
) -> torch.Tensor:
    """Gather selected records, dequantize, zero-fill and apply tail RoPE."""
    buffers = _as_buffers(packed_buffers, layer_id)
    if physical_indices.shape != raw_indices.shape:
        raise ValueError("physical_indices and raw_indices must have equal shape")
    n_queries, selected_k = physical_indices.shape
    flat_rows = n_queries * selected_k
    out2d = output.reshape(-1, config.HEAD_DIM)
    if out2d.shape[0] < flat_rows:
        raise ValueError("output workspace is too small")
    if flat_rows == 0:
        return output
    from .triton import (
        _rope_tail_complex_inplace_kernel,
        _unpack_gather_bf16_kernel,
    )

    _unpack_gather_bf16_kernel[(flat_rows,)](
        buffers.values,
        buffers.bitmaps,
        buffers.scales,
        physical_indices,
        raw_indices,
        topk_lengths,
        out2d,
        n_queries,
        selected_k,
        HEAD_DIM=config.HEAD_DIM,
        KEEP_K=config.PACKED_KEPT_VALUES,
        BITMAP_WORDS=config.BITMAP_WORDS,
        BLOCK_D=config.HEAD_DIM,
        num_warps=8,
    )
    if not freqs_cis.is_complex() or not freqs_cis.is_contiguous():
        raise ValueError("freqs_cis must be a contiguous complex tensor")
    # This is a view only; unlike .real.contiguous()/.imag.contiguous(), it
    # performs no CUDA allocation and can be retained by the packed pool.
    freq_pairs = torch.view_as_real(freqs_cis)
    _rope_tail_complex_inplace_kernel[(flat_rows,)](
        out2d,
        raw_indices,
        freq_pairs,
        flat_rows,
        HEAD_DIM=config.HEAD_DIM,
        NOPE_DIM=config.NOPE_DIM,
        ROPE_PAIRS=config.ROPE_DIM // 2,
        num_warps=1,
    )
    return output


def unpack_gather_native(
    packed_buffers,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    freqs_cis,
    native_workspace: NativeWorkspace,
    *,
    layer_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize native hybrid pages and return cache plus remapped indices."""
    buffers = _as_buffers(packed_buffers, layer_id)
    n_queries, selected_k = physical_indices.shape
    if n_queries > native_workspace.max_queries:
        raise ValueError(
            "native workspace query capacity exceeded: "
            f"{n_queries} > {native_workspace.max_queries}; "
            "route this extend through sparse prefill"
        )
    if selected_k > native_workspace.selected_k:
        raise ValueError(
            "native workspace top-k capacity exceeded: "
            f"{selected_k} > {native_workspace.selected_k}"
        )
    rows = n_queries * selected_k
    temp = native_workspace.temporary_indices[:n_queries, :selected_k]
    if config.fused_enabled():
        config.validate_packed_static_config()
        unpack_gather_native_fused(
            buffers,
            physical_indices,
            raw_indices,
            topk_lengths,
            freqs_cis,
            native_workspace,
        )
        return native_workspace.native_bytes, temp
    if native_workspace.dense_bf16 is None:
        raise RuntimeError("Packed reconstruction requires a dense BF16 workspace")
    dense = native_workspace.dense_bf16[:rows]
    unpack_gather_bf16(
        buffers,
        physical_indices,
        raw_indices,
        topk_lengths,
        freqs_cis,
        dense,
    )
    if rows:
        from .triton import _bf16_to_native_kernel

        _bf16_to_native_kernel[(rows,)](
            dense,
            buffers.bitmaps,
            buffers.values,
            buffers.scales,
            physical_indices,
            native_workspace.native_bytes,
            temp,
            rows,
            page_size=native_workspace.page_size,
            bytes_per_page=native_workspace.bytes_per_page,
            HEAD_DIM=config.HEAD_DIM,
            NOPE_DIM=config.NOPE_DIM,
            KEEP_K=config.PACKED_KEPT_VALUES,
            BITMAP_WORDS=config.BITMAP_WORDS,
            BLOCK_D=config.HEAD_DIM,
            num_warps=8,
        )
    # The existing FlashMLA consumer reads only the prefix described by
    # ``topk_lengths``. Returning the preallocated dense range directly avoids
    # torch.arange/where allocations in decode and keeps graph replay static.
    # Invalid padding slots are reconstructed as zeros but are outside that
    # prefix and therefore never consumed.
    return native_workspace.native_bytes, temp


def unpack_gather_native_fused(
    packed_buffers,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    freqs_cis,
    native_workspace: NativeWorkspace,
    *,
    layer_id: int | None = None,
) -> None:
    """Run the allocation-free Fused packed-to-native CUDA adapter."""
    buffers = _as_buffers(packed_buffers, layer_id)
    if physical_indices.shape != raw_indices.shape:
        raise ValueError("physical_indices and raw_indices must have equal shape")
    n_queries, selected_k = physical_indices.shape
    if n_queries > native_workspace.max_queries:
        raise ValueError("native workspace query capacity exceeded")
    if selected_k > native_workspace.selected_k:
        raise ValueError("native workspace top-k capacity exceeded")
    if not freqs_cis.is_complex() or not freqs_cis.is_contiguous():
        raise ValueError("freqs_cis must be a contiguous complex tensor")
    from .fused import packed_to_native

    packed_to_native(
        buffers.values,
        buffers.bitmaps,
        buffers.scales,
        physical_indices,
        raw_indices,
        topk_lengths,
        torch.view_as_real(freqs_cis),
        native_workspace.native_bytes,
        native_workspace.page_size,
        native_workspace.bytes_per_page,
    )
