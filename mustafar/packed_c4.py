"""Persistent 328-byte TopMag50 C4 layout and production wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from . import config
from .bitmap import bitmap_to_bits, mask_to_bitmap


class PackedC4Buffers(NamedTuple):
    values: torch.Tensor
    bitmaps: torch.Tensor
    scales: torch.Tensor


@dataclass
class NativeC4Workspace:
    """Reusable buffers used to preserve the existing FlashMLA consumer."""

    raw: torch.Tensor
    dense: torch.Tensor
    temporary_indices: torch.Tensor
    page_size: int
    bytes_per_page: int

    @classmethod
    def allocate(
        cls,
        max_batch: int,
        selected_k: int,
        page_size: int,
        device: torch.device | str,
    ) -> "NativeC4Workspace":
        rows = max_batch * selected_k
        pages = (rows + page_size - 1) // page_size
        bytes_per_page = (
            (config.NATIVE_C4_BYTES * page_size + 575) // 576
        ) * 576
        raw = torch.zeros(
            pages, bytes_per_page, dtype=torch.uint8, device=device
        )
        dense = torch.empty(
            rows, config.HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        temporary_indices = torch.arange(
            rows, dtype=torch.int32, device=device
        ).reshape(max_batch, selected_k)
        return cls(raw, dense, temporary_indices, page_size, bytes_per_page)


def _as_buffers(packed_buffers, layer_id: int | None = None) -> PackedC4Buffers:
    if isinstance(packed_buffers, PackedC4Buffers):
        return packed_buffers
    if hasattr(packed_buffers, "get_packed_buffers"):
        if layer_id is None:
            raise ValueError("layer_id is required for a packed C4 pool")
        return PackedC4Buffers(*packed_buffers.get_packed_buffers(layer_id))
    return PackedC4Buffers(*packed_buffers)


def _plan_rows(compressor_plan: object) -> torch.Tensor:
    return compressor_plan[1].view(torch.int32)


def _ue8m0_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for SGLang's ceil-to-UE8M0 FP8 quantization."""
    if x.shape[0] == 0:
        return (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty(
                (0, config.BITMAP_WORDS), dtype=torch.uint8, device=x.device
            ),
        )
    tiles = x.float().reshape(-1, config.BITMAP_WORDS, config.TILE_SIZE)
    abs_max = tiles.abs().amax(dim=-1)
    raw_scale = abs_max.clamp_min(1.0e-4) / config.FP8_E4M3_MAX
    exponent = torch.ceil(torch.log2(raw_scale)).to(torch.int32)
    scale_codes = (exponent + 127).to(torch.uint8)
    scales = torch.exp2(exponent.float())
    quantized = torch.clamp(
        tiles / scales.unsqueeze(-1),
        -config.FP8_E4M3_MAX,
        config.FP8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    return quantized.reshape(-1, config.HEAD_DIM), scale_codes


def pack_c4_rows_ref(
    latent: torch.Tensor,
    keep_mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Torch row-format reference, independent of page locations."""
    if keep_mask.dtype != torch.bool or keep_mask.shape != latent.shape:
        raise ValueError("keep_mask must be bool and match latent")
    if latent.shape[-1] != config.HEAD_DIM:
        raise ValueError("packed C4 rows must have 512 coordinates")
    if latent.shape[0] and not bool(
        (keep_mask.sum(-1) == config.PACKED_KEEP).all()
    ):
        raise ValueError("every packed C4 row must retain exactly 256 coordinates")
    masked = latent.float().masked_fill(~keep_mask, 0.0)
    inv_rms = torch.rsqrt(masked.square().mean(-1, keepdim=True) + norm_eps)
    normalized = (masked * inv_rms * norm_weight.float()).to(torch.bfloat16)
    quantized, scales = _ue8m0_quantize(normalized)
    columns = torch.nonzero(keep_mask, as_tuple=False)[:, 1].reshape(
        latent.shape[0], config.PACKED_KEEP
    )
    codes = quantized.view(torch.uint8).gather(1, columns)
    bitmaps = mask_to_bitmap(keep_mask).view(torch.uint64)
    return codes, bitmaps, scales


def pack_c4_rows(
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
    """Pack valid C4 compressor rows directly into persistent paged storage."""
    config.validate_packed_static_config()
    if latent.shape[-1] != config.HEAD_DIM:
        raise ValueError(f"packed C4 expects dim 512, got {latent.shape}")
    if keep_mask.shape != latent.shape or keep_mask.dtype != torch.bool:
        raise ValueError("keep_mask must be bool and match latent")
    buffers = _as_buffers(packed_buffers, layer_id)
    if latent.numel() == 0:
        return
    from .triton import _pack_c4_fp8_kernel

    plan_rows = _plan_rows(compressor_plan)
    _pack_c4_fp8_kernel[(latent.shape[0],)](
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
        KEEP_K=config.PACKED_KEEP,
        TILE_SIZE=config.TILE_SIZE,
        BITMAP_WORDS=config.BITMAP_WORDS,
        FP8_MAX=config.FP8_E4M3_MAX,
        IS_DECODE=bool(compressor_plan.is_decode),
        BLOCK_D=config.HEAD_DIM,
        num_warps=8,
    )


def _freq_parts(freqs_cis) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(freqs_cis, tuple):
        return freqs_cis
    return freqs_cis.real.contiguous(), freqs_cis.imag.contiguous()


def unpack_gather_c4_bf16(
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
    from .triton import _rope_tail_inplace_kernel, _unpack_gather_c4_bf16_kernel

    _unpack_gather_c4_bf16_kernel[(flat_rows,)](
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
        KEEP_K=config.PACKED_KEEP,
        BITMAP_WORDS=config.BITMAP_WORDS,
        BLOCK_D=config.HEAD_DIM,
        num_warps=8,
    )
    cos, sin = _freq_parts(freqs_cis)
    _rope_tail_inplace_kernel[(flat_rows,)](
        out2d,
        raw_indices,
        cos,
        sin,
        flat_rows,
        HEAD_DIM=config.HEAD_DIM,
        NOPE_DIM=config.NOPE_DIM,
        ROPE_PAIRS=config.ROPE_DIM // 2,
        num_warps=1,
    )
    return output


def unpack_gather_c4_native(
    packed_buffers,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    freqs_cis,
    native_workspace: NativeC4Workspace,
    temporary_indices: torch.Tensor | None = None,
    *,
    layer_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize native hybrid pages and return cache plus remapped indices."""
    buffers = _as_buffers(packed_buffers, layer_id)
    n_queries, selected_k = physical_indices.shape
    rows = n_queries * selected_k
    temp = (
        native_workspace.temporary_indices[:n_queries, :selected_k]
        if temporary_indices is None
        else temporary_indices
    )
    dense = native_workspace.dense[:rows]
    unpack_gather_c4_bf16(
        buffers,
        physical_indices,
        raw_indices,
        topk_lengths,
        freqs_cis,
        dense,
    )
    if rows:
        from .triton import _bf16_to_native_c4_kernel

        _bf16_to_native_c4_kernel[(rows,)](
            dense,
            buffers.bitmaps,
            buffers.values,
            buffers.scales,
            physical_indices,
            native_workspace.raw,
            temp,
            rows,
            page_size=native_workspace.page_size,
            bytes_per_page=native_workspace.bytes_per_page,
            HEAD_DIM=config.HEAD_DIM,
            NOPE_DIM=config.NOPE_DIM,
            KEEP_K=config.PACKED_KEEP,
            BITMAP_WORDS=config.BITMAP_WORDS,
            BLOCK_D=config.HEAD_DIM,
            num_warps=8,
        )
    # The existing FlashMLA consumer reads only the prefix described by
    # ``topk_lengths``.  Returning the preallocated dense range directly avoids
    # torch.arange/where allocations in decode and keeps graph replay static.
    # Invalid padding slots are reconstructed as zeros but are outside that
    # prefix and therefore never consumed.
    return native_workspace.raw, temp


def unpack_c4_rows_ref(
    values: torch.Tensor,
    bitmaps: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Reference pre-RoPE BF16 reconstruction for unit tests."""
    bits = bitmap_to_bits(bitmaps)
    rank = bits.cumsum(-1) - 1
    codes = values.gather(1, rank.clamp_min(0))
    q = codes.view(torch.float8_e4m3fn).float()
    scale = torch.exp2(scales.to(torch.int32).float() - 127)
    scale = scale.repeat_interleave(config.TILE_SIZE, dim=-1)
    return torch.where(bits, q * scale, torch.zeros_like(q)).to(torch.bfloat16)


def packed_storage_report(buffers: PackedC4Buffers, occupied_rows: int) -> dict:
    storage_bytes = sum(t.untyped_storage().nbytes() for t in buffers)
    return {
        "logical_bytes_per_row": config.PACKED_C4_BYTES,
        "occupied_rows": int(occupied_rows),
        "occupied_bytes": int(occupied_rows) * config.PACKED_C4_BYTES,
        "storage_bytes": int(storage_bytes),
        "logical_compression": config.NATIVE_C4_BYTES / config.PACKED_C4_BYTES,
    }


def project_request_storage(
    seq_len: int = 128 * 1024,
    c4_layers: int = 21,
    c4_page_size: int = 64,
) -> dict[str, int | float]:
    """Logical and page-rounded C4 bytes for one request."""
    rows_per_layer = (seq_len + 3) // 4
    pages_per_layer = (rows_per_layer + c4_page_size - 1) // c4_page_size
    native_page_bytes = (
        (config.NATIVE_C4_BYTES * c4_page_size + 575) // 576
    ) * 576
    packed_page_bytes = config.PACKED_C4_BYTES * c4_page_size
    logical_native = rows_per_layer * c4_layers * config.NATIVE_C4_BYTES
    logical_packed = rows_per_layer * c4_layers * config.PACKED_C4_BYTES
    allocated_native = pages_per_layer * c4_layers * native_page_bytes
    allocated_packed = pages_per_layer * c4_layers * packed_page_bytes
    return {
        "seq_len": seq_len,
        "c4_layers": c4_layers,
        "rows_per_layer": rows_per_layer,
        "pages_per_layer": pages_per_layer,
        "logical_native_bytes": logical_native,
        "logical_packed_bytes": logical_packed,
        "allocated_native_bytes": allocated_native,
        "allocated_packed_bytes": allocated_packed,
        "native_page_padding_bytes": allocated_native - logical_native,
        "packed_page_padding_bytes": allocated_packed - logical_packed,
        "logical_compression": logical_native / logical_packed,
        "allocated_compression": allocated_native / allocated_packed,
    }
