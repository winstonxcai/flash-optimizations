"""SM80-compatible fixed-stride C4 pack and unpack/gather kernels."""

from __future__ import annotations

from functools import lru_cache

import torch

from .. import config, reference

try:
    import triton
    import triton.language as tl
except ImportError:  # Local CPU development does not require Triton.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None


@lru_cache(maxsize=None)
def _bitmap_weights(device: str) -> torch.Tensor:
    return torch.tensor(
        [-(1 << 63)] + [1 << shift for shift in range(62, -1, -1)],
        dtype=torch.int64,
        device=torch.device(device),
    )


@lru_cache(maxsize=16)
def _freqs_real(data_ptr: int, device: str, rows: int, freqs_cis: torch.Tensor):
    del data_ptr, device, rows
    if freqs_cis.is_complex():
        return torch.view_as_real(freqs_cis).reshape(-1, config.ROPE_DIM).float().contiguous()
    return freqs_cis.reshape(-1, config.ROPE_DIM).float().contiguous()


if triton is not None:

    @triton.jit
    def _round_to_nearest_even(x):
        base = tl.floor(x)
        fraction = x - base
        tie_increment = (base.to(tl.int32) & 1).to(tl.float32)
        return tl.where(
            fraction > 0.5,
            base + 1.0,
            tl.where(fraction < 0.5, base, base + tie_increment),
        )

    @triton.jit
    def _encode_e4m3fn(x):
        """Software E4M3FN encoder for SM80 (raw uint8 result)."""
        sign = x < 0.0
        magnitude = tl.abs(x)
        is_subnormal = magnitude < 0.015625  # 2^-6
        sub_mantissa = _round_to_nearest_even(magnitude * 512.0).to(tl.int32)
        sub_carry = sub_mantissa == 8
        sub_exponent = tl.where(sub_carry, 1, 0)
        sub_mantissa = tl.where(sub_carry, 0, sub_mantissa)

        safe = tl.maximum(magnitude, 0.015625)
        unbiased = tl.floor(tl.log2(safe))
        exponent = unbiased.to(tl.int32) + 7
        mantissa_exact = (safe * tl.exp2(-unbiased) - 1.0) * 8.0
        mantissa = _round_to_nearest_even(mantissa_exact).to(tl.int32)
        carry = mantissa == 8
        exponent += carry.to(tl.int32)
        mantissa = tl.where(carry, 0, mantissa)

        exponent = tl.where(is_subnormal, sub_exponent, exponent)
        mantissa = tl.where(is_subnormal, sub_mantissa, mantissa)
        exponent = tl.minimum(tl.maximum(exponent, 0), 15)
        max_mantissa = tl.where(exponent == 15, 6, 7)
        mantissa = tl.minimum(tl.maximum(mantissa, 0), max_mantissa)
        bits = (sign.to(tl.int32) << 7) | (exponent << 3) | mantissa
        return tl.where(magnitude == 0.0, 0, bits).to(tl.uint8)

    @triton.jit
    def _decode_e4m3fn(bits):
        """Software E4M3FN decoder for SM80."""
        raw = bits.to(tl.int32)
        sign = (raw >> 7) & 1
        exponent = (raw >> 3) & 0xF
        mantissa = raw & 0x7
        subnormal = mantissa.to(tl.float32) * 0.001953125  # 2^-9
        normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(
            exponent.to(tl.float32) - 7.0
        )
        value = tl.where(exponent == 0, subnormal, normal)
        return tl.where(sign == 0, value, -value)

    @triton.jit
    def _pack_kernel(
        latent,
        keep_mask,
        norm_weight,
        locations,
        packed_values,
        bitmaps,
        packed_scales,
        bitmap_weights,
        n_rows,
        norm_eps: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KEEP_DIM: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        NUM_TILES: tl.constexpr,
        FP8_MAX: tl.constexpr,
    ):
        row = tl.program_id(0)
        d = tl.arange(0, HEAD_DIM)
        row_ok = row < n_rows
        x = tl.load(latent + row * HEAD_DIM + d, mask=row_ok, other=0.0).to(tl.float32)
        keep = tl.load(keep_mask + row * HEAD_DIM + d, mask=row_ok, other=0).to(tl.int1)
        x = tl.where(keep, x, 0.0)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + norm_eps)
        weight = tl.load(norm_weight + d).to(tl.float32)
        normed = x * inv_rms * weight
        location = tl.load(locations + row, mask=row_ok, other=-1).to(tl.int64)
        valid_row = row_ok & (location >= 0)

        rank = tl.cumsum(keep.to(tl.int32), axis=0) - 1
        for tile in tl.static_range(NUM_TILES):
            in_tile = (d >= tile * TILE_SIZE) & (d < (tile + 1) * TILE_SIZE)
            maxabs = tl.max(tl.where(in_tile, tl.abs(normed), 0.0), axis=0)
            maxabs = tl.maximum(maxabs, 1.0e-8)
            exponent = tl.ceil(tl.log2(maxabs / FP8_MAX))
            exponent = tl.minimum(tl.maximum(exponent, -127.0), 128.0)
            scale = tl.exp2(exponent)
            quant = tl.minimum(tl.maximum(normed / scale, -FP8_MAX), FP8_MAX)
            tl.store(
                packed_values + location * KEEP_DIM + rank,
                _encode_e4m3fn(quant),
                mask=valid_row & keep & in_tile,
            )
            tl.store(
                packed_scales + location * NUM_TILES + tile,
                (exponent + 127.0).to(tl.uint8),
                mask=valid_row,
            )

            lane = d - tile * TILE_SIZE
            weight_bits = tl.load(
                bitmap_weights + lane,
                mask=in_tile,
                other=0,
            )
            word = tl.sum(tl.where(in_tile & keep, weight_bits, 0), axis=0)
            tl.store(bitmaps + location * NUM_TILES + tile, word, mask=valid_row)

    @triton.jit
    def _unpack_kernel(
        packed_values,
        bitmaps,
        packed_scales,
        physical_indices,
        raw_indices,
        freqs,
        output,
        n_selected,
        max_position,
        HEAD_DIM: tl.constexpr,
        KEEP_DIM: tl.constexpr,
        TILE_SIZE: tl.constexpr,
        NUM_TILES: tl.constexpr,
        NOPE_DIM: tl.constexpr,
        C4_RATIO: tl.constexpr,
    ):
        row = tl.program_id(0)
        d = tl.arange(0, HEAD_DIM)
        row_ok = row < n_selected
        location = tl.load(physical_indices + row, mask=row_ok, other=-1).to(tl.int64)
        raw = tl.load(raw_indices + row, mask=row_ok, other=-1).to(tl.int64)
        valid = row_ok & (location >= 0) & (raw >= 0)
        tile = d // TILE_SIZE
        lane = d % TILE_SIZE
        word = tl.load(bitmaps + location * NUM_TILES + tile, mask=valid, other=0)
        keep = ((word >> (63 - lane)) & 1).to(tl.int1)
        rank = tl.cumsum(keep.to(tl.int32), axis=0) - 1
        code_bits = tl.load(
            packed_values + location * KEEP_DIM + rank,
            mask=valid & keep,
            other=0,
        )
        code = _decode_e4m3fn(code_bits)
        scale_u8 = tl.load(
            packed_scales + location * NUM_TILES + tile,
            mask=valid,
            other=127,
        ).to(tl.float32)
        dense = tl.where(keep, code * tl.exp2(scale_u8 - 127.0), 0.0)

        peer_d = tl.where((d & 1) == 0, d + 1, d - 1)
        peer = tl.gather(dense, peer_d, axis=0)
        pair = (d - NOPE_DIM) // 2
        position = tl.minimum(raw * C4_RATIO, max_position - 1)
        cos = tl.load(freqs + position * 64 + 2 * pair, mask=valid & (d >= NOPE_DIM), other=1.0)
        sin = tl.load(freqs + position * 64 + 2 * pair + 1, mask=valid & (d >= NOPE_DIM), other=0.0)
        rotated = tl.where(
            (d & 1) == 0,
            dense * cos - peer * sin,
            peer * sin + dense * cos,
        )
        result = tl.where(d >= NOPE_DIM, rotated, dense)
        tl.store(output + row * HEAD_DIM + d, result, mask=row_ok)


def pack_c4_rows(
    latent: torch.Tensor,
    keep_mask: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    locations: torch.Tensor,
    packed_pool,
) -> None:
    """Pack rows, using Triton on CUDA and the exact Torch reference otherwise."""
    if not latent.is_cuda or triton is None:
        return reference.pack_c4_rows_reference(
            latent, keep_mask, norm_weight, norm_eps, locations, packed_pool
        )
    if latent.shape[-1] != config.HEAD_DIM or keep_mask.shape != latent.shape:
        raise ValueError("latent and keep_mask must both be [N, 512]")
    values = packed_pool.packed_values.reshape(-1)
    bitmaps = packed_pool.bitmap.reshape(-1)
    scales = packed_pool.packed_scales.reshape(-1)
    _pack_kernel[(latent.shape[0],)](
        latent.contiguous(),
        keep_mask.contiguous(),
        norm_weight.contiguous(),
        locations.contiguous(),
        values,
        bitmaps,
        scales,
        _bitmap_weights(str(latent.device)),
        latent.shape[0],
        norm_eps=norm_eps,
        HEAD_DIM=config.HEAD_DIM,
        KEEP_DIM=config.KEEP_DIM,
        TILE_SIZE=config.TILE_SIZE,
        NUM_TILES=config.SCALE_TILES,
        FP8_MAX=torch.finfo(reference.FP8_DTYPE).max,
        num_warps=8,
        num_stages=1,
    )


def unpack_gather_c4(
    packed_pool,
    physical_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    freqs_cis: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Densify selected rows into ``output`` and apply RoPE in the same launch."""
    if not physical_indices.is_cuda or triton is None:
        return reference.unpack_gather_c4_reference(
            packed_pool, physical_indices, raw_indices, freqs_cis, output
        )
    if physical_indices.shape != raw_indices.shape:
        raise ValueError("physical_indices and raw_indices must have identical shape")
    flat_physical = physical_indices.contiguous().reshape(-1)
    flat_raw = raw_indices.contiguous().reshape(-1)
    freqs = _freqs_real(
        freqs_cis.data_ptr(), str(freqs_cis.device), freqs_cis.shape[0], freqs_cis
    )
    _unpack_kernel[(flat_physical.numel(),)](
        packed_pool.packed_values.reshape(-1),
        packed_pool.bitmap.reshape(-1),
        packed_pool.packed_scales.reshape(-1),
        flat_physical,
        flat_raw,
        freqs,
        output.reshape(-1),
        flat_physical.numel(),
        freqs_cis.shape[0],
        HEAD_DIM=config.HEAD_DIM,
        KEEP_DIM=config.KEEP_DIM,
        TILE_SIZE=config.TILE_SIZE,
        NUM_TILES=config.SCALE_TILES,
        NOPE_DIM=config.NOPE_DIM,
        C4_RATIO=config.C4_RATIO,
        num_warps=8,
        num_stages=1,
    )
    return output
