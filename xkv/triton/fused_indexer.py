"""Fused paged coefficient reconstruction consumed by sparse attention."""
import triton
import triton.language as tl

import torch

from .. import config
from ..reference import fp8_dtype, vrt_bf16_for


@triton.jit
def _reconstruct_kernel(buf_fp8, buf_u8, buf_i32, loc, vrt, freqs, out, n, max_pos,
                        PAGE_SIZE: tl.constexpr, PAGE_BYTES: tl.constexpr,
                        BLOCK_M: tl.constexpr, NUM_HALVES: tl.constexpr,
                        BLOCK_NOPE_H: tl.constexpr, NOPE_DIM: tl.constexpr,
                        ROPE_DIM: tl.constexpr, COEFF_DIM: tl.constexpr,
                        SCALE_TILES: tl.constexpr, TILE_SIZE: tl.constexpr,
                        META_BYTES: tl.constexpr, BYTES: tl.constexpr,
                        VRT_STRIDE: tl.constexpr, FREQ_STRIDE: tl.constexpr,
                        OUT_STRIDE: tl.constexpr):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = m < n
    l = tl.load(loc + m, mask=mm, other=0).to(tl.int64)
    base = (l // PAGE_SIZE) * PAGE_BYTES + (l % PAGE_SIZE) * BYTES
    p = tl.load(buf_i32 + (base + META_BYTES) // 4, mask=mm, other=0)
    p = tl.minimum(tl.maximum(p, 0), max_pos - 1)
    pair = tl.arange(0, ROPE_DIM // 2)
    co = tl.load(freqs + p[:, None] * FREQ_STRIDE + 2 * pair[None, :], mask=mm[:, None], other=1.)
    si = tl.load(freqs + p[:, None] * FREQ_STRIDE + 2 * pair[None, :] + 1, mask=mm[:, None], other=0.)
    k = tl.arange(0, TILE_SIZE)
    for h in tl.range(0, NUM_HALVES):
        c = h * BLOCK_NOPE_H + tl.arange(0, BLOCK_NOPE_H)
        acc = tl.zeros((BLOCK_M, BLOCK_NOPE_H), tl.float32)
        for t in tl.static_range(SCALE_TILES):
            a = tl.load(buf_fp8 + base[:, None] + t * TILE_SIZE + k[None, :], mask=mm[:, None], other=0.).to(tl.float32)
            s = tl.load(buf_u8 + base + COEFF_DIM + t, mask=mm, other=0).to(tl.float32)
            a = (a * tl.exp2(s[:, None] - 127.)).to(tl.bfloat16)
            rows = (t * TILE_SIZE + k)[:, None] * VRT_STRIDE
            acc += tl.dot(a, tl.load(vrt + rows + c[None, :]))
        tl.store(out + m[:, None] * OUT_STRIDE + c[None, :], acc.to(tl.bfloat16), mask=mm[:, None] & (c[None, :] < NOPE_DIM))
    ar = tl.zeros((BLOCK_M, ROPE_DIM // 2), tl.float32)
    ai = tl.zeros((BLOCK_M, ROPE_DIM // 2), tl.float32)
    for t in tl.static_range(SCALE_TILES):
        a = tl.load(buf_fp8 + base[:, None] + t * TILE_SIZE + k[None, :], mask=mm[:, None], other=0.).to(tl.float32)
        s = tl.load(buf_u8 + base + COEFF_DIM + t, mask=mm, other=0).to(tl.float32)
        a = (a * tl.exp2(s[:, None] - 127.)).to(tl.bfloat16)
        rows = (t * TILE_SIZE + k)[:, None] * VRT_STRIDE
        ar += tl.dot(a, tl.load(vrt + rows + (NOPE_DIM + 2 * pair)[None, :]))
        ai += tl.dot(a, tl.load(vrt + rows + (NOPE_DIM + 2 * pair + 1)[None, :]))
    nr, ni = ar * co - ai * si, ar * si + ai * co
    tl.store(out + m[:, None] * OUT_STRIDE + NOPE_DIM + 2 * pair[None, :], nr.to(tl.bfloat16), mask=mm[:, None])
    tl.store(out + m[:, None] * OUT_STRIDE + NOPE_DIM + 2 * pair[None, :] + 1, ni.to(tl.bfloat16), mask=mm[:, None])


def reconstruct(coeff_buf, flat_token_ids, *, page_size, layer_id, out, freqs_cis, vrt=None):
    if not flat_token_ids.numel():
        return
    vrt = vrt if vrt is not None else vrt_bf16_for(layer_id, flat_token_ids.device)
    if vrt is None:
        return
    freqs = torch.view_as_real(freqs_cis.contiguous()).reshape(-1, config.ROPE_DIM).float().contiguous()
    _reconstruct_kernel[(triton.cdiv(flat_token_ids.shape[0], config.BLOCK_M),)](
        coeff_buf.view(fp8_dtype).reshape(-1),
        coeff_buf.reshape(-1), coeff_buf.view(torch.int32).reshape(-1), flat_token_ids, vrt, freqs, out,
        flat_token_ids.shape[0], freqs_cis.shape[0], PAGE_SIZE=page_size, PAGE_BYTES=coeff_buf.shape[-1],
        BLOCK_M=config.BLOCK_M, NUM_HALVES=config.NUM_HALVES, BLOCK_NOPE_H=config.BLOCK_NOPE_H,
        NOPE_DIM=config.NOPE_DIM, ROPE_DIM=config.ROPE_DIM, COEFF_DIM=config.COEFF_DIM,
        SCALE_TILES=config.SCALE_TILES, TILE_SIZE=config.TILE_SIZE, META_BYTES=config.COEFF_SCALE_BYTES,
        BYTES=config.BYTES_PER_TOKEN, VRT_STRIDE=config.HEAD_DIM, FREQ_STRIDE=config.ROPE_DIM,
        OUT_STRIDE=out.shape[-1], num_warps=config.NUM_WARPS, num_stages=config.NUM_STAGES,
    )
