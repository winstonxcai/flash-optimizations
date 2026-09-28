"""Triton coefficient writes for the page-addressable score cache."""
import torch
import triton
import triton.language as tl

from .. import config
from ..reference import fp8_dtype, store_torch


@triton.jit
def _store_kernel(buf_fp8, buf_u8, buf_i32, loc, coeff, scales, pos, valid,
                  PAGE_SIZE: tl.constexpr, PAGE_BYTES: tl.constexpr,
                  NUM_COEFF: tl.constexpr, NUM_SCALE: tl.constexpr,
                  BYTES: tl.constexpr, META_BYTES: tl.constexpr,
                  BLOCK_COEFF: tl.constexpr, BLOCK_SCALE: tl.constexpr,
                  HAS_VALID: tl.constexpr):
    tid = tl.program_id(0)
    l = tl.load(loc + tid)
    base = (l // PAGE_SIZE) * PAGE_BYTES + (l % PAGE_SIZE) * BYTES
    # Decode keeps a row per plan entry and masks the non-boundary ones here,
    # rather than dropping them in torch: a dropped row is a data-dependent
    # shape, which cuda-graph capture cannot record. A masked store is not.
    keep = tl.load(valid + tid).to(tl.int1) if HAS_VALID else True
    c = tl.arange(0, BLOCK_COEFF)
    tl.store(buf_fp8 + base + c,
             tl.load(coeff + tid * NUM_COEFF + c, mask=(c < NUM_COEFF) & keep, other=0.),
             mask=(c < NUM_COEFF) & keep)
    s = tl.arange(0, BLOCK_SCALE)
    tl.store(buf_u8 + base + NUM_COEFF + s,
             tl.load(scales + tid * NUM_SCALE + s, mask=(s < NUM_SCALE) & keep, other=0),
             mask=(s < NUM_SCALE) & keep)
    tl.store(buf_i32 + (base + META_BYTES) // 4, tl.load(pos + tid), mask=keep)


def store(buf, loc, coeff_fp8, scale_u8, pos, page_size, valid=None):
    # The torch fallback is the eager path (small prefill chunks). Decode always
    # goes through the kernel, since a masked store has no host-sync form and a
    # decode step may be running inside a captured graph.
    if valid is None and loc.numel() < 16:
        return store_torch(buf, loc, coeff_fp8, scale_u8, pos, page_size)
    _store_kernel[(loc.shape[0],)](
        buf.view(fp8_dtype).reshape(-1), buf.reshape(-1), buf.view(torch.int32).reshape(-1),
        loc, coeff_fp8, scale_u8, pos, loc if valid is None else valid,
        PAGE_SIZE=page_size, PAGE_BYTES=buf.shape[-1], NUM_COEFF=config.COEFF_DIM,
        NUM_SCALE=config.SCALE_TILES, BYTES=config.BYTES_PER_TOKEN,
        META_BYTES=config.COEFF_SCALE_BYTES, BLOCK_COEFF=triton.next_power_of_2(config.COEFF_DIM),
        BLOCK_SCALE=triton.next_power_of_2(config.SCALE_TILES), HAS_VALID=valid is not None,
    )
