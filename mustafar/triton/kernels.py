"""Triton kernels for physically-sparse TopMag compression (Stage 0).

Compression-only: these pack/unpack pre-transform c4 latent rows. They never
touch the memory pool, the decoder, or the store. Host logic lives in
mustafar/sparse.py; this subfolder holds only @triton.jit kernels.

Layout conventions (shared with sparse.py, single source of truth in code):
  - keep-mask: bool [n, HEAD_DIM], True = keep (exact-global TopMag,
    computed once from the unmodified latent via reference.topmag_keep_mask).
  - packed:    [n, KEEP_K] in the latent's dtype, columns in ascending
    keep-column order (rank = flat cumsum of the mask - 1).
  - bitmap:    [n, 8] int64, word w covers cols 64w..64w+63; bit (63 - lane)
    of word w is 1 iff col 64w+lane is kept (MSB = lane 0, upstream
    mustafar-upstream convention). int64 not uint64 (torch storage); signed
    `>>` + `& 1` extracts the correct bit even for the top bit (stored as
    -2**63).

Kernel B is the exact inverse of A: same cumsum rank, masked load instead of
masked scatter, `other=0.0` so pruned coords land as exactly 0.
"""
import triton
import triton.language as tl


@triton.jit
def _pack_ccomp_kernel(
    x_ptr,        # [n, HEAD_DIM] latent values (fp32/bf16/fp16)
    mask_ptr,     # [n, HEAD_DIM] int8 keep-mask (0/1), True = keep
    packed_ptr,   # [n, KEEP_K] output, same dtype as x
    n,
    HEAD_DIM: tl.constexpr,
    KEEP_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    vals = tl.load(x_ptr + row * HEAD_DIM + offs)
    bits = tl.load(mask_ptr + row * HEAD_DIM + offs).to(tl.int1)
    rank = tl.cumsum(bits.to(tl.int32), axis=0) - 1     # global packed rank 0..KEEP_K-1
    idx = (row.to(tl.int64) * KEEP_K + rank.to(tl.int64))
    tl.store(packed_ptr + idx, vals, mask=bits)


@triton.jit
def _unpack_ccomp_kernel(
    packed_ptr,   # [n, KEEP_K] packed values
    bitmap_ptr,   # [n, BITMAP_WORDS] int64 bitmaps (MSB = lane 0)
    out_ptr,      # [n, HEAD_DIM] dense output, same dtype as packed
    n,
    HEAD_DIM: tl.constexpr,
    KEEP_K: tl.constexpr,
    BITMAP_WORDS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    word = offs // 64
    lane = offs % 64
    bm = tl.load(bitmap_ptr + row * BITMAP_WORDS + word)      # 8 distinct int64, broadcast
    bits = ((bm >> (63 - lane)) & 1).to(tl.int1)
    rank = tl.cumsum(bits.to(tl.int32), axis=0) - 1
    idx = (row.to(tl.int64) * KEEP_K + rank.to(tl.int64))
    val = tl.load(packed_ptr + idx, mask=bits, other=0.0)
    tl.store(out_ptr + row * HEAD_DIM + offs, val)
