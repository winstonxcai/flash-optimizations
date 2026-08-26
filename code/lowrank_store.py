#!/usr/bin/env python3
"""Low-rank store kernel for DeepSeek-V4 CSA: store 192-dim coefficients in the KV pool.

The 0.5.15 serving bench reconstructed the CSA latent to full 512 dims before the fused
store -> stored bytes identical to native -> compression was pure prefill overhead with no
memory/bandwidth win. This module implements the real fix: store the **192-dim W3
coefficients** (not the reconstructed latent) in a dedicated pool at 200 B/token vs the
native 584 B -> **2.92x KV reduction** for the CSA cache -> decode reads fewer bytes and
the memory ceiling rises.

    STORE (per CSA layer, all TP ranks):  coeffs = RMSNorm(latent) @ Vr        (Vr: 512x192)
        quantized to fp8 + per-64-tile ue8m0 scale, plus the RoPE position (int32),
        written into a new DeepSeekV4LowRankPool; the fused norm+rope+store is skipped.
    READ  (prefill + decode):  gather coeffs -> dequant -> coeffs @ Vr^T -> 512 bf16
        -> RoPE at the stored position -> existing flash_mla_sparse_fwd.

The compressor latent is a ReplicatedLinear output, so all TP ranks see the same latent and
each rank writes an identical per-rank replica (exactly what the native fused store does).

Injected files (SGLANG_OPT_LOWRANK_KV_STORE=1 gates all runtime behavior):
    compressor_v2.py             -> store hook between compress_forward and the fused store
    deepseek_v4_memory_pool.py   -> DeepSeekV4LowRankPool class + construction switch
    pool_configurator.py         -> budget: c4 compressed fraction uses 200 B/token
    deepseek_v4_backend.py       -> decode recon (_forward_decode_lowrank) + prefill recon

v1 note: decode runs with --disable-cuda-graph (plain eager torch recon), trading v1 decode
latency for correctness. The benchmark target is the memory ceiling, not v1 decode speed.
"""
import os
import shutil
import time as _time
from typing import List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

# --- layout constants (200 B/token = 192 fp8 + 3 ue8m0 + 1 pad + 4 int32 pos) ---------
# XKV_COEFF_DIM overrides the basis rank: 192 (default, 200 B/token) or 512 (full
# rank, 524 B/token -- LOSSLESS diagnostic: coeffs = normed @ E where E is the full
# orthonormal eigenbasis of A, so coeffs @ E.T = normed exactly).
#
# PAD_BYTES is derived, NOT hardcoded: the int32 pos at byte COEFF_SCALE_BYTES is
# written/read via buf.view(int32) and must be 4-byte aligned, and BYTES_PER_TOKEN
# must stay 4-aligned so consecutive records can never straddle an int32 slot. A
# fixed PAD=1 works only for COEFF_DIM=192 (196/200 are both %4==0); at 512 the raw
# 521/525 are %4==1 and the misaligned pos write clobbers the last two ue8m0 scale
# bytes (garbage scale >= 247 -> exp2(247-127) overflows fp32 -> Inf -> NaN in the
# recon). (4 - (D+T)%4)%4 gives PAD=1 at 192 and PAD=0 at 512.
COEFF_DIM = int(os.environ.get("XKV_COEFF_DIM", "192"))  # rank-192 coefficients
HEAD_DIM = 512              # full compressor latent dim
ROPE_DIM = 64               # rotary tail dims
NOPE_DIM = HEAD_DIM - ROPE_DIM          # 448
TILE_SIZE = 64
SCALE_TILES = COEFF_DIM // TILE_SIZE    # 3 (8 for full rank)
POS_BYTES = 4                            # int32
PAD_BYTES = (4 - (COEFF_DIM + SCALE_TILES) % 4) % 4     # 1 at 192, 0 at 512
COEFF_SCALE_BYTES = COEFF_DIM + SCALE_TILES + PAD_BYTES   # 196 / 520, always %4==0
BYTES_PER_TOKEN = COEFF_SCALE_BYTES + POS_BYTES           # 200 / 524, always %4==0

# fused recon launch config (BLOCK_M swept in the self-test; no triton.autotune
# on the hot decode path -- house policy). BLOCK_NOPE pads the 448-col nope dot
# to a power of two (tl block shapes must be pow2); cols 448..511 of acc hold
# the unrotated rope and are dropped by the store mask.
BLOCK_M = 32
NUM_WARPS = 8
# which caller fed the last recon (decode_lowrank sets "decode" right before its
# call; backend prefill path never touches it -> stays "prefill")
_RECON_SOURCE = "prefill"
NUM_STAGES = 1
BLOCK_NOPE = triton.next_power_of_2(NOPE_DIM)      # 512
# NOPE output is computed in halves of BLOCK_NOPE_H cols so the b_full operand of
# each tl.dot is [64, BLOCK_NOPE_H] bf16. BLOCK_NOPE_H=256 -> 32 KB dynamic SMEM
# per dot, keeping the kernel under the 48 KB default SMEM carveout. A full
# [64,512] b_full would need 64 KB dynamic SMEM and the >48 KB opt-in launch
# path, which fails with CUDA_ERROR_OUT_OF_MEMORY on a near-full GPU (the
# 32k-prefill recon crash). Peak SMEM here = one 32 KB b_full + 2x4 KB rope.
BLOCK_NOPE_H = 256
NUM_HALVES = BLOCK_NOPE // BLOCK_NOPE_H          # 2

fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn

# --- runtime state ----------------------------------------------------------
_cur_layer: Optional[int] = None
_basis_dir: str = ""
_Vr: dict = {}                # layer -> [512,192] fp32 orthonormal basis (columns, CPU)
_VrT: dict = {}               # layer -> [192,512] fp32 (CPU)
_Vr_dev: dict = {}            # (layer, device) -> Vr on device
_VrT_dev: dict = {}           # (layer, device) -> VrT on device
_VrT_bf16_dev: dict = {}      # (layer, device) -> VrT [192,512] bf16 (dot operand)
_freqs_cis: Optional[torch.Tensor] = None
_freqs_real: Optional[torch.Tensor] = None   # [max_len,64] fp32 interleaved cos/sin


def lowrank_enabled() -> bool:
    return os.environ.get("SGLANG_OPT_LOWRANK_KV_STORE") == "1"


def set_cur_layer(layer_id: int) -> None:
    global _cur_layer
    _cur_layer = layer_id


def _dbg(msg, **kw):
    if os.environ.get("XKV_DEBUG") == "1":
        import json
        line = json.dumps({"lowrank": msg, **kw}, default=str)
        with open(os.path.join(_ctrl_dir(), "debug.log"), "a") as f:
            f.write(line + "\n")


def _tdbg(msg, **kw):
    """Stage-timing log for decode_lowrank, gated on XKV_DECODE_TIMING=1.
    Written to ctrl/timing.log independent of XKV_DEBUG so clean ITL runs
    (XKV_DEBUG=0) still get the per-stage breakdown."""
    if os.environ.get("XKV_DECODE_TIMING") == "1":
        import json
        line = json.dumps({"timing": msg, **kw}, default=str)
        with open(os.path.join(_ctrl_dir(), "timing.log"), "a") as f:
            f.write(line + "\n")


def _ctrl_dir() -> str:
    return os.environ.get(
        "SG_CTRL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctrl")
    )


# --- basis (per-layer Vr from the xKV fixed basis A_i = Vr Vr^T) --------------
def _load_Vr(layer_id: int) -> Optional[torch.Tensor]:
    """Vr = top-192 orthonormal eigenvectors of A_i (A is a projector P = VrVr^T;
    any orthonormal basis of range(A) gives the same reconstruction VrVr^T = A).
    Kept on CPU; moved to the compute device via _vr_for/_vrt_for (each TP rank
    runs its own process, so the device differs per rank)."""
    if layer_id in _Vr:
        return _Vr[layer_id]
    path = os.path.join(_basis_dir, f"A_{layer_id:03d}.pt")
    if not os.path.exists(path):
        return None
    A = torch.load(path, map_location="cpu")
    A = A.to(torch.float32)
    try:
        # A symmetric PSD -> eigh; top-192 eigenvectors span range(A).
        evals, evecs = torch.linalg.eigh((A + A.T) / 2)
        Vr = evecs[:, -COEFF_DIM:].contiguous()
        _Vr[layer_id] = Vr
        _VrT[layer_id] = Vr.T.contiguous()
        return Vr
    except Exception:
        return None


def _vr_for(layer_id: int, device) -> Optional[torch.Tensor]:
    """Vr [512,192] on `device` (cached per (layer, device))."""
    key = (layer_id, str(device))
    if key not in _Vr_dev:
        Vr = _load_Vr(layer_id)
        if Vr is None:
            return None
        _Vr_dev[key] = Vr.to(device)
        _VrT_dev[key] = Vr.T.contiguous().to(device)
    return _Vr_dev[key]


def _vrt_for(layer_id: int, device) -> Optional[torch.Tensor]:
    """VrT [192,512] on `device` (cached per (layer, device))."""
    if _vr_for(layer_id, device) is None:
        return None
    return _VrT_dev[(layer_id, str(device))]


def _vrt_bf16_for(layer_id: int, device) -> Optional[torch.Tensor]:
    """VrT [192,512] bf16 on `device` (the dot operand of the fused recon)."""
    key = (layer_id, str(device))
    if key not in _VrT_bf16_dev:
        VrT = _vrt_for(layer_id, device)
        if VrT is None:
            return None
        _VrT_bf16_dev[key] = VrT.to(torch.bfloat16).contiguous()
    return _VrT_bf16_dev[key]


def _freqs_real_for(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Flattened interleaved cos/sin table for the kernel, matching the fused
    store's layout: `torch.view_as_real(freqs_cis).flatten(-2)` so element
    2k=cos_k, 2k+1=sin_k, one 64-wide row per position."""
    global _freqs_real
    if _freqs_real is None:
        _freqs_real = (
            torch.view_as_real(freqs_cis.contiguous())
            .reshape(-1, ROPE_DIM)
            .to(torch.float32)
            .contiguous()
        )
    return _freqs_real


def set_basis_dir(d: str) -> None:
    global _basis_dir
    _basis_dir = d


# --- ue8m0 quantization (mirrors the fused store's per-64-tile scale) ---------
def _quant_ue8m0(x: torch.Tensor, tile: int = TILE_SIZE):
    """x: [n, COEFF_DIM] fp32 -> (fp8 [n,192], scale_u8 [n,3]).
    Matches _quant_k_cache_fused_kernel's convention exactly:
      max_abs_clamped = max(|x_tile|, EPS);  scale = max_abs_clamped / FP8_MAX
      s = uint8(ceil(log2(scale)) + 127);    fp8 = clamp(x / 2^(s-127), [min,max])
    (was: e = floor(log2(maxabs)) without the +127 offset -> fp8 = x*2^125
    overflowed float8_e4m3fn to NaN, poisoning every recon)."""
    n = x.shape[0]
    fp8_info = torch.finfo(fp8_dtype)
    fp8_max = fp8_info.max
    fp8_min = fp8_info.min
    xt = x.view(n, SCALE_TILES, tile)
    maxabs = xt.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = maxabs / fp8_max
    ceil_log2 = torch.ceil(torch.log2(scale))
    s = (ceil_log2 + 127.0).clamp(0.0, 255.0)
    x_scaled = xt / (2.0 ** ceil_log2)
    fp8 = x_scaled.clamp(fp8_min, fp8_max).to(fp8_dtype)
    return fp8.reshape(n, COEFF_DIM), s.to(torch.uint8).reshape(n, SCALE_TILES)


def _dequant_ue8m0(coeff_fp8: torch.Tensor, scale: torch.Tensor, n: int):
    """coeff_fp8 [n,192] fp8, scale [n,3] uint8 -> [n,192] fp32."""
    c = coeff_fp8.to(torch.float32).view(n, SCALE_TILES, TILE_SIZE)
    s = (2.0 ** (scale.to(torch.float32) - 127.0)).unsqueeze(-1)
    return (c * s).reshape(n, COEFF_DIM)


# --- store accessor (torch; page-addressable pool, 200 B/token) --------------
def _set_coeff_buffer_torch(
    buf: torch.Tensor, loc: torch.Tensor, coeff_fp8: torch.Tensor,
    scale_u8: torch.Tensor, pos: torch.Tensor, page_size: int,
) -> None:
    """buf: uint8 [num_pages, page_bytes]. loc: flat compressed-token locs
    (page*page_size + offset, same addressing the fused store uses)."""
    n = loc.shape[0]
    assert buf.dtype == torch.uint8 and buf.is_contiguous()
    page_bytes = buf.shape[-1]
    page = loc // page_size
    off = loc % page_size
    base = page * page_bytes + off * BYTES_PER_TOKEN          # [n] byte offsets

    flat_fp8 = buf.view(fp8_dtype).reshape(-1)
    cidx = base[:, None] + torch.arange(COEFF_DIM, device=loc.device)[None, :]
    flat_fp8[cidx] = coeff_fp8

    flat_u8 = buf.reshape(-1)
    sidx = base[:, None] + COEFF_DIM + torch.arange(SCALE_TILES, device=loc.device)[None, :]
    flat_u8[sidx] = scale_u8

    flat_i32 = buf.view(torch.int32).reshape(-1)
    pidx = (base + COEFF_SCALE_BYTES) // 4
    flat_i32[pidx] = pos


@triton.jit
def _set_coeff_kernel(
    buf_fp8_ptr, buf_uint8_ptr, buf_int32_ptr,
    loc_ptr, coeff_ptr, scale_ptr, pos_ptr,
    PAGE_SIZE: tl.constexpr, PAGE_BYTES: tl.constexpr,
    NUM_COEFF: tl.constexpr, NUM_SCALE: tl.constexpr,
    BYTES_PER_TOKEN: tl.constexpr, COEFF_SCALE_BYTES: tl.constexpr,
    BLOCK_COEFF: tl.constexpr, BLOCK_SCALE: tl.constexpr,
):
    tid = tl.program_id(0)
    loc = tl.load(loc_ptr + tid)
    page = loc // PAGE_SIZE
    off = loc % PAGE_SIZE
    base = page * PAGE_BYTES + off * BYTES_PER_TOKEN

    cr = tl.arange(0, BLOCK_COEFF)
    cmask = cr < NUM_COEFF
    c = tl.load(coeff_ptr + tid * NUM_COEFF + cr, mask=cmask, other=0.0)
    tl.store(buf_fp8_ptr + base + cr, c, mask=cmask)

    sr = tl.arange(0, BLOCK_SCALE)
    smask = sr < NUM_SCALE
    s = tl.load(scale_ptr + tid * NUM_SCALE + sr, mask=smask, other=0)
    tl.store(buf_uint8_ptr + base + NUM_COEFF + sr, s, mask=smask)

    tl.store(buf_int32_ptr + (base + COEFF_SCALE_BYTES) // 4,
             tl.load(pos_ptr + tid))


def _set_coeff_buffer(
    buf, loc, coeff_fp8, scale_u8, pos, page_size,
    use_triton: bool = False,
) -> None:
    if use_triton and loc.numel() >= 16:
        page_bytes = buf.shape[-1]
        _set_coeff_kernel[(loc.shape[0],)](
            buf.view(fp8_dtype).reshape(-1),
            buf.reshape(-1),
            buf.view(torch.int32).reshape(-1),
            loc, coeff_fp8, scale_u8, pos,
            PAGE_SIZE=page_size, PAGE_BYTES=page_bytes,
            NUM_COEFF=COEFF_DIM, NUM_SCALE=SCALE_TILES,
            BYTES_PER_TOKEN=BYTES_PER_TOKEN,
            COEFF_SCALE_BYTES=COEFF_SCALE_BYTES,
            BLOCK_COEFF=triton.next_power_of_2(COEFF_DIM),
            BLOCK_SCALE=triton.next_power_of_2(SCALE_TILES),
        )
    else:
        _set_coeff_buffer_torch(buf, loc, coeff_fp8, scale_u8, pos, page_size)


# --- READ kernel: fused low-rank recon (gather -> dequant -> GEMM -> rope) ---
# One program handles BLOCK_M tokens. Replaces the eager torch chain (fp32
# [n,512] recon write + bf16 copy_) with a single launch: the latent is built in
# fp32 accumulators and written straight to the bf16 workspace slice the sparse
# attention reads -- the fp32 HBM round-trip and the copy_ are gone.
#
# Mirrors house kernels exactly: _set_coeff_kernel's byte addressing,
# _dequantize_k_cache_paged_kernel's per-64-tile ue8m0 dequant, and
# _compress_norm_rope_kernel's interleaved freqs + strided rope stores. The rope
# real/imag are split as TWO dots over even/odd VrT columns (not tl.split on the
# full acc -- the acc is a tl.dot register tensor and tl.split of an MMA layout
# is the fragile part; a [M,64] dot is cleanly avoidable at +12.5% FLOPs).

@triton.jit
def _recon_lowrank_kernel(
    buf_fp8_ptr, buf_uint8_ptr, buf_int32_ptr,
    loc_ptr, vrt_bf16_ptr, freqs_real_ptr, out_ptr,
    N, MAX_POS,
    PAGE_SIZE: tl.constexpr, PAGE_BYTES: tl.constexpr,
    COEFF_DIM: tl.constexpr, SCALE_TILES: tl.constexpr,
    TILE_SIZE: tl.constexpr, COEFF_SCALE_BYTES: tl.constexpr,
    BYTES_PER_TOKEN: tl.constexpr,
    NOPE_DIM: tl.constexpr, BLOCK_NOPE: tl.constexpr,
    BLOCK_NOPE_H: tl.constexpr, NUM_HALVES: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    OUT_STRIDE_0: tl.constexpr, VRT_STRIDE_N: tl.constexpr,
    FREQS_STRIDE: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m < N

    # flat pool loc -> byte offset of the 200-B/token record
    loc = tl.load(loc_ptr + m, mask=m_mask, other=0).to(tl.int64)
    base = (loc // PAGE_SIZE) * PAGE_BYTES + (loc % PAGE_SIZE) * BYTES_PER_TOKEN

    # stored RoPE position (int32 at byte COEFF_SCALE_BYTES), clamped so a stale
    # loc can never gather freqs_cis OOB (that poisoned the eager stream once).
    pos = tl.load(buf_int32_ptr + (base + COEFF_SCALE_BYTES) // 4,
                  mask=m_mask, other=0)
    pos = tl.minimum(tl.maximum(pos, 0), MAX_POS - 1)

    # interleaved freqs row `pos`: 2k=cos, 2k+1=sin
    pair = tl.arange(0, ROPE_DIM // 2)
    f_cos = tl.load(freqs_real_ptr + pos[:, None] * FREQS_STRIDE + (2 * pair)[None, :],
                    mask=m_mask[:, None], other=1.0)
    f_sin = tl.load(freqs_real_ptr + pos[:, None] * FREQS_STRIDE + (2 * pair + 1)[None, :],
                    mask=m_mask[:, None], other=0.0)

    k = tl.arange(0, TILE_SIZE)
    cp = tl.arange(0, ROPE_DIM // 2)

    # NOPE output computed in halves: the b_full operand of each tl.dot is
    # [64, BLOCK_NOPE_H] bf16 = 32 KB SMEM (a full [64,512] operand would need
    # 64 KB dynamic SMEM, which trips the >48 KB opt-in launch path and fails
    # with CUDA_ERROR_OUT_OF_MEMORY on a near-full GPU). tl.range keeps only one
    # half's b_full live per iteration; rope dots run in a separate pass below.
    out_row = m[:, None] * OUT_STRIDE_0
    for half in tl.range(0, NUM_HALVES):
        cn_h = half * BLOCK_NOPE_H + tl.arange(0, BLOCK_NOPE_H)
        acc_h = tl.zeros((BLOCK_M, BLOCK_NOPE_H), dtype=tl.float32)
        for t in tl.static_range(SCALE_TILES):
            fp8_vals = tl.load(
                buf_fp8_ptr + base[:, None] + (t * TILE_SIZE + k)[None, :],
                mask=m_mask[:, None], other=0.0)
            s = tl.load(buf_uint8_ptr + base + COEFF_DIM + t, mask=m_mask, other=0)
            sp2 = tl.exp2((s.to(tl.float32) - 127.0))
            a = (fp8_vals.to(tl.float32) * sp2[:, None]).to(tl.bfloat16)
            row = (t * TILE_SIZE + k)[:, None] * VRT_STRIDE_N
            acc_h += tl.dot(a, tl.load(vrt_bf16_ptr + row + cn_h[None, :]))
        tl.store(out_ptr + out_row + cn_h[None, :], acc_h.to(tl.bfloat16),
                 mask=m_mask[:, None] & (cn_h[None, :] < NOPE_DIM))

    # ROPE real/imag (b_rope is [64,32] = 4 KB SMEM each); kept out of the nope
    # loop so each accumulator sums exactly SCALE_TILES dots.
    acc_r = tl.zeros((BLOCK_M, ROPE_DIM // 2), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, ROPE_DIM // 2), dtype=tl.float32)
    for t in tl.static_range(SCALE_TILES):
        fp8_vals = tl.load(
            buf_fp8_ptr + base[:, None] + (t * TILE_SIZE + k)[None, :],
            mask=m_mask[:, None], other=0.0)
        s = tl.load(buf_uint8_ptr + base + COEFF_DIM + t, mask=m_mask, other=0)
        sp2 = tl.exp2((s.to(tl.float32) - 127.0))
        a = (fp8_vals.to(tl.float32) * sp2[:, None]).to(tl.bfloat16)
        row = (t * TILE_SIZE + k)[:, None] * VRT_STRIDE_N
        acc_r += tl.dot(a, tl.load(vrt_bf16_ptr + row + (NOPE_DIM + 2 * cp)[None, :]))
        acc_i += tl.dot(a, tl.load(vrt_bf16_ptr + row + (NOPE_DIM + 2 * cp + 1)[None, :]))

    nr = acc_r * f_cos - acc_i * f_sin
    ni = acc_r * f_sin + acc_i * f_cos
    tl.store(out_ptr + out_row + (NOPE_DIM + 2 * cp)[None, :],
             nr.to(tl.bfloat16), mask=m_mask[:, None])
    tl.store(out_ptr + out_row + (NOPE_DIM + 2 * cp + 1)[None, :],
             ni.to(tl.bfloat16), mask=m_mask[:, None])


def _dequantize_lowrank_k_cache_paged_triton(
    coeff_buf, flat_token_ids, *, page_size, layer_id, out,
):
    """Triton fused recon: gather -> dequant -> GEMM -> rope -> bf16, one launch."""
    n = flat_token_ids.shape[0]
    VrT_bf16 = _vrt_bf16_for(layer_id, flat_token_ids.device)
    if VrT_bf16 is None:
        _dbg("recon_no_basis", layer=layer_id)
        return
    freqs_real = _freqs_real_for(_freqs_cis)
    page_bytes = coeff_buf.shape[-1]
    loc = flat_token_ids.contiguous()
    if os.environ.get("XKV_DEBUG") == "1":
        free_gb, _tot = torch.cuda.mem_get_info(flat_token_ids.device)
        _dbg("recon_prelaunch", layer=layer_id, n=n, free_gb=round(free_gb / 1e9, 3),
             reserved_gb=round(torch.cuda.memory_reserved(flat_token_ids.device) / 1e9, 3))
    _kt0 = None
    if os.environ.get("XKV_DECODE_TIMING") == "1":
        torch.cuda.synchronize()
        _kt0 = _time.time()
    _recon_lowrank_kernel[(triton.cdiv(n, BLOCK_M),)](
        coeff_buf.view(fp8_dtype).reshape(-1),
        coeff_buf.reshape(-1),
        coeff_buf.view(torch.int32).reshape(-1),
        loc,
        VrT_bf16,
        freqs_real,
        out,
        n,
        _freqs_cis.shape[0],
        PAGE_SIZE=page_size, PAGE_BYTES=page_bytes,
        COEFF_DIM=COEFF_DIM, SCALE_TILES=SCALE_TILES,
        TILE_SIZE=TILE_SIZE, COEFF_SCALE_BYTES=COEFF_SCALE_BYTES,
        BYTES_PER_TOKEN=BYTES_PER_TOKEN,
        NOPE_DIM=NOPE_DIM, BLOCK_NOPE=BLOCK_NOPE,
        BLOCK_NOPE_H=BLOCK_NOPE_H, NUM_HALVES=NUM_HALVES, ROPE_DIM=ROPE_DIM,
        OUT_STRIDE_0=out.shape[-1], VRT_STRIDE_N=HEAD_DIM,
        FREQS_STRIDE=ROPE_DIM,
        BLOCK_M=BLOCK_M,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    if os.environ.get("XKV_DECODE_TIMING") == "1":
        torch.cuda.synchronize()
        _tdbg("recon_kernel", layer=layer_id, n=n,
              ms=round((_time.time() - _kt0) * 1e3, 3))
    if os.environ.get("XKV_DEBUG") == "1":
        _dbg("recon_triton", layer=layer_id, n=n,
             loc_min=int(loc.min().item()), loc_max=int(loc.max().item()),
             out_absmean=float(out.abs().mean().item()),
             out_nan=int(torch.isnan(out).sum().item()),
             source=_RECON_SOURCE)


# --- STORE hook (injected into compressor_v2._forward_compress_all_in_one) ---
def store_compressed_lowrank(
    kv_compressed, plan, norm, compress_ratio, is_indexer,
    kv_cache, page_size, out_loc, freqs_cis_cache,
) -> bool:
    """Returns True if the compressed store was handled (low-rank); False falls
    through to the native fused store.

    When lowrank is enabled for the c4 latent (ratio==4, not indexer), this MUST
    return True on every path: the c4 pool is the 200-B layout and the native
    fused store writes 584 B/token into it -> shape mismatch + crash. A failed
    low-rank store therefore SKIPS the store (logged loudly) instead of falling
    through; the benchmark only enables low-rank with the basis present."""
    if not lowrank_enabled():
        return False
    if is_indexer or compress_ratio != 4:
        return False
    lid = _cur_layer
    if lid is None:
        _dbg("store_skip_no_layer")
        return True
    if not _basis_dir:
        set_basis_dir(os.environ.get(
            "SG_LOWRANK_BASIS",
            os.path.join(_ctrl_dir(), "basis"),
        ))
    global _freqs_cis, _freqs_real
    if _freqs_cis is None and freqs_cis_cache is not None:
        _freqs_cis = freqs_cis_cache.detach()
        _freqs_real = None          # invalidate the flattened interleaved table
    try:
        x = kv_compressed.detach().to(torch.float32)
        Vr = _vr_for(lid, x.device)
        if Vr is None:
            _dbg("store_skip_no_basis", layer=lid)
            return True
        plan_i = plan[1].view(torch.int32)                   # [rows, 4]
        seq_len = plan_i[:, 0].to(torch.int64)
        col1 = plan_i[:, 1].to(torch.int64)
        is_decode = bool(getattr(plan, "is_decode", False))
        if is_decode:
            valid = seq_len % compress_ratio == 0
            ragged = torch.arange(seq_len.shape[0], device=seq_len.device)
        else:
            valid = seq_len != -1
            ragged = col1 & 0xFFFF
        if x.shape[0] == plan_i.shape[0]:
            x = x[valid]
            seq_len = seq_len[valid]
            ragged = ragged[valid]
        if x.shape[0] == 0:
            return True
        loc = out_loc[ragged]
        pos = (seq_len - compress_ratio).to(torch.int32)   # int32 -> int32 pool slot

        # coeffs = RMSNorm(latent) @ Vr  (exact fused-kernel norm math)
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
        normed = normed * norm.weight.to(torch.float32)
        coeffs = normed @ Vr                                # [n, 192]
        coeff_fp8, scale_u8 = _quant_ue8m0(coeffs)
        _set_coeff_buffer(kv_cache, loc, coeff_fp8, scale_u8, pos, page_size)
        if os.environ.get("XKV_STORE_RTDBG") == "1":
            # Round-trip isolation: does dequant(coeffs) @ VrT recover the
            # projected latent (the same P@normed the native xKV store writes)?
            # proj_clean = normed @ Vr @ VrT (fp32, no quant); proj_quant is the
            # ue8m0 round trip. A large |proj_quant - proj_clean| means the coeff
            # quant path is broken; a small one means recon values are fine and
            # the failure is downstream (rope / workspace / attention).
            cq = _dequant_ue8m0(coeff_fp8, scale_u8, x.shape[0])
            VrT_d = _vrt_for(lid, x.device)
            if VrT_d is not None:
                proj_clean = normed @ Vr @ VrT_d
                proj_quant = cq @ VrT_d
                dq = (proj_quant - proj_clean).abs()
                perr = (normed - proj_clean).abs()   # projection loss vs full latent
                _dbg("store_rtdbg", layer=lid, rows=int(x.shape[0]),
                     is_decode=is_decode,
                     max_diff=float(dq.max().item()),
                     mean_diff=float(dq.mean().item()),
                     clean_absmax=float(proj_clean.abs().max().item()),
                     clean_absmean=float(proj_clean.abs().mean().item()),
                     normed_absmean=float(normed.abs().mean().item()),
                     proj_err_mean=float(perr.mean().item()),
                     proj_err_max=float(perr.max().item()),
                     pos_min=int(pos.min().item()), pos_max=int(pos.max().item()))
        if os.environ.get("XKV_DEBUG") == "1":
            _dbg("store", layer=lid, rows=int(x.shape[0]),
                 is_decode=is_decode, first_loc=int(loc[0].item()),
                 first_pos=int(pos[0].item()),
                 loc_min=int(loc.min().item()), loc_max=int(loc.max().item()))
        return True
    except Exception as e:
        _dbg("store_error", err=repr(e))
        return True            # low-rank pool layout: never fall through to fused store


# --- READ hook: low-rank recon (gather -> dequant -> expand -> rope -> bf16) ---
def dequantize_lowrank_k_cache_paged(
    coeff_buf, flat_token_ids, *, page_size, layer_id, out,
):
    """Reconstruct [n,1,512] bf16 from the 200-B/token coeff pool.

    Dispatch: fused Triton recon (XKV_RECON_TRITON=1, default) for n>=16, else
    the eager torch chain (XKV_RECON_TRITON=0 forces torch -- the A/B control
    and the numeric reference for the self-test).
    """
    global _freqs_cis
    if _freqs_cis is None:
        _dbg("recon_no_freqs", layer=layer_id)
        return
    n = flat_token_ids.shape[0]
    if n == 0:
        return
    if os.environ.get("XKV_RECON_TRITON", "1") == "1" and n >= 16:
        _dequantize_lowrank_k_cache_paged_triton(
            coeff_buf, flat_token_ids, page_size=page_size,
            layer_id=layer_id, out=out)
    else:
        _dequantize_lowrank_k_cache_paged_torch(
            coeff_buf, flat_token_ids, page_size=page_size,
            layer_id=layer_id, out=out)


def _dequantize_lowrank_k_cache_paged_torch(
    coeff_buf, flat_token_ids, *, page_size, layer_id, out,
):
    """Reconstruct [n,1,512] bf16 from the 200-B/token coeff pool (eager torch).

    coeff_buf: uint8 [num_pages, page_bytes] (the LowRankPool key buffer).
    flat_token_ids: [n] flat compressed-token locs (page*page_size + offset).
    out: [n,1,HEAD_DIM] bf16 (a slice of the sparse workspace).
    """
    global _freqs_cis
    if _freqs_cis is None:
        _dbg("recon_no_freqs", layer=layer_id)
        return

    n = flat_token_ids.shape[0]
    page_bytes = coeff_buf.shape[-1]
    page = flat_token_ids // page_size
    off = flat_token_ids % page_size
    base = page * page_bytes + off * BYTES_PER_TOKEN

    flat_fp8 = coeff_buf.view(fp8_dtype).reshape(-1)
    cidx = base[:, None] + torch.arange(COEFF_DIM, device=flat_token_ids.device)[None, :]
    coeff_fp8 = flat_fp8[cidx].reshape(n, COEFF_DIM)         # [n,192]

    flat_u8 = coeff_buf.reshape(-1)
    sidx = base[:, None] + COEFF_DIM + torch.arange(SCALE_TILES, device=flat_token_ids.device)[None, :]
    scale_u8 = flat_u8[sidx].reshape(n, SCALE_TILES)         # [n,3]

    flat_i32 = coeff_buf.view(torch.int32).reshape(-1)
    pidx = (base + COEFF_SCALE_BYTES) // 4
    # Defensive clamp: a stale/OOB loc must never gather freqs_cis OOB (that
    # illegal access poisons the stream and surfaces as CUBLAS_INTERNAL_ERROR
    # at the recon GEMM below).
    pos = flat_i32[pidx].clamp(0, _freqs_cis.shape[0] - 1)       # [n]

    coeffs = _dequant_ue8m0(coeff_fp8, scale_u8, n)
    VrT = _vrt_for(layer_id, coeffs.device)
    if VrT is None:
        _dbg("recon_no_basis", layer=layer_id)
        return
    recon = coeffs @ VrT                                     # [n,512] fp32
    _apply_rope_tail(recon, _freqs_cis, pos)
    out.copy_(recon.unsqueeze(1).to(torch.bfloat16))        # out is [n,1,512]
    if os.environ.get("XKV_DEBUG") == "1":
        _dbg("recon_torch", layer=layer_id, n=n,
             loc_min=int(flat_token_ids.min().item()),
             loc_max=int(flat_token_ids.max().item()))


def _apply_rope_tail(recon, freqs_cis, pos):
    """Rotate the last ROPE_DIM dims at position pos (matches fused kernel)."""
    n = recon.shape[0]
    # freqs_cis is complex64 [max_len, ROPE_DIM//2] (torch.polar table); the
    # fused kernel uses view_as_real(...).flatten(-2) so element 2k=cos, 2k+1=sin.
    f = freqs_cis[pos.to(torch.long)]                        # complex64 [n, 32]
    fc = torch.view_as_real(f)                               # [n, 32, 2] = (cos, sin)
    cos = fc[..., 0].to(torch.float32)
    sin = fc[..., 1].to(torch.float32)
    tail = recon[:, NOPE_DIM:].view(n, ROPE_DIM // 2, 2)     # (real, imag)
    real = tail[..., 0]
    imag = tail[..., 1]
    nr = real * cos - imag * sin
    ni = real * sin + imag * cos
    recon[:, NOPE_DIM:] = torch.stack([nr, ni], dim=-1).view(n, ROPE_DIM)


# --- DECODE forward: flat-workspace sparse attention over recon+swa ----------
def decode_lowrank(
    self, *, q, layer_id, forward_batch, token_to_kv_pool,
    core_attn_metadata, attn_sink,
):
    """Decode-time CSA attention with the low-rank pool. Mirrors
    _forward_prefill_sparse: gather unique swa + low-rank c4 tokens into a flat
    bf16 workspace, rebase per-query indices, call flash_mla_sparse_fwd."""
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    q_flat = q.squeeze(1)                                    # [B, h_q, d]
    B = q_flat.shape[0]

    swa_page_size = token_to_kv_pool.swa_page_size
    c4_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
    coeff_buf = token_to_kv_pool.get_extra_key_buffer(layer_id)
    swa_buf = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

    swa_page_indices = core_attn_metadata.swa_page_indices    # [B, K_swa]
    swa_topk_lengths = core_attn_metadata.swa_topk_lengths    # [B]
    c4_page_indices = core_attn_metadata.c4_sparse_page_indices  # [B, K_c4]
    c4_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths  # [B]
    if os.environ.get("XKV_DECODE_LOCDBG") == "1":
        _sl = getattr(core_attn_metadata, "seq_lens", None)
        if _sl is None:
            _sl = getattr(forward_batch, "seq_lens", None)
        _ri = getattr(core_attn_metadata, "c4_sparse_raw_indices", None)
        _pt = getattr(core_attn_metadata, "page_table", torch.empty(0))
        _dbg("decode_locdbg",
             B=B, seq_lens=[int(s) for s in (_sl.reshape(-1)[:8] if _sl is not None else [])],
             c4_page_size=c4_page_size,
             c4_topk_lengths=[int(t) for t in c4_topk_lengths[:4]],
             c4_shape=list(c4_page_indices.shape),
             c4_row0=[int(v) for v in c4_page_indices.reshape(-1)[:24]],
             raw_idx0=[int(v) for v in _ri.reshape(-1)[:24]] if _ri is not None else None,
             pt_shape=list(_pt.shape),
             pt_row0=[int(v) for v in _pt.reshape(-1)[:24]])

    def _match(x, value):
        if x is None or x.shape[0] == B:
            return x
        if x.shape[0] > B:
            return x[:B]
        return torch.nn.functional.pad(x, (0, 0, 0, B - x.shape[0]), value=value)

    swa_page_indices = _match(swa_page_indices, 0)
    swa_topk_lengths = _match(swa_topk_lengths, 1)
    c4_page_indices = _match(c4_page_indices, -1)
    c4_topk_lengths = _match(c4_topk_lengths, 1)
    if swa_page_indices.ndim == 2:
        swa_page_indices = swa_page_indices.unsqueeze(1)
    if c4_page_indices.ndim == 2:
        c4_page_indices = c4_page_indices.unsqueeze(1)

    # --- XKV_LOWRANK: decode c4 locs are already in the store's pool-slot space.
    # Once c4_sparse_raw_indices is allocated for decode (backend patch routes
    # the indexer to the v1 topk path), transform_output writes
    #   S(k) = page_table[req, k>>6] << 6 | (k&63)
    # with k = the c4 token index — exactly the loc the coeff store wrote at.
    # (The v2 path, used when raw_indices is None, emits raw-stride-4 locs
    # M(k) = page_table[req, k>>4] << 6 | (4*(k&15)) that do NOT match the
    # store's page layout and land in never-written pages -> EMPTY.)
    # c4_sparse_page_indices therefore needs NO translation when v1 is active.
    pt = getattr(core_attn_metadata, "page_table", None)
    raw = getattr(core_attn_metadata, "c4_sparse_raw_indices", None)
    if raw is not None and raw.numel() > 0 and raw.ndim >= 2:
        if os.environ.get("XKV_DECODE_LOCDBG") == "1":
            r2 = raw.to(torch.int64)
            S2 = c4_page_indices.to(torch.int64)
            if S2.ndim == 3:
                S2 = S2.squeeze(1)
            v = (r2 >= 0)
            _dbg("decode_locraw",
                 n_pts=int(v.sum().item()),
                 raw_idx0=[int(x) for x in r2.reshape(-1)[:12]],
                 s_loc0=[int(x) for x in S2.reshape(-1)[:12]],
                 s_min=int(S2[v].min().item()) if v.any() else None,
                 s_max=int(S2[v].max().item()) if v.any() else None,
                 pt_shape=list(pt.shape) if pt is not None else None)
        # v1 active: c4_sparse_page_indices already holds S(k). Pass through.
        pass
    elif pt is not None and pt.numel() > 0 and pt.ndim >= 2:
        # Fallback (raw unavailable): invert M(k) per loc via the request page
        # table. Works for locs whose raw block was actually allocated; raw-
        # stride locs beyond the allocated pages are unrecoverable.
        pt = pt if pt.ndim == 2 else pt.squeeze(1)
        c4_out = torch.full_like(c4_page_indices, -1)
        for b in range(c4_page_indices.shape[0]):
            row = pt[b].to(torch.int64)
            used = row > 0
            blk = torch.nonzero(used, as_tuple=False).squeeze(-1)
            if blk.numel() == 0:
                continue
            phys = row[blk]
            inv = torch.full(
                (int(phys.max()) + 1,), -1, dtype=torch.int64, device=pt.device
            )
            inv[phys] = blk
            L = c4_page_indices[b].to(torch.int64)
            valid = L >= 0
            P = L >> 6
            in4 = (L & 63) >> 2
            Pc = P.clamp(0, inv.numel() - 1)
            b_idx = inv[Pc]
            k = b_idx * 16 + in4
            kc = k.clamp(0, row.numel() - 1)
            S = (row[kc] << 6) | (k & 63)
            bad = (~valid) | (P >= inv.numel()) | (b_idx < 0) | (k >= row.numel())
            c4_out[b] = torch.where(bad, L, S)
        c4_page_indices = c4_out

    dev = q_flat.device
    # swa_topk_lengths = min(seq_len, SWA_WINDOW) — TOKEN counts (clamped in
    # get_swa_topk_lengths), NOT pages. swa_page_indices are FLAT swa-cache row
    # locs (already translated through full_to_swa, one per window position),
    # NOT page indices. Same for c4: c4_sparse_topk_lengths = clamp(seq_len//4,
    # max=512) counts compressed-TOKEN locs, and each c4_sparse_page_indices
    # entry is ALREADY a FLAT c4-pool loc — topk_transform packs
    # (page_table[block]<<6)|in_page. Treating them as pages re-expands
    # (flat_loc*64+in_page) and OOBs the coeff buffer once locs span the pool
    # (a lone 32k req stayed in bounds only because its 512 locs were small
    # and contiguous; at high concurrency the pool is nearly full).
    swa_lens = swa_topk_lengths.to(torch.int64)                      # [B] tokens
    c4_lens = c4_topk_lengths.to(torch.int64)                        # [B] tokens
    n_att = (swa_lens + c4_lens).to(torch.int32)                     # [B] attended tokens

    OFFSET = 1 << 40
    max_swa_tok = int(swa_lens.max().item()) if B else 0
    max_c4_tok = int(c4_lens.max().item()) if B else 0
    max_att = max_swa_tok + max_c4_tok
    topk = max(128, ((max_att + 127) // 128) * 128)
    if max_att == 0:
        return torch.zeros_like(q_flat)                              # nothing to attend

    # swa: each column is one already-translated flat row loc — keep as-is, no
    # page expansion (expanding flat_row * page_size + in_page OOBs the swa ring
    # at long context, e.g. ring loc * 256 past the 564k-row buffer at 32k).
    swa_flat = swa_page_indices.squeeze(1)                            # [B,W_swa]
    # c4: each column is already a FLAT c4-pool loc (page*64+in_page packed by
    # topk_transform) — keep as-is, no page expansion.
    need_c4 = max_c4_tok
    c4_flat = c4_page_indices[..., :need_c4].reshape(B, -1)           # [B,W_c4]
    W_swa, W_c4 = swa_flat.shape[1], c4_flat.shape[1]

    # Metadata pads c4 entries with -1 (empty topk slots) and clamps
    # c4_sparse_topk_lengths to a minimum of 1, so a query with ZERO real c4
    # locs can still carry a length-1 "-1 loc". Exclude invalid locs here so
    # negative addresses never reach the recon.
    swa_page_ok = (swa_page_indices.squeeze(1) >= 0)                  # [B,W_swa]
    c4_page_ok = (c4_flat >= 0)                                       # [B,W_c4]
    swa_mask = (torch.arange(W_swa, device=dev)[None, :] < swa_lens[:, None]) & swa_page_ok
    c4_mask = (torch.arange(W_c4, device=dev)[None, :] < c4_lens[:, None]) & c4_page_ok

    # Deduplicate across queries into a flat bf16 workspace; swa locs stay
    # < OFFSET, c4 locs = OFFSET + flat loc (same flat loc may exist in both).
    # The swa/c4 tensors are int32 and OFFSET = 2^40 wraps to 0 in int32
    # (torch keeps int32 + 2^40 in int32), so the tag is INVISIBLE unless we
    # promote to int64 FIRST. This silent int32-wrap bug classified every loc
    # as c4 (n_swa=0), routed swa rows through the coeff recon, and was the
    # real cause of the empty long-context output.
    _tseg = None
    if os.environ.get("XKV_DECODE_TIMING") == "1":
        torch.cuda.synchronize()
        _td = {"t0": _time.time()}

        def _tseg(k):
            torch.cuda.synchronize()
            now = _time.time()
            _td[k] = round((now - _td["t0"]) * 1e3, 3)
            _td["t0"] = now

    encoded = torch.cat(
        [swa_flat.reshape(-1).to(torch.int64),
         c4_flat.reshape(-1).to(torch.int64) + OFFSET], dim=0)
    valid_enc = torch.cat([swa_mask.reshape(-1), c4_mask.reshape(-1)])
    u, inv = torch.unique(encoded[valid_enc], return_inverse=True)
    n_u = u.shape[0]
    if _tseg is not None:
        _tseg("unique")
    workspace = self.sparse_prefill_workspace.get(n_u)           # [N,1,512] bf16

    n_swa = int((u < OFFSET).sum())
    swa_u = u[:n_swa].to(torch.int32)
    c4_u = (u[n_swa:] - OFFSET).to(torch.int32)
    if os.environ.get("XKV_DECODE_LOCDBG") == "1" and n_u > n_swa:
        n_sw_ = int(swa_mask.sum()), int(c4_mask.sum())
        enc = encoded[valid_enc]
        big = enc[enc >= OFFSET + 1000]
        n_swa_flat = B * W_swa
        enc_swa = encoded[:n_swa_flat]
        enc_c4 = encoded[n_swa_flat:]
        _dbg("decode_recon_in", layer=layer_id,
             n_c4u=int(c4_u.shape[0]), n_swa=int(n_swa), n_u=int(n_u),
             B=int(B), W_c4=int(W_c4), W_swa=int(W_swa),
             valid_swa=n_sw_[0], valid_c4=n_sw_[1],
             swa_max=int(swa_flat.max().item()),
             c4_flat_max=int(c4_flat.max().item()),
             c4_u_min=int(c4_u.min().item()), c4_u_max=int(c4_u.max().item()),
             u_min=int(u.min().item()), u_max=int(u.max().item()),
             enc_swa_min=int(enc_swa.min().item()) if enc_swa.numel() else None,
             enc_swa_max=int(enc_swa.max().item()) if enc_swa.numel() else None,
             enc_c4_min=int(enc_c4.min().item()) if enc_c4.numel() else None,
             enc_c4_max=int(enc_c4.max().item()) if enc_c4.numel() else None,
             n_big=int(big.numel()),
             enc0=[int(x) for x in encoded.reshape(-1)[:8]],
             enc_tail=[int(x) for x in encoded.reshape(-1)[-8:]],
             swa0=[int(x) for x in swa_flat.reshape(-1)[:12]],
             c4flat0=[int(x) for x in c4_flat.reshape(-1)[:12]],
             c4u0=[int(x) for x in c4_u.reshape(-1)[:12]],
             u_tail=[int(x) for x in u.reshape(-1)[-8:]])
    global _RECON_SOURCE
    _RECON_SOURCE = "decode"
    if n_swa > 0:
        dequantize_k_cache_paged(swa_buf, swa_u, page_size=swa_page_size,
                                 out=workspace[:n_swa])
    if _tseg is not None:
        _tseg("swa_dequant")
    if n_u > n_swa:
        dequantize_lowrank_k_cache_paged(
            coeff_buf, c4_u, page_size=c4_page_size,
            layer_id=layer_id, out=workspace[n_swa:],
        )
    if _tseg is not None:
        _tseg("c4_recon")

    # Map every (query, position) to its workspace row; invalid -> -1.
    inv_map = torch.full((encoded.shape[0],), -1, dtype=torch.int32, device=dev)
    inv_map[valid_enc] = inv.to(torch.int32)

    # Pack per query: [valid swa rows..., valid c4 rows...] at the front, pad to
    # topk with -1. Attention is order-invariant, so row order within a query is
    # irrelevant; the kernel reads exactly n_att[b] entries per query.
    combined = torch.full((B, topk), -1, dtype=torch.int32, device=dev)
    swa_block = inv_map[:B * W_swa].reshape(B, W_swa)
    c4_block = inv_map[B * W_swa:].reshape(B, W_c4)

    r_idx = torch.arange(B, device=dev)
    col_swa = torch.arange(W_swa, device=dev)[None, :].expand(B, W_swa)
    m_swa = col_swa < swa_lens[:, None]
    combined[r_idx[:, None].expand(B, W_swa)[m_swa], col_swa[m_swa]] = swa_block[m_swa]
    col_c4 = swa_lens[:, None] + torch.arange(W_c4, device=dev)[None, :]
    m_c4 = torch.arange(W_c4, device=dev)[None, :] < c4_lens[:, None]
    combined[r_idx[:, None].expand(B, W_c4)[m_c4], col_c4[m_c4]] = c4_block[m_c4]
    if _tseg is not None:
        _tseg("build")

    o, _, _ = flash_mla_sparse_fwd(
        q=q_flat, kv=workspace, indices=combined.unsqueeze(1),
        sm_scale=self.softmax_scale, d_v=self.head_dim_v,
        attn_sink=attn_sink, topk_length=n_att,
    )
    if _tseg is not None:
        _tseg("attn")
        _rank = 0
        try:
            import torch.distributed as _tdist
            if _tdist.is_initialized():
                _rank = _tdist.get_rank()
        except Exception:
            pass
        _tdbg("decode_step", layer=layer_id, rank=_rank, B=int(B),
              n_swa=int(n_swa), n_c4=int(n_u - n_swa),
              W_swa=int(W_swa), W_c4=int(W_c4),
              n_att0=int(n_att.reshape(-1)[0].item()) if B else 0,
              total_ms=round(sum(v for k, v in _td.items() if k != "t0"), 3),
              segments=_td)
    return o


# =============================================================================
# Patch / unpatch
# =============================================================================
# The 4 injected targets live in the serving source tree. SG_LOWRANK_SRC lets
# the patch target a pristine clone (/sgl-workspace/sglang-lowrank/python) so
# the low-rank kernel never collides with the sg_capture-patched default source.
_SRC_ROOT = os.environ.get(
    "SG_LOWRANK_SRC", "/sgl-workspace/sglang/python")
COMPRESSOR_V2 = f"{_SRC_ROOT}/sglang/srt/layers/attention/dsv4/compressor_v2.py"
MEM_POOL = f"{_SRC_ROOT}/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
POOL_CFG = f"{_SRC_ROOT}/sglang/srt/model_executor/pool_configurator.py"
DSV4_BACKEND = f"{_SRC_ROOT}/sglang/srt/layers/attention/deepseek_v4_backend.py"
TRANSFER_DIR = "/mnt/host_root/home/jovyan/winstonxcai/transferibility/xkv_decode"
MARKER = "## XKV_LOWRANK"

_IMPORT_BLOCK = (
    "\n"
    + MARKER
    + " (import)\n"
    "import sys as _sg_lr_sys\n"
    f"if {TRANSFER_DIR!r} not in _sg_lr_sys.path:\n"
    f"    _sg_lr_sys.path.insert(0, {TRANSFER_DIR!r})\n"
    "try:\n"
    "    import lowrank_store as _sg_lr\n"
    "except Exception as _sg_lr_e:\n"
    "    _sg_lr = None\n"
)

# --- pool class injected into deepseek_v4_memory_pool.py (after imports) ------
_POOL_CLASS = '''
class DeepSeekV4LowRankPool(KVCache):
    """KV pool for the rank-192 CSA coefficients: 200 B/token
    (192 fp8 + 3 ue8m0 scale + 1 pad + 4 int32 RoPE position)."""

    coeff_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.coeff_dim = COEFF_DIM
        self._create_buffer()

    def get_bytes_per_token(self) -> int:
        return BYTES_PER_TOKEN

    def _create_buffer(self):
        page_bytes = self.page_size * self.get_bytes_per_token()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                self.coeff_buffer = [
                    torch.zeros(
                        (self.size + self.page_size + 1) // self.page_size,
                        page_bytes,
                        dtype=self.coeff_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.coeff_buffer[layer_id]

    def get_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError()

    def get_value_buffer(self, *args, **kwargs):
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def set_coeff_buffer(
        self, layer_id: int, loc: torch.Tensor, coeff_fp8: torch.Tensor,
        scale_ue8m0: torch.Tensor, position: torch.Tensor,
    ) -> None:
        buf = self.coeff_buffer[layer_id]
        _sg_lr._set_coeff_buffer(
            buf, loc, coeff_fp8, scale_ue8m0, position, self.page_size)
'''

_FORWARD_UNIFIED_HOOK = (
    "\n"
    "        " + MARKER + " (attr)\n"
    "        if _sg_lr is not None:\n"
    "            _sg_lr.set_cur_layer(layer_id)\n"
)

_STORE_CALL = (
    "\n"
    "        " + MARKER + " (store)\n"
    "        if _sg_lr is not None and _sg_lr.store_compressed_lowrank(\n"
    "            kv_compressed, plan, norm, compress_ratio, is_indexer,\n"
    "            kv_cache, page_size, out_loc, freqs_cis_cache,\n"
    "        ):\n"
    "            return\n"
)

_BACKEND_DECODE_BRANCH = (
    "\n"
    "            " + MARKER + " (decode_lowrank)\n"
    "            if (_sg_lr is not None and _sg_lr.lowrank_enabled()\n"
    "                    and compress_ratio == 4\n"
    "                    and forward_batch.forward_mode.is_decode()):\n"
    "                return _sg_lr.decode_lowrank(\n"
    "                    self, q=q, layer_id=layer_id,\n"
    "                    forward_batch=forward_batch,\n"
    "                    token_to_kv_pool=token_to_kv_pool,\n"
    "                    core_attn_metadata=core_attn_metadata,\n"
    "                    attn_sink=attn_sink,\n"
    "                )\n"
)


def _apply(path, edits):
    with open(path) as f:
        s = f.read()
    if MARKER in s:
        print(f"[patch] {path}: already patched")
        return
    for anchor, new, tag in edits:
        assert s.count(anchor) == 1, (
            f"{path} anchor '{tag}' count={s.count(anchor)}")
        s = s.replace(anchor, new, 1)
    with open(path, "w") as f:
        f.write(s)
    print(f"[patch] {path}: ok")


def patch():
    os.makedirs(_ctrl_dir(), exist_ok=True)

    _apply(COMPRESSOR_V2, [
        ("from __future__ import annotations\n", "from __future__ import annotations\n" + _IMPORT_BLOCK, "import"),
        ("    ) -> None:\n"
         "        if forward_batch.forward_mode.is_idle():\n"
         "            return\n",
         "    ) -> None:\n"
         "        if forward_batch.forward_mode.is_idle():\n"
         "            return\n" + _FORWARD_UNIFIED_HOOK, "forward_unified"),
        ("        # Step 2: norm + rope + store\n"
         "        compress_norm_rope_store(\n",
         _STORE_CALL +
         "        # Step 2: norm + rope + store\n"
         "        compress_norm_rope_store(\n", "store_hook"),
    ])

    _apply(MEM_POOL, [
        ("from __future__ import annotations\n", "from __future__ import annotations\n" + _IMPORT_BLOCK, "import"),
        ("class DeepSeekV4IndexerPool(KVCache):\n", _POOL_CLASS + "\n\nclass DeepSeekV4IndexerPool(KVCache):\n", "pool_class"),
        ("            c4_kv_pool_type = DeepSeekV4SingleKVPool\n",
         "            c4_kv_pool_type = DeepSeekV4SingleKVPool\n"
         "            if _sg_lr is not None and _sg_lr.lowrank_enabled():\n"
         "                c4_kv_pool_type = DeepSeekV4LowRankPool\n", "pool_switch"),
        ("        buf_groups = [\n"
         "            self.c4_kv_pool.kv_buffer,\n",
         "        if _sg_lr is not None and _sg_lr.lowrank_enabled():\n"
         "            _c4_kv_buffers = self.c4_kv_pool.coeff_buffer\n"
         "        else:\n"
         "            _c4_kv_buffers = self.c4_kv_pool.kv_buffer\n"
         "        buf_groups = [\n"
         "            _c4_kv_buffers,\n", "buf_groups"),
    ])

    # budget: ONLY the c4 compressed-latent fraction swaps 584 -> 200 B/token
    # when low-rank (SWA/full + c128 stay native). This grows full_token ->
    # _compute_dsv4_sizes yields more c4 slots per byte = the memory-ceiling win.
    _apply(POOL_CFG, [
        ("from __future__ import annotations\n", "from __future__ import annotations\n" + _IMPORT_BLOCK, "import"),
        ("            + c4_frac * kv_bytes * self.num_layers_ca4\n",
         "            + c4_frac * (kv_bytes if not (_sg_lr is not None and _sg_lr.lowrank_enabled()) else _sg_lr.BYTES_PER_TOKEN) * self.num_layers_ca4\n", "budget"),
    ])

    _apply(DSV4_BACKEND, [
        ("from __future__ import annotations\n", "from __future__ import annotations\n" + _IMPORT_BLOCK, "import"),
        ("        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)\n"
         "        if is_prefill:\n"
         "            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)\n",
         "        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)\n"
         "        if is_prefill or (_sg_lr is not None and _sg_lr.lowrank_enabled()):\n"
         "            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)\n", "raw_indices_decode"),
        ("            if save_kv_cache:\n"
         "                self.store_cache(layer_id, swa_k, forward_batch)\n",
         "            if save_kv_cache:\n"
         "                self.store_cache(layer_id, swa_k, forward_batch)\n"
         + _BACKEND_DECODE_BRANCH, "decode_branch"),
        ("        if compressed_slice is not None:\n"
         "            dequantize_k_cache_paged(\n"
         "                extra_k_cache,\n"
         "                flat_token_ids,\n"
         "                page_size=extra_page_size,\n"
         "                out=compressed_slice,\n"
         "            )\n",
         "        if compressed_slice is not None:\n"
         "            if (_sg_lr is not None and _sg_lr.lowrank_enabled()\n"
         "                    and compress_ratio == 4):\n"
         "                _sg_lr.dequantize_lowrank_k_cache_paged(\n"
         "                    extra_k_cache, flat_token_ids,\n"
         "                    page_size=extra_page_size,\n"
         "                    layer_id=layer_id, out=compressed_slice,\n"
         "                )\n"
         "            else:\n"
         "                dequantize_k_cache_paged(\n"
         "                    extra_k_cache,\n"
         "                    flat_token_ids,\n"
         "                    page_size=extra_page_size,\n"
         "                    out=compressed_slice,\n"
         "                )\n", "prefill_dequant"),
        ("            if extra_k_cache is not None:\n"
         "                page_sizes = {\n"
         "                    4: token_to_kv_pool.page_size // 4,\n"
         "                    128: token_to_kv_pool.page_size // 128,\n"
         "                }\n"
         "                extra_k_cache = extra_k_cache[\n"
         "                    :, : page_sizes[compress_ratio] * k_cache_total_dim\n"
         "                ].view(\n"
         "                    extra_k_cache.shape[0],\n"
         "                    page_sizes[compress_ratio],\n"
         "                    1,\n"
         "                    k_cache_total_dim,\n"
         "                )\n",
         "            if extra_k_cache is not None:\n"
         "                if _sg_lr is not None and _sg_lr.lowrank_enabled() and compress_ratio == 4:\n"
         "                    # low-rank pool: keep the raw coeff buffer (200 B/token).\n"
         "                    # The sparse recon paths fetch get_extra_key_buffer directly.\n"
         "                    pass\n"
         "                else:\n"
         "                    page_sizes = {\n"
         "                        4: token_to_kv_pool.page_size // 4,\n"
         "                        128: token_to_kv_pool.page_size // 128,\n"
         "                    }\n"
         "                    extra_k_cache = extra_k_cache[\n"
         "                        :, : page_sizes[compress_ratio] * k_cache_total_dim\n"
         "                    ].view(\n"
         "                        extra_k_cache.shape[0],\n"
         "                        page_sizes[compress_ratio],\n"
         "                        1,\n"
         "                        k_cache_total_dim,\n"
         "                    )\n", "view_guard"),
    ])


def unpatch():
    import glob
    for p in (COMPRESSOR_V2, MEM_POOL, POOL_CFG, DSV4_BACKEND):
        bak = p + ".lr.bak"
        if os.path.exists(bak):
            shutil.copy(bak, p)
            print(f"[unpatch] restored {p}")
        else:
            # no backup: try to strip injected markers (best-effort)
            with open(p) as f:
                s = f.read()
            if MARKER in s:
                print(f"[unpatch] WARN {p} has marker but no backup; manual restore needed")
    # restore by stripping injected blocks is fragile; recommend git checkout
    print("[unpatch] note: SGLang sources are git-tracked; `git checkout` on the 4 files fully restores")


def _patch_with_backup():
    for p in (COMPRESSOR_V2, MEM_POOL, POOL_CFG, DSV4_BACKEND):
        if not os.path.exists(p + ".lr.bak"):
            shutil.copy(p, p + ".lr.bak")


def _cmd_patch():
    _patch_with_backup()
    patch()


def _cmd_unpatch():
    for p in (COMPRESSOR_V2, MEM_POOL, POOL_CFG, DSV4_BACKEND):
        bak = p + ".lr.bak"
        if os.path.exists(bak):
            shutil.copy(bak, p)
            print(f"[unpatch] restored {p}")


def _cmd_verify():
    for p in (COMPRESSOR_V2, MEM_POOL, POOL_CFG, DSV4_BACKEND):
        with open(p) as f:
            s = f.read()
        print(f"{p}: marker={'present' if MARKER in s else 'absent'}")


def _cmd_selftest():
    """Numeric check: fused Triton recon vs the eager torch path on cuda:0.

    Run inside the eval container on GPUs 0-3:
      CUDA_VISIBLE_DEVICES=0,1,2,3 \
      PYTHONPATH=/sgl-workspace/sglang-lowrank/python \
      python /mnt/host_root/home/jovyan/winstonxcai/transferibility/xkv_decode/lowrank_store.py selftest
    """
    import time
    torch.manual_seed(0)
    dev = "cuda:0"
    lid = 7
    n, max_len, page_size, n_pages = 1024, 4096, 64, 64
    global _freqs_cis, _freqs_real
    _freqs_cis = None
    _freqs_real = None

    # orthonormal basis Vr [512,192] (columns span range(A))
    Vr, _ = torch.linalg.qr(torch.randn(HEAD_DIM, COEFF_DIM, device=dev))
    _Vr[lid] = Vr.cpu()
    _VrT[lid] = Vr.T.cpu()

    # arbitrary-but-fixed freqs table [max_len, 32] complex
    theta = torch.arange(ROPE_DIM // 2, device=dev).float() * (
        2 * torch.pi / (ROPE_DIM // 2)
    )
    freqs = torch.polar(
        torch.ones(max_len, ROPE_DIM // 2, device=dev),
        theta[None, :] * torch.arange(max_len, device=dev)[:, None],
    )
    _freqs_cis = freqs.detach()

    # random coeffs -> fp8+ue8m0 -> populate a page-addressable pool
    page_bytes = page_size * BYTES_PER_TOKEN
    buf = torch.zeros(n_pages, page_bytes, dtype=torch.uint8, device=dev)
    loc = torch.randint(0, n_pages * page_size, (n,), device=dev)
    coeffs = torch.randn(n, COEFF_DIM, device=dev)
    coeff_fp8, scale_u8 = _quant_ue8m0(coeffs)
    pos = torch.randint(0, max_len, (n,), device=dev, dtype=torch.int32)
    _set_coeff_buffer_torch(buf, loc, coeff_fp8, scale_u8, pos, page_size)

    out_t = torch.zeros(n, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    out_r = torch.zeros(n, 1, HEAD_DIM, dtype=torch.bfloat16, device=dev)

    t0 = time.time()
    _dequantize_lowrank_k_cache_paged_torch(
        buf, loc, page_size=page_size, layer_id=lid, out=out_t)
    torch.cuda.synchronize()
    dt = time.time() - t0

    t0 = time.time()
    _dequantize_lowrank_k_cache_paged_triton(
        buf, loc, page_size=page_size, layer_id=lid, out=out_r)
    torch.cuda.synchronize()
    dr = time.time() - t0

    md = (out_t.float() - out_r.float()).abs().max().item()
    ok = torch.allclose(out_t, out_r, atol=0.01, rtol=0.02)
    print(f"[selftest] n={n} torch={dt * 1e3:.1f}ms triton={dr * 1e3:.1f}ms "
          f"max_abs_diff={md:.6f} allclose(0.01,0.02)={ok}")
    if not ok:
        print("[selftest] FAIL")
        raise SystemExit(1)
    print("[selftest] OK")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"patch": _cmd_patch, "unpatch": _cmd_unpatch, "verify": _cmd_verify,
     "selftest": _cmd_selftest}[cmd]()
