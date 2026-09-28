"""Torch reference operations and basis caches for the low-rank store."""
import os
from typing import Optional

import torch

from . import config

# The coefficient record's fp8 format. E4M3FN, not the fnuz variant triton names
# fp8e4b8: the store kernel compiles a `tl.store` of these values on sm90 and
# triton only accepts fp8e4nv/fp8e4b15/fp8e5 as store types, so fnuz makes
# `_store_kernel` fail to compile outright. Every reader (`reconstruct_torch`,
# `fused_indexer.reconstruct`) views the pool through this same constant, so the
# byte layout stays self-consistent; E4M3FN's wider range (448 vs fnuz's 240)
# also makes `quantize`'s clamp bound the storage format's real maximum.
fp8_dtype = torch.float8_e4m3fn
_basis_dir = ""
_Vr, _VrT, _Vr_dev, _VrT_dev, _VrT_bf16_dev = {}, {}, {}, {}, {}
_freqs_cis = None


def set_basis_dir(path: str) -> None:
    global _basis_dir
    _basis_dir = path


def set_freqs(freqs: Optional[torch.Tensor]) -> None:
    global _freqs_cis
    if freqs is not None:
        _freqs_cis = freqs.detach()


def _load_vr(layer_id: int) -> Optional[torch.Tensor]:
    if layer_id in _Vr:
        return _Vr[layer_id]
    path = os.path.join(_basis_dir, f"A_{layer_id:03d}.pt")
    if not os.path.exists(path):
        return None
    try:
        a = torch.load(path, map_location="cpu").float()
        _, evecs = torch.linalg.eigh((a + a.T) / 2)
        vr = evecs[:, -config.COEFF_DIM:].contiguous()
    except Exception:
        return None
    _Vr[layer_id] = vr
    _VrT[layer_id] = vr.T.contiguous()
    return vr


def vr_for(layer_id: int, device) -> Optional[torch.Tensor]:
    key = (layer_id, str(device))
    if key not in _Vr_dev:
        vr = _load_vr(layer_id)
        if vr is None:
            return None
        _Vr_dev[key] = vr.to(device)
        _VrT_dev[key] = vr.T.contiguous().to(device)
    return _Vr_dev[key]


def vrt_for(layer_id: int, device) -> Optional[torch.Tensor]:
    return _VrT_dev.get((layer_id, str(device))) if vr_for(layer_id, device) is not None else None


def vrt_bf16_for(layer_id: int, device) -> Optional[torch.Tensor]:
    key = (layer_id, str(device))
    if key not in _VrT_bf16_dev:
        vrt = vrt_for(layer_id, device)
        if vrt is None:
            return None
        _VrT_bf16_dev[key] = vrt.to(torch.bfloat16).contiguous()
    return _VrT_bf16_dev[key]


def prewarm(layer_ids, device) -> int:
    """Fault every layer's basis into the device caches up front.

    `vr_for` lazily `torch.load`s from disk and copies to the device on first
    use. The first use is a decode step, which runs inside a cuda graph, so a
    miss there is a host-side load and an allocation in the captured region.
    Returns the number of layers that resolved; missing bases are reported by the
    caller's own skip path, not here.
    """
    return sum(1 for l in layer_ids if vr_for(l, device) is not None)



def quantize(x: torch.Tensor):
    n = x.shape[0]
    info = torch.finfo(fp8_dtype)
    xt = x.float().view(n, config.SCALE_TILES, config.TILE_SIZE)
    maxabs = xt.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    exponent = torch.ceil(torch.log2(maxabs / info.max)).clamp(-127.0, 128.0)
    scaled = (xt / (2.0 ** exponent)).clamp(info.min, info.max)
    return scaled.to(fp8_dtype).reshape(n, config.COEFF_DIM), (exponent + 127).to(torch.uint8).reshape(n, config.SCALE_TILES)


def dequantize(coeff_fp8, scale, n: Optional[int] = None):
    n = coeff_fp8.shape[0] if n is None else n
    c = coeff_fp8.float().view(n, config.SCALE_TILES, config.TILE_SIZE)
    s = (2.0 ** (scale.float() - 127.0)).unsqueeze(-1)
    return (c * s).reshape(n, config.COEFF_DIM)


def store_torch(buf, loc, coeff_fp8, scale_u8, pos, page_size, valid=None):
    """Eager fallback store. `valid` filters rows before the scatter, which is
    only safe off-graph -- the capture path masks inside the triton kernel."""
    if valid is not None:
        keep = valid.to(torch.bool)
        loc, coeff_fp8, scale_u8, pos = loc[keep], coeff_fp8[keep], scale_u8[keep], pos[keep]
        if loc.numel() == 0:
            return
    page_bytes = buf.shape[-1]
    base = (loc // page_size) * page_bytes + (loc % page_size) * config.BYTES_PER_TOKEN
    fp8 = buf.view(fp8_dtype).reshape(-1)
    fp8[base[:, None] + torch.arange(config.COEFF_DIM, device=loc.device)] = coeff_fp8
    flat = buf.reshape(-1)
    flat[base[:, None] + config.COEFF_DIM + torch.arange(config.SCALE_TILES, device=loc.device)] = scale_u8
    buf.view(torch.int32).reshape(-1)[(base + config.COEFF_SCALE_BYTES) // 4] = pos


def apply_rope_tail(recon, freqs_cis, pos):
    n = recon.shape[0]
    fc = torch.view_as_real(freqs_cis[pos.long()])
    tail = recon[:, config.NOPE_DIM:].view(n, config.ROPE_DIM // 2, 2)
    real, imag = tail[..., 0].clone(), tail[..., 1].clone()
    tail[..., 0] = real * fc[..., 0] - imag * fc[..., 1]
    tail[..., 1] = real * fc[..., 1] + imag * fc[..., 0]


def reconstruct_torch(coeff_buf, flat_token_ids, *, page_size, layer_id, out):
    if _freqs_cis is None or not flat_token_ids.numel():
        return
    page_bytes = coeff_buf.shape[-1]
    base = (flat_token_ids // page_size) * page_bytes + (flat_token_ids % page_size) * config.BYTES_PER_TOKEN
    flat = coeff_buf.reshape(-1)
    idx = torch.arange(config.COEFF_DIM, device=flat_token_ids.device)
    coeff = coeff_buf.view(fp8_dtype).reshape(-1)[base[:, None] + idx]
    scales = flat[base[:, None] + config.COEFF_DIM + torch.arange(config.SCALE_TILES, device=flat_token_ids.device)]
    pos = coeff_buf.view(torch.int32).reshape(-1)[(base + config.COEFF_SCALE_BYTES) // 4].clamp(0, _freqs_cis.shape[0] - 1)
    vrt = vrt_for(layer_id, coeff.device)
    if vrt is None:
        return
    recon = dequantize(coeff, scales) @ vrt
    apply_rope_tail(recon, _freqs_cis, pos)
    out.copy_(recon.unsqueeze(1).to(torch.bfloat16))
