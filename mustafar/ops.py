"""Store-time TopMag pruning hook + patch CLI (native c4-latent scope).

This module is imported by the sglang hook as `import mustafar as _sg_lr`.
The injected call `_sg_lr.maybe_prune(kv_compressed)` zeroes the smallest-|.|
(1 - XKV_TOPMAG_KEEP) fraction of each c4 latent vector in place, right before
the native fused store. Everything else (memory pool, layout, decode) is the
stock DeepSeek-V4 build — no lowrank KV anywhere.
"""
import json
import os
import shutil
import subprocess
from typing import Dict

import torch

from . import config
from . import reference
from .pool import PackedC4Pool
from .triton import pack_c4_rows, unpack_gather_c4


_packed_pools: Dict[int, PackedC4Pool] = {}
_freqs_by_layer: Dict[int, torch.Tensor] = {}


def _dbg(msg: str, **fields) -> None:
    if os.environ.get("XKV_DEBUG") == "1":
        line = json.dumps({"mustafar": msg, **fields}, default=str)
        os.makedirs(config.ctrl_dir(), exist_ok=True)
        with open(os.path.join(config.ctrl_dir(), "debug.log"), "a") as f:
            f.write(line + "\n")


_dbg("import", src=__file__, head_dim=config.HEAD_DIM,
     keep=config.topmag_keep())


def packed_enabled() -> bool:
    config.validate_packed_config()
    return config.packed_enabled()


def register_packed_pool(
    packed_values: torch.Tensor,
    bitmap: torch.Tensor,
    packed_scales: torch.Tensor,
) -> PackedC4Pool:
    pool = PackedC4Pool(packed_values, bitmap, packed_scales)
    pool.validate()
    _packed_pools[packed_values.data_ptr()] = pool
    return pool


def get_packed_pool(packed_values: torch.Tensor) -> PackedC4Pool:
    try:
        return _packed_pools[packed_values.data_ptr()]
    except KeyError as exc:
        raise KeyError(
            "C4 packed_values tensor was not registered with mustafar"
        ) from exc


def packed_bytes_per_row() -> int:
    return config.PACKED_BYTES_PER_ROW


def make_keep_mask(kv_compressed: torch.Tensor) -> torch.Tensor:
    return reference.topmag_keep_mask(kv_compressed, config.topmag_keep())


def maybe_prune(kv_compressed) -> None:
    """Hook injected into compressor_v2 just before the native c4 fused store.

    Zeroes the smallest-|.| (1 - XKV_TOPMAG_KEEP) fraction of each latent
    vector in place, gated on SGLANG_OPT_TOPMAG=1. The native store then
    quantizes+writes the pruned latent exactly as usual.
    """
    if not config.topmag_enabled():
        return
    if kv_compressed is None or kv_compressed.numel() == 0:
        return
    if kv_compressed.shape[-1] < config.HEAD_DIM:
        # Non-c4 latent (e.g. the 128-dim indexer) — leave untouched.
        _dbg("prune_skip", dim=int(kv_compressed.shape[-1]))
        return
    try:
        reference.topmag_zero(kv_compressed, config.topmag_keep())
        if os.environ.get("XKV_DEBUG") == "1":
            _dbg("prune", rows=int(kv_compressed.shape[0]),
                 zeroed=config.topmag_zero_count())
    except Exception as e:
        _dbg("prune_error", err=repr(e),
             dim=int(kv_compressed.shape[-1]),
             ndim=kv_compressed.dim())


def set_layer_context(layer_id: int, norm=None, freqs_cis=None) -> None:
    if norm is not None:
        norm._mustafar_layer_id = int(layer_id)
    if freqs_cis is not None:
        _freqs_by_layer[int(layer_id)] = freqs_cis


def store_packed_c4(
    kv_compressed,
    plan,
    norm,
    compress_ratio,
    is_indexer,
    kv_cache,
    out_loc,
    freqs_cis_cache,
) -> bool:
    """Graph-safe replacement for the native C4 norm/RoPE/store operation."""
    if not packed_enabled() or is_indexer or compress_ratio != config.C4_RATIO:
        return False
    if kv_compressed is None or kv_compressed.numel() == 0:
        return True
    if kv_compressed.shape[-1] != config.HEAD_DIM:
        raise ValueError("packed C4 store received a non-512-dimensional latent")
    pool = get_packed_pool(kv_cache)
    plan_i = plan[1].view(torch.int32)
    seq_len = plan_i[:, 0].long()
    column = plan_i[:, 1].long()
    is_decode = bool(getattr(plan, "is_decode", False))
    if is_decode:
        valid = seq_len.remainder(config.C4_RATIO) == 0
        ragged = torch.arange(seq_len.shape[0], device=seq_len.device)
    else:
        valid = seq_len != -1
        ragged = column & 0xFFFF
    locations = out_loc[ragged].to(torch.int64)
    locations = torch.where(valid, locations, torch.full_like(locations, -1))
    keep_mask = make_keep_mask(kv_compressed)
    pack_c4_rows(
        kv_compressed,
        keep_mask,
        norm.weight,
        norm.variance_epsilon,
        locations,
        pool,
    )
    layer_id = getattr(norm, "_mustafar_layer_id", None)
    if layer_id is not None:
        _freqs_by_layer[int(layer_id)] = freqs_cis_cache
    _dbg(
        "pack",
        rows=int(kv_compressed.shape[0]),
        bytes=config.PACKED_BYTES_PER_ROW,
    )
    return True


def unpack_packed_c4(
    packed_values,
    physical_indices,
    raw_indices,
    *,
    layer_id,
    out,
) -> None:
    freqs = _freqs_by_layer.get(int(layer_id))
    if freqs is None:
        raise RuntimeError(f"missing C4 RoPE table for layer {layer_id}")
    unpack_gather_c4(
        get_packed_pool(packed_values), physical_indices, raw_indices, freqs, out
    )


def decode_packed(
    self,
    *,
    q,
    layer_id,
    token_to_kv_pool,
    core_attn_metadata,
    attn_sink,
):
    """Gather to a fixed workspace, then invoke native sparse attention."""
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    q_flat = q.squeeze(1)
    batch = q_flat.shape[0]
    swa_indices = core_attn_metadata.swa_page_indices
    c4_indices = core_attn_metadata.c4_sparse_page_indices
    raw_indices = core_attn_metadata.c4_sparse_raw_indices
    if raw_indices is None:
        raise RuntimeError("packed decode requires c4_sparse_raw_indices")
    if swa_indices.ndim == 3:
        swa_indices = swa_indices.squeeze(1)
    if c4_indices.ndim == 3:
        c4_indices = c4_indices.squeeze(1)
    swa_indices = swa_indices[:batch]
    c4_indices = c4_indices[:batch, : config.C4_TOPK]
    raw_indices = raw_indices[:batch, : config.C4_TOPK]
    swa_len = core_attn_metadata.swa_topk_lengths[:batch].long()
    c4_len = core_attn_metadata.c4_sparse_topk_lengths[:batch].long()
    swa_width = swa_indices.shape[-1]
    width = swa_width + config.C4_TOPK
    workspace = self.sparse_prefill_workspace.get(batch * width)
    workspace_view = workspace[: batch * width].view(
        batch, width, config.HEAD_DIM
    )
    swa_out = workspace_view[:, :swa_width].reshape(
        -1, 1, config.HEAD_DIM
    )
    dequantize_k_cache_paged(
        token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
        swa_indices.clamp_min(0).reshape(-1),
        page_size=token_to_kv_pool.swa_page_size,
        out=swa_out,
    )
    unpack_packed_c4(
        token_to_kv_pool.get_extra_key_buffer(layer_id),
        c4_indices,
        raw_indices,
        layer_id=layer_id,
        out=workspace_view[:, swa_width:],
    )
    positions = torch.arange(width, device=q.device, dtype=torch.int32)[None, :]
    base = torch.arange(batch, device=q.device, dtype=torch.int32)[:, None] * width
    c4_slot = positions - swa_len[:, None]
    combined = torch.where(
        positions < swa_len[:, None],
        base + positions,
        base + swa_width + c4_slot,
    )
    attended = (swa_len + c4_len).to(torch.int32)
    combined = torch.where(
        positions < attended[:, None], combined, torch.full_like(combined, -1)
    )
    return flash_mla_sparse_fwd(
        q=q_flat,
        kv=workspace[: batch * width],
        indices=combined.unsqueeze(1),
        sm_scale=self.softmax_scale,
        d_v=self.head_dim_v,
        attn_sink=attn_sink,
        topk_length=attended,
    )[0]


# --- patch / unpatch / verify --------------------------------------------------
def _apply(path: str, edits) -> None:
    with open(path) as f:
        s = f.read()
    if config.MARKER in s:
        print(f"[mustafar] {path}: already patched")
        return
    for anchor, new in edits:
        if s.count(anchor) != 1:
            raise AssertionError(
                f"[mustafar] anchor count != 1 in {path}: {anchor[:70]!r}")
        s = s.replace(anchor, new, 1)
    with open(path, "w") as f:
        f.write(s)
    print(f"[mustafar] {path}: ok")


def _import_block() -> str:
    return ("\n" + config.MARKER + " (import)\n"
            "import sys as _sg_lr_sys\n"
            f"if {config.PACKAGE_ROOT!r} not in _sg_lr_sys.path:\n"
            f"    _sg_lr_sys.path.insert(0, {config.PACKAGE_ROOT!r})\n"
            "import mustafar as _sg_lr\n")


def patch() -> None:
    os.makedirs(config.ctrl_dir(), exist_ok=True)
    stale = [p for p in config.OLD_PATCH_FILES
             if "XKV_LOWRANK" in open(p).read()]
    if stale:
        raise SystemExit(
            f"[mustafar] old XKV_LOWRANK patch still present in {stale}; "
            "run 'python -m mustafar unpatch' first")

    _apply(config.COMPRESSOR_V2, [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        ("        # Step 2: norm + rope + store\n"
         "        compress_norm_rope_store(\n",
         "        " + config.MARKER + " (topmag)\n"
         "        _sg_lr.maybe_prune(kv_compressed)\n"
         "        # Step 2: norm + rope + store\n"
         "        compress_norm_rope_store(\n"),
    ])


def unpatch() -> None:
    """Restore all 4 previously-patched files to pristine, clearing the old
    lowrank injection (transferibility's `## XKV_LOWRANK`)."""
    for p in config.OLD_PATCH_FILES:
        bak = p + ".lr.bak"
        if os.path.exists(bak):
            shutil.copy(bak, p)
            print(f"[mustafar] restored {p} from .lr.bak")
        else:
            subprocess.run(["git", "checkout", "--", p], cwd=config.SRC_ROOT,
                           check=False)
            print(f"[mustafar] git checkout {p}")


def verify() -> None:
    for p in config.OLD_PATCH_FILES:
        with open(p) as f:
            s = f.read()
        print(f"{p}: mustafar_markers={s.count(config.MARKER)} "
              f"old_lowrank={'PRESENT' if 'XKV_LOWRANK' in s else 'absent'}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"patch": patch, "unpatch": unpatch, "verify": verify}[cmd]()
