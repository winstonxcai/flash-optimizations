"""Runtime integration for the W3 CSA low-rank KV store."""
import os
import shutil
from typing import Optional

import torch

from . import config
from . import reference
from .triton import fused_indexer, score_cache

_cur_layer: Optional[int] = None
_freqs_cis = None


def lowrank_enabled():
    return config.lowrank_enabled()


def set_cur_layer(layer_id):
    global _cur_layer
    _cur_layer = layer_id


def set_basis_dir(path):
    reference.set_basis_dir(path)


def _debug(msg, **fields):
    if os.environ.get("XKV_DEBUG") != "1":
        return
    import json
    os.makedirs(config.ctrl_dir(), exist_ok=True)
    with open(os.path.join(config.ctrl_dir(), "debug.log"), "a") as f:
        f.write(json.dumps({"lowrank": msg, **fields}, default=str) + "\n")


def _set_freqs(freqs):
    global _freqs_cis
    if freqs is not None:
        _freqs_cis = freqs.detach()
        reference.set_freqs(_freqs_cis)


def store_compressed_lowrank(kv_compressed, plan, norm, compress_ratio,
                             is_indexer, kv_cache, page_size, out_loc,
                             freqs_cis_cache):
    if not lowrank_enabled() or is_indexer or compress_ratio != 4:
        return False
    if _cur_layer is None:
        return True
    if not reference._basis_dir:
        reference.set_basis_dir(config.basis_dir())
    _set_freqs(freqs_cis_cache)
    x = kv_compressed.detach().float()
    vr = reference.vr_for(_cur_layer, x.device)
    if vr is None:
        _debug("store_skip_no_basis", layer=_cur_layer)
        return True
    plan_i = plan[1].view(torch.int32)
    seq_len, col1 = plan_i[:, 0].long(), plan_i[:, 1].long()
    is_decode = bool(getattr(plan, "is_decode", False))
    valid = seq_len % compress_ratio == 0 if is_decode else seq_len != -1
    ragged = torch.arange(seq_len.shape[0], device=seq_len.device) if is_decode else col1 & 0xFFFF
    if x.shape[0] == plan_i.shape[0]:
        x, seq_len, ragged = x[valid], seq_len[valid], ragged[valid]
    if not x.shape[0]:
        return True
    loc = out_loc[ragged]
    pos = (seq_len - compress_ratio).to(torch.int32)
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    normed = normed * norm.weight.float()
    coeff_fp8, scale_u8 = reference.quantize(normed @ vr)
    score_cache.store(kv_cache, loc, coeff_fp8, scale_u8, pos, page_size)
    return True


def dequantize_lowrank_k_cache_paged(coeff_buf, flat_token_ids, *, page_size,
                                     layer_id, out):
    if _freqs_cis is None or not flat_token_ids.numel():
        return
    if os.environ.get("XKV_RECON_TRITON", "1") == "1" and flat_token_ids.numel() >= 16:
        fused_indexer.reconstruct(coeff_buf, flat_token_ids, page_size=page_size,
                                  layer_id=layer_id, out=out, freqs_cis=_freqs_cis)
    else:
        reference.reconstruct_torch(coeff_buf, flat_token_ids, page_size=page_size,
                                    layer_id=layer_id, out=out)


def decode_lowrank(self, *, q, layer_id, forward_batch, token_to_kv_pool,
                   core_attn_metadata, attn_sink):
    """Reconstruct unique SWA/CSA locations into the sparse-attention workspace."""
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from sglang.srt.layers.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged

    q_flat = q.squeeze(1)
    batch = q_flat.shape[0]
    swa_idx = core_attn_metadata.swa_page_indices
    swa_len = core_attn_metadata.swa_topk_lengths
    c4_idx = core_attn_metadata.c4_sparse_page_indices
    c4_len = core_attn_metadata.c4_sparse_topk_lengths

    def match(x, value):
        if x.shape[0] == batch:
            return x
        if x.shape[0] > batch:
            return x[:batch]
        return torch.nn.functional.pad(x, (0, 0, 0, batch - x.shape[0]), value=value)

    swa_idx, swa_len = match(swa_idx, 0), match(swa_len, 1)
    c4_idx, c4_len = match(c4_idx, -1), match(c4_len, 1)
    if swa_idx.ndim == 2:
        swa_idx = swa_idx.unsqueeze(1)
    if c4_idx.ndim == 2:
        c4_idx = c4_idx.unsqueeze(1)
    swa = swa_idx.squeeze(1)
    c4 = c4_idx.reshape(batch, -1)
    swa_len, c4_len = swa_len.long(), c4_len.long()
    attended = (swa_len + c4_len).to(torch.int32)
    max_att = int(attended.max().item()) if batch else 0
    if max_att == 0:
        return torch.zeros_like(q_flat)
    width = max(128, ((max_att + 127) // 128) * 128)
    swa_ok = (torch.arange(swa.shape[1], device=q.device)[None, :] < swa_len[:, None]) & (swa >= 0)
    c4_ok = (torch.arange(c4.shape[1], device=q.device)[None, :] < c4_len[:, None]) & (c4 >= 0)
    offset = 1 << 40
    encoded = torch.cat([swa.long().reshape(-1), c4.long().reshape(-1) + offset])
    valid = torch.cat([swa_ok.reshape(-1), c4_ok.reshape(-1)])
    unique, inverse = torch.unique(encoded[valid], return_inverse=True)
    workspace = self.sparse_prefill_workspace.get(unique.shape[0])
    n_swa = int((unique < offset).sum().item())
    swa_u = unique[:n_swa].int()
    c4_u = (unique[n_swa:] - offset).int()
    if n_swa:
        dequantize_k_cache_paged(token_to_kv_pool.get_swa_key_buffer_radix(layer_id), swa_u,
                                 page_size=token_to_kv_pool.swa_page_size, out=workspace[:n_swa])
    if c4_u.numel():
        dequantize_lowrank_k_cache_paged(token_to_kv_pool.get_extra_key_buffer(layer_id), c4_u,
            page_size=token_to_kv_pool.get_extra_key_page_size(layer_id), layer_id=layer_id,
            out=workspace[n_swa:])
    inv_map = torch.full((encoded.numel(),), -1, dtype=torch.int32, device=q.device)
    inv_map[valid] = inverse.int()
    combined = torch.full((batch, width), -1, dtype=torch.int32, device=q.device)
    sw = inv_map[:batch * swa.shape[1]].reshape(batch, -1)
    c4w = inv_map[batch * swa.shape[1]:].reshape(batch, -1)
    rows = torch.arange(batch, device=q.device)[:, None]
    sm = torch.arange(swa.shape[1], device=q.device)[None, :] < swa_len[:, None]
    cm = torch.arange(c4.shape[1], device=q.device)[None, :] < c4_len[:, None]
    combined[rows.expand_as(sw)[sm], torch.arange(swa.shape[1], device=q.device)[None, :].expand_as(sw)[sm]] = sw[sm]
    ccols = swa_len[:, None] + torch.arange(c4.shape[1], device=q.device)[None, :]
    combined[rows.expand_as(c4w)[cm], ccols.expand_as(c4w)[cm]] = c4w[cm]
    return flash_mla_sparse_fwd(q=q_flat, kv=workspace, indices=combined.unsqueeze(1),
        sm_scale=self.softmax_scale, d_v=self.head_dim_v, attn_sink=attn_sink,
        topk_length=attended)[0]


def _apply(path, edits):
    with open(path) as f:
        text = f.read()
    if config.MARKER in text:
        return
    for anchor, replacement in edits:
        if text.count(anchor) != 1:
            raise AssertionError(f"patch anchor count for {path}: {text.count(anchor)}")
        text = text.replace(anchor, replacement, 1)
    with open(path, "w") as f:
        f.write(text)


def _import_block():
    return ("\n" + config.MARKER + " (import)\n"
            "import sys as _sg_lr_sys\n"
            f"if {config.PACKAGE_ROOT!r} not in _sg_lr_sys.path:\n"
            f"    _sg_lr_sys.path.insert(0, {config.PACKAGE_ROOT!r})\n"
            "try:\n    import xkv as _sg_lr\nexcept Exception:\n    _sg_lr = None\n")


def patch():
    os.makedirs(config.ctrl_dir(), exist_ok=True)
    _apply(config.COMPRESSOR_V2, [
        ("from __future__ import annotations\n", "from __future__ import annotations\n" + _import_block()),
        ("        if forward_batch.forward_mode.is_idle():\n            return\n",
         "        if forward_batch.forward_mode.is_idle():\n            return\n        if _sg_lr is not None:\n            _sg_lr.set_cur_layer(layer_id)\n"),
        ("        # Step 2: norm + rope + store\n        compress_norm_rope_store(\n",
         "        if _sg_lr is not None and _sg_lr.store_compressed_lowrank(\n            kv_compressed, plan, norm, compress_ratio, is_indexer, kv_cache, page_size, out_loc, freqs_cis_cache):\n            return\n        # Step 2: norm + rope + store\n        compress_norm_rope_store(\n")])
    _apply(config.MEM_POOL, [
        ("from __future__ import annotations\n", "from __future__ import annotations\n" + _import_block()),
        ("class DeepSeekV4IndexerPool(KVCache):\n", _pool_class() + "\n\nclass DeepSeekV4IndexerPool(KVCache):\n"),
        ("            c4_kv_pool_type = DeepSeekV4SingleKVPool\n", "            c4_kv_pool_type = DeepSeekV4SingleKVPool\n            if _sg_lr is not None and _sg_lr.lowrank_enabled():\n                c4_kv_pool_type = DeepSeekV4LowRankPool\n"),
        ("        buf_groups = [\n            self.c4_kv_pool.kv_buffer,\n", "        _c4_buffers = (self.c4_kv_pool.coeff_buffer if _sg_lr is not None and _sg_lr.lowrank_enabled() else self.c4_kv_pool.kv_buffer)\n        buf_groups = [\n            _c4_buffers,\n"),
    ])
    _apply(config.POOL_CFG, [
        ("from __future__ import annotations\n", "from __future__ import annotations\n" + _import_block()),
        ("            + c4_frac * kv_bytes * self.num_layers_ca4\n", "            + c4_frac * (kv_bytes if _sg_lr is None or not _sg_lr.lowrank_enabled() else _sg_lr.BYTES_PER_TOKEN) * self.num_layers_ca4\n")])
    _apply(config.DSV4_BACKEND, [
        ("from __future__ import annotations\n", "from __future__ import annotations\n" + _import_block()),
        ("            if save_kv_cache:\n                self.store_cache(layer_id, swa_k, forward_batch)\n",
         "            if save_kv_cache:\n                self.store_cache(layer_id, swa_k, forward_batch)\n            if (_sg_lr is not None and _sg_lr.lowrank_enabled() and compress_ratio == 4 and forward_batch.forward_mode.is_decode()):\n                return _sg_lr.decode_lowrank(self, q=q, layer_id=layer_id, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool, core_attn_metadata=core_attn_metadata, attn_sink=attn_sink)\n"),
        ("        if compressed_slice is not None:\n            dequantize_k_cache_paged(\n                extra_k_cache,\n                flat_token_ids,\n                page_size=extra_page_size,\n                out=compressed_slice,\n            )\n",
         "        if compressed_slice is not None:\n            if (_sg_lr is not None and _sg_lr.lowrank_enabled() and compress_ratio == 4):\n                _sg_lr.dequantize_lowrank_k_cache_paged(extra_k_cache, flat_token_ids, page_size=extra_page_size, layer_id=layer_id, out=compressed_slice)\n            else:\n                dequantize_k_cache_paged(extra_k_cache, flat_token_ids, page_size=extra_page_size, out=compressed_slice)\n"),
        ("        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)\n        if is_prefill:\n            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)\n",
         "        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)\n        if is_prefill or (_sg_lr is not None and _sg_lr.lowrank_enabled()):\n            self.c4_sparse_raw_indices = torch.empty_like(self.c4_sparse_page_indices)\n"),
        ("            if extra_k_cache is not None:\n                page_sizes = {\n                    4: token_to_kv_pool.page_size // 4,\n                    128: token_to_kv_pool.page_size // 128,\n                }\n                extra_k_cache = extra_k_cache[\n                    :, : page_sizes[compress_ratio] * k_cache_total_dim\n                ].view(\n                    extra_k_cache.shape[0],\n                    page_sizes[compress_ratio],\n                    1,\n                    k_cache_total_dim,\n                )\n",
         "            if extra_k_cache is not None and not (_sg_lr is not None and _sg_lr.lowrank_enabled() and compress_ratio == 4):\n                page_sizes = {\n                    4: token_to_kv_pool.page_size // 4,\n                    128: token_to_kv_pool.page_size // 128,\n                }\n                extra_k_cache = extra_k_cache[:, : page_sizes[compress_ratio] * k_cache_total_dim].view(extra_k_cache.shape[0], page_sizes[compress_ratio], 1, k_cache_total_dim)\n")])


def _pool_class():
    return '''class DeepSeekV4LowRankPool(KVCache):
    coeff_buffer_dtype = torch.uint8
    def __init__(self, size, page_size, dtype, qk_nope_head_dim, qk_rope_head_dim, layer_num, device, enable_memory_saver, start_layer=None, end_layer=None):
        super().__init__(size, page_size, dtype, layer_num, device, enable_memory_saver, start_layer, end_layer)
        self._create_buffer()
    def get_bytes_per_token(self):
        return _sg_lr.BYTES_PER_TOKEN
    def _create_buffer(self):
        page_bytes = self.page_size * self.get_bytes_per_token()
        self.coeff_buffer = [torch.zeros((self.size + self.page_size + 1) // self.page_size, page_bytes, dtype=torch.uint8, device=self.device) for _ in range(self.layer_num)]
    def get_key_buffer(self, layer_id): return self.coeff_buffer[layer_id]
    def get_kv_buffer(self, *args, **kwargs): raise NotImplementedError()
    def get_value_buffer(self, *args, **kwargs): raise NotImplementedError()
    def set_kv_buffer(self, *args, **kwargs): raise NotImplementedError()
'''


def unpatch():
    for path in (config.COMPRESSOR_V2, config.MEM_POOL, config.POOL_CFG, config.DSV4_BACKEND):
        backup = path + ".lr.bak"
        if os.path.exists(backup):
            shutil.copy(backup, path)


def verify():
    return {path: config.MARKER in open(path).read() for path in (config.COMPRESSOR_V2, config.MEM_POOL, config.POOL_CFG, config.DSV4_BACKEND)}
