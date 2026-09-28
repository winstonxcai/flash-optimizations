"""Runtime integration for the W3 CSA low-rank KV store.

Three things run inside the captured decode region and so must be static-shaped:
the coefficient store (`store_compressed_lowrank`), the reconstruction gather
(`dequantize_lowrank_k_cache_paged`), and the sparse-attention call itself
(`decode_lowrank`). Each is written here to build fixed-shape tensors and let a
trivial mask carry the data dependence, rather than compacting rows or deduping
token sets the way the eager prototype did.
"""
import os
from typing import Optional

import torch

from . import calib, config, reference
from .triton import fused_indexer, score_cache

_cur_layer: Optional[int] = None
_freqs_cis = None
_prewarmed = False


def _native_dequantize_k_cache_paged():
    """The stock page-gather dequantizer. Its module moved in SGLang v0.5.18."""
    try:
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
    except ImportError:
        from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
    return dequantize_k_cache_paged


def lowrank_enabled():
    return config.lowrank_enabled()


def set_cur_layer(layer_id):
    """Called once per layer per forward, so it stays O(1) after the first call.

    The first call is also where the basis is pulled onto the device: `vr_for`
    lazily `torch.load`s per layer, and a cache miss on the first captured decode
    step would fault inside graph capture.
    """
    global _cur_layer, _prewarmed
    _cur_layer = layer_id
    if not _prewarmed:
        _prewarmed = True
        if lowrank_enabled():
            reference.prewarm(config.CSA_LAYERS,
                              torch.device("cuda", torch.cuda.current_device()))


def set_basis_dir(path):
    reference.set_basis_dir(path)


def init_decode_workspace(backend, *, max_bs, c4_topk, swa_window):
    """Allocate the decode reconstruction workspace at its full graph capacity.

    Deliberately not `backend.sparse_prefill_workspace`: that one reallocates on
    growth, which is safe only because sparse prefill runs eagerly. Capturing the
    decode path needs an allocation that never moves.
    """
    rows = max(1, max_bs) * (c4_topk + swa_window)
    backend.xkv_decode_workspace = torch.empty(
        (rows, 1, config.HEAD_DIM), dtype=torch.bfloat16, device=backend.device
    )
    backend.xkv_decode_capacity = rows
    _debug("init_decode_workspace", rows=rows, max_bs=max_bs,
           c4_topk=c4_topk, swa_window=swa_window)


def _decode_workspace(backend, rows, device):
    """The decode workspace, sized for `rows`, growing only off-graph.

    `init_decode_workspace` runs from the backend's `init_cuda_graph_state`, so
    it exists only when a decode graph runner was built. With decode graphs
    disabled (the A/B's eager arm, and the calibration legs) the attribute is
    never created and the first decode allocates here instead -- which is safe,
    because eager decode is exactly the regime where a moving buffer is fine.
    """
    workspace = getattr(backend, "xkv_decode_workspace", None)
    if workspace is not None and rows <= workspace.shape[0]:
        return workspace[:rows]
    workspace = torch.empty((rows, 1, config.HEAD_DIM),
                            dtype=torch.bfloat16, device=device)
    backend.xkv_decode_workspace = workspace
    backend.xkv_decode_capacity = rows
    _debug("decode_workspace_lazy", rows=rows)
    return workspace


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
    """Project this step's latent onto the shared basis and write the record.

    Runs inside the captured decode region, so the decode branch computes
    coefficients for EVERY plan row and hands the store a `valid` mask instead of
    dropping rows. Upstream's own compressor does the same thing for the same
    reason: `_forward_unified_hip` zero-masks non-boundary decode rows rather
    than compacting them.
    """
    if calib.enabled():
        calib.maybe_capture(_cur_layer, kv_compressed, plan, norm,
                            compress_ratio, is_indexer)
        return True
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
    if is_decode:
        # Decode is the captured path: keep every row, mask in the kernel.
        # Non-boundary rows carry out_loc 0 upstream, so leaving them unwritten
        # is strictly safer than upstream's zero-write over slot 0.
        assert x.shape[0] == plan_i.shape[0], (
            f"decode store expected one plan row per token, got "
            f"{x.shape[0]} tokens / {plan_i.shape[0]} rows"
        )
        valid = seq_len % compress_ratio == 0
        ragged = torch.arange(seq_len.shape[0], device=seq_len.device)
    else:
        # Prefill runs eagerly (prefill graphs stay disabled), so compaction is
        # free here and keeps the store off the -1-padded rows entirely.
        valid = seq_len != -1
        ragged = col1 & 0xFFFF
        if x.shape[0] == plan_i.shape[0]:
            x, seq_len, ragged = x[valid], seq_len[valid], ragged[valid]
            valid = None
    if not x.shape[0]:
        return True

    loc = out_loc[ragged]
    pos = (seq_len - compress_ratio).to(torch.int32)
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    normed = normed * norm.weight.float()
    coeff_fp8, scale_u8 = reference.quantize(normed @ vr)
    score_cache.store(kv_cache, loc, coeff_fp8, scale_u8, pos, page_size, valid=valid)
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


def decode_lowrank(self, *, q, layer_id, token_to_kv_pool, attn_sink,
                   swa_page_indices, swa_topk_lengths,
                   c4_page_indices, c4_topk_lengths):
    """Sparse attention over a reconstructed, fixed-shape decode workspace.

    Every tensor here has a shape fixed at graph-capture time; only the index
    *values* depend on the batch. That is the whole point: the eager prototype
    deduped the attended token set with `torch.unique` and sized its workspace
    from `int(x.max().item())`, which no capture can record.

    The layout mirrors `_forward_prefill_sparse`: one flat row per attended
    slot, compressed region first, SWA region second, then per-query indices
    rebased into that flat space with `-1` padding beyond `topk_length`.

    Caller contract: `q` is (b, 1, h_q, d_qk) and the return is (b, h_q, d_v),
    matching what `flash_mla_with_kvcache(...).squeeze(1)` would have produced.
    The four index/length tensors are the caller's *matched* locals: the backend
    rebinds those to q's batch dimension via `match_num_queries` without writing
    back to the metadata, so the metadata fields can disagree with `q.shape[0]`.
    """
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    q_flat = q.squeeze(1)
    batch = q_flat.shape[0]
    device = q.device

    swa_idx = swa_page_indices.reshape(batch, -1)
    swa_len = swa_topk_lengths.reshape(batch).to(torch.int32)
    c4_idx = c4_page_indices.reshape(batch, -1)
    c4_len = c4_topk_lengths.reshape(batch).to(torch.int32)

    w_c4, w_swa = c4_idx.shape[1], swa_idx.shape[1]
    n_c4, n_swa = batch * w_c4, batch * w_swa
    rows = n_c4 + n_swa

    workspace = _decode_workspace(self, rows, device)

    # -1-padded slots are clamped to 0 so the gather stays in bounds; the
    # attention masks them out via topk_length. Both dequant paths clamp too.
    dequantize_lowrank_k_cache_paged(
        token_to_kv_pool.get_extra_key_buffer(layer_id),
        c4_idx.reshape(-1).clamp_min(0),
        page_size=token_to_kv_pool.get_extra_key_page_size(layer_id),
        layer_id=layer_id, out=workspace[:n_c4])
    _native_dequantize_k_cache_paged()(
        token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
        swa_idx.reshape(-1).clamp_min(0),
        page_size=token_to_kv_pool.swa_window_size, out=workspace[n_c4:])

    rows_arange = torch.arange(batch, device=device)[:, None]
    c4_cols = torch.where(
        torch.arange(w_c4, device=device)[None, :] < c4_len[:, None],
        rows_arange * w_c4 + torch.arange(w_c4, device=device)[None, :], -1)
    swa_cols = torch.where(
        torch.arange(w_swa, device=device)[None, :] < swa_len[:, None],
        n_c4 + rows_arange * w_swa + torch.arange(w_swa, device=device)[None, :], -1)
    combined = torch.cat([c4_cols, swa_cols], dim=1).unsqueeze(1).to(torch.int32)

    return flash_mla_sparse_fwd(
        q=q_flat, kv=workspace, indices=combined,
        sm_scale=self.softmax_scale, d_v=self.head_dim_v, attn_sink=attn_sink,
        topk_length=c4_len + swa_len)[0]


# Tree patching lives in .patching so the same edit declarations can also be
# censused read-only against a different SGLang revision (patching.drift).
from .patching import patch, unpatch, verify, drift  # noqa: E402,F401
