"""Runtime integration for the STAR-CSA c4 low-rank store.

Two things run inside the captured decode region and so must be static-shaped:
the store (`store_lowrank`) and the gather+reconstruct (`dequantize_lowrank_k_cache_paged`).
Neither compacts rows or dedupes token sets; the mask rides in the data, not in
the shape.

The row->location mapping is the vendor's own, taken from the torch fallback in
``_forward_unified_hip`` (which is unreferenced in v0.5.18 but documents the
live kernel's contract exactly): on decode every row is stored and non-boundary
rows are zeroed onto ``out_loc == 0``; on prefill the plan rows are already
one-per-compress-unit and the write location is ``out_loc[plan[:,1] & 0xFFFF]``.
"""

import json
import os
from typing import Optional

import torch

from . import config, reference
from .analysis import capture

_cur_layer: Optional[int] = None
_prewarmed = False


def starkv_enabled() -> bool:
    return config.starkv_enabled()


def set_basis_dir(path: str) -> None:
    reference.set_basis_dir(path)


def set_cur_layer(layer_id: int) -> None:
    """Called once per layer per forward, so it stays O(1) after the first call.

    The first call is also where the bases are pulled onto the device:
    `basis_for` lazily loads per layer, and a cache miss on the first captured
    decode step would fault inside graph capture.
    """
    global _cur_layer, _prewarmed
    _cur_layer = layer_id
    if not _prewarmed:
        _prewarmed = True
        if config.starkv_enabled():
            reference.set_basis_dir(config.basis_dir())
            device = torch.device("cuda", torch.cuda.current_device())
            resolved = reference.prewarm(config.STARKV_LAYERS, device)
            _debug("prewarm", layers=resolved, expected=len(config.STARKV_LAYERS))


def _debug(msg: str, **fields) -> None:
    if not config.debug_enabled():
        return
    os.makedirs(config.ctrl_dir(), exist_ok=True)
    with open(os.path.join(config.ctrl_dir(), "debug.log"), "a") as f:
        f.write(json.dumps({"starkv": msg, **fields}, default=str) + "\n")


# --- store ------------------------------------------------------------------


def capture_enabled() -> bool:
    return capture.enabled()


def capture_decode_attn(layer_id, *, q, indices, lengths) -> None:
    capture.capture_decode_attn(layer_id, q, indices, lengths)


def _store_inputs(kv_compressed, norm, plan, compress_ratio, out_loc, freqs_cis_cache):
    """The vendor's norm+rope, the write locations and the boundary mask.

    Shared by the compression path and the capture hook so a capture records
    exactly the tensor a store would have written, computed the same way.
    """
    plan_i = plan[1].view(torch.int32)
    seq_len = plan_i[:, 0].long()
    positions = (seq_len - compress_ratio).clamp_min(0).to(torch.int32)
    normed = reference.rms_norm(kv_compressed.detach().float(), norm.weight,
                                norm.variance_epsilon)
    tail = reference.apply_rope(normed, freqs_cis_cache, positions)
    if plan.is_decode:
        loc = out_loc
        boundary = seq_len % compress_ratio == 0
    else:
        # plan_c rows are already one-per-compress-unit, so only the write
        # location is indirect; no row is dropped.
        loc = out_loc[(plan_i[:, 1].long() & 0xFFFF)]
        boundary = torch.ones_like(seq_len, dtype=torch.bool)
    return normed, tail, loc, positions, boundary


def store_lowrank(
    kv_compressed,
    plan,
    norm,
    compress_ratio,
    is_indexer,
    kv_cache,
    page_size,
    out_loc,
    freqs_cis_cache,
    bf16_store=False,
):
    """Project this step's latent onto the layer basis and write the record.

    Returns True when the store is ours (so the caller must not fall through to
    the native writer, which would put a 584-byte record into a compact pool).
    Declining is only ever safe for a target we do not own: the indexer pool,
    the c128 ratio, and the HIP-only unified-KV path are all left alone.
    """
    if is_indexer or compress_ratio != 4 or bf16_store:
        return False
    if capture.enabled():
        # Capture mode: record the native sample and let the native store run.
        # This wins over the compression path on purpose -- the whole point is to
        # measure the compression, so nothing may be recorded through it.
        #
        # Decode only, deliberately: the attention hook samples decode steps, and
        # the join is on the c4 slot, so a prefill row here would both dilute the
        # sample and (because a chunked prefill writes thousands of rows at once)
        # close the store's window long before the attention's.
        if _cur_layer is None or kv_cache is None or not plan.is_decode:
            return False
        normed, tail, loc, positions, boundary = _store_inputs(
            kv_compressed, norm, plan, compress_ratio, out_loc, freqs_cis_cache
        )
        capture.capture_store(_cur_layer, normed, tail, loc, positions, boundary)
        return False
    if not config.starkv_enabled():
        return False
    if _cur_layer is None or kv_cache is None:
        # Enabled, but we cannot place this write. Skipping is the only safe
        # option: a native record in this pool would be read as a low-rank one.
        _debug("store_skip_no_layer", layer=_cur_layer)
        return True

    d = reference.basis_for(_cur_layer, kv_compressed.device)
    if d is None:
        _debug("store_skip_no_basis", layer=_cur_layer)
        return True

    normed, tail, loc, positions, boundary = _store_inputs(
        kv_compressed, norm, plan, compress_ratio, out_loc, freqs_cis_cache
    )
    z = normed[:, : config.NOPE_DIM] @ d
    records = reference.pack_record(z, tail)
    if plan.is_decode:
        # Non-boundary decode rows carry slot 0; zero them the way the vendor
        # does rather than dropping them, which keeps the op fixed-shape.
        records = torch.where(boundary[:, None], records, records.new_zeros(()))

    reference.store_records(kv_cache, loc, records, page_size)
    if config.debug_enabled():
        _telemetry(_cur_layer, normed, z, d, rows=normed.shape[0])
    return True


def _telemetry(layer, normed, z, d, rows) -> None:
    """One summary line per store call: the retention curve, not a per-token log.

    A per-token line here would grow at hundreds of lines/s and throttle decode,
    which is the opposite of what the instrument is for. Both figures are
    reported because they fail differently: `basis` is what the rank costs,
    `quant` is what the fp8 record costs on top of it.
    """
    nope = normed[:, : config.NOPE_DIM]
    den = nope.pow(2).sum(-1).clamp_min(1e-12)
    codes, scales = reference.quantize(z)
    z_q = reference.dequantize(codes, scales, config.RANK)
    basis = reference.reconstruct(z, d).pow(2).sum(-1) / den
    quant = reference.reconstruct(z_q, d).pow(2).sum(-1) / den
    entry = {
        "layer": layer,
        "rows": int(rows),
        "rank": config.RANK,
        "basis_mean": float(basis.mean()),
        "basis_min": float(basis.min()),
        "quant_mean": float(quant.mean()),
        "quant_min": float(quant.min()),
    }
    os.makedirs(config.ctrl_dir(), exist_ok=True)
    with open(os.path.join(config.ctrl_dir(), "retention.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


# --- read -------------------------------------------------------------------


def _native_dequantize_k_cache_paged():
    """The native page-gather dequantizer. Its module moved in v0.5.18."""
    try:
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
    except ImportError:
        from sglang.srt.layers.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
    return dequantize_k_cache_paged


def dequantize_lowrank_k_cache_paged(
    rec_buf, flat_token_ids, *, page_size, layer_id, out
):
    """Gather compact records and rebuild [D z | tail] into `out`.

    `out` is the (rows, 1, HEAD_DIM) bf16 workspace slot the sparse kernel
    reads, exactly as the native dequantizer fills it.
    """
    if not config.recon_mode():
        raise NotImplementedError(
            "the latent read path is kernel work; see starkv/triton/. "
            "Use STARKV_RECON=1 for the reconstruct MVE."
        )
    if not flat_token_ids.numel():
        return
    d = reference.basis_for(layer_id, flat_token_ids.device)
    if d is None:
        raise RuntimeError(f"[starkv] no basis for layer {layer_id}")
    records = reference.gather_records(rec_buf, flat_token_ids, page_size)
    out.copy_(
        reference.reconstruct_records(records, d).to(torch.bfloat16).unsqueeze(1)
    )


def init_decode_workspace(backend, *, max_bs, c4_topk, swa_window):
    """Allocate the decode workspace at its full graph capacity.

    Deliberately not `backend.sparse_prefill_workspace`: that one reallocates on
    growth, which is safe only because sparse prefill runs eagerly. Capturing the
    decode path needs an allocation that never moves.
    """
    rows = max(1, max_bs) * (c4_topk + swa_window)
    backend.starkv_decode_workspace = torch.empty(
        (rows, 1, config.HEAD_DIM), dtype=torch.bfloat16, device=backend.device
    )
    backend.starkv_decode_capacity = rows
    _debug("init_decode_workspace", rows=rows, max_bs=max_bs, c4_topk=c4_topk,
           swa_window=swa_window)


def _decode_workspace(backend, rows, device):
    """The decode workspace, sized for `rows`, growing only off-graph.

    `init_decode_workspace` runs from the backend's `init_cuda_graph_state`, so
    it exists only when a decode graph runner was built. With decode graphs
    disabled (the eager legs, and the calibration/capture runs) the attribute is
    never created and the first decode allocates here instead -- which is safe,
    because eager decode is exactly the regime where a moving buffer is fine.
    """
    workspace = getattr(backend, "starkv_decode_workspace", None)
    if workspace is not None and rows <= workspace.shape[0]:
        return workspace[:rows]
    workspace = torch.empty(
        (rows, 1, config.HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    backend.starkv_decode_workspace = workspace
    backend.starkv_decode_capacity = rows
    _debug("decode_workspace_lazy", rows=rows)
    return workspace


def decode_lowrank(
    self,
    *,
    q,
    layer_id,
    token_to_kv_pool,
    attn_sink,
    swa_page_indices,
    swa_topk_lengths,
    c4_page_indices,
    c4_topk_lengths,
):
    """Sparse attention over a reconstructed, fixed-shape workspace.

    Every tensor here has a shape fixed at graph-capture time; only the index
    *values* depend on the batch. That is the point: the eager prototype deduped
    the attended token set with `torch.unique` and sized its workspace from
    `int(x.max().item())`, which no capture can record.

    The layout mirrors `_forward_prefill_sparse`: one flat row per attended slot,
    c4 region first, SWA region second, then per-query indices rebased into that
    flat space with `-1` padding beyond `topk_length`.

    Caller contract: `q` is (b, 1, h_q, d_qk) and the return is (b, h_q, d_v),
    matching what `flash_mla_with_kvcache(...).squeeze(1)` would have produced.
    The four index/length tensors are the caller's *matched* locals: the backend
    rebinds those to q's batch dimension via `match_num_queries` without writing
    back to the metadata, so the metadata fields can disagree with q.shape[0].
    """
    if not config.recon_mode():
        raise NotImplementedError(
            "the latent read path is kernel work; see starkv/triton/. "
            "Use STARKV_RECON=1 for the reconstruct MVE."
        )
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

    # -1-padded slots are clamped to 0 so the gather stays in bounds; attention
    # masks them out via topk_length. The native dequantizer clamps too.
    dequantize_lowrank_k_cache_paged(
        token_to_kv_pool.get_extra_key_buffer(layer_id),
        c4_idx.reshape(-1).clamp_min(0),
        page_size=token_to_kv_pool.get_extra_key_page_size(layer_id),
        layer_id=layer_id,
        out=workspace[:n_c4],
    )
    _native_dequantize_k_cache_paged()(
        token_to_kv_pool.get_swa_key_buffer_radix(layer_id),
        swa_idx.reshape(-1).clamp_min(0),
        page_size=token_to_kv_pool.swa_window_size,
        out=workspace[n_c4:],
    )

    rows_arange = torch.arange(batch, device=device)[:, None]
    c4_cols = torch.where(
        torch.arange(w_c4, device=device)[None, :] < c4_len[:, None],
        rows_arange * w_c4 + torch.arange(w_c4, device=device)[None, :],
        -1,
    )
    swa_cols = torch.where(
        torch.arange(w_swa, device=device)[None, :] < swa_len[:, None],
        n_c4 + rows_arange * w_swa + torch.arange(w_swa, device=device)[None, :],
        -1,
    )
    combined = torch.cat([c4_cols, swa_cols], dim=1).unsqueeze(1).to(torch.int32)

    return flash_mla_sparse_fwd(
        q=q_flat,
        kv=workspace,
        indices=combined,
        sm_scale=self.softmax_scale,
        d_v=self.head_dim_v,
        attn_sink=attn_sink,
        topk_length=c4_len + swa_len,
    )[0]


# Tree patching lives in .patching so the same edit declarations can also be
# censused read-only against a different SGLang revision (patching.drift).
from .patching import drift, patch, unpatch, verify  # noqa: E402,F401
