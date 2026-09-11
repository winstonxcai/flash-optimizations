"""Direct 328-byte sparse MLA: the c4 attention leg without reassembly.

The production packed path reconstructs each packed row into the 584-byte
FlashMLA-native layout and hands that to ``flash_mla_with_kvcache``
(``unpack_gather_native``, and cuda/fused.cu under ``_FUSED``). This module
instead feeds the packed record straight to ``mustafar._sparse``, which
decompresses fp8 E4M3 + UE8M0 scales in the shared-memory fill, applies the tail
RoPE in-kernel, and runs the QK^T and PV products on the same tile.

The softmax stays on the Python side between the two passes, mirroring
dhjoo98/mustafar's own reference wiring; the log-sum-exp it produces is what
lets the c4 leg be merged with the native SWA+sink leg.
"""

from __future__ import annotations

import importlib
import threading

import torch

from . import config

_extension = None
_load_error: Exception | None = None
_load_lock = threading.Lock()

# log2(e): the kernel and flash_mla both work in natural units; the merge is
# base-2, so the conversion happens once, here.
_LOG2E = 1.4426950408889634


def _load():
    global _extension, _load_error
    if _extension is not None:
        return _extension
    with _load_lock:
        if _extension is not None:
            return _extension
        if _load_error is not None:
            raise RuntimeError(
                "Sparse MLA was requested, but the CUDA extension is unavailable"
            ) from _load_error
        try:
            _extension = importlib.import_module("mustafar._sparse")
        except Exception as exc:
            _load_error = exc
            raise RuntimeError(
                "Sparse MLA was requested, but mustafar._sparse could not "
                "be loaded"
            ) from exc
    return _extension


def sparse_available() -> bool:
    """Whether the extension imports; device support is checked at launch."""
    try:
        _load()
    except RuntimeError:
        return False
    return True


def _as_int32(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype == torch.int32 else t.to(torch.int32)


def _valid_mask(
    physical: torch.Tensor, raw: torch.Tensor, topk_lengths
) -> torch.Tensor:
    """Slots the reference path treats as absent.

    Mirrors ``_unpack_gather_bf16_kernel``'s validity predicate: a slot is real
    only when it is in range, has a physical record and a raw position.
    """
    keep = physical >= 0
    if raw is not None:
        keep = keep & (raw >= 0)
    if topk_lengths is not None:
        topk_lengths = _as_int32(topk_lengths)
        arange = torch.arange(physical.shape[1], device=physical.device)
        keep = keep & (arange[None, :] < topk_lengths[:, None])
    return keep


def _prepare(q, physical, raw, freqs_cis, topk_lengths):
    """Validate the call and normalize indices/frequencies for the kernels.

    Shared by :func:`scores` and :func:`c4_leg` so the per-dimension probe tier
    reads the packed record through exactly the same plumbing as the real leg.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Sparse MLA requires CUDA")
    if q.dtype != torch.bfloat16:
        raise ValueError(f"sparse MLA expects bf16 q, got {q.dtype}")
    if q.shape[-1] != config.HEAD_DIM:
        raise ValueError(f"sparse MLA expects dim {config.HEAD_DIM}, got {q.shape}")
    if physical.shape != raw.shape:
        raise ValueError("physical and raw indices must have equal shape")

    physical = _as_int32(physical).contiguous()
    raw = _as_int32(raw).contiguous()
    lengths = _as_int32(topk_lengths).contiguous() if topk_lengths is not None else None
    if lengths is None:
        lengths = torch.full(
            (physical.shape[0],), physical.shape[1], dtype=torch.int32,
            device=physical.device,
        )
    if not freqs_cis.is_complex() or not freqs_cis.is_contiguous():
        raise ValueError("freqs_cis must be a contiguous complex tensor")
    return physical, raw, lengths, torch.view_as_real(freqs_cis)


def scores(
    q: torch.Tensor,
    values: torch.Tensor,
    bitmaps: torch.Tensor,
    scales: torch.Tensor,
    physical: torch.Tensor,
    raw: torch.Tensor,
    freqs_cis: torch.Tensor,
    sm_scale: float,
    *,
    topk_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Raw scaled QK^T over the packed rows, float32 ``[queries, heads, k]``.

    Unmasked and unnormalized: invalid slots are whatever the kernel computed
    over a zeroed row, so this is the right handle for a per-dimension probe but
    not for attention. :func:`c4_leg` applies the mask and the softmax.
    """
    physical, raw, lengths, freq_pairs = _prepare(
        q, physical, raw, freqs_cis, topk_lengths
    )
    return _load().sparse_scores(
        q.contiguous(), values, bitmaps, scales, physical, raw, lengths,
        freq_pairs, float(sm_scale),
    )


def c4_leg(
    q: torch.Tensor,
    values: torch.Tensor,
    bitmaps: torch.Tensor,
    scales: torch.Tensor,
    physical: torch.Tensor,
    raw: torch.Tensor,
    freqs_cis: torch.Tensor,
    sm_scale: float,
    *,
    topk_lengths: torch.Tensor | None = None,
):
    """Compute the packed c4 attention leg.

    Returns ``(output, lse)`` with ``output`` bf16 ``[queries, heads, 512]`` and
    ``lse`` float32 ``[queries, heads]`` in **base 2**, matching the convention
    ``flash_mla_with_kvcache`` reports so the two legs can be merged.

    ``q`` must be bf16 ``[queries, heads, 512]``. ``bitmaps``/``scales`` are the
    ``[rows, 8]`` packed ABI; ``physical``/``raw`` are ``[queries, selected_k]``.
    """
    # Validate before dlopen so a CUDA-less host fails on the guard rather than
    # on a missing extension.
    physical, raw, lengths, freq_pairs = _prepare(
        q, physical, raw, freqs_cis, topk_lengths
    )
    extension = _load()

    scores = extension.sparse_scores(
        q.contiguous(), values, bitmaps, scales, physical, raw, lengths,
        freq_pairs, float(sm_scale),
    )

    keep = _valid_mask(physical, raw, lengths)
    scores = scores.masked_fill(~keep[:, None, :], float("-inf"))

    # base-2 log-sum-exp; -inf rows collapse to -inf, which merge_lse drops.
    lse = torch.logsumexp(scores, dim=-1) * _LOG2E
    probs = torch.softmax(scores, dim=-1)
    # Zero the invalid slots so they contribute nothing to P . V. softmax
    # already yields 0 for -inf, but bf16 rounding of an all-masked row can
    # produce NaN, so be explicit.
    probs = torch.where(keep[:, None, :], probs, torch.zeros_like(probs))
    probs = probs.to(torch.bfloat16).contiguous()

    out = extension.sparse_output(
        probs, values, bitmaps, scales, physical, raw, lengths, freq_pairs
    )
    return out, lse


def _merge_weights(lse_a: torch.Tensor, lse_b: torch.Tensor):
    """Base-2 split-softmax weights plus the running max, NaN-free.

    An empty leg has lse = -inf. If BOTH are -inf the max is -inf too and
    ``-inf - -inf`` is NaN, so the max is clamped to a finite value first; the
    weights then all collapse to 0 and the caller's denominator guard takes over.
    """
    stack = torch.stack([lse_a, lse_b], dim=0)
    m = torch.max(stack, dim=0).values
    safe_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    weights = torch.exp2(stack - safe_m.unsqueeze(0))
    return weights, torch.where(torch.isfinite(m), m, safe_m)


def merge_lse(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Split-softmax merge of two attention legs, both base-2 log-sum-exp.

    ``attn_sink`` must be counted in exactly one of the two legs; this function
    only combines what it is given.
    """
    if lse_a.shape != lse_b.shape:
        raise ValueError("lse tensors must have equal shape")
    weights, _ = _merge_weights(lse_a, lse_b)
    total = weights.sum(dim=0)
    merged = (
        out_a.float() * weights[0].unsqueeze(-1)
        + out_b.float() * weights[1].unsqueeze(-1)
    )
    merged = merged / total.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
    return merged.to(out_a.dtype)


def merge_lse_lse(lse_a: torch.Tensor, lse_b: torch.Tensor) -> torch.Tensor:
    """Combined base-2 log-sum-exp of two legs, for callers that want it."""
    weights, m = _merge_weights(lse_a, lse_b)
    total = weights.sum(dim=0)
    return m + torch.log2(total.clamp_min(torch.finfo(torch.float32).tiny))


def sparse_forward(
    q: torch.Tensor,
    packed_buffers,
    physical: torch.Tensor,
    raw: torch.Tensor,
    freqs_cis: torch.Tensor,
    sm_scale: float,
    *,
    topk_lengths: torch.Tensor | None = None,
    swa_leg=None,
):
    """Full c4 leg plus an optional native SWA+sink leg, merged.

    ``swa_leg`` is a zero-argument callable returning the native leg's
    ``(output, lse)`` -- injected rather than imported so this module stays free
    of any SGLang dependency. When omitted, only the c4 leg is returned.
    """
    from .packed import PackedBuffers, _as_buffers

    buffers = (
        packed_buffers
        if isinstance(packed_buffers, PackedBuffers)
        else _as_buffers(packed_buffers)
    )
    out_c4, lse_c4 = c4_leg(
        q, buffers.values, buffers.bitmaps, buffers.scales, physical, raw,
        freqs_cis, sm_scale, topk_lengths=topk_lengths,
    )
    if swa_leg is None:
        return out_c4, lse_c4
    out_swa, lse_swa = swa_leg()
    return merge_lse(out_swa, lse_swa, out_c4, lse_c4), merge_lse_lse(lse_swa, lse_c4)
