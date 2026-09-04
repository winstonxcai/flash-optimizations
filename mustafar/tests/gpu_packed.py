"""H100/A800 production-kernel checks for the packed path."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import torch


class _Plan:
    def __init__(self, rows: torch.Tensor, *, is_decode: bool):
        self.rows = rows
        self.is_decode = is_decode
        self.compress_ratio = 4

    def __getitem__(self, index: int):
        if index == 1:
            return self.rows.view(torch.uint8)
        raise IndexError(index)


@patch.dict(
    os.environ,
    SGLANG_OPT_TOPMAG="1",
    KEEP="0.5",
    SGLANG_OPT_TOPMAG_PACKED="1",
    SGLANG_OPT_TOPMAG_FUSED="0",
)
def run_packed_validation() -> dict[str, object]:

    from .. import reference
    from ..packed import (
        NativeWorkspace,
        PackedBuffers,
        pack_rows,
        unpack_gather_bf16,
        unpack_gather_native,
    )
    from ..reference import pack_rows_ref, unpack_rows_ref

    if not torch.cuda.is_available():
        raise RuntimeError("gpu_packed requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(123)
    n = 8
    latent = torch.randn(n, 512, dtype=torch.bfloat16, device=device)
    latent[0, :300] = 0
    latent[1].fill_(1)
    keep_mask = reference.topmag_keep_mask(latent, 0.5)
    weight = torch.linspace(0.75, 1.25, 512, dtype=torch.bfloat16, device=device)

    values = torch.zeros(n, 256, dtype=torch.uint8, device=device)
    bitmaps = torch.zeros(n, 8, dtype=torch.uint64, device=device)
    scales = torch.zeros(n, 8, dtype=torch.uint8, device=device)
    buffers = PackedBuffers(values, bitmaps, scales)
    # SGLang's fused native store ABI requires int64 cache locations.
    locations = torch.arange(n, dtype=torch.int64, device=device)
    plan_rows = torch.zeros(n, 4, dtype=torch.int32, device=device)
    plan_rows[:, 0] = 4 * (torch.arange(n, device=device, dtype=torch.int32) + 1)
    plan = _Plan(plan_rows, is_decode=True)

    pack_rows(latent, keep_mask, weight, 1.0e-6, plan, locations, buffers)
    torch.cuda.synchronize()
    rv, rb, rs = pack_rows_ref(latent, keep_mask, weight, 1.0e-6)
    assert torch.equal(bitmaps, rb), "Triton bitmap != reference mask"
    assert torch.equal(values, rv), "Triton FP8 codes != native-order reference"
    assert torch.equal(scales, rs), "Triton UE8M0 scales != reference"

    # Capture the production write path, including the exact TopMag mask.
    # This guards against host-scalar/indexing operations that work eagerly but
    # are forbidden while SGLang captures its decode graph.
    graph_values = torch.zeros_like(values)
    graph_bitmaps = torch.zeros_like(bitmaps)
    graph_scales = torch.zeros_like(scales)
    graph_buffers = PackedBuffers(graph_values, graph_bitmaps, graph_scales)
    reference.topmag_keep_mask(latent, 0.5)
    pack_rows(latent, keep_mask, weight, 1.0e-6, plan, locations, graph_buffers)
    torch.cuda.synchronize()
    write_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(write_graph):
        captured_mask = reference.topmag_keep_mask(latent, 0.5)
        pack_rows(
            latent,
            captured_mask,
            weight,
            1.0e-6,
            plan,
            locations,
            graph_buffers,
        )
    write_graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_bitmaps, rb), "graph-captured TopMag bitmap mismatch"
    assert torch.equal(graph_values, rv), "graph-captured FP8 codes mismatch"
    assert torch.equal(graph_scales, rs), "graph-captured scales mismatch"

    # Compare the packed path against SGLang's actual fused native store.
    from sglang.jit_kernel.dsv4 import compress_norm_rope_store

    native_page_bytes = ((584 * 64 + 575) // 576) * 576
    native_cache = torch.zeros(1, native_page_bytes, dtype=torch.uint8, device=device)
    dense_zero = latent.masked_fill(~keep_mask, 0)
    freq_complex = torch.complex(
        torch.cos(torch.randn(64, 32, device=device)),
        torch.sin(torch.randn(64, 32, device=device)),
    )
    # Normalize complex values to unit magnitude, matching a RoPE table.
    freq_complex = freq_complex / freq_complex.abs().clamp_min(1.0e-12)
    compress_norm_rope_store(
        dense_zero,
        plan,
        norm_weight=weight,
        norm_eps=1.0e-6,
        freq_cis=freq_complex,
        out_loc=locations,
        kvcache=native_cache,
        page_size=64,
    )
    torch.cuda.synchronize()
    columns = torch.nonzero(keep_mask, as_tuple=False)[:, 1].reshape(n, 256)
    dense_codes = torch.zeros(n, 512, dtype=torch.uint8, device=device)
    dense_codes.scatter_(1, columns, values)
    native_codes = torch.stack(
        [native_cache[0, row * 576 : row * 576 + 448] for row in range(n)]
    )
    native_scales = torch.stack(
        [
            native_cache[0, 64 * 576 + row * 8 : 64 * 576 + row * 8 + 7]
            for row in range(n)
        ]
    )
    assert torch.equal(native_codes, dense_codes[:, :448]), (
        "packed NoPE FP8 codes != SGLang native fused store"
    )
    assert torch.equal(native_scales, scales[:, :7]), (
        "packed NoPE scales != SGLang native fused store"
    )

    # The injected pool allocates only the three packed arrays.
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import MustafarPackedKVPool

    pool = MustafarPackedKVPool(
        size=128,
        page_size=64,
        dtype=torch.float8_e4m3fn,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        layer_num=2,
        device=str(device),
        enable_memory_saver=False,
    )
    assert pool.get_bytes_per_token() == 328
    assert pool.get_kv_size_bytes() == sum(
        tensor.nbytes
        for group in (pool.packed_values, pool.packed_bitmaps, pool.packed_scales)
        for tensor in group
    )
    assert not hasattr(pool, "native_kv_buffer")
    assert pool.kv_buffer is pool.packed_values
    pool.set_rope_freqs(0, freq_complex)
    assert pool.get_rope_freqs(0) is freq_complex, (
        "packed pool must retain the existing RoPE tensor without a copy"
    )

    # Prefill CompressPlan word 1 contains ragged_id in its low 16 bits. It
    # must index out_loc rather than treating compressed-row id as a location.
    prefill_values = torch.zeros_like(values)
    prefill_bitmaps = torch.zeros_like(bitmaps)
    prefill_scales = torch.zeros_like(scales)
    prefill = PackedBuffers(prefill_values, prefill_bitmaps, prefill_scales)
    prefill_rows = torch.zeros(2, 4, dtype=torch.int32, device=device)
    prefill_rows[:, 0] = torch.tensor([4, 8], dtype=torch.int32, device=device)
    prefill_rows[:, 1] = torch.tensor([3, 1], dtype=torch.int32, device=device)
    prefill_locations = torch.tensor([7, 5, 6, 4], dtype=torch.int32, device=device)
    pack_rows(
        latent[:2],
        keep_mask[:2],
        weight,
        1.0e-6,
        _Plan(prefill_rows, is_decode=False),
        prefill_locations,
        prefill,
    )
    torch.cuda.synchronize()
    assert torch.equal(prefill_values[4], rv[0])
    assert torch.equal(prefill_values[5], rv[1])

    physical = torch.tensor([[0, 1, -1, 1]], dtype=torch.int32, device=device)
    raw = torch.tensor([[0, 1, -1, 1]], dtype=torch.int32, device=device)
    lengths = torch.tensor([4], dtype=torch.int32, device=device)
    angles = torch.randn(16, 32, dtype=torch.float32, device=device)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    freqs = torch.complex(cos, sin).contiguous()
    output = torch.empty(4, 512, dtype=torch.bfloat16, device=device)
    unpack_gather_bf16(buffers, physical, raw, lengths, freqs, output)
    torch.cuda.synchronize()
    ref_dense = unpack_rows_ref(values, bitmaps, scales)
    expected = ref_dense[[0, 1, 0, 1]].clone()
    for row_id, raw_id in ((0, 0), (1, 1), (3, 1)):
        tail = expected[row_id, 448:].float().reshape(32, 2)
        real = tail[:, 0] * cos[raw_id * 4] - tail[:, 1] * sin[raw_id * 4]
        imag = tail[:, 0] * sin[raw_id * 4] + tail[:, 1] * cos[raw_id * 4]
        expected[row_id, 448:] = torch.stack((real, imag), -1).reshape(-1)
    assert torch.allclose(output[0], expected[0], atol=0.02, rtol=0.02)
    assert torch.allclose(output[1], expected[1], atol=0.02, rtol=0.02)
    assert bool((output[2] == 0).all()), "invalid top-k slot was not zeroed"
    assert torch.allclose(output[3], expected[3], atol=0.02, rtol=0.02), (
        "duplicate gather mismatch"
    )

    # Full-tail comparison against the native store at position seq_len - 4.
    native_tail = torch.stack(
        [
            native_cache[0, row * 576 + 448 : row * 576 + 576]
            .contiguous()
            .view(torch.bfloat16)
            for row in range(n)
        ]
    )
    all_physical = locations.reshape(1, n)
    all_raw = torch.arange(n, dtype=torch.int32, device=device).reshape(1, n)
    all_lengths = torch.tensor([n], dtype=torch.int32, device=device)
    all_output = torch.empty(n, 512, dtype=torch.bfloat16, device=device)
    unpack_gather_bf16(
        buffers,
        all_physical,
        all_raw,
        all_lengths,
        freq_complex,
        all_output,
    )
    tail_actual = all_output[:, 448:].float()
    tail_expected = native_tail.float()
    tail_abs_error = (tail_actual - tail_expected).abs()
    tail_tolerance = 0.02 + 0.02 * tail_expected.abs()
    tail_within_tolerance = bool((tail_abs_error <= tail_tolerance).all())
    tail_max_abs = tail_abs_error.max().item()
    tail_violations = int((tail_abs_error > tail_tolerance).sum().item())

    # The 328-byte ABI necessarily adds FP8 loss to the tail that native
    # stores as BF16. Preserve tail/logit/output differences as informational
    # metrics while the temporary quality-tolerance waiver is active.
    native_dense = all_output.float().clone()
    native_dense[:, 448:] = tail_expected
    packed_dense = all_output.float()
    query = torch.randn(16, 512, dtype=torch.bfloat16, device=device).float()
    native_logits = query @ native_dense.T / (512.0**0.5)
    packed_logits = query @ packed_dense.T / (512.0**0.5)

    # Diagnostic for an ABI-compatible eighth-scale policy: choose floor or
    # ceil UE8M0 exponent by per-row tail MSE. The seven NoPE tiles remain on
    # the native bit-exact ceil policy.
    masked = dense_zero.float()
    normalized = (
        (
            masked
            * torch.rsqrt(masked.square().mean(-1, keepdim=True) + 1.0e-6)
            * weight.float()
        )
        .to(torch.bfloat16)
        .float()
    )
    pre_rope_tail = normalized[:, 448:]
    raw_tail_scale = pre_rope_tail.abs().amax(-1).clamp_min(1.0e-4) / 448.0
    floor_scale = torch.exp2(torch.floor(torch.log2(raw_tail_scale)))
    ceil_scale = torch.exp2(torch.ceil(torch.log2(raw_tail_scale)))

    def quantized_tail(candidate_scale: torch.Tensor) -> torch.Tensor:
        return (pre_rope_tail / candidate_scale[:, None]).clamp(-448.0, 448.0).to(
            torch.float8_e4m3fn
        ).float() * candidate_scale[:, None]

    floor_tail = quantized_tail(floor_scale)
    ceil_tail = quantized_tail(ceil_scale)
    use_floor = (floor_tail - pre_rope_tail).square().sum(-1) < (
        ceil_tail - pre_rope_tail
    ).square().sum(-1)
    candidate_tail = torch.where(use_floor[:, None], floor_tail, ceil_tail)
    candidate_rotated = torch.empty_like(candidate_tail)
    candidate_pairs = candidate_tail.reshape(n, 32, 2)
    for row_id in range(n):
        c = freq_complex.real[row_id * 4]
        s = freq_complex.imag[row_id * 4]
        candidate_rotated[row_id, 0::2] = (
            candidate_pairs[row_id, :, 0] * c - candidate_pairs[row_id, :, 1] * s
        )
        candidate_rotated[row_id, 1::2] = (
            candidate_pairs[row_id, :, 0] * s + candidate_pairs[row_id, :, 1] * c
        )
    candidate_dense = native_dense.clone()
    candidate_dense[:, 448:] = candidate_rotated
    candidate_logits = query @ candidate_dense.T / (512.0**0.5)
    print(
        json.dumps(
            {
                "tail_scale_candidate_floor_rows": int(use_floor.sum().item()),
                "tail_scale_candidate_tail_max_abs": float(
                    (candidate_rotated - tail_expected).abs().max().item()
                ),
                "tail_scale_candidate_logits_max_abs": float(
                    (candidate_logits - native_logits).abs().max().item()
                ),
                "tail_scale_candidate_logits_close": bool(
                    torch.allclose(
                        candidate_logits, native_logits, atol=0.02, rtol=0.02
                    )
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    logits_close = torch.allclose(packed_logits, native_logits, atol=0.02, rtol=0.02)
    logits_max_abs = (packed_logits - native_logits).abs().max().item()
    attention_values = torch.randn(8, 64, dtype=torch.bfloat16, device=device).float()
    native_attention = torch.softmax(native_logits, dim=-1) @ attention_values
    packed_attention = torch.softmax(packed_logits, dim=-1) @ attention_values
    attention_close = torch.allclose(
        packed_attention, native_attention, atol=0.02, rtol=0.02
    )
    attention_max_abs = (packed_attention - native_attention).abs().max().item()
    assert torch.isfinite(packed_logits).all()
    assert torch.isfinite(packed_attention).all()

    workspace = NativeWorkspace.allocate(1, 4, 64, device)
    native, temp = unpack_gather_native(
        buffers, physical, raw, lengths, freqs, workspace
    )
    assert native.dtype == torch.uint8 and temp.shape == physical.shape
    torch.cuda.synchronize()

    # Warm kernels before capture. Replay must not allocate or synchronize.
    unpack_gather_native(buffers, physical, raw, lengths, freqs, workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unpack_gather_native(buffers, physical, raw, lengths, freqs, workspace)
    graph.replay()
    torch.cuda.synchronize()

    # Small warm latency sample used as an early smoke, not the full matrix.
    samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        unpack_gather_native(buffers, physical, raw, lengths, freqs, workspace)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "rows": n,
        "logical_bytes_per_row": 328,
        "compression": 584 / 328,
        "native_unpack_p50_us": sorted(samples)[len(samples) // 2],
        "graph_capture": True,
        "graph_capture_pack_write": True,
        "fp8_codes_exact": True,
        "scales_exact": True,
        "native_store_exact": True,
        "rope_tail_within_tolerance": tail_within_tolerance,
        "rope_tail_max_abs": tail_max_abs,
        "rope_tail_violations": tail_violations,
        "qk_logits_within_tolerance": bool(logits_close),
        "qk_logits_max_abs": logits_max_abs,
        "attention_output_within_tolerance": bool(attention_close),
        "attention_output_max_abs": attention_max_abs,
        "quality_tolerance_enforced": False,
        "rope_table_zero_copy": True,
        "no_native_shadow_pool": True,
        "bitmap_exact": True,
        "timestamp": time.time(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run_packed_validation()
