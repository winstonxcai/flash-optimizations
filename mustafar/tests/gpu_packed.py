"""H100/A800 production-kernel checks for the packed C4 Stage-1 path."""

from __future__ import annotations

import json
import os
import time

import torch


class _Plan:
    def __init__(self, rows: torch.Tensor, *, is_decode: bool):
        self.rows = rows
        self.is_decode = is_decode

    def __getitem__(self, index: int):
        if index == 1:
            return self.rows.view(torch.uint8)
        raise IndexError(index)


def run() -> dict[str, object]:
    os.environ.setdefault("SGLANG_OPT_TOPMAG", "1")
    os.environ.setdefault("XKV_TOPMAG_KEEP", "0.5")
    os.environ.setdefault("SGLANG_OPT_TOPMAG_PACKED_C4", "1")

    from .. import config, reference
    from ..packed_c4 import (
        NativeC4Workspace,
        PackedC4Buffers,
        pack_c4_rows,
        pack_c4_rows_ref,
        unpack_c4_rows_ref,
        unpack_gather_c4_bf16,
        unpack_gather_c4_native,
    )

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
    buffers = PackedC4Buffers(values, bitmaps, scales)
    locations = torch.arange(n, dtype=torch.int32, device=device)
    plan_rows = torch.zeros(n, 4, dtype=torch.int32, device=device)
    plan_rows[:, 0] = 4 * (torch.arange(n, device=device, dtype=torch.int32) + 1)
    plan = _Plan(plan_rows, is_decode=True)

    pack_c4_rows(latent, keep_mask, weight, 1.0e-6, plan, locations, buffers)
    torch.cuda.synchronize()
    rv, rb, rs = pack_c4_rows_ref(latent, keep_mask, weight, 1.0e-6)
    assert torch.equal(bitmaps, rb), "Triton bitmap != reference mask"
    assert torch.equal(values, rv), "Triton FP8 codes != native-order reference"
    assert torch.equal(scales, rs), "Triton UE8M0 scales != reference"

    # Prefill CompressPlan word 1 contains ragged_id in its low 16 bits. It
    # must index out_loc rather than treating compressed-row id as a location.
    prefill_values = torch.zeros_like(values)
    prefill_bitmaps = torch.zeros_like(bitmaps)
    prefill_scales = torch.zeros_like(scales)
    prefill = PackedC4Buffers(prefill_values, prefill_bitmaps, prefill_scales)
    prefill_rows = torch.zeros(2, 4, dtype=torch.int32, device=device)
    prefill_rows[:, 0] = torch.tensor([4, 8], dtype=torch.int32, device=device)
    prefill_rows[:, 1] = torch.tensor([3, 1], dtype=torch.int32, device=device)
    prefill_locations = torch.tensor(
        [7, 5, 6, 4], dtype=torch.int32, device=device
    )
    pack_c4_rows(
        latent[:2], keep_mask[:2], weight, 1.0e-6,
        _Plan(prefill_rows, is_decode=False), prefill_locations, prefill,
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
    output = torch.empty(4, 512, dtype=torch.bfloat16, device=device)
    unpack_gather_c4_bf16(buffers, physical, raw, lengths, (cos, sin), output)
    torch.cuda.synchronize()
    ref_dense = unpack_c4_rows_ref(values, bitmaps, scales)
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

    workspace = NativeC4Workspace.allocate(1, 4, 64, device)
    native, temp = unpack_gather_c4_native(
        buffers, physical, raw, lengths, (cos, sin), workspace
    )
    assert native.dtype == torch.uint8 and temp.shape == physical.shape
    torch.cuda.synchronize()

    # Warm kernels before capture. Replay must not allocate or synchronize.
    unpack_gather_c4_native(buffers, physical, raw, lengths, (cos, sin), workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unpack_gather_c4_native(
            buffers, physical, raw, lengths, (cos, sin), workspace
        )
    graph.replay()
    torch.cuda.synchronize()

    # Small warm latency sample used as an early smoke, not the full matrix.
    samples = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        unpack_gather_c4_native(
            buffers, physical, raw, lengths, (cos, sin), workspace
        )
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
        "fp8_codes_exact": True,
        "scales_exact": True,
        "bitmap_exact": True,
        "timestamp": time.time(),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    run()
