"""Synthetic Fused correctness, stream, and CUDA-graph tests."""

from __future__ import annotations

import argparse
import json
import os
from unittest.mock import patch

import torch

from .. import reference
from ..packed import (
    NativeWorkspace,
    PackedBuffers,
    unpack_gather_native,
    unpack_gather_native_fused,
)
from ..reference import pack_rows_ref


def _native_rows(
    workspace: NativeWorkspace, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    scales = []
    for row in range(rows):
        page, offset = divmod(row, workspace.page_size)
        values.append(
            workspace.native_bytes[
                page,
                offset * 576 : offset * 576 + 576,
            ]
        )
        scales.append(
            workspace.native_bytes[
                page,
                workspace.page_size * 576 + offset * 8 : workspace.page_size * 576
                + offset * 8
                + 8,
            ]
        )
    if not values:
        device = workspace.native_bytes.device
        return (
            torch.empty((0, 576), dtype=torch.uint8, device=device),
            torch.empty((0, 8), dtype=torch.uint8, device=device),
        )
    return torch.stack(values), torch.stack(scales)


def _make_buffers(pool_rows: int, device: torch.device) -> PackedBuffers:
    latent = torch.randn(pool_rows, 512, dtype=torch.bfloat16, device=device)
    masks = reference.topmag_keep_mask(latent, 0.5)

    # A deliberately uneven row: empty/full tiles and nonuniform partial tiles.
    counts = (0, 64, 1, 63, 32, 32, 64, 0)
    special = torch.zeros(512, dtype=torch.bool, device=device)
    for tile, count in enumerate(counts):
        special[tile * 64 : tile * 64 + count] = True
    assert int(special.sum()) == 256
    masks[0] = special
    weight = torch.linspace(0.75, 1.25, 512, dtype=torch.bfloat16, device=device)
    values, bitmaps, scales = pack_rows_ref(latent, masks, weight, 1.0e-6)
    return PackedBuffers(values, bitmaps, scales)


def _run_shape(batch: int, selected_k: int = 512) -> dict[str, object]:
    device = torch.device("cuda")
    # Keep the synthetic pool page-major even for the reduced sanitizer case.
    minimum_pool_rows = batch * selected_k + 64
    pool_rows = ((minimum_pool_rows + 63) // 64) * 64
    flat_buffers = _make_buffers(pool_rows, device)
    # Production SGLang pools are page-major, while the kernel consumes their
    # contiguous storage as logical rows. Exercise that exact runtime ABI.
    buffers = PackedBuffers(
        flat_buffers.values.reshape(-1, 64, 256),
        flat_buffers.bitmaps.reshape(-1, 64, 8),
        flat_buffers.scales.reshape(-1, 64, 8),
    )
    physical = torch.arange(batch * selected_k, dtype=torch.int32, device=device)
    physical = physical.reshape(batch, selected_k) % pool_rows
    raw = torch.arange(selected_k, dtype=torch.int32, device=device)[None, :]
    raw = raw.expand(batch, -1).contiguous()
    lengths = torch.full((batch,), selected_k, dtype=torch.int32, device=device)
    if batch > 1:
        lengths[0] = 0
        lengths[1] = 257
    angles = torch.randn(4096, 32, dtype=torch.float32, device=device)
    freqs = torch.complex(torch.cos(angles), torch.sin(angles)).contiguous()

    packed = NativeWorkspace.allocate(batch, selected_k, 61, device, with_dense=True)
    fused = NativeWorkspace.allocate(batch, selected_k, 61, device, with_dense=False)
    unpack_gather_native(buffers, physical, raw, lengths, freqs, packed)
    unpack_gather_native_fused(buffers, physical, raw, lengths, freqs, fused)
    torch.cuda.synchronize()
    rows = batch * selected_k
    native1, scales1 = _native_rows(packed, rows)
    native2, scales2 = _native_rows(fused, rows)

    valid = (
        torch.arange(selected_k, device=device)[None, :] < lengths[:, None]
    ).reshape(-1)
    assert torch.equal(native2[valid, :448], native1[valid, :448])
    assert torch.equal(scales2[valid, :7], scales1[valid, :7])
    tail1 = native1[valid, 448:].contiguous().view(torch.bfloat16).float()
    tail2 = native2[valid, 448:].contiguous().view(torch.bfloat16).float()
    assert torch.allclose(tail2, tail1, atol=0.02, rtol=0.02)
    assert bool((native2[~valid] == 0).all())
    assert bool((scales2[~valid] == 0).all())
    assert bool((scales2[:, 7] == 0).all())

    # Negative, stale physical, stale raw, and duplicate locations.
    edge_physical = torch.tensor(
        [[0, 0, -1, pool_rows, 1, 2]], dtype=torch.int32, device=device
    )
    edge_raw = torch.tensor(
        [[0, 511, 1, 2, -1, 4096]], dtype=torch.int32, device=device
    )
    edge_lengths = torch.tensor([6], dtype=torch.int32, device=device)
    edge = NativeWorkspace.allocate(1, 6, 4, device, with_dense=False)
    edge.native_bytes.fill_(0xA5)
    unpack_gather_native_fused(
        buffers, edge_physical, edge_raw, edge_lengths, freqs, edge
    )
    torch.cuda.synchronize()
    edge_values, edge_scales = _native_rows(edge, 6)
    assert bool((edge_values[2:] == 0).all())
    assert bool((edge_scales[2:] == 0).all())
    assert bool((edge_values[:2] != 0).any())

    # A non-default stream must own the launch.
    stream_workspace = NativeWorkspace.allocate(
        batch, selected_k, 61, device, with_dense=False
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream_workspace.native_bytes.fill_(0xCC)
        unpack_gather_native_fused(
            buffers, physical, raw, lengths, freqs, stream_workspace
        )
    torch.cuda.current_stream().wait_stream(stream)
    stream_values, stream_scales = _native_rows(stream_workspace, rows)
    assert torch.equal(stream_values, native2)
    assert torch.equal(stream_scales, scales2)

    # Warm once before capture so module loading and marker output stay outside.
    unpack_gather_native_fused(buffers, physical, raw, lengths, freqs, fused)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unpack_gather_native_fused(buffers, physical, raw, lengths, freqs, fused)
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated(device)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated(device)
    assert allocated_after == allocated_before
    graph_values, graph_scales = _native_rows(fused, rows)
    assert torch.equal(graph_values, native2)
    assert torch.equal(graph_scales, scales2)

    return {
        "batch": batch,
        "selected_k": selected_k,
        "rows": rows,
        "valid_rows": int(valid.sum().item()),
        "graph_replay": True,
        "replay_allocation_bytes": allocated_after - allocated_before,
    }


# The reference leg must stay Triton regardless of the caller's serving mode.
@patch.dict(
    os.environ,
    SGLANG_OPT_TOPMAG="1",
    KEEP="0.5",
    SGLANG_OPT_TOPMAG_PACKED="1",
    SGLANG_OPT_TOPMAG_FUSED="0",
)
def run_fused_validation(sanitizer_case: bool = False) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("gpu_fused requires CUDA")
    torch.manual_seed(2026)
    batches = (1,) if sanitizer_case else (1, 2, 4, 8, 16)
    shapes = [_run_shape(batch, 8 if sanitizer_case else 512) for batch in batches]
    result = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "sanitizer_case": sanitizer_case,
        "nope_and_scales_exact": True,
        "rope_tolerance": {"atol": 0.02, "rtol": 0.02},
        "invalid_rows_zero": True,
        "non_default_stream": True,
        "shapes": shapes,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanitizer-case", action="store_true")
    args = parser.parse_args()
    run_fused_validation(args.sanitizer_case)
