"""H100/A800 Packed pack and gather/unpack microbenchmark matrix."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import median
from unittest.mock import patch

import torch


class _DecodePlan:
    def __init__(self, rows: torch.Tensor):
        self.rows = rows
        self.is_decode = True

    def __getitem__(self, index: int):
        if index == 1:
            return self.rows.view(torch.uint8)
        raise IndexError(index)


def _timings_us(fn, warmup: int = 20, repeats: int = 100) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return median(samples), samples[min(len(samples) - 1, int(0.95 * len(samples)))]


@patch.dict(
    os.environ,
    SGLANG_OPT_TOPMAG="1",
    KEEP="0.5",
    SGLANG_OPT_TOPMAG_PACKED="1",
    SGLANG_OPT_TOPMAG_FUSED="0",
)
def run_packed_benchmark(output_dir: str | Path | None = None) -> dict[str, object]:

    from .. import reference
    from ..packed import (
        NativeWorkspace,
        PackedBuffers,
        pack_rows,
        unpack_gather_bf16,
        unpack_gather_native,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("bench_packed requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(11)
    batch_sizes = (1, 2, 4, 8, 16, 32, 64)
    selected_counts = (64, 128, 256, 512)
    max_rows = max(batch_sizes) * max(selected_counts)

    latent = torch.randn(max_rows, 512, dtype=torch.bfloat16, device=device)
    mask = reference.topmag_keep_mask(latent, 0.5)
    weight = torch.ones(512, dtype=torch.bfloat16, device=device)
    values = torch.zeros(max_rows, 256, dtype=torch.uint8, device=device)
    bitmaps = torch.zeros(max_rows, 8, dtype=torch.uint64, device=device)
    scales = torch.zeros(max_rows, 8, dtype=torch.uint8, device=device)
    buffers = PackedBuffers(values, bitmaps, scales)
    locations = torch.arange(max_rows, dtype=torch.int32, device=device)
    plan_rows = torch.zeros(max_rows, 4, dtype=torch.int32, device=device)
    plan_rows[:, 0] = 4
    plan = _DecodePlan(plan_rows)
    pack_rows(latent, mask, weight, 1.0e-6, plan, locations, buffers)

    max_position = max(selected_counts) * 4 + 1
    angles = torch.randn(max_position, 32, dtype=torch.float32, device=device)
    freqs = torch.complex(torch.cos(angles), torch.sin(angles)).contiguous()
    rows: list[dict[str, object]] = []

    for batch in batch_sizes:
        local_plan = _DecodePlan(plan_rows[:batch])

        def fn(batch=batch, local_plan=local_plan):
            return pack_rows(
                latent[:batch],
                mask[:batch],
                weight,
                1.0e-6,
                local_plan,
                locations[:batch],
                buffers,
            )

        p50, p95 = _timings_us(fn)
        input_bytes = batch * (512 * 2 + 512 + 512 * 2)
        output_bytes = batch * 328
        rows.append(
            {
                "operation": "pack",
                "batch": batch,
                "selected": 0,
                "p50_us": p50,
                "p95_us": p95,
                "effective_gbps": (input_bytes + output_bytes) / p50 / 1.0e3,
                "per_layer_us": p50,
                "projected_21_layer_us": p50 * 21,
            }
        )

    for batch in batch_sizes:
        for selected in selected_counts:
            count = batch * selected
            physical = locations[:count].reshape(batch, selected)
            raw = torch.arange(selected, dtype=torch.int32, device=device)[None, :]
            raw = raw.expand(batch, -1).contiguous()
            lengths = torch.full((batch,), selected, dtype=torch.int32, device=device)
            output = torch.empty(count, 512, dtype=torch.bfloat16, device=device)

            def bf16_fn(physical=physical, raw=raw, lengths=lengths, output=output):
                return unpack_gather_bf16(
                    buffers, physical, raw, lengths, freqs, output
                )

            p50, p95 = _timings_us(bf16_fn)
            bytes_moved = count * (328 + 512 * 2)
            rows.append(
                {
                    "operation": "unpack_bf16",
                    "batch": batch,
                    "selected": selected,
                    "p50_us": p50,
                    "p95_us": p95,
                    "effective_gbps": bytes_moved / p50 / 1.0e3,
                    "per_layer_us": p50,
                    "projected_21_layer_us": p50 * 21,
                }
            )

            workspace = NativeWorkspace.allocate(batch, selected, 64, device)

            def native_fn(
                physical=physical, raw=raw, lengths=lengths, workspace=workspace
            ):
                return unpack_gather_native(
                    buffers, physical, raw, lengths, freqs, workspace
                )

            p50, p95 = _timings_us(native_fn)
            # Packed read + dense BF16 intermediate + native 584-byte write.
            bytes_moved = count * (328 + 512 * 2 + 584)
            rows.append(
                {
                    "operation": "unpack_native",
                    "batch": batch,
                    "selected": selected,
                    "p50_us": p50,
                    "p95_us": p95,
                    "effective_gbps": bytes_moved / p50 / 1.0e3,
                    "per_layer_us": p50,
                    "projected_21_layer_us": p50 * 21,
                }
            )

    result = {
        "gpu": torch.cuda.get_device_name(),
        "logical_bytes_per_row": 328,
        "native_bytes_per_row": 584,
        "logical_compression": 584 / 328,
        "rows": rows,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "packed-microbench.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        with (output_path / "packed-microbench.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return result


if __name__ == "__main__":
    run_packed_benchmark(os.environ.get("MUSTAFAR_RESULTS_DIR"))
