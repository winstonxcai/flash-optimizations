"""Packed Triton versus Fused CUDA reconstruction microbenchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median
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


def _samples(fn, *, repeats: int = 500) -> list[float]:
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))


def _summary(samples: list[float]) -> dict[str, float]:
    return {
        "p50_us": median(samples),
        "p95_us": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
    }


# Pin the dispatcher to Triton; the fused leg calls the CUDA adapter directly.
# Restore the caller's environment even if the benchmark fails.
@patch.dict(
    os.environ,
    SGLANG_OPT_TOPMAG="1",
    KEEP="0.5",
    SGLANG_OPT_TOPMAG_PACKED="1",
    SGLANG_OPT_TOPMAG_FUSED="0",
)
def run_fused_benchmark(output_dir: str | Path | None = None) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("bench_fused requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(2027)
    selected_k = 512
    max_batch = 16
    pool_rows = max_batch * selected_k
    latent = torch.randn(pool_rows, 512, dtype=torch.bfloat16, device=device)
    mask = reference.topmag_keep_mask(latent, 0.5)
    weight = torch.ones(512, dtype=torch.bfloat16, device=device)
    buffers = PackedBuffers(*pack_rows_ref(latent, mask, weight, 1.0e-6))
    angles = torch.randn(4096, 32, dtype=torch.float32, device=device)
    freqs = torch.complex(torch.cos(angles), torch.sin(angles)).contiguous()
    all_rows: list[dict[str, object]] = []

    for batch in (1, 2, 4, 8, 16):
        physical = torch.arange(
            batch * selected_k, dtype=torch.int32, device=device
        ).reshape(batch, selected_k)
        raw = torch.arange(selected_k, dtype=torch.int32, device=device)[None]
        raw = raw.expand(batch, -1).contiguous()
        lengths = torch.full((batch,), selected_k, dtype=torch.int32, device=device)
        packed = NativeWorkspace.allocate(
            batch, selected_k, 64, device, with_dense=True
        )
        fused = NativeWorkspace.allocate(
            batch, selected_k, 64, device, with_dense=False
        )
        eager_fns = {
            "packed": lambda physical=physical, raw=raw, lengths=lengths, packed=packed: (
                unpack_gather_native(buffers, physical, raw, lengths, freqs, packed)
            ),
            "fused": lambda physical=physical, raw=raw, lengths=lengths, fused=fused: (
                unpack_gather_native_fused(
                    buffers, physical, raw, lengths, freqs, fused
                )
            ),
        }
        graph_fns = {}
        for mode, fn in eager_fns.items():
            fn()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graph_fns[mode] = graph.replay

        for round_id in range(3):
            for execution, functions in (("eager", eager_fns), ("graph", graph_fns)):
                for mode, fn in functions.items():
                    all_rows.append(
                        {
                            "batch": batch,
                            "selected_k": selected_k,
                            "round": round_id + 1,
                            "execution": execution,
                            "mode": mode,
                            **_summary(_samples(fn)),
                        }
                    )

    b1_graph = [
        row for row in all_rows if row["batch"] == 1 and row["execution"] == "graph"
    ]
    packed_medians = [row["p50_us"] for row in b1_graph if row["mode"] == "packed"]
    fused_medians = [row["p50_us"] for row in b1_graph if row["mode"] == "fused"]
    repeatable_speedup = all(
        new < old for new, old in zip(fused_medians, packed_medians)
    )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "selected_k": selected_k,
        "samples_per_round": 500,
        "rounds": 3,
        "batch1_graph_packed_median_us": median(packed_medians),
        "batch1_graph_fused_median_us": median(fused_medians),
        "batch1_graph_speedup": median(packed_medians) / median(fused_medians),
        "performance_gate_passed": repeatable_speedup,
        "rows": all_rows,
    }
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "microbenchmark.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    if not repeatable_speedup:
        raise RuntimeError("Fused failed the repeatable batch-1 graph speedup gate")
    return result


if __name__ == "__main__":
    run_fused_benchmark(os.environ.get("MUSTAFAR_FUSED_RESULTS_DIR"))
