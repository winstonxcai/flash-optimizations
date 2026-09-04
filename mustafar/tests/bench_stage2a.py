"""Stage-1 Triton versus Stage-2A CUDA reconstruction microbenchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median

import torch

from .. import reference
from ..packed_c4 import (
    NativeC4Workspace,
    PackedC4Buffers,
    pack_c4_rows_ref,
    unpack_gather_c4_native,
    unpack_gather_c4_native_stage2a,
)


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


def run(output_dir: str | Path | None = None) -> dict[str, object]:
    os.environ.setdefault("SGLANG_OPT_TOPMAG", "1")
    os.environ.setdefault("XKV_TOPMAG_KEEP", "0.5")
    os.environ.setdefault("SGLANG_OPT_TOPMAG_PACKED_C4", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("bench_stage2a requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(2027)
    selected_k = 512
    max_batch = 16
    pool_rows = max_batch * selected_k
    latent = torch.randn(pool_rows, 512, dtype=torch.bfloat16, device=device)
    mask = reference.topmag_keep_mask(latent, 0.5)
    weight = torch.ones(512, dtype=torch.bfloat16, device=device)
    buffers = PackedC4Buffers(*pack_c4_rows_ref(latent, mask, weight, 1.0e-6))
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
        stage1 = NativeC4Workspace.allocate(
            batch, selected_k, 64, device, with_dense=True
        )
        stage2a = NativeC4Workspace.allocate(
            batch, selected_k, 64, device, with_dense=False
        )
        eager_fns = {
            "stage1": lambda: unpack_gather_c4_native(
                buffers, physical, raw, lengths, freqs, stage1
            ),
            "stage2a": lambda: unpack_gather_c4_native_stage2a(
                buffers, physical, raw, lengths, freqs, stage2a
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
        row for row in all_rows
        if row["batch"] == 1 and row["execution"] == "graph"
    ]
    stage1_medians = [row["p50_us"] for row in b1_graph if row["mode"] == "stage1"]
    stage2a_medians = [row["p50_us"] for row in b1_graph if row["mode"] == "stage2a"]
    repeatable_speedup = all(
        new < old for new, old in zip(stage2a_medians, stage1_medians)
    )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "selected_k": selected_k,
        "samples_per_round": 500,
        "rounds": 3,
        "batch1_graph_stage1_median_us": median(stage1_medians),
        "batch1_graph_stage2a_median_us": median(stage2a_medians),
        "batch1_graph_speedup": median(stage1_medians) / median(stage2a_medians),
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
        raise RuntimeError("Stage 2A failed the repeatable batch-1 graph speedup gate")
    return result


if __name__ == "__main__":
    run(os.environ.get("MUSTAFAR_STAGE2A_RESULTS_DIR"))
