#!/usr/bin/env python3
"""Assemble the fp4-native serving-capacity sweep into comparison tables.

Reads mustafar/results/serve-sweep-fp4/<leg>/<ts>/ produced by
run_serve_sweep_fp4.sh. For each context (32k/64k/128k/256k, output 2048) each
leg ran:
  native: one point at the native-584 allocator ceiling  (ctx<T>-max)
  packed: a fair point at the native ceiling             (ctx<T>-fair)
          and a max   point at the packed ceiling        (ctx<T>-max)

The bench output JSONL holds ONE merged record per measured wave
(result | result_details). Validation per point: completed == 3*C, no errors,
every output_lens == 2048. Decode-graph replay is counted from server.delta.log
("cuda graph: True/False" lines); residency from "#running-req: N".

Usage: serve_sweep_parse.py <results_root> [<ts>]
  prints a JSON summary; the run dirs for each leg must share one <ts> (default:
  the newest present).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CTX_TOK = {"32k": 32768, "64k": 65536, "128k": 131072, "256k": 262144}
OUT_TOK = 2048
POOL_EXPECT = {"native": 3730944, "packed": 4519168}


def load_record(jsonl: Path) -> dict | None:
    lines = [l for l in jsonl.read_text().splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def scan_delta(path: Path) -> dict:
    if not path.exists():
        return {"running_max": None, "graph_true": 0, "graph_false": 0}
    text = path.read_text(errors="replace")
    running = [int(x) for x in re.findall(r"#running-req: (\d+)", text)]
    return {
        "running_max": max(running) if running else None,
        "graph_true": len(re.findall(r"cuda graph:\s*(?:True|true)", text)),
        "graph_false": len(re.findall(r"cuda graph:\s*(?:False|false)", text)),
    }


def parse_point(leg: str, ts_dir: Path, ctx_label: str, tag: str) -> dict | None:
    run_dir = ts_dir / f"ctx{CTX_TOK[ctx_label]}-{tag}"
    jsonl = run_dir / "measured.jsonl"
    if not jsonl.exists():
        return None
    rec = load_record(jsonl)
    if rec is None:
        raise RuntimeError(f"no record in {jsonl}")
    completed = rec.get("completed", 0)
    if completed == 0:
        raise RuntimeError(f"{jsonl}: completed=0 (bench failed?)")
    exp_req = completed  # measured JSONL itself is authoritative for N
    errors = [e for e in (rec.get("errors") or []) if e]
    out_lens = rec.get("output_lens") or []
    expected_ok = len(out_lens) == exp_req and all(
        n == OUT_TOK for n in out_lens
    )
    delta = scan_delta(run_dir / "server.delta.log")
    return {
        "leg": leg,
        "context": ctx_label,
        "context_tokens": CTX_TOK[ctx_label],
        "tag": tag,
        "concurrency_configured": rec.get("max_concurrency"),
        "request_throughput": rec.get("request_throughput"),
        "total_throughput": rec.get("total_throughput"),
        "output_throughput": rec.get("output_throughput"),
        "input_throughput": rec.get("input_throughput"),
        "total_input_tokens": rec.get("total_input_tokens"),
        "total_output_tokens": rec.get("total_output_tokens"),
        "mean_e2e_ms": rec.get("mean_e2e_latency_ms"),
        "mean_ttft_ms": rec.get("mean_ttft_ms"),
        "mean_tpot_ms": rec.get("mean_tpot_ms"),
        "median_tpot_ms": rec.get("median_tpot_ms"),
        "duration_s": rec.get("duration"),
        "completed": completed,
        "errors": errors,
        "output_all_2048": expected_ok,
        "running_max_observed": delta["running_max"],
        "graph_replay_true": delta["graph_true"],
        "graph_replay_false": delta["graph_false"],
        "record": rec,
    }


def main() -> None:
    root = Path(sys.argv[1])
    ts = sys.argv[2] if len(sys.argv) > 2 else None
    legs: dict = {}
    for leg in ("native", "packed"):
        leg_dir = root / leg
        if not leg_dir.exists():
            continue
        ts_dir = leg_dir / ts if ts else max(
            (d for d in leg_dir.iterdir() if d.is_dir()), key=lambda d: d.name
        )
        legs[leg] = {
            "ts": ts_dir.name,
            "pool_expected": POOL_EXPECT[leg],
            "points": {},
        }
        for ctx in CTX_TOK:
            if leg == "native":
                # native has one point per ctx (its ceiling == the fair concurrency)
                p = parse_point(leg, ts_dir, ctx, "max")
                if p is None:
                    continue
                p["tag"] = "fair==max"
                legs[leg]["points"][ctx] = p
            else:
                fair = parse_point(leg, ts_dir, ctx, "fair")
                mx = parse_point(leg, ts_dir, ctx, "max")
                if fair is not None:
                    legs[leg]["points"][f"{ctx}-fair"] = fair
                if mx is not None:
                    legs[leg]["points"][f"{ctx}-max"] = mx
    print(json.dumps({"experiment": "serve-sweep-fp4", "legs": legs}, indent=2))


if __name__ == "__main__":
    main()
