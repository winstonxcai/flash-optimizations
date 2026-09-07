#!/usr/bin/env python3
"""Pure-Python parsing/assembly for the local TP4 experiments (no Modal).

Ports the regex/JSON logic from mustafar/scripts/modal/app.py so the bash
drivers (mustafar/scripts/local/tp4_*.sh) only orchestrate server launches
and `sglang.bench_serving` runs; this module owns the parsing, validation,
and result-JSON assembly.

Subcommands:
  ceiling <log>                        parsed pool-ceiling JSON (stdout)
  resident <log> <label>               resident_requests_with_output for a label (32k/64k/128k)
  scan <log>                           max running/queued/full-token-usage from a log segment
  validate-record <jsonl> <expected_requests> <expected_out_tokens>
                                       fail closed on a bench_serving record
  assemble-capacity <results_dir> <out_json>
  assemble-graphdecode <results_dir> <out_json>
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

MODEL_REVISION = "ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17"
PAGE_SIZE = 256
OUTPUT_TOKENS = 16
CONTEXTS = (("32k", 32 * 1024), ("64k", 64 * 1024), ("128k", 128 * 1024))
GRAPH_DECODE_CONTEXTS = 64 * 1024
GRAPH_DECODE_OUTPUT_TOKENS = 128
GRAPH_DECODE_POINTS = (("c1", 1, 2), ("c2", 2, 4))  # label, concurrency, num_prompts


def parse_pool_ceiling(text: str) -> dict:
    memory = re.search(
        r"bytes_per_full_token=([0-9.]+), available_bytes=([0-9.]+) GB, "
        r"c128_state_fixed=([0-9.]+) GB, full_token=([0-9]+)",
        text,
    )
    pools = re.search(
        r"DSV4 pool sizes: full=([0-9]+), swa=([0-9]+), c4=([0-9]+), "
        r"c128=([0-9]+), c4_state=([0-9]+), c128_state=([0-9]+)",
        text,
    )
    maximum = re.search(r"max_total_num_tokens=([0-9]+)", text)
    if memory is None or pools is None or maximum is None:
        raise RuntimeError("could not parse DSV4 pool allocation from server log")

    full_tokens = int(maximum.group(1))
    packed = re.search(r"logical_row_bytes=328 allocated_bytes=([0-9]+)", text)
    contexts: dict = {}
    for label, ctx in CONTEXTS:
        occupied = math.ceil((ctx + OUTPUT_TOKENS) / PAGE_SIZE) * PAGE_SIZE
        contexts[label] = {
            "context_tokens": ctx,
            "output_tokens": OUTPUT_TOKENS,
            "page_rounded_tokens_per_request": occupied,
            "theoretical_context_only": full_tokens // ctx,
            "resident_requests_with_output": full_tokens // occupied,
            "remaining_tokens": full_tokens % occupied,
        }
    return {
        "bytes_per_full_token": float(memory.group(1)),
        "available_gib_reported": float(memory.group(2)),
        "c128_state_fixed_gib_reported": float(memory.group(3)),
        "full_tokens": full_tokens,
        "pools": {
            "full": int(pools.group(1)),
            "swa": int(pools.group(2)),
            "c4": int(pools.group(3)),
            "c128": int(pools.group(4)),
            "c4_state": int(pools.group(5)),
            "c128_state": int(pools.group(6)),
        },
        "packed_c4_allocated_bytes": int(packed.group(1)) if packed else None,
        "contexts": contexts,
    }


def scan_log(text: str) -> dict:
    return {
        "max_running_observed": max(
            (int(x) for x in re.findall(r"#running-req: ([0-9]+)", text)), default=0
        ),
        "max_queued_observed": max(
            (int(x) for x in re.findall(r"#queue-req: ([0-9]+)", text)), default=0
        ),
        "max_full_token_usage": max(
            (float(x) for x in re.findall(r"full token usage: ([0-9.]+)", text)),
            default=0.0,
        ),
    }


def load_records(jsonl: Path) -> list:
    if not jsonl.exists():
        return []
    return [
        json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()
    ]


def load_meta(meta: Path) -> dict:
    result = {}
    if meta.exists():
        for line in meta.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def validate_decode_graph_record(
    record: dict | None, *, expected_requests: int, expected_output_tokens: int
) -> dict:
    if record is None:
        raise RuntimeError("bench_serving did not write a result record")
    if record.get("completed") != expected_requests:
        raise RuntimeError(
            "unexpected completed count: %s != %s"
            % (record.get("completed"), expected_requests)
        )
    errors = [error for error in record.get("errors", []) if error]
    if errors:
        raise RuntimeError("bench_serving returned request errors: %s" % errors)
    output_lens = record.get("output_lens", [])
    if len(output_lens) != expected_requests or any(
        length != expected_output_tokens for length in output_lens
    ):
        raise RuntimeError(
            "bench_serving did not generate the requested decode length: %s"
            % output_lens
        )
    server_info = record.get("server_info") or {}
    graph_config = server_info.get("cuda_graph_config") or {}
    decode = graph_config.get("decode") or {}
    backend = decode.get("backend")
    batches = decode.get("bs") or []
    if backend == "disabled" or not {1, 2}.issubset(set(batches)):
        raise RuntimeError(
            "decode CUDA graph was not configured for batch 1 and 2: %s" % decode
        )
    return {"decode_graph_backend": backend, "decode_graph_batches": batches}


def _read_ready_sec(results_dir: Path, sub: str, name: str) -> float:
    meta = results_dir / sub / f"{name}.ready_sec"
    try:
        return float(meta.read_text().strip())
    except (OSError, ValueError):
        return 0.0


def _scan_or_zero(delta: Path) -> dict:
    if delta.exists():
        return scan_log(delta.read_text(errors="replace"))
    return {
        "max_running_observed": 0,
        "max_queued_observed": 0,
        "max_full_token_usage": 0.0,
    }


def assemble_capacity(results_dir: Path, out_path: Path) -> dict:
    results_dir = Path(results_dir)
    run_dir = results_dir / "official-capacity"
    legs: dict = {}
    for name, packed in (("topmag50_packed", True), ("native", False)):
        log = results_dir / f"capacity-{name}-server.log"
        if not log.exists() or "Memory pool end." not in log.read_text(errors="replace"):
            continue
        leg = {
            "packed": packed,
            "seconds_to_server_ready": _read_ready_sec(
                results_dir, "official-capacity", name
            ),
            **parse_pool_ceiling(log.read_text(errors="replace")),
            "benchmarks": {},
        }
        for label, _ctx in CONTEXTS:
            jsonl = run_dir / f"{name}-{label}.jsonl"
            if not jsonl.exists():
                continue
            delta = run_dir / f"{name}-{label}.logdelta"
            bench_meta = load_meta(run_dir / f"{name}-{label}.benchmeta")
            records = load_records(jsonl)
            leg["benchmarks"][label] = {
                "official_module": "sglang.bench_serving",
                "concurrency": int(
                    leg["contexts"][label]["resident_requests_with_output"]
                ),
                "returncode": int(bench_meta.get("returncode", 0)),
                "seconds": float(bench_meta.get("seconds", 0.0)),
                "record": records[-1] if records else None,
                **_scan_or_zero(delta),
            }
        legs[name] = leg

    result = {
        "experiment": "tp4-memory-bound-ceiling",
        "model_revision": MODEL_REVISION,
        "tp": 4,
        "mem_fraction_static": 0.93,
        "max_running_requests": 64,
        "cuda_graph": False,
        "method": "official python -m sglang.bench_serving at allocator ceiling",
        "legs": legs,
    }
    if "topmag50_packed" in legs and "native" in legs:
        result["full_token_gain"] = (
            legs["topmag50_packed"]["full_tokens"] / legs["native"]["full_tokens"]
        )
    _write(out_path, result)
    return result


def assemble_graphdecode(results_dir: Path, out_path: Path) -> dict:
    results_dir = Path(results_dir)
    run_dir = results_dir / "official-graph-decode"
    legs: dict = {}
    for name, packed in (("topmag50_packed", True), ("native", False)):
        log = results_dir / f"graph-decode-{name}-server.log"
        if not log.exists() or "Memory pool end." not in log.read_text(errors="replace"):
            continue
        leg = {
            "packed": packed,
            "seconds_to_server_ready": _read_ready_sec(
                results_dir, "official-graph-decode", name
            ),
            **parse_pool_ceiling(log.read_text(errors="replace")),
            "benchmarks": {},
        }
        for label, concurrency, num_prompts in GRAPH_DECODE_POINTS:
            jsonl = run_dir / f"{name}-{label}.jsonl"
            if not jsonl.exists():
                continue
            records = load_records(jsonl)
            record = records[-1] if records else None
            meta = load_meta(run_dir / f"{name}-{label}.meta")
            expected_req = int(meta.get("num_prompts", num_prompts))
            expected_out = int(
                meta.get("output_tokens", GRAPH_DECODE_OUTPUT_TOKENS)
            )
            graph_evidence = validate_decode_graph_record(
                record,
                expected_requests=expected_req,
                expected_output_tokens=expected_out,
            )
            delta = run_dir / f"{name}-{label}.logdelta"
            leg["benchmarks"][label] = {
                "official_module": "sglang.bench_serving",
                "concurrency": concurrency,
                "num_prompts": num_prompts,
                "record": record,
                "graph_replay_log_values": (
                    re.findall(
                        r"cuda graph:\s*(True|False)",
                        delta.read_text(errors="replace"),
                        re.IGNORECASE,
                    )
                    if delta.exists()
                    else []
                ),
                **graph_evidence,
            }
        legs[name] = leg

    result = {
        "experiment": "tp4-graph-decode-ab",
        "model_revision": MODEL_REVISION,
        "tp": 4,
        "context_tokens": GRAPH_DECODE_CONTEXTS,
        "output_tokens": GRAPH_DECODE_OUTPUT_TOKENS,
        "mem_fraction_static": 0.93,
        "max_running_requests": 2,
        "cuda_graph_config": {
            "decode": {"backend": "full", "max_bs": 2, "bs": [1, 2]},
            "prefill": {"backend": "disabled"},
        },
        "method": "official python -m sglang.bench_serving",
        "legs": legs,
    }
    _write(out_path, result)
    return result


def _write(out_path: Path, result: dict) -> None:
    Path(out_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "ceiling":
        print(
            json.dumps(
                parse_pool_ceiling(Path(sys.argv[2]).read_text(errors="replace")),
                sort_keys=True,
            )
        )
    elif cmd == "resident":
        ceiling = parse_pool_ceiling(
            Path(sys.argv[2]).read_text(errors="replace")
        )
        print(ceiling["contexts"][sys.argv[3]]["resident_requests_with_output"])
    elif cmd == "scan":
        print(json.dumps(scan_log(Path(sys.argv[2]).read_text(errors="replace")), sort_keys=True))
    elif cmd == "validate-record":
        records = load_records(Path(sys.argv[2]))
        evidence = validate_decode_graph_record(
            records[-1] if records else None,
            expected_requests=int(sys.argv[3]),
            expected_output_tokens=int(sys.argv[4]),
        )
        print(json.dumps(evidence, sort_keys=True))
    elif cmd == "assemble-capacity":
        assemble_capacity(Path(sys.argv[2]), Path(sys.argv[3]))
        print("wrote %s" % sys.argv[3], flush=True)
    elif cmd == "assemble-graphdecode":
        assemble_graphdecode(Path(sys.argv[2]), Path(sys.argv[3]))
        print("wrote %s" % sys.argv[3], flush=True)
    else:
        raise SystemExit("unknown subcommand: %s" % cmd)


if __name__ == "__main__":
    main()
