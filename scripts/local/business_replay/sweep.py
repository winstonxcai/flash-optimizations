"""Own a single serving boot and index existing LongSWE replay artifacts.

Author: Winston Cai.
The shell entry point resolves serving defaults. This module owns orchestration,
configuration identity and reporting; numerical replay accounting stays in runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

if __package__:
    from . import runner
else:
    import runner

# Explicitly record relevant knobs, never the complete process environment.
SETTING_NAMES = (
    "REMNANT_INTEGRATION CONTAINER HOST_REPO REPO_CT RESULTS_HOST SGLANG_ROOT_CT SGLANG_PY "
    "MODEL_CT MODEL_NAME MEM_FRAC TP PORT MASTER_PORT CTX_LEN MAX_RUN CHUNK DECODE_CFG "
    "SERVER_SEED C4_KEEP_COUNT C4_LAYER_KEEP_COUNTS STORE_IMPLEMENTATION "
    "FRONTEND_DIAGNOSTICS DSV4_ASYNC_PREPARATION DSV4_PREPARATION_WORKERS "
    "HICACHE_RATIO REPLAY_RUNNER REPLAY_ADAPTER REPLAY_DATASET REPLAY_MANIFEST "
    "REPLAY_SCHEDULER REPLAY_SESSION_POPULATION REPLAY_DRAIN_TIMEOUT REPLAY_TTFT_SLO_MS "
    "REPLAY_MINIMUM_SUCCESS_RATE CUDA_VISIBLE_DEVICES OMP_NUM_THREADS MKL_NUM_THREADS "
    "TOKENIZERS_PARALLELISM"
).split()
REPORT_FIELDS = (
    "concurrency repeat measurement_complete response_success_pass ttft_slo_pass "
    "request_throughput completion_token_throughput prompt_token_throughput "
    "ttft_mean_ms ttft_p90_ms e2e_mean_ms tpot_mean_ms successful failed success_rate "
    "device_hit_rate host_hit_rate total_hit_rate duration_s artifact_path error"
).split()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    runner.atomic_json(path, value)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive(value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("must be a positive integer")
    return int(value)


def concurrencies(value: str) -> list[int]:
    try:
        values = [positive(item) for item in value.split(",")]
    except argparse.ArgumentTypeError as error:
        raise argparse.ArgumentTypeError("use comma-separated positive integers") from error
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("concurrency values must be distinct")
    return values


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sweep-lswb.sh", description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("native", "kernel"), help="server mode; one boot per sweep")
    parser.add_argument("--concurrencies", type=concurrencies, required=True, help="ordered distinct counts, e.g. 20,22,24,26,28,30")
    parser.add_argument("--duration-s", type=positive, default=1200, help="dispatch window per replay in seconds")
    parser.add_argument("--repeats", type=positive, default=1, help="consecutive repetitions per concurrency")
    parser.add_argument("--tag", required=True, help="descriptive label; a unique suffix is added")
    parser.add_argument("--expected-image-id", required=True, help="full sha256 image ID shared by comparison modes")
    parser.add_argument("--ttft-slo-ms", type=float, default=10000.0, help="strict TTFT p90 ceiling in milliseconds")
    parser.add_argument("--minimum-success-rate", type=float, default=0.99, help="inclusive response-success floor")
    parser.add_argument("--point-timeout-s", type=positive, help="complete replay bound; defaults to duration + 900 seconds")
    parser.add_argument("--dry-run", action="store_true", help="read-only preflight; no GPU allocation, model load or files")
    args = parser.parse_args(arguments)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.tag):
        parser.error("--tag must be a simple label containing letters, numbers, _, - or .")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.expected_image_id):
        parser.error("--expected-image-id must be a full sha256 Docker image ID")
    if not finite_number(args.ttft_slo_ms) or args.ttft_slo_ms <= 0:
        parser.error("--ttft-slo-ms must be finite and positive")
    if not finite_number(args.minimum_success_rate) or not 0 <= args.minimum_success_rate <= 1:
        parser.error("--minimum-success-rate must be finite and in [0, 1]")
    args.point_timeout_s = args.point_timeout_s or args.duration_s + 900
    if args.point_timeout_s <= args.duration_s:
        parser.error("--point-timeout-s must exceed --duration-s to allow flushing and drain")
    return args


def command(arguments: list[str], env: dict[str, str], *, timeout: int = 60,
            log: Path | None = None, allowed: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    """Bound local children; remote serving processes remain owned by serve.sh."""
    with log.open("a") if log else open(os.devnull, "w") as sink:
        process = subprocess.Popen(arguments, cwd=env["HOST_REPO"], env=env,
                                   stdout=sink if log else subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
    result = subprocess.CompletedProcess(arguments, process.returncode, output or "")
    if result.returncode not in allowed:
        raise RuntimeError(f"Command exited {result.returncode}: {shlex.join(arguments)}; "
                           f"{('see ' + str(log)) if log else result.stdout[-1500:]}")
    return result


def helper(env: dict[str, str], body: str) -> str:
    local = str(Path(env["HOST_REPO"]) / "scripts/local")
    return command(["bash", "-c", 'set -eu; . "$1/env.sh"; ' + body, "sweep", local], env).stdout


def source_identity(env: dict[str, str]) -> dict:
    return json.loads(helper(env, "check_sources"))


def graph_coverage(graphs: dict, highest: int, padding_disabled: bool = False) -> None:
    decode = graphs.get("decode", {})
    sizes = decode.get("bs") or []
    if not isinstance(sizes, list) or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError(f"Invalid decode graph sizes: {sizes!r}")
    covered = highest in sizes if padding_disabled else any(size >= highest for size in sizes)
    if decode.get("backend") != "full" or not covered:
        raise ValueError(f"Decode graphs must cover C{highest}: {decode}")


def check_ports(env: dict[str, str]) -> None:
    for key in ("PORT", "MASTER_PORT"):
        port = int(env[key])
        if not 1 <= port <= 65535:
            raise ValueError(f"{key} must be in [1, 65535]")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as error:
                raise RuntimeError(f"Refusing occupied {key}={port}: {error}") from error
    if env["PORT"] == env["MASTER_PORT"]:
        raise ValueError("PORT and MASTER_PORT must differ")


def check_owner_record(env: dict[str, str]) -> None:
    pidfile = env.get("SERVER_PID_FILE_CT", f"/tmp/remnant-sglang-{env['PORT']}.pid")
    result = command(["docker", "exec", env["CONTAINER"], "test", "-e", pidfile], env,
                     allowed=(0, 1))
    if result.returncode == 0:
        raise RuntimeError(f"Refusing existing server ownership record: {pidfile}")
    if result.stdout.strip():
        raise RuntimeError(f"Cannot inspect server ownership record: {result.stdout}")


def source_hashes(env: dict[str, str]) -> dict[str, str]:
    root = Path(env["HOST_REPO"])
    files = [root / "scripts" / name for name in (
        "sources.py", "local/env.sh", "local/eval-env.sh", "local/serve.sh",
        "local/run-server.sh", "local/bench-lswb.sh", "local/sweep-lswb.sh",
        "local/business_replay/sweep.py", "local/business_replay/request_slots.py",
    )]
    files.extend(Path(env[name]) for name in ("REPLAY_RUNNER", "REPLAY_ADAPTER", "REPLAY_MANIFEST"))
    return {str(path.resolve()): sha256(path) for path in sorted(set(files))}


def validate_hicache_receipt(env: dict[str, str], image: str, sources: dict) -> dict | None:
    """Require independently collected reset evidence; do not infer GPU safety from source text."""
    if not env.get("HICACHE_RATIO"):
        return None
    ratio = float(env["HICACHE_RATIO"])
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("HICACHE_RATIO must be finite and positive")
    receipt_path = env.get("HICACHE_RESET_RECEIPT")
    if not receipt_path:
        raise ValueError("HiCache sweep requires HICACHE_RESET_RECEIPT from reset qualification "
                         "for this image/fork pair; GPU/host cache clearing and transfer drain are unverified")
    path = Path(receipt_path).resolve()
    receipt = read_json(path)
    expected = {"image_id": image, "integration": env["REMNANT_INTEGRATION"],
                "mode": env["SWEEP_MODE"], "hicache_ratio": ratio,
                "fork_commits": {k: v["commit"] for k, v in sources.items() if k != "parent"}}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"HiCache reset receipt {key} mismatch: expected {value!r}")
    for key in ("gpu_cache_empty", "host_cache_empty", "transfers_drained"):
        if receipt.get("checks", {}).get(key) is not True:
            raise ValueError(f"HiCache reset receipt lacks passing {key}")
    evidence = Path(receipt["evidence_path"])
    if not evidence.is_absolute() or sha256(evidence) != receipt.get("evidence_sha256"):
        raise ValueError("HiCache reset evidence must be an absolute path with a matching SHA256")
    return {"path": str(path), "sha256": sha256(path), "receipt": receipt}


def prepare(args: argparse.Namespace, supplied: dict[str, str]) -> tuple[dict, dict[str, str]]:
    env = dict(supplied)
    for key in ("HOST_REPO", "RESULTS_HOST", "REPLAY_RUNNER", "REPLAY_ADAPTER", "REPLAY_DATASET", "REPLAY_MANIFEST"):
        env[key] = str(Path(env[key]).resolve())
    if not Path(env["RESULTS_HOST"]).is_relative_to(Path(env["HOST_REPO"])):
        raise ValueError("RESULTS_HOST must be under HOST_REPO for the serving metrics mount")
    env.setdefault("REMNANT_INTEGRATION", "v0.5.18")
    env.setdefault("SERVER_SEED", "20260919")
    env.setdefault("REPLAY_SCHEDULER", runner.SCHEDULER)
    env["SWEEP_MODE"] = args.mode
    env["REPLAY_MINIMUM_SUCCESS_RATE"] = str(args.minimum_success_rate)
    if env.get("NSYS_OUTPUT") or env.get("REMNANT_CAPTURE_DIR"):
        raise ValueError("Sweep does not support profiling or latent-capture overrides")
    if env.get("C4_KEEP_COUNT", "256") != "256" or env.get("C4_LAYER_KEEP_COUNTS"):
        raise ValueError("Native/Kernel sweeps require the default 256-survivor profile")
    if args.mode == "kernel":
        if env.get("STORE_IMPLEMENTATION", "kernel") != "kernel":
            raise ValueError("Packed Kernel sweep requires STORE_IMPLEMENTATION=kernel")
        env["STORE_IMPLEMENTATION"] = "kernel"
    tp = positive(env["TP"])
    devices = env.get("CUDA_VISIBLE_DEVICES", "").split(",") if env.get("CUDA_VISIBLE_DEVICES") else []
    valid_devices = all(re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", device) for device in devices)
    if (devices or not args.dry_run) and (len(devices) != tp or len(set(devices)) != tp or not valid_devices):
        raise ValueError(f"GPUQ CUDA_VISIBLE_DEVICES must contain exactly TP={tp} distinct assigned devices")
    if devices:
        env["GPUS"] = ",".join(devices)
    for name in ("MEM_FRAC",):
        value = float(env[name])
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be finite and in (0, 1]")
    if positive(env["MAX_RUN"]) < max(args.concurrencies):
        raise ValueError("MAX_RUN must cover the highest requested concurrency")
    graph_coverage(json.loads(env["DECODE_CFG"]), max(args.concurrencies))
    helper(env, '. "$1/eval-env.sh"; require_replay_config')
    sources = source_identity(env)
    if env["REMNANT_INTEGRATION"] not in ("v0.5.18", "v0.5.22"):
        raise ValueError("Unsupported REMNANT_INTEGRATION")
    if env["REMNANT_INTEGRATION"] == "v0.5.22":
        helper(env, 'declare -F check_container_sources >/dev/null || { '
               'echo "Selected launchers lack v0.5.22 integration support" >&2; exit 1; }; '
               'check_container_sources')
    else:
        helper(env, 'if declare -F check_container_sources >/dev/null; then check_container_sources; fi')
    container = json.loads(command(["docker", "inspect", env["CONTAINER"]], env).stdout)[0]
    if container["Image"] != args.expected_image_id:
        raise ValueError(f"Container image mismatch: expected {args.expected_image_id}, actual {container['Image']}")
    if not container["State"]["Running"]:
        raise ValueError("Selected container is not running; start the existing container before queueing")
    check_ports(env)
    check_owner_record(env)
    policy = argparse.Namespace(
        scheduler=env["REPLAY_SCHEDULER"], max_concurrency=max(args.concurrencies),
        max_duration=args.duration_s, arrival_mode="immediate", warmup_manifest=None,
        session_population=int(env["REPLAY_SESSION_POPULATION"]) if env.get("REPLAY_SESSION_POPULATION") else None,
        drain_timeout=float(env["REPLAY_DRAIN_TIMEOUT"]) if env.get("REPLAY_DRAIN_TIMEOUT") else None,
        ttft_slo_ms=float(env["REPLAY_TTFT_SLO_MS"]) if env.get("REPLAY_TTFT_SLO_MS") else None,
    )
    if policy.scheduler == "request-slot-fifo":
        if policy.ttft_slo_ms is not None and policy.ttft_slo_ms != args.ttft_slo_ms:
            raise ValueError("REPLAY_TTFT_SLO_MS conflicts with --ttft-slo-ms")
        policy.ttft_slo_ms = args.ttft_slo_ms
        env["REPLAY_TTFT_SLO_MS"] = str(args.ttft_slo_ms)
    protocol = runner.validate_policy(policy)
    records, _ = runner.load_dataset_manifest(Path(env["REPLAY_MANIFEST"]), Path(env["REPLAY_DATASET"]), 4916)
    if policy.session_population is not None and policy.session_population > len({r.user_id for r in records}):
        raise ValueError("Session population exceeds the frozen dataset's user count")
    manifest = {
        "schema_version": 1, "mode": args.mode, "tag": args.tag,
        "duration_s": args.duration_s, "point_timeout_s": args.point_timeout_s,
        "ttft_slo_ms": args.ttft_slo_ms, "minimum_success_rate": args.minimum_success_rate,
        "repetitions_basis": "Consecutive repetitions within one server boot; not independent startups",
        "points": [{"concurrency": c, "repeat": r} for c in args.concurrencies for r in range(1, args.repeats + 1)],
        "protocol": protocol, "image_id": container["Image"], "sources": sources,
        "source_hashes": source_hashes(env), "settings": {k: env[k] for k in SETTING_NAMES if k in env},
        "node": socket.gethostname(), "gpuq_job_id": env.get("GPUQ_JOB_ID"),
        "container_labels": {k: v for k, v in (container.get("Config", {}).get("Labels") or {}).items()
                             if k.startswith("remnant.")},
        "hicache_reset_qualification": validate_hicache_receipt(env, container["Image"], sources),
    }
    return manifest, env


def server_info(env: dict[str, str]) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{env['PORT']}/server_info", timeout=30) as response:
        return json.load(response)


def check_server(info: dict, env: dict[str, str], mode: str, highest: int) -> dict:
    expected = {
        "model_path": env["MODEL_CT"], "served_model_name": env["MODEL_NAME"],
        "export_metrics_to_file": True, "enable_metrics": True,
        "tp_size": int(env["TP"]), "mem_fraction_static": float(env["MEM_FRAC"]),
        "random_seed": int(env["SERVER_SEED"]), "chunked_prefill_size": int(env["CHUNK"]),
        "context_length": int(env["CTX_LEN"]), "max_running_requests": int(env["MAX_RUN"]),
        "enable_hierarchical_cache": bool(env.get("HICACHE_RATIO")),
        "dsv4_c4_cache_format": "native" if mode == "native" else "remnant",
    }
    if mode == "kernel":
        expected.update(dsv4_c4_cache_implementation="kernel", dsv4_c4_cache_store_implementation="kernel",
                        dsv4_c4_cache_keep_count=256, dsv4_c4_cache_layer_keep_counts=None)
    if env.get("HICACHE_RATIO"):
        expected.update(hicache_ratio=float(env["HICACHE_RATIO"]), hicache_write_policy="write_through",
                        hicache_io_backend="kernel", hicache_mem_layout="layer_first", hicache_storage_backend=None)
    for key, value in expected.items():
        if key not in info or info[key] != value:
            raise ValueError(f"Server configuration mismatch {key}: expected {value!r}, got {info.get(key)!r}")
    graph_coverage(info.get("cuda_graph_config", {}), highest, info.get("disable_cuda_graph_padding", False))
    # Snapshot stable fields only; throughput and memory-use counters naturally change.
    keys = list(expected) + ["model_path", "tokenizer_path", "kv_cache_dtype", "cuda_graph_config",
                             "max_total_num_tokens", "export_metrics_to_file", "export_metrics_to_file_dir"]
    return {key: info.get(key) for key in keys}


def check_frozen(manifest: dict, env: dict[str, str]) -> None:
    current = source_identity(env)
    if {k: v for k, v in current.items() if k != "parent"} != {
        k: v for k, v in manifest["sources"].items() if k != "parent"
    } or source_hashes(env) != manifest["source_hashes"]:
        raise RuntimeError("Frozen source, dataset manifest or launcher content changed during sweep")
    actual = command(["docker", "inspect", "--format", "{{.Image}}", env["CONTAINER"]], env).stdout.strip()
    if actual != manifest["image_id"]:
        raise RuntimeError("Container image changed during sweep")


def collect_point(path: Path, point: dict, manifest: dict, returncode: int) -> dict:
    """Validate existing runner output and retain its original numerical values."""
    required = ("client.log", "server.log", "server-info.json", "request-timings.jsonl", "cache-flush.log")
    for relative in required:
        if not (path / relative).is_file():
            raise ValueError(f"Missing replay artifact: {path / relative}")
    summary = read_json(path / "client/summary.json")
    done = read_json(path / "client/done.json")
    timing = read_json(path / "request-timing-validation.json")
    window = read_json(path / "replay-window.json")
    if not timing.get("complete") or timing.get("partial") or timing.get("errors"):
        raise ValueError(f"Incomplete request-timing artifacts: {timing}")
    if not finite_number(window.get("ended_at")) or window["ended_at"] <= window["started_at"]:
        raise ValueError("Replay window is incomplete")
    if done.get("protocol") != manifest["protocol"] or summary.get("protocol") != manifest["protocol"]:
        raise ValueError("Replay protocol does not match sweep")
    if done.get("valid") != summary.get("valid") or not (path / "cache-flush.log").is_file():
        raise ValueError("Missing cache flush evidence or inconsistent completion marker")
    clients = [json.loads(line) for line in (path / "client/responses.jsonl").read_text().splitlines() if line]
    if len({record["request_id"] for record in clients}) != len(clients):
        raise ValueError("Duplicate replay responses")
    exported = [json.loads(line) for line in (path / "request-timings.jsonl").read_text().splitlines() if line]
    if len(exported) != timing.get("records") or len({record["id"] for record in exported}) != len(exported):
        raise ValueError("Request timing record count/identity does not match collection metadata")
    activity = [json.loads(line) for line in (path / "client/active_requests.jsonl").read_text().splitlines() if line]
    dataset = read_json(path / "client/dataset_manifest.json")
    client_manifest = read_json(path / "client/manifest.json")
    replay = client_manifest["replay"]
    if replay.get("max_concurrency") != point["concurrency"] or replay.get("max_duration_s") != manifest["duration_s"]:
        raise ValueError("Replay concurrency/duration differs from requested point")
    if dataset.get("source_manifest_sha256") != manifest["source_hashes"][manifest["settings"]["REPLAY_MANIFEST"]]:
        raise ValueError("Replay used a different frozen dataset manifest")
    records = [runner.Record(**{**record, "path": Path(record["path"])}) for record in dataset["records"]]
    if not finite_number(summary.get("duration_s")) or summary["duration_s"] <= 0:
        raise ValueError("Invalid replay duration")
    if manifest["protocol"] == runner.PROTOCOL:
        recalculated = runner.summarize(records, clients, activity, summary["duration_s"], manifest["minimum_success_rate"])
    else:
        slots = runner.request_slots
        by_id = {record.request_id: record for record in records}
        responses = {record["request_id"]: record for record in clients}
        if len(responses) != len(clients):
            raise ValueError("Duplicate replay responses")
        outcome = slots.ReplayOutcome(summary["duration_s"], summary["total_elapsed_s"], activity=activity)
        for event in activity:
            if event.get("event") == "ready":
                result = responses.pop(event["request_id"], None)
                outcome.states.append(slots.RequestState(
                    by_id[event["request_id"]], event["user_sequence"], event["ready_offset_s"],
                    event["session_slot_id"], result.get("dispatch_offset_s") if result else None, result,
                ))
        if responses:
            raise ValueError("Responses without matching readiness events")
        recalculated = slots.summarize(outcome, records, manifest["minimum_success_rate"],
                                     manifest["ttft_slo_ms"], runner.summarize, runner.distribution)
    mismatches = [key for key, value in recalculated.items() if summary.get(key) != value]
    if mismatches:
        raise ValueError(f"Summary does not match raw records: {mismatches}")
    if summary.get("instrumentation_errors", 0):
        raise ValueError("Replay reports invalid client timing instrumentation")
    if summary.get("minimum_success_rate") != manifest["minimum_success_rate"]:
        raise ValueError("Response threshold differs from sweep")
    if not summary.get("counts", {}).get("attempted"):
        raise ValueError("Replay attempted no requests")
    if returncode != (0 if summary["valid"] else 1):
        raise ValueError(f"Replay exit {returncode} is inconsistent with saved response validity")
    queue_inclusive = manifest["protocol"] != runner.PROTOCOL
    ttft = summary["ready_ttft_ms"] if queue_inclusive else summary["ttft_ms"]
    e2e = summary["ready_latency_ms"] if queue_inclusive else summary["latency_ms"]
    for stats, names in ((ttft, ("mean", "p90")), (e2e, ("mean",)), (summary["tpot_ms"], ("mean",))):
        if any(not finite_number(stats.get(key)) or stats[key] < 0 for key in names):
            raise ValueError("Missing/nonfinite latency measurement")
    cache, counts = summary["cache"], summary["counts"]
    return {
        **point, "measurement_complete": True,
        "response_success_pass": summary["success_rate"] >= manifest["minimum_success_rate"],
        "ttft_slo_pass": ttft["p90"] < manifest["ttft_slo_ms"] and (
            not queue_inclusive or summary["slo"]["status"] == "pass"),
        **{key: summary[key] for key in ("request_throughput", "completion_token_throughput", "prompt_token_throughput", "success_rate", "duration_s")},
        "ttft_mean_ms": ttft["mean"], "ttft_p90_ms": ttft["p90"], "e2e_mean_ms": e2e["mean"],
        "tpot_mean_ms": summary["tpot_ms"]["mean"],
        "successful": counts.get("successful_in_window", counts["successful"]), "failed": counts["failed"],
        **{key: cache.get(key) for key in ("device_hit_rate", "host_hit_rate", "total_hit_rate")},
        "artifact_path": str(path), "error": "",
    }


def save_report(root: Path, rows: list[dict], status: str, error: str | None = None) -> None:
    atomic_json(root / "summary.json", {"schema_version": 1, "status": status, "error": error, "points": rows})
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "summary.csv")


def print_report(rows: list[dict]) -> None:
    print("C   REPEAT COMPLETE RESPONSE SLO     REQ/S   OUTPUT TOK/S TTFT-p90(s)")
    for row in rows:
        render = lambda key: "N/A" if row.get(key) is None else ("PASS" if row[key] else "FAIL")
        print(f"{row['concurrency']:<3} {row['repeat']:<6} {render('measurement_complete'):<8} "
              f"{render('response_success_pass'):<8} {render('ttft_slo_pass'):<7} "
              f"{row.get('request_throughput', 0):.4f}  {row.get('completion_token_throughput', 0):10.2f} "
              f"{row.get('ttft_p90_ms', 0)/1000:11.3f}")


def execute(args: argparse.Namespace, manifest: dict, env: dict[str, str]) -> Path:
    base = Path(env["RESULTS_HOST"]) / "lswb-replay"
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{args.tag}-{args.mode}-{time.strftime('%Y%m%d_%H%M%S')}-", dir=base))
    boot = root / "boot"
    boot.mkdir()
    env = {**env, "LOG_HOST": str(boot), "REQUEST_METRICS_DIR": str(boot / "request-metrics"),
           "SERVER_PID_FILE_CT": f"/tmp/remnant-{root.name}.pid"}
    manifest["artifact_root"] = str(root)
    manifest["server_ownership"] = {key: env[key] for key in ("LOG_HOST", "REQUEST_METRICS_DIR", "SERVER_PID_FILE_CT")}
    manifest["gpus"] = command(["nvidia-smi", "--id=" + env["GPUS"], "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"], env).stdout.splitlines()
    # Include explicit unsets so inherited HiCache/protocol knobs cannot change reproduction.
    relevant = SETTING_NAMES + ["HICACHE_RESET_RECEIPT"]
    unset = [key for key in relevant if key not in env and key != "CUDA_VISIBLE_DEVICES"]
    assignments = [f"{key}={value}" for key, value in manifest["settings"].items() if key != "CUDA_VISIBLE_DEVICES"]
    if env.get("HICACHE_RESET_RECEIPT"):
        assignments.append(f"HICACHE_RESET_RECEIPT={env['HICACHE_RESET_RECEIPT']}")
    manifest["reproduction_command"] = shlex.join(["env", *[item for key in unset for item in ("-u", key)], *assignments,
        "bash", str(Path(env["HOST_REPO"]) / "scripts/local/sweep-lswb.sh"), *sys.argv[1:]])
    atomic_json(root / "manifest.json", manifest)
    rows: list[dict] = []
    save_report(root, rows, "starting")
    local = Path(env["HOST_REPO"]) / "scripts/local"
    owns_server = False
    failure: BaseException | None = None
    try:
        check_ports(env)
        check_owner_record(env)
        owns_server = True  # The fresh PID path can only identify this boot, even if startup fails.
        command(["bash", str(local / "serve.sh"), args.mode], env, timeout=1800, log=boot / "startup.log")
        info = server_info(env)
        stable = check_server(info, env, args.mode, max(args.concurrencies))
        atomic_json(boot / "server-info.json", info)
        for point in manifest["points"]:
            path = root / f"c{point['concurrency']}" / f"repeat-{point['repeat']:02d}"
            row = {**point, "measurement_complete": False, "response_success_pass": None,
                   "ttft_slo_pass": None, "artifact_path": str(path), "error": ""}
            rows.append(row)
            save_report(root, rows, "running")
            try:
                check_frozen(manifest, env)
                if check_server(server_info(env), env, args.mode, max(args.concurrencies)) != stable:
                    raise RuntimeError("Server settings or pool capacity changed during sweep")
                point_env = {**env, "LSWB_RUN_ROOT": str(path)}
                result = command(["bash", str(local / "bench-lswb.sh"), root.name, env["PORT"],
                                  str(point["concurrency"]), str(args.duration_s)], point_env,
                                 timeout=args.point_timeout_s, log=root / "replay.log", allowed=(0, 1))
                check_frozen(manifest, env)
                if check_server(server_info(env), env, args.mode, max(args.concurrencies)) != stable:
                    raise RuntimeError("Server settings or pool capacity changed during measurement")
                rows[-1] = collect_point(path, point, manifest, result.returncode)
                save_report(root, rows, "running")
                print_report([rows[-1]])
            except BaseException as error:
                rows[-1].update(measurement_complete=False, response_success_pass=None,
                                ttft_slo_pass=None, error=str(error))
                raise
        check_frozen(manifest, env)
        if check_server(server_info(env), env, args.mode, max(args.concurrencies)) != stable:
            raise RuntimeError("Server settings changed during final measurement")
    except BaseException as error:
        failure = error
    finally:
        if owns_server:
            # A second SIGINT/TERM must not interrupt cleanup of our server group.
            old = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
            try:
                command(["bash", str(local / "serve.sh"), args.mode, "stop"], env, timeout=60, log=boot / "cleanup.log")
            except BaseException as error:
                failure = RuntimeError(f"{failure or 'Sweep complete'}; server cleanup failed: {error}")
            finally:
                for sig, handler in old.items():
                    signal.signal(sig, handler)
        save_report(root, rows, "failed" if failure else "complete", str(failure) if failure else None)
        print_report(rows)
        print(f"Sweep artifacts: {root}", flush=True)
    if failure:
        raise RuntimeError(f"Sweep failed; retained {root}: {failure}") from failure
    return root


def main() -> int:
    args = parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"Sweep interrupted by signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        manifest, env = prepare(args, dict(os.environ))
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
        else:
            execute(args, manifest, env)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, argparse.ArgumentTypeError, subprocess.SubprocessError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
