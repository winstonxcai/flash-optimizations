#!/usr/bin/env bash
# =====================================================================
# bench-lswb.sh <tag> [port] [concurrency] [duration_s] [nsys_windows]
# Author: Winston Cai.
#   LongSWE-Bench replay client against an ALREADY-RUNNING server (bring it
#   up first with serve.sh native|packed). Replays 4,916 recorded Claude
#   business conversations (~144k prompt tok/req, short decodes) over OpenAI
#   SSE for a fixed window -- a prefix-reuse workload that shows whether the
#   larger packed KV pool retains more shared prefixes (device cache hits).
#
#   tag   = results label (e.g. native, packed)
#   port  = server port, default $PORT (from env.sh)
#   C,DUR = concurrency and window (default 15 / 1200 s, the official heaviest
#           point used in the report)
#
# Runs the business_replay runner on THIS host (host python3 has `requests`).
# Results: <RESULTS_HOST>/lswb-replay/<tag>/<ts>/  (+ client.log, gpu-samples).
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"
. "$(dirname "$0")/eval-env.sh"

TAG=${1:-} PORT=${2:-$PORT} C=${3:-15} DUR=${4:-1200}
NSYS_WINDOWS=${5:-}
[ -n "$TAG" ] || { echo "usage: $0 <tag> [port] [concurrency] [duration_s]"; exit 1; }
require_replay_config
[[ "$C" =~ ^[1-9][0-9]*$ ]] || { echo "concurrency must be a positive integer" >&2; exit 2; }
[[ "$DUR" =~ ^[1-9][0-9]*$ ]] || { echo "duration must be a positive integer" >&2; exit 2; }

# Reject malformed/overlapping windows before starting the replay client.
if [ -n "$NSYS_WINDOWS" ]; then
  [[ "$NSYS_WINDOWS" =~ ^[0-9]+:[0-9]+(,[0-9]+:[0-9]+)*$ ]] || { echo "capture windows must be start:end pairs" >&2; exit 2; }
  IFS=, read -r -a windows <<< "$NSYS_WINDOWS"
  previous_end=0
  for window in "${windows[@]}"; do
    start=$((10#${window%:*})) end=$((10#${window#*:}))
    (( previous_end <= start && start < end && end <= DUR )) || { echo "invalid capture window $window for duration $DUR" >&2; exit 2; }
    previous_end=$end
  done
fi

health () { curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
health || { echo "FATAL: no server on 127.0.0.1:$PORT (run serve.sh first)"; exit 1; }

RUN_ROOT="$RESULTS_HOST/lswb-replay/$TAG/$(ts)"
mkdir -p "$RUN_ROOT/client"

# Record the resolved server configuration and reject eager decode caused by
# an undersized graph setup before spending a replay window on it.
curl -fsS -m 30 "http://127.0.0.1:$PORT/server_info" > "$RUN_ROOT/server-info.json" || exit 1
python3 - "$RUN_ROOT" "$C" "${SERVER_SEED:-}" "${REQUEST_METRICS_DIR:-}" \
  "$HOST_REPO" "$REPO_CT" "$LOG_HOST" <<'PY_PREFLIGHT' || exit 1
import json
import sys
from pathlib import Path

run_root, concurrency, seed, expected_metrics, host_repo, repo_ct, log_host = sys.argv[1:]
run_root = Path(run_root)
with open(run_root / "server-info.json") as stream:
    info = json.load(stream)
concurrency = int(concurrency)
decode = info.get("cuda_graph_config", {}).get("decode", {})
sizes = decode.get("bs") or []
covered = any(size >= concurrency for size in sizes)
if info.get("disable_cuda_graph_padding", False):
    covered = concurrency in sizes
if decode.get("backend") != "full" or not covered:
    raise SystemExit(
        f"Replay c{concurrency} requires full decode graphs covering that batch; "
        f"server reports {decode}. Restart serve.sh with the extended graph config."
    )
print(f"[lswb] decode graph preflight OK: c{concurrency}, captured sizes={sizes}")
if seed and info.get("random_seed") != int(seed):
    raise SystemExit(f"Server seed mismatch: expected {seed}, got {info.get('random_seed')}")
mode = "native" if info.get("dsv4_c4_cache_format") == "native" else info.get("dsv4_c4_cache_implementation")
if mode not in ("native", "triton", "kernel"):
    raise SystemExit(f"Unknown server cache mode: {mode!r}")
metrics_dir = None
if info.get("export_metrics_to_file"):
    if not info.get("enable_metrics") or not info.get("log_requests") or info.get("log_requests_level") != 0:
        raise SystemExit("Request timing export requires --enable-metrics and metadata-only --log-requests-level 0")
    source = info.get("export_metrics_to_file_dir") or ""
    prefix = repo_ct.rstrip("/") + "/"
    if not source.startswith(prefix):
        raise SystemExit(f"Export directory must be under container repository {repo_ct}: {source}")
    metrics_dir = str((Path(host_repo) / source[len(prefix):]).resolve())
    if not Path(metrics_dir).is_relative_to(Path(host_repo).resolve()) or not Path(metrics_dir).is_dir():
        raise SystemExit(f"Export directory is not accessible under the host repository: {metrics_dir}")
    if expected_metrics and Path(expected_metrics).resolve() != Path(metrics_dir):
        raise SystemExit(f"Export directory mismatch: expected {expected_metrics}, got {metrics_dir}")
elif expected_metrics:
    raise SystemExit("Server request metrics export is disabled")
server_log = Path(log_host) / f"serve_{mode}.log"
if metrics_dir and not server_log.is_file():
    raise SystemExit(f"Request timing run requires the boot's server log: {server_log}")
(run_root / "collection-config.json").write_text(json.dumps({
    "schema_version": 1, "mode": mode, "server_seed": info.get("random_seed"),
    "metrics_dir": metrics_dir, "server_log": str(server_log),
}, indent=2) + "\n")
PY_PREFLIGHT

# Start every replay with an empty prefix cache, including c28 after c24 on
# the same server. Wait for outstanding requests to drain; do not launch the
# client or Nsight capture if any worker cannot flush. Model/graphs stay loaded.
echo "[lswb] waiting for idle server and flushing prefix cache"
if ! curl --fail-with-body -sS --max-time 330 -X POST \
    "http://127.0.0.1:$PORT/flush_cache?timeout=300" \
    > "$RUN_ROOT/cache-flush.log" 2>&1; then
  echo "FATAL: prefix cache flush failed; replay not started (see $RUN_ROOT/cache-flush.log)" >&2
  cat "$RUN_ROOT/cache-flush.log" >&2
  exit 1
fi
cat "$RUN_ROOT/cache-flush.log"

echo "== lswb replay tag=$TAG port=$PORT c$C @ ${DUR}s -> $RUN_ROOT =="
python3 - "$RUN_ROOT" <<'PY_WINDOW' || exit 1
import json
import time
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "collection-config.json").read_text())
log = Path(config["server_log"])
identity = None
if log.is_file():
    stat = log.stat()
    identity = {"device": stat.st_dev, "inode": stat.st_ino, "offset": stat.st_size}
(root / "replay-window.json").write_text(json.dumps({
    "started_at": time.time(), "server_log_identity": identity,
}, indent=2) + "\n")
PY_WINDOW

# Reuse SGLang's completed-request exporter. Keep raw files and validate only
# records received in this replay, allowing bounded time for async disk writes.
collect_artifacts () {
  python3 - "$RUN_ROOT" "$1" <<'PY_COLLECT'
import json
import math
import shutil
import sys
import time
from pathlib import Path

root, partial = Path(sys.argv[1]), sys.argv[2] == "partial"
config = json.loads((root / "collection-config.json").read_text())
window = json.loads((root / "replay-window.json").read_text())
window.setdefault("ended_at", time.time())
(root / "replay-window.json").write_text(json.dumps(window, indent=2) + "\n")
log, identity = Path(config["server_log"]), window["server_log_identity"]
def copy_server_log():
    errors = []
    if identity:
        try:
            stat = log.stat()
            if (stat.st_dev, stat.st_ino) != (identity["device"], identity["inode"]) or stat.st_size < identity["offset"]:
                raise ValueError("server log was replaced/truncated during replay")
            with log.open("rb") as source:
                source.seek(identity["offset"])
                (root / "server.log").write_bytes(source.read())
            text = (root / "server.log").read_text(errors="replace")
            if "Failed to write perf metrics to file" in text:
                errors.append("SGLang request exporter reported a write failure")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    elif config["metrics_dir"]:
        errors.append("missing server log identity")
    return errors

if not config["metrics_dir"]:
    errors = copy_server_log()
    if errors:
        raise SystemExit("; ".join(errors))
    sys.exit(0)

summary_path = root / "client" / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
expected = summary["counts"]["successful"] if summary else None
raw_dir = root / "request-metrics"
raw_dir.mkdir(exist_ok=True)
deadline = time.monotonic() + (0 if partial else 30)
previous = None
stable_since = time.monotonic()

def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

while True:
    files = sorted(Path(config["metrics_dir"]).glob("sglang-request-metrics-*.log"))
    fingerprint = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    now = time.monotonic()
    if fingerprint != previous:
        stable_since, previous = now, fingerprint
    records, errors, seen, valid = [], [], set(), 0
    for path in files:
        shutil.copyfile(path, raw_dir / path.name)
        for line_no, line in enumerate((raw_dir / path.name).read_text().splitlines(), 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                received = record.get("request_received_ts")
                if not finite_number(received):
                    raise ValueError("missing/nonfinite request_received_ts")
                if not window["started_at"] <= received <= window["ended_at"]:
                    continue
                rid = record.get("id")
                if not isinstance(rid, str) or not rid:
                    raise ValueError("missing request ID")
                if rid in seen:
                    errors.append(f"duplicate request ID: {rid}")
                seen.add(rid)
                records.append(record)
                finish = record.get("finish_reason")
                if not isinstance(finish, dict):
                    errors.append(f"{rid}: missing finish reason")
                    continue
                if finish.get("type") == "abort":
                    continue  # Preserve failures; they need not have first output.
                fields = ("request_received_ts", "forward_entry_time", "prefill_finished_time",
                          "first_output_ts", "request_finished_ts")
                stamps = [record.get(k) for k in fields]
                queue = record.get("queue_time")
                if not all(finite_number(v) and v > 0 for v in stamps) or not finite_number(queue) or queue < 0:
                    errors.append(f"{rid}: incomplete/nonfinite/negative request timing")
                elif stamps != sorted(stamps):
                    errors.append(f"{rid}: nonmonotonic timing timestamps")
                else:
                    valid += 1
            except (ValueError, TypeError) as exc:
                errors.append(f"{path.name}:{line_no}: {exc}")
    enough = expected is not None and expected > 0 and valid >= expected
    if partial or now >= deadline or (enough and now - stable_since >= 2):
        break
    time.sleep(0.5)

errors.extend(copy_server_log())  # Include async write errors emitted while settling.
(root / "request-timings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
if not enough:
    errors.append(f"valid exported records {valid} < client successful count {expected}, or missing/empty client summary")
validation = {
    "schema_version": 1, "complete": not partial and not errors,
    "server_seed": config["server_seed"], "mode": config["mode"],
    "records": len(records), "valid_timing_records": valid,
    "client_successful": expected, "errors": errors,
    "coverage_basis": "count lower bound; not a verified per-request client/server join",
    "client_join_verified": False, "partial": partial,
}
(root / "request-timing-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
print(f"[lswb] request timing: {len(records)} records, {valid} valid, complete={validation['complete']}")
if errors and not partial:
    raise SystemExit("Request timing collection failed: " + "; ".join(errors[:8]))
PY_COLLECT
}

SAMPLER=
CLIENT=
PROFILER=
cleanup () {
  [ -z "$PROFILER" ] || kill "$PROFILER" 2>/dev/null || true
  [ -z "$PROFILER" ] || wait "$PROFILER" 2>/dev/null || true
  [ -z "$SAMPLER" ] || kill "$SAMPLER" 2>/dev/null || true
  [ -z "$CLIENT" ] || kill "$CLIENT" 2>/dev/null || true
  [ -z "$SAMPLER" ] || wait "$SAMPLER" 2>/dev/null || true
  [ -z "$CLIENT" ] || wait "$CLIENT" 2>/dev/null || true
  collect_artifacts partial || true
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

( cd "$REPLAY_DIR" && exec "${PYTHON:-python3}" -B "$REPLAY_RUNNER" \
    --result-root "$RUN_ROOT/client" \
    --dataset-root "$REPLAY_DATASET" \
    --dataset-manifest-input "$REPLAY_MANIFEST" \
    --adapter "$REPLAY_ADAPTER" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL_NAME" \
    --max-requests 4916 --max-concurrency "$C" --max-duration "$DUR" \
    --arrival-mode immediate --time-scale 60 --max-gap 30 \
    --timeout 21600 --minimum-success-rate 0.99 \
    --expected-protocol business-user-replay-v2 \
    --audit-level candidate --return-cached-tokens-details \
    > "$RUN_ROOT/client.log" 2>&1 ) &
CLIENT=$!

# Capture offsets are measured from client launch, independently of trace export.
# The child owns profiler API calls and always closes an active range on exit.
if [ -n "$NSYS_WINDOWS" ]; then
  python3 - "$PORT" "$NSYS_WINDOWS" "$DUR" > "$RUN_ROOT/nsys-windows.jsonl" 2>&1 <<'PY_CAPTURE' &
import json
import signal
import sys
import threading
import time
import urllib.request

port, specification, duration = sys.argv[1:]
origin = time.monotonic()
cancelled = threading.Event()
active = False
signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
signal.signal(signal.SIGINT, lambda *_: cancelled.set())

def profile(action):
    payload = {"activities": ["CUDA_PROFILER"]} if action == "start" else {}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{action}_profile",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # cudaProfilerStop may block while Nsight exports the completed range.
    # Keep a bounded timeout without mistaking a successful export for failure.
    requested_at = time.monotonic() - origin
    with urllib.request.urlopen(request, timeout=180) as response:
        response.read()
    print(json.dumps({"action": action, "requested_s": requested_at,
                      "elapsed_s": time.monotonic() - origin}), flush=True)

try:
    windows = [tuple(map(int, pair.split(":"))) for pair in specification.split(",")]
    previous_end = 0
    for start, end in windows:
        if not previous_end <= start < end <= int(duration):
            raise ValueError(f"Invalid ordered capture window {start}:{end} for duration {duration}")
        previous_end = end
    for start, end in windows:
        if cancelled.wait(max(0, origin + start - time.monotonic())):
            raise RuntimeError("Benchmark ended before all capture windows completed")
        active = True  # Even a timed-out start may have reached the server.
        profile("start")
        if cancelled.wait(max(0, origin + end - time.monotonic())):
            raise RuntimeError("Benchmark ended during capture")
        profile("stop")
        active = False
finally:
    if active:
        profile("stop")
PY_CAPTURE
  PROFILER=$!
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  ( while kill -0 "$CLIENT" 2>/dev/null; do
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        >> "$RUN_ROOT/gpu-samples.csv" 2>/dev/null || true
      kill -0 "$CLIENT" 2>/dev/null || break
      sleep 20
    done ) &
  SAMPLER=$!
else
  echo "[lswb] warning: nvidia-smi not found; GPU sampling disabled" >&2
fi

wait "$CLIENT"
RC=$?
CLIENT=
if [ -n "$PROFILER" ]; then
  kill "$PROFILER" 2>/dev/null || true
  wait "$PROFILER"
  PROFILE_RC=$?
  PROFILER=
  [ "$PROFILE_RC" = 0 ] || { echo "[lswb] capture failed; see $RUN_ROOT/nsys-windows.jsonl" >&2; RC=1; }
fi
[ -z "$SAMPLER" ] || kill "$SAMPLER" 2>/dev/null || true
[ -z "$SAMPLER" ] || wait "$SAMPLER" 2>/dev/null || true
SAMPLER=
if ! collect_artifacts full; then
  RC=1
fi
trap - EXIT INT TERM

echo "[lswb] client rc=$RC"; echo "==== client.log ===="; cat "$RUN_ROOT/client.log"
echo "[lswb] artifacts at $RUN_ROOT"
exit $RC
