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

echo "== lswb replay tag=$TAG port=$PORT c$C @ ${DUR}s -> $RUN_ROOT =="
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
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

( cd "$REPLAY_DIR" && "${PYTHON:-python3}" -B "$REPLAY_RUNNER" \
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
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
    print(json.dumps({"action": action, "elapsed_s": time.monotonic() - origin}), flush=True)

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
trap - EXIT INT TERM

echo "[lswb] client rc=$RC"; echo "==== client.log ===="; cat "$RUN_ROOT/client.log"
echo "[lswb] artifacts at $RUN_ROOT"
exit $RC
