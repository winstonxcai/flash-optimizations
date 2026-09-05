#!/usr/bin/env bash
# =====================================================================
# bench-lswb.sh <tag> [port] [concurrency] [duration_s]
#   LongSWE-Bench replay client against an ALREADY-RUNNING server (bring it
#   up first with serve.sh native|packed). Replays 4,916 recorded Claude
#   business conversations (~144k prompt tok/req, short decodes) over OpenAI
#   SSE for a fixed window -- a prefix-reuse workload that shows whether the
#   larger packed KV pool retains more shared prefixes (device cache hits).
#
#   tag   = results label (e.g. native, packed)
#   port  = server port, default $PORT (30212)
#   C,DUR = concurrency and window (default 15 / 1200 s, the official heaviest
#           point used in the report)
#
# Runs the business_replay runner on THIS host (host python3 has `requests`).
# Results: <RESULTS_HOST>/lswb-replay/<tag>/<ts>/  (+ client.log, gpu-samples).
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

TAG=${1:-} PORT=${2:-$PORT} C=${3:-15} DUR=${4:-1200}
[ -n "$TAG" ] || { echo "usage: $0 <tag> [port] [concurrency] [duration_s]"; exit 1; }

health () { curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
health || { echo "FATAL: no server on 127.0.0.1:$PORT (run serve.sh first)"; exit 1; }

RUN_ROOT="$RESULTS_HOST/lswb-replay/$TAG/$(ts)"
mkdir -p "$RUN_ROOT/client"

echo "== lswb replay tag=$TAG port=$PORT c$C @ ${DUR}s -> $RUN_ROOT =="
# sample GPUs during the client window
( for _ in $(seq 1 300); do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$RUN_ROOT/gpu-samples.csv"
    sleep 20
  done ) &
SAMPLER=$!

( cd "$REPLAY_DIR" && /usr/bin/python3 -B "$REPLAY_RUNNER" \
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
    > "$RUN_ROOT/client.log" 2>&1 )
RC=$?
kill "$SAMPLER" 2>/dev/null

echo "[lswb] client rc=$RC"; echo "==== client.log ===="; cat "$RUN_ROOT/client.log"
echo "[lswb] artifacts at $RUN_ROOT"
exit $RC
