#!/usr/bin/env bash
# =====================================================================
# eval-lb2.sh <tag> [port] [out.json]
#   LongBench v2 full eval against an ALREADY-RUNNING server (bring it up
#   with serve.sh native|packed). Runs every feasible sample of the official
#   503-question set over the server's /v1/chat/completions endpoint.
#
#   The 30 samples whose prompt exceeds the 1M context cap are dropped (the
#   set's longest is 4.6M tokens -- truncation would bias the answer).
#   473 samples remain (~65M prompt tokens). Resumable: rerun with the same
#   --out and finished samples are skipped.
#
#   tag     = results label (e.g. native-0731, packed-0731)
#   port    = server port, default $PORT (30212)
#   out     = results json (default <RESULTS_HOST>/lb2-full/<tag>/<ts>/results.json)
#
# Client = lb2_serve_eval.py (this folder); runs on the host, concurrency 6.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

TAG=${1:-} PORT=${2:-$PORT} OUT=${3:-}
[ -n "$TAG" ] || { echo "usage: $0 <tag> [port] [out.json]"; exit 1; }

curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "FATAL: no server on 127.0.0.1:$PORT (run serve.sh first)"; exit 1; }

if [ -z "$OUT" ]; then
  OUT="$RESULTS_HOST/lb2-full/$TAG/$(ts)/results.json"
fi

echo "== LongBench v2 full: tag=$TAG server=127.0.0.1:$PORT -> $OUT =="
python3 "$(dirname "$0")/lb2_serve_eval.py" \
  --server "http://127.0.0.1:$PORT" --model "$MODEL_NAME" --tag "$TAG" \
  --data "$LB2_DATA" --tokens "$LB2_TOKENS" --out "$OUT"
echo "[lb2] results at $OUT"
