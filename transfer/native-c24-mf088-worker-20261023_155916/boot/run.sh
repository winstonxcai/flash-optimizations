#!/usr/bin/env bash
# Author: Winston Cai. Native C24 preparation-worker replay.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=$(cd -- "$ROOT/../.." && pwd)
cd "$REPO"

export PYTHONDONTWRITEBYTECODE=1 GIT_CONFIG_COUNT=4
trees=("$REPO" "$REPO/third_party/sglang" \
       "$REPO/third_party/flashmla" "$REPO/third_party/flashmla/csrc/cutlass")
for i in 0 1 2 3; do
  export "GIT_CONFIG_KEY_$i=safe.directory" "GIT_CONFIG_VALUE_$i=${trees[$i]}"
done

[ "$(git rev-parse HEAD)" = "$(cat "$ROOT/parent-commit.txt")" ]
[ "$(docker inspect "$CONTAINER" --format '{{.Image}}')" = "$(cat "$ROOT/image-id.txt")" ]
sha256sum -c "$ROOT/launcher-sha256.txt"

export SGLANG_ROOT_CT="/mnt/host_root$REPO/third_party/sglang"
export SGLANG_PY="$SGLANG_ROOT_CT/python"
export PORT=30651 MASTER_PORT=30051
export GPUS="${CUDA_VISIBLE_DEVICES:?GPUQ must assign devices}"
IFS=, read -r -a devices <<< "$GPUS"
[ "${#devices[@]}" -eq 4 ]

export TP=4 MEM_FRAC=0.88
export SERVER_SEED=20260919 CTX_LEN=1048576 MAX_RUN=256 CHUNK=8192
export DSV4_ASYNC_PREPARATION=1 FRONTEND_DIAGNOSTICS=1
export LOG_HOST="$ROOT"
export REQUEST_METRICS_DIR="$ROOT/request-metrics"
export SERVER_PID_FILE_CT="/tmp/$(basename "$ROOT").pid"
export REPLAY_RUNNER="$REPO/scripts/local/business_replay/runner.py"
export REPLAY_ADAPTER="$REPO/scripts/local/business_replay/adapters/openai_sse.py"
export REPLAY_SCHEDULER=fixed-active-user-pool

unset HICACHE_RATIO STORE_IMPLEMENTATION C4_KEEP_COUNT C4_LAYER_KEEP_COUNTS
unset REPLAY_SESSION_POPULATION REPLAY_DRAIN_TIMEOUT REPLAY_TTFT_SLO_MS
unset NSYS_OUTPUT REMNANT_CAPTURE_DIR

. scripts/local/env.sh
. scripts/local/eval-env.sh
export DECODE_CFG="$DECODE_CFG_EXT"
require_replay_config
check_sources > "$ROOT/runtime-source-manifest.json"

if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "Port $PORT already occupied; refusing to replace another server" >&2
  exit 1
fi

active=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$active" = 1 ]; then
    bash scripts/local/serve.sh native stop || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

active=1
bash scripts/local/serve.sh native

grep -q 'DSV4 asynchronous input preparation enabled' "$ROOT/serve_native.log"
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" > "$ROOT/server-info.json"

python3 - "$ROOT/server-info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
expected = {
    "dsv4_c4_cache_format": "native",
    "tp_size": 4,
    "mem_fraction_static": 0.88,
    "enable_hierarchical_cache": False,
    "random_seed": 20260919,
    "chunked_prefill_size": 8192,
    "max_running_requests": 256,
}
for key, value in expected.items():
    if info.get(key) != value:
        raise SystemExit(f"{key}: expected {value!r}, got {info.get(key)!r}")
print("Native worker configuration PASS")
PY

curl --fail-with-body -sS --max-time 30 -X POST \
  "http://127.0.0.1:$PORT/freeze_gc" > "$ROOT/gc-freeze.log"

rc=0
timeout --foreground --kill-after=60 1800 \
  bash scripts/local/bench-lswb.sh "$(basename "$ROOT")" "$PORT" 24 1200 || rc=$?
printf '%s\n' "$rc" > "$ROOT/replay-exit-code.txt"
exit "$rc"
