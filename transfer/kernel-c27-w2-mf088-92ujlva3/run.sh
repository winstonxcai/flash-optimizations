#!/usr/bin/env bash
# Author: Winston Cai.
set -euo pipefail
cd -- "$1"
BATCH=$2

export GIT_CONFIG_COUNT=4
trees=("$PWD" "$PWD/third_party/sglang" "$PWD/third_party/flashmla" "$PWD/third_party/flashmla/csrc/cutlass")
for i in 0 1 2 3; do
  export "GIT_CONFIG_KEY_$i=safe.directory" "GIT_CONFIG_VALUE_$i=${trees[$i]}"
done

export HOST_REPO="$PWD" REPO_CT="/mnt/host_root$PWD"
export SGLANG_ROOT_CT="$REPO_CT/third_party/sglang"
export SGLANG_PY="$SGLANG_ROOT_CT/python"
. scripts/local/env.sh
. scripts/local/eval-env.sh

check_sources > "$BATCH/source-at-start.json"
require_replay_config
git diff --quiet "$(cat "$BATCH/parent-commit.txt")" -- scripts/local third_party
[ -z "$(git status --porcelain --untracked-files=no)" ]
sha256sum -c "$BATCH/launcher-sha256.txt"
[ "$(docker inspect "$CONTAINER" --format '{{.Image}}')" = "$(cat "$BATCH/image-id.txt")" ]

export GPUS="${CUDA_VISIBLE_DEVICES:?GPUQ must assign devices}" TP=4
IFS=, read -r -a devices <<< "$GPUS"
[ "${#devices[@]}" -eq 4 ] || exit 2

export PORT=30707 MASTER_PORT=30107 MEM_FRAC=0.88 SERVER_SEED=20260919
export CTX_LEN=1048576 MAX_RUN=256 CHUNK=8192 DECODE_CFG="$DECODE_CFG_EXT"
export C4_KEEP_COUNT=256 STORE_IMPLEMENTATION=kernel
export DSV4_ASYNC_PREPARATION=1 DSV4_PREPARATION_WORKERS=2 FRONTEND_DIAGNOSTICS=1
export LOG_HOST="$BATCH/boot" REQUEST_METRICS_DIR="$BATCH/boot/request-metrics"
export SERVER_PID_FILE_CT="/tmp/$(basename "$BATCH").pid"
export REPLAY_RUNNER="$PWD/scripts/local/business_replay/runner.py"
export REPLAY_ADAPTER="$PWD/scripts/local/business_replay/adapters/openai_sse.py"
export REPLAY_SCHEDULER=fixed-active-user-pool
unset HICACHE_RATIO C4_LAYER_KEEP_COUNTS NSYS_OUTPUT REMNANT_CAPTURE_DIR
unset REPLAY_SESSION_POPULATION REPLAY_DRAIN_TIMEOUT REPLAY_TTFT_SLO_MS
printf '%s\n' "$GPUS" > "$BATCH/assigned-gpus.txt"

started=0
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [ "$started" -eq 1 ]; then
    bash scripts/local/serve.sh kernel stop || status=1
  fi
  printf '%s\n' "$status" > "$BATCH/job-exit-code.txt"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "Port $PORT already occupied" >&2
  exit 1
fi

started=1
bash scripts/local/serve.sh kernel
grep -Fq 'DSV4 asynchronous input preparation enabled: 2 CPU worker(s)' \
  "$LOG_HOST/serve_kernel.log"

curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" \
  > "$LOG_HOST/server-info.json"

python3 - "$LOG_HOST/server-info.json" <<'CONFIG'
import json, sys
info = json.load(open(sys.argv[1]))
expected = dict(
    tp_size=4, mem_fraction_static=.88, enable_hierarchical_cache=False,
    random_seed=20260919, chunked_prefill_size=8192, max_running_requests=256,
    dsv4_c4_cache_format="remnant", dsv4_c4_cache_implementation="kernel",
    dsv4_c4_cache_store_implementation="kernel", dsv4_c4_cache_keep_count=256,
    dsv4_c4_cache_layer_keep_counts=None,
)
for key, value in expected.items():
    if key not in info or info[key] != value:
        raise SystemExit(f"{key}: expected {value!r}, got {info.get(key)!r}")
CONFIG

curl --fail-with-body -sS --max-time 30 -X POST \
  "http://127.0.0.1:$PORT/freeze_gc" > "$LOG_HOST/gc-freeze.log"

TAG=$(basename "$BATCH")
printf '%s\n' "$RESULTS_HOST/lswb-replay/$TAG" > "$BATCH/result-root.txt"

set +e
timeout --foreground --kill-after=60 1800 \
  bash scripts/local/bench-lswb.sh "$TAG" "$PORT" 27 1200
status=$?
set -e
printf '%s\n' "$status" > "$LOG_HOST/replay-exit-code.txt"
exit "$status"
