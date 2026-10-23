#!/usr/bin/env bash
# Author: Winston Cai. C28 frontend off/on comparison for one Packed mode.
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")" && pwd)
REPO=$(cd -- "$ROOT/../.." && pwd)
cd "$REPO"

MODE=${1:?Pass triton or kernel}
case "$MODE" in
  triton) export PORT=30641 MASTER_PORT=30041 ;;
  kernel) export PORT=30642 MASTER_PORT=30042 ;;
  *) echo "Expected triton or kernel"; exit 2 ;;
esac

export PYTHONDONTWRITEBYTECODE=1 GIT_CONFIG_COUNT=4
paths=("$REPO" "$REPO/third_party/sglang" "$REPO/third_party/flashmla" "$REPO/third_party/flashmla/csrc/cutlass")
for i in 0 1 2 3; do
  export "GIT_CONFIG_KEY_$i=safe.directory" "GIT_CONFIG_VALUE_$i=${paths[$i]}"
done

export SGLANG_ROOT_CT="/mnt/host_root$REPO/third_party/sglang"
export SGLANG_PY="$SGLANG_ROOT_CT/python"
. scripts/local/env.sh

export REPLAY_RUNNER="$REPO/scripts/local/business_replay/runner.py"
export REPLAY_ADAPTER="$REPO/scripts/local/business_replay/adapters/openai_sse.py"
. scripts/local/eval-env.sh
require_replay_config

: "${CUDA_VISIBLE_DEVICES:?GPUQ must assign devices}"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[ "${#devices[@]}" -eq 4 ] || exit 1
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$ROOT/assigned-gpus-$MODE.txt"

export GPUS="$CUDA_VISIBLE_DEVICES" TP=4 MEM_FRAC=0.88
export SERVER_SEED=20260919 CTX_LEN=1048576 MAX_RUN=256 CHUNK=8192
export C4_KEEP_COUNT=256 FRONTEND_DIAGNOSTICS=1
export DECODE_CFG="$DECODE_CFG_EXT" STORE_IMPLEMENTATION="$MODE"
export REPLAY_SCHEDULER=fixed-active-user-pool

unset HICACHE_RATIO C4_LAYER_KEEP_COUNTS
unset REPLAY_SESSION_POPULATION REPLAY_DRAIN_TIMEOUT REPLAY_TTFT_SLO_MS
unset NSYS_OUTPUT REMNANT_CAPTURE_DIR

active=
failed=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ -n "$active" ]; then
    bash scripts/local/serve.sh "$active" stop || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for prep in 0 1; do
  export DSV4_ASYNC_PREPARATION=$prep
  CELL="$MODE-async$prep"
  export LOG_HOST="$ROOT/$CELL"
  export REQUEST_METRICS_DIR="$LOG_HOST/request-metrics"
  export SERVER_PID_FILE_CT="/tmp/remnant-$(basename "$ROOT")-$CELL.pid"
  mkdir "$LOG_HOST"

  [ "$(git rev-parse HEAD)" = "$(cat "$ROOT/parent-commit.txt")" ]
  [ -z "$(git status --porcelain --untracked-files=no)" ]
  sha256sum -c "$ROOT/launcher-sha256.txt"
  [ "$(docker inspect "$CONTAINER" --format '{{.Image}}')" = "$(cat "$ROOT/image-id.txt")" ]
  check_sources > "$LOG_HOST/source-manifest.json"

  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "Port $PORT already serves another process" >&2
    exit 1
  fi

  active=$MODE
  bash scripts/local/serve.sh "$MODE"

  curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" \
    > "$LOG_HOST/server-info.json"

  python3 - "$LOG_HOST/server-info.json" "$MODE" <<'PY_CONFIG'
import json, sys
info = json.load(open(sys.argv[1]))
expected = dict(
    dsv4_c4_cache_format="remnant",
    dsv4_c4_cache_keep_count=256,
    dsv4_c4_cache_layer_keep_counts=None,
    tp_size=4, mem_fraction_static=0.88, random_seed=20260919,
    enable_hierarchical_cache=False, chunked_prefill_size=8192,
    max_running_requests=256, tokenizer_worker_num=1,
    detokenizer_worker_num=1,
    dsv4_c4_cache_implementation=sys.argv[2],
    dsv4_c4_cache_store_implementation=sys.argv[2],
)
for key, value in expected.items():
    if key not in info or info[key] != value:
        raise SystemExit(f"{key}: expected {value!r}, got {info.get(key)!r}")
PY_CONFIG

  if [ "$prep" = 1 ]; then
    grep -q 'DSV4 asynchronous input preparation enabled' \
      "$LOG_HOST/serve_$MODE.log"
  elif grep -q 'DSV4 asynchronous input preparation enabled' \
      "$LOG_HOST/serve_$MODE.log"; then
    echo "Unexpected worker in synchronous control" >&2
    exit 1
  fi

  curl --fail-with-body -sS --max-time 30 -X POST \
    "http://127.0.0.1:$PORT/freeze_gc" > "$LOG_HOST/gc-freeze.log"

  TAG="$MODE-k256-c28-mf088-$CELL-$(basename "$ROOT")"
  RESULT="$RESULTS_HOST/lswb-replay/$TAG"
  printf '%s\n' "$RESULT" > "$LOG_HOST/result-root.txt"

  set +e
  timeout --foreground --kill-after=60 1800 \
    bash scripts/local/bench-lswb.sh "$TAG" "$PORT" 28 1200
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$LOG_HOST/replay-exit-code.txt"

  python3 - "$RESULT" "$REPO" <<'PY_ANALYZE'
import importlib.util, json, sys
from pathlib import Path

root, repo = map(Path, sys.argv[1:])
runs = list(root.glob("*/replay-window.json"))
if len(runs) != 1:
    raise SystemExit(f"Expected one replay under {root}, got {len(runs)}")
run = runs[0].parent
spec = importlib.util.spec_from_file_location(
    "a", repo / "benchmarks/lswb-replay/c24-packed256-store-decode-ablation-n1-20261014/analyze.py"
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
s = a.read(run / "client/summary.json")
p = a.first_output_pipeline_diagnostics(
    a.lines(run / "client/responses.jsonl"),
    a.lines(run / "request-timings.jsonl"),
    a.read(run / "replay-window.json")["started_at"],
)
p["post_180s_sample_gate"] = p["cohorts"]["after_180s"]["count"] >= 200
p["response_summary_valid"] = s["valid"]
(run / "first-output-pipeline.json").write_text(json.dumps(p, indent=2) + "\n")
print(run, "pipeline_valid=", p["valid"], "response_valid=", s["valid"],
      "req/s=", s["request_throughput"], "TTFT-p90-ms=", s["ttft_ms"]["p90"])
if not p["valid"] or not p["post_180s_sample_gate"]:
    raise SystemExit("Diagnostic coverage failed; stopping remaining cells")
PY_ANALYZE

  if [ "$rc" -ne 0 ]; then
    [ "$rc" -eq 1 ] || exit "$rc"
    python3 - "$RESULT" <<'PY_GATE'
import json, sys
from pathlib import Path
files = list(Path(sys.argv[1]).glob("*/client/summary.json"))
if len(files) != 1 or json.loads(files[0].read_text())["valid"] is not False:
    raise SystemExit("Unexpected replay failure; stopping remaining cells")
PY_GATE
    failed=1
  fi

  bash scripts/local/serve.sh "$MODE" stop
  active=
done

echo "Both preparation settings finished; response failures are preserved."
exit "$failed"
