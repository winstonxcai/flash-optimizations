#!/usr/bin/env bash
# Standalone serving benchmark used by the production image and Modal.
set -u

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DECODE_CFG_WAS_SET=${DECODE_CFG+x}
. "$DIR/env.sh"
REPO=$HOST_REPO
[ -n "$DECODE_CFG_WAS_SET" ] || export DECODE_CFG="$DECODE_CFG_EXT"

ts () { date +%Y%m%d_%H%M%S; }
die () { echo "FATAL: $*" >&2; exit 1; }
usage () {
  cat >&2 <<'EOF'
usage: bench-serving-image.sh <native|packed> <input_tokens> <output_tokens> <concurrency>

The server is started and stopped inside this process group. MODEL_PATH must
point at the official DeepSeek-V4-Flash-0731 checkpoint.
EOF
}
usage_err () { echo "bench-serving-image.sh: $*" >&2; usage; exit 2; }

server_pid=
bench_pid=
cleanup () {
  [ -n "$bench_pid$server_pid" ] || return 0
  for pid in "$bench_pid" "$server_pid"; do
    [ -z "$pid" ] || kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "$bench_pid" "$server_pid"; do
    [ -z "$pid" ] || kill -KILL -- "-$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

validate () {
  python3 - "$1" "$((3 * $2))" "$3" <<'PY'
import json, sys
record = None
with open(sys.argv[1]) as stream:
    for line in stream:
        if line.strip():
            record = json.loads(line)
expected = int(sys.argv[2])
outlen = int(sys.argv[3])
if record is None:
    print("  valid: no benchmark summary found")
    sys.exit(1)
valid = (
    record.get("completed") == expected
    and all(n == outlen for n in (record.get("output_lens") or []))
    and not any(record.get("errors") or [])
)
print(f"  valid: completed={record.get('completed')} expected={expected} all_outlen={valid}")
sys.exit(0 if valid else 1)
PY
}

summary_line () {
  awk '
    /Request throughput \(req\/s\):/     {r=$NF}
    /Total token throughput \(tok\/s\):/ {t=$NF}
    /Median TTFT \(ms\):/                {tt=$NF}
    /Median TPOT \(ms\):/                {tp=$NF}
    /Median E2E Latency \(ms\):/         {e=$NF}
    END {printf "req/s=%.4f tok/s=%.1f ttft_ms=%.1f tpot_ms=%.1f e2e_ms=%.1f", r,t,tt,tp,e}' "$1"
}

solo_wave () {
  setsid "$python" -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port "$port" \
    --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --dataset-name random \
    --random-input-len "$input" --random-output-len "$output" \
    --random-range-ratio 1.0 --num-prompts "$3" --max-concurrency "$conc" \
    --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
    --output-file "$1" --output-details --seed "$seed" > "$2" 2>&1 &
  bench_pid=$!
  wait "$bench_pid"
  local rc=$?
  bench_pid=
  return "$rc"
}

main () {
  if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    usage
    return 0
  fi
  mode=${1:-native}; input=${2:-32768}; output=${3:-2048}; conc=${4:-8}
  (( $# <= 4 )) || usage_err "expected at most 4 arguments"
  case "$mode" in native|packed) ;; *) usage_err "mode must be native or packed" ;; esac
  for value in "$input" "$output" "$conc"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || usage_err "counts must be positive integers"
  done
  (( conc <= 136 )) || usage_err "concurrency cannot exceed 136"
  : "${MODEL_PATH:?Set MODEL_PATH to the official DeepSeek-V4-Flash-0731 checkpoint}"
  for tool in "${PYTHON:-python3}" curl setsid; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
  done

  python=${PYTHON:-python3}
  port=${PORT:-30211}
  seed=${SEED:-42}
  results_dir=${RESULTS_DIR:-$REPO/logs/bench-serving}
  export SG_LOWRANK_SRC="$SGLANG_ROOT/python"
  export PYTHONPATH="/opt/sglang-runtime-fixes:$SG_LOWRANK_SRC:$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1

  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    die "a server is already running on port $port; set a different PORT"
  fi
  mkdir -p "$results_dir"
  run=$(mktemp -d "$results_dir/$(ts)-$mode-XXXXXX") || die "could not create run directory"
  echo "Results: $run"
  echo "boot: mode=$mode tp=${TP:-4} mem=${MEM_FRAC:-0.88} ctx=${CTX_LEN:-1048576} max_run=${MAX_RUN:-256} chunk=${CHUNK:-8192} fp8_kv port=$port"

  setsid "$REPO/scripts/local/run-server.sh" "$mode" "$MODEL_PATH" 127.0.0.1 "$port" >"$run/server.log" 2>&1 &
  server_pid=$!
  local n=0
  until curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    n=$((n + 1))
    kill -0 "$server_pid" 2>/dev/null || { tail -n 40 "$run/server.log" >&2; die "server failed to start"; }
    [ "$n" -lt 240 ] || { tail -n 40 "$run/server.log" >&2; die "server not healthy after ~20 min"; }
    sleep 5
  done
  echo "server UP on port $port (pid $server_pid)"

  solo_wave "$run/warmup.jsonl" "$run/warmup.log" "$conc" || die "warmup wave failed"
  curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 || die "server down after warmup"
  solo_wave "$run/measured.jsonl" "$run/measured.log" "$((3 * conc))" || die "measured wave failed"
  validate "$run/measured.jsonl" "$conc" "$output" || {
    echo "standalone point FAILED (validation): $run/measured.jsonl" >&2
    return 1
  }
  echo "  measured: $(summary_line "$run/measured.log")"
  echo "Completed: $run/measured.jsonl"
}

main "$@"
