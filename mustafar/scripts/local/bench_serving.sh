#!/usr/bin/env bash
# Usage: MODEL_PATH=/weights SGLANG_ROOT=/sglang bash bench_serving.sh [mode] [input] [output] [concurrency]
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: MODEL_PATH=/weights SGLANG_ROOT=/sglang bash $0 [native|packed|packed_fused] [32768] [2048] [8]"
  echo "Optional env: PYTHON, RESULTS_DIR, PORT. Fixed TP4, one warm-up and one measured wave."
  exit 0
fi
mode=${1:-native}; input=${2:-32768}; output=${3:-2048}; concurrency=${4:-8}
case "$mode" in native|packed|packed_fused) ;; *) echo "Unknown mode: $mode" >&2; exit 2 ;; esac
for n in "$input" "$output" "$concurrency"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "Counts and limits must be positive integers" >&2; exit 2; }
done
(( $# <= 4 && concurrency <= 16 )) || { echo "Expected four arguments and concurrency <= 16" >&2; exit 2; }
: "${MODEL_PATH:?Set MODEL_PATH to the official 0731 checkpoint}"
for tool in "${PYTHON:-python3}" curl jq setsid; do command -v "$tool" >/dev/null; done
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
if [[ -n "${SGLANG_ROOT:-}" ]]; then
  export SG_LOWRANK_SRC="$SGLANG_ROOT/python"
  export PYTHONPATH="$SG_LOWRANK_SRC:$PYTHONPATH"
fi
for name in ${!SGLANG_OPT_TOPMAG@} ${!XKV_TOPMAG@}; do unset "$name"; done
export SGLANG_OPT_TOPMAG=0 XKV_TOPMAG_KEEP=1.0 SGLANG_OPT_TOPMAG_PACKED_C4=0 SGLANG_OPT_TOPMAG_STAGE2A=0
if [[ "$mode" != native ]]; then
  export SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=0.5 SGLANG_OPT_TOPMAG_PACKED_C4=1
fi
[[ "$mode" != packed_fused ]] || export SGLANG_OPT_TOPMAG_STAGE2A=1
python=${PYTHON:-python3}; port=${PORT:-30211}
if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
  echo "A server is already running on port $port; choose a different PORT" >&2
  exit 1
fi
mkdir -p "${RESULTS_DIR:-$repo/mustafar/logs/bench-serving}"
run=$(mktemp -d "${RESULTS_DIR:-$repo/mustafar/logs/bench-serving}/$(date -u +%Y%m%dT%H%M%SZ)-$mode-XXXXXX")
echo "Results: $run"
server_pid=; bench_pid=
cleanup() {
  for pid in "$bench_pid" "$server_pid"; do
    [[ -z "$pid" ]] || kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "$bench_pid" "$server_pid"; do
    [[ -z "$pid" ]] || kill -KILL -- "-$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Fixed serving setup; edit here to change it for both local and Modal runs.
graph='{"decode":{"backend":"full","max_bs":16,"bs":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},"prefill":{"backend":"disabled"}}'
setsid "$python" -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name deepseek-v4-flash-0731 \
  --tp 4 --trust-remote-code --moe-runner-backend flashinfer_mxfp4 \
  --mem-fraction-static 0.90 --context-length "$((input + output + 2048))" \
  --max-running-requests 16 --chunked-prefill-size 4096 \
  --swa-full-tokens-ratio 0.1 --page-size 256 --host 127.0.0.1 --port "$port" \
  --cuda-graph-config "$graph" --skip-server-warmup --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 --watchdog-timeout 1800 --random-seed 7301 \
  >"$run/server.log" 2>&1 &
server_pid=$!
until curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
  kill -0 "$server_pid" 2>/dev/null || { tail -n 40 "$run/server.log"; exit 1; }
  sleep 2
done
for phase in warmup measured; do
  setsid "$python" -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port "$port" \
    --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --dataset-name random \
    --random-input-len "$input" --random-output-len "$output" --random-range-ratio 1.0 \
    --max-concurrency "$concurrency" --num-prompts "$concurrency" --request-rate inf \
    --warmup-requests 0 --flush-cache --tokenize-prompt --output-details --seed 7301 \
    --output-file "$run/$phase.jsonl" >"$run/$phase.log" 2>&1 &
  bench_pid=$!
  wait "$bench_pid"
  bench_pid=
  cat "$run/$phase.log"
  jq -se --argjson n "$concurrency" --argjson input "$input" --argjson output "$output" \
    'last | .completed == $n and (.errors | length) == $n and all(.errors[]; . == "")
     and (.input_lens | length) == $n and all(.input_lens[]; . == $input)
     and (.output_lens | length) == $n and all(.output_lens[]; . == $output)' \
    "$run/$phase.jsonl" >/dev/null
done
echo "Completed: $run/measured.jsonl"
