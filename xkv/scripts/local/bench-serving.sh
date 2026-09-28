#!/usr/bin/env bash
# =====================================================================
# bench-serving.sh -- one bench driver, two interfaces.
#
#   1) Report-grade protocol inside the xkv container:
#        bench-serving.sh fair    <ctx> [C]     native vs xkv, SAME C
#        bench-serving.sh graphab <ctx> [C]     xkv, decode graphs full vs off
#      Each leg boots with serve.sh inside $CONTAINER (sources env.sh), so a
#      `fair` run leaves two fresh-boot points and `graphab` leaves the two arms
#      of the decode-graph A/B, all at the same concurrency and context.
#
#   2) Standalone single-config measurement on the current host:
#        MODEL_PATH=/checkpoint bash bench-serving.sh <native|xkv> <in> <out> <C>
#
# Per point both paths share the report protocol: fresh server, warm-up wave of
# C, then 3 measured waves (3C requests) via official sglang.bench_serving,
# flush-cache, seed 42, exact <ctx|in> in / <out> out, and the ShareGPT corpus
# pinned with --dataset-path (the Hub copy is gone upstream). The 3C requests
# must all complete with <out>-token outputs and no errors or the point FAILED.
#
# Container results: <RESULTS_HOST>/<mode>-ctx<ctx>-<ts>/
# Standalone results: <RESULTS_DIR>/<ts>-<mode>-<rand>/
# =====================================================================
set -u

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$DIR/../../.." && pwd)

ts () { date +%Y%m%d_%H%M%S; }
die () { echo "FATAL: $*" >&2; exit 1; }

help_usage () {
  cat <<'EOF'
usage: bench-serving.sh fair    <ctx> [C]     native vs xkv at the same C (default 32)
       bench-serving.sh graphab <ctx> [C]     xkv decode graphs full vs disabled
       bench-serving.sh <native|xkv> <in> <out> <concurrency>

  fair      one fresh boot per leg, same C, same context. The headline serving
            comparison. Results: <RESULTS_HOST>/fair-ctx<ctx>-<ts>/.
  graphab   two fresh boots of the SAME xkv leg, decode graphs full then
            disabled, same C. Isolates what graph capture buys. Results:
            <RESULTS_HOST>/graphab-ctx<ctx>-<ts>/.
  native|xkv standalone single-config measurement on the current host; boots one
            TP4 server with the report configuration. MODEL_PATH must point at
            the official 0731 checkpoint. Optional env: PYTHON, RESULTS_DIR,
            PORT, SEED, MEM_FRAC, CTX_LEN, MAX_RUN, CHUNK, DECODE_CFG, MODEL_NAME.
EOF
}
usage_err () { echo "bench-serving.sh: $*" >&2; help_usage >&2; exit 2; }

server_pid=
bench_pid=
cleanup () {
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

# ------------------------- shared (host-side) --------------------------
validate () {  # $1=measured.jsonl(host) $2=C $3=outlen -> 0 if 3C all complete w/ <outlen>
  python3 - "$1" "$((3 * $2))" "$3" <<'PY'
import json, sys
rec = None
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        rec = json.loads(line)
exp = int(sys.argv[2]); outlen = int(sys.argv[3])
ok = rec and rec.get("completed") == exp \
     and all(n == outlen for n in (rec.get("output_lens") or [])) \
     and not [e for e in (rec.get("errors") or []) if e]
print(f"  valid: completed={rec.get('completed') if rec else None} expected={exp} all_outlen={ok}")
sys.exit(0 if ok else 1)
PY
}

summary_line () {  # $1=measured.log -> one readable line
  awk '
    /Request throughput \(req\/s\):/       {r=$NF}
    /Total token throughput \(tok\/s\):/   {t=$NF}
    /Median TTFT \(ms\):/                  {tt=$NF}
    /Median TPOT \(ms\):/                  {tp=$NF}
    /Median E2E Latency \(ms\):/           {e=$NF}
    END {printf "req/s=%.4f tok/s=%.1f ttft_ms=%.1f tpot_ms=%.1f e2e_ms=%.1f", r,t,tt,tp,e}'
}

# ------------------------- container protocol --------------------------
bench_wave () {  # $1=C $2=N $3=out.jsonl(ct) $4=out.log(host)
  ct "export PYTHONPATH=$SGLANG_PY; cd $SGLANG_PY
      python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model $MODEL_CT --tokenizer $MODEL_CT \
        --dataset-name random --dataset-path $SHAREGPT_CT \
        --random-input-len $CTX --random-output-len $OUTLEN \
        --random-range-ratio 1.0 --num-prompts $2 --max-concurrency $1 \
        --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
        --output-file $3 --output-details --seed $SEED" > "$4" 2>&1
}

# Boot one leg, measure one point, validate, kill.
boot_measure_kill () {  # $1=leg $2=C $3=arm label (optional, for the log dir)
  # Positional refs, not $leg/$arm: every word of a `local` is expanded before
  # the builtin assigns anything, so under `set -u` naming the new locals here
  # aborts with "leg: unbound variable". (mustafar's copy uses $1 for the same
  # reason; this was introduced when the `arm` parameter was added.)
  local leg=$1 C=$2 arm=${3:-} dir="$RUN_ROOT/$1${3:-}" POOL rc
  mkdir -p "$dir"
  echo "==== [$leg$arm] ctx=$CTX C=$C $(date -u +%H:%M:%S)Z ===="
  bash "$DIR/serve.sh" "$leg" || die "serve.sh $leg failed"
  POOL=$(pool_of "$LOG_HOST/serve_$leg$( [ "$DECODE_CFG" = "$DECODE_CFG_OFF" ] && echo _nograph ).log")
  echo "  pool=$POOL"
  echo "  pool=$POOL" > "$dir/pool.txt"
  bench_wave "$C" "$C" "/tmp/${leg}${arm}-warm.jsonl" "$dir/warmup.log"
  if ! health; then
    echo "  [$leg$arm] SERVER DOWN after warmup -> FAILED" | tee -a "$RUN_ROOT/RESULT.txt"
    bash "$DIR/serve.sh" "$leg" stop
    return 1
  fi
  bench_wave "$C" "$((3 * C))" "$(to_ct "$dir")/measured.jsonl" "$dir/measured.log"
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$dir/measured.jsonl" ] && validate "$dir/measured.jsonl" "$C" "$OUTLEN"; then
    echo "  [$leg$arm] $(summary_line "$dir/measured.log")" | tee -a "$RUN_ROOT/RESULT.txt"
    echo "  $leg$arm OK" | tee -a "$RUN_ROOT/RESULT.txt"
  else
    echo "  $leg$arm FAILED (rc=$rc)" | tee -a "$RUN_ROOT/RESULT.txt"
  fi
  bash "$DIR/serve.sh" "$leg" stop
}

report_main () {  # $1=fair|graphab $2=ctx [$3=C]
  local mode=$1
  . "$DIR/env.sh"
  CTX=${2:-}; C=${3:-32}
  [ -n "$CTX" ] || usage_err "missing <ctx>"
  [[ "$CTX" =~ ^[1-9][0-9]*$ ]] || usage_err "ctx must be a positive integer (got '$CTX')"
  [[ "$C" =~ ^[1-9][0-9]*$ ]] || usage_err "C must be a positive integer (got '$C')"
  (( C <= 136 )) || usage_err "C=$C exceeds the extended decode-graph coverage cap of 136"
  RUN_ROOT="$RESULTS_HOST/$mode-ctx$CTX-$(ts)"
  mkdir -p "$RUN_ROOT"

  case "$mode" in
    fair)
      boot_measure_kill native "$C"
      boot_measure_kill xkv "$C"
      ;;
    graphab)
      DECODE_CFG="$DECODE_CFG_EXT"  boot_measure_kill xkv "$C" ; # arm 1: on-graph
      DECODE_CFG="$DECODE_CFG_OFF"  boot_measure_kill xkv "$C" _nograph
      ;;
  esac

  echo "=== artifacts at $RUN_ROOT ==="
  cat "$RUN_ROOT/RESULT.txt" 2>/dev/null
}

# ------------------------- standalone (self-boot) ----------------------
solo_wave () {  # $1=jsonl $2=log $3=num-prompts
  setsid "$python" -m sglang.bench_serving \
    --backend sglang --host 127.0.0.1 --port "$port" \
    --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" --dataset-name random \
    --dataset-path "${SHAREGPT_HOST:-}" \
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

standalone_main () {  # $1=mode [$2=input $3=output $4=concurrency]
  mode=${1:-native}; input=${2:-65536}; output=${3:-2048}; conc=${4:-32}
  (( $# <= 4 )) || usage_err "expected at most 4 arguments (mode input output concurrency)"
  case "$mode" in
    native|xkv) ;;
    *) usage_err "unknown mode '$mode'" ;;
  esac
  for n in "$input" "$output" "$conc"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || usage_err "counts must be positive integers (got '$n')"
  done
  (( conc <= 136 )) || usage_err "concurrency $conc exceeds 136 (extended decode-graph coverage cap)"
  : "${MODEL_PATH:?Set MODEL_PATH to the official DeepSeek-V4-Flash-0731 checkpoint}"
  : "${SHAREGPT_HOST:?Set SHAREGPT_HOST to the local ShareGPT json (the Hub copy is gone upstream)}"
  for tool in "${PYTHON:-python3}" curl jq setsid; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
  done

  python=${PYTHON:-python3}; port=${PORT:-30213}; seed=${SEED:-42}
  results_dir=${RESULTS_DIR:-$REPO/xkv/logs/bench-serving}
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1

  for name in ${!SGLANG_OPT_LOWRANK_KV_STORE@} ${!XKV_@}; do
    unset "$name" 2>/dev/null || true
  done
  if [ "$mode" = xkv ]; then
    export SGLANG_OPT_LOWRANK_KV_STORE=1 XKV_RECON_TRITON=1 XKV_COEFF_DIM=192
    export SG_LOWRANK_BASIS="$BASIS_HOST"
  fi

  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    die "a server is already running on port $port; set a different PORT"
  fi
  mkdir -p "$results_dir"
  run=$(mktemp -d "$results_dir/$(ts)-$mode-XXXXXX") || die "could not create a run dir under $results_dir"
  echo "Results: $run"
  echo "boot: mode=$mode tp=${TP:-4} mem=${MEM_FRAC:-0.88} ctx=${CTX_LEN:-1048576} max_run=${MAX_RUN:-256} chunk=${CHUNK:-8192} fp8_kv port=$port"

  setsid "$python" -c 'from sglang.cli.main import main; main()' serve \
    --model-path "$MODEL_PATH" --served-model-name "${MODEL_NAME:-deepseek-v4-flash}" \
    --tp "${TP:-4}" --trust-remote-code --mem-fraction-static "${MEM_FRAC:-0.88}" \
    --context-length "${CTX_LEN:-1048576}" --max-running-requests "${MAX_RUN:-256}" \
    --chunked-prefill-size "${CHUNK:-8192}" \
    --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4 \
    --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
    --host 127.0.0.1 --port "$port" --cuda-graph-config "$DECODE_CFG" \
    --skip-server-warmup --watchdog-timeout 1800 \
    >"$run/server.log" 2>&1 &
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
  if ! curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    die "server DOWN after warmup"
  fi
  solo_wave "$run/measured.jsonl" "$run/measured.log" "$((3 * conc))" || die "measured wave failed"
  if validate "$run/measured.jsonl" "$conc" "$output"; then
    echo "  measured: $(summary_line "$run/measured.log")"
    echo "Completed: $run/measured.jsonl"
  else
    echo "standalone point FAILED (validation): $run/measured.jsonl" >&2
    exit 1
  fi
  exit 0
}

# ------------------------------- dispatch ------------------------------
MODE=${1:-}
case "$MODE" in
  --help|-h) help_usage; exit 0 ;;
  fair|graphab) report_main "$@" ;;
  native|xkv) standalone_main "$@" ;;
  *) usage_err "unknown mode '$MODE'" ;;
esac
