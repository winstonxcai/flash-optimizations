#!/usr/bin/env bash
# =====================================================================
# bench-serving.sh -- one bench driver, two interfaces.
#
#   1) Report-grade dual-leg protocol (fair/max) inside the eval container:
#        bench-serving.sh <fair|max> <ctx> [C_fair]
#      fair -> measure Native AND Packed at the SAME concurrency C_nat, where
#              C_nat = Native's allocator ceiling for <ctx> (floor(pool/(ctx+2048))).
#              [C_fair] overrides the shared concurrency for both legs.
#      max  -> measure each leg at its OWN allocator ceiling
#              (Native at C_nat, Packed at C_pck = floor(pool_pck/(ctx+2048))).
#      Each leg boots with serve.sh inside the $CONTAINER (sources env.sh).
#
#   2) Standalone single-config measurement on the current host:
#        MODEL_PATH=/checkpoint bash bench-serving.sh <native|packed|fused> \
#                                        <in> <out> <concurrency>
#      One configuration per call, no container / env.sh dependency: self-boots
#      one TP4 server with the report configuration (fp8 KV, mem-frac 0.88, 1M
#      ctx, extended decode graphs). This is the interface Modal
#      (app.py::bench_serving) and tests/test_bench_serving.py drive.
#
# Per point both paths share the report protocol: fresh server, extended decode
# CUDA graphs (decode on-graph up to max_bs 136), one warm-up wave of C, then 3
# measured waves (3C requests) via official sglang.bench_serving, flush-cache,
# seed 42, exact <ctx|in> in / <out> out. The 3C requests must all complete with
# <out>-token outputs and no errors or the point is FAILED.
#
# fair/max results:   <RESULTS_HOST>/serving/<mode>-ctx<ctx>-<ts>/
# standalone results: <RESULTS_DIR>/<ts>-<mode>-<rand>/  (server.log,
#                      warmup.log/jsonl, measured.log/jsonl;
#                      default <repo>/mustafar/logs/bench-serving/)
# =====================================================================
set -u

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$DIR/../../.." && pwd)

# Extended decode graphs so decode stays on-graph up to the packed ceiling --
# the report config for BOTH interfaces (env.sh's small default targets the
# agentic evals only). This standalone path cannot source env.sh, so the EXT
# literal is duplicated here -- keep it in sync with env.sh's DECODE_CFG_EXT.
# Overridable via DECODE_CFG.
export DECODE_CFG=${DECODE_CFG:-'{"decode":{"backend":"full","max_bs":136,"bs":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,24,28,32,34,40,48,56,64,68,80,96,112,120,136]},"prefill":{"backend":"disabled"}}'}

ts () { date +%Y%m%d_%H%M%S; }
die () { echo "FATAL: $*" >&2; exit 1; }

help_usage () {
  cat <<'EOF'
usage: bench-serving.sh <fair|max> <ctx> [C_fair]
       bench-serving.sh <native|packed|fused> <in> <out> <concurrency>

  fair|max  report-grade dual-leg serving protocol inside the eval container
            (each leg boots with serve.sh; warm-up wave of C then 3 measured
            waves). fair = both legs at Native's allocator ceiling, or at
            [C_fair]; max = each leg at its own allocator ceiling.
            Results: <RESULTS_HOST>/serving/<mode>-ctx<ctx>-<ts>/.

  native|packed|fused
            standalone single-config measurement on the current host: self-boots
            one TP4 server with the report configuration (fp8 KV, mem-frac
            0.88, 1M ctx, extended decode graphs), warm-up wave of
            <concurrency> then 3 measured waves. MODEL_PATH must point at the
            official 0731 checkpoint. Results: <RESULTS_DIR>/<ts>-<mode>-<rand>/.
            Optional env: PYTHON, RESULTS_DIR, PORT (default 30211), SEED,
            MEM_FRAC, CTX_LEN, MAX_RUN, CHUNK, DECODE_CFG, MODEL_NAME.
EOF
}
usage_err () { echo "bench-serving.sh: $*" >&2; help_usage >&2; exit 2; }

# Process-group teardown for the standalone server/client (setsid children run
# in their own process groups). fair/max servers are stopped by serve.sh, not
# here, so these stay empty on that path.
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
validate () {  # $1=measured.jsonl(host) $2=C $3=outlen -> 0 if 3C all complete w/ <outlen>, no errors
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

ceiling () {  # $1=pool $2=req_len -> floor(pool/req_len)
  python3 -c "print(int(int('$1') // int('$2')))"
}

# ------------------------- fair/max (container) ------------------------
# One bench_serving run (in-container) writing raw jsonl + a stdout summary log.
bench_wave () {  # $1=C $2=N $3=out.jsonl(ct) $4=out.log(host)
  ct "export PYTHONPATH=$SGLANG_PY; cd $SGLANG_PY
      python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model $MODEL_CT --tokenizer $MODEL_CT \
        --dataset-name random --random-input-len $CTX --random-output-len $OUTLEN \
        --random-range-ratio 1.0 --num-prompts $2 --max-concurrency $1 \
        --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
        --output-file $3 --output-details --seed $SEED" > "$4" 2>&1
}

# ---- boot one leg, measure one point, validate, kill ----
# Learns the pool from the boot log; if C is empty it uses the leg's own
# allocator ceiling (floor(pool/(ctx+2048))).
boot_measure_kill () {  # $1=leg $2=C(empty = leg ceiling)
  local leg=$1 C=${2:-} dir="$RUN_ROOT/$1" POOL rc
  mkdir -p "$dir"
  echo "==== [$leg] ctx=$CTX C='${C:-ceiling}' $(date -u +%H:%M:%S)Z ===="
  bash "$DIR/serve.sh" "$leg" || die "serve.sh $leg failed"
  POOL=$(pool_of "$LOG_HOST/serve_$leg.log"); echo "  pool=$POOL"
  [ -n "$C" ] || C=$(ceiling "$POOL" "$REQ_LEN")
  echo "  using C=$C"
  bench_wave "$C" "$C" "/tmp/${leg}-warm.jsonl" "$dir/warmup.log"
  if ! health; then
    echo "  [$leg] SERVER DOWN after warmup -> FAILED" | tee -a "$RUN_ROOT/RESULT.txt"
    bash "$DIR/serve.sh" "$leg" stop
    return 1
  fi
  bench_wave "$C" "$((3 * C))" "$(to_ct "$dir")/measured.jsonl" "$dir/measured.log"
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$dir/measured.jsonl" ] && validate "$dir/measured.jsonl" "$C" "$OUTLEN"; then
    echo "  [$leg] $(summary_line "$dir/measured.log")" | tee -a "$RUN_ROOT/RESULT.txt"
    echo "  $leg OK" | tee -a "$RUN_ROOT/RESULT.txt"
  else
    echo "  $leg FAILED (rc=$rc)" | tee -a "$RUN_ROOT/RESULT.txt"
  fi
  bash "$DIR/serve.sh" "$leg" stop
}

fairmax_main () {  # $1=mode $2=ctx [$3=C_fair]
  local mode=$1 c_fair=${3:-}
  . "$DIR/env.sh"
  CTX=${2:-}
  [ -n "$CTX" ] || usage_err "missing <ctx>"
  [[ "$CTX" =~ ^[1-9][0-9]*$ ]] || usage_err "ctx must be a positive integer (got '$CTX')"
  if [ -n "$c_fair" ]; then
    [[ "$c_fair" =~ ^[1-9][0-9]*$ ]] || usage_err "C_fair must be a positive integer (got '$c_fair')"
  fi
  RUN_ROOT="$RESULTS_HOST/serving/$mode-ctx$CTX-$(ts)"
  mkdir -p "$RUN_ROOT"
  REQ_LEN=$((CTX + OUTLEN))

  case "$mode" in
    fair)
      if [ -n "$c_fair" ]; then
        boot_measure_kill native "$c_fair"
        boot_measure_kill packed "$c_fair"
      else
        # shared fair C = native's allocator ceiling: probe native's pool first
        bash "$DIR/serve.sh" native || die "serve.sh native failed"
        NATIVE_POOL=$(pool_of "$LOG_HOST/serve_native.log")
        bash "$DIR/serve.sh" native stop
        echo "native pool=$NATIVE_POOL"
        C_SHARED=$(ceiling "$NATIVE_POOL" "$REQ_LEN")
        boot_measure_kill native "$C_SHARED"
        boot_measure_kill packed "$C_SHARED"
      fi
      ;;
    max)
      boot_measure_kill native ""   # C = native ceiling
      boot_measure_kill packed ""   # C = packed ceiling
      ;;
  esac

  echo "=== artifacts at $RUN_ROOT ==="
  cat "$RUN_ROOT/RESULT.txt" 2>/dev/null
}

# ------------------------- standalone (self-boot) ----------------------
# One wave of the official bench_serving client (host python, own process group
# so the EXIT trap can tear it down together with the server).
solo_wave () {  # $1=jsonl $2=log $3=num-prompts -> rc of client
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

standalone_main () {  # $1=mode [$2=input $3=output $4=concurrency]
  mode=${1:-native}; input=${2:-32768}; output=${3:-2048}; conc=${4:-8}
  (( $# <= 4 )) || usage_err "expected at most 4 arguments (mode input output concurrency)"
  case "$mode" in
    native|packed|fused) ;;
    *) usage_err "unknown mode '$mode'" ;;
  esac
  for n in "$input" "$output" "$conc"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || usage_err "counts must be positive integers (got '$n')"
  done
  (( conc <= 136 )) || usage_err "concurrency $conc exceeds 136 (extended decode-graph coverage cap)"
  : "${MODEL_PATH:?Set MODEL_PATH to the official DeepSeek-V4-Flash-0731 checkpoint}"
  for tool in "${PYTHON:-python3}" curl jq setsid; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
  done

  python=${PYTHON:-python3}; port=${PORT:-30211}; seed=${SEED:-42}
  results_dir=${RESULTS_DIR:-$REPO/mustafar/logs/bench-serving}
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
  if [ -n "${SGLANG_ROOT:-}" ]; then
    export SG_LOWRANK_SRC="$SGLANG_ROOT/python"
    export PYTHONPATH="$SG_LOWRANK_SRC:$PYTHONPATH"
  fi

  # TopMag switches on the CURRENT runtime names (mustafar/config.py); clear any
  # inherited or legacy pre-refactor ones first, then set the mode's values.
  for name in ${!SGLANG_OPT_TOPMAG@} ${!KEEP@} XKV_TOPMAG_KEEP SGLANG_OPT_TOPMAG_PACKED_C4; do
    unset "$name" 2>/dev/null || true
  done
  export SGLANG_OPT_TOPMAG=0 KEEP=1.0 SGLANG_OPT_TOPMAG_PACKED=0 SGLANG_OPT_TOPMAG_FUSED=0
  if [ "$mode" != native ]; then
    export SGLANG_OPT_TOPMAG=1 KEEP=0.5 SGLANG_OPT_TOPMAG_PACKED=1
  fi
  [ "$mode" != fused ] || export SGLANG_OPT_TOPMAG_FUSED=1

  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    die "a server is already running on port $port; set a different PORT"
  fi
  mkdir -p "$results_dir"
  run=$(mktemp -d "$results_dir/$(ts)-$mode-XXXXXX") || die "could not create a run dir under $results_dir"
  echo "Results: $run"
  echo "boot: mode=$mode tp=${TP:-4} mem=${MEM_FRAC:-0.88} ctx=${CTX_LEN:-1048576} max_run=${MAX_RUN:-256} chunk=${CHUNK:-8192} fp8_kv port=$port"

  # Report-config server (mirrors serve.sh flags, host python, localhost-only).
  # Host python has no `sglang` console script and cli/main.py has no __main__
  # guard, so reach the new `serve` entry through its exact console-script body.
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
  fair|max) fairmax_main "$@" ;;
  native|packed|fused) standalone_main "$@" ;;
  *) usage_err "unknown mode '$MODE'" ;;
esac
