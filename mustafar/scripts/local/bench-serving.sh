#!/usr/bin/env bash
# =====================================================================
# bench-serving.sh <fair|max> <ctx> [C_fair]
#   Low-level shared protocol behind bench-fair.sh and bench-max.sh.
#
#   fair  -> measure Native AND Packed at the SAME concurrency C_nat, where
#            C_nat = Native's allocator ceiling for <ctx> (floor(pool_nat/(ctx+2048))).
#            [C_fair] overrides the shared concurrency for both legs.
#   max   -> measure each leg at its OWN allocator ceiling
#            (Native at C_nat, Packed at C_pck = floor(pool_pck/(ctx+2048))).
#
# Per point (mirrors the report protocol): fresh boot of the leg server with
# extended decode CUDA graphs (decode on-graph up to max_bs 136), one warm-up
# wave of C, then 3 measured waves (3C requests) via official
# sglang.bench_serving, flush-cache, seed 42, exact <ctx> in / 2048 out.
# The 3C requests must all complete with 2048-token outputs and no errors or
# the point is FAILED. Results: <RESULTS_HOST>/serving/<mode>-<ctx>-<ts>/.
#
# Structure is deliberately 2 boots total: boot native (learns pool -> C_nat),
# measure it, boot packed (learns pool -> C_pck), measure it.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

MODE=${1:-} CTX=${2:-} C_FAIR=${3:-}
[ "$MODE" = fair ] || [ "$MODE" = max ] || { echo "usage: $0 <fair|max> <ctx> [C_fair]"; exit 1; }
[ -n "$CTX" ] || { echo "usage: $0 <fair|max> <ctx>"; exit 1; }

# Extended decode graphs so decode stays on-graph up to the packed ceiling.
export DECODE_CFG='{"decode":{"backend":"full","max_bs":136,"bs":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,24,28,32,34,40,48,56,64,68,80,96,112,120,136]},"prefill":{"backend":"disabled"}}'

RUN_ROOT="$RESULTS_HOST/serving/$MODE-ctx$CTX-$(ts)"
mkdir -p "$RUN_ROOT"

die () { echo "FATAL: $*" >&2; exit 1; }

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

validate () {  # $1=measured.jsonl(host) $2=C -> 0 if 3C completed, all 2048, no errors
  python3 - "$1" "$((3 * $2))" <<'PY'
import json, sys
rec = None
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        rec = json.loads(line)
exp = int(sys.argv[2])
ok = rec and rec.get("completed") == exp \
     and all(n == 2048 for n in (rec.get("output_lens") or [])) \
     and not [e for e in (rec.get("errors") or []) if e]
print(f"  valid: completed={rec.get('completed') if rec else None} expected={exp} all2048={ok}")
sys.exit(0 if ok else 1)
PY
}

summary_line () {  # $1=measured.log(host) -> one readable line
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

# ---- boot one leg, measure one point, validate, kill ----
# Learns the pool from the boot log; if C is empty it uses the leg's own
# allocator ceiling (floor(pool/(ctx+2048))).
boot_measure_kill () {  # $1=leg $2=C(empty = leg ceiling)
  local leg=$1 C=${2:-} dir="$RUN_ROOT/$1" POOL rc
  mkdir -p "$dir"
  echo "==== [$leg] ctx=$CTX C='${C:-ceiling}' $(date -u +%H:%M:%S)Z ===="
  bash "$(dirname "$0")/serve.sh" "$leg" || die "serve.sh $leg failed"
  POOL=$(pool_of "$LOG_HOST/serve_$leg.log"); echo "  pool=$POOL"
  [ -n "$C" ] || C=$(ceiling "$POOL" "$REQ_LEN")
  echo "  using C=$C"
  bench_wave "$C" "$C" "/tmp/${leg}-warm.jsonl" "$dir/warmup.log"
  if ! health; then
    echo "  [$leg] SERVER DOWN after warmup -> FAILED" | tee -a "$RUN_ROOT/RESULT.txt"
    bash "$(dirname "$0")/serve.sh" "$leg" stop
    return 1
  fi
  bench_wave "$C" "$((3 * C))" "$(to_ct "$dir")/measured.jsonl" "$dir/measured.log"
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$dir/measured.jsonl" ] && validate "$dir/measured.jsonl" "$C"; then
    echo "  [$leg] $(summary_line "$dir/measured.log")" | tee -a "$RUN_ROOT/RESULT.txt"
    echo "  $leg OK" | tee -a "$RUN_ROOT/RESULT.txt"
  else
    echo "  $leg FAILED (rc=$rc)" | tee -a "$RUN_ROOT/RESULT.txt"
  fi
  bash "$(dirname "$0")/serve.sh" "$leg" stop
}

REQ_LEN=$((CTX + OUTLEN))

case "$MODE" in
  fair)
    if [ -n "$C_FAIR" ]; then
      boot_measure_kill native "$C_FAIR"
      boot_measure_kill packed "$C_FAIR"
    else
      # shared fair C = native's allocator ceiling: probe native's pool first
      bash "$(dirname "$0")/serve.sh" native || die "serve.sh native failed"
      NATIVE_POOL=$(pool_of "$LOG_HOST/serve_native.log")
      bash "$(dirname "$0")/serve.sh" native stop
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
