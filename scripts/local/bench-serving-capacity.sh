#!/usr/bin/env bash
# Fair/max serving capacity measurements against the local Docker container.
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
usage: bench-serving-capacity.sh <fair|max> <ctx> [C_fair]

fair measures native and packed at the same concurrency.
max measures each layout at its own allocator ceiling.
EOF
}
usage_err () { echo "bench-serving-capacity.sh: $*" >&2; usage; exit 2; }

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

ceiling () { python3 -c "print(int(int('$1') // int('$2')))"; }

bench_wave () {
  ct "export PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT; cd $SGLANG_PY
      python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model $MODEL_CT --tokenizer $MODEL_CT \
        --dataset-name random --random-input-len $CTX --random-output-len $OUTLEN \
        --random-range-ratio 1.0 --num-prompts $2 --max-concurrency $1 \
        --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
        --output-file $3 --output-details --seed $SEED" > "$4" 2>&1
}

measure_leg () {
  local leg=$1 C=${2:-} dir="$RUN_ROOT/$1" pool rc=1
  mkdir -p "$dir"
  echo "==== [$leg] ctx=$CTX C='${C:-ceiling}' $(date -u +%H:%M:%S)Z ===="
  bash "$DIR/serve.sh" "$leg" || die "serve.sh $leg failed"
  pool=$(pool_of "$LOG_HOST/serve_$leg.log")
  echo "  pool=$pool"
  [ -n "$pool" ] || {
    echo "  [$leg] could not determine allocator pool" | tee -a "$RUN_ROOT/RESULT.txt"
    bash "$DIR/serve.sh" "$leg" stop || true
    return 1
  }
  [ -n "$C" ] || C=$(ceiling "$pool" "$REQ_LEN")
  [[ "$C" =~ ^[1-9][0-9]*$ ]] || {
    echo "  [$leg] invalid concurrency ceiling: $C" | tee -a "$RUN_ROOT/RESULT.txt"
    bash "$DIR/serve.sh" "$leg" stop || true
    return 1
  }
  echo "  using C=$C"
  bench_wave "$C" "$C" "/tmp/${leg}-warm.jsonl" "$dir/warmup.log" || {
    echo "  [$leg] warmup failed" | tee -a "$RUN_ROOT/RESULT.txt"
    bash "$DIR/serve.sh" "$leg" stop || true
    return 1
  }
  if ! health; then
    echo "  [$leg] server down after warmup" | tee -a "$RUN_ROOT/RESULT.txt"
    bash "$DIR/serve.sh" "$leg" stop || true
    return 1
  fi
  bench_wave "$C" "$((3 * C))" "$(to_ct "$dir")/measured.jsonl" "$dir/measured.log"
  rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$dir/measured.jsonl" ] && validate "$dir/measured.jsonl" "$C" "$OUTLEN"; then
    echo "  [$leg] $(summary_line "$dir/measured.log")" | tee -a "$RUN_ROOT/RESULT.txt"
    echo "  $leg OK" | tee -a "$RUN_ROOT/RESULT.txt"
  else
    echo "  [$leg] FAILED (rc=$rc)" | tee -a "$RUN_ROOT/RESULT.txt"
    rc=1
  fi
  bash "$DIR/serve.sh" "$leg" stop || rc=1
  return "$rc"
}

main () {
  if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    usage
    return 0
  fi
  local mode=${1:-} fair_c=${3:-} rc=0 ctx=${2:-}
  [ "$mode" = fair ] || [ "$mode" = max ] || usage_err "mode must be fair or max"
  [[ "$ctx" =~ ^[1-9][0-9]*$ ]] || usage_err "ctx must be a positive integer"
  if [ -n "$fair_c" ]; then
    [[ "$fair_c" =~ ^[1-9][0-9]*$ ]] || usage_err "C_fair must be a positive integer"
  fi
  RUN_ROOT="$RESULTS_HOST/serving/$mode-ctx$ctx-$(ts)"
  mkdir -p "$RUN_ROOT"
  CTX=$ctx
  REQ_LEN=$((CTX + OUTLEN))
  case "$mode" in
    fair)
      if [ -n "$fair_c" ]; then
        measure_leg native "$fair_c" || rc=1
        measure_leg packed "$fair_c" || rc=1
      else
        bash "$DIR/serve.sh" native || die "serve.sh native failed"
        native_pool=$(pool_of "$LOG_HOST/serve_native.log")
        bash "$DIR/serve.sh" native stop || rc=1
        [ -n "$native_pool" ] || die "could not determine native allocator pool"
        shared_c=$(ceiling "$native_pool" "$REQ_LEN")
        measure_leg native "$shared_c" || rc=1
        measure_leg packed "$shared_c" || rc=1
      fi
      ;;
    max)
      measure_leg native || rc=1
      measure_leg packed || rc=1
      ;;
  esac
  echo "=== artifacts at $RUN_ROOT ==="
  cat "$RUN_ROOT/RESULT.txt" 2>/dev/null || true
  return "$rc"
}

main "$@"
