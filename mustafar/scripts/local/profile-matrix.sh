#!/usr/bin/env bash
# =====================================================================
# profile-matrix.sh -- Step 5 of the decode-cost-attribution plan.
#
#   profile-matrix.sh            the full matrix: 4 modes x 3 passes,
#                                both concurrencies off one boot each.
#                                That is 12 boots, ~2.2 h.
#   profile-matrix.sh smoke      one mode, one pass, one concurrency --
#                                enough to prove the whole chain (boot ->
#                                warm-up -> wave -> mid-wave profile ->
#                                traces -> validate) before the long run.
#
# Per (mode, pass): ONE boot, then both concurrency points off it.
# Per point: a warm-up wave at C, then a measured wave of 3C requests with
# the profiler fired mid-wave. The profiler counts real server steps, so
# firing it at an idle server captures nothing; firing it after the wave
# ended captures nothing either. It is fired PROFILE_DELAY seconds into the
# measured wave. 20 decode steps at ~17 ms/step is ~0.35 s of stepping, so
# it completes long before a wave of 32k-in/2k-out requests drains.
#
# Modes rotate per pass (native->packed->optimized->topmag, then rotated
# by one) so that any thermal/neighbour drift decorrelates from mode
# instead of loading onto whichever mode is measured last.
#
# The wave's own throughput numbers are NOT the deliverable and are not
# compared across modes -- the in-kernel profiler perturbs timing badly.
# The traces are the deliverable; the client's role is to hold the server
# at the pinned concurrency while the profiler samples it.
#
# The bench client runs INSIDE the container (host python has no sglang).
# `--dataset-path $SHAREGPT_CT` is required: sglang >= 0.5.18 fetches the
# ShareGPT json from the Hub when the path is empty, and that repo is gone
# upstream. See SHAREGPT_* in env.sh.
#
# Overrides: MODES_OVERRIDE, PASSES, CONCS, IN_LEN, OUT_LEN, STEPS,
#            PROFILE_DELAY, OUTDIR
# =====================================================================
set -u
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$DIR/env.sh"

# The extended decode graphs: C=25 is above the small config's max_bs 15,
# so without this the C=25 point silently falls off-graph and stops being
# comparable to the report's ladder point. serve.sh inherits it.
export DECODE_CFG="$DECODE_CFG_EXT"

SMOKE=0
[ "${1:-}" = smoke ] && SMOKE=1

if [ "$SMOKE" = 1 ]; then
  # Still overridable: the point of the smoke is to prove the chain, and the
  # chain that matters is two points off one boot (the second one only works if
  # the first closed its profile session). CONCS="8 25" exercises that.
  MODES=(${MODES_OVERRIDE:-native}); PASSES=1; CONCS=(${CONCS:-8})
else
  MODES=(${MODES_OVERRIDE:-native packed optimized topmag})
  PASSES=${PASSES:-3}
  CONCS=(${CONCS:-8 25})
fi
IN_LEN=${IN_LEN:-32768}
OUT_LEN=${OUT_LEN:-2048}
STEPS=${STEPS:-20}
PROFILE_DELAY=${PROFILE_DELAY:-30}

RUN_TAG=$(date +%Y%m%d)
OUTDIR=${OUTDIR:-$RESULTS_HOST/profile-$RUN_TAG}
RUN_ROOT=${RUN_ROOT:-$RESULTS_HOST/profile-run-$RUN_TAG-$([ "$SMOKE" = 1 ] && echo smoke || echo full)}
mkdir -p "$OUTDIR" "$RUN_ROOT"
RESULT="$RUN_ROOT/RESULT.txt"

echo "== profile matrix: modes=${MODES[*]} passes=$PASSES concs=${CONCS[*]}"
echo "   in=$IN_LEN out=$OUT_LEN steps=$STEPS delay=${PROFILE_DELAY}s"
echo "   traces -> $OUTDIR"
echo "   run    -> $RUN_ROOT"

# ------------------------------- helpers -------------------------------
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

summary_line () {  # $1=measured.log -> one readable line (informational only)
  awk '
    /Request throughput \(req\/s\):/       {r=$NF}
    /Total token throughput \(tok\/s\):/   {t=$NF}
    /Median TPOT \(ms\):/                  {tp=$NF}
    END {printf "req/s=%.4f tok/s=%.1f tpot_ms=%.1f", r,t,tp}'
}

# One bench_serving wave, in-container. --max-concurrency $1, $2 prompts.
wave () {  # $1=C $2=N $3=out.jsonl(host) $4=out.log(host)
  ct "export PYTHONPATH=$SGLANG_PY; cd $SGLANG_PY
      export HF_ENDPOINT=https://hf-mirror.com
      python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model $MODEL_CT --tokenizer $MODEL_CT \
        --dataset-name random --dataset-path $SHAREGPT_CT \
        --random-input-len $IN_LEN --random-output-len $OUT_LEN \
        --random-range-ratio 1.0 --num-prompts $2 --max-concurrency $1 \
        --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
        --output-file $(to_ct "$3") --output-details --seed $SEED" > "$4" 2>&1
}

# Assert the leg is the leg we think it is. The dispatch marker is written
# by the fork at boot; native serves the stock tree and has none.
assert_leg () {  # $1=mode
  local mode=$1 log="$LOG_HOST/serve_$mode.log"
  case "$mode" in
    native)
      if grep -qa "MUSTAFAR\|mustafar" "$log" 2>/dev/null; then
        echo "  WARN: native boot log mentions mustafar -- tree may be patched"
      fi ;;
    *)
      grep -aoE "MUSTAFAR_FUSED_DISPATCH=[a-z_]*" "$log" | head -1 | sed 's/^/  /'
      ;;
  esac
}

# One measured point: warm-up, then the measured wave with the profiler
# fired mid-wave.
point () {  # $1=mode $2=C $3=pass
  local mode=$1 C=$2 pass=$3
  local pid="${mode}-c${C}-r${pass}"
  local dir="$RUN_ROOT/$pid" bpid
  mkdir -p "$dir"
  echo "---- point $pid $(date -u +%H:%M:%S)Z"

  wave "$C" "$C" "$dir/warmup.jsonl" "$dir/warmup.log" \
    || echo "  WARN: warm-up wave failed (continuing; the measured wave flushes cache anyway)"
  if ! health; then
    echo "  FAIL $pid: server down after warm-up" | tee -a "$RESULT"
    return 1
  fi

  wave "$C" "$((3 * C))" "$dir/measured.jsonl" "$dir/measured.log" &
  bpid=$!
  sleep "$PROFILE_DELAY"

  if bash "$DIR/profile-ranks.sh" "$pid" --steps "$STEPS" --outdir "$OUTDIR" \
       > "$dir/profile.log" 2>&1; then
    echo "  profile OK -> $OUTDIR/$pid-TP-{0..3}-DECODE.trace.json.gz"
  else
    echo "  WARN $pid: profiler did not land all traces (see $dir/profile.log)"
  fi

  wait "$bpid" || true
  if [ -f "$dir/measured.jsonl" ] && validate "$dir/measured.jsonl" "$C" "$OUT_LEN"; then
    echo "  OK $pid: $(summary_line < "$dir/measured.log")" | tee -a "$RESULT"
  else
    echo "  FAIL $pid: measured wave did not validate" | tee -a "$RESULT"
    return 1
  fi
}

# --------------------------------- run ---------------------------------
n=${#MODES[@]}
for pass in $(seq 1 "$PASSES"); do
  for i in $(seq 0 $((n - 1))); do
    mode=${MODES[$(( (i + pass - 1) % n ))]}
    echo "==== pass=$pass mode=$mode $(date -u +%H:%M:%S)Z ===="
    bash "$DIR/serve.sh" "$mode" || { echo "  FAIL: serve.sh $mode did not come up" | tee -a "$RESULT"; continue; }
    assert_leg "$mode"
    for C in "${CONCS[@]}"; do
      point "$mode" "$C" "$pass"
    done
    bash "$DIR/serve.sh" "$mode" stop
  done
done

echo "=== matrix done. traces in $OUTDIR ==="
ls -la "$OUTDIR" | tail -60
echo "=== points ==="
cat "$RESULT" 2>/dev/null
