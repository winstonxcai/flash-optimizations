#!/usr/bin/env bash
# STOCK-native re-run of the fp4 serving-capacity points, per the user-approved
# baseline decision ("Native = genuinely untouched stock 0731", TOPMAG=0, dense
# 584-byte C4), so the report's Native column matches the LSWB/SWE-bench legs.
#
# The previously-reported "Native" random-serving numbers ran with TOPMAG=1 +
# PACKED_C4=0 (pruned-native 584-B) -- a packing-only control, not stock. This
# driver re-measures the native operating points as TRUE stock:
#
#   Phase A (sweep, no expandable_segments, pool gate 3,730,944):
#     ctx32768-max C107 | ctx65536-max C55 | ctx131072-max C28 | ctx262144-max C14
#   Phase B (OPTIONAL, RUN_V384=1; cancelled 2026-09-05 per user):
#     384k shallow C9, expandable_segments, pool gate 3,610,880
#
# Same protocol as run_serve_sweep_fp4.sh: official sglang.bench_serving, warmup 1
# wave of C then measured 3 waves (3C), random inputs at exact ctx, 2048 outputs,
# ratio 1.0, seed 42, flush-cache. Results:
#   mustafar/results/serve-sweep-fp4/stock/<ts>/<point>/
#   logs/serve_sweep_stock.log | logs/run-serve-sweep-stock.log
set -u
HOST_REPO=/home/jovyan/winstonxcai/flash-optimizations
CONT_REPO=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations
CONTAINER=ruler-eval
MODEL=/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731
LAUNCH_SWEEP=$CONT_REPO/mustafar/scripts/local/launch_inner_serve_sweep_stock_tp4.sh
LAUNCH_V384=$CONT_REPO/mustafar/scripts/local/launch_inner_verify384k_stock_tp4.sh
LOG_BASE=$HOST_REPO/mustafar/logs
RESULT_BASE=$HOST_REPO/mustafar/results/serve-sweep-fp4/stock
CT_LOG=$LOG_BASE/serve_sweep_stock.log
TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$RESULT_BASE/$TS"
PORT=30212
OUTLEN=2048
SEED=42
mkdir -p "$RUN_ROOT"

die () { echo "FATAL: $*" >&2; exit 1; }

wait_health () {   # $1 poll cap in 5s steps
  local n=${1:-180} i
  for i in $(seq 1 "$n"); do
    curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  health OK after ~$((i*5))s"; return 0; }
    sleep 5
  done
  echo "  health TIMEOUT after ~$((n*5))s" >&2
  return 1
}

health () { curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

kill_port () {
  ps -eo pid,args | grep "[s]glang.launch_server" | grep -- "--port $PORT" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
}

boot_leg () {   # $1 inner-launcher(container path) $2 pool_expect
  local launcher=$1 pool_expect=$2
  kill_port
  echo "== boot stock ($(date -u +%H:%M:%S)Z) via $(basename "$launcher") =="
  : > "$CT_LOG"
  docker exec "$CONTAINER" bash -c "bash $launcher" || die "inner launcher failed"
  wait_health || { tail -30 "$CT_LOG"; die "boot failed (log: $CT_LOG)"; }
  sleep 3
  local POOL
  POOL=$(grep -aoE "max_total_num_tokens=[0-9]+" "$CT_LOG" | head -1 | grep -oE "[0-9]+")
  echo "  pool=$POOL (expect $pool_expect)"
  [ "$POOL" = "$pool_expect" ] || { tail -30 "$CT_LOG"; die "pool $POOL != $pool_expect (C table invalid)"; }
  {
    echo "=== boot markers stock ($(date -u)) ==="
    grep -aoE "logical_row_bytes=[0-9]+ layers=[0-9]+|max_total_num_tokens=[0-9]+|available_gpu_mem=[0-9.]+ GB|Capture target decode CUDA graph end[^,]*elapsed=[0-9.]+ s|is fired up and ready" "$CT_LOG" | head -20
  } > "$RUN_ROOT/boot-markers.txt"
  cat "$RUN_ROOT/boot-markers.txt"
}

bench_wave () {   # $1 ctx $2 C $3 N $4 outfile(container) $5 log(host)
  local ctx=$1 C=$2 N=$3 OUT=$4 LOUT=$5
  docker exec "$CONTAINER" bash -c "
    export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
    cd /sgl-workspace/sglang-lowrank/python
    python3 -m sglang.bench_serving \
      --backend sglang --host 127.0.0.1 --port $PORT \
      --model $MODEL --tokenizer $MODEL \
      --dataset-name random --random-input-len $ctx --random-output-len $OUTLEN \
      --random-range-ratio 1.0 --num-prompts $N --max-concurrency $C \
      --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
      --output-file $OUT --output-details --seed $SEED" > "$LOUT" 2>&1
  return $?
}

validate () {   # $1 point-dir(host) $2 C
  local dir=$1 C=$2
  python3 - "$dir/measured.jsonl" "$((3*C))" <<'PYEOF' || return 1
import json,sys
rec=None
for line in open(sys.argv[1]):
    line=line.strip()
    if line: rec=json.loads(line)
exp=int(sys.argv[2])
ok = rec and rec.get("completed")==exp and all(n==2048 for n in (rec.get("output_lens") or [])) and not [e for e in (rec.get("errors") or []) if e]
print(f"  valid: completed={rec.get('completed') if rec else None} expected={exp} all2048={ok}")
sys.exit(0 if ok else 1)
PYEOF
}

run_point () {   # $1 label $2 ctx $3 C
  local label=$1 ctx=$2 C=$3
  local dir="$RUN_ROOT/$label"
  mkdir -p "$dir"
  local OUT_CT="/mnt/host_root$dir/measured.jsonl"
  local OFFSET; OFFSET=$(wc -c < "$CT_LOG")
  echo "==== $label ctx=$ctx C=$C ($(date -u +%H:%M:%S)Z) ===="
  bench_wave "$ctx" "$C" "$C" "/tmp/stock-warm.jsonl" "$dir/warmup.log"
  echo "  warmup rc=$? (${C} prompts)"
  if ! health; then echo "  SERVER DOWN after warmup"; echo "  $label FAILED (server down after warmup)" | tee -a "$RUN_ROOT/RESULT.txt"; return 1; fi
  bench_wave "$ctx" "$C" "$((3*C))" "$OUT_CT" "$dir/measured.log"
  local RC=$?
  echo "  measured rc=$RC ($((3*C)) prompts)"
  tail -c +$((OFFSET + 1)) "$CT_LOG" > "$dir/server.delta.log" 2>/dev/null || true
  if [ $RC -eq 0 ] && [ -f "$dir/measured.jsonl" ] && validate "$dir" "$C"; then
    echo "  $label OK" | tee -a "$RUN_ROOT/RESULT.txt"
    return 0
  fi
  echo "  $label FAILED (rc=$RC)" | tee -a "$RUN_ROOT/RESULT.txt"
  return 1
}

echo "=== stock-native serving re-run $(date -u +%H:%M:%S)Z ==="

# Phase A: 32k/64k/128k/256k at the native allocator ceiling (C107/55/28/14),
#           no expandable_segments, pool 3,730,944.
boot_leg "$LAUNCH_SWEEP" 3730944
run_point ctx32768-max 32768 107
run_point ctx65536-max 65536 55
run_point ctx131072-max 131072 28
run_point ctx262144-max 262144 14
kill_port

# Phase B (OPTIONAL, default OFF since 2026-09-05 user cancel): 384k shallow native
# point (C9), expandable_segments pool 3,610,880. Enable with RUN_V384=1.
if [ "${RUN_V384:-0}" = "1" ]; then
  boot_leg "$LAUNCH_V384" 3610880
  run_point ctx393216-max 393216 9
  kill_port
else
  echo "Phase B (384k verify) SKIPPED (RUN_V384!=1) -- 32k/64k/128k/256k sweep only"
fi

echo "=== stock-native re-run done $(date -u +%H:%M:%S)Z; artifacts at $RUN_ROOT ==="
exit 0
