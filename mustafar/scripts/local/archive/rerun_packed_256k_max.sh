#!/usr/bin/env bash
# Re-run the one failed sweep point: packed ctx262144-max.
#
# ctx262144-max (packed, C=17, 256k ctx) OOM-crashed the server mid-measurement
# on the ORIGINAL long-lived boot (21:37Z, "CUDA out of memory ... 1.81 GiB").
# This rerun uses a FRESH boot to distinguish:
#   (a) intrinsic 256k-depth workspace OOM  -> crash again at C=17
#   (b) fragmentation over the old server's 2.8h lifetime -> succeeds at C=17
# Falls back to C=16 then C=15 until one measured wave passes. Records the
# outcome in a RESULT file; does NOT clobber the original point dir.
#
# Results: results/serve-sweep-fp4/packed/<ts>-retry256k/ctx262144-max/  (chosen C)
set -u
CONTAINER=ruler-eval
HOST_REPO=/home/jovyan/winstonxcai/flash-optimizations
CONT_REPO=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations
MODEL=/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731
INNER_LAUNCH=$CONT_REPO/mustafar/scripts/local/launch_inner_serve_sweep_fp4_tp4.sh
LOG_BASE=$HOST_REPO/mustafar/logs
# NOTE: launch_inner_serve_sweep_fp4_tp4.sh hardcodes its server log to
# serve_sweep_packed.log for LEG=packed, so CT_LOG must point there for the
# pool-marker grep to work. Original run log is preserved as *.orig_run.log.
CT_LOG=$LOG_BASE/serve_sweep_packed.log
RESULT_BASE=$HOST_REPO/mustafar/results/serve-sweep-fp4/packed
TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$RESULT_BASE/${TS}-retry256k"
PORT=30211
CTX=262144
OUTLEN=2048
SEED=42
mkdir -p "$RUN_ROOT"

die () { echo "FATAL: $*" >&2; exit 1; }

health () { curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

wait_health () {
  for i in $(seq 1 200); do health && { echo "  health OK after ~$((i*5))s"; return 0; }; sleep 5; done
  return 1
}

kill_server () {
  ps -eo pid,args | grep "[s]glang.launch_server" | grep -- "--port $PORT" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 5
}

boot () {   # fresh packed boot on 30211 (GPUs 4-7)
  kill_server
  echo "== booting fresh packed server $(date -u +%H:%M:%S)Z =="
  : > "$CT_LOG"
  docker exec "$CONTAINER" bash -c "LEG=packed bash $INNER_LAUNCH" || die "inner launcher failed"
  wait_health || { tail -30 "$CT_LOG"; die "boot failed"; }
  POOL=$(grep -aoE "max_total_num_tokens=[0-9]+" "$CT_LOG" | head -1 | grep -oE "[0-9]+")
  echo "  pool=$POOL"
  [ "$POOL" = "4519168" ] || { tail -30 "$CT_LOG"; die "pool $POOL != 4519168"; }
}

bench () {   # $1 C $2 num_prompts $3 outfile(container) $4 log(host)
  local C=$1 N=$2 OUT=$3 LOUT=$4
  docker exec "$CONTAINER" bash -c "
    export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
    cd /sgl-workspace/sglang-lowrank/python
    python3 -m sglang.bench_serving \
      --backend sglang --host 127.0.0.1 --port $PORT \
      --model $MODEL --tokenizer $MODEL \
      --dataset-name random --random-input-len $CTX --random-output-len $OUTLEN \
      --random-range-ratio 1.0 --num-prompts $N --max-concurrency $C \
      --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
      --output-file $OUT --output-details --seed $SEED" > "$LOUT" 2>&1
}

echo "=== retry ctx262144-max packed: $(date -u +%H:%M:%S)Z ==="
boot

CHOSEN_C=""
for C in 17 16 15; do
  echo "---- trying C=$C $(date -u +%H:%M:%S)Z ----"
  # ensure server alive (a prior attempt may have OOM-killed it)
  if ! health; then echo "  server down -- rebooting"; boot; fi
  DIR="$RUN_ROOT/ctx${CTX}-max"
  mkdir -p "$DIR"
  OFFSET=$(wc -c < "$CT_LOG")
  # warmup
  if ! bench "$C" "$C" "/tmp/r256k-warm.jsonl" "$DIR/warmup-c${C}.log"; then echo "  warmup rc=$?"; fi
  if ! health; then echo "  server DIED during warmup C=$C"; continue; fi
  echo "  warmup ok (${C} prompts)"
  # measured
  if ! bench "$C" "$((3*C))" "/mnt/host_root$DIR/measured.jsonl" "$DIR/measured-c${C}.log"; then
    echo "  measured rc=$? (C=$C)"
  fi
  tail -c +$((OFFSET + 1)) "$CT_LOG" > "$DIR/server-c${C}.delta.log" 2>/dev/null || true
  # validate
  if health && [ -f "$DIR/measured.jsonl" ]; then
    python3 - "$DIR/measured.jsonl" "$((3*C))" <<'PYEOF' || { echo "  invalid record for C=$C"; continue; }
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
    if [ $? -eq 0 ]; then CHOSEN_C=$C; break; fi
  else
    echo "  no valid measured.jsonl / server unhealthy for C=$C (rc=$?)"
  fi
done

if [ -n "$CHOSEN_C" ]; then
  echo "RESULT: packed ctx262144-max SUCCEEDED at C=$CHOSEN_C" | tee "$RUN_ROOT/RESULT.txt"
else
  echo "RESULT: packed ctx262144-max FAILED at all tried C (17/16/15)" | tee "$RUN_ROOT/RESULT.txt"
fi
kill_server
echo "=== retry done $(date -u +%H:%M:%S)Z; artifacts at $RUN_ROOT ==="
[ -n "$CHOSEN_C" ] && exit 0 || exit 1
