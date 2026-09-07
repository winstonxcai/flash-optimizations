#!/usr/bin/env bash
# Verify the "shallow-concurrency" hypothesis for the fp4-native max-concurrency
# table: at ctx 384k (393216 input + 2048 output) the pools are ~full at the same
# geometry as the old FP8-era C9-vs-C11 point that measured +14.5% throughput.
#
#   native pool 3,730,944 (no expandable_segments) -> C=9 at 384k
#   packed pool 4,519,168                              -> C=11 at 384k
#   But a 384k-context prefill OOMs on the default allocator (2.38 GiB TVM-side
#   malloc vs 1.94 GiB driver-free while PyTorch held 2.87 GiB reserved-but-
#   unallocated). The launcher therefore sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
#   which shifts the measured pools to native 3,610,880 / packed 4,373,760 --
#   still C=9 vs C=11 at 393216+2048 (9*395264=3,557,376; 11*395264=4,347,904).
#
# Points (official sglang.bench_serving, warmup C + measured 3C, output 2048,
# seed 42, flush-cache, random-range-ratio 1.0):
#   native-ctx393216-max    C=9    (native's only operating point == its ceiling)
#   packed-ctx393216-fair   C=9    (packed at native's ceiling)
#   packed-ctx393216-max    C=11   (packed at its own ceiling)
#
# Boots reuse launch_inner_serve_sweep_fp4_tp4.sh (fp4-native stack, extended
# decode graphs max_bs 136, GPU 4-7). Results:
#   results/serve-sweep-fp4/verify384k-shallow/<ts>/<point>/
#   logs/verify384k-shallow.log
set -u
HOST_REPO=/home/jovyan/winstonxcai/flash-optimizations
CONT_REPO=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations
CONTAINER=ruler-eval
MODEL=/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731
INNER_LAUNCH=$CONT_REPO/mustafar/scripts/local/launch_inner_verify384k_tp4.sh
LOG_BASE=$HOST_REPO/mustafar/logs
RESULT_BASE=$HOST_REPO/mustafar/results/serve-sweep-fp4/verify384k-shallow
TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$RESULT_BASE/$TS"
CTX=393216
OUTLEN=2048
SEED=42
PORT=""
CT_LOG=""
mkdir -p "$RUN_ROOT"

die () { echo "FATAL: $*" >&2; exit 1; }

wait_health () {
  local port=$1 n=${2:-180} i
  for i in $(seq 1 "$n"); do
    curl -fsS -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "  health OK after ~$((i*5))s"; return 0; }
    sleep 5
  done
  echo "  health TIMEOUT after ~$((n*5))s on $port" >&2
  return 1
}

health () { curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

kill_port () {
  local port=$1
  ps -eo pid,args | grep "[s]glang.launch_server" | grep -- "--port $port" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
}

boot_leg () {   # $1 leg
  local leg=$1
  case "$leg" in
    packed) PORT=30211; POOL_EXPECT=4373760 ;;
    native) PORT=30212; POOL_EXPECT=3610880 ;;
    *) die "bad leg $leg" ;;
  esac
  CT_LOG=$LOG_BASE/serve_sweep_${leg}.log
  kill_port "$PORT"
  echo "== boot $leg (port $PORT) $(date -u +%H:%M:%S)Z =="
  : > "$CT_LOG"
  docker exec "$CONTAINER" bash -c "LEG=$leg bash $INNER_LAUNCH" || die "inner launcher failed ($leg)"
  wait_health "$PORT" || { tail -30 "$CT_LOG"; die "boot failed $leg"; }
  sleep 3
  POOL=$(grep -aoE "max_total_num_tokens=[0-9]+" "$CT_LOG" | head -1 | grep -oE "[0-9]+")
  echo "  pool=$POOL (expect $POOL_EXPECT)"
  [ "$POOL" = "$POOL_EXPECT" ] || { tail -30 "$CT_LOG"; die "pool $POOL != $POOL_EXPECT"; }
  {
    echo "=== boot markers $leg ($(date -u)) ==="
    grep -aoE "logical_row_bytes=[0-9]+ layers=[0-9]+|max_total_num_tokens=[0-9]+|available_gpu_mem=[0-9.]+ GB|Capture target decode CUDA graph end[^,]*elapsed=[0-9.]+ s|is fired up and ready" "$CT_LOG" | head -20
  } > "$RUN_ROOT/boot-markers-${leg}.txt"
  cat "$RUN_ROOT/boot-markers-${leg}.txt"
}

bench_wave () {   # $1 C $2 N $3 outfile(container) $4 log(host)
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

run_point () {   # $1 label $2 C $3 leg
  local label=$1 C=$2 leg=$3
  local dir="$RUN_ROOT/$label"
  mkdir -p "$dir"
  local OUT_CT="/mnt/host_root$dir/measured.jsonl"
  local OFFSET; OFFSET=$(wc -c < "$CT_LOG")
  echo "==== $label C=$C ($(date -u +%H:%M:%S)Z) ===="
  bench_wave "$C" "$C" "/tmp/v384-warm.jsonl" "$dir/warmup.log"
  echo "  warmup rc=$? (${C} prompts)"
  if ! health; then echo "  SERVER DOWN after warmup"; echo "  $label FAILED (server down after warmup)" | tee -a "$RUN_ROOT/RESULT.txt"; return 1; fi
  bench_wave "$C" "$((3*C))" "$OUT_CT" "$dir/measured.log"
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

echo "=== verify384k-shallow $(date -u +%H:%M:%S)Z ==="

# native leg: its single ceiling point C=9
boot_leg native
run_point native-ctx393216-max 9 native
kill_port 30212

# packed leg: fair (C=9) then max (C=11) on one boot
boot_leg packed
run_point packed-ctx393216-fair 9 packed
run_point packed-ctx393216-max 11 packed
kill_port 30211

echo "=== verify384k done $(date -u +%H:%M:%S)Z; artifacts at $RUN_ROOT ==="
exit 0
