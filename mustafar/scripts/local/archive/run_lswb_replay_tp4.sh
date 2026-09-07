#!/usr/bin/env bash
# Orchestrator (runs on HOST 10.72.1.175) for the LongCodeBench business-replay
# run of the STAGE-1 packed 0731 model + fork-untouched control, both TP4 on
# GPUs 0,1,2,3 (a free TP4 slice on this shared node; was 4-7 until wl71785 freed 0-3).
#
#   LEG          inner launcher                         port  MASTER_PORT
#   packed       launch_inner_packed_fp4_tp4_lswb_30211.sh  30211  29626
#   untouched    launch_inner_untouched_fp4_tp4_lswb_30212.sh 30212 29628
#
# Usage:  run_lswb_replay_tp4.sh <packed|untouched|both> [EXPERT_MODE]
#   EXPERT_MODE=native (default, fp4 via flashinfer_mxfp4 cutlass) | dequant (fallback).
#   MUST be identical for packed and untouched legs so packing is the only delta.
#
# Per leg: boot server inside ruler-eval -> wait /health -> verify boot markers ->
# run business_replay client (c15 @1200s) -> collect artifacts -> kill server.
# Results land in flash-optimizations/mustafar/results/lswb-replay/<leg>/<ts>/.
set -u
LEGARG=${1:-both}
EXPERT_MODE=${EXPERT_MODE:-native}

HOST_REPO=/home/jovyan/winstonxcai/flash-optimizations
CONT_REPO=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations
CONTAINER=ruler-eval
MODEL_PATH_CONT=/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731
BENCH=/home/jovyan/wenyuhong/benchmarks
RUNNER=/usr/bin/python3
RESULT_BASE=$HOST_REPO/mustafar/results/lswb-replay
TS=$(date +%Y%m%d_%H%M%S)
C=15
DUR=1200

die () { echo "FATAL: $*" >&2; exit 1; }

preflight () {
  docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" || die "container $CONTAINER not running"
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n '1,4p')
  for u in $used; do
    [ "$u" -lt 200 ] || die "GPUs 0-3 not free (used ${u} MiB). Aborting before launch."
  done
}

wait_health () {
  local port=$1
  local waited=0
  while [ $waited -lt 420 ]; do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "health OK after ~${waited}s (port $port)"; return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  echo "health TIMEOUT after ${waited}s (port $port)"; return 1
}

kill_server () {
  docker exec "$CONTAINER" bash -c "pkill -9 -f 'sglang.launch_server' " >/dev/null 2>&1
  sleep 3
  echo "killed lswb-replay server in $CONTAINER"
}

run_leg () {
  local leg=$1 inner=$2 port=$3
  local RUN_ROOT="$RESULT_BASE/$leg/$TS"
  local CLIENT_DIR="$RUN_ROOT/client"
  mkdir -p "$CLIENT_DIR"

  echo "================ [$leg] boot ($EXPERT_MODE) $(date -u +%H:%M:%S) ================"
  docker exec "$CONTAINER" bash -c "EXPERT_MODE=$EXPERT_MODE bash $CONT_REPO/mustafar/scripts/local/$inner" || die "launcher failed for $leg"
  wait_health "$port" || { echo "---- boot log tail ----"; docker exec "$CONTAINER" bash -c "tail -60 $CONT_REPO/mustafar/logs/serve_lswb_replay_$leg.log"; kill_server; die "boot failed for $leg"; }

  # Boot marker excerpt
  {
    echo "=== boot markers ($leg, EXPERT_MODE=$EXPERT_MODE, $(date -u)) ==="
    docker exec "$CONTAINER" bash -c "grep -aE 'Dequantized FP4|quant_method|Auto-detected DSV4|Mustafar packed C4 pool|max_total_num_tokens=|context_len=|Capture target decode CUDA graph end|layers=21' $CONT_REPO/mustafar/logs/serve_lswb_replay_$leg.log | head -30"
  } > "$RUN_ROOT/boot-markers.txt"
  cat "$RUN_ROOT/boot-markers.txt"

  # Health + model id + smoke
  curl -fsS "http://127.0.0.1:$port/v1/models" > "$RUN_ROOT/models.json" && echo "models: $(head -c 300 "$RUN_ROOT/models.json")"

  echo "================ [$leg] client c$C @ ${DUR}s $(date -u +%H:%M:%S) ================"
  # GPU sampler during the client window
  ( for i in $(seq 1 300); do
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$RUN_ROOT/gpu-samples.csv"
      sleep 20
    done ) &
  SAMPLER=$!

  ( cd "$BENCH" && "$RUNNER" -B harnesses/business_replay/runner.py \
      --result-root "$CLIENT_DIR" \
      --dataset-root "$BENCH/datasets/h20-dsv4pro/longcodebench_openai" \
      --dataset-manifest-input "$BENCH/cache/official-longswebench/longswebench-openai-v1.json" \
      --adapter "$BENCH/harnesses/business_replay/adapters/openai_sse.py" \
      --base-url "http://127.0.0.1:$port" \
      --model deepseek-v4-flash \
      --max-requests 4916 --max-concurrency "$C" --max-duration "$DUR" \
      --arrival-mode immediate --time-scale 60 --max-gap 30 \
      --timeout 21600 --minimum-success-rate 0.99 \
      --expected-protocol business-user-replay-v2 \
      --audit-level candidate --return-cached-tokens-details \
      > "$RUN_ROOT/client.log" 2>&1 )
  CLIENT_RC=$?
  kill "$SAMPLER" 2>/dev/null
  echo "[$leg] client rc=$CLIENT_RC"
  echo "==== [$leg] client.log ===="; cat "$RUN_ROOT/client.log"

  kill_server
  echo "[$leg] artifacts at $RUN_ROOT"
  [ $CLIENT_RC -eq 0 ] || echo "[$leg] WARNING: runner exited rc=$CLIENT_RC (valid=false)"
}

preflight

case "$LEGARG" in
  packed)    run_leg packed    launch_inner_packed_fp4_tp4_lswb_30211.sh    30211 ;;
  untouched) run_leg untouched launch_inner_untouched_fp4_tp4_lswb_30212.sh 30212 ;;
  both)
    run_leg packed    launch_inner_packed_fp4_tp4_lswb_30211.sh    30211
    run_leg untouched launch_inner_untouched_fp4_tp4_lswb_30212.sh 30212
    ;;
  *) die "unknown leg '$LEGARG' (packed|untouched|both)" ;;
esac

echo "==== done $(date -u +%H:%M:%S) ===="
