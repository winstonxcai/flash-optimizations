#!/usr/bin/env bash
# Local replica of modal app.py::tp4_capacity_ceiling — dense vs packed TP4
# capacity ceiling using the official `sglang.bench_serving`, one page-aligned
# point per context (32k/64k/128k) x 2 legs (packed, native).
#
#   Usage: ./tp4_capacity_ceiling.sh        (4 free H100s; port ${PORT:-30211})
#   Requires: `python -m mustafar patch` applied to /sgl-workspace/sglang-lowrank.
#   Output:  mustafar/results/tp4-capacity-ceiling.json (+ .partial.json after
#            every point; server logs under mustafar/logs/capacity-*.log)
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
. "$DIR/common.sh"

RUN_DIR="$RESULTS_HOST/official-capacity"
RUN_DIR_CT="/mnt/host_root$RUN_DIR"
mkdir -p "$RUN_DIR"
START=$(date +%s)

for LEG in topmag50_packed native; do
  [ "$LEG" = topmag50_packed ] && PACKED=1 || PACKED=0
  HOST_LOG="$LOG_HOST/capacity-$LEG-server.log"
  LOG_CT="/mnt/host_root$HOST_LOG"
  : > "$HOST_LOG"
  kill_server "$PORT"

  echo "== launching $LEG server (packed=$PACKED) port=$PORT"
  docker exec ruler-eval bash -c "
    export CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$MASTER_PORT
    export SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=$KEEP SGLANG_OPT_TOPMAG_PACKED_C4=$PACKED
    export PYTHONPATH=$PYTHONPATH
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
    cd /sgl-workspace/sglang-lowrank/python
    nohup python3 -m sglang.launch_server \
      --model-path $MODEL --served-model-name deepseek-v4-flash \
      --tp 4 --trust-remote-code --mem-fraction-static 0.93 \
      --context-length 135168 --max-running-requests 64 \
      --chunked-prefill-size 4096 --fp8-gemm-backend triton \
      --host 0.0.0.0 --port $PORT --disable-cuda-graph \
      --skip-server-warmup --reasoning-parser deepseek-v4 \
      --tool-call-parser deepseekv4 --watchdog-timeout 1800 \
      > '$LOG_CT' 2>&1 &
    echo launched pid \$!
  "
  LEG_START=$(date +%s)
  wait_pool "$HOST_LOG" 240 || exit 1
  echo "$(( $(date +%s) - LEG_START ))" > "$RUN_DIR/$LEG.ready_sec"

  for CTX in 32k 64k 128k; do
    case $CTX in 32k) TLEN=32768;; 64k) TLEN=65536;; 128k) TLEN=131072;; esac
    RESIDENT=$(python3 "$DIR/tp4_parse.py" resident "$HOST_LOG" "$CTX")
    OFFSET=$(wc -c < "$HOST_LOG")
    BENCH_START=$(date +%s)
    echo "== [$LEG $CTX] C=$RESIDENT len=$TLEN out=16"
    docker exec ruler-eval bash -c "
      export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
      cd /sgl-workspace/sglang-lowrank/python
      python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model $MODEL --tokenizer $MODEL \
        --dataset-name random --random-input-len $TLEN --random-output-len 16 \
        --random-range-ratio 1.0 --num-prompts $RESIDENT --max-concurrency $RESIDENT \
        --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
        --output-file $RUN_DIR_CT/$LEG-$CTX.jsonl --output-details --seed 42
    "
    RC=$?
    BENCH_SEC=$(( $(date +%s) - BENCH_START ))
    if [ -s "$HOST_LOG" ]; then
      tail -c +$((OFFSET + 1)) "$HOST_LOG" > "$RUN_DIR/$LEG-$CTX.logdelta" 2>/dev/null || true
    else
      : > "$RUN_DIR/$LEG-$CTX.logdelta"
    fi
    printf 'returncode=%s\nseconds=%s\n' "$RC" "$BENCH_SEC" > "$RUN_DIR/$LEG-$CTX.benchmeta"
    python3 "$DIR/tp4_parse.py" assemble-capacity "$RESULTS_HOST" "$RESULTS_HOST/tp4-capacity-ceiling.partial.json"
    if [ "$RC" -ne 0 ]; then
      echo "[$LEG $CTX] bench FAILED rc=$RC (see $RUN_DIR/$LEG-$CTX.logdelta)" >&2
    fi
  done
  kill_server "$PORT"
done

TOTAL=$(( $(date +%s) - START ))
echo "$TOTAL" > "$RUN_DIR/total_wall_sec"
python3 "$DIR/tp4_parse.py" assemble-capacity "$RESULTS_HOST" "$RESULTS_HOST/tp4-capacity-ceiling.json"
echo "DONE -> $RESULTS_HOST/tp4-capacity-ceiling.json"
