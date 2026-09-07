#!/usr/bin/env bash
# Local replica of modal app.py::tp4_graph_decode_ab — short native/packed TP4
# decode A/B with BS1/BS2 CUDA graphs at 64k input, 128 output tokens, at
# concurrency 1 and 2 (num-prompts 2 and 4). Fail-closed on output length,
# request errors, and missing decode-graph config.
#
#   Usage: ./tp4_graph_decode_ab.sh         (4 free H100s; port ${PORT:-30211})
#   Requires: `python -m mustafar patch` applied to /sgl-workspace/sglang-lowrank.
#   Output:  mustafar/results/tp4-graph-decode-ab.json (+ .partial.json after
#            every point; server logs under mustafar/logs/graph-decode-*.log)
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
. "$DIR/common.sh"

RUN_DIR="$RESULTS_HOST/official-graph-decode"
RUN_DIR_CT="/mnt/host_root$RUN_DIR"
mkdir -p "$RUN_DIR"
TLEN=65536
OUTLEN=128
GRAPH_CONFIG='{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}'
START=$(date +%s)

for LEG in topmag50_packed native; do
  [ "$LEG" = topmag50_packed ] && PACKED=1 || PACKED=0
  HOST_LOG="$LOG_HOST/graph-decode-$LEG-server.log"
  LOG_CT="/mnt/host_root$HOST_LOG"
  : > "$HOST_LOG"
  kill_server "$PORT"

  echo "== launching $LEG graph-decode server (packed=$PACKED) port=$PORT"
  docker exec ruler-eval bash -c "
    export CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$MASTER_PORT
    export SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=$KEEP SGLANG_OPT_TOPMAG_PACKED_C4=$PACKED
    export PYTHONPATH=$PYTHONPATH
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
    cd /sgl-workspace/sglang-lowrank/python
    nohup python3 -m sglang.launch_server \
      --model-path $MODEL --served-model-name deepseek-v4-flash \
      --tp 4 --trust-remote-code --mem-fraction-static 0.93 \
      --context-length 135168 --max-running-requests 2 \
      --chunked-prefill-size 4096 --fp8-gemm-backend triton \
      --host 0.0.0.0 --port $PORT \
      --cuda-graph-config '$GRAPH_CONFIG' \
      --skip-server-warmup --reasoning-parser deepseek-v4 \
      --tool-call-parser deepseekv4 --watchdog-timeout 1800 \
      > '$LOG_CT' 2>&1 &
    echo launched pid \$!
  "
  LEG_START=$(date +%s)
  wait_pool "$HOST_LOG" 240 || exit 1
  echo "$(( $(date +%s) - LEG_START ))" > "$RUN_DIR/$LEG.ready_sec"

  for CP in c1 c2; do
    case $CP in c1) CONC=1; NP=2; SEED=4200;; c2) CONC=2; NP=4; SEED=4201;; esac
    OFFSET=$(wc -c < "$HOST_LOG")
    BENCH_START=$(date +%s)
    echo "== [$LEG $CP] C=$CONC NP=$NP len=$TLEN out=$OUTLEN"
    docker exec ruler-eval bash -c "
      export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
      cd /sgl-workspace/sglang-lowrank/python
      python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model $MODEL --tokenizer $MODEL \
        --dataset-name random --random-input-len $TLEN --random-output-len $OUTLEN \
        --random-range-ratio 1.0 --num-prompts $NP --max-concurrency $CONC \
        --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
        --output-file $RUN_DIR_CT/$LEG-$CP.jsonl --output-details --seed $SEED
    "
    RC=$?
    BENCH_SEC=$(( $(date +%s) - BENCH_START ))
    if [ -s "$HOST_LOG" ]; then
      tail -c +$((OFFSET + 1)) "$HOST_LOG" > "$RUN_DIR/$LEG-$CP.logdelta" 2>/dev/null || true
    else
      : > "$RUN_DIR/$LEG-$CP.logdelta"
    fi
    printf 'num_prompts=%s\noutput_tokens=%s\nreturncode=%s\nseconds=%s\n' "$NP" "$OUTLEN" "$RC" "$BENCH_SEC" > "$RUN_DIR/$LEG-$CP.meta"
    if [ "$RC" -ne 0 ]; then
      echo "[$LEG $CP] bench FAILED rc=$RC (see $RUN_DIR/$LEG-$CP.logdelta)" >&2
      exit 1
    fi
    python3 "$DIR/tp4_parse.py" validate-record "$RUN_DIR/$LEG-$CP.jsonl" "$NP" "$OUTLEN"
    python3 "$DIR/tp4_parse.py" assemble-graphdecode "$RESULTS_HOST" "$RESULTS_HOST/tp4-graph-decode-ab.partial.json"
  done
  kill_server "$PORT"
done

TOTAL=$(( $(date +%s) - START ))
echo "$TOTAL" > "$RUN_DIR/total_wall_sec"
python3 "$DIR/tp4_parse.py" assemble-graphdecode "$RESULTS_HOST" "$RESULTS_HOST/tp4-graph-decode-ab.json"
echo "DONE -> $RESULTS_HOST/tp4-graph-decode-ab.json"
