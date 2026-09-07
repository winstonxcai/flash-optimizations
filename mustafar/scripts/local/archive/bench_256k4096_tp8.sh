#!/usr/bin/env bash
# Speed measurement on the TP8 (ctx 524288, packed C4) server: 256k input,
# 4096 output, concurrency 1 and 2. Server must already be running on $PORT.
#   Usage: ./bench_256k4096_tp8.sh
#   Output: mustafar/results/official-256k4096/tp8-c{1,2}.jsonl (+ .meta)
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
. "$DIR/common.sh"

RUN_DIR="$RESULTS_HOST/official-256k4096"
RUN_DIR_CT="/mnt/host_root$RUN_DIR"
mkdir -p "$RUN_DIR"
TLEN=262144
OUTLEN=4096

for CP in c1 c2; do
  case $CP in c1) CONC=1; NP=2; SEED=4300;; c2) CONC=2; NP=4; SEED=4301;; esac
  START=$(date +%s)
  echo "== [tp8 $CP] C=$CONC NP=$NP len=$TLEN out=$OUTLEN"
  docker exec ruler-eval bash -c "
    export HF_ENDPOINT=https://hf-mirror.com
    export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
    cd /sgl-workspace/sglang-lowrank/python
    python3 -m sglang.bench_serving \
      --backend sglang --host 127.0.0.1 --port $PORT \
      --model $MODEL --tokenizer $MODEL \
      --dataset-name random --random-input-len $TLEN --random-output-len $OUTLEN \
      --random-range-ratio 1.0 --num-prompts $NP --max-concurrency $CONC \
      --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
      --output-file $RUN_DIR_CT/tp8-$CP.jsonl --output-details --seed $SEED
  "
  RC=$?
  SEC=$(( $(date +%s) - START ))
  printf 'num_prompts=%s\noutput_tokens=%s\nreturncode=%s\nseconds=%s\n' "$NP" "$OUTLEN" "$RC" "$SEC" > "$RUN_DIR/tp8-$CP.meta"
  if [ "$RC" -ne 0 ]; then
    echo "[tp8 $CP] bench FAILED rc=$RC" >&2
    exit 1
  fi
  echo "[tp8 $CP] done in ${SEC}s rc=0"
done
echo "ALL_256K_DONE"
