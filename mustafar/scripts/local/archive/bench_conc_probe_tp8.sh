#!/usr/bin/env bash
# Probe safe concurrency on the TP8 packed server via bench_serving.
# Decode-heavy workload (short input, 1k output) so the decode batch size
# is what's actually stressed. No --flush-cache (keeps each leg short).
#   Output: mustafar/results/concurrency-probe/tp8-conc-<C>.jsonl
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
. "$DIR/common.sh"

RUN_DIR="$RESULTS_HOST/concurrency-probe"
RUN_DIR_CT="/mnt/host_root$RUN_DIR"
mkdir -p "$RUN_DIR"
TLEN=131072
OUTLEN=2048
SEED=4400

for CONC in 1 2 4 6 8; do
  NP=$CONC
  SEED=$((4400 + CONC))
  echo "== [tp8 conc=$CONC] C=$CONC NP=$NP len=$TLEN out=$OUTLEN"
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
      --output-file $RUN_DIR_CT/tp8-conc-$CONC.jsonl --output-details --seed $SEED
  " > "$RUN_DIR/tp8-conc-$CONC.benchout" 2>&1
  RC=$?
  echo "[tp8 conc=$CONC] rc=$RC"
  grep -E "Successful requests|Request throughput|Output token throughput|Total token throughput|Mean TTFT|Mean TPOT|Mean ITL" "$RUN_DIR/tp8-conc-$CONC.benchout"
done
echo "CONC_PROBE_DONE"
