#!/usr/bin/env bash
# =====================================================================
# calibrate.sh -- refit the W3 rank-192 basis from live prefill traffic.
#
#   calibrate.sh [num_prompts] [input_len]
#
# The store needs one 512x512 second-moment matrix per CSA-layer triple. This
# boots the PATCHED tree with XKV_CALIB=1 (the store's hook accumulates x^T x
# instead of writing records), drives `num_prompts` long prefills at
# `input_len`, waits out one flush interval, then finalizes into $BASIS_HOST.
#
# Outputs:
#   $XKV_CALIB_OUT/S_all.pt                    raw per-layer running sums
#   $BASIS_HOST/A_<layer>.pt                   one basis per layer (shared per triple)
#
# The server's answers during the capture pass are meaningless by construction
# (the store is bypassed, so there is nothing to attend over) -- they are
# discarded. Only the accumulated second moments are used.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

NUM=${1:-16}
LEN=${2:-65536}
FLUSH_S=${XKV_CALIB_FLUSH_S:-30}
CALIB_HOST=${XKV_CALIB_OUT:-$HOST_REPO/xkv/ctrl/calib}
CALIB_CT=$(to_ct "$CALIB_HOST")
BASIS_CT=$(to_ct "$BASIS_HOST")
LOG="$LOG_HOST/calib-$(ts).log"

mkdir -p "$CALIB_HOST" "$BASIS_HOST"
echo "== calibrating: $NUM prompts x $LEN tokens -> $BASIS_HOST (log: $LOG)"

kill_port
ct "mkdir -p $CALIB_CT && rm -f $CALIB_CT/S_all.pt $CALIB_CT/.S_all.tmp"

# Boot the patched tree with the store BYPASSED into capture mode. The store's
# env must still be on (that is what arms the compressor hook), but XKV_CALIB=1
# short-circuits it before any record is written.
echo "== boot xkv (capture mode)"
XKV_CALIB=1 XKV_CALIB_OUT="$CALIB_CT" XKV_CALIB_FLUSH_S="$FLUSH_S" \
  DECODE_CFG="$DECODE_CFG_OFF" \
  ct "
    cd $SGLANG_PY_FORK
    export CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$MASTER_PORT
    export PYTHONPATH=$SGLANG_PY_FORK:$REPO_CT
    export SGLANG_OPT_LOWRANK_KV_STORE=1 XKV_RECON_TRITON=1 XKV_COEFF_DIM=192
    export XKV_CALIB=1 XKV_CALIB_OUT=$CALIB_CT XKV_CALIB_FLUSH_S=$FLUSH_S
    export SG_LOWRANK_BASIS=$BASIS_CT
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    nohup sglang serve \
      --model-path $MODEL_CT --served-model-name $MODEL_NAME \
      --tp $TP --trust-remote-code --mem-fraction-static $MEM_FRAC \
      --context-length $CTX_LEN --max-running-requests $MAX_RUN \
      --chunked-prefill-size $CHUNK \
      --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4 \
      --host 0.0.0.0 --port $PORT \
      --cuda-graph-config '$DECODE_CFG_OFF' \
      --skip-server-warmup --watchdog-timeout 1800 \
      > $(to_ct "$LOG") 2>&1 &
  "
wait_health 240 || { tail -40 "$LOG"; exit 1; }

echo "== drive $NUM x $LEN-token prefills (short outputs; answers discarded)"
ct "cd $SGLANG_PY_FORK && python3 -m sglang.bench_serving \
      --backend sglang --host 127.0.0.1 --port $PORT \
      --model $MODEL_CT --tokenizer $MODEL_CT \
      --dataset-name random --dataset-path $SHAREGPT_CT \
      --random-input-len $LEN --random-output-len 8 \
      --random-range-ratio 1.0 --num-prompts $NUM --max-concurrency 4 \
      --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
      --output-file /dev/null --seed $SEED" > "$LOG.bench" 2>&1 \
  || echo "   (bench client returned nonzero; capture sums still count)"

# The worker only writes on a timer, so outlive one interval before tearing down.
echo "== waiting $((FLUSH_S + 5))s for the final flush"
sleep $((FLUSH_S + 5))
kill_port

echo "== finalize"
SG_LOWRANK_BASIS="$BASIS_CT" XKV_CALIB_OUT="$CALIB_CT" \
  ct "cd $REPO_CT && SG_LOWRANK_BASIS=$BASIS_CT XKV_CALIB_OUT=$CALIB_CT python3 -m xkv calib_finalize" \
  | tee -a "$LOG"

echo "== bases in $BASIS_HOST:"; ls -1 "$BASIS_HOST" | head -30
