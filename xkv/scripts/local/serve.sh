#!/usr/bin/env bash
# =====================================================================
# serve.sh -- boot (or stop) a DeepSeek-V4-Flash-0731 TP server for the
# xkv study on this machine, inside the xkv container.
#
#   serve.sh native             untouched 0731, stock 584-byte C4, no xkv
#   serve.sh xkv                200-byte low-rank C4 record (xkv store ON)
#   serve.sh <mode> stop        kill the server on $PORT
#
# All modes use the fp4-native MoE runner (flashinfer_mxfp4), mem-frac 0.88,
# 1M ctx cap, fp8 KV, and DeepSeek reasoning/tool parsers. The legs differ by
# WHICH source tree serves and by the xkv store envs: native serves the pristine
# tree (SGLANG_PY_STOCK, byte-identical sglang); xkv serves the patched fork
# (SGLANG_PY_FORK) with SGLANG_OPT_LOWRANK_KV_STORE=1 and the calibrated basis.
#
# Env overrides (all optional): PORT, GPUS, MASTER_PORT, TP, DECODE_CFG,
# MEM_FRAC, CTX_LEN, MAX_RUN, CHUNK, BASIS_HOST, SG_LOWRANK_BASIS, EXTRA_ENVS.
# Passing DECODE_CFG=$DECODE_CFG_OFF is the eager (no-decode-graph) arm of the
# A/B; EXTRA_ENVS appends raw KEY=VALUE gates last so one can be turned off.
# Boot log:
#   <LOG_HOST>/serve_<native|xkv>[_nograph].log
# Server is left RUNNING; use "serve.sh <mode> stop" to tear it down.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

MODE=${1:-}; ACTION=${2:-boot}
case "$MODE" in
  native|xkv) ;;
  *) echo "usage: $0 <native|xkv> [stop]"; exit 1 ;;
esac

# The A/B arm is named from the effective decode config, so the two arms of the
# graph comparison never overwrite each other's boot log.
ARM=$([ "$DECODE_CFG" = "$DECODE_CFG_OFF" ] && echo _nograph || echo "")
SERVE_LOG="$LOG_HOST/serve_${MODE}${ARM}.log"   # host-side log path
SERVE_LOG_CT=$(to_ct "$SERVE_LOG")              # same file inside container

if [ "$ACTION" = stop ]; then
  kill_port
  echo "stopped $MODE server on port $PORT"
  exit 0
fi

kill_port

# --- per-mode env + tree -----------------------------------------------
# Every mode names the whole gate set explicitly: a value left over from another
# mode would otherwise be silently honored at launch.
TREE="$SGLANG_PY_STOCK"
CT_PYTHONPATH="$SGLANG_PY_STOCK"
MODE_ENVS=(SGLANG_OPT_LOWRANK_KV_STORE=0)
if [ "$MODE" = xkv ]; then
  TREE="$SGLANG_PY_FORK"
  CT_PYTHONPATH="$SGLANG_PY_FORK:$REPO_CT"
  MODE_ENVS=("${LOWRANK_ENVS[@]}" "SG_LOWRANK_BASIS=$(to_ct "$BASIS_HOST")")
fi
# Appended last, so a caller can turn one gate off to isolate it (qa-arms.sh
# boots the fork tree with the store disabled, or with the torch reconstruct
# instead of the fused triton one) without a second copy of this launch line.
EXTRA_ENVS=${EXTRA_ENVS:-}

echo "== serve $MODE$ARM on gpus=$GPUS port=$PORT master=$MASTER_PORT (log: $SERVE_LOG) =="
: > "$SERVE_LOG"   # truncate for a clean boot log (host side)

ct "
  cd $TREE
  export CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$MASTER_PORT
  export ${MODE_ENVS[*]} $EXTRA_ENVS
  export PYTHONPATH=$CT_PYTHONPATH
  export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup sglang serve \
    --model-path $MODEL_CT --served-model-name $MODEL_NAME \
    --tp $TP --trust-remote-code --mem-fraction-static $MEM_FRAC \
    --context-length $CTX_LEN --max-running-requests $MAX_RUN \
    --chunked-prefill-size $CHUNK \
    --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4 \
    --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
    --host 0.0.0.0 --port $PORT \
    --cuda-graph-config '$DECODE_CFG' \
    --skip-server-warmup --watchdog-timeout 1800 \
    > $SERVE_LOG_CT 2>&1 &
  echo \"launched pid \$!\"
"

wait_health 240 || { tail -40 "$SERVE_LOG"; exit 1; }
sleep 3
POOL=$(pool_of "$SERVE_LOG")
echo "  pool(max_total_num_tokens)=${POOL:-UNKNOWN}"
boot_markers "$SERVE_LOG" | sed 's/^/  /'
echo "serve $MODE$ARM UP on port $PORT -- leave running, tear down with: $0 $MODE stop"
