#!/usr/bin/env bash
# =====================================================================
# serve.sh -- boot (or stop) a DeepSeek-V4-Flash-0731 TP server for the
# mustafar study on this machine, inside the sglang container.
#
#   serve.sh native             untouched 0731, stock 584-byte C4 (TopMag OFF)
#   serve.sh packed             328-byte packed C4 (TopMag50, packed)
#   serve.sh <native|packed> stop     kill the server on $PORT
#
# Both modes use the fp4-native MoE runner (flashinfer_mxfp4), mem-frac 0.88,
# 1M ctx cap, fp8 KV, and DeepSeek reasoning/tool parsers (needed by the
# agentic evals; harmless for benches). The ONLY difference between the legs
# is the three TopMag envs + the fork PYTHONPATH for packed.
#
# Env overrides (all optional): PORT, GPUS, MASTER_PORT, TP, DECODE_CFG,
# MEM_FRAC, CTX_LEN, MAX_RUN, CHUNK. Boot log:
#   <LOG_HOST>/serve_<native|packed>.log
# Server is left RUNNING; use "serve.sh <mode> stop" to tear it down.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

MODE=${1:-}; ACTION=${2:-boot}
[ "$MODE" = native ] || [ "$MODE" = packed ] || { echo "usage: $0 <native|packed> [stop]"; exit 1; }

SERVE_LOG="$LOG_HOST/serve_$MODE.log"          # host-side log path
SERVE_LOG_CT=$(to_ct "$SERVE_LOG")             # same file inside container

if [ "$ACTION" = stop ]; then
  kill_port
  echo "stopped $MODE server on port $PORT"
  exit 0
fi

kill_port

# --- per-mode env -----------------------------------------------------
MODE_ENVS=()          # each entry exported inside the container before launch
if [ "$MODE" = packed ]; then
  MODE_ENVS=(SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=0.5 SGLANG_OPT_TOPMAG_PACKED_C4=1)
  CT_PYTHONPATH="$SGLANG_PY:$REPO_CT"
else
  CT_PYTHONPATH="$SGLANG_PY"
fi

echo "== serve $MODE on gpus=$GPUS port=$PORT master=$MASTER_PORT (log: $SERVE_LOG) =="
: > "$SERVE_LOG"   # truncate for a clean boot log (host side)

ct "
  cd $SGLANG_PY
  export CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$MASTER_PORT
  ${MODE_ENVS[*]:+export ${MODE_ENVS[*]}}
  export PYTHONPATH=$CT_PYTHONPATH
  export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup python3 -m sglang.launch_server \
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
echo "serve $MODE UP on port $PORT -- leave running, tear down with: $0 $MODE stop"
