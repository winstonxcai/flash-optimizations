#!/usr/bin/env bash
# =====================================================================
# serve.sh -- boot (or stop) a DeepSeek-V4-Flash-0731 TP server for the
# remnant study on this machine, inside the sglang container.
#
#   serve.sh native             untouched 0731, stock 584-byte C4 (default)
#   serve.sh packed             328-byte packed C4 (Remnant)
#   serve.sh <mode> stop        kill the server on $PORT
#
# All modes use the fp4-native MoE runner (flashinfer_mxfp4), mem-frac 0.88,
# 1M ctx cap, fp8 KV, and DeepSeek reasoning/tool parsers (needed by the
# agentic evals; harmless for benches). The legs differ by WHICH source tree
# serves the Remnant fork (SGLANG_PY_FORK). Native uses the fork's default
# cache format; packed passes --dsv4-c4-cache-format remnant.
#
# HICACHE=1 (optional) additionally enables SGLang's hierarchical cache
# (GPU L1 <-> CPU DRAM L2) with the locked remnant settings:
#   env SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 and flags
#   --enable-hierarchical-cache --hicache-ratio 2.75
#   --hicache-write-policy write_through --hicache-io-backend direct
#   --hicache-mem-layout page_first_direct
# (no L3/storage). Boot log gains a _hicache suffix.
#
# Env overrides (all optional): PORT, GPUS, MASTER_PORT, TP, DECODE_CFG,
# MEM_FRAC, CTX_LEN, MAX_RUN, CHUNK, HICACHE. Boot log:
#   <LOG_HOST>/serve_<native|packed>[,_hicache].log
# Server is left RUNNING; use "serve.sh <mode> stop" to tear it down.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

MODE=${1:-}; ACTION=${2:-boot}
case "$MODE" in
  native|packed) ;;
  *) echo "usage: $0 <native|packed> [stop]"; exit 1 ;;
esac
HICACHE=${HICACHE:-0}
[ "$HICACHE" = 1 ] || [ "$HICACHE" = 0 ] || { echo "HICACHE must be 0 or 1"; exit 1; }

SERVE_LOG="$LOG_HOST/serve_${MODE}$([ "$HICACHE" = 1 ] && echo _hicache).log"  # host-side log path
SERVE_LOG_CT=$(to_ct "$SERVE_LOG")             # same file inside container

if [ "$ACTION" = stop ]; then
  kill_port
  echo "stopped $MODE server on port $PORT"
  exit 0
fi

kill_port

# --- per-mode flag + tree -----------------------------------------------
TREE="$SGLANG_PY_FORK"
case "$MODE" in
  native)
    CACHE_FORMAT=native
    CT_PYTHONPATH="$TREE"
    ;;
  packed)
    CACHE_FORMAT=remnant
    CT_PYTHONPATH="$TREE"
    ;;
esac

# --- hierarchical-cache (HiCache L2) env + flags ---------------------
HICACHE_ARGS=()
if [ "$HICACHE" = 1 ]; then
  HICACHE_ARGS=(--enable-hierarchical-cache --hicache-ratio 2.75 \
    --hicache-write-policy write_through --hicache-io-backend direct \
    --hicache-mem-layout page_first_direct)
fi

echo "== serve $MODE on gpus=$GPUS port=$PORT master=$MASTER_PORT (log: $SERVE_LOG) =="
: > "$SERVE_LOG"   # truncate for a clean boot log (host side)

ct "
  cd $TREE
  export CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$MASTER_PORT
  [ \"$HICACHE\" = 1 ] && export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
  export PYTHONPATH=$CT_PYTHONPATH
  export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup sglang serve \
    --model-path $MODEL_CT --served-model-name $MODEL_NAME \
    --tp $TP --trust-remote-code --mem-fraction-static $MEM_FRAC \
    --context-length $CTX_LEN --max-running-requests $MAX_RUN \
    --chunked-prefill-size $CHUNK \
    --dsv4-c4-cache-format $CACHE_FORMAT \
    --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4 \
    --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
    --host 0.0.0.0 --port $PORT \
    --cuda-graph-config '$DECODE_CFG' \
    ${HICACHE_ARGS[*]} \
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
