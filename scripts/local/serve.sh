#!/usr/bin/env bash
# =====================================================================
# serve.sh -- boot (or stop) a production-fork DeepSeek-V4-Flash-0731 TP server
# on the local GPU node, inside the reproducible SGLang container.
#
#   serve.sh native             untouched 0731, stock 584-byte C4 (default)
#   serve.sh packed             328-byte packed C4 (Remnant)
#   serve.sh <mode> stop        kill the server on $PORT
#
# Both modes use the same production fork and differ only in cache format.
# They use the fp4-native MoE runner (flashinfer_mxfp4), mem-frac 0.88,
# 1M ctx cap, fp8 KV, and DeepSeek reasoning/tool parsers (needed by the
# agentic evals; harmless for benches). Native uses the fork's default cache
# format; packed passes --dsv4-c4-cache-format remnant.
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
case "$ACTION" in
  boot|stop) ;;
  *) echo "usage: $0 <native|packed> [stop]"; exit 1 ;;
esac
HICACHE=${HICACHE:-0}
[ "$HICACHE" = 1 ] || [ "$HICACHE" = 0 ] || { echo "HICACHE must be 0 or 1"; exit 1; }

SERVE_LOG="$LOG_HOST/serve_${MODE}$([ "$HICACHE" = 1 ] && echo _hicache).log"  # host-side log path
SERVE_LOG_CT=$(to_ct "$SERVE_LOG")             # same file inside container
mkdir -p "$LOG_HOST"

if [ "$ACTION" = stop ]; then
  kill_port
  echo "stopped $MODE server on port $PORT"
  exit 0
fi

kill_port

# --- shared production launcher ---------------------------------------
TREE="$SGLANG_ROOT_CT"
DECODE_CFG_QUOTED=$(printf '%q' "$DECODE_CFG")

echo "== serve $MODE on gpus=$GPUS port=$PORT master=$MASTER_PORT (log: $SERVE_LOG) =="
: > "$SERVE_LOG"   # truncate for a clean boot log (host side)

ct "
  cd $TREE
  export CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$MASTER_PORT
  export MODEL_NAME=$MODEL_NAME TP=$TP MEM_FRAC=$MEM_FRAC CTX_LEN=$CTX_LEN
  export MAX_RUN=$MAX_RUN CHUNK=$CHUNK HICACHE=$HICACHE
  export DECODE_CFG=$DECODE_CFG_QUOTED
  export PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT
  export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  nohup bash $REPO_CT/scripts/local/run-server.sh $MODE $MODEL_CT 0.0.0.0 $PORT \
    > $SERVE_LOG_CT 2>&1 &
  echo \"launched pid \$!\"
"

wait_health 240 || { tail -40 "$SERVE_LOG"; exit 1; }
sleep 3
POOL=$(pool_of "$SERVE_LOG")
echo "  pool(max_total_num_tokens)=${POOL:-UNKNOWN}"
boot_markers "$SERVE_LOG" | sed 's/^/  /'
echo "serve $MODE UP on port $PORT -- leave running, tear down with: $0 $MODE stop"
