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
# Env overrides (all optional): PORT, GPUS, MASTER_PORT, TP, DECODE_CFG,
# MEM_FRAC, CTX_LEN, MAX_RUN, CHUNK. Boot log:
#   <LOG_HOST>/serve_<native|packed>.log
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
SERVE_LOG="$LOG_HOST/serve_${MODE}.log"  # host-side log path
SERVE_LOG_CT=$(to_ct "$SERVE_LOG")             # same file inside container
mkdir -p "$LOG_HOST"

if [ "$ACTION" = stop ]; then
  kill_port
  echo "stopped $MODE server on port $PORT"
  exit 0
fi

kill_port

if health; then
  echo "FATAL: an unowned server is still healthy on port $PORT; refusing to replace it" >&2
  exit 1
fi

# --- shared production launcher ---------------------------------------
TREE="$SGLANG_ROOT_CT"

echo "== serve $MODE on gpus=$GPUS port=$PORT master=$MASTER_PORT (log: $SERVE_LOG) =="
: > "$SERVE_LOG"   # truncate for a clean boot log (host side)

ct_script "$TREE" "$GPUS" "$MASTER_PORT" "$MODEL_NAME" "$TP" "$MEM_FRAC" \
  "$CTX_LEN" "$MAX_RUN" "$CHUNK" "$DECODE_CFG" "$SGLANG_PY" "$REPO_CT" \
  "$MODE" "$MODEL_CT" "$PORT" "$SERVE_LOG_CT" "$SERVER_PID_FILE_CT" <<'BASH'
set -u
tree=$1
gpus=$2
master_port=$3
model_name=$4
tp=$5
mem_frac=$6
ctx_len=$7
max_run=$8
chunk=$9
decode_cfg=${10}
sglang_py=${11}
repo_ct=${12}
mode=${13}
model_ct=${14}
port=${15}
serve_log_ct=${16}
pid_file=${17}

cd -- "$tree"
export CUDA_VISIBLE_DEVICES="$gpus" MASTER_PORT="$master_port"
export MODEL_NAME="$model_name" TP="$tp" MEM_FRAC="$mem_frac" CTX_LEN="$ctx_len"
export MAX_RUN="$max_run" CHUNK="$chunk" DECODE_CFG="$decode_cfg"
export PYTHONPATH="/opt/sglang-runtime-fixes:$sglang_py:$repo_ct"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
command -v setsid >/dev/null 2>&1 || { echo "setsid is required" >&2; exit 127; }
nohup setsid bash "$repo_ct/scripts/local/run-server.sh" "$mode" "$model_ct" 0.0.0.0 "$port" \
  > "$serve_log_ct" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$pid_file"
echo "launched pid $pid"
BASH

wait_health 240 || { tail -40 "$SERVE_LOG"; exit 1; }
sleep 3
POOL=$(pool_of "$SERVE_LOG")
echo "  pool(max_total_num_tokens)=${POOL:-UNKNOWN}"
boot_markers "$SERVE_LOG" | sed 's/^/  /'
echo "serve $MODE UP on port $PORT -- leave running, tear down with: $0 $MODE stop"
