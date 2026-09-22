#!/usr/bin/env bash
# Start one production-fork SGLang server in the foreground.
#
# Usage: run-server.sh <native|packed> <model> <host> <port>
# The caller owns process groups, logging, and cleanup. Shared server flags
# live here so local-container and Modal launches cannot silently diverge.
set -u

MODE=${1:-}
MODEL=${2:-}
HOST=${3:-127.0.0.1}
PORT_VALUE=${4:-}

case "$MODE" in
  native) CACHE_FORMAT=native ;;
  packed) CACHE_FORMAT=remnant ;;
  *) echo "usage: $0 <native|packed> <model> <host> <port>" >&2; exit 2 ;;
esac
[ -n "$MODEL" ] || { echo "missing model path" >&2; exit 2; }
[ -n "$PORT_VALUE" ] || { echo "missing port" >&2; exit 2; }
case "${HICACHE:-0}" in
  0|1) ;;
  *) echo "HICACHE must be 0 or 1" >&2; exit 2 ;;
esac

args=(
  serve
  --model-path "$MODEL"
  --served-model-name "${MODEL_NAME:-deepseek-v4-flash}"
  --tp "${TP:-4}"
  --trust-remote-code
  --mem-fraction-static "${MEM_FRAC:-0.88}"
  --context-length "${CTX_LEN:-1048576}"
  --max-running-requests "${MAX_RUN:-256}"
  --chunked-prefill-size "${CHUNK:-8192}"
  --dsv4-c4-cache-format "$CACHE_FORMAT"
  --kv-cache-dtype fp8_e4m3
  --moe-runner-backend flashinfer_mxfp4
  --reasoning-parser deepseek-v4
  --tool-call-parser deepseekv4
  --host "$HOST"
  --port "$PORT_VALUE"
  --cuda-graph-config "${DECODE_CFG:?DECODE_CFG is required}"
  --skip-server-warmup
  --watchdog-timeout 1800
)

if [ "${HICACHE:-0}" = 1 ]; then
  export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
  args+=(
    --enable-hierarchical-cache
    --hicache-ratio 2.75
    --hicache-write-policy write_through
    --hicache-io-backend direct
    --hicache-mem-layout page_first_direct
  )
fi

exec sglang "${args[@]}"
