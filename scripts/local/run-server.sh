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

# The local-container launcher sets this after starting us in a dedicated
# session. Record identity before exec so cleanup can recognize this exact
# process even while SGLang is still importing or initializing.
if [ -n "${SERVER_PID_FILE:-}" ]; then
  stat_line=$(<"/proc/$$/stat")
  stat_tail=${stat_line##*) }
  read -r -a stat_fields <<< "$stat_tail"
  process_group=${stat_fields[2]:-}
  session_id=${stat_fields[3]:-}
  start_ticks=${stat_fields[19]:-}
  if [ "$process_group" != "$$" ] || [ "$session_id" != "$$" ] || [ -z "$start_ticks" ]; then
    echo "server must start in its own session to enable safe cleanup" >&2
    exit 1
  fi
  pid_file_tmp="$SERVER_PID_FILE.$$"
  printf '%s %s %s %s %s\n' "$$" "$process_group" "$session_id" "$start_ticks" "$PORT_VALUE" > "$pid_file_tmp"
  mv -f -- "$pid_file_tmp" "$SERVER_PID_FILE"
fi

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

exec sglang "${args[@]}"
