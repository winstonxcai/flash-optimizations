#!/usr/bin/env bash
# Start one production-fork SGLang server in the foreground.
#
# Usage: run-server.sh <native|triton|kernel> <model> <host> <port>
# Author: Winston Cai.
# The caller owns process groups, logging, and cleanup. Shared server flags
# live here so local-container and Modal launches cannot silently diverge.
set -u

MODE=${1:-}
MODEL=${2:-}
HOST=${3:-127.0.0.1}
PORT_VALUE=${4:-}

case "$MODE" in
  native) CACHE_FORMAT=native; CACHE_IMPLEMENTATION=kernel ;;
  triton|kernel) CACHE_FORMAT=remnant; CACHE_IMPLEMENTATION=$MODE ;;
  packed) CACHE_FORMAT=remnant; CACHE_IMPLEMENTATION=kernel ;;
  *) echo "usage: $0 <native|triton|kernel> <model> <host> <port> (packed aliases kernel)" >&2; exit 2 ;;
esac
case "${STORE_IMPLEMENTATION:-}" in
  ""|triton|kernel) ;;
  *) echo "STORE_IMPLEMENTATION must be triton or kernel" >&2; exit 2 ;;
esac
[ -n "$MODEL" ] || { echo "missing model path" >&2; exit 2; }
[ -n "$PORT_VALUE" ] || { echo "missing port" >&2; exit 2; }
if [ -n "${SERVER_SEED:-}" ]; then
  [[ "$SERVER_SEED" =~ ^[0-9]{1,10}$ ]] && (( 10#$SERVER_SEED <= 2147483647 )) || {
    echo "SERVER_SEED must be an integer in [0, 2147483647]" >&2; exit 2;
  }
fi
if [ -n "${HICACHE_RATIO:-}" ]; then
  [[ "$HICACHE_RATIO" =~ ^[0-9]+([.][0-9]+)?$ && -n "${HICACHE_RATIO//[.0]/}" ]] || {
    echo "HICACHE_RATIO must be a positive finite decimal; unset it to disable HiCache" >&2; exit 2;
  }
fi

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
  --dsv4-c4-cache-implementation "$CACHE_IMPLEMENTATION"
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

if [ -n "${STORE_IMPLEMENTATION:-}" ]; then
  args+=(--dsv4-c4-cache-store-implementation "$STORE_IMPLEMENTATION")
fi
if [ -n "${C4_KEEP_COUNT:-}" ]; then
  args+=(--dsv4-c4-cache-keep-count "$C4_KEEP_COUNT")
fi
if [ -n "${C4_LAYER_KEEP_COUNTS:-}" ]; then
  args+=(--dsv4-c4-cache-layer-keep-counts "$C4_LAYER_KEEP_COUNTS")
fi
if [ -n "${HICACHE_RATIO:-}" ]; then
  # CPU L2 only. Keep layout and transfer policy explicit across both formats.
  args+=(--enable-hierarchical-cache --hicache-ratio "$HICACHE_RATIO"
    --hicache-write-policy write_through --hicache-io-backend kernel
    --hicache-mem-layout layer_first)
fi

if [ -n "${SERVER_SEED:-}" ]; then
  args+=(--random-seed "$SERVER_SEED")
fi
if [ -n "${REQUEST_METRICS_DIR:-}" ]; then
  [ -d "$REQUEST_METRICS_DIR" ] || { echo "request metrics directory does not exist: $REQUEST_METRICS_DIR" >&2; exit 2; }
  args+=(--enable-metrics --log-requests --log-requests-level 0
    --export-metrics-to-file --export-metrics-to-file-dir "$REQUEST_METRICS_DIR")
fi

if [ -n "${REMNANT_CAPTURE_DIR:-}" ]; then
  [ "$MODE" = native ] || { echo "capture requires native mode" >&2; exit 2; }
  [ -z "${SERVER_SEED:-}" ] || [ "$SERVER_SEED" = 42 ] || { echo "native fixture capture requires SERVER_SEED=42" >&2; exit 2; }
  args+=(--disable-radix-cache --disable-overlap-schedule)
  [ -n "${SERVER_SEED:-}" ] || args+=(--random-seed 42)
fi

# Nsight attaches at launch but collects nothing until /start_profile asks
# SGLang for CUDA_PROFILER. Do not enable startup profiling: loading weights
# and creating CUDA graphs must finish before the capture range starts.
if [ -n "${NSYS_OUTPUT:-}" ]; then
  command -v nsys >/dev/null 2>&1 || { echo "Nsight Systems (nsys) is required" >&2; exit 127; }
  NSYS_GRAPH_TRACE=${NSYS_GRAPH_TRACE:-node}
  case "$NSYS_GRAPH_TRACE" in
    node|graph) ;;
    *) echo "NSYS_GRAPH_TRACE must be node or graph" >&2; exit 2 ;;
  esac
  # These must be set before SGLang imports its profiler helpers.
  export SGLANG_ENABLE_NVTX_SCHEDULER=1 SGLANG_ENABLE_NVTX_OPERATIONS=1
  python3 -c 'from sglang.srt.utils.nvtx_utils import batch_profile_range, validate_nvtx_backend; validate_nvtx_backend()' || {
    echo "Nsight requires built-in torch.cuda.nvtx and the reviewed batch profiling hooks in the active SGLang source" >&2
    exit 1
  }
  echo "Nsight capture inactive until CUDA_PROFILER start; output: $NSYS_OUTPUT"
  exec nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --cuda-graph-trace="$NSYS_GRAPH_TRACE" \
    --capture-range=cudaProfilerApi --capture-range-end=repeat:999:async \
    --kill=none --output="$NSYS_OUTPUT" sglang "${args[@]}"
fi

exec sglang "${args[@]}"
