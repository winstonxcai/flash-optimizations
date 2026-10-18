#!/usr/bin/env bash
# =====================================================================
# serve.sh -- boot (or stop) a production-fork DeepSeek-V4-Flash-0731 TP server
# on the local GPU node, inside the reproducible SGLang container.
# Author: Winston Cai.
#
#   serve.sh native             untouched 0731, stock 584-byte C4 (default)
#   serve.sh triton             Packed C4 with Triton packing/reconstruction
#   serve.sh kernel             Packed C4 with current optimized dispatch
#   serve.sh packed             compatibility alias for kernel
#   serve.sh <mode> profile     Nsight Systems, capture waits for profiler API
#   serve.sh <mode> stop        kill the server on $PORT
#
# All modes use the same production fork.
# They use the fp4-native MoE runner (flashinfer_mxfp4), mem-frac 0.88,
# 1M ctx cap, fp8 KV, and DeepSeek reasoning/tool parsers (needed by the
# agentic evals; harmless for benches). Native uses the fork's default cache
# format; triton/kernel default to 328-byte Packed records. Triton also supports
# startup per-layer survivor counts with smaller/larger value planes.
#
# Env overrides (all optional): PORT, GPUS, MASTER_PORT, TP, DECODE_CFG,
# MEM_FRAC, CTX_LEN, MAX_RUN, CHUNK, SERVER_SEED, REQUEST_METRICS_DIR,
# C4_KEEP_COUNT, C4_LAYER_KEEP_COUNTS. Boot log:
#   <LOG_HOST>/serve_<native|triton|kernel>.log
# Server is left RUNNING; use "serve.sh <mode> stop" to tear it down.
# =====================================================================
set -u
DECODE_CFG_WAS_SET=${DECODE_CFG+x}
. "$(dirname "$0")/env.sh"
# Standalone serving must cover replay c24/c28, like the capacity drivers.
# Keep an explicit caller override for smaller agentic workloads.
[ -n "$DECODE_CFG_WAS_SET" ] || DECODE_CFG="$DECODE_CFG_EXT"

MODE=${1:-}; ACTION=${2:-boot}
case "$MODE" in
  native|triton|kernel|packed) ;;
  *) echo "usage: $0 <native|triton|kernel> [boot|profile|stop] (packed aliases kernel)"; exit 1 ;;
esac
case "$ACTION" in
  boot|profile|stop) ;;
  *) echo "usage: $0 <native|triton|kernel> [boot|profile|stop]"; exit 1 ;;
esac
SERVE_LOG="$LOG_HOST/serve_${MODE}.log"  # host-side log path
SERVE_LOG_CT=$(to_ct "$SERVE_LOG")             # same file inside container
mkdir -p "$LOG_HOST"

if [ "$ACTION" = stop ]; then
  if ! kill_port; then
    echo "FATAL: could not verify or stop the server process record for port $PORT" >&2
    exit 1
  fi
  if health; then
    echo "FATAL: server still responds on port $PORT after stop attempt" >&2
    exit 1
  fi
  echo "stopped $MODE server on port $PORT"
  exit 0
fi

case "${STORE_IMPLEMENTATION:-}" in
  ""|triton|kernel) ;;
  *) echo "STORE_IMPLEMENTATION must be triton or kernel" >&2; exit 2 ;;
esac

# Resolve the same model-wide profile as the server before replacing a healthy
# process. Configuration reads only; no model weights or GPU work are loaded.
if [ -n "${C4_KEEP_COUNT:-}" ] || [ -n "${C4_LAYER_KEEP_COUNTS:-}" ]; then
  docker exec -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT" "$CONTAINER" python3 -c '
import sys
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.utils.hf_transformers_utils import get_config
from sglang.srt.remnant import config
for field in ("dsv4_c4_cache_keep_count", "dsv4_c4_cache_layer_keep_counts"):
    assert field in ServerArgs.__dataclass_fields__, "mounted source lacks per-layer Packed counts"
model, mode, count, overrides, store = sys.argv[1:]
# Match ModelConfig: its loader applies the registered SGLang model schema.
# Transformers AutoConfig alone can omit the V4 compression metadata.
hf = get_config(model, trust_remote_code=True)
if mode != "native" and not is_deepseek_v4(hf):
    raise ValueError("Packed layer counts require a DeepSeek V4 model")
config.configure_cache_format(
    "native" if mode == "native" else "remnant",
    "triton" if mode == "triton" else "kernel", store or None,
    keep_count=int(count) if count else 256, layer_keep_counts=overrides or None,
    compression_ratios=getattr(hf, "compress_ratios", None),
)
if mode != "native" and not hasattr(hf, "compress_ratios"):
    raise ValueError("Packed layer counts require model compress_ratios")
print("Packed C4 profile:", config.packed_profile())
' "$MODEL_CT" "$MODE" "${C4_KEEP_COUNT:-}" "${C4_LAYER_KEEP_COUNTS:-}" "${STORE_IMPLEMENTATION:-}" || exit 1
fi

# Probe the actual mounted source before stopping an existing server.
if [ -n "${STORE_IMPLEMENTATION:-}" ]; then
  docker exec -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT" "$CONTAINER" python3 -c '
from sglang.srt.server_args import ServerArgs
assert "dsv4_c4_cache_store_implementation" in ServerArgs.__dataclass_fields__, "active source lacks store override; select updated host Python source"
' || exit 1
fi

# Diagnostic exports are opt-in and belong to one boot. Probe the mounted
# Python source before replacing a healthy server; no model is loaded here.
REQUEST_METRICS_DIR_CT=
if [ -n "${SERVER_SEED:-}" ]; then
  [[ "$SERVER_SEED" =~ ^[0-9]{1,10}$ ]] && (( 10#$SERVER_SEED <= 2147483647 )) || {
    echo "SERVER_SEED must be an integer in [0, 2147483647]" >&2; exit 2;
  }
fi
if [ -n "${REQUEST_METRICS_DIR:-}" ]; then
  case "$REQUEST_METRICS_DIR" in
    "$HOST_REPO"/*) ;;
    *) echo "REQUEST_METRICS_DIR must be an absolute directory under $HOST_REPO" >&2; exit 2 ;;
  esac
  [ ! -e "$REQUEST_METRICS_DIR" ] || { echo "request metrics directory already exists; choose a fresh boot directory: $REQUEST_METRICS_DIR" >&2; exit 2; }
  docker exec -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT" "$CONTAINER" python3 -c '
import inspect
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
stats = APIServerReqTimeStats()
stats.first_token_time = 1.0
assert "first_output_ts" in stats.convert_to_output_meta_info(), "active SGLang source lacks first_output_ts; pull/select the updated Python fork"
print("Request timing source:", inspect.getsourcefile(APIServerReqTimeStats))
' || exit 1
  mkdir -p -- "$(dirname -- "$REQUEST_METRICS_DIR")" || exit 1
  mkdir -- "$REQUEST_METRICS_DIR" || exit 1
  REQUEST_METRICS_DIR_CT=$(to_ct "$REQUEST_METRICS_DIR")
fi

# Validate profiling before replacing a running server. The output directory
# is unique and host-mounted, so reports survive container recreation.
NSYS_OUTPUT_CT=
if [ "$ACTION" = profile ]; then
  NSYS_GRAPH_TRACE=${NSYS_GRAPH_TRACE:-node}
  case "$NSYS_GRAPH_TRACE" in
    node|graph) ;;
    *) echo "NSYS_GRAPH_TRACE must be node or graph" >&2; exit 2 ;;
  esac
  docker exec "$CONTAINER" nsys --version || exit $?
  docker exec -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT" "$CONTAINER" python3 -c \
    'import inspect; from sglang.srt.utils.nvtx_utils import batch_profile_range, validate_nvtx_backend; validate_nvtx_backend(); print("Batch profiling source:", inspect.getsourcefile(batch_profile_range))' || {
      echo "profile requires built-in torch.cuda.nvtx and updated SGLang hooks; select the reviewed fork source" >&2
      exit 1
    }
  profile_dir=$(mktemp -d "$LOG_HOST/nsys-${MODE}-$(ts)-XXXXXX") || exit 1
  NSYS_OUTPUT_CT=$(to_ct "$profile_dir/server")
  echo "Nsight capture waits for CUDA profiler API; reports: $profile_dir"
fi

if ! kill_port; then
  echo "FATAL: could not verify/stop the previous server record for port $PORT; refusing to launch another" >&2
  exit 1
fi

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
  "$MODE" "$MODEL_CT" "$PORT" "$SERVE_LOG_CT" "$SERVER_PID_FILE_CT" "$NSYS_OUTPUT_CT" "${NSYS_GRAPH_TRACE:-node}" \
  "${SERVER_SEED:-}" "$REQUEST_METRICS_DIR_CT" "${STORE_IMPLEMENTATION:-}" "${C4_KEEP_COUNT:-}" "${C4_LAYER_KEEP_COUNTS:-}" <<'BASH'
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
nsys_output=${18}
nsys_graph_trace=${19}
server_seed=${20}
request_metrics_dir=${21}
store_implementation=${22}
c4_keep_count=${23}
c4_layer_keep_counts=${24}

cd -- "$tree"
export CUDA_VISIBLE_DEVICES="$gpus" MASTER_PORT="$master_port"
export MODEL_NAME="$model_name" TP="$tp" MEM_FRAC="$mem_frac" CTX_LEN="$ctx_len"
export MAX_RUN="$max_run" CHUNK="$chunk" DECODE_CFG="$decode_cfg"
export SERVER_PID_FILE="$pid_file" NSYS_OUTPUT="$nsys_output"
export NSYS_GRAPH_TRACE="$nsys_graph_trace"
export SERVER_SEED="$server_seed" REQUEST_METRICS_DIR="$request_metrics_dir"
export STORE_IMPLEMENTATION="$store_implementation"
export C4_KEEP_COUNT="$c4_keep_count" C4_LAYER_KEEP_COUNTS="$c4_layer_keep_counts"
export PYTHONPATH="/opt/sglang-runtime-fixes:$sglang_py:$repo_ct"
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
# All TP workers are local; keep Gloo off hostname-selected external interfaces.
export GLOO_SOCKET_IFNAME=lo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
command -v setsid >/dev/null 2>&1 || { echo "setsid is required" >&2; exit 127; }
nohup setsid bash "$repo_ct/scripts/local/run-server.sh" "$mode" "$model_ct" 0.0.0.0 "$port" \
  > "$serve_log_ct" 2>&1 &
pid=$!
echo "launched pid $pid"
BASH
launch_status=$?
if [ "$launch_status" -ne 0 ]; then
  echo "FATAL: container launch failed ($launch_status)" >&2
  exit "$launch_status"
fi

# Cold starts compile MHC and DeepGEMM kernels before graph creation.
# Allow 40 minutes for this initialization, with cleanup on timeout.
if ! wait_health 480; then
  tail -40 "$SERVE_LOG"
  if ! kill_port; then
    echo "FATAL: could not verify or terminate the timed-out server process group" >&2
  fi
  if health; then
    echo "FATAL: server still responds on port $PORT after startup cleanup" >&2
  fi
  exit 1
fi
sleep 3
POOL=$(pool_of "$SERVE_LOG")
echo "  pool(max_total_num_tokens)=${POOL:-UNKNOWN}"
boot_markers "$SERVE_LOG" | sed 's/^/  /'
echo "serve $MODE UP on port $PORT -- leave running, tear down with: $0 $MODE stop"
