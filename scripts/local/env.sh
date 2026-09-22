#!/usr/bin/env bash
# =====================================================================
# env.sh -- shared runtime configuration and helpers for local driver scripts
#
# TO PORT TO A NEW MACHINE/GPU NODE: override the MACHINE CONFIG values below
# (or export them before invoking a script). All drivers source this file:
#   . "$(dirname "$0")/env.sh"
# =====================================================================
set -u

# ------------------------------ MACHINE CONFIG -------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HOST_REPO=${HOST_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
CONTAINER=${CONTAINER:-remnant}

# The Dockerfile is the source of truth for serving. It installs the reviewed
# production fork at this path; neither native nor packed serving imports the
# repository's old third_party/sglang checkout anymore.
REPO_CT=${REPO_CT:-/mnt/host_root$HOST_REPO}
SGLANG_ROOT_CT=${SGLANG_ROOT_CT:-/sgl-workspace/sglang-remnant}
SGLANG_PY=${SGLANG_PY:-$SGLANG_ROOT_CT/python}
SGLANG_ROOT=${SGLANG_ROOT:-$SGLANG_ROOT_CT}
MODEL_PATH=${MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731}
MODEL_CT=${MODEL_CT:-/mnt/host_root$MODEL_PATH}

RESULTS_HOST=${RESULTS_HOST:-$HOST_REPO/results}
LOG_HOST=${LOG_HOST:-$HOST_REPO/logs}

# -----------------------------------------------------------------------------

# Run defaults (override before sourcing / on the command line).
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-30212}
MASTER_PORT=${MASTER_PORT:-29638}
TP=${TP:-4}
MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
MEM_FRAC=${MEM_FRAC:-0.88}
CTX_LEN=${CTX_LEN:-1048576}
MAX_RUN=${MAX_RUN:-256}
CHUNK=${CHUNK:-8192}
OUTLEN=${OUTLEN:-2048}     # bench_serving output length
SEED=${SEED:-42}

# Decode CUDA-graph configs live in config/*.json so every launcher reads the
# same definitions. DECODE_CFG defaults to the small agentic configuration;
# serving capacity drivers explicitly select the extended configuration.
CONFIG_DIR=${CONFIG_DIR:-$SCRIPT_DIR/config}
DECODE_CFG_SMALL=${DECODE_CFG_SMALL:-$(<"$CONFIG_DIR/decode-graphs-small.json")}
DECODE_CFG_EXT=${DECODE_CFG_EXT:-$(<"$CONFIG_DIR/decode-graphs-extended.json")}
DECODE_CFG=${DECODE_CFG:-$DECODE_CFG_SMALL}

ts () { date +%Y%m%d_%H%M%S; }

# Container-visible path of a host path. The local container mounts / at
# /mnt/host_root, while Modal runs the same scripts directly and does not use
# this helper.
to_ct () { echo "$REPO_CT${1#$HOST_REPO}"; }

# Run one shell command inside the sglang container.
#   ct <cmd...>          -> docker exec $CONTAINER bash -c "<cmd>"
ct () { docker exec "$CONTAINER" bash -c "$*"; }

# ------------------------------ server helpers ------------------------------
# These manage a server that serve.sh brought up inside $CONTAINER. The launch
# log is written host-side so it can be grepped here.

health () { curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

wait_health () {  # [$1=poll cap in 5s steps]
  local n=${1:-180} i
  for i in $(seq 1 "$n"); do
    health && { echo "  health OK after ~$((i * 5))s"; return 0; }
    sleep 5
  done
  echo "  health TIMEOUT after ~$((n * 5))s" >&2
  return 1
}

# Kill any production-fork server inside the container on $PORT.
kill_port () {
  ct "pkill -9 -f 'sglang(\.launch_server| serve).*--port $PORT'" 2>/dev/null
  sleep 4
}

# Boot markers we care about, printed from a host launch log.
boot_markers () {  # $1=host launch log
  grep -aoE "logical_row_bytes=[0-9]+ layers=[0-9]+|max_total_num_tokens=[0-9]+|Dequantized FP4|is fired up and ready" "$1" | head -20
}

pool_of () {  # $1=host launch log -> max_total_num_tokens ("" if not found yet)
  grep -aoE "max_total_num_tokens=[0-9]+" "$1" | head -1 | grep -oE "[0-9]+"
}
