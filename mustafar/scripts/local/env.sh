#!/usr/bin/env bash
# =====================================================================
# env.sh -- shared config + tiny helpers for the mustafar driver scripts
#           (serve.sh, bench-serving.sh, bench-fair.sh, bench-max.sh,
#            bench-lswb.sh, eval-lb2.sh, eval-sangfor.sh, eval-swe.sh)
#
# TO PORT TO A NEW MACHINE/GPU NODE: edit ONLY the "MACHINE CONFIG"
# block below. Everything else is generic. All drivers source this file:
#   . "$(dirname "$0")/env.sh"
# =====================================================================
set -u

# ------------------------------ MACHINE CONFIG -------------------------------
# SGLang runs inside a container on this host; host python has no torch/sglang,
# so model work happens via `docker exec`. Host paths mirror into the container
# under /mnt/host_root (hence the *_CT twins).
CONTAINER=${CONTAINER:-ruler-eval}                       # sglang container name
HOST_REPO=${HOST_REPO:-/home/jovyan/winstonxcai/flash-optimizations}
REPO_CT=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations   # repo, as seen in $CONTAINER
MODEL_CT=${MODEL_CT:-/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731}
SGLANG_PY=/sgl-workspace/sglang-lowrank/python           # mustafar-fork sglang source (in $CONTAINER)

RESULTS_HOST=${RESULTS_HOST:-$HOST_REPO/mustafar/results}
LOG_HOST=${LOG_HOST:-$HOST_REPO/mustafar/logs}

# LongSWE-Bench replay client (business_replay) -- lives on THIS host.
REPLAY_DIR=${REPLAY_DIR:-/home/jovyan/wenyuhong/benchmarks}
REPLAY_RUNNER=$REPLAY_DIR/harnesses/business_replay/runner.py
REPLAY_ADAPTER=$REPLAY_DIR/harnesses/business_replay/adapters/openai_sse.py
REPLAY_DATASET=$REPLAY_DIR/datasets/h20-dsv4pro/longcodebench_openai
REPLAY_MANIFEST=$REPLAY_DIR/cache/official-longswebench/longswebench-openai-v1.json

# LongBench v2 dataset (host paths; the full 503-question set + per-id token counts).
LB2_DATA=${LB2_DATA:-/home/jovyan/winstonxcai/transferibility/data/longbench/lb2_data.json}
LB2_TOKENS=${LB2_TOKENS:-/home/jovyan/winstonxcai/transferibility/data/longbench/lb2_tokens.json}

# Remote agentic-eval box that runs the Sangfor / SWE-bench clients (they reach
# our local sglang server over http). A docker_env_config JSON on that box holds
# the env keys (experiment_env.ANTHROPIC_BASE_URL / ...MODEL) incl. the auth
# token -- we reference existing configs and never read/echo their contents.
EVAL_SSH=${EVAL_SSH:-"sshpass -p a ssh -o StrictHostKeyChecking=no root@10.57.3.76"}
EVAL_YJY=/data/zc/workplace_zhq/YJYBench
EVAL_VENV=$EVAL_YJY/.venv/bin/python
EVAL_CFG=${EVAL_CFG:-$EVAL_YJY/test_env/docker_env_config_dsv4_0731.json}  # points at the canonical serve port on this host
# -----------------------------------------------------------------------------

# Run defaults (override before sourcing / on the command line).
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-30212}
MASTER_PORT=${MASTER_PORT:-29628}
TP=${TP:-4}
MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
MEM_FRAC=${MEM_FRAC:-0.88}
CTX_LEN=${CTX_LEN:-1048576}
MAX_RUN=${MAX_RUN:-256}
CHUNK=${CHUNK:-8192}
OUTLEN=${OUTLEN:-2048}     # bench_serving output length
SEED=${SEED:-42}

# Decode CUDA-graph config. Default = small (agentic-eval concurrency); the
# serving bench drivers override with an extended list so decode stays on-graph
# up to the packed allocator ceiling (~129). prefill graphs stay off.
DECODE_CFG=${DECODE_CFG:-'{"decode":{"backend":"full","max_bs":15,"bs":[1,2,3,4,5,6,7,8,10,12,14,15]},"prefill":{"backend":"disabled"}}'}

mkdir -p "$RESULTS_HOST" "$LOG_HOST"

ts () { date +%Y%m%d_%H%M%S; }

# Container-visible path of a HOST path under $HOST_REPO (this repo is mounted
# inside the container at $REPO_CT = /mnt/host_root/home/.../flash-optimizations).
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

# Kill any sglang.launch_server inside the container on $PORT.
kill_port () {
  ct "pkill -9 -f 'sglang.launch_server.*--port $PORT'" 2>/dev/null
  sleep 4
}

# Boot markers we care about, printed from a host launch log.
boot_markers () {  # $1=host launch log
  grep -aoE "logical_row_bytes=[0-9]+ layers=[0-9]+|max_total_num_tokens=[0-9]+|Dequantized FP4|is fired up and ready" "$1" | head -20
}

pool_of () {  # $1=host launch log -> max_total_num_tokens ("" if not found yet)
  grep -aoE "max_total_num_tokens=[0-9]+" "$1" | head -1 | grep -oE "[0-9]+"
}
