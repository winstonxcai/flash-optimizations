#!/usr/bin/env bash
# =====================================================================
# env.sh -- shared config + tiny helpers for the mustafar driver scripts
#           (container.sh, serve.sh, bench-serving.sh, bench-lswb.sh,
#            eval-lb2.sh, agentic-eval.sh, hicache-ladder.sh, lswb-row.sh)
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
CONTAINER=${CONTAINER:-remnant}                          # sglang container name (v0.5.18; anchors are remnant-only)
HOST_REPO=${HOST_REPO:-/home/jovyan/winstonxcai/flash-optimizations}
REPO_CT=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations   # repo, as seen in $CONTAINER
MODEL_CT=${MODEL_CT:-/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731}
# Two serving source trees live side-by-side in the container. STOCK is the
# pristine, byte-identical sglang (the image's own /sgl-workspace/sglang);
# FORK is the mustafar-patched clone at /sgl-workspace/sglang-lowrank created
# at runtime by container.sh. `native` serves STOCK; `packed` serves FORK.
# SGLANG_PY is the resolved active python (used for bench clients; either tree
# imports sglang.bench_serving). remnant (the active study container) ships
# both at v0.5.18; ruler-eval is the frozen v0.5.15 legacy box (also on 30212,
# GPUs 4-7 / MASTER 29628). Defaults below target remnant on 30212; don't run
# both containers at once -- give one a distinct PORT.
SGLANG_PY_STOCK=${SGLANG_PY_STOCK:-/sgl-workspace/sglang/python}
SGLANG_PY_FORK=${SGLANG_PY_FORK:-/sgl-workspace/sglang-lowrank/python}
SGLANG_PY=${SGLANG_PY:-$SGLANG_PY_FORK}

RESULTS_HOST=${RESULTS_HOST:-$HOST_REPO/mustafar/results}
LOG_HOST=${LOG_HOST:-$HOST_REPO/mustafar/logs}

# LongSWE-Bench replay client (business_replay) -- lives on THIS host.
REPLAY_DIR=${REPLAY_DIR:-/home/jovyan/wenyuhong/benchmarks}
REPLAY_RUNNER=$REPLAY_DIR/harnesses/business_replay/runner.py
REPLAY_ADAPTER=$REPLAY_DIR/harnesses/business_replay/adapters/openai_sse.py
REPLAY_DATASET=$REPLAY_DIR/datasets/h20-dsv4pro/longcodebench_openai
REPLAY_MANIFEST=$REPLAY_DIR/cache/official-longswebench/longswebench-openai-v1.json

# LongBench v2 dataset (host paths; the full 503-question set + per-id token counts).
LB2_DATA=${LB2_DATA:-$HOST_REPO/mustafar/data/lb2_data.json}
LB2_TOKENS=${LB2_TOKENS:-$HOST_REPO/mustafar/data/lb2_tokens.json}

# Remote agentic-eval box that runs the Sangfor / SWE-bench clients (they reach
# our local sglang server over http). A docker_env_config JSON on that box holds
# the env keys (experiment_env.ANTHROPIC_BASE_URL / ...MODEL) incl. the auth
# token -- we reference existing configs and never read/echo their contents.
EVAL_SSH=${EVAL_SSH:-"sshpass -p a ssh -o StrictHostKeyChecking=no root@10.57.3.76"}
EVAL_YJY=/data/zc/workplace_zhq/YJYBench
EVAL_VENV=$EVAL_YJY/.venv/bin/python
EVAL_CFG=${EVAL_CFG:-$EVAL_YJY/test_env/docker_env_config_dsv4_0731.json}  # base-url points at the canonical serve port ($PORT; remnant 30212 default)
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

# Decode CUDA-graph configs (prefill graphs stay off). DECODE_CFG default =
# SMALL (agentic-eval concurrency, and SLO legs with C<=15); the serving/bench
# drivers and C>15 legs use EXT so decode stays on-graph up to the packed
# allocator ceiling (~129). bench-serving.sh's standalone path carries its own
# copy of EXT because it cannot source env.sh -- keep the two in sync.
DECODE_CFG_SMALL='{"decode":{"backend":"full","max_bs":15,"bs":[1,2,3,4,5,6,7,8,10,12,14,15]},"prefill":{"backend":"disabled"}}'
DECODE_CFG_EXT='{"decode":{"backend":"full","max_bs":136,"bs":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,24,28,32,34,40,48,56,64,68,80,96,112,120,136]},"prefill":{"backend":"disabled"}}'
DECODE_CFG=${DECODE_CFG:-$DECODE_CFG_SMALL}

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

# Kill any sglang server inside the container on $PORT: matches both the
# legacy `python3 -m sglang.launch_server` and the current `sglang serve`
# entrypoints (ruler-eval v0.5.15 still boots the former; remnant v0.5.18 the
# latter).
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
