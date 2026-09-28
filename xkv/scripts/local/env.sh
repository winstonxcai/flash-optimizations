#!/usr/bin/env bash
# =====================================================================
# env.sh -- shared config + tiny helpers for the xKV driver scripts
#           (container.sh, serve.sh, bench-serving.sh)
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
#
# This box is DEDICATED TO xKV and is deliberately separate from `remnant`
# (the mustafar study container, which also runs on this host at PORT 30212 /
# MASTER 29638 and GPUs 0-3). Run only one of the two at a time, or give the
# second one a distinct GPU set / PORT / MASTER_PORT.
CONTAINER=${CONTAINER:-xkv}
HOST_REPO=${HOST_REPO:-/home/jovyan/winstonxcai/flash-optimizations}
REPO_CT=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations   # repo, as seen in $CONTAINER
MODEL_CT=${MODEL_CT:-/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731}
# Two serving source trees live side-by-side in the container. STOCK is the
# pristine, byte-identical sglang (the image's own /sgl-workspace/sglang);
# FORK is the xkv-patched clone at /sgl-workspace/sglang-lowrank created at
# runtime by container.sh. `native` serves STOCK; `xkv` serves FORK.
# SGLANG_PY is the resolved active python (used for bench clients; either tree
# imports sglang.bench_serving).
SGLANG_PY_STOCK=${SGLANG_PY_STOCK:-/sgl-workspace/sglang/python}
SGLANG_PY_FORK=${SGLANG_PY_FORK:-/sgl-workspace/sglang-lowrank/python}
SGLANG_PY=${SGLANG_PY:-$SGLANG_PY_FORK}

RESULTS_HOST=${RESULTS_HOST:-$HOST_REPO/xkv/results}
LOG_HOST=${LOG_HOST:-$HOST_REPO/xkv/logs}
# -----------------------------------------------------------------------------

# Run defaults (override before sourcing / on the command line).
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-30213}
MASTER_PORT=${MASTER_PORT:-29639}
TP=${TP:-4}
MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
MEM_FRAC=${MEM_FRAC:-0.88}
CTX_LEN=${CTX_LEN:-1048576}
MAX_RUN=${MAX_RUN:-256}
CHUNK=${CHUNK:-8192}
OUTLEN=${OUTLEN:-2048}     # bench_serving output length
SEED=${SEED:-42}

# ShareGPT json that `bench_serving --dataset-name random` samples token ids
# from. sglang >= v0.5.18 hard-requires it and fetches it from the Hub when
# --dataset-path is empty -- but that repo is gone upstream (401/404), so pass
# this local copy explicitly or the bench client dies before sending a request.
# Same bytes mustafar's serving protocol uses (md5 8d2f1dcd...).
SHAREGPT_HOST=${SHAREGPT_HOST:-/home/jovyan/zongyi/vllm-bench/ShareGPT_V3_unfiltered_cleaned_split.json}
SHAREGPT_CT=${SHAREGPT_CT:-/mnt/host_root$SHAREGPT_HOST}

# Decode CUDA-graph configs (prefill graphs stay off). DECODE_CFG_EXT covers the
# 64k fair point at C=32 and the xkv leg's wider allocator ceiling; DECODE_CFG_OFF
# is the eager control arm of the graph A/B. DECODE_CFG default = EXT, since the
# whole point of this study is decode-on-graph.
DECODE_CFG_EXT='{"decode":{"backend":"full","max_bs":136,"bs":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,24,28,32,34,40,48,56,64,68,80,96,112,120,136]},"prefill":{"backend":"disabled"}}'
DECODE_CFG_OFF='{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}'
DECODE_CFG=${DECODE_CFG:-$DECODE_CFG_EXT}

# xKV store env for the `xkv` leg: the low-rank store on, fused triton recon,
# rank-192 coefficients, and the calibrated basis directory (A_<layer>.pt).
LOWRANK_ENVS=(SGLANG_OPT_LOWRANK_KV_STORE=1 XKV_RECON_TRITON=1 XKV_COEFF_DIM=192)
BASIS_HOST=${BASIS_HOST:-$HOST_REPO/xkv/ctrl/basis}   # 512x512 per-CSA-layer second moments

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
# entrypoints.
kill_port () {
  ct "pkill -9 -f 'sglang(\.launch_server| serve).*--port $PORT'" 2>/dev/null
  sleep 4
}

# Boot markers we care about, printed from a host launch log.
boot_markers () {  # $1=host launch log
  grep -aoE "logical_row_bytes=[0-9]+ layers=[0-9]+|max_total_num_tokens=[0-9]+|Dequantized FP4|is fired up and ready|cuda graph" "$1" | head -20
}

pool_of () {  # $1=host launch log -> max_total_num_tokens ("" if not found yet)
  grep -aoE "max_total_num_tokens=[0-9]+" "$1" | head -1 | grep -oE "[0-9]+"
}
