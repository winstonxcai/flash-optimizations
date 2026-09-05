#!/usr/bin/env bash
# Shared env + helpers for the local (Modal-free) Mustafar Stage-1 driver
# scripts in this directory.
#
# Everything runs on the local H100 box inside the `ruler-eval` SGLang
# container (host python has no torch/sglang). Host paths are mirrored into
# the container under /mnt/host_root, so host $HOST_* vars pair with the
# container-visible $*_CT vars.
#
# Override any default before sourcing, e.g.:
#   PORT=30212 GPUS=4,5,6,7 . "$(dirname "$0")/common.sh"
set -u

# ---- layout ----
HOST_REPO=${HOST_REPO:-/home/jovyan/winstonxcai/flash-optimizations}
HOST_MUSTAFAR="$HOST_REPO/mustafar"
REPO_CT=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations   # container view of repo root

# ---- env defaults (override before sourcing) ----
PORT=${PORT:-30211}
GPUS=${GPUS:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29622}
KEEP=${XKV_TOPMAG_KEEP:-0.5}
MODEL_HOST=/mnt/public_data/sgl-project/DeepSeek-V4-Flash-FP8
MODEL=${MODEL:-/mnt/host_root$MODEL_HOST}          # container-visible model path
PYTHONPATH=/sgl-workspace/sglang-lowrank/python:$REPO_CT

RESULTS_HOST=${RESULTS_HOST:-$HOST_MUSTAFAR/results}
LOG_HOST=${LOG_HOST:-$HOST_MUSTAFAR/logs}
RESULTS_CT=/mnt/host_root$RESULTS_HOST             # container-visible results dir
mkdir -p "$RESULTS_HOST" "$LOG_HOST"

# ---- run a command in ruler-eval with the TopMag env ----
#   $* = command line (bash -c string)
rulerexec() {
  docker exec ruler-eval bash -c "
    export CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$MASTER_PORT
    export SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=$KEEP
    export PYTHONPATH=$PYTHONPATH
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
    $*"
}

# ---- kill any sglang server listening on a port (host pid space) ----
kill_server() {
  ps -eo pid,args | grep '[s]glang.launch_server' | grep " $1 " | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 2
}

# ---- wait until the server log reports readiness / KV-pool allocation ----
wait_ready() {  # $1=host log, $2=poll count (5s each)
  local log=$1 n=${2:-120} i
  for i in $(seq 1 "$n"); do
    grep -q 'is fired up and ready' "$log" 2>/dev/null && { echo "server ready (${i}x5s)"; return 0; }
    sleep 5
  done
  echo "server NOT ready after $((n*5))s" >&2; tail -20 "$log" >&2; return 1
}

wait_pool() {  # $1=host log, $2=poll count (5s each)
  local log=$1 n=${2:-240} i
  for i in $(seq 1 "$n"); do
    if grep -q 'max_total_num_tokens=' "$log" 2>/dev/null && grep -q 'Memory pool end.' "$log" 2>/dev/null; then
      echo "pools allocated (${i}x5s)"; return 0
    fi
    sleep 5
  done
  echo "pool allocation NOT seen after $((n*5))s" >&2; tail -20 "$log" >&2; return 1
}
