#!/usr/bin/env bash
# =====================================================================
# env.sh -- shared config + tiny helpers for the starkv driver scripts
#           (container.sh, serve.sh, capture.sh, stability.sh)
#
# TO PORT TO A NEW MACHINE/GPU NODE: edit ONLY the "MACHINE CONFIG"
# block below. Everything else is generic. All drivers source this file:
#   . "$(dirname "$0")/env.sh"
# =====================================================================
set -u

# ------------------------------ MACHINE CONFIG -------------------------------
# GPU work on this host is scheduled by GPUQ, and GPUQ's rules shape this file:
#
#   * No container is ever launched with `--gpus all`. A GPU leg runs in a
#     throwaway container pinned to exactly the devices the scheduler granted
#     (`--gpus "\"device=$CUDA_VISIBLE_DEVICES\""` -- the inner quotes are
#     load-bearing, see the note in serve.sh), launched from a `gpuq run` job.
#   * CUDA_VISIBLE_DEVICES is never set by hand for a GPU launch. GPUQ sets it
#     after acquiring the allocation, and `require_allocation` below refuses to
#     proceed without it.
#   * The long-lived `starkv` container is a CPU-only control box -- it is
#     created without `--gpus`, so a stray process inside it cannot reach a GPU
#     at all. It exists for `selftest` (which needs torch) and for the pristine
#     tree census.
#
# That is why the fork tree lives on the HOST rather than inside a container:
# a throwaway container is created and destroyed per run, so anything it needs
# to see has to be on a bind mount. /home is mounted identically in every
# container (`-v /home:/home`), so a host path is the same path inside -- no
# /mnt/host_root translation for the fork.
CONTAINER=${CONTAINER:-starkv}                  # CPU-only control container
RUN_CONTAINER=${RUN_CONTAINER:-starkv-run}      # throwaway GPU container
IMAGE=${IMAGE:-starkv:v0.5.18}
GPUQ_PROJECT=${GPUQ_PROJECT:-starkv}
HOST_REPO=${HOST_REPO:-/home/jovyan/winstonxcai/flash-optimizations}
REPO_CT=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations   # repo, as seen in a container
MODEL_CT=${MODEL_CT:-/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731}

# The fork tree: cloned, patched and censused on the HOST, then bind-mounted
# read-only into whichever container serves it. Host path == container path.
FORK_HOST=${FORK_HOST:-/home/jovyan/winstonxcai/sglang-starkv}
SGLANG_PY_FORK=$FORK_HOST/python
# The pristine tree ships in the image, so every container has it at this path.
# Only the control container is ever checked that it is still unpatched.
SGLANG_PY_NATIVE=/sgl-workspace/sglang/python

RESULTS_HOST=${RESULTS_HOST:-$HOST_REPO/starkv/results}
CTRL_HOST=${CTRL_HOST:-$HOST_REPO/starkv/ctrl}
CAPTURES_HOST=${CAPTURES_HOST:-$HOST_REPO/starkv/captures}
BASIS_HOST=${BASIS_HOST:-$HOST_REPO/starkv/ctrl/basis}   # D_<layer>_r<rank>.pt

# Job artifacts the HOST-SIDE payload writes. A `gpuq run` job executes as the
# unprivileged gpuq-runner account, which cannot write into this repo, so every
# file a job produces on the host lands under the submitting user's designated
# results directory instead -- including the server boot log, which is written
# by serve.sh from inside the job. The container-side outputs (captures) are
# written through the bind mount as root and go straight to $CAPTURES_HOST.
GPUQ_OUT=${GPUQ_OUT:-/home/jovyan/gpuq-results/starkv}
LOG_HOST=${LOG_HOST:-$GPUQ_OUT/logs}

# LongSWE-Bench replay client (business_replay) -- lives on THIS host.
REPLAY_DIR=${REPLAY_DIR:-/home/jovyan/wenyuhong/benchmarks}
REPLAY_RUNNER=$REPLAY_DIR/harnesses/business_replay/runner.py
REPLAY_ADAPTER=$REPLAY_DIR/harnesses/business_replay/adapters/openai_sse.py
REPLAY_DATASET=$REPLAY_DIR/datasets/h20-dsv4pro/longcodebench_openai
REPLAY_MANIFEST=$REPLAY_DIR/cache/official-longswebench/longswebench-openai-v1.json
REPLAY_REQUESTS=${REPLAY_REQUESTS:-4916}
# -----------------------------------------------------------------------------

# Run defaults (override before sourcing / on the command line).
#
# Note there is no GPUS variable any more. A GPU leg takes its device list from
# CUDA_VISIBLE_DEVICES, which GPUQ sets when it grants the allocation, and
# `require_allocation` refuses to boot without one. A hand-set device list would
# be indistinguishable from a guess about what is free.
PORT=${PORT:-30213}
MASTER_PORT=${MASTER_PORT:-29648}
TP=${TP:-4}
MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
MEM_FRAC=${MEM_FRAC:-0.88}
CTX_LEN=${CTX_LEN:-1048576}
MAX_RUN=${MAX_RUN:-256}
CHUNK=${CHUNK:-8192}
SEED=${SEED:-42}

# Decode CUDA-graph configs. The measurement legs all run EAGER: the capture
# hooks have to fire on every decode step, and a captured graph replays its
# kernels without re-entering python, so a graphed capture leg would record one
# step and then nothing. DECODE_CFG_GRAPH is the throughput arm for when the
# study moves on from measurement to serving comparison.
DECODE_CFG_OFF='{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}'
DECODE_CFG_GRAPH='{"decode":{"backend":"full","max_bs":136,"bs":[1,2,4,8,16,24,32,48,64,80,96,112,136]},"prefill":{"backend":"disabled"}}'
DECODE_CFG=${DECODE_CFG:-$DECODE_CFG_OFF}

# The reconstruct MVE's record rank. Must match the rank the basis files were
# fitted at, or `basis_for` finds no file and the store declines.
STARKV_RANK=${STARKV_RANK:-320}

# Only $GPUQ_OUT is created here: the repo dirs are made by the operator
# (container.sh), and a job account has no business creating them.
mkdir -p "$GPUQ_OUT" "$LOG_HOST"

ts () { date +%Y%m%d_%H%M%S; }

# Container-visible path of a HOST path. Every container here is launched with
# `-v /:/mnt/host_root`, so the mapping is a flat prefix over ANY absolute host
# path -- not just paths under $HOST_REPO. (It used to substitute $REPO_CT for a
# $HOST_REPO prefix, which silently mangled any path outside the repo: a log
# under $GPUQ_OUT came out as $REPO_CT/$GPUQ_OUT, the container's redirect
# failed, and the leg died at launch with an empty boot log.)
HOST_ROOT_CT=/mnt/host_root
to_ct () { echo "$HOST_ROOT_CT$1"; }

# Run one shell command in the CPU-only control container (prep, patch, drift,
# selftest). Never used for anything that could touch a GPU: the container is
# created without `--gpus`, so it has no device nodes to reach.
ct () { docker exec "$CONTAINER" bash -c "$*"; }

# ------------------------------ server helpers ------------------------------
# A GPU leg is a throwaway container pinned to the devices GPUQ granted. The
# launch log is written to a host path so it can be grepped from here.

require_allocation () {
  # GPUQ exports CUDA_VISIBLE_DEVICES after it acquires the allocation. There is
  # deliberately no fallback to GPUS or to "the free cards": on a shared host,
  # guessing is how one study lands on another's devices.
  DEVICES=${CUDA_VISIBLE_DEVICES:-}
  [ -n "$DEVICES" ] || {
    echo "FATAL: no GPU allocation. This must run as a 'gpuq run' job (which" >&2
    echo "       sets CUDA_VISIBLE_DEVICES), or with DEVICES exported from a lease." >&2
    exit 1
  }
  # A device list, not a count and not "all".
  case "$DEVICES" in
    all|*[!0-9,]*) echo "FATAL: CUDA_VISIBLE_DEVICES='$DEVICES' is not a device list" >&2; exit 1 ;;
  esac
  echo "== allocation: gpuq job ${GPUQ_JOB_ID:-<none>} project ${GPUQ_PROJECT:-?} devices $DEVICES"
}

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

# Boot markers we care about, printed from a host launch log.
boot_markers () {  # $1=host launch log
  grep -aoE "max_total_num_tokens=[0-9]+|is fired up and ready|cuda graph" "$1" | head -20
}

pool_of () {  # $1=host launch log -> max_total_num_tokens ("" if not found yet)
  grep -aoE "max_total_num_tokens=[0-9]+" "$1" | head -1 | grep -oE "[0-9]+"
}
