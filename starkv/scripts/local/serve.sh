#!/usr/bin/env bash
# =====================================================================
# serve.sh -- boot (or stop) a DeepSeek-V4-Flash-0731 TP server for the
# starkv study, in a container pinned to the GPUQ allocation.
#
#   serve.sh native               untouched 0731, stock 584-byte C4, no starkv
#   serve.sh starkv-recon         compact C4 record, rebuilt to 512-D on read
#   serve.sh capture <tag>        fork tree, store OFF, capture hooks writing
#                                 to captures/<tag>
#   serve.sh <mode> stop          stop the running leg
#
# This script is normally invoked by capture.sh or stability.sh, which run as
# `gpuq run` jobs and therefore already hold the allocation. It can also be run
# by hand, but only with CUDA_VISIBLE_DEVICES exported from a lease you hold --
# see require_allocation.
#
# Every mode uses the fp4-native MoE runner (flashinfer_mxfp4), mem-frac 0.88,
# 1M ctx cap, fp8 KV, and the DeepSeek reasoning/tool parsers. The legs differ
# by WHICH source tree serves and by the starkv envs: native serves the image's
# pristine tree (SGLANG_PY_NATIVE, byte-identical sglang); starkv-recon and
# capture serve the patched fork on the host (SGLANG_PY_FORK), with
# SGLANG_OPT_STARKV=1 for the recon leg. Capture serves the same fork with the
# store OFF so every recorded value is native.
#
# The capture leg MUST run eager (DECODE_CFG_OFF): the hooks are python calls on
# the decode path, and a captured CUDA graph replays its kernels without
# re-entering python, so a graphed capture records one step and then nothing.
#
# The container is THROWAWAY (`--rm`) and pinned with
# `--gpus "device=$CUDA_VISIBLE_DEVICES"`. It is never launched with `--gpus all`
# -- see the GPUQ note at the top of env.sh. The fork tree is bind-mounted from
# the host, which is why the same path resolves inside and out.
#
# Env overrides (all optional): PORT, MASTER_PORT, TP, DECODE_CFG, MEM_FRAC,
# CTX_LEN, MAX_RUN, CHUNK, BASIS_HOST, STARKV_RANK, STARKV_DEBUG. The device list
# is NOT an override -- it comes from CUDA_VISIBLE_DEVICES, set by GPUQ.
# Boot log: <LOG_HOST>/serve_<mode>.log
# The server is left RUNNING; use "serve.sh <mode> stop" to tear it down.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

MODE=${1:-}; ACTION=boot; TAG=""
case "$MODE" in
  native|starkv-recon|capture) ;;
  *) echo "usage: $0 <native|starkv-recon|capture> [stop]"; exit 1 ;;
esac

# Two argument shapes, because "capture" is the only mode whose $2 could be
# either an action or a value:
#   native|starkv-recon  <mode> [stop]
#   capture              capture <tag> | capture stop
TAG=""
if [ "$MODE" = capture ]; then
  case "${2:-}" in
    stop) ACTION=stop ;;
    *)    TAG=${2:-} ;;
  esac
  # The tag names both the capture directory and the boot log, so two capture
  # sessions never overwrite each other.
  [ -n "$TAG" ] || [ "$ACTION" = stop ] || { echo "usage: $0 capture <tag>"; exit 1; }
else
  ACTION=${2:-boot}
fi

SERVE_LOG="$LOG_HOST/serve_${MODE}${TAG:+_$TAG}.log"   # host-side log path
SERVE_LOG_CT=$(to_ct "$SERVE_LOG")                     # same file in the container

if [ "$ACTION" = stop ]; then
  docker stop -t 30 "$RUN_CONTAINER" >/dev/null 2>&1
  docker rm -f "$RUN_CONTAINER" >/dev/null 2>&1
  echo "stopped $MODE leg (container $RUN_CONTAINER)"
  exit 0
fi

require_allocation

# --- per-mode env + tree -----------------------------------------------
# Every mode names the whole gate set explicitly: a value left over from another
# mode would otherwise be silently honored at launch.
TREE="$SGLANG_PY_NATIVE"
CT_PYTHONPATH="$SGLANG_PY_NATIVE"
MODE_ENVS=(SGLANG_OPT_STARKV=0)
MOUNT_ARGS=()
case "$MODE" in
  starkv-recon)
    TREE="$SGLANG_PY_FORK"
    CT_PYTHONPATH="$SGLANG_PY_FORK:$REPO_CT"
    # Read-only: the leg must never write into the tree it is serving from.
    MOUNT_ARGS+=(-v "$FORK_HOST:$FORK_HOST:ro")
    MODE_ENVS=(SGLANG_OPT_STARKV=1 STARKV_RECON=1 STARKV_RANK="$STARKV_RANK"
               STARKV_BASIS="$(to_ct "$BASIS_HOST")")
    if [ -n "${STARKV_DEBUG:-}" ]; then
      MODE_ENVS+=("STARKV_DEBUG=$STARKV_DEBUG")
    fi
    ;;
  capture)
    TREE="$SGLANG_PY_FORK"
    CT_PYTHONPATH="$SGLANG_PY_FORK:$REPO_CT"
    MOUNT_ARGS+=(-v "$FORK_HOST:$FORK_HOST:ro")
    # Store OFF: the hooks must record native values, never compressed ones.
    MODE_ENVS=(SGLANG_OPT_STARKV=0 STARKV_CAPTURE="$TAG"
               STARKV_CAPTURES="$(to_ct "$CAPTURES_HOST")")
    # Window shape, passed through when set. How many rows a capture yields is
    # decided here, and it is the binding constraint on the analysis: the
    # held-out split needs half the sample to hold `rank` rows and the drift
    # split needs a quarter to, so a 256-step window leaves every rank above
    # ~192 unmeasurable. Exposed as an override rather than a constant because
    # the right window is a function of what the measurement is being asked.
    for kv in STARKV_CAPTURE_STEPS STARKV_CAPTURE_ROWS STARKV_CAPTURE_HEADS; do
      v=${!kv:-}
      [ -n "$v" ] && MODE_ENVS+=("$kv=$v")
    done
    ;;
esac

echo "== serve $MODE${TAG:+ ($TAG)} on devices=$DEVICES port=$PORT master=$MASTER_PORT"
echo "   tree=$TREE  log=$SERVE_LOG"
: > "$SERVE_LOG"   # truncate for a clean boot log

# The launch log is redirected INSIDE the container's shell: `docker run -d`
# would otherwise leave it in the container's own log, which dies with the
# container. A stale leg would also hold the GPUs, so clear one first.
docker rm -f "$RUN_CONTAINER" >/dev/null 2>&1

# Two things are NOT passed:
#
#   * CUDA_VISIBLE_DEVICES. `--gpus device=4,5,6,7` maps those cards to 0,1,2,3
#     *inside* the container, so re-exporting the host ids would name devices
#     that do not exist here. The runner's mapping is the truth.
#   * a bare `--gpus device=$DEVICES`. The --gpus value is a comma-separated
#     list of key=value pairs, so an unquoted "device=4,5,6,7" is split into
#     `device=4` plus the bare counts 5, 6 and 7 -- the last of which wins,
#     yielding Count=7 with DeviceIDs=["4"]. `docker create` accepts that;
#     `docker run` refuses it ("cannot set both Count and DeviceIDs"). The inner
#     quotes below are what tell docker's CSV parser that the commas belong to
#     the device list, giving DeviceIDs=["4","5","6","7"] and Count=0.
ENV_ARGS=()
for kv in "${MODE_ENVS[@]}"; do ENV_ARGS+=(-e "$kv"); done

# NOT `--rm`: a leg that dies at launch has to be inspectable (`docker logs`),
# and cleanup is already explicit -- `docker rm -f` immediately above and in the
# `stop` action, so a leftover container never survives a run.
docker run -d --name "$RUN_CONTAINER" \
  --network host --shm-size 120g \
  --gpus "\"device=$DEVICES\"" \
  -v /:/mnt/host_root -v /home:/home "${MOUNT_ARGS[@]}" \
  -w "$TREE" \
  -e "MASTER_PORT=$MASTER_PORT" \
  -e "PYTHONPATH=$CT_PYTHONPATH" \
  -e "NCCL_IB_DISABLE=1" -e "NCCL_SOCKET_IFNAME=lo" \
  -e "NCCL_P2P_LEVEL=NVL" -e "NCCL_PROTO=Simple" -e "NCCL_ALGO=Ring" \
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  "${ENV_ARGS[@]}" \
  "$IMAGE" \
  bash -c "exec sglang serve \
    --model-path $MODEL_CT --served-model-name $MODEL_NAME \
    --tp $TP --trust-remote-code --mem-fraction-static $MEM_FRAC \
    --context-length $CTX_LEN --max-running-requests $MAX_RUN \
    --chunked-prefill-size $CHUNK \
    --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4 \
    --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
    --host 0.0.0.0 --port $PORT \
    --cuda-graph-config '$DECODE_CFG' \
    --skip-server-warmup --watchdog-timeout 1800 \
    > $SERVE_LOG_CT 2>&1" \
  >/dev/null || { echo "FATAL: docker run failed" >&2; exit 1; }

# A leg that dies at launch -- a bad flag, a failed output redirect, a tree that
# is not there -- must fail HERE, in seconds, rather than as a 20-minute health
# timeout that says only "TIMEOUT". Give it a moment to fall over, then look.
sleep 5
docker inspect -f '{{.State.Running}}' "$RUN_CONTAINER" 2>/dev/null | grep -q true || {
  echo "FATAL: $RUN_CONTAINER exited within 5s of launch:" >&2
  docker logs "$RUN_CONTAINER" 2>&1 | tail -30 >&2
  [ -s "$SERVE_LOG" ] && { echo "--- $SERVE_LOG ---" >&2; tail -30 "$SERVE_LOG" >&2; }
  exit 1
}

wait_health 240 || { tail -40 "$SERVE_LOG"; exit 1; }
sleep 3
POOL=$(pool_of "$SERVE_LOG")
echo "  pool(max_total_num_tokens)=${POOL:-UNKNOWN}"
boot_markers "$SERVE_LOG" | sed 's/^/  /'
echo "serve $MODE${TAG:+ ($TAG)} UP on port $PORT -- leave running, tear down with: $0 $MODE stop"
