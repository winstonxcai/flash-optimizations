#!/usr/bin/env bash
# =====================================================================
# kernel-run.sh -- run the mustafar GPU suites in a container pinned to a
#               GPUQ-granted device.
#
#   gpuq run --project mustafar --gpus 1 --timeout 30m \
#     --cwd /home/jovyan/winstonxcai/flash-optimizations \
#     -- bash mustafar/scripts/local/kernel-run.sh
#
#   gpuq lease acquire --project mustafar --gpus 1 --gpu-ids 5 \
#     --ttl 45m --keep-idle --detach          # -> lease-...
#   MUSTAFAR_DEVICES=5 MUSTAFAR_LEASE=lease-... \
#     bash mustafar/scripts/local/kernel-run.sh  # the container is the CUDA process
#
#   kernel-run.sh --validity-only  validity only
#   kernel-run.sh --speed-only     speed only
#
# This script never chooses a card and never uses `--gpus all`. It takes the
# device ids from exactly one of two places, in this order:
#
#   1. CUDA_VISIBLE_DEVICES, when running as a `gpuq run` job (the daemon sets
#      it to the ids it allocated).
#   2. MUSTAFAR_DEVICES plus MUSTAFAR_LEASE, when a container has to be launched
#      from outside the scheduler -- see the note below. The lease is required
#      so the ids are traceable to a reservation rather than to someone's guess.
#
# Why the second path exists: jobs on this host run as the unprivileged
# gpuq-runner user, which is not in the `docker` group, so a `gpuq run` job
# cannot start a container here at all. The suites need the image's torch and
# sglang, so the container has to be launched by a user who can reach the docker
# socket -- but still pinned to a device the scheduler has granted.
#
# The container is a throwaway from $IMAGE rather than a `docker exec` into
# `remnant`: remnant is a long-lived serving box brought up with `--gpus all`,
# and `docker exec` inherits none of the caller's environment, so a run inside it
# would see all eight cards and validity.py's `cuda:0` would land on whichever
# comes first -- on this host, someone else's sglang. The suites driven here
# (validity, speed) import only the *stock* sglang tree, which the image already
# exposes through its editable install, so no clone or patch step is needed and
# the run cannot disturb the patched lowrank tree.
#
# Unlike a `gpuq run` job, nothing supervises this process, so the script bounds
# itself: `timeout` on the docker run and a trap that removes the container.
#
# Artifacts land under $GPUQ_OUTPUT_DIR when set (jobs), else $RESULTS_HOST, and
# are also exported as MUSTAFAR_RESULTS_DIR for speed.py's speed.{json,csv}.
# =====================================================================
set -euo pipefail

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
. "$DIR/env.sh"

die () { echo "FATAL: $*" >&2; exit 1; }

# --- resolve the device ids; never guess ------------------------------
if [ -n "${GPUQ_JOB_ID:-}" ]; then
  CONTEXT="job ${GPUQ_JOB_ID}"
  DEVICES=${CUDA_VISIBLE_DEVICES:-}
  [ -n "$DEVICES" ] || die "job ${GPUQ_JOB_ID} was started without CUDA_VISIBLE_DEVICES"
else
  [ -n "${MUSTAFAR_DEVICES:-}" ] || die "no allocation: run this under 'gpuq run', or set MUSTAFAR_DEVICES (and MUSTAFAR_LEASE) from a lease you hold"
  [ -n "${MUSTAFAR_LEASE:-}" ] || die "MUSTAFAR_DEVICES=${MUSTAFAR_DEVICES} names no lease; set MUSTAFAR_LEASE to the lease that granted it"
  CONTEXT="lease ${MUSTAFAR_LEASE}"
  DEVICES=$MUSTAFAR_DEVICES
fi

MODE=${1:-both}
case "$MODE" in
  both|--validity-only|--speed-only) ;;
  *) die "unknown argument: $MODE (expected --validity-only or --speed-only)" ;;
esac

IMAGE=${IMAGE:-remnant:v0.5.18}
# Host path, and the same path inside the container: /home is bind-mounted
# identically, so no /mnt/host_root translation is needed here.
OUT=${GPUQ_OUTPUT_DIR:-$RESULTS_HOST/sparse-$(date -u +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
STOCK=/sgl-workspace/sglang
NAME=mustafar-kernel-run
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}

echo "== ${CONTEXT}  project=${GPUQ_PROJECT:-mustafar}  devices=${DEVICES}"
echo "== image ${IMAGE}   out ${OUT}   timeout ${RUN_TIMEOUT}s"

# The validity leg is every stage across every leg against native; the speed leg
# is the same comparison timed against the reassembling path. pipefail so a failed
# assertion in validity cannot be masked by `tee` and let speed run on a kernel
# that just failed.
INNER="set -eo pipefail"
if [ "$MODE" != "--speed-only" ]; then
  INNER="$INNER
python3 -m mustafar.tests.validity 2>&1 | tee $OUT/validity.log"
fi
if [ "$MODE" != "--validity-only" ]; then
  INNER="$INNER
python3 -m mustafar.tests.speed 2>&1 | tee $OUT/speed.log"
fi

cleanup () { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker rm -f "$NAME" >/dev/null 2>&1 || true
# -k so a container that ignores SIGTERM is killed rather than orphaned.
timeout -k 30 "$RUN_TIMEOUT" docker run --rm --name "$NAME" \
  --network host --shm-size 120g \
  --gpus "device=${DEVICES}" \
  -v /:/mnt/host_root -v /home:/home \
  -e "PYTHONPATH=$REPO_CT" \
  -e "MUSTAFAR_RESULTS_DIR=$OUT" \
  -w "$STOCK" \
  "$IMAGE" bash -c "$INNER"

echo "== done; artifacts in $OUT"

