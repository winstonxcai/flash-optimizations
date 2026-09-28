#!/usr/bin/env bash
# =====================================================================
# capture.sh <tag> [duration_s] [concurrency]
#
# Collect the per-layer samples `spectrum.py` measures, by booting the fork tree
# with the store OFF and the capture hooks ON, driving the LongSWE-Bench replay
# against it, and shutting it down.
#
#   tag    = capture label; samples land in <CAPTURES_HOST>/<tag>/
#   DUR,C  = replay window and concurrency (default 600 s / 15)
#   STEPS,R= decode steps per layer, row guard (default 256 / 16384)
#
# Window shape, exported when invoking this script (see serve.sh):
#   STARKV_CAPTURE_STEPS  decode steps recorded per layer (default 256; arg 4)
#   STARKV_CAPTURE_ROWS   runaway guard on stored rows per layer (default 16384; arg 5)
#
# STARKV_CAPTURE_STEPS is the number that decides whether the measurement can
# answer its own question, and it is worth sizing before the job rather than
# after. Each step yields roughly two real store rows (decode stores a row per
# compress unit, and only the boundaries are kept), and spectrum.py's splits
# need half the sample to hold `rank` rows (held-out) and a quarter to (drift).
# The 256-step default yields ~490 rows per layer, which leaves every rank above
# ~192 unmeasurable -- fine for a smoke test, not for the r <= 320 the decision
# rule turns on. ~1400 steps puts ~2700 rows behind each layer and covers the
# full rank sweep.
#
# This is a GPUQ entrypoint -- submit it, do not run it by hand:
#
#   gpuq run --project starkv --gpus 4 --timeout 60m \
#     --output /home/jovyan/gpuq-results/starkv \
#     --cwd /home/jovyan/winstonxcai/flash-optimizations \
#     -- bash starkv/scripts/local/capture.sh <tag> [dur] [conc] [steps] [rows]
#
# The job holds ONE allocation for the whole capture: the server runs in a
# throwaway container pinned to the granted devices, and the replay client runs
# on the host in the same job, reaching the server over --network host. Nothing
# here uses `docker run --gpus all`.
#
# Why the replay and nothing else: the sample has to be the distribution the
# model actually stores during a real coding session, and the replay is the
# recorded shape of one -- ~144k-token prompts, short decodes, many concurrent
# requests. A synthetic prompt set would sample a distribution nobody serves.
#
# The store is OFF for this leg (SGLANG_OPT_STARKV=0), so every recorded value is
# native ground truth: nothing is measured through the compression under test.
#
# The measurement afterwards needs no GPU at all:
#   cd flash-optimizations
#   python3 -m starkv.analysis.spectrum starkv/captures/<tag>
# =====================================================================
set -u
DIR=$(cd -- "$(dirname -- "$0")" && pwd)
. "$DIR/env.sh"

TAG=${1:-} DUR=${2:-600} C=${3:-15} STEPS=${4:-} ROWS=${5:-}
[ -n "$TAG" ] || { echo "usage: $0 <tag> [duration_s] [concurrency] [steps] [rows]"; exit 1; }

# Window shape is a positional argument, not an exported env var, because this
# runs as a `gpuq run` job and there is no guarantee the submitting shell's
# environment reaches it -- but the argv does, unchanged.
[ -n "$STEPS" ] && export STARKV_CAPTURE_STEPS="$STEPS"
[ -n "$ROWS" ] && export STARKV_CAPTURE_ROWS="$ROWS"

require_allocation

RUN_ROOT="$GPUQ_OUT/capture/$TAG/$(ts)"
mkdir -p "$RUN_ROOT"

echo "== capture tag=$TAG c$C @ ${DUR}s -> $CAPTURES_HOST/$TAG"

# Boot the capture leg (eager, store off, hooks on) and make sure it comes down
# again even if the client fails -- a leftover leg would hold the devices.
#
# The trap is armed BEFORE the boot, not after: a serve.sh that fails partway
# (a bad launch, a health timeout) has still created a container, and `stop` is
# idempotent, so arming it first is what actually guarantees the devices are
# released. Arming it after `serve.sh` returns left exactly the crashed leg
# behind that it was meant to clean up.
trap 'bash "$DIR/serve.sh" capture stop' EXIT
bash "$DIR/serve.sh" capture "$TAG" || exit 1

( cd "$REPLAY_DIR" && "${HOST_PY:-python3}" -B "$REPLAY_RUNNER" \
    --result-root "$RUN_ROOT/client" \
    --dataset-root "$REPLAY_DATASET" \
    --dataset-manifest-input "$REPLAY_MANIFEST" \
    --adapter "$REPLAY_ADAPTER" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL_NAME" \
    --max-requests "$REPLAY_REQUESTS" --max-concurrency "$C" --max-duration "$DUR" \
    --arrival-mode immediate --time-scale 60 --max-gap 30 \
    --timeout 21600 --minimum-success-rate 0.99 \
    --expected-protocol business-user-replay-v2 \
    --audit-level candidate --return-cached-tokens-details \
    > "$RUN_ROOT/client.log" 2>&1 )
RC=$?
echo "[capture] client rc=$RC (log: $RUN_ROOT/client.log)"

# Stop here rather than at the trap, so the atexit flush and the listing below
# both see a settled directory.
bash "$DIR/serve.sh" capture stop
trap - EXIT

echo "== captured files =="
ls -la "$CAPTURES_HOST/$TAG" 2>/dev/null || echo "  (nothing written -- did the hooks see any decode steps?)"

cat <<EOF

[capture] samples:  $CAPTURES_HOST/$TAG
[capture] run log:  $RUN_ROOT/client.log
[capture] measure (CPU-only, host, no GPU):
  cd $HOST_REPO
  python3 -m starkv.analysis.spectrum starkv/captures/$TAG
EOF
exit $RC
