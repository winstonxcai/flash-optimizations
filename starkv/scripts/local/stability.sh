#!/usr/bin/env bash
# =====================================================================
# stability.sh -- the long-decode collapse ladder, as a two-phase GPUQ job.
#
#   stability.sh resolve <rank>              print the replay prompt for a rank
#   stability.sh reference <tag> [rank] [lengths]   boot native, generate + score
#   stability.sh compare   <tag> [lengths]          boot starkv-recon, score it
#
#   tag      = result label; results land in $GPUQ_OUT/stability-<tag>.{json,md}
#   rank     = which long replay prompt (1..N) from data/long-prompts.jsonl (default 1)
#   lengths  = comma-separated decode lengths to report (default 2048,8192,32768)
#
# Both GPU phases are GPUQ entrypoints -- submit them, do not run them by hand:
#
#   gpuq run --project starkv --gpus 4 --timeout 60m \
#     --output /home/jovyan/gpuq-results/starkv \
#     --cwd /home/jovyan/winstonxcai/flash-optimizations \
#     -- bash starkv/scripts/local/stability.sh reference <tag> 1
#   ... then, once that finishes ...
#     -- bash starkv/scripts/local/stability.sh compare <tag>
#
# Why two phases rather than one run over two servers: at the memory fraction
# this model needs, one leg already claims the devices it was granted, and GPUQ
# grants one allocation per job. So phase 1 generates the reference continuation
# on the native leg and writes it -- text plus that leg's per-position logprobs
# -- under $GPUQ_OUT; phase 2 boots the other leg and scores the byte-identical
# text. Both phases agree on the token sequence by construction, which is the
# only way a teacher-forced drift number means anything.
#
# What it measures, and why the shape matters more than the endpoint: retention
# loss does not announce itself as a wrong answer, it announces itself as babble
# thousands of decode steps in. So phase 2 reports |dlogprob| and top-1
# agreement *bucketed by decode position*, plus a temperature-0 degeneration
# profile (distinct-n, longest repeated n-gram) on each leg. A curve that grows
# with position is the failure; a flat one is the basis holding. See
# analysis/stability.py for the details.
#
# Both legs boot eager (DECODE_CFG_OFF) -- these are correctness runs, and the
# ladder has to see every decode step.
# =====================================================================
set -u
DIR=$(cd -- "$(dirname -- "$0")" && pwd)
. "$DIR/env.sh"

PHASE=${1:-}
PROMPT_LIST="$HOST_REPO/starkv/data/long-prompts.jsonl"
HOST_PY=${HOST_PY:-python3}

resolve_rank () {  # $1=rank -> the replay request path for that rank
  "$HOST_PY" - "$PROMPT_LIST" "$1" <<'PY'
import json, sys
path, rank = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(path) if l.strip()]
row = next((r for r in rows if r["rank"] == rank), None)
if row is None:
    raise SystemExit(f"rank {rank} not in {path} (have {len(rows)})")
print(row["path"])
PY
}

case "$PHASE" in
  resolve)
    RANK=${2:-1}
    resolve_rank "$RANK" || exit 1
    ;;

  reference)
    TAG=${2:-}; RANK=${3:-1}; LENGTHS=${4:-2048,8192,32768}
    [ -n "$TAG" ] || { echo "usage: $0 reference <tag> [rank] [lengths]" >&2; exit 1; }
    PROMPT=$(resolve_rank "$RANK") || exit 1
    require_allocation
    echo "== stability reference tag=$TAG rank=$RANK lengths=$LENGTHS"
    echo "   prompt: $PROMPT"

    # Armed before the boot: a serve.sh that fails partway has still created a
    # container, and leaving it would hold the granted devices. `stop` is
    # idempotent, so arming it early costs nothing.
    trap 'bash "$DIR/serve.sh" native stop' EXIT
    bash "$DIR/serve.sh" native || exit 1

    ( cd "$HOST_REPO" && "$HOST_PY" -m starkv.analysis.stability reference \
        --server "http://127.0.0.1:$PORT" --prompt "$PROMPT" \
        --lengths "$LENGTHS" --tag "$TAG" --out "$GPUQ_OUT/stability-$TAG" )
    RC=$?

    bash "$DIR/serve.sh" native stop
    trap - EXIT
    echo
    echo "[stability] reference saved to $GPUQ_OUT/stability-$TAG.json"
    echo "[stability] next job: stability.sh compare $TAG '$LENGTHS'"
    exit $RC
    ;;

  compare)
    TAG=${2:-}; LENGTHS=${3:-2048,8192,32768}
    [ -n "$TAG" ] || { echo "usage: $0 compare <tag> [lengths]" >&2; exit 1; }
    [ -f "$GPUQ_OUT/stability-$TAG.json" ] || {
      echo "FATAL: $GPUQ_OUT/stability-$TAG.json missing; run the reference phase first" >&2
      exit 1; }
    require_allocation
    echo "== stability compare tag=$TAG lengths=$LENGTHS (vs the saved reference)"

    # Armed before the boot -- see the note in the reference phase.
    trap 'bash "$DIR/serve.sh" starkv-recon stop' EXIT
    bash "$DIR/serve.sh" starkv-recon || exit 1

    ( cd "$HOST_REPO" && "$HOST_PY" -m starkv.analysis.stability compare \
        --server "http://127.0.0.1:$PORT" \
        --lengths "$LENGTHS" --tag "$TAG" --out "$GPUQ_OUT/stability-$TAG" )
    RC=$?

    bash "$DIR/serve.sh" starkv-recon stop
    trap - EXIT
    echo
    echo "[stability] results: $GPUQ_OUT/stability-$TAG.md  /  $GPUQ_OUT/stability-$TAG.json"
    echo "[stability] copy into the repo with:"
    echo "  cp $GPUQ_OUT/stability-$TAG.* $RESULTS_HOST/"
    exit $RC
    ;;

  *) echo "usage: $0 <resolve|reference|compare> ..." >&2; exit 2 ;;
esac
