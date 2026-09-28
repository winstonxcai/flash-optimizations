#!/usr/bin/env bash
# =====================================================================
# profile-ranks.sh -- trigger a per-rank torch profile on a RUNNING server.
#
#   profile-ranks.sh <profile_id> [options]
#
#     --steps N        decode steps to capture          (default 20)
#     --stages S       comma list of stages to WAIT FOR (default decode)
#     --outdir DIR     HOST dir for traces  (default $RESULTS_HOST/profiles/<date>)
#     --prefix P       filename prefix                  (default empty)
#     --timeout S      seconds to wait for all traces   (default 300)
#     --no-shapes      drop record_shapes (smaller traces; loses the batch)
#
#   Writes, per rank:  <outdir>/<prefix><profile_id>-TP-<0..3>-<STAGE>.trace.json.gz
#
# Why this exists
# ---------------
# The 2026-09-15 traces that every mustafar decode number currently rests on
# were produced by an ad-hoc request that was never committed, so the exact
# capture parameters are unrecoverable -- and it now turns out they were wrong
# in ways that mattered (see mustafar/tools/trace_ranks.py). This pins them.
#
# What the server actually supports (sglang v0.5.18). The scheduler picks ONE
# of two implementations off envs.SGLANG_PROFILE_V2 (srt/environ.py:411,
# default False), and they differ in ways that matter here:
#
#   V2 = False  -- the live path in this container.
#     srt/managers/scheduler_components/profiler_manager.py. It ACCEPTS
#     profile_stages AND NEVER STORES IT: _init_profile hardcodes a target of
#     num_steps for prefill and decode alike (lines 139-143), and
#     _profile_batch_predicate (397-418) starts whichever stage it meets first.
#     So a request for ["decode"] profiles prefill as well. Filenames carry the
#     ForwardMode name, uppercased: -DECODE and -EXTEND.
#
#   V2 = True
#     srt/utils/profile_utils.py. This one DOES honour profile_stages
#     (interesting_stages = profile_stages or ["prefill","decode"]) and names
#     files with the lowercase API stage: -decode, -prefill.
#
# So --stages is a FILTER ON WHAT WE WAIT FOR, not a request to the server. The
# DECODE trace we came for lands either way; the prefill legs land beside it
# and are ignored. Matching is case-insensitive with prefill/extend as synonyms
# so one invocation works on either implementation.
#
# Both implementations share:
#   - num_steps MUST be set:        start_step is asserted to be null.
#   - profile_by_stage MUST be true.
#   - merge_profiles MUST be false: there is no cross-rank merge in this
#     version, so per-rank files are the only option and cross-rank comparison
#     is on per-rank windows -- valid because NCCL holds the ranks in lockstep.
#
# One POST covers all 4 TP ranks; the scheduler fans it out.
#
# With num_steps set the stage we asked for auto-stops. That does NOT mean the
# session is closed: on V2=False the OTHER stage opens on the next batch of its
# kind and stays open until it has seen num_steps of them, so a session can
# outlive the point that started it -- and can start after that point has
# already tidied up. An open session turns the next /start_profile into a
# silent no-op. So this script clears one on the way in, and closes the one it
# opened on the way out.
#
# The profile must be fired WHILE the client is at the target concurrency --
# the trigger counts real server steps, so an idle server profiles nothing.
# =====================================================================
set -u
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$DIR/env.sh"

PROFILE_ID=${1:-}
[ -n "$PROFILE_ID" ] || { echo "usage: $0 <profile_id> [--steps N] [--outdir DIR] ..." >&2; exit 2; }
shift

STEPS=20
STAGES=decode
OUTDIR=
PREFIX=
TIMEOUT=300
SHAPES=true
while [ $# -gt 0 ]; do
  case "$1" in
    --steps)   STEPS=$2; shift 2 ;;
    --stages)  STAGES=$2; shift 2 ;;
    --outdir)  OUTDIR=$2; shift 2 ;;
    --prefix)  PREFIX=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --no-shapes) SHAPES=false; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# STAGE list is only used to know which filenames to wait for.
IFS=',' read -r -a STAGE_ARR <<< "$STAGES"
[ ${#STAGE_ARR[@]} -gt 0 ] || { echo "no stages given" >&2; exit 2; }

if [ -z "$OUTDIR" ]; then
  OUTDIR="$RESULTS_HOST/profiles/$(date +%Y%m%d)"
fi
mkdir -p "$OUTDIR" || exit 1
OUTDIR_CT=$(to_ct "$OUTDIR")     # the SERVER writes the files, so it needs a
                                 # container-visible path, not the host one
health || {
  echo "FATAL: no server on port $PORT -- profile-ranks.sh profiles a running server" >&2
  exit 1
}

# The stage names the API uses are not the stage names the trace files use (see
# header): decode/prefill are the API's, DECODE/EXTEND are what V2=False writes.
# Map an API stage to the filename spellings that could satisfy it.
stage_spellings () {  # $1 = API stage -> candidate filename spellings
  case "$1" in
    decode)  echo "decode" ;;
    prefill) echo "prefill extend" ;;
    *)       echo "$1" ;;
  esac
}

# The V2=False writer upper-cases the stage, so match case-insensitively rather
# than guessing the case for whichever implementation is live.
find_trace () {  # $1=rank $2=API stage -> print the path, exit 0 if it exists
  local r=$1 s f
  for s in $(stage_spellings "$2"); do
    f=$(find "$OUTDIR" -maxdepth 1 -type f \
          -iname "${PREFIX}${PROFILE_ID}-TP-${r}-${s}.trace.json.gz" \
          -print -quit 2>/dev/null)
    [ -n "$f" ] && { echo "$f"; return 0; }
  done
  return 1
}

# Wait for all 4 ranks x each requested stage before declaring success.
wait_traces () {  # $1 = seconds
  local deadline=$(( $(date +%s) + $1 )) want=$(( 4 * ${#STAGE_ARR[@]} )) got=0 r s
  while [ "$(date +%s)" -lt "$deadline" ]; do
    got=0
    for r in 0 1 2 3; do
      for s in "${STAGE_ARR[@]}"; do
        find_trace "$r" "$s" > /dev/null && got=$((got + 1))
      done
    done
    [ "$got" -ge "$want" ] && { echo "  all $want traces present"; return 0; }
    sleep 5
  done
  echo "  TIMEOUT: only $got/$want traces appeared in ${1}s" >&2
  return 1
}

# Close a profile session, if one is open. The server AWAITS this, so a return
# means the trace is flushed and profile_in_progress is false.
#
# It is NOT idempotent. Stopping a session that is not open raises RuntimeError
# in the scheduler and the endpoint answers HTTP 500 (observed: "Profiling is
# not in progress. Call /start_profile first."). "No session open" is the state
# we want, so 500 counts as success here. No -f, because we want the code.
#
# The long -m is the flush itself: a prefill leg is ~70 MB per rank.
stop_profile () {
  local code
  code=$(curl -sS -m 900 -o /dev/null -w '%{http_code}' \
           -X POST "http://127.0.0.1:$PORT/stop_profile" \
           -H 'Content-Type: application/json' -d '{}' 2>/dev/null) || code=000
  case "$code" in
    200) echo "   stop_profile: closed an open session" ;;
    500) echo "   stop_profile: none open (clean already)" ;;
    *)   echo "   stop_profile: unexpected HTTP $code" ;;
  esac
  return 0
}

# The server answers a FIXED 200 "Start profiling." whether or not it actually
# started one (http_server.py:1159-1166), so the only honest check is whether
# the traces turn up afterwards.
start_profile () {
  local resp
  resp=$(curl -fsS -m 30 -X POST "http://127.0.0.1:$PORT/start_profile" \
           -H 'Content-Type: application/json' -d "$BODY" 2>&1) || resp="(request failed)"
  echo "   response: $resp"
  case "$resp" in
    *'Start profiling'*) return 0 ;;
    *) echo "FATAL: /start_profile returned an unexpected response" >&2; return 1 ;;
  esac
}

if [ "$SHAPES" = true ]; then SHAPES_JSON=true; else SHAPES_JSON=false; fi

# start_step / merge_profiles are deliberately absent: the server asserts both
# are unset (see header). stages is a JSON array built from the comma list.
STAGES_JSON=$(printf '"%s",' "${STAGE_ARR[@]}"); STAGES_JSON="[${STAGES_JSON%,}]"

BODY=$(cat <<JSON
{
  "output_dir": "$OUTDIR_CT",
  "num_steps": $STEPS,
  "profile_by_stage": true,
  "profile_stages": $STAGES_JSON,
  "activities": ["CPU", "GPU"],
  "profile_id": "$PROFILE_ID",
  "profile_prefix": "$PREFIX",
  "record_shapes": $SHAPES_JSON
}
JSON
)

echo "== profiling id=$PROFILE_ID steps=$STEPS stages=$STAGES shapes=$SHAPES"
echo "   outdir (host)      = $OUTDIR"
echo "   outdir (container) = $OUTDIR_CT"

# A session left open by an earlier point makes /start_profile a silent no-op,
# so clear one before asking. On V2=False a point can genuinely leave one open:
# after the DECODE target is met, EXTEND starts on the next prefill batch and
# only closes once it has seen num_steps prefills of its own. If the wave drains
# first, that session outlives the point -- and it can start AFTER any stop we
# do on the way out, which is why the cleanup has to be here, at entry.
stop_profile
start_profile || exit 1

if ! wait_traces "$TIMEOUT"; then
  # Usually a stale session that slipped in between the stop above and the
  # start. Clear it and take one more run at it.
  echo "  retrying once: clearing any stale session" >&2
  stop_profile
  start_profile || exit 1
  wait_traces "$TIMEOUT" || { stop_profile; exit 1; }
fi

# The traces we wanted have landed. Close the session so the next point starts
# clean rather than relying on EXTEND reaching its target on its own.
stop_profile

echo "== traces in $OUTDIR"
ls -la "$OUTDIR" | grep -- "$PROFILE_ID" | sed 's/^/   /'
