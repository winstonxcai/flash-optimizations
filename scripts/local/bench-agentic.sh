#!/usr/bin/env bash
# =====================================================================
# bench-agentic.sh <sangfor|swe> <instance-list> [run-id]
#   Entry point for the agentic benchmarks: launches an agentic benchmark (Claude Code
#   agents driving a live local sglang server) on the remote YJYBench box. The
#   server must already be up (serve.sh native|packed); the agents reach it via
#   the docker_env_config's ANTHROPIC_BASE_URL.
#
#   instance-list   path to a NEWLINE list of task ids, one per line.
#                   LOCAL paths are uploaded to the eval box's instance_file/
#                   (use an absolute /data/... path to reference an existing
#                   file already there).
#   run-id          results label (default remnant-<bench>-<basename>-<ts>)
#   BASE_URL        optional override -- if set, a per-run copy of EVAL_CFG is
#                   made with only ANTHROPIC_BASE_URL patched (the auth token
#                   is copied verbatim and never read or echoed). Default: reuse
#                   EVAL_CFG as-is (it already points at the canonical serve
#                   port on this host).
#
# The run is launched DETACHED on the eval box (nohup) because agents run for
# hours; this script prints the run-id, the remote launch log, and how to poll.
# Config (eval-env.sh): EVAL_SSH, EVAL_SCP, EVAL_YJY, EVAL_VENV, EVAL_CFG.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"
. "$(dirname "$0")/eval-env.sh"

BENCH=${1:-} INSTANCE=${2:-} RID=${3:-}
[ -n "$BENCH" ] || { echo "usage: $0 <sangfor|swe> <instance-list> [run-id]"; exit 1; }
[ -n "$INSTANCE" ] || { echo "usage: $0 <sangfor|swe> <instance-list>"; exit 1; }
require_remote_eval_config

case "$BENCH" in
  sangfor) BENCH_NAME=Sangfor-Bench; DATASET_NAME= ;;
  swe)     BENCH_NAME=SWE-bench; DATASET_NAME=SWE-bench_Verified ;;
  *) echo "unknown bench '$BENCH' (sangfor|swe)"; exit 1 ;;
esac

[ -z "$RID" ] && RID="remnant-$BENCH-$(basename "$INSTANCE" .txt)-$(date +%Y%m%d_%H%M%S)"

# --- resolve instance list to a path already on the eval box -----------------
INST_REMOTE="$INSTANCE"
if [ -f "$INSTANCE" ]; then   # local file -> upload
  : "${EVAL_SCP:?Set EVAL_SCP to upload local instance lists}"
  BASE=$(basename "$INSTANCE")
  $EVAL_SCP "$INSTANCE" "$EVAL_YJY/instance_file/" 2>/dev/null \
    || { echo "FATAL: could not upload $INSTANCE"; exit 1; }
  INST_REMOTE="$EVAL_YJY/instance_file/$BASE"
fi

# --- config: reuse EVAL_CFG, or make a base-url-patched per-run copy ---------
CFG_REMOTE="$EVAL_CFG"
if [ -n "${BASE_URL:-}" ]; then
  RID_SAFE=$(echo "$BENCH-$RID" | tr -c 'a-zA-Z0-9.-' '_')
  CFG_REMOTE="$EVAL_YJY/test_env/docker_env_config_remnant_$RID_SAFE.json"
  echo ">> patching base-url of $EVAL_CFG -> $CFG_REMOTE (BASE_URL=$BASE_URL)"
  # Read the reference config remotely, patch ONLY the *_BASE_URL key, write the
  # copy. The auth token is copied verbatim -- never read or echoed here.
  if ! $EVAL_SSH "$EVAL_VENV" - "$EVAL_CFG" "$CFG_REMOTE" "$BASE_URL" <<'PY'
import json
import sys
src, dst, url = sys.argv[1:]
j = json.load(open(src))
env = j.get('experiment_env', j)
for k in list(env.keys()):
    if 'BASE_URL' in k.upper():
        env[k] = url
json.dump(j, open(dst, 'w'), indent=2)
print('patched', dst)
PY
  then
    echo "FATAL: could not create remote evaluation config" >&2
    exit 1
  fi
fi

echo "== $BENCH benchmark: run_id=$RID instance=$INST_REMOTE cfg=$CFG_REMOTE =="
echo "   launching DETACHED on the eval box ..."
# runs inside the heredoc: yjybench creates results/<run_id>/ and agents drive
# the server for hours; nohup keeps it alive after the ssh session ends.
if ! $EVAL_SSH bash -s -- "$EVAL_YJY" "$EVAL_VENV" "$BENCH_NAME" \
  "$DATASET_NAME" "$RID" "$INST_REMOTE" "$CFG_REMOTE" <<'EOF'
set -eu
yjy=$1
venv=$2
bench_name=$3
dataset_name=$4
rid=$5
instance_file=$6
config_file=$7
cd -- "$yjy"
args=(--benchmark "$bench_name" --agent_type cc --agent_mode vibe --mode e2e
  --run_id "$rid" --max_workers 8 --timeout 18000 --exp_name "$rid"
  --instance_file "$instance_file" --docker_env_config "$config_file")
[ -z "$dataset_name" ] || args+=(--dataset "$dataset_name")
nohup "$venv" -m yjybench.cli "${args[@]}" \
  > "results/${rid}_launch.log" 2>&1 &
echo "started pid $! (launch log: $yjy/results/${rid}_launch.log)"
EOF
then
  echo "FATAL: remote benchmark launch failed" >&2
  exit 1
fi

echo "== poll progress:  $EVAL_SSH 'tail -5 $EVAL_YJY/results/$RID/*/run.log' =="
