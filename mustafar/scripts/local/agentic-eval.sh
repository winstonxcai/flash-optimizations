#!/usr/bin/env bash
# =====================================================================
# agentic-eval.sh <sangfor|swe> <instance-list> [run-id]
#   Internal driver behind eval-sangfor.sh and eval-swe.sh: launches an
#   agentic eval (Claude Code agents driving a live local sglang server) on
#   the remote YJYBench box. The server must already be up (serve.sh
#   native|packed); the agents reach it via the docker_env_config's
#   ANTHROPIC_BASE_URL.
#
#   instance-list   path to a NEWLINE list of task ids, one per line.
#                   LOCAL paths are uploaded to the eval box's instance_file/
#                   (use an absolute /data/... path to reference an existing
#                   file already there).
#   run-id          results label (default mustafar-<bench>-<basename>-<ts>)
#   BASE_URL        optional override -- if set, a per-run copy of EVAL_CFG is
#                   made with only ANTHROPIC_BASE_URL patched (the auth token
#                   is copied verbatim and never read or echoed). Default: reuse
#                   EVAL_CFG as-is (it already points at the canonical serve
#                   port on this host).
#
# The run is launched DETACHED on the eval box (nohup) because agents run for
# hours; this script prints the run-id, the remote launch log, and how to poll.
# Config (env.sh): EVAL_SSH, EVAL_YJY, EVAL_VENV, EVAL_CFG.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

BENCH=${1:-} INSTANCE=${2:-} RID=${3:-}
[ -n "$BENCH" ] || { echo "usage: $0 <sangfor|swe> <instance-list> [run-id]"; exit 1; }
[ -n "$INSTANCE" ] || { echo "usage: $0 <sangfor|swe> <instance-list>"; exit 1; }

case "$BENCH" in
  sangfor) BENCH_ARGS="--benchmark Sangfor-Bench" ;;
  swe)     BENCH_ARGS="--benchmark SWE-bench --dataset SWE-bench_Verified" ;;
  *) echo "unknown bench '$BENCH' (sangfor|swe)"; exit 1 ;;
esac

# --- resolve instance list to a path already on the eval box -----------------
INST_REMOTE="$INSTANCE"
if [ -f "$INSTANCE" ]; then   # local file -> upload
  BASE=$(basename "$INSTANCE")
  /usr/bin/sshpass -p a scp -o StrictHostKeyChecking=no "$INSTANCE" \
    root@10.57.3.76:$EVAL_YJY/instance_file/ 2>/dev/null \
    || { echo "FATAL: could not upload $INSTANCE"; exit 1; }
  INST_REMOTE="$EVAL_YJY/instance_file/$BASE"
fi

# --- config: reuse EVAL_CFG, or make a base-url-patched per-run copy ---------
CFG_REMOTE="$EVAL_CFG"
if [ -n "${BASE_URL:-}" ]; then
  RID_SAFE=$(echo "$BENCH-$RID" | tr -c 'a-zA-Z0-9.-' '_')
  CFG_REMOTE="$EVAL_YJY/test_env/docker_env_config_mustafar_$RID_SAFE.json"
  echo ">> patching base-url of $EVAL_CFG -> $CFG_REMOTE (BASE_URL=$BASE_URL)"
  # Read the reference config remotely, patch ONLY the *_BASE_URL key, write the
  # copy. The auth token is copied verbatim -- never read or echoed here.
  $EVAL_SSH "$EVAL_VENV -" <<PY
import json
src = '$EVAL_CFG'
dst = '$CFG_REMOTE'
url = '$BASE_URL'
j = json.load(open(src))
env = j.get('experiment_env', j)
for k in list(env.keys()):
    if 'BASE_URL' in k.upper():
        env[k] = url
json.dump(j, open(dst, 'w'), indent=2)
print('patched', dst)
PY
fi

[ -z "$RID" ] && RID="mustafar-$BENCH-$(basename "$INSTANCE" .txt)-$(date +%Y%m%d_%H%M%S)"

echo "== $BENCH eval: run_id=$RID instance=$INST_REMOTE cfg=$CFG_REMOTE =="
echo "   launching DETACHED on the eval box ..."
# runs inside the heredoc: yjybench creates results/<run_id>/ and agents drive
# the server for hours; nohup keeps it alive after the ssh session ends.
$EVAL_SSH bash -s <<EOF
cd $EVAL_YJY || exit 1
nohup $EVAL_VENV -m yjybench.cli \\
  $BENCH_ARGS --agent_type cc --agent_mode vibe --mode e2e \\
  --run_id $RID --max_workers 8 --timeout 18000 --exp_name $RID \\
  --instance_file $INST_REMOTE \\
  --docker_env_config $CFG_REMOTE \\
  > results/${RID}_launch.log 2>&1 &
echo "started pid \$! (launch log: $EVAL_YJY/results/${RID}_launch.log)"
EOF

echo "== poll progress:  $EVAL_SSH 'tail -5 $EVAL_YJY/results/$RID/*/run.log' =="
