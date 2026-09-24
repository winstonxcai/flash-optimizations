#!/usr/bin/env bash
# =====================================================================
# env.sh -- shared runtime configuration and helpers for local driver scripts
#
# TO PORT TO A NEW MACHINE/GPU NODE: override the MACHINE CONFIG values below
# (or export them before invoking a script). All drivers source this file:
#   . "$(dirname "$0")/env.sh"
# =====================================================================
set -u

# ------------------------------ MACHINE CONFIG -------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HOST_REPO=${HOST_REPO:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
CONTAINER=${CONTAINER:-remnant}

# The Dockerfile is the source of truth for serving. It installs the reviewed
# production fork at this path; neither native nor packed serving imports the
# repository's old third_party/sglang checkout anymore.
REPO_CT=${REPO_CT:-/mnt/host_root$HOST_REPO}
SGLANG_ROOT_CT=${SGLANG_ROOT_CT:-/sgl-workspace/sglang-remnant}
SGLANG_PY=${SGLANG_PY:-$SGLANG_ROOT_CT/python}
SGLANG_ROOT=${SGLANG_ROOT:-$SGLANG_ROOT_CT}
MODEL_PATH=${MODEL_PATH:-/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731}
MODEL_CT=${MODEL_CT:-/mnt/host_root$MODEL_PATH}

RESULTS_HOST=${RESULTS_HOST:-$HOST_REPO/results}
LOG_HOST=${LOG_HOST:-$HOST_REPO/logs}

# -----------------------------------------------------------------------------

# Run defaults (override before sourcing / on the command line).
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-30212}
MASTER_PORT=${MASTER_PORT:-29638}
SERVER_PID_FILE_CT=${SERVER_PID_FILE_CT:-/tmp/remnant-sglang-${PORT}.pid}
TP=${TP:-4}
MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash}
MEM_FRAC=${MEM_FRAC:-0.88}
CTX_LEN=${CTX_LEN:-1048576}
MAX_RUN=${MAX_RUN:-256}
CHUNK=${CHUNK:-8192}
OUTLEN=${OUTLEN:-2048}     # bench_serving output length
SEED=${SEED:-42}

# Decode CUDA-graph configs live in config/*.json so every launcher reads the
# same definitions. DECODE_CFG defaults to the small agentic configuration;
# serving capacity drivers explicitly select the extended configuration.
CONFIG_DIR=${CONFIG_DIR:-$SCRIPT_DIR/config}
DECODE_CFG_SMALL=${DECODE_CFG_SMALL:-$(<"$CONFIG_DIR/decode-graphs-small.json")}
DECODE_CFG_EXT=${DECODE_CFG_EXT:-$(<"$CONFIG_DIR/decode-graphs-extended.json")}
DECODE_CFG=${DECODE_CFG:-$DECODE_CFG_SMALL}

ts () { date +%Y%m%d_%H%M%S; }

# Container-visible path of a host path. The local container mounts / at
# /mnt/host_root, while Modal runs the same scripts directly and does not use
# this helper.
to_ct () { echo "$REPO_CT${1#$HOST_REPO}"; }

# Run a script on stdin, passing values as positional arguments. Use this for
# container commands so paths and JSON values are not interpolated into bash -c.
ct_script () { docker exec "$CONTAINER" bash -s -- "$@"; }

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

# Stop the production-fork server recorded for $PORT. The launcher creates a
# dedicated process group, so this does not kill unrelated processes.
kill_port () {
  ct_script "$SERVER_PID_FILE_CT" "$PORT" <<'BASH' 2>/dev/null
set -u
pid_file=$1
port=$2
status=0
remove_pid_file=1
terminate_group() {
  kill -TERM -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 -- "-$pgid" 2>/dev/null || break
    sleep 0.25
  done
  kill -KILL -- "-$pgid" 2>/dev/null || true
  for _ in $(seq 1 4); do
    kill -0 -- "-$pgid" 2>/dev/null || return 0
    sleep 0.25
  done
  return 1
}
if [ -s "$pid_file" ]; then
  if ! read -r pid pgid sid start_ticks recorded_port < "$pid_file"; then
    status=1
    remove_pid_file=0
  elif [[ "${pid:-}" =~ ^[1-9][0-9]*$ && "${pgid:-}" =~ ^[1-9][0-9]*$ &&
        "${sid:-}" =~ ^[1-9][0-9]*$ && "${start_ticks:-}" =~ ^[0-9]+$ &&
        "${recorded_port:-}" == "$port" && "${pid:-}" == "${pgid:-}" &&
        "${pgid:-}" == "${sid:-}" ]]; then
    if [ -r "/proc/$pid/stat" ]; then
      stat_line=$(<"/proc/$pid/stat")
      stat_tail=${stat_line##*) }
      read -r -a stat_fields <<< "$stat_tail"
      current_pgid=${stat_fields[2]:-}
      current_sid=${stat_fields[3]:-}
      current_start_ticks=${stat_fields[19]:-}
      if [ "$current_pgid" = "$pgid" ] && [ "$current_sid" = "$sid" ] &&
         [ "$current_start_ticks" = "$start_ticks" ] && kill -0 -- "-$pgid" 2>/dev/null; then
        if ! terminate_group; then
          echo "server process group $pgid did not exit after SIGKILL" >&2
          status=1
          remove_pid_file=0
        fi
      else
        echo "ignoring stale or mismatched server PID record for port $port" >&2
        status=1
        remove_pid_file=0
      fi
    elif kill -0 -- "-$pgid" 2>/dev/null; then
      # The session leader may have exited while child workers remain. The
      # still-existing process group retains its ID, so terminate that group.
      if ! terminate_group; then
        echo "server process group $pgid did not exit after SIGKILL" >&2
        status=1
        remove_pid_file=0
      fi
    fi
  else
    echo "cannot verify server PID record for port $port" >&2
    status=1
    remove_pid_file=0
  fi
  [ "$remove_pid_file" = 0 ] || rm -f -- "$pid_file"
fi
exit "$status"
BASH
  local rc=$?
  sleep 4
  return "$rc"
}

# Boot markers we care about, printed from a host launch log.
boot_markers () {  # $1=host launch log
  grep -aoE "logical_row_bytes=[0-9]+ layers=[0-9]+|max_total_num_tokens=[0-9]+|Dequantized FP4|is fired up and ready" "$1" | head -20
}

pool_of () {  # $1=host launch log -> max_total_num_tokens ("" if not found yet)
  grep -aoE "max_total_num_tokens=[0-9]+" "$1" | head -1 | grep -oE "[0-9]+"
}
