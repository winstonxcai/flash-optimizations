#!/usr/bin/env bash
# Own Docker setup checks, mounts, GPU selection, deadlines and container cleanup.
# Use an existing capture-only image; never build, pull, or replace production.
# Author: Winston Cai.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
runner=/sgl-workspace/sglang-remnant/python/sglang/benchmark/remnant/calibrate.py

# These two actions run inside the prepared container, using this same script.
preflight_image() {
  local port=$1 root=/sgl-workspace/sglang-remnant/python/sglang package operation
  require_marker() {
    grep -q "$2" "$1" || { echo "Prepared image lacks $2: $1" >&2; exit 1; }
  }
  require_marker "$root/srt/remnant/capture.py" observe_raw
  require_marker "$root/srt/layers/attention/dsv4/compressor_v2.py" observe_raw
  require_marker "$root/srt/layers/attention/deepseek_v4_backend.py" observe_attention
  require_marker "$root/srt/models/deepseek_v4.py" observe_output
  package=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/sgl_kernel")')
  for operation in flashmla_ops remnant_ops; do
    compgen -G "$package/$operation*.so" >/dev/null || {
      echo "Missing baked $operation in $package" >&2; exit 1;
    }
  done
  require_marker "$package/flash_mla.py" remnant_sparse_decode_fwd
  for operation in curl setsid timeout; do
    command -v "$operation" >/dev/null || { echo "Missing container tool: $operation" >&2; exit 1; }
  done
  exec python3 "$runner" capture-preflight --root /capture --publish-root /published --model /model --port "$port"
}

remaining_seconds() {
  local left=$(( $1 - SECONDS ))
  (( left > 0 )) || { echo 'Capture budget exhausted during startup or replay' >&2; return 1; }
  echo "$left"
}

stop_group() {
  local pid=$1 attempt
  [[ -n "$pid" ]] || return 0
  kill -TERM -- "-$pid" 2>/dev/null || true
  for ((attempt=0; attempt<15; attempt++)); do
    kill -0 -- "-$pid" 2>/dev/null || break
    sleep 1
  done
  # A session leader may exit before its workers; stop the whole owned group.
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup_capture() {
  local status=$?
  trap - EXIT
  trap '' INT TERM
  stop_group "$capture_client_pid"
  stop_group "$capture_server_pid"
  exit "$status"
}

run_capture() {
  local root=$1 published=$2 model=$3 client=$4 deadline=$5 port=$6 stop_at remaining
  # Convert the host deadline once. Bash's elapsed-seconds clock then covers
  # mounted-data preflight, server startup and replay; Docker supplies a hard cap.
  stop_at=$(( SECONDS + deadline - $(date +%s) ))
  capture_server_pid= capture_client_pid=
  trap cleanup_capture EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  remaining=$(remaining_seconds "$stop_at")
  set +m
  setsid timeout --foreground --signal=TERM --kill-after=5s "${remaining}s" \
    python3 "$client" capture-preflight --root "$root" --publish-root "$published" --model "$model" --port "$port" &
  capture_client_pid=$!
  wait "$capture_client_pid"
  stop_group "$capture_client_pid"
  capture_client_pid=

  unset NSYS_OUTPUT REQUEST_METRICS_DIR STORE_IMPLEMENTATION
  export TP=4 MAX_RUN=2 MEM_FRAC=0.88 CHUNK=8192 CTX_LEN=1048576 SERVER_SEED=42
  export DECODE_CFG='{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}'
  export REMNANT_CAPTURE_DIR="$root" SERVER_PID_FILE="$root/server.pid"
  export SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
  remaining=$(remaining_seconds "$stop_at")
  # No job control: setsid keeps each recorded PID as its session/group leader.
  setsid bash "$repo/scripts/local/run-server.sh" native "$model" 127.0.0.1 "$port" >"$root/server.log" 2>&1 &
  capture_server_pid=$!
  while true; do
    kill -0 "$capture_server_pid" 2>/dev/null || {
      echo "Native capture server exited; inspect $root/server.log" >&2; exit 1;
    }
    remaining=$(remaining_seconds "$stop_at")
    if curl --fail --silent --output /dev/null --max-time "$(( remaining < 3 ? remaining : 3 ))" \
      "http://127.0.0.1:$port/health"; then
      break
    fi
    remaining=$(remaining_seconds "$stop_at")
    sleep "$(( remaining < 2 ? remaining : 2 ))"
  done
  remaining=$(remaining_seconds "$stop_at")
  setsid timeout --foreground --signal=TERM --kill-after=5s "${remaining}s" \
    python3 "$client" capture-client --root "$root" --endpoint "http://127.0.0.1:$port" \
    --seconds "$remaining" --publish-root "$published" &
  capture_client_pid=$!
  wait "$capture_client_pid"
  exit 0
}

usage() {
  echo "usage: $0 run <image> <model> <scratch> <published> <seconds> [port]"
  echo "Requires a prepared local capture-only image and four allocated, idle GPUs."
}
[[ $# -gt 0 ]] || { usage >&2; exit 2; }
case "$1" in
  _preflight) [[ $# -eq 2 ]] || exit 2; preflight_image "$2" ;;
  _capture)
    [[ $# -eq 7 && "$6" =~ ^[0-9]{1,12}$ && "$7" =~ ^[0-9]{1,5}$ ]] || exit 2
    shift; run_capture "$@"
    ;;
  -h|--help) usage; exit 0 ;;
  build) echo 'Image setup is separate; this launcher never builds images.' >&2; exit 2 ;;
  run)
    if [[ "${2:-}" = --help || "${2:-}" = -h ]]; then usage; exit 0; fi
    ;;
  *) usage >&2; exit 2 ;;
esac
[[ $# -ge 6 && $# -le 7 ]] || { usage >&2; exit 2; }
shift
image=$1; model=$2; scratch=$3; published=$4; seconds=$5; port=${6:-30312}
[[ "$seconds" =~ ^[1-9][0-9]{0,8}$ ]] || { echo 'Positive bounded seconds required' >&2; exit 2; }
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || { echo 'Port must be in 1..65535' >&2; exit 2; }
for tool in docker nvidia-smi timeout; do
  command -v "$tool" >/dev/null || { echo "Required tool unavailable: $tool" >&2; exit 1; }
done
for path in "$model" "$scratch" "$published"; do
  [[ "$path" = /* && "$path" != / ]] || { echo 'Absolute, non-root directories required' >&2; exit 2; }
done
[[ -d "$model" && -f "$scratch/capture-config.json" && -f "$scratch/replay.jsonl" ]] || {
  echo 'Prepared model, capture-config.json and replay.jsonl required' >&2; exit 2;
}
mkdir -p -- "$published"
model=$(cd -- "$model" && pwd -P)
scratch=$(cd -- "$scratch" && pwd -P)
published=$(cd -- "$published" && pwd -P)
[[ "$model" != / && "$scratch" != / && "$published" != / && "$scratch" != "$published" ]] || {
  echo 'Resolved directories must be non-root; scratch and publication must differ' >&2; exit 2;
}
docker_timeout() {
  local limit=$1; shift
  timeout --foreground --signal=TERM --kill-after=5s "$limit" docker "$@"
}
metadata=$(docker_timeout 30s image inspect --format '{{.Id}} {{index .Config.Labels "remnant.capture-only"}}' "$image") || {
  echo "Capture image '$image' is unavailable locally; nothing will be built or pulled." >&2; exit 1;
}
read -r image_id capture_only <<< "$metadata"
[[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ && "$capture_only" = true ]] || {
  echo "Image '$image' must have remnant.capture-only=true" >&2; exit 1;
}
gpus=${CAPTURE_GPUS:-0,1,2,3}
IFS=, read -r -a gpu_ids <<< "$gpus"
[[ ${#gpu_ids[@]} = 4 && "$gpus" != *, ]] || { echo 'CAPTURE_GPUS must identify four distinct GPUs' >&2; exit 2; }
for ((i=0; i<4; i++)); do
  if [[ "${gpu_ids[i]}" =~ ^[0-9]{1,3}$ ]]; then
    gpu_ids[i]=$((10#${gpu_ids[i]}))
  else
    [[ "${gpu_ids[i]}" =~ ^GPU-[[:alnum:]-]+$ ]] || { echo 'Invalid explicit GPU ID' >&2; exit 2; }
  fi
  for ((j=0; j<i; j++)); do
    [[ "${gpu_ids[i]}" != "${gpu_ids[j]}" ]] || { echo 'Duplicate GPU IDs' >&2; exit 2; }
  done
done
gpus=$(IFS=,; echo "${gpu_ids[*]}")
owned=$(mktemp -d)
cleanup() {
  local status=$? cid_file cid stopped failed=0
  trap - EXIT
  trap '' INT TERM
  for cid_file in "$owned/preflight.cid" "$owned/capture.cid"; do
    [[ -s "$cid_file" ]] || continue
    read -r cid < "$cid_file" || true
    if [[ ! "$cid" =~ ^[0-9a-f]{64}$ ]]; then
      echo "Invalid owned container identity: $cid_file" >&2; failed=1; continue
    fi
    if ! stopped=$(docker_timeout 25s stop --time 15 "$cid" 2>&1); then
      if [[ "$stopped" != *"No such container"* ]]; then
        echo "Cannot stop owned capture container $cid: $stopped" >&2; failed=1
      fi
    fi
  done
  if (( failed )); then
    echo "Retained container identities for recovery: $owned" >&2
    status=1
  else
    rm -f -- "$owned/preflight.cid" "$owned/capture.cid"
    rmdir -- "$owned" || status=1
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
common=(--pull never --rm --network host
  --mount "type=bind,src=$scratch,dst=/capture"
  --mount "type=bind,src=$published,dst=/published"
  --mount "type=bind,src=$model,dst=/model,readonly"
  --mount "type=bind,src=$repo,dst=/capture-repo,readonly")
# Check the prepared image and actual mounts without exposing GPUs.
docker_timeout 60s run --name "remnant-capture-preflight-$$" --cidfile "$owned/preflight.cid" \
  "${common[@]}" --entrypoint bash "$image_id" /capture-repo/scripts/local/capture-latent.sh _preflight "$port"
occupied=$(timeout --foreground --signal=TERM --kill-after=5s 30s \
  nvidia-smi -i "$gpus" --query-compute-apps=pid --format=csv,noheader)
[[ -z "$occupied" ]] || { echo "Selected GPUs $gpus are occupied; no production process will be stopped" >&2; exit 1; }
# This deadline includes container startup, model loading, replay and writer drain.
# The container action subtracts startup time before invoking the replay client.
deadline=$(( $(date +%s) + seconds ))
echo "Capture mounts: $scratch -> $published; image: $image_id; GPUs: $gpus"
docker_timeout "${seconds}s" run --name "remnant-capture-$$" --cidfile "$owned/capture.cid" \
  "${common[@]}" --gpus "\"device=$gpus\"" --entrypoint bash "$image_id" \
  /capture-repo/scripts/local/capture-latent.sh _capture /capture /published /model "$runner" "$deadline" "$port"
