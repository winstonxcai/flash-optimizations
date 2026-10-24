#!/usr/bin/env bash
# One owned server per replay sweep. Author: Winston Cai.
set -euo pipefail
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Help must not require Docker, data or a configured serving environment.
for arg in "$@"; do
  if [ "$arg" = --help ] || [ "$arg" = -h ]; then
    exec "${PYTHON:-python3}" "$DIR/business_replay/sweep.py" --help
  fi
done
# Match serve.sh's extended graph default instead of env.sh's small graph set.
DECODE_CFG_WAS_SET=${DECODE_CFG+x}
. "$DIR/env.sh"
. "$DIR/eval-env.sh"
[ -n "$DECODE_CFG_WAS_SET" ] || DECODE_CFG=$DECODE_CFG_EXT
export HOST_REPO REPO_CT CONTAINER SGLANG_ROOT_CT SGLANG_PY MODEL_CT MODEL_NAME
export RESULTS_HOST PORT MASTER_PORT TP MEM_FRAC CTX_LEN MAX_RUN CHUNK DECODE_CFG
export REPLAY_RUNNER REPLAY_ADAPTER REPLAY_DATASET REPLAY_MANIFEST
export REMNANT_INTEGRATION=${REMNANT_INTEGRATION:-v0.5.18}
exec "${PYTHON:-python3}" "$DIR/business_replay/sweep.py" "$@"
