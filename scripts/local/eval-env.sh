#!/usr/bin/env bash
# Local replay paths and optional connection settings for external evaluations.
# Author: Winston Cai.
#
# This file is intentionally separate from env.sh. Serving and kernel scripts
# should not require remote benchmark infrastructure to be configured.
set -u

EVAL_SSH=${EVAL_SSH:-}
EVAL_SCP=${EVAL_SCP:-}
EVAL_YJY=${EVAL_YJY:-}
EVAL_VENV=${EVAL_VENV:-}
EVAL_CFG=${EVAL_CFG:-}

# Code defaults belong to this checkout, never to the optional legacy data root.
REPLAY_LOCAL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPLAY_RUNNER=${REPLAY_RUNNER:-$REPLAY_LOCAL_DIR/business_replay/runner.py}
REPLAY_ADAPTER=${REPLAY_ADAPTER:-$REPLAY_LOCAL_DIR/business_replay/adapters/openai_sse.py}
REPLAY_DIR=${REPLAY_DIR:-}
if [ -n "$REPLAY_DIR" ]; then
  # Anchor a relative legacy root before removing the launcher's old cd.
  [[ "$REPLAY_DIR" = /* ]] || REPLAY_DIR="$PWD/$REPLAY_DIR"
  REPLAY_DATASET=${REPLAY_DATASET:-$REPLAY_DIR/datasets/h20-dsv4pro/longcodebench_openai}
  REPLAY_MANIFEST=${REPLAY_MANIFEST:-$REPLAY_DIR/cache/official-longswebench/longswebench-openai-v1.json}
fi
REPLAY_DATASET=${REPLAY_DATASET:-$REPLAY_LOCAL_DIR/../../data/lswb-replay/records}
REPLAY_MANIFEST=${REPLAY_MANIFEST:-$REPLAY_LOCAL_DIR/../../data/lswb-replay/manifest.json}

require_remote_eval_config() {
  : "${EVAL_SSH:?Set EVAL_SSH to the remote SSH command}"
  : "${EVAL_YJY:?Set EVAL_YJY to the remote YJYBench directory}"
  : "${EVAL_VENV:?Set EVAL_VENV to the remote evaluation Python}"
  : "${EVAL_CFG:?Set EVAL_CFG to the remote evaluation config}"
}

require_replay_config() {
  local name path base=${REPLAY_DIR:-$PWD}
  for name in REPLAY_RUNNER REPLAY_ADAPTER REPLAY_DATASET REPLAY_MANIFEST; do
    path=${!name}
    if [ -z "$path" ]; then
      printf 'FATAL: Set %s to the replay %s path (or set REPLAY_DIR for legacy dataset/manifest defaults).\n' \
        "$name" "${name#REPLAY_}" >&2
      return 1
    fi
    # Previously the client ran inside REPLAY_DIR. Preserve relative override
    # semantics there; without it, use the caller's working directory.
    [[ "$path" = /* ]] || path="$base/$path"
    printf -v "$name" '%s' "$path"
    if [ "$name" = REPLAY_DATASET ]; then
      if [ ! -d "$path" ] || [ ! -r "$path" ] || [ ! -x "$path" ]; then
        printf 'FATAL: REPLAY_DATASET must be a readable/searchable directory of replay JSON records: %s. Prepare data/longswebench.jsonl with convert_lswb.py --format replay or set REPLAY_DATASET.\n' "$path" >&2
        return 1
      fi
    elif [ ! -f "$path" ] || [ ! -r "$path" ]; then
      printf 'FATAL: %s must be a readable file: %s. Set %s to an existing %s file.\n' \
        "$name" "$path" "$name" "${name#REPLAY_}" >&2
      return 1
    fi
  done
}
