#!/usr/bin/env bash
# Optional paths and connection settings for external evaluations.
#
# This file is intentionally separate from env.sh. Serving and kernel scripts
# should not require remote benchmark infrastructure to be configured.
set -u

EVAL_SSH=${EVAL_SSH:-}
EVAL_SCP=${EVAL_SCP:-}
EVAL_YJY=${EVAL_YJY:-}
EVAL_VENV=${EVAL_VENV:-}
EVAL_CFG=${EVAL_CFG:-}

REPLAY_DIR=${REPLAY_DIR:-}
if [ -n "$REPLAY_DIR" ]; then
  REPLAY_RUNNER=${REPLAY_RUNNER:-$REPLAY_DIR/harnesses/business_replay/runner.py}
  REPLAY_ADAPTER=${REPLAY_ADAPTER:-$REPLAY_DIR/harnesses/business_replay/adapters/openai_sse.py}
  REPLAY_DATASET=${REPLAY_DATASET:-$REPLAY_DIR/datasets/h20-dsv4pro/longcodebench_openai}
  REPLAY_MANIFEST=${REPLAY_MANIFEST:-$REPLAY_DIR/cache/official-longswebench/longswebench-openai-v1.json}
fi

require_remote_eval_config() {
  : "${EVAL_SSH:?Set EVAL_SSH to the remote SSH command}"
  : "${EVAL_YJY:?Set EVAL_YJY to the remote YJYBench directory}"
  : "${EVAL_VENV:?Set EVAL_VENV to the remote evaluation Python}"
  : "${EVAL_CFG:?Set EVAL_CFG to the remote evaluation config}"
}

require_replay_config() {
  : "${REPLAY_DIR:?Set REPLAY_DIR to the LongSWE-Bench client directory}"
  : "${REPLAY_RUNNER:?Set REPLAY_RUNNER to the replay runner}"
  : "${REPLAY_ADAPTER:?Set REPLAY_ADAPTER to the replay adapter}"
  : "${REPLAY_DATASET:?Set REPLAY_DATASET to the replay dataset}"
  : "${REPLAY_MANIFEST:?Set REPLAY_MANIFEST to the replay manifest}"
}
