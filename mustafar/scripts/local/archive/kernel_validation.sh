#!/usr/bin/env bash
# Local replica of modal app.py::kernel_validation — compile and validate the
# Stage-1 Triton kernels on one H100.
#
#   Usage: ./kernel_validation.sh [GPU]      (default GPU 0)
#   Requires: `python -m mustafar patch` applied to /sgl-workspace/sglang-lowrank.
#   Output:  mustafar/results/kernel-validation.json (+ .stdout log)
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
GPU=${1:-0}
export GPUS=$GPU
. "$DIR/common.sh"

OUTLOG="$RESULTS_HOST/kernel_validation.stdout"
echo "== gpu_packed on H100:$GPU (TopMag packed C4)"
docker exec ruler-eval bash -c "
  export CUDA_VISIBLE_DEVICES=$GPU
  export SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=$KEEP SGLANG_OPT_TOPMAG_PACKED_C4=1
  export PYTHONPATH=$PYTHONPATH
  cd $REPO_CT
  python3 -m mustafar.tests.gpu_packed
" 2>&1 | tee "$OUTLOG"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
  echo "kernel validation FAILED rc=$RC (see $OUTLOG)" >&2
  exit 1
fi
tail -1 "$OUTLOG" > "$RESULTS_HOST/kernel-validation.json"
echo "OK -> $RESULTS_HOST/kernel-validation.json"
