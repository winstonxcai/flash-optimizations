#!/usr/bin/env bash
# Local replica of modal app.py::microbenchmark — packed C4 pack/gather/unpack
# microbenchmark matrix (batch x top-k) on one H100.
#
#   Usage: ./microbenchmark.sh [GPU]          (default GPU 0)
#   Requires: `python -m mustafar patch` applied to /sgl-workspace/sglang-lowrank.
#   Output:  mustafar/results/microbenchmark.json + packed-c4-microbench.{json,csv}
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
GPU=${1:-0}
export GPUS=$GPU
. "$DIR/common.sh"

OUTLOG="$RESULTS_HOST/microbenchmark.stdout"
echo "== bench_packed on H100:$GPU (TopMag packed C4)"
docker exec ruler-eval bash -c "
  export CUDA_VISIBLE_DEVICES=$GPU
  export SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=$KEEP SGLANG_OPT_TOPMAG_PACKED_C4=1
  export PYTHONPATH=$PYTHONPATH
  export MUSTAFAR_RESULTS_DIR=$RESULTS_CT
  cd $REPO_CT
  python3 -m mustafar.tests.bench_packed
" 2>&1 | tee "$OUTLOG"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then
  echo "microbenchmark FAILED rc=$RC (see $OUTLOG)" >&2
  exit 1
fi
tail -1 "$OUTLOG" > "$RESULTS_HOST/microbenchmark.json"
echo "OK -> $RESULTS_HOST/microbenchmark.json"
echo "     -> $RESULTS_HOST/packed-c4-microbench.{json,csv}"
