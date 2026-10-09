#!/usr/bin/env bash
# Isolated microbenchmark (no nsys/ncu): parity + alternating complete-call
# native/direct decode CUDA graphs, including combine. Reuses the h64/b32 fixture
# and flashmla harness. Launches a fresh child of pilot.py per mode swap.
set -u
CT=remnant-pilot-20261009
SRC=/home/jovyan/winstonxcai/flash-optimizations
echo "=== 1) swap in the lineinfo artifact; SAVE the exact release binary ==="
docker exec "$CT" bash -lc '
set -eu
R=/usr/local/lib/python3.12/dist-packages/sgl_kernel/flashmla_ops.cpython-312-x86_64-linux-gnu.so
if [ ! -f "$R.release" ]; then cp -f "$R" "$R.release"; echo "saved release -> $R.release"; fi
cp -f /tmp/flashmla_ops.lineinfo.so "$R"
echo "resolved:"; python3 -c "import sgl_kernel.flashmla_ops as e; print(e.__file__)"
sha256sum "$R" "$R.release"
'
echo "=== 2) copy fixture + scripts into container ==="
docker cp "$SRC/third_party/flashmla/tests/." "$CT:/tmp/pilot/"
docker cp "$SRC/results/lineinfo/pilot/pilot.py" "$CT:/tmp/pilot/pilot.py"
docker cp "$SRC/results/lineinfo/pilot/mb_parity_timing.py" "$CT:/tmp/pilot/mb_parity_timing.py"
echo "=== 3) run isolated microbenchmark ==="
docker exec --privileged "$CT" bash -lc '
set -eu
export PILOT_FIXTURE_DIR=/tmp/pilot
export PYTHONPATH=/tmp/pilot
export CUDA_VISIBLE_DEVICES=0
export PILOT_LINEINFO_SHA=b44d6826065aaa5aa5d2526ded8e299b3b86f9cc915bfc6f81739c432310e5b3
python3 /tmp/pilot/mb_parity_timing.py
'
RC=$?
echo "=== 4) restore the release binary ==="
docker exec "$CT" bash -lc '
R=/usr/local/lib/python3.12/dist-packages/sgl_kernel/flashmla_ops.cpython-312-x86_64-linux-gnu.so
cp -f "$R.release" "$R"; sha256sum "$R"
'
echo "MB_RC=$RC"
