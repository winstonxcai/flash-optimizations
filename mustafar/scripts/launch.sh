#!/usr/bin/env bash
# Launch port 30211 with dense-zero TopMag50 in the native 584-byte C4 pool.
# Set SGLANG_OPT_TOPMAG_PACKED_C4=1 to launch the packed Stage-1 path instead.
# Requires: `python -m mustafar patch` applied to the active SGLang source.
set -u
KEEP=${XKV_TOPMAG_KEEP:-0.5}
PACKED=${SGLANG_OPT_TOPMAG_PACKED_C4:-0}
# Host-side paths resolve against the live repo on THIS host. Inside the
# container the same files appear under /mnt/host_root.
HOST_BASE=/home/jovyan/winstonxcai/flash-optimizations/mustafar
BASE=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar
LOG=$BASE/logs/serve_topmag.log
mkdir -p "$HOST_BASE/logs"

ps -eo pid,args | grep "[s]glang.launch_server" | grep "30211" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 3

docker exec ruler-eval bash -c "
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29622
export SGLANG_OPT_TOPMAG=1
export XKV_TOPMAG_KEEP=$KEEP
export SGLANG_OPT_TOPMAG_PACKED_C4=$PACKED
export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
cd /sgl-workspace/sglang-lowrank/python
nohup python3 -m sglang.launch_server \
  --model-path /mnt/host_root/mnt/public_data/sgl-project/DeepSeek-V4-Flash-FP8 \
  --served-model-name deepseek-v4-flash \
  --tp 4 --trust-remote-code --mem-fraction-static 0.93 \
  --context-length 135168 --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --fp8-gemm-backend triton --host 0.0.0.0 --port 30211 \
  --disable-cuda-graph --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 > '$LOG' 2>&1 &
echo 'launched pid '\\$!
"
for i in $(seq 1 72); do
  if docker exec ruler-eval bash -c "grep -q 'is fired up and ready' '$LOG'" 2>/dev/null; then
    echo "server ready in ~$((i*5))s"; break
  fi
  sleep 5
done
tail -3 "$LOG" 2>/dev/null
