#!/usr/bin/env bash
# Launch 30211 with store-time TopMag pruning on the NATIVE c4 latent.
#
# NO lowrank KV: the pool stays the native 584 B/token c4 layout, the memory
# pool and decode are stock DeepSeek-V4. The only change is the injected
# `_sg_lr.maybe_prune(kv_compressed)` before the fused native store.
# Requires: `python -m mustafar patch` applied to the active sglang source.
set -u
KEEP=${XKV_TOPMAG_KEEP:-0.5}
# Host-side paths resolve against the live repo on THIS host. Inside the
# container the same files appear under /mnt/host_root (a different mount on
# the host, so never use it for host-side mkdir/rm).
HOST_BASE=/home/jovyan/winstonxcai/flash-optimizations/mustafar
BASE=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar
LOG=$BASE/logs/serve_topmag.log
mkdir -p "$HOST_BASE/logs" "$HOST_BASE/ctrl"
rm -f "$HOST_BASE/ctrl/debug.log"

ps -eo pid,args | grep "[s]glang.launch_server" | grep "30211" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 3

docker exec ruler-eval bash -c "
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29622
export SGLANG_OPT_TOPMAG=1
export XKV_TOPMAG_KEEP=$KEEP
# XKV_DEBUG=0 for long runs: =1 grows ctrl/debug.log ~380 lines/s (~1 GB/12 h)
# and throttles decode. Pruning itself is unaffected (the _dbg guard short-circuits).
export XKV_DEBUG=0
export SG_CTRL_DIR=$BASE/ctrl
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
echo 'launched pid '\\\$!
"
for i in $(seq 1 72); do
  if docker exec ruler-eval bash -c "grep -q 'is fired up and ready' '$LOG'" 2>/dev/null; then
    echo "server ready in ~$((i*5))s"; break
  fi
  sleep 5
done
tail -3 "$LOG" 2>/dev/null
