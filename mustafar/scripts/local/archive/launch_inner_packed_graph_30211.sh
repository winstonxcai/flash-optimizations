#!/usr/bin/env bash
# Inner launcher for the Stage-1 PACKED serving (report-comparable + live eval):
# decode CUDA graphs FULL (bs 1,2), prefill graphs disabled, ctx 196608 (fits
# eval requests up to ~164k input + 32k completion), fp8 triton.
# GPUs 4-7, port 30211. Runs INSIDE ruler-eval.
set -u
KEEP=${XKV_TOPMAG_KEEP:-0.5}
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_packed_graph_196k_30211.log
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29624
export SGLANG_OPT_TOPMAG=1
export XKV_TOPMAG_KEEP=$KEEP
export SGLANG_OPT_TOPMAG_PACKED_C4=1
export PYTHONPATH=/sgl-workspace/sglang-lowrank/python:/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
cd /sgl-workspace/sglang-lowrank/python
nohup python3 -m sglang.launch_server \
  --model-path /mnt/host_root/mnt/public_data/sgl-project/DeepSeek-V4-Flash-FP8 \
  --served-model-name deepseek-v4-flash \
  --tp 4 --trust-remote-code --mem-fraction-static 0.93 \
  --context-length 196608 --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --fp8-gemm-backend triton --host 0.0.0.0 --port 30211 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":2,"bs":[1,2]},"prefill":{"backend":"disabled"}}' \
  --skip-server-warmup --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --watchdog-timeout 1800 \
  > "$LOG" 2>&1 &
echo "launched packed-graph-serving pid $!"
