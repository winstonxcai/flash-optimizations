#!/usr/bin/env bash
# Inner launcher for Stage-1 PACKED serving on TP8 with 500k-class context.
# decode CUDA graphs FULL capturing batch sizes 1,2,4,8 (max_bs 8) so that
# 4-8 concurrent deepswe requests all decode on-graph (~130 tok/s/stream)
# instead of falling off to eager (~7 tok/s). prefill graphs disabled,
# ctx 524288 (512k), fp8 triton. GPUs 0-7, port 30211. INSIDE ruler-eval.
set -u
KEEP=${XKV_TOPMAG_KEEP:-0.5}
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_packed_graph_tp8_500k_bs8_30211.log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29626
export SGLANG_OPT_TOPMAG=1
export XKV_TOPMAG_KEEP=$KEEP
export SGLANG_OPT_TOPMAG_PACKED_C4=1
export PYTHONPATH=/sgl-workspace/sglang-lowrank/python:/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
cd /sgl-workspace/sglang-lowrank/python
nohup python3 -m sglang.launch_server \
  --model-path /mnt/host_root/mnt/public_data/sgl-project/DeepSeek-V4-Flash-FP8 \
  --served-model-name deepseek-v4-flash \
  --tp 8 --trust-remote-code --mem-fraction-static 0.93 \
  --context-length 524288 --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --fp8-gemm-backend triton --host 0.0.0.0 --port 30211 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":8,"bs":[1,2,4,8]},"prefill":{"backend":"disabled"}}' \
  --skip-server-warmup --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --watchdog-timeout 1800 \
  > "$LOG" 2>&1 &
echo "launched packed-graph tp8 bs8 pid $!"
