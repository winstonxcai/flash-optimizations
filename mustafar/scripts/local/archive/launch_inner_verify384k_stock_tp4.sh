#!/usr/bin/env bash
# Inner launcher (INSIDE ruler-eval) for the STOCK-native 384k shallow-concurrency
# verification point. Byte-identical to launch_inner_verify384k_tp4.sh's native leg
# EXCEPT no TopMag envs and PYTHONPATH without flash-optimizations (untouched stock
# 0731, dense 584-byte C4).
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is required: a 384k-context
# prefill OOMs on the default allocator. Pool expectation (expandable_segments):
#   3,610,880  -> C=9 at 393216+2048 (9*395264=3,557,376). Gate in the host driver.
# Port 30212 / master 29628, GPUs 4-7, decode CUDA graphs max_bs 136 (same extended
# config as the sweep legs).
set -u
EXPERT_MODE=${EXPERT_MODE:-native}   # native (fp4 mxfp4 cutlass) | dequant (fallback)
GPUS=${GPUS:-4,5,6,7}
CHUNK=${CHUNK:-8192}

GRAPH_CFG='{"decode":{"backend":"full","max_bs":136,"bs":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,24,28,32,34,40,48,56,64,68,80,96,112,120,136]},"prefill":{"backend":"disabled"}}'

PORT=30212
MPORT=29628
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_sweep_stock.log

export CUDA_VISIBLE_DEVICES=$GPUS
export MASTER_PORT=$MPORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
MOE_ARGS="--moe-runner-backend flashinfer_mxfp4"
if [ "$EXPERT_MODE" = dequant ]; then
  export SGLANG_DSV4_FP4_DEQUANT=1
  MOE_ARGS=""
fi

cd /sgl-workspace/sglang-lowrank/python
nohup python3 -m sglang.launch_server \
  --model-path /mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --tp 4 --trust-remote-code --mem-fraction-static 0.88 \
  --context-length 1048576 --max-running-requests 256 \
  --chunked-prefill-size $CHUNK \
  --kv-cache-dtype fp8_e4m3 $MOE_ARGS \
  --host 0.0.0.0 --port $PORT \
  --cuda-graph-config "$GRAPH_CFG" \
  --skip-server-warmup --watchdog-timeout 1800 \
  > "$LOG" 2>&1 &
echo "launched verify384k STOCK (expert=$EXPERT_MODE, chunk=$CHUNK, gpus=$GPUS) pid $! port $PORT log=$LOG"
