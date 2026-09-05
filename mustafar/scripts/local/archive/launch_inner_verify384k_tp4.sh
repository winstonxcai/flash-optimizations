#!/usr/bin/env bash
# Inner launcher (INSIDE ruler-eval) for the 384k shallow-concurrency verification.
# Byte-identical to launch_inner_serve_sweep_fp4_tp4.sh EXCEPT:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (fixes the cross-allocator
#     CUDA OOM seen when prefilling 384k-context requests: a 2.38 GiB TVM-side
#     malloc failed with only 1.94 GiB driver-free while PyTorch held 2.87 GiB
#     reserved-but-unallocated. expandable_segments returns it to the driver.)
#   - chunked-prefill-size configurable via CHUNK (default 8192 to match the report
#     rows; drop to 4096/2048 only if the workspace still does not fit).
#
# Same pool expectation must hold (native 3730944 / packed 4519168): the fork sizes
# max_total_num_tokens from the mem-frac budget, so expandable_segments must NOT move
# it -- the verify driver gates on it.
#
#   LEG=native -> 30212 / 29628 ; LEG=packed -> 30211 / 29626 (GPUs 4-7)
set -u
LEG=${LEG:-native}          # native | packed
EXPERT_MODE=${EXPERT_MODE:-native}   # native (fp4 mxfp4 cutlass) | dequant (fallback)
GPUS=${GPUS:-4,5,6,7}
CHUNK=${CHUNK:-8192}

GRAPH_CFG='{"decode":{"backend":"full","max_bs":136,"bs":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,24,28,32,34,40,48,56,64,68,80,96,112,120,136]},"prefill":{"backend":"disabled"}}'

if [ "$LEG" = packed ]; then
  PORT=30211; MPORT=29626; PACKED=1
else
  PORT=30212; MPORT=29628; PACKED=0
fi
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_sweep_${LEG}.log

export CUDA_VISIBLE_DEVICES=$GPUS
export MASTER_PORT=$MPORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_OPT_TOPMAG=1
export XKV_TOPMAG_KEEP=0.5
export SGLANG_OPT_TOPMAG_PACKED_C4=$PACKED
export PYTHONPATH=/sgl-workspace/sglang-lowrank/python:/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations
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
echo "launched verify384k $LEG (packed_c4=$PACKED, expert=$EXPERT_MODE, chunk=$CHUNK, gpus=$GPUS) pid $! port $PORT log=$LOG"
