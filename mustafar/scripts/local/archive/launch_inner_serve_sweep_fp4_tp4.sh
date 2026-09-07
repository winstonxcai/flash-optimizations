#!/usr/bin/env bash
# Inner launcher (INSIDE ruler-eval) for the fp4-native serving-capacity sweep
# (results land in mustafar/results/serve-sweep-fp4/). Boots ONE TP4 leg of the
# Stage-1 packed DeepSeek-V4-Flash-0731 stack on host GPUs $GPUS with the EXTENDED
# decode-CUDA-graph config; the host driver then runs official sglang.bench_serving
# at the allocator-ceiling concurrency points (32k/64k/128k/256k, output 2048).
#
#   LEG=native   TopMag50 pruning, NATIVE 584-byte C4 store (PACKED_C4=0) -> 30212 / 29628
#   LEG=packed   TopMag50 pruning, PACKED 328-byte C4 store  (PACKED_C4=1) -> 30211 / 29626
#
# Only PACKED_C4 differs between legs (packing-only delta). Serve flags mirror the
# LSWB fp4-native baseline (mem-frac 0.88, ctx 1M, max_running 256, chunked 8192,
# kv fp8_e4m3, --moe-runner-backend flashinfer_mxfp4, no parsers, no cache-report,
# no DSPARK). Decode CUDA graphs are EXTENDED from LSWB's bs[1..15]/max_bs 15 to
# sparse buckets through max_bs 136 so decode stays on-graph at the pool-derived
# concurrency (native C107 / packed C129 resident at 32k). Pool is sized BEFORE
# graph capture in the fork, so a larger max_bs does not shrink max_total_num_tokens.
set -u
LEG=${LEG:-native}          # native | packed
EXPERT_MODE=${EXPERT_MODE:-native}   # native (fp4 mxfp4 cutlass) | dequant (fallback)
GPUS=${GPUS:-4,5,6,7}

GRAPH_CFG='{"decode":{"backend":"full","max_bs":136,"bs":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,24,28,32,34,40,48,56,64,68,80,96,112,120,136]},"prefill":{"backend":"disabled"}}'

if [ "$LEG" = packed ]; then
  PORT=30211; MPORT=29626; PACKED=1
else
  PORT=30212; MPORT=29628; PACKED=0
fi
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_sweep_${LEG}.log

export CUDA_VISIBLE_DEVICES=$GPUS
export MASTER_PORT=$MPORT
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
  --chunked-prefill-size 8192 \
  --kv-cache-dtype fp8_e4m3 $MOE_ARGS \
  --host 0.0.0.0 --port $PORT \
  --cuda-graph-config "$GRAPH_CFG" \
  --skip-server-warmup --watchdog-timeout 1800 \
  > "$LOG" 2>&1 &
echo "launched serve-sweep $LEG (packed_c4=$PACKED, expert=$EXPERT_MODE, gpus=$GPUS) pid $! port $PORT log=$LOG"
