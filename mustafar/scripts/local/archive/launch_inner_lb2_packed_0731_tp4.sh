#!/usr/bin/env bash
# Inner launcher (INSIDE ruler-eval) for the Sangfor-Bench hard50 rerun on the
# 0731 checkpoint, PACKED 328-B leg (TopMag50). TP4 on GPUs 4-7.
#
# This is launch_inner_packed_fp4_tp4_lswb_30211.sh with exactly three deltas for
# a LIVE agentic eval (Sangfor), per the serving-config review:
#   1. + parsers: --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4
#      (Sangfor agents must receive structured tool_calls; LSWB/serving ran no
#      parsers only for token-count parity, which does NOT apply here).
#   2. + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (1M ctx cap: a single
#      very large prefill OOMs the default allocator, as the 384k verify showed;
#      costs ~120k pool slots, irrelevant for a task-level eval).
#   3. decode CUDA graphs UNCHANGED from LSWB: full bs[1..15] max_bs 15, prefill
#      off. 8 workers = <=8 concurrent decode streams, all on-graph, ~0 pad waste.
#
# Everything else matches the validated 0731 fp4-native stack (mem-frac 0.88, ctx
# 1M, max_running 256, chunked_prefill 8192, kv fp8_e4m3, flashinfer_mxfp4, no
# cache-report, no DSPARK, skip-warmup, watchdog 1800). Radix/prefix cache stays on
# (default) -- agentic turns re-send the conversation, prefix reuse is the point.
#
# GPUs 4-7 default (the stock-sweep slot, freed once run_serve_sweep_stock.sh exits
# and kills its server on 30212). EXPERT_MODE=native (fp4-native mxfp4 cutlass,
# validated with packed C4 in the LSWB replay); dequant fallback kept for parity.
set -u
EXPERT_MODE=${EXPERT_MODE:-native}   # native | dequant
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-30212}
MPORT=${MPORT:-29628}
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_lb2_packed_0731.log
export CUDA_VISIBLE_DEVICES=$GPUS
export MASTER_PORT=$MPORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_OPT_TOPMAG=1
export XKV_TOPMAG_KEEP=0.5
export SGLANG_OPT_TOPMAG_PACKED_C4=1
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
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --host 0.0.0.0 --port $PORT \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":15,"bs":[1,2,3,4,5,6,7,8,10,12,14,15]},"prefill":{"backend":"disabled"}}' \
  --skip-server-warmup --watchdog-timeout 1800 \
  > "$LOG" 2>&1 &
echo "launched sangfor-0731 PACKED (${EXPERT_MODE}, gpus=$GPUS, port=$PORT) pid $! log=$LOG"
