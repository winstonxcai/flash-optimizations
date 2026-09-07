#!/usr/bin/env bash
# Inner launcher (INSIDE ruler-eval) for the STAGE-1 PACKED TP4 leg of the
# LongCodeBench business-replay run, on the OFFICIAL 0731 checkpoint.
#
# Expert path is switchable for the Step-0 boot gate:
#   EXPERT_MODE=native  (default) -> --moe-runner-backend flashinfer_mxfp4, NO
#     SGLANG_DSV4_FP4_DEQUANT. fp4-packed experts run through
#     Mxfp4FlashinferCutlassMoEMethod (SM90/Hopper cutlass path), matching the
#     official baseline's own MoE kernel path. UNVALIDATED with packed C4.
#   EXPERT_MODE=dequant          -> SGLANG_DSV4_FP4_DEQUANT=1, drop the runner
#     flag (auto -> fp8-triton). Proven path from the TP8 legs (fp4->fp8 at load,
#     value-preserving). ~2x expert weight memory vs native.
# THE COMPARISON ISOLATES PACKING ONLY IF BOTH LEGS SHARE THE SAME EXPERT MODE.
#
# TopMag50 packed C4 on the 21 CSA layers (SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=0.5
# SGLANG_OPT_TOPMAG_PACKED_C4=1). Serve flags mirror the official device baseline
# (mem-frac 0.88, ctx 1M, max_running_requests 256, chunked_prefill 8192, kv
# fp8_e4m3, decode CUDA graphs bs[1..15] max_bs 15, no reasoning/tool parser, no
# enable-cache-report) MINUS DSPARK (packed-C4 guard forbids speculative decode)
# and minus the upstream engine. GPUs 0,1,2,3, port 30211, MASTER_PORT 29626.
set -u
EXPERT_MODE=${EXPERT_MODE:-native}   # native | dequant
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_lswb_replay_packed.log
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29626
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
  --host 0.0.0.0 --port 30211 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":15,"bs":[1,2,3,4,5,6,7,8,10,12,14,15]},"prefill":{"backend":"disabled"}}' \
  --skip-server-warmup --watchdog-timeout 1800 \
  > "$LOG" 2>&1 &
echo "launched lswb-replay packed (${EXPERT_MODE}) pid $! log=$LOG"
