#!/usr/bin/env bash
# Inner launcher for the UNTOUCHED baseline on the OFFICIAL 0731 checkpoint,
# matched to the packed Stage-1 leg so the comparison isolates packing.
# Serving config, fp4 handling, and MoE path are IDENTICAL to the packed server
# (official 0731, SGLANG_DSV4_FP4_DEQUANT=1 -> fp4->fp8 at load, value-preserving,
# fp8-triton MoE, tp8, ctx 524288, decode CUDA graphs full bs[1,2,4,8], prefill
# disabled, mem-fraction 0.93). The ONLY difference from the packed leg is that the
# three Mustafar env vars are ABSENT: SGLANG_OPT_TOPMAG, XKV_TOPMAG_KEEP,
# SGLANG_OPT_TOPMAG_PACKED_C4. This gives a fair apples-to-apples untouched baseline:
# expert math runs on the same fp8-triton path as the packed leg, so any quality or
# capacity delta is attributable to TopMag50 packing alone.
# GPUs 0-7, port 30211. INSIDE ruler-eval.
set -u
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_untouched_0731_dequant_tp8_500k_bs8_30211.log
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29628
export SGLANG_DSV4_FP4_DEQUANT=1
export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
cd /sgl-workspace/sglang-lowrank/python
nohup python3 -m sglang.launch_server \
  --model-path /mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --tp 8 --trust-remote-code --mem-fraction-static 0.93 \
  --context-length 524288 --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --fp8-gemm-backend triton --host 0.0.0.0 --port 30211 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":8,"bs":[1,2,4,8]},"prefill":{"backend":"disabled"}}' \
  --skip-server-warmup --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --watchdog-timeout 1800 \
  > "$LOG" 2>&1 &
echo "launched untouched 0731 dequant tp8 bs8 pid $!"
