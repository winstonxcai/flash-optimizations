#!/usr/bin/env bash
# Inner launcher for the NATIVE (untouched, out-of-the-box csa) 25-new-task rerun at
# --context-length 262144 on GPUs 4-7. Runs INSIDE ruler-eval. Port 30211
# (docker_env_config_dsv4-windowed points there). Native = SGLANG_OPT_TOPMAG=0,
# XKV_TOPMAG_KEEP=1.0, no SGLANG_OPT_TOPMAG_PACKED_C4 — the MUSTAFAR patch is env-gated
# and every zeroing/packed branch falls through to stock. MASTER_PORT 29624 distinct
# from server B's 29623.
set -u
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_native25A_262k.log
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29624
export SGLANG_OPT_TOPMAG=0
export XKV_TOPMAG_KEEP=1.0
export XKV_DEBUG=0
export SG_CTRL_DIR=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/ctrl
export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
cd /sgl-workspace/sglang-lowrank/python
nohup python3 -m sglang.launch_server \
  --model-path /mnt/host_root/mnt/public_data/sgl-project/DeepSeek-V4-Flash-FP8 \
  --served-model-name deepseek-v4-flash \
  --tp 4 --trust-remote-code --mem-fraction-static 0.93 \
  --context-length 262144 --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --fp8-gemm-backend triton --host 0.0.0.0 --port 30211 \
  --disable-cuda-graph --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  > "$LOG" 2>&1 &
echo "launched native25A-262k pid $!"
