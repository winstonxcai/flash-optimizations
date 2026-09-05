#!/usr/bin/env bash
# Inner launcher (INSIDE ruler-eval) for the LongBench v2 full NATIVE leg on the
# 0731 checkpoint, TRUE stock (TopMag OFF, dense 584-byte C4) -- the exact mirror
# of launch_inner_sangfor_packed_0731_tp4.sh with only the three TopMag envs
# removed, so Native and Packed differ solely in the C4 treatment. TP4 on GPUs
# 4-7. DeepSeek reasoning/tool parsers are ON (a quality eval, not a token-count
# parity bench): the model must emit structured content, same as the packed leg.
#
# fp4-native MoE runner (flashinfer_mxfp4), mem-frac 0.88, ctx 1M, max_running
# 256, chunked_prefill 8192, kv fp8_e4m3, decode CUDA graphs bs[1..15] max_bs 15
# (matches the packed leg: LB2 concurrency is ~6), expandable_segments retained
# for 1M-ctx prefills, watchdog 1800. Radix/prefix cache stays on.
set -u
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-30212}
MPORT=${MPORT:-29628}
LOG=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_lb2_native_0731.log
export CUDA_VISIBLE_DEVICES=$GPUS
export MASTER_PORT=$MPORT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NO SGLANG_OPT_TOPMAG / XKV_TOPMAG_KEEP / SGLANG_OPT_TOPMAG_PACKED_C4 -- native stock
export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL NCCL_PROTO=Simple NCCL_ALGO=Ring
cd /sgl-workspace/sglang-lowrank/python
nohup python3 -m sglang.launch_server \
  --model-path /mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash \
  --tp 4 --trust-remote-code --mem-fraction-static 0.88 \
  --context-length 1048576 --max-running-requests 256 \
  --chunked-prefill-size 8192 \
  --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4 \
  --reasoning-parser deepseek-v4 --tool-call-parser deepseekv4 \
  --host 0.0.0.0 --port $PORT \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":15,"bs":[1,2,3,4,5,6,7,8,10,12,14,15]},"prefill":{"backend":"disabled"}}' \
  --skip-server-warmup --watchdog-timeout 1800 \
  > "$LOG" 2>&1 &
echo "launched LB2-native-0731 STOCK (gpus=$GPUS, port=$PORT) pid $! log=$LOG"
