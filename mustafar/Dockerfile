# TopMag 50% store-time pruning on the native c4 latent — reproducible server image
#
# Build:
#   cd flash-optimizations
#   docker build -t topmag-server:v0.5.15 -f mustafar/Dockerfile .
#
# Run (any H100 host with >=4 GPUs + the fp8 weights on disk):
#   docker run --gpus all -p 30211:30211 \
#     -v /path/to/DeepSeek-V4-Flash-FP8:/model \
#     -e MODEL_PATH=/model \
#     -e TP=4 -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e MASTER_PORT=29622 \
#     -e SGLANG_OPT_TOPMAG=1 -e XKV_TOPMAG_KEEP=0.5 -e XKV_DEBUG=0 \
#     topmag-server:v0.5.15
#
# The weights (274 GB fp8) are intentionally NOT baked in; they are mounted.
# Base image ships the sglang source at the exact commit the server ran
# (/sgl-workspace/sglang, f63458b, incl. dsv4/), so the build is offline-safe.
FROM lmsysorg/sglang:v0.5.15-cu130

# 1. Reproduce the active source tree (sglang-lowrank) as a local clone of the
#    base image's own /sgl-workspace/sglang at the same commit the server used.
RUN git clone /sgl-workspace/sglang /sgl-workspace/sglang-lowrank \
    && cd /sgl-workspace/sglang-lowrank && git checkout f63458b

# 2. Ship the mustafar package. PACKAGE_ROOT resolves to the parent dir of the
#    mustafar/ folder, so the sglang hook's `import mustafar` finds it without
#    an XKV_PACKAGE_DIR override.
COPY mustafar/ /opt/mustafar/flash-optimizations/mustafar/

# 3. Bake in the TopMag hook into the ACTIVE tree (PYTHONPATH-pinned at run),
#    then verify the marker is present.
ENV SG_LOWRANK_SRC=/sgl-workspace/sglang-lowrank/python
RUN cd /opt/mustafar/flash-optimizations \
    && python3 -m mustafar patch \
    && python3 -m mustafar verify

# 4. Store-time TopMag on by default; config via env at run time. NCCL
#    settings reproduce the tested single-host 4-GPU run (override at run
#    time with -e if the target host uses InfiniBand / different fabrics).
ENV SGLANG_OPT_TOPMAG=1 \
    XKV_TOPMAG_KEEP=0.5 \
    XKV_DEBUG=0 \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=lo \
    NCCL_P2P_LEVEL=NVL \
    NCCL_PROTO=Simple \
    NCCL_ALGO=Ring

# 5. Launch the server. Mirrors flash-optimizations/mustafar/scripts/launch.sh
#    (same flags; decode ~2-4 tok/s on long agent contexts by design:
#    --fp8-gemm-backend triton + --disable-cuda-graph).
CMD ["bash","-c", "cd /sgl-workspace/sglang-lowrank/python && \
  exec python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH:?} \
    --served-model-name deepseek-v4-flash \
    --tp ${TP:-4} --trust-remote-code --mem-fraction-static 0.93 \
    --context-length 135168 --max-running-requests 64 \
    --chunked-prefill-size 4096 \
    --fp8-gemm-backend triton --host 0.0.0.0 --port ${PORT:-30211} \
    --disable-cuda-graph --reasoning-parser deepseek-v4 \
    --tool-call-parser deepseekv4"]
