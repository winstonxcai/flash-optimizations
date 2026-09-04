# Packed TopMag50 cache — DeepSeek-V4-Flash-0731 server image
#
# Build:
#   cd flash-optimizations
#   docker build -t topmag-server:v0.5.17-0731 -f mustafar/Dockerfile .
#
# Open a shell (any H100 host with >=4 GPUs + the 0731 weights on disk):
#   docker run --rm -it --gpus all \
#     -v /path/to/DeepSeek-V4-Flash-0731:/model \
#     -e MODEL_PATH=/model \
#     -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     topmag-server:v0.5.17-0731 bash
# Launch benchmarks explicitly with mustafar/scripts/local/bench_serving.sh.
#
# The weights are intentionally NOT baked in; they are mounted.
# v0.5.17 is the pinned SGLang tree used for the official 0731 bring-up.
FROM lmsysorg/sglang:v0.5.17-cu130

# 1. Reproduce the active source tree (sglang-lowrank) as a local clone of the
#    base image's own /sgl-workspace/sglang at the same tag the server uses.
RUN git clone /sgl-workspace/sglang /sgl-workspace/sglang-lowrank \
    && cd /sgl-workspace/sglang-lowrank && git checkout v0.5.17

# 2. Ship the mustafar package. PACKAGE_ROOT resolves to the parent dir of the
#    mustafar/ folder, so the sglang hook's `import mustafar` finds it without
#    an XKV_PACKAGE_DIR override.
COPY mustafar/ /opt/mustafar/flash-optimizations/mustafar/

# Build the Fused adapter for L4/H100 and validate its CPU ABI before GPU use.
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"
RUN cd /opt/mustafar/flash-optimizations \
    && MAX_JOBS=4 python3 mustafar/cuda/setup.py build_ext --inplace \
    && python3 -m mustafar.tests.fused_cpu

# 3. Bake in the TopMag hook into the ACTIVE tree (PYTHONPATH-pinned at run),
#    then verify the complete patch against its original backups.
ENV SG_LOWRANK_SRC=/sgl-workspace/sglang-lowrank/python
RUN cd /opt/mustafar/flash-optimizations \
    && python3 -m mustafar patch \
    && python3 -m mustafar verify

# Modal injects its runtime dependencies after this Dockerfile and currently
# supplies typing_extensions 4.12.2, while the base pydantic_core imports
# Sentinel (available in newer releases). A one-package target directory keeps
# the server import-compatible.
RUN python3 -m pip install --no-cache-dir --target /opt/sglang-runtime-fixes \
    "typing_extensions==4.16.0"

# 4. Store-time TopMag on by default; config via env at run time. NCCL
#    settings reproduce the tested single-host 4-GPU run (override at run
#    time with -e if the target host uses InfiniBand / different fabrics).
ENV SGLANG_OPT_TOPMAG=1 \
    KEEP=0.5 \
    NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=lo \
    NCCL_P2P_LEVEL=NVL \
    NCCL_PROTO=Simple \
    NCCL_ALGO=Ring \
    PYTHONPATH=/opt/sglang-runtime-fixes:/sgl-workspace/sglang-lowrank/python:/opt/mustafar/flash-optimizations

# No automatic server startup; override the base image's default command.
CMD ["bash"]
