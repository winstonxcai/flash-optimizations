# Packed TopMag50 cache — DeepSeek-V4-Flash-0731 server image
#
# Build:
#   cd flash-optimizations
#   docker build -t topmag-server:v0.5.18-0731 -f remnant/docker/modal.Dockerfile .
#
# Open a shell (any H100 host with >=4 GPUs + the 0731 weights on disk):
#   docker run --rm -it --gpus all \
#     -v /path/to/DeepSeek-V4-Flash-0731:/model \
#     -e MODEL_PATH=/model \
#     -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     topmag-server:v0.5.18-0731 bash
# Launch benchmarks explicitly with remnant/scripts/local/bench-serving.sh.
#
# The weights are intentionally NOT baked in; they are mounted.
# The image uses the pinned v0.5.18 fork branch; the outer repository records
# the reviewed commit as a submodule.
FROM lmsysorg/sglang:v0.5.18-cu130

# 1. Use the productized fork directly. The submodule pins the same branch in
#    the outer repository; the image clones the pushed branch for Modal builds.
ARG REMNANT_SGLANG_REF=remnant/v0.5.18
ARG REMNANT_SGLANG_COMMIT=17ebde37cb
RUN git clone --depth 1 --branch ${REMNANT_SGLANG_REF} \
    https://github.com/winstonxcai/sglang.git /sgl-workspace/sglang-remnant \
    && cd /sgl-workspace/sglang-remnant \
    && git checkout ${REMNANT_SGLANG_COMMIT}

# 2. Modal adds the benchmark scripts below. The SGLang fork contains the
#    Remnant runtime and no outer-package import or runtime patch is needed.
ENV SG_LOWRANK_SRC=/sgl-workspace/sglang-remnant/python

# Modal injects its runtime dependencies after this Dockerfile and currently
# supplies typing_extensions 4.12.2, while the base pydantic_core imports
# Sentinel (available in newer releases). A one-package target directory keeps
# the server import-compatible.
RUN python3 -m pip install --no-cache-dir --target /opt/sglang-runtime-fixes \
    "typing_extensions==4.16.0"

# 3. NCCL settings reproduce the tested single-host 4-GPU run (override at run
#    time with -e if the target host uses InfiniBand / different fabrics).
ENV NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=lo \
    NCCL_P2P_LEVEL=NVL \
    NCCL_PROTO=Simple \
    NCCL_ALGO=Ring \
    PYTHONPATH=/opt/sglang-runtime-fixes:/sgl-workspace/sglang-remnant/python:/opt/remnant/flash-optimizations

# No automatic server startup; override the base image's default command.
CMD ["bash"]
