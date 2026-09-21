# Reproducible DeepSeek-V4-Flash-0731 server image.
#
# Build from the repository root:
#   docker build -t remnant:v0.5.18 -f Dockerfile .
#
# This is the single server image for local and Modal runs. The model weights
# remain mounted at runtime; the SGLang fork and its exact reviewed commit are
# pinned below. FlashMLA is pinned by the fork's CMake FetchContent revision.
FROM lmsysorg/sglang:v0.5.18-cu130

ARG REMNANT_SGLANG_REF=remnant/v0.5.18
ARG REMNANT_SGLANG_COMMIT=84647d6284ff640617de539bb97455d3eec7740f
RUN git clone --depth 1 --branch ${REMNANT_SGLANG_REF} \
    https://github.com/winstonxcai/sglang.git /sgl-workspace/sglang-remnant \
    && cd /sgl-workspace/sglang-remnant \
    && git checkout ${REMNANT_SGLANG_COMMIT}

ENV SG_LOWRANK_SRC=/sgl-workspace/sglang-remnant/python

RUN python3 -m pip install --no-cache-dir --target /opt/sglang-runtime-fixes \
    "typing_extensions==4.16.0"

ENV NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=lo \
    NCCL_P2P_LEVEL=NVL \
    NCCL_PROTO=Simple \
    NCCL_ALGO=Ring \
    PYTHONPATH=/opt/sglang-runtime-fixes:/sgl-workspace/sglang-remnant/python:/opt/flash-optimizations

CMD ["bash"]
