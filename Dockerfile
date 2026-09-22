# Reproducible DeepSeek-V4-Flash-0731 server image.
#
# Build from the repository root:
#   docker build -t remnant:v0.5.18 -f Dockerfile .
#
# This is the single server image for local and Modal runs. The model weights
# remain mounted at runtime; the checked-out fork sources are copied from the
# third_party submodules in this repository.
FROM lmsysorg/sglang:v0.5.18-cu130

COPY third_party/sglang /sgl-workspace/sglang-remnant
COPY third_party/flashmla /opt/flashmla-remnant

ENV SG_LOWRANK_SRC=/sgl-workspace/sglang-remnant/python
ENV REMNANT_FLASHMLA_SOURCE_DIR=/opt/flashmla-remnant

RUN python3 -m pip install --no-cache-dir --target /opt/sglang-runtime-fixes \
    "typing_extensions==4.16.0"

ENV NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=lo \
    NCCL_P2P_LEVEL=NVL \
    NCCL_PROTO=Simple \
    NCCL_ALGO=Ring \
    PYTHONPATH=/opt/sglang-runtime-fixes:/sgl-workspace/sglang-remnant/python

CMD ["bash"]
