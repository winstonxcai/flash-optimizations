# Remnant local server image — DeepSeek-V4-Flash-0731, two trees, NO bake
#
# This is the local v0.5.18 server image. The pinned fork is mounted from the
# outer repository's third_party/sglang submodule at runtime.
#
# Build:
#   cd flash-optimizations
#   docker build -t remnant:v0.5.18 -f remnant/docker/local.Dockerfile .
#
# Run (container.sh does this for you):
#   docker run -d --name remnant --network host --gpus all --shm-size 120g \
#     -v /:/mnt/host_root -v /home:/home remnant:v0.5.18
#
# The weights and the fork are mounted at runtime (weights at
# /mnt/host_root/..., fork via the flash-optimizations repo at $REPO_CT).
FROM lmsysorg/sglang:v0.5.18-cu130

# NOTE: no typing_extensions runtime-fixes step here (unlike modal.Dockerfile).
# That image needs it because the Modal platform injects typing_extensions 4.12.2
# after the build; this self-contained local image already ships 4.16.0 and
# pydantic_core imports Sentinel fine, so an override would add nothing (and
# would require PyPI egress at build time).

# Cache format is selected by serve.sh with --dsv4-c4-cache-format. Native is
# the fork's default; packed is an explicit Remnant startup mode.
ENV PYTHONPATH=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/third_party/sglang/python

# No automatic server startup; override the base image's default command.
CMD ["bash"]
