# Local server image — DeepSeek-V4-Flash-0731, two trees, NO bake
#
# Build from the repository root:
#   docker build -t remnant:v0.5.18 -f docker/local.Dockerfile .
#
# The weights and the SGLang fork are mounted at runtime.
FROM lmsysorg/sglang:v0.5.18-cu130

# Cache format is selected by serve.sh with --dsv4-c4-cache-format. Native is
# the fork's default; packed is an explicit Remnant startup mode.
ENV PYTHONPATH=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/third_party/sglang/python

CMD ["bash"]
