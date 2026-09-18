# Remnant local server image — DeepSeek-V4-Flash-0731, two trees, NO bake
#
# This is the "native untouched AND patched" local container. Unlike
# docker/modal.Dockerfile it does NOT clone sglang-lowrank and does NOT apply the
# Remnant patch at build time. The pristine /sgl-workspace/sglang tree ships in
# the base image (v0.5.18); the patched /sgl-workspace/sglang-lowrank tree is
# created at runtime by remnant/scripts/local/container.sh (git clone + the
# read-only `remnant drift` check). Patching happens through the user's scripts,
# never baked in.
#
# Build:
#   cd flash-optimizations
#   docker build -t remnant:v0.5.18 -f remnant/docker/local.Dockerfile .
#
# Run (container.sh does this for you):
#   docker run -d --name remnant --network host --gpus all --shm-size 120g \
#     -v /:/mnt/host_root -v /home:/home remnant:v0.5.18
#
# The weights and the remnant repo are NOT baked in; both are mounted
# (weights at /mnt/host_root/..., remnant via the flash-optimizations repo at
# $REPO_CT). The COPY below is a self-contained fallback so `import remnant`
# resolves even without the mount.
FROM lmsysorg/sglang:v0.5.18-cu130

# Ship the remnant package (no clone, no patch at build). PACKAGE_ROOT resolves
# to the parent of remnant/, so `import remnant` finds it.
COPY remnant/ /opt/remnant/flash-optimizations/remnant/

# NOTE: no typing_extensions runtime-fixes step here (unlike modal.Dockerfile).
# That image needs it because the Modal platform injects typing_extensions 4.12.2
# after the build; this self-contained local image already ships 4.16.0 and
# pydantic_core imports Sentinel fine, so an override would add nothing (and
# would require PyPI egress at build time).

# No TopMag/NCCL defaults baked: serve.sh sets SGLANG_OPT_TOPMAG/KEEP and the
# NCCL overrides per leg (native | packed) at run time. Native must boot with
# packing off and no remnant import; baking SGLANG_OPT_TOPMAG=1 here would
# break it.
ENV PYTHONPATH=/opt/remnant/flash-optimizations

# No automatic server startup; override the base image's default command.
CMD ["bash"]
