# Remnant local server image — DeepSeek-V4-Flash-0731, two trees, NO bake
#
# This is the "native untouched AND patched" local container. Unlike
# docker/modal.Dockerfile it does NOT clone sglang-lowrank and does NOT apply the
# Mustafar patch at build time. The pristine /sgl-workspace/sglang tree ships in
# the base image (v0.5.18); the patched /sgl-workspace/sglang-lowrank tree is
# created at runtime by mustafar/scripts/local/container.sh (git clone + the
# read-only `mustafar drift` check). Patching happens through the user's scripts,
# never baked in.
#
# Build:
#   cd flash-optimizations
#   docker build -t remnant:v0.5.18 -f mustafar/docker/local.Dockerfile .
#
# Run (container.sh does this for you):
#   docker run -d --name remnant --network host --gpus all --shm-size 120g \
#     -v /:/mnt/host_root -v /home:/home remnant:v0.5.18
#
# The weights and the mustafar repo are NOT baked in; both are mounted
# (weights at /mnt/host_root/..., mustafar via the flash-optimizations repo at
# $REPO_CT). The COPY below is a self-contained fallback so `import mustafar`
# resolves even without the mount.
FROM lmsysorg/sglang:v0.5.18-cu130

# Ship the mustafar package (no clone, no patch at build). PACKAGE_ROOT resolves
# to the parent of mustafar/, so `import mustafar` finds it.
COPY mustafar/ /opt/mustafar/flash-optimizations/mustafar/

# NOTE: no typing_extensions runtime-fixes step here (unlike modal.Dockerfile).
# That image needs it because the Modal platform injects typing_extensions 4.12.2
# after the build; this self-contained local image already ships 4.16.0 and
# pydantic_core imports Sentinel fine, so an override would add nothing (and
# would require PyPI egress at build time).

# No TopMag/NCCL defaults baked: serve.sh sets SGLANG_OPT_TOPMAG/KEEP and the
# NCCL overrides per leg (native | packed) at run time. Native must boot with
# packing off and no mustafar import; baking SGLANG_OPT_TOPMAG=1 here would
# break it.
ENV PYTHONPATH=/opt/mustafar/flash-optimizations

# No automatic server startup; override the base image's default command.
CMD ["bash"]
