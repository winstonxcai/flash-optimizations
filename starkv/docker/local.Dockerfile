# STAR-CSA local server image — DeepSeek-V4-Flash-0731, two trees, NO bake
#
# Dedicated starkv box, separate from the sibling containers. Like them it does
# NOT clone the fork tree and does NOT apply the starkv patch at build time. The
# pristine /sgl-workspace/sglang tree ships in the base image (v0.5.18); the
# patched /sgl-workspace/sglang-starkv tree is created at runtime by
# starkv/scripts/local/container.sh (clone + the read-only `starkv drift`
# check). Patching happens through the user's scripts, never baked in.
#
# Build:
#   cd flash-optimizations
#   docker build -t starkv:v0.5.18 -f starkv/docker/local.Dockerfile .
#
# Run (container.sh does this for you):
#   docker run -d --name starkv --network host --gpus all --shm-size 120g \
#     -v /:/mnt/host_root -v /home:/home starkv:v0.5.18 bash -c 'sleep infinity'
#
# The weights and this repo are NOT baked in; both are mounted (weights at
# /mnt/host_root/..., starkv via the flash-optimizations repo at $REPO_CT). The
# COPY below is a self-contained fallback so `import starkv` resolves even
# without the mount.
FROM lmsysorg/sglang:v0.5.18-cu130

# Ship the starkv package (no clone, no patch at build). PACKAGE_ROOT resolves
# to the parent of starkv/, so `import starkv` finds it.
COPY starkv/ /opt/starkv/flash-optimizations/starkv/

# No SGLANG_OPT_STARKV default baked: serve.sh sets it per leg
# (native | starkv-recon) at run time. Native must boot with the store off and
# no starkv import; baking it here would break the native leg.
ENV PYTHONPATH=/opt/starkv/flash-optimizations

# No automatic server startup; override the base image's default command.
CMD ["bash"]
