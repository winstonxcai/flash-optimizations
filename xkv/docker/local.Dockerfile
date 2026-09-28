# xKV local server image — DeepSeek-V4-Flash-0731, two trees, NO bake
#
# Dedicated xKV box, separate from `remnant` (mustafar). Like remnant's image it
# does NOT clone sglang-lowrank and does NOT apply the xkv patch at build time.
# The pristine /sgl-workspace/sglang tree ships in the base image (v0.5.18); the
# patched /sgl-workspace/sglang-lowrank tree is created at runtime by
# xkv/scripts/local/container.sh (git clone + the read-only `xkv drift` check).
# Patching happens through the user's scripts, never baked in.
#
# Build:
#   cd flash-optimizations
#   docker build -t xkv:v0.5.18 -f xkv/docker/local.Dockerfile .
#
# Run (container.sh does this for you):
#   docker run -d --name xkv --network host --gpus all --shm-size 120g \
#     -v /:/mnt/host_root -v /home:/home xkv:v0.5.18 bash -c 'sleep infinity'
#
# The weights and the xkv repo are NOT baked in; both are mounted (weights at
# /mnt/host_root/..., xkv via the flash-optimizations repo at $REPO_CT). The COPY
# below is a self-contained fallback so `import xkv` resolves even without the
# mount.
FROM lmsysorg/sglang:v0.5.18-cu130

# Ship the xkv package (no clone, no patch at build). PACKAGE_ROOT resolves to
# the parent of xkv/, so `import xkv` finds it.
COPY xkv/ /opt/xkv/flash-optimizations/xkv/

# No SGLANG_OPT_LOWRANK_KV_STORE default baked: serve.sh sets it per leg
# (native | xkv) at run time. Native must boot with the store off and no xkv
# import; baking it here would break the native leg.
ENV PYTHONPATH=/opt/xkv/flash-optimizations

# No automatic server startup; override the base image's default command.
CMD ["bash"]
