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

# Compile the local SM90 extensions while building the image. These layers are
# invalidated when either fork's copied source changes, so unchanged Modal runs
# reuse the compiled artifacts without spending H100 time on a rebuild.
RUN set -eux; \
    export TORCH_CUDA_ARCH_LIST=9.0; \
    build_root=/tmp/remnant-flashmla-build; \
    cmake \
      -S /sgl-workspace/sglang-remnant/python/sglang/kernels/aot \
      -B "$build_root" \
      -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_BELOW_SM90=OFF \
      -DSGL_KERNEL_ENABLE_FA3=OFF \
      -DSGL_KERNEL_COMPILE_THREADS=1 \
      -DSGL_KERNEL_ENABLE_FLASHMLA_SM100=OFF \
      -DREMNANT_FLASHMLA_SOURCE_DIR=/opt/flashmla-remnant \
      -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
      -DCUDA_VERSION=13.0; \
    cmake --build "$build_root" --target remnant_ops --parallel 2; \
    cmake --build "$build_root" --target flashmla_ops --parallel 2; \
    package_dir="$(python3 -c 'import pathlib, sysconfig; print(pathlib.Path(sysconfig.get_paths()["purelib"]) / "sgl_kernel")')"; \
    cp "$build_root"/remnant_ops*.so "$package_dir"/; \
    cp "$build_root"/flashmla_ops*.so "$package_dir"/; \
    cp /sgl-workspace/sglang-remnant/python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py "$package_dir"/flash_mla.py; \
    test -n "$(find "$package_dir" -maxdepth 1 -type f -name 'remnant_ops*.so' -print -quit)"; \
    test -n "$(find "$package_dir" -maxdepth 1 -type f -name 'flashmla_ops*.so' -print -quit)"; \
    grep -q remnant_buffers "$package_dir"/flash_mla.py; \
    grep -q remnant_sparse_decode_fwd "$package_dir"/flash_mla.py; \
    rm -rf "$build_root"

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
