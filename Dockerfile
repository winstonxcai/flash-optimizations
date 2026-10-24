# Reproducible DeepSeek-V4-Flash-0731 server image.
# Author: Winston Cai.
#
# Build from the repository root:
#   docker build -t remnant:v0.5.18 -f Dockerfile .
#
# This is the single server image for local and Modal runs. The model weights
# remain mounted at runtime; the checked-out fork sources are copied from the
# third_party submodules in this repository.
ARG REMNANT_BASE_IMAGE=lmsysorg/sglang:v0.5.18-cu130
FROM ${REMNANT_BASE_IMAGE}
ARG REMNANT_INTEGRATION=v0.5.18
ARG REMNANT_SGLANG_SOURCE=third_party/sglang-v0.5.18
ARG REMNANT_FLASHMLA_SOURCE=third_party/flashmla-v0.5.18
ARG REMNANT_SGLANG_SHA
ARG REMNANT_FLASHMLA_SHA
ENV REMNANT_BUILT_INTEGRATION=${REMNANT_INTEGRATION}
LABEL remnant.integration=${REMNANT_INTEGRATION} \
      remnant.sglang=${REMNANT_SGLANG_SHA} \
      remnant.flashmla=${REMNANT_FLASHMLA_SHA}

# Keep COPY sources literal for both Docker and Modal's Dockerfile parser.
COPY third_party/sglang-v0.5.18/python/sglang/kernels /sgl-workspace/sglang-remnant/python/sglang/kernels
COPY third_party/flashmla-v0.5.18 /opt/flashmla-remnant
RUN test "$REMNANT_INTEGRATION" = v0.5.18 \
    && test "$REMNANT_SGLANG_SOURCE" = third_party/sglang-v0.5.18 \
    && test "$REMNANT_FLASHMLA_SOURCE" = third_party/flashmla-v0.5.18 \
    && python3 -c 'import importlib.metadata, os; expected=os.environ["REMNANT_BUILT_INTEGRATION"].lstrip("v"); actual=importlib.metadata.version("sglang"); assert actual == expected or actual.startswith(expected+"."), (expected, actual)' \
    && test -f /opt/flashmla-remnant/csrc/sm90/decode/sparse_fp8/instantiations/model1_remnant_h64.cu

# Compile the local SM90 extensions while building the image. These layers are
# invalidated by kernel source changes, not Python capture/client changes.
# Unchanged CUDA artifacts are reused without spending H100 time on a rebuild.
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

COPY third_party/sglang-v0.5.18 /sgl-workspace/sglang-remnant

ENV SG_LOWRANK_SRC=/sgl-workspace/sglang-remnant/python
ENV REMNANT_FLASHMLA_SOURCE_DIR=/opt/flashmla-remnant

# Keep this small build dependency in Git: VDI build workers cannot reliably
# reach PyPI or mirrors. Install offline and enforce the upstream wheel digest.
COPY third_party/wheels /opt/remnant-wheels
RUN python3 -m pip install --disable-pip-version-check --no-cache-dir --no-deps \
    --no-index --find-links /opt/remnant-wheels --require-hashes \
    --target /opt/sglang-runtime-fixes \
    --requirement /opt/remnant-wheels/requirements.txt

ENV NCCL_IB_DISABLE=1 \
    NCCL_SOCKET_IFNAME=lo \
    NCCL_P2P_LEVEL=NVL \
    NCCL_PROTO=Simple \
    NCCL_ALGO=Ring \
    PYTHONPATH=/opt/sglang-runtime-fixes:/sgl-workspace/sglang-remnant/python

CMD ["bash"]
