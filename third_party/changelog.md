<!-- Author: Winston Cai. -->

# Remnant fork change map

This file records the files changed by the Remnant work in the two large
upstream repositories. It is generated against the pinned fork bases, not
against whichever checkout happens to be open in the submodule directory.

The submodules remain independent repositories. This manifest lives in the
parent repository so it is visible without opening either fork.

## FlashMLA

- Fork: `winstonxcai/FlashMLA`
- Upstream base: `sgl-project/FlashMLA`
- Base commit: `05e26647fe840b8baedae486c2d86d5ce4efeb7c`
- Remnant branch: `remnant/flashmla-v0518`
- Current branch head: `a4b20774a4d6b9a0263fa2ff719dbb056bd9c8e2`
- Comparison: `05e26647..origin/remnant/flashmla-v0518`

### Production/API integration

- `csrc/api/api.cpp` — exported Remnant sparse-decode entry point.
- `csrc/api/sparse_decode.h` — public decode declarations.
- `csrc/params.h` — Remnant decode parameters.
- `csrc/sm90/decode/sparse_fp8/config.h` — SM90 Remnant configuration.
- `csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu` — SM90 instantiation wiring.
- `csrc/sm90/decode/sparse_fp8/instantiations/model1_remnant_h64.cu` — H64 Remnant instantiation.
- `csrc/sm90/decode/sparse_fp8/instantiations/model1_remnant_h128.cu` — H128 Remnant instantiation.
- `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` — Remnant loader and decode implementation.
- `csrc/sm90/decode/sparse_fp8/splitkv_mla.h` — loader/decode declarations.
- `flash_mla/__init__.py` — Python export wiring.
- `flash_mla/flash_mla_interface.py` — Python API dispatch.
- `setup.py` — build/source integration.

### Remnant validation and measurement

- `tests/lib.py` — shared upstream test generators plus the 328-byte Packed fixtures.
- `tests/test_flash_mla_remnant_decoding.py` — direct-vs-Native and independent-reference validity tests.
- `benchmark/remnant/microbench.py` (SGLang fork) — production-function Native/Packed Triton/Packed Kernel decode and store microbenchmark (protocol remnant-three-modes-v1).

The former `tests/remnant_fixture.py` was removed after its consumers migrated
to `tests/lib.py`; the separate FlashMLA `benchmark/bench_remnant_decode.py`
was retired in favor of the SGLang benchmark above.

`benchmark/bench_flash_mla.py` is intentionally not listed: it is an upstream
FlashMLA benchmark inherited unchanged for FlashInfer/FlashMLA comparison.

## SGLang

- Fork: `winstonxcai/sglang`
- Upstream base: `sgl-project/sglang`
- Base commit: `71de97b264b04dcd514cf904003028aefe9775c8` (`v0.5.18`)
- Remnant branch: `remnant/v0.5.18`
- Current branch head: `40d07208195c31f75f30ee0c12f3c8c6c1fd2793`
- Comparison: `71de97b264b04dcd514cf904003028aefe9775c8..40d07208195c31f75f30ee0c12f3c8c6c1fd2793`

### Runtime and build integration

Per-layer Packed Triton extension (GPU qualification pending): startup count
and model-layer overrides resolve to immutable layouts. Values allocate at the
configured width, and local-stage pool accounting sums C4 record sizes. Triton
store/gather/Native materialization derive geometry from their layer buffers.
Non-256 counts require Triton throughout and reject HiCache; optimized ABIs
remain at 256. Parent `scripts/local/run-server.sh` and `serve.sh` forward the
options and preflight the mounted model/source before replacing a server.
The existing parent `scripts/modal/kernels.py` sanitizer selection includes
all-count packing and mixed-layer graph replay; no remote run is implied.

- `python/sglang/srt/observability/req_time_stats.py` — expose the existing tokenizer first-output timestamp in request metadata for TTFT attribution through the stock JSON exporter.
- `test/registered/unit/observability/test_req_time_stats.py` — first-output metadata, single-token completion, unset timestamps, and cross-process clock rebasing checks.
- `python/sglang/srt/utils/nvtx_utils.py` — gated CPU-only scheduler batch metadata and graph-replay annotations using built-in `torch.cuda.nvtx`, with a startup probe and no external NVTX package.
- `python/sglang/srt/managers/scheduler.py` — step/rank metadata nested inside the existing scheduler span.
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` — exact padded-batch annotation around the graph replay launch.
- `python/sglang/srt/server_args.py` — cache-format and Triton/Kernel implementation flags.
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` — per-layer Packed allocation, geometry and model-to-C4 mapping.
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py` — decode dispatch.
- `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` — packed writes.
- `python/sglang/srt/models/deepseek_v4.py` — opt-in capture of the attention output after inverse RoPE.
- `python/sglang/srt/layers/attention/dsv4/indexer.py` — raw-index plumbing.
- `python/sglang/srt/model_executor/pool_configurator.py` — stage-local sum of Packed C4 record sizes.
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` — Packed L2 HiCache routing and unsupported-layout checks.
- `python/sglang/srt/mem_cache/memory_pool_host.py` — coordinated host transfers for Packed values, bitmaps and scales.
- `python/sglang/srt/remnant/__init__.py` — internal package exports.
- `python/sglang/srt/remnant/bitmap.py` — bitmap helpers.
- `python/sglang/srt/remnant/config.py` — startup format/implementation configuration and validation.
- `python/sglang/srt/remnant/packed.py` — packing/reconstruction dispatch, reusable Triton workspace and opt-in fused unpack/RoPE reader; split remains the serving default.
- `python/sglang/srt/remnant/reference.py` — reference reconstruction.
- `python/sglang/srt/remnant/triton/__init__.py` — Triton package exports.
- `python/sglang/srt/remnant/triton/kernels.py` — shared guarded gather, split RoPE and an experimental fused reader preserving intermediate BF16 rounding; Native-page materialization and prefill plan mapping.
- `python/sglang/srt/remnant/triton/fused_store.py` — fused FP32 selection, normalization and Packed C4 storage.
- `python/sglang/kernels/aot/cmake/flashmla.cmake` — FlashMLA fork build pinning.
- `python/sglang/kernels/aot/csrc/flashmla_extension.cc` — extension bindings.
- `python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py` — Python dispatch wrapper.
- `python/sglang/kernels/aot/CMakeLists.txt` — Packed-to-Native adapter source and standalone build target.
- `python/sglang/kernels/aot/csrc/common_extension.cc` — adapter operator registration.
- `python/sglang/kernels/aot/csrc/remnant/packed_to_native.cu` — fused Packed-to-Native CUDA reconstruction.
- `python/sglang/kernels/aot/csrc/remnant_extension.cc` — standalone adapter extension registration.
- `python/sglang/kernels/aot/include/sgl_kernel_ops.h` — adapter declaration.

### SGLang validation and benchmarks

- `test/registered/unit/utils/test_nvtx_utils.py` — CPU annotation contracts, disabled-path behavior and exception cleanup.
- `test/registered/unit/test_dsv4_c4_cache_format.py` — startup argument validation.
- `test/registered/attention/unittests/dsv4/test_remnant_pool.py` — pool layout/capacity.
- `test/registered/attention/unittests/dsv4/test_remnant_hicache.py` — Packed host-cache transfer tests.
- `test/registered/attention/unittests/dsv4/test_remnant_pack.py` — pack correctness and CPU checks for explicit fused-reader dispatch.
- `test/registered/attention/unittests/dsv4/test_remnant_fused_store.py` — fused-store bytes, fallback, destinations and graph replay.
- `test/registered/attention/unittests/dsv4/test_remnant_compressor_store.py` — serving compressor routing.
- `test/registered/attention/unittests/dsv4/test_remnant_backend.py` — backend reconstruction/dispatch.
- `test/registered/attention/unittests/dsv4/test_remnant_cuda_graph.py` — graph replay, including fused reads with changing Packed planes, indices, lengths and frequencies.
- `test/registered/attention/unittests/dsv4/test_remnant_flashmla_direct.py` — direct FlashMLA parity.
- `python/sglang/test/kernels/deepseek_v4/test_remnant_pack_kernel.py` — pack kernel checks.
- `python/sglang/test/kernels/deepseek_v4/test_remnant_unpack_kernel.py` — unpack correctness, compilation reuse, exact fused/split BF16 parity, pruned complex partners, and exact historical Packed parity.
- `python/sglang/test/kernels/deepseek_v4/fixtures/main_packed/` — immutable historical Packed source, hashes, and parity contract; test-only, never used by production.
- `benchmark/remnant/README.md` — benchmark documentation.
- `benchmark/remnant/microbench.py` — explicit Native/Adapter/Direct decode matrix with JSON/CSV output.

## Regenerating the file list

From the parent repository:

```sh
git -C third_party/flashmla diff --name-status \
  05e26647fe840b8baedae486c2d86d5ce4efeb7c \
  origin/remnant/flashmla-v0518

git -C third_party/sglang diff --name-status \
  71de97b264b04dcd514cf904003028aefe9775c8 \
  origin/remnant/v0.5.18
```

If either fork receives new Remnant commits, update the branch head and the
corresponding lists here in the same parent-repository change.

### Three-mode measurement ownership

- `benchmark/remnant/microbench.py` — independent references, mode-owned decode/store execution, graph timing and warmed eager capture; configurable 64..448 survivors in steps of 64, with Kernel restricted to 256 and Native unchanged.
- `benchmark/remnant/test_microbench.py` — CLI/schema/matrix contracts, survivor-profile configuration, paired raw-input identities and H100 numerical/graph coverage.
- `benchmark/remnant/README.md` — new baseline/protocol and maintained commands.
- `python/sglang/benchmark/remnant/cases.py` — shared production store/decode matrix, deterministic seeds, statistics, result serialization, timing coverage checks and Nsight Compute export validation. Modal uses this contract rather than duplicating it.
- `python/sglang/benchmark/remnant/__init__.py` — benchmark package marker.
- The opt-in `prefill-read` suite in these existing benchmark files compares
  unchanged Native with Packed192/Packed256 on shared inputs, measures historical
  C4/SWA reconstruction, sparse attention and their complete path, and records
  independent validity plus warm/rotating cache policy. Parent Modal timing and
  traffic profiling reuse the same factories; no runtime dispatch is changed.
- `test/registered/attention/unittests/dsv4/test_remnant_fused_store.py` — consumes the shared production store case definitions.

Historical Adapter/Direct benchmarking and its Native gate are preserved in the
parent experiment archive; this rewrite changes no production runtime dispatch.

### Opt-in capture storage

- Test audit (2026-10-18; `0e94e2b32b`): remove the unreachable
  dense-pruning branch and its unused runtime flags/helper; retain the active
  keep-mask reference and frozen compatibility fixtures. Compressor/startup
  checks retain Native dispatch and raw-FP32 preservation coverage.
- Capture/replay regressions: empty CSA positions use integer indices; replay
  validates and decodes already-compacted Native rows without masking twice.
  Both storage schemas cover padding, duplicates and empty CSA/SWA selections;
  matching empty-domain LSE values report zero error. Numerical tolerances are
  unchanged. Live H100 validation remains outstanding.

- `python/sglang/srt/remnant/capture.py` — schema-2 owner-only shared copies,
  audited TP payloads and per-layer constants; inactive without capture settings.
- `python/sglang/benchmark/remnant/calibrate.py` — legacy/shared snapshot reader,
  exact conversion, dependency hashes, per-issue bundles, bounded extraction and
  prepared-capture data preflight for the shell launcher.
- `test/registered/attention/unittests/dsv4/test_remnant_calibration.py` — schema
  round trips, partial/duplicate selections, audit/hash failures, publication and
  disabled/non-owner copy contracts.

The capture hooks are opt-in and inactive during normal serving. The recorded
pin includes their source; it does not establish live GPU validation. This
publication passed 65 focused CPU checks with 36 CUDA-dependent cases skipped;
no GPU, model-load, or image-build run was performed for this push.

- `python/sglang/benchmark/remnant/quantization.py` — independent FP64 normalization and FP32-to-BF16/FP8 quantization-cell validity; exact scale/value interval checks.
- `python/sglang/srt/remnant/triton/fused_store.py` — explicit pairwise FMA operand order matches unchanged Triton packing at nonuniform-FP32-weight quantization boundaries.
- `test/registered/attention/unittests/dsv4/test_remnant_fused_store.py` — regression coverage for the three production-sized cases exposing FMA-order byte differences.

### Independent Packed store override

- `python/sglang/srt/server_args.py` and `python/sglang/srt/remnant/config.py` — optional store override, defaults and validation.
- `python/sglang/srt/runtime_context.py` — resolved cache/store-mode publication in spawned workers.
- `python/sglang/srt/remnant/__init__.py` — resolved store accessor.
- `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` — storage dispatch independent of decode.
- `test/registered/unit/test_dsv4_c4_cache_format.py` — all four combinations in spawned workers and reset/validation contracts.

Decode and numerical kernels are unchanged.
