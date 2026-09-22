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
- `benchmark/remnant/microbench.py` (SGLang fork) — production-facing Native/Adapter/Direct decode microbenchmark.

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
- Current branch head: `1961f1383db3f95fc40699bac7c6823ac09c87ac`
- Comparison: `71de97b264b04dcd514cf904003028aefe9775c8..origin/remnant/v0.5.18`

### Runtime and build integration

- `python/sglang/srt/server_args.py` — `--dsv4-c4-cache-format` flag.
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py` — packed pool.
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py` — decode dispatch.
- `python/sglang/srt/layers/attention/dsv4/compressor_v2.py` — packed writes.
- `python/sglang/srt/layers/attention/dsv4/indexer.py` — raw-index plumbing.
- `python/sglang/srt/model_executor/pool_configurator.py` — packed capacity accounting.
- `python/sglang/srt/remnant/__init__.py` — internal package exports.
- `python/sglang/srt/remnant/bitmap.py` — bitmap helpers.
- `python/sglang/srt/remnant/config.py` — format configuration and validation.
- `python/sglang/srt/remnant/packed.py` — packed storage helpers.
- `python/sglang/srt/remnant/reference.py` — reference reconstruction.
- `python/sglang/srt/remnant/triton/__init__.py` — Triton package exports.
- `python/sglang/srt/remnant/triton/kernels.py` — packed reconstruction kernels.
- `python/sglang/kernels/aot/cmake/flashmla.cmake` — FlashMLA fork build pinning.
- `python/sglang/kernels/aot/csrc/flashmla_extension.cc` — extension bindings.
- `python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py` — Python dispatch wrapper.

### SGLang validation and benchmarks

- `test/registered/unit/test_dsv4_c4_cache_format.py` — startup argument validation.
- `test/registered/attention/unittests/dsv4/test_remnant_pool.py` — pool layout/capacity.
- `test/registered/attention/unittests/dsv4/test_remnant_pack.py` — pack correctness.
- `test/registered/attention/unittests/dsv4/test_remnant_backend.py` — backend reconstruction/dispatch.
- `test/registered/attention/unittests/dsv4/test_remnant_cuda_graph.py` — graph replay.
- `test/registered/attention/unittests/dsv4/test_remnant_flashmla_direct.py` — direct FlashMLA parity.
- `python/sglang/test/kernels/deepseek_v4/test_remnant_pack_kernel.py` — pack kernel checks.
- `python/sglang/test/kernels/deepseek_v4/test_remnant_unpack_kernel.py` — unpack kernel checks.
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
