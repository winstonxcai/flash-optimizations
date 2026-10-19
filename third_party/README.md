# Remnant forks

Author: Winston Cai.

This guide describes the current integration and its ownership. The
[changelog](CHANGELOG.md) records commit-based milestones; benchmark manifests
record what was actually tested. Neither a branch name nor a matching Git pin
proves which binary an existing container loaded.

## Source identities

| Fork | Upstream base | Development branch |
|---|---|---|
| [SGLang](https://github.com/winstonxcai/sglang) | `sgl-project/sglang`, v0.5.18, `71de97b264b04dcd514cf904003028aefe9775c8` | `remnant/v0.5.18` |
| [FlashMLA](https://github.com/winstonxcai/FlashMLA) | `sgl-project/FlashMLA`, `05e26647fe840b8baedae486c2d86d5ce4efeb7c` | `remnant/flashmla-v0518` |

The parent Git links are authoritative for **current pins**. Read them rather
than copying a branch head from documentation:

```bash
git ls-tree HEAD third_party/sglang third_party/flashmla
git -C third_party/sglang status --short --branch
git -C third_party/flashmla status --short --branch
git -C third_party/sglang rev-parse HEAD
git -C third_party/flashmla rev-parse HEAD
```

Ordinary builds require each fork's HEAD to equal its parent pin and both forks
to be clean, including untracked files. Follow the root [source safeguards](../AGENTS.md#git-and-source-ownership)
when synchronizing; never discard or stash work automatically. Approved source
overrides must record both commits and submitted-content hashes.

`scripts/sources.py` is the read-only gate used before local container/serving/
benchmark work and before Modal image definitions. It checks nested submodules
too; it never synchronizes them. Model-free Modal experiments may explicitly set
`REMNANT_SGLANG_SOURCE_OVERRIDE` / `REMNANT_FLASHMLA_SOURCE_OVERRIDE` to Git
checkouts, including dirty worktrees. Their commits, expected pins, dirty state,
content hashes and nested-source identities are retained in the build manifest.
Ordinary serving does not accept these overrides. A source check does not prove
that an existing container's loaded binaries match; source changes need a rebuild.

## Integration map

Paths below are relative to the owning fork. Native is the default; Packed is
selected by `--dsv4-c4-cache-format remnant`.

| Capability | Owner and entry point | Contract / checks |
|---|---|---|
| Configuration and allocation | SGLang `srt/remnant/config.py`, `srt/mem_cache/deepseek_v4_memory_pool.py` under `python/sglang/` | Startup layouts, local-stage capacity accounting; startup/pool tests |
| Selection and storage | SGLang `srt/layers/attention/dsv4/compressor_v2.py`, `srt/remnant/triton/fused_store.py` | Raw-FP32 producer; independent store selection and byte-parity tests |
| Packed reconstruction | SGLang `srt/remnant/packed.py`, `srt/remnant/triton/kernels.py` | Triton gather/RoPE and Native materialization; changing-input graph tests |
| CUDA Adapter | SGLang `python/sglang/kernels/aot/csrc/remnant/packed_to_native.cu` | Reconstruct Native workspace before stock FlashMLA; 256 survivors only |
| Direct attention | FlashMLA `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` | Read Packed records inside attention; output/LSE, edge and graph tests |
| Host cache | SGLang `srt/mem_cache/memory_pool_host.py`, `hybrid_cache/hybrid_pool_assembler.py` | Coordinated value/bitmap/scale transfers; 256 survivors only |
| Capture and replay | SGLang `srt/remnant/capture.py`, `benchmark/remnant/calibrate.py` under `python/sglang/` | Opt-in owner-only capture, audited TP payloads, strict source hashes |
| Profiling and request timing | SGLang `srt/utils/nvtx_utils.py`, scheduler, graph runner, request statistics | Gated built-in PyTorch NVTX; tokenizer first-output timestamp |
| Measurements | SGLang `benchmark/remnant/microbench.py` | Shared decode/store and prefill-read fixtures; [protocol and commands](sglang/benchmark/remnant/README.md) |

### SGLang file map

Inventory relative to [sglang/](sglang/). Directory/glob entries cover changed
files within them, not every upstream file in that directory.

- **Runtime and numerics:** `python/sglang/srt/remnant/` owns configuration,
  bitmap/reference helpers, pack/reconstruction dispatch, capture and Triton
  kernels. `python/sglang/srt/runtime_context.py` publishes resolved settings
  to spawned workers; `python/sglang/srt/server_args.py` validates startup options.
- **Model integration:** `python/sglang/srt/layers/attention/deepseek_v4_backend.py`
  dispatches decode; `python/sglang/srt/layers/attention/dsv4/compressor_v2.py`
  dispatches stores; `python/sglang/srt/layers/attention/dsv4/indexer.py` carries
  raw positions. `python/sglang/srt/models/deepseek_v4.py` captures the attention
  output after inverse RoPE when requested.
- **Pools and capacity:** `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`,
  `python/sglang/srt/model_executor/pool_configurator.py`,
  `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` and
  `python/sglang/srt/mem_cache/memory_pool_host.py` own layer geometry, capacity
  and host transfers.
- **Build and Adapter:** `python/sglang/kernels/aot/CMakeLists.txt`,
  `python/sglang/kernels/aot/cmake/flashmla.cmake`,
  `python/sglang/kernels/aot/csrc/common_extension.cc`,
  `python/sglang/kernels/aot/csrc/flashmla_extension.cc`,
  `python/sglang/kernels/aot/csrc/remnant/packed_to_native.cu`,
  `python/sglang/kernels/aot/csrc/remnant_extension.cc`,
  `python/sglang/kernels/aot/include/sgl_kernel_ops.h` and
  `python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py` provide bindings and
  builds from the supplied local FlashMLA source, not a moving remote checkout.
- **Observability:** `python/sglang/srt/utils/nvtx_utils.py`,
  `python/sglang/srt/managers/scheduler.py`,
  `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` and
  `python/sglang/srt/observability/req_time_stats.py` own batch/rank/step markers,
  padded graph sizes and first-output metadata. No external `nvtx` package is required.
- **Benchmarks and calibration:** `benchmark/remnant/` owns the maintained CLI,
  protocol and harness tests. `python/sglang/benchmark/remnant/` owns shared case
  definitions, seeds, statistics, JSON/CSV and profile-export checks, capture
  conversion/preflight and independent quantization-cell validation.
- **Recorded-request serving client:** `python/sglang/benchmark/datasets/openai_dataset.py`
  accepts recorded `prompt_len` estimates without forwarding that metadata or
  locally rendering model-specific chat templates. Zero explicitly marks missing
  accounting. `python/sglang/benchmark/serving.py` includes tool-call-only output
  in first-output timing and rejects missing streaming usage when requested.
  CPU regressions live in `test/registered/bench_fn/test_bench_serving_reasoning_stream.py`.
- **Integration tests:** `test/registered/attention/unittests/dsv4/test_remnant_*.py`
  covers backend, calibration, compressor stores, graphs, Direct, fused stores,
  HiCache, packing and pools. Startup and observability contracts live in
  `test/registered/unit/test_dsv4_c4_cache_format.py`,
  `test/registered/unit/observability/test_req_time_stats.py` and
  `test/registered/unit/utils/test_nvtx_utils.py`.
- **Low-level tests:**
  `python/sglang/test/kernels/deepseek_v4/test_remnant_pack_kernel.py` and
  `python/sglang/test/kernels/deepseek_v4/test_remnant_unpack_kernel.py` cover
  numerical, layout, compilation-reuse and fused/split parity contracts.
  `python/sglang/test/kernels/deepseek_v4/fixtures/main_packed/` is an immutable,
  hash-recorded historical reference, never a production dependency.

### FlashMLA file map

Inventory relative to [flashmla/](flashmla/).

- **API and build:** `csrc/api/api.cpp`, `csrc/api/sparse_decode.h`,
  `csrc/params.h`, `flash_mla/__init__.py`, `flash_mla/flash_mla_interface.py`
  and `setup.py` expose and build Direct. `csrc/api/sparse_fwd.h` guards SM100
  APIs so the SM90-only build remains possible.
- **SM90 decode:** `csrc/sm90/decode/sparse_fp8/config.h`,
  `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` and
  `csrc/sm90/decode/sparse_fp8/splitkv_mla.h` own Packed loading and attention,
  including H64 eight-warp producer synchronization/readiness.
  `csrc/sm90/decode/sparse_fp8/instantiations/model1_persistent_h64.cu`,
  `csrc/sm90/decode/sparse_fp8/instantiations/model1_remnant_h64.cu` and
  `csrc/sm90/decode/sparse_fp8/instantiations/model1_remnant_h128.cu` wire instantiations.
- **Tests:** `tests/lib.py` owns shared fixtures;
  `tests/test_flash_mla_remnant_decoding.py` checks Native/reference output and
  LSE, extreme bitmaps, partial/invalid selections and graph replay. The former
  separate Remnant fixture and benchmark were consolidated into these fixtures
  and SGLang's maintained microbenchmark.
- **Workspace metadata:** `.DS_Store` is ignored and no longer tracked
  (FlashMLA `97f3faa`). This cleanup changes no runtime or kernel code.

## Runtime contracts

- Default Packed: 256 of 512 coordinates, **328 bytes** = 256 FP8 values +
  64 bitmap bytes + eight UE8M0 scales. Native uses **584 logical bytes**.
- Triton supports startup per-layer counts **64..448 in steps of 64**, with
  `count + 72` logical bytes in separate planes. These are fixed startup layouts,
  not adaptive budgets. Non-256 layouts require Triton store/reconstruction and
  no HiCache; CUDA store, Adapter and Direct remain 256-only.
- Match the FP32 producer: select raw magnitudes, mask, RMSNorm, prescribed BF16
  rounding, then UE8M0/E4M3 packing. Fused-store FMA operand order matters at
  quantization boundaries; historical BF16-input screens do not prove FP32 parity.
- Preserve raw logical positions versus physical locations, SWA, RoPE, sink,
  partial lengths, duplicates and invalid entries. Store selection is independent
  of decode; its default follows the selected decode implementation.
- The experimental fused prefill reader (`fused_rope=True`, benchmark
  `--prefill-reader fused`) preserves intermediate BF16 rounding. Split remains
  the normal serving reader; do not confuse this experiment with the CUDA Adapter.
- Capture and profiling hooks are opt-in. First-output metadata is a tokenizer
  timestamp, not an isolated GPU measurement. Kernel scope is **H100/SM90**;
  no SM100 qualification is implied.

## Validation evidence

These are retained records, not tests of every later commit. Read submitted
source hashes, binary identities and validity files in each collection before
reusing its conclusions. Test presence and CUDA skips are not passes.

| Scope | Evidence | Limits |
|---|---|---|
| Default-256 decode/store | [H100 three-mode reference](../benchmarks/microbench/decode-store-h100-three-mode-20261014/README.md): 114 GPU checks, 33 timing cases, 12 Compute captures; recorded SGLang `7b2592eb9048`, FlashMLA `57e9560`, plus submitted-content hashes | Model-free; not qualification of the current pin pair or serving TPOT |
| Prefill reconstruction | [Split-reader study](../benchmarks/microbench/prefill-read-20261018/README.md) and [fused-reader study](../benchmarks/microbench/prefill-read-fused-rope-20261018/README.md): fused study reports 291 validation checks, 67 under each sanitizer, 36 timing cases and 108 profiles, including all-count read parity and changing-input graphs | Dirty source snapshot based on SGLang `0098b4277f` / FlashMLA `57e9560`, not a clean current-pin run; no production promotion or serving/quality gain |
| Matched-load serving | [C24 triplicate](../benchmarks/lswb-replay/c24-native-packed256-n3-20261018/README.md): three independent unprofiled runs per mode, all meeting the 99% success criterion and TTFT p90 below 10 s; [128k/8k study](../benchmarks/serving/random-128k-8k-c24-capacity-n1-20261018/README.md): six error-free single-run cells with exact output lengths | Triplicate still has 15–21 failures/run. Collections use different images and lack full loaded-runtime/extension identity; no statistical equivalence, task-quality or replicated SLO-capacity claim |
| Capture/configuration/HiCache | Focused regression entry points in the file map | No complete retained qualification of the current pair's live TP4 capture or all host-cache/non-256 serving combinations; historical CPU capture checks are not GPU validation |

The fused-prefill GPU job timed out during CSV export after capture. Remaining
exports were recovered locally through NVIDIA's report API and checked against
86 existing CLI exports; this does not make the original job exit successful.
Read parity does not qualify all non-256 stores, mixed-layer serving or HiCache.

Training cache/checkpoint access and historical source-identity compatibility
belong in the [training guide](../training/README.md#cache-and-checkpoints).

## Maintenance

1. Change and test the owning fork; preserve upstream attribution and licenses.
2. Update this guide when ownership, affected files or contracts change. Update
   evidence only from recorded runs, preserving their original identities.
3. Commit and push fork changes before publishing the parent pins. Add a dated
   [changelog](CHANGELOG.md) milestone for meaningful integrations/fixes, with
   parent commit, both fork pins and validation scope. Do not log every tuning attempt.

Use `Unreleased` until the parent integration commit exists; fill in its SHA in
a follow-up documentation update. Never guess a future commit hash.

From the parent root, regenerate the committed file map against the fixed bases:

```bash
git -C third_party/sglang diff --name-status \
  71de97b264b04dcd514cf904003028aefe9775c8 \
  "$(git rev-parse HEAD:third_party/sglang)"
git -C third_party/flashmla diff --name-status \
  05e26647fe840b8baedae486c2d86d5ce4efeb7c \
  "$(git rev-parse HEAD:third_party/flashmla)"
git log --oneline -- third_party/sglang third_party/flashmla
```

Use `git ls-tree <parent-commit> third_party/sglang third_party/flashmla` to
recover an integration's exact pair. Git history remains the detailed record;
this guide and the changelog are its curated index.
