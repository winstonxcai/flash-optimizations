# Remnant forks

Author: Winston Cai.

This guide describes the current integration and its ownership. The
[changelog](CHANGELOG.md) records commit-based milestones; benchmark manifests
record what was actually tested. Neither a branch name nor a matching Git pin
proves which binary an existing container loaded.

## Source identities

Both versions use the existing [SGLang](https://github.com/winstonxcai/sglang) and
[FlashMLA](https://github.com/winstonxcai/FlashMLA) repositories, not new forks.
Folder suffixes identify the SGLang integration pair, not a FlashMLA release.

| Checkout | Exact upstream base | Development branch | Current pin |
|---|---|---|---|
| `sglang-v0.5.18/` | `71de97b264b04dcd514cf904003028aefe9775c8` | `remnant/v0.5.18` | `6687a7c` |
| `flashmla-v0.5.18/` | `05e26647fe840b8baedae486c2d86d5ce4efeb7c` | `remnant/flashmla-v0518` | `97f3faa` |
| `sglang-v0.5.22/` | `baa9befa381511cd4d56f83980951e5a45fffc6d` (peeled release commit) | `codex/glm53-packed` | `23e2e08` |
| `flashmla-v0.5.22/` | `3620b63fc4ebe33950c50e1b7701b6607f44150e` (selected by SGLang v0.5.22) | `remnant/flashmla-v0522` | `3eb8594` |

Both FlashMLA checkouts pin CUTLASS `147f5673d0c1c3dcf66f78d677fd647e4a020219`.
The default remains v0.5.18; `REMNANT_INTEGRATION=v0.5.22` selects the other pair.

The parent Git links are authoritative for **current pins**. Read them rather
than copying a branch head from documentation:

```bash
git ls-tree HEAD third_party/sglang-v0.5.18 third_party/flashmla-v0.5.18 \
  third_party/sglang-v0.5.22 third_party/flashmla-v0.5.22
git -C third_party/sglang-v0.5.18 status --short --branch
git -C third_party/flashmla-v0.5.18 status --short --branch
git -C third_party/sglang-v0.5.18 rev-parse HEAD
git -C third_party/flashmla-v0.5.18 rev-parse HEAD
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

## Parallel v0.5.22 integration

Parent branch `codex/sglang-v0.5.22` starts from local `main` `6fa18c7` in
`data/source-snapshots/sglang-v0.5.22/`. It semantically ports only the integration
delta `88e3b1a..62cb99a`; it does not merge the archived experiment history.
The original `codex/sglang-v0522-port` worktree remains archived unchanged.
The branch subsequently integrates GitHub main `ffb11cb`, preserving its newer
v0.5.18 SGLang pin `6687a7c`, frontend diagnostics/preparation controls, HiCache
launcher options and benchmark defaults. The v0.5.22 pair is unchanged. Main's
custom asynchronous-preparation controls remain v0.5.18-specific; v0.5.22 keeps
upstream conversion offload, and unsupported opt-ins fail the CPU preflight.

Target: 0731 on H100/SM90, **V4 584-byte logical Native layout**, not V4.1.

| Area | Decision | Contract / adaptation |
|---|---|---|
| Bitmap/reference/Triton | Port unchanged | FP32 magnitude selection, mask/RMSNorm, BF16 rounding and UE8M0/E4M3; 64..448 survivors by 64 |
| Compressor/configuration/capture | Adapt | Grouped `srt/arg_groups/fields/model.py` options, scoped context publication and opt-in observers; Native producer preserved |
| Pools and allocator | Adapt | Upstream V4 layout selection, local-stage ownership, reserved pages and physical-byte accounting |
| Prefill/decode and graphs | Adapt | New layout/metadata APIs, head padding and FlashMLA arguments; unchanged SWA/indexer algorithms and raw positions |
| CUDA store/Adapter | Port unchanged | 256-only ABI, independent store/decode selection and explicit dispatch |
| FlashMLA Direct | Adapt | `csrc/kernels/sm90/decode/sparse/`, V4 instantiations and `csrc/api/sparse_decode.cpp`; preserve parameter prefix and upstream APIs |
| HiCache | Adapt | New host-budget interfaces; coordinated value/bitmap/scale transfers, 256-only/backend rejection checks |
| Frontend conversion | Already upstream | Retain upstream executor; no duplicate custom offload patch |
| Profiling/calibration/benchmark | Adapt | Existing entry points and tests; experimental prefill reader stays experimental |
| V4.1 / SM100 / tuning / full-model serving | Defer | No qualification or promotion implied |

V4 synchronization corrections `31877b4..3eb8594` replace divergent aligned
barriers and add explicit local/peer KV-ready acquires. Arithmetic and persistent
layout are unchanged, but v0.5.22 Native is **not the untouched historical
binary**. The [qualification record](../benchmarks/microbench/v0522-qualification-20261023/README.md)
retains failed baselines as well as completed core numerical, allocation and
memcheck/racecheck/synccheck gates. HiCache GPU, full-model, distributed execution
and serving remain pending.

### Reproduce the selected profile

These local CPU commands do not build images or allocate GPUs:

```bash
REMNANT_INTEGRATION=v0.5.18 python3 scripts/sources.py .
REMNANT_INTEGRATION=v0.5.22 python3 scripts/sources.py .
REMNANT_INTEGRATION=v0.5.18 python -m unittest scripts.modal.tests
REMNANT_INTEGRATION=v0.5.22 python -m unittest scripts.modal.tests
```

Local and Modal launchers resolve the same source pair and base image.
v0.5.18 uses `Dockerfile.v0.5.18`, image `remnant:v0.5.18`, container `remnant`;
v0.5.22 uses `Dockerfile.v0.5.22`, image `remnant:v0.5.22`, container
`remnant-v0522`. Container-internal paths stay unchanged. Capture requires matching
serving-image source/profile labels for either version. Never mix compiled pairs.

The kernel launcher reuses v0.5.22 serving AOT binaries, CPU-precompiles Native
store JIT, checks SM90 payloads and Linux CPU cases, and collects strict numerical
and sanitizer coverage through `validate_kernels`. Profiler availability is not
a correctness prerequisite. Its default selection includes HiCache; the archived
core qualification explicitly deferred those four GPU cases and substituted
H128 graph coverage in its frozen sanitizer set. Do not label that evidence as
HiCache qualification. `REMNANT_ALLOCATION_REPORT` selects the optional allocation
JSON; constructor-based tests reconcile planes, padding/reserves, unchanged pools
and workspaces at matched capacity.

**Initial reorganization checks:** 70 local CPU launcher/source tests passed under each
profile, including mocked image-build arguments, source/image rejection and
preservation of the distinct default matrices. All 20 training integration tests
passed; both versioned calibration runners load on CPU. Reproduce with
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest training.tests.test_integration.TrainingIntegrationTests`.
Syntax/import and exact recursive pin checks are separate from GPU checks.
After reconciling main `ffb11cb` and fixing container preparation, **72 launcher/
source tests per profile and 19 replay tests pass** in the existing local Torch
environment. The new prep regression executes the actual shell checks with
fake Docker against both source layouts and rejects missing/stale fields.
The separate lightweight-launcher test's accidental Torch dependency remains
unresolved; these results do not establish a Modal-only fresh-install check.
The curated GPU receipts belong to
parent runtime `e68f9c4` and image `im-vOlukc1AW0gozz9mlAmg1m`, not a new parent
build. **No image was rebuilt and no GPU run was performed after reorganization.**
Raw artifacts remain in the archived worktree/Modal volume, with original paths
and hashes unchanged. Historical v0.5.18 performance is not v0.5.22 performance.

## Default v0.5.18 integration map

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
| Profiling and request timing | SGLang `srt/utils/nvtx_utils.py`, scheduler, graph runner, request statistics | Gated built-in PyTorch NVTX; tokenizer first-output timestamp; opt-in first-output IPC boundaries |
| Measurements | SGLang `benchmark/remnant/microbench.py` | Shared decode/store and prefill-read fixtures; [protocol and commands](sglang-v0.5.18/benchmark/remnant/README.md) |

### SGLang file map

Inventory relative to [sglang-v0.5.18/](sglang-v0.5.18/). Directory/glob entries cover changed
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
  `managers/scheduler_components/output_streamer.py`, `detokenizer_manager.py`,
  `multi_tokenizer_mixin.py` and `tokenizer_manager.py` additionally carry
  first-output transport stamps when `SGLANG_FRONTEND_DIAGNOSTICS=1` at startup.
  The local launcher exposes `FRONTEND_DIAGNOSTICS=1`, requires timing export,
  and preflights the mounted Python implementation before replacing a server.
  Stamps bracket CPU payload serialization/IPC/handling, not GPU completion or
  visible SSE content. API state latches the first packet across later snapshots;
  unset fields stay absent. Disabling the flag preserves normal dispatch.
  The [frontend diagnostic](../benchmarks/lswb-replay/frontend-event-loop-20261022/README.md)
  records CPU coverage and the bounded VDI/GPUQ attribution protocol. Its source
  is isolated from newer local microbenchmark changes; native binaries are unchanged.
- **DSV4 input preparation:** `python/sglang/srt/entrypoints/openai/serving_base.py`,
  `serving_chat.py`, `entrypoints/http_server.py` and `environ.py` own the opt-in
  `SGLANG_DSV4_ASYNC_PREPARATION` text-chat worker. The local launcher exposes
  `DSV4_ASYNC_PREPARATION=1` and checks the actual mounted implementation on CPU
  before replacing a server. `DSV4_PREPARATION_WORKERS=1|2|4` bounds conversion
  admission and executor size (default 1); counts above one need full-model
  qualification. Opt-in frontend diagnostics also separate preparation
  queue/start/finish/API-resume stamps, carried by `managers/io_struct.py`
  through `tokenizer_manager.py` to `req_time_stats.py`. The API loop
  receives outputs; cancellation retains admission until conversion completes,
  and shutdown drains owned work. Default off creates no worker. The controlled
  chat-only diagnostic is qualified separately from mixed-endpoint tokenizer use;
  no kernel, cache, token IDs or sampling change is intended. Regression checks
  are in the existing serving-chat unit suite.
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

Inventory relative to [flashmla-v0.5.18/](flashmla-v0.5.18/).

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

The following are historical v0.5.18 records, not v0.5.22 validation or tests of
every later commit. Read submitted source hashes, binary identities and validity files in each collection before
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
git -C third_party/sglang-v0.5.18 diff --name-status \
  71de97b264b04dcd514cf904003028aefe9775c8 \
  "$(git rev-parse HEAD:third_party/sglang-v0.5.18)"
git -C third_party/flashmla-v0.5.18 diff --name-status \
  05e26647fe840b8baedae486c2d86d5ce4efeb7c \
  "$(git rev-parse HEAD:third_party/flashmla-v0.5.18)"
git log --oneline -- third_party/sglang-v0.5.18 third_party/flashmla-v0.5.18
```

Use `git ls-tree <parent-commit> third_party/sglang-v0.5.18 third_party/flashmla-v0.5.18` to
recover an integration's exact pair. Git history remains the detailed record;
this guide and the changelog are its curated index.

## GLM-5.3-Flash candidate (v0.5.22 only)

Author: Winston Cai.

GLM Packed is opt-in with `--glm5-mla-cache-format remnant` and
`--glm5-mla-cache-implementation adapter|direct` (Direct by default). Native
remains unchanged. The fork owns `srt/remnant/glm_config.py`, `glm.py`, and
`glm_pool.py`: two planes of 256 raw E4M3 bytes and eight uint64 bitmap words,
with no DeepSeek scale plane. Raw latent selection precedes the existing GLM
RMSNorm; the producer stores once and the DSA backend skips the ordinary write.
Only hybrid full-attention storage changes; KDA, indexer and tail ownership stays
with the existing pools. Full-page relocation and retraction carry both planes.
The Adapter shares one 128-query workspace; Direct specializes only TileLang's
KV tile loads. All Native FP8 attention and combine arithmetic is retained.

Shared launchers select `MODEL_FAMILY=glm53`; Modal image selection additionally
uses `REMNANT_MODEL_FAMILY=glm53 REMNANT_INTEGRATION=v0.5.22`. Maintained
`kernels.py::validate_kernels --model-family glm53` owns model-free qualification.
Full weights/serving are outside the current $15 authorization. See
`benchmarks/microbench/glm53-packed/README.md` and its canonical JSON for exact
coverage and promotion status. FlashMLA and both v0.5.18 pins are unchanged.

Original runtime qualification (historical): 90 H100 numerical and 11 synccheck
tests passed; racecheck/memcheck and independent review were incomplete then.
Current optimized-loader focused qualification and decode results are recorded
below. Broad runtime qualification remains incomplete; Packed stays opt-in.

GLM attention microbenchmark: the maintained entry point now accepts
`--model-family glm53 --suite glm-attention`, comparing unpruned Native with
Packed Direct in 128 paired graph cases. The fork owns the GLM fixture/reference
module and shared timing primitives. Modal validates explicit per-tool testcase
coverage and matching production runtime/compiled-artifact identities before
timing. This does not change production attention kernels or serving defaults.


Early harness failures are historical: the initial four-bank memory-accounting
assertion was replaced with graph-private segment accounting. That repair and
the resulting full graph path have since executed on H100. Failed attempts and
the earlier incomplete128-case comparison remain in the canonical experiment
record; the completed48-case optimized decode scope is described below.

GLM exploratory timing is explicitly selected with
`benchmark_kernels --model-family glm53 --suite glm-attention --allow-incomplete-qualification`
plus source-bound `--qualification-runs`. Missing sanitizer coverage remains
recorded and the experiment is unqualified; this does not promote Packed or
change the default gate. See the canonical GLM experiment README/ledger.

GLM Packed cooperative loader: `sglang-v0.5.22/python/sglang/kernels/jit/include/sgl_kernel/glm_packed_decode.cuh`
is embedded by the Packed TileLang specialization. It owns raw-byte expansion only;
the 320-byte cache ABI and Native arithmetic remain unchanged. Decode-only
benchmark scope uses `--model-family glm53 --suite glm-attention --phase decode`.

The optimized GLM decode comparison uses `--phase decode` (48 cases; default
`all` still requires 128). The helper is embedded through TileLang `import_source`;
the producer/320-byte ABI and Native arithmetic are unchanged. The maintained
`test_glm5_remnant.py` owns an exact-cubin Driver launcher for one-CTA sanitizer
checks; modules outlive graphs and explicit streams own launches. CPU references
and verified binary reuse avoid instrumenting unchanged reference/framework paths.
The optional seven-row/batched-transfer fixture trial was rejected and reverted
because it did not improve tool runtime and its memcheck was incomplete.

All 48 paired cases completed at SGLang `246ee03c1ad5`, with 1.60% batch-weighted
overhead. Final fork `23e2e08` has the identical Git tree. Focused sync/race/memcheck
passes; full-grid/combine instrumentation and broad qualification remain incomplete.
See the experiment README, canonical JSON and decode CSV for identities, every
case, costs and limits. Native remains default; rollback is Native or `889edf56`.
