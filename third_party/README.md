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
| `sglang-v0.5.22/` | `baa9befa381511cd4d56f83980951e5a45fffc6d` (peeled release commit) | `exp/ds41-flash/upstream-decode` | `6ba5517ea595` |
| `flashmla-v0.5.22/` | `3620b63fc4ebe33950c50e1b7701b6607f44150e` (selected by SGLang v0.5.22) | `exp/ds41-flash/upstream-decode` | `858bb1c` |

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
to be clean, including untracked files. Follow the root [source safeguards](../docs/agents/git-and-sources.md)
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

## Experimental V4.1 FP4 on SM90

Qualified decode-only follow-up: SGLang `8084dc44`, FlashMLA `858bb1c` unchanged.
The isolated original Native control now carries synchronization-only corrections
at `6fbbf3a`; the complete patch and original base are preserved in the experiment
record. It is identified as synchronization-corrected upstream Native, not the
untouched failed binary. Protocol v5 and `v41-decode` qualify two production-page
decode graph/stream cases per sanitizer before the ten-case timing matrix.
CPU preflight, three ordinary H100 cases, all three strict decode sanitizers and
independent reviews passed. Packed took 2.56× as long as synchronization-corrected
Native across ten cases (range 1.72×–3.42×). Observed follow-up spend was $1.63.


Upstream comparison candidate: SGLang `3c595a2b`, FlashMLA `858bb1c` unchanged.
The v4 microbenchmark defines Native as the original upstream H100 auto
FP8/BF16 path (584 bytes), built separately at FlashMLA `3620b63`; the previous
custom dense FP4 control is now `dense_fp4` (288 bytes). Packed remains 224
bytes. Prior v1–v3 ratios and allocator fixtures used the custom dense FP4
control and do not measure upstream Native. CPU matrix, ratios and rounding
contracts passed. On frozen parent `7e913f5`, the completed Modal CPU preflight
passed 27 tests + 15 subtests, followed by 43 ordinary H100 numerical/graph/stream
cases. Strict synccheck then reported 4,192 divergent-thread barrier errors in
the original upstream H64 sparse kernel (`splitkv_mla.cuh:418`); no complete
synccheck XML was produced. Racecheck was interrupted after 7/38 cases, memcheck
was not run, and timing stayed blocked. This comparison is unqualified;
independent fresh review is also pending. The baseline remains unchanged.
No runtime or kernel arithmetic changes. Source/binary identities bind the
separate control to the new receipts; the required sanitizer matrix includes
both global page sizes (38 cases/tool). All task containers are stopped;
conservative cumulative compute is $18.15 under the approved $25 ceiling,
retaining the $2 reserve. Prior allocator results compare custom dense FP4
against Packed, not upstream `auto`.
See the single [experiment record](../benchmarks/microbench/v41-packed-fp4/README.md).
Rollback omits the timing suite or restores SGLang `4f8fb091`; runtime `auto`
continues to select its existing path.

Follow-up candidate: SGLang `4f8fb091`, FlashMLA `858bb1c`. Effective prefill
resolution now accepts a Q8 CLI choice disabled by the environment; shared page
validation and named CUDA dispatch modes reduce duplicate contracts. A synthetic
C2 decode test exercises the actual producer, request ring and FP4 store.
Interrupted sanitizer evidence is audited before complete sibling receipts can
be reused. These pins are pending fresh validation; prior qualification does not
transfer automatically. See the experiment record for budget and deferred gates.

Previous review-fix integration: SGLang `c4708870`, FlashMLA `4e72722`, based on
parent `af12dd9`. Those pins added effective Q8KV8 override rejection, retained
prefill policy, aligned cache base/page checks, named FP4 layouts and explicit
fixed-matrix CLI validation. Parent qualification uses
`scripts/modal/qualification.py` to reject failed tools, compare complete runtime
identities and check receipts on CPU before GPU timing. Those V4.1 source identities passed the fresh model-free numerical, graph/stream, allocator,
34-case/tool sanitizer and review gates using parent runner `2707db9`. The final
v3 ten-case timing is complete; historical evidence is kept separate below.

Prefill still reconstructs the selected requests' full compressed history into
an inherited BF16 scratch buffer. Its high-watermark allocation scales roughly
as `1024 * (sum(floor(sequence_length/compression_ratio)) + SWA_union)` bytes
per GPU. At C1, 32 requests of 128K need about 4 GiB before the SWA union and
padding. The microbenchmark's small scratch fixture does not establish serving
memory safety or concurrency capacity. This review does not redesign prefill.

The isolated `ds41-flash-packed` candidate starts from parent
`9419d0e18e1e7677c0e48b8e40ad54bbc30010ca`; the original integration/worktrees
remain unchanged. Its original model-specific selector is replaced in the merged
v0.5.22 interface by `--kv-cache-format native|direct`. Native always selects the
existing FP4 global-cache path; Direct selects Packed FP4. Both require
SM90 FlashMLA and V4.1 ratio-1/2 shared global caches; HiCache, distributed cache
transfer, speculation/DSpark, replay and legacy layout overrides are rejected.

- **Storage/producer ownership:** SGLang's `kernels/ops/attention/dsv4/fp4_cache.py`
  owns software SM90 E2M1 packing and BF16 reconstruction. It takes normalized
  BF16 rows, applies upstream RoPE/BF16 rounding, quantizes with per-16 E4M3
  scales, then optionally retains the 256 largest dequantized magnitudes.
  Ties retain lower coordinates; codes/scales are unchanged. Indexer input is
  unpruned. Native is 288 bytes; Packed is 224 (128 codes + 64 bitmap + 32 scales).
- **Runtime ownership:** grouped model args, the existing remnant configuration
  boundary, `deepseek_v4_memory_pool.py`, `pool_configurator.py`, and
  `deepseek_v4_backend.py` select layouts, budget physical reserved rows and
  dispatch. Existing SWA storage and FP8 Remnant interfaces are retained.
- **FP4 prefill ownership:** FlashMLA `sm90/prefill/sparse/phase1.cuh` has an
  opt-in named-barrier policy and two FP4-only D512 instantiations. Separate
  `fp4_sparse_prefill_fwd`/`flash_mla_fp4_sparse_fwd` entries serve reconstructed
  BF16 caches and SWA-only layers in explicit FP4 modes. Stock/auto retain the
  existing policy. SGLang owns AOT registration and explicit-mode dispatch.
- **Direct ownership:** FlashMLA's separate `flash_mla_with_fp4_kvcache` entry
  point and SM90 template specializations read global FP4 beside V4 SWA. Packed
  data rows hold ascending survivors and eight MSB-first uint64 bitmap words;
  even ranks occupy low nibbles. Pages store all data rows then all scale rows.
  Scale lookup uses original coordinate groups. Global records already have
  RoPE applied. SGLang's AOT CMake, extension registration and Python binding
  expose the same contract without importing experiment code.
- **Validation:** model-free fixtures extend the existing tests and Modal runner.
  CPU native build/precompile passed before H100, along with 24 CPU tests +
  15 subtests, six receipt/dispatch tests, 17 launcher regressions and 103 V4
  CPU regressions + 45 subtests. Scoped style/registry checks passed. H100
  validation passed 39 ordinary cases and 34/34 under each strict memcheck,
  racecheck and synccheck tool, with zero errors/hazards. Independent source
  and artifact review found no unresolved finding. Complete e/f tool receipts
  match one runtime/source/binary identity; overall interruptions remain failed.
- **Timing/allocation:** the fresh v3 H64/window128/SWA-page256 matrix has ten
  cases across C1/page256 and C2/page128. Packed/Native geometric-mean latency
  changes are +1.30% overall, +1.72% C1 and +0.87% C2 (range −1.02%..+2.71%).
  These warm synthetic graph results establish no serving or repeatable speedup.
  The fixed-cap 1 GiB allocator fixture again gains 21.96% token capacity;
  global records shrink 22.22%. Historical v1 timing is separately retained.
- **Scope/rollback:** experimental and opt-in; no weights, full-model loads,
  serving, quality, H20, SM100 execution, GPUQ or profiling qualification.
  Follow-up compute is conservatively about $14.87 before settlement, under
  the $20 ceiling with its reserve intact. All task allocations are stopped.
  Full-history prefill scratch scaling above remains a serving limitation.
  Select `--kv-cache-format native` or restore recorded baseline pins to roll back.

See the [single experiment record](../benchmarks/microbench/v41-packed-fp4/README.md)
for exact sources, commands, all cases, costs, failed attempts and limitations.
Broader GitHub lint/gating statuses are separate from these focused Modal gates.

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

Baseline target: 0731 on H100/SM90, **V4 584-byte logical Native layout**.
The separate experimental V4.1 candidate above extends this baseline.

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
selected by `--kv-cache-format direct` on v0.5.22 (the unchanged v0.5.18 pin uses
`--dsv4-c4-cache-format remnant --dsv4-c4-cache-implementation kernel`).

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

Native remains the default. `--kv-cache-format direct` selects the 320-byte
Packed MLA pool and TileLang Direct. Adapter remains an explicit internal test
and benchmark input. Packed requires the pinned GLM text
geometry, FP8 KV and TileLang sparse attention. Startup rejects unsupported
combinations before pool allocation, including a non-BF16 resolved `auto` dtype.

Eleven MLA layers own two persistent planes per slot: 256 raw E4M3 bytes and
eight uint64 bitmap words (320 bytes). Selection precedes the existing RMSNorm;
the producer writes once and the backend skips the ordinary KV write. The
existing KDA, indexer, tail and allocator ownership remains unchanged. Adapter
reconstructs into a shared 128-query workspace; Direct expands inside the
TileLang KV tile and reuses it for QK and PV. Native arithmetic is retained.

| Contract | Owner under `sglang-v0.5.22/python/` |
|---|---|
| Geometry and startup | `sglang/srt/remnant/glm_config.py` |
| Producer and Adapter | `sglang/srt/remnant/glm.py` |
| Persistent planes and lifecycle | `sglang/srt/remnant/glm_pool.py` |
| Direct TileLang and loader | `sglang/kernels/ops/attention/dsa/tilelang_kernel.py`, `sglang/kernels/jit/include/sgl_kernel/glm_packed_decode.cuh` |
| Fixtures and timing | `sglang/benchmark/remnant/glm_attention.py` |
| CPU compile/artifact manifest | `sglang/benchmark/remnant/glm_artifacts.py` |

The parent `scripts/modal/kernels.py` owns image preparation, execution and
export; `scripts/modal/kernel_qualification.py` owns qualification policy.
Set `REMNANT_INTEGRATION=v0.5.22` and `REMNANT_MODEL_FAMILY=glm53` to select
GLM. `REMNANT_COMPILED_IMAGE` and `REMNANT_GLM_CPU_REBUILD_FROM` are mutually
exclusive. Ordinary reuse checks remain strict; the explicit CPU rebuild can
verify a limited, recorded formatting transition and retained AOT binaries.

The attention benchmark uses `--model-family glm53 --suite glm-attention`;
`--phase decode` runs 48 cases, while default `all` requires 128. Resume requires
a complete verified build manifest plus matching source, artifact and fixture
identities. Individual sanitizer-tool runs remain partial qualification.
See [the experiment index](../benchmarks/microbench/glm53-packed/README.md) for
commands, all cases, budget, source identities and historical evidence.

Historical optimized decode at SGLang `246ee03c1ad5`: all 48 paired cases and
30 rounds per case completed on one H100, with +1.6010% batch-weighted overhead.
Focused Direct numerical/graph and three-tool checks, and Adapter >2 GiB
numerical and three-tool checks, subsequently passed at `9185c6d25115`. These
results qualify their recorded sources; they do not automatically qualify the
current pin. The expanded producer/pool/public-dispatch GPU test has not run.
Full-grid/combine sanitizer instrumentation, model quality and H20 execution
remain unverified. The four-H200 text-only serving scope is recorded below.

### Full prefill kernel comparison

The opt-in 64-row Packed TileLang prefill candidate is exercised by the
maintained GLM benchmark with `--phase prefill --packed-prefill-tile 64`. It
uses two splits and reuses the unpacked latent tile in QK and PV. The H200
suite completed 80 paired cases at both local head counts with focused
sanitizers. Packed Direct was faster in 70 cases and missed the strict
per-case parity criterion. The serving launcher uses its production dispatch.
See the [GLM experiment guide](../benchmarks/microbench/glm53-packed/README.md)
for reproduction and the source-bound evidence.

### Full-model H200 smoke test

The [SGLang cookbook](https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash)
uses TileLang DSA and DeepGEMM on H200. Its verified high-throughput recipe
uses TP8/EP8 with BF16 KV. The Remnant smoke test uses TP4/EP4, FP8 E4M3 KV,
text-only loading, an 8K context and decode graphs up to batch four. Native
and Packed Direct use the same checkpoint and launcher settings. The smoke
uses the Python radix-tree backend supplied by the verified image.

From the parent worktree, with a Modal profile that can access the verified
GLM image, run:

```sh
export REMNANT_INTEGRATION=v0.5.22 REMNANT_MODEL_FAMILY=glm53
export REMNANT_COMPILED_IMAGE=<verified-cpu-built-image>
modal run scripts/modal/serve.py::download_model
modal run scripts/modal/serve.py::prepare_glm_serving
modal run scripts/modal/serve.py::smoke_glm_serving --run-id <unique-run-id>
```

For a deliberate experiment against an uncommitted candidate, set
`REMNANT_SGLANG_SOURCE_OVERRIDE` to its checkout. The runner records the
source content hash and verifies the compiled-source digest before overlaying
Python files. Ordinary launches still require exact clean pins.

The checkpoint function downloads the pinned official FP8 weights once to
`glm53-flash`. The CPU preflight verifies checkpoint receipts, shard bytes,
the config hash and the LLM CLI. The four-H200 function mounts the checkpoint
read-only and starts one loopback server at a time. Each mode must return HTTP
200 from `/health` and a nonempty OpenAI-compatible chat completion. Logs and
responses go to `remnant-stage2a-results/glm53-serving/<unique-run-id>`.

The [four-H200 run](https://modal.com/apps/caiw/main/ap-kOVXUiAnMymNtwN0njkeOe)
loaded all 62 official FP8 shards and captured decode graphs on every rank in
both modes. Both returned HTTP 200 for `/health` and `/v1/chat/completions`.
Each response contained `1, 2, 3, 4, 5` with finish reason `stop`. The tested
Modal serving image is `im-VvBWmlcKXOGyVuHTE6kVxx` in `caiw`. It used an
explicit source override with benchmark-only prefill edits alongside the
serving fixes. The 64-row prefill experiment was outside this smoke scope.
The [qualification record](../benchmarks/microbench/glm53-packed/qualification.json)
contains the exact image and source identities, commands, response bodies,
log hashes and costs. This was a single-request smoke per mode. Serving
performance, capacity and multimodal behavior remain unverified.

Rollback is Native format or the pre-optimization SGLang pin `889edf56`. The
v0.5.22 FlashMLA pin and both v0.5.18 pins are unchanged.
The fragment reconstruction follow-up stages Packed FP4 rows and prefix ranks
once, reusing V4's fragment selectors with nibble adaptation. Producer geometry
and persistent formats stay fixed. The maintained microbenchmark now includes
`v41-fp4-decode`. Its active v3 matrix uses H64/SWA128 with 256-token SWA pages, batches
8/16/24/28/32, and both C1/page256 and C2/page128 (ten cases), with
actual captured-output checks and fresh dense Native FP4 controls. Timing is
blocked until identical source/binary/tool receipts pass 34 cases per sanitizer.
The prior source pins passed scoped model-free correctness and v1 timing.
Fresh-context source/artifact review passed with no blocking findings. The v3
fixture correction passed its two existing matrix/contract tests on Modal CPU
(2 passed in 0.14 s) and changes no runtime/kernel code. Historical v1 measurements
retain their original source identities. Fresh v3 GPU timing is recorded in the
review-fix milestone; historical timing remains unchanged. V2 incorrectly equated
the 128-token
window with page128; the runtime factory requires page256. The measured
allocator fixture and capacity prediction already used page256.


## Combined serving interface

On v0.5.22, `--kv-cache-format {native,direct}` replaces all model-specific
format/reader selectors. Removed flags produce ordinary unknown-option errors.
`direct` maps to Packed C4/FlashMLA for V4, Packed FP4/FlashMLA for V4.1, and
Packed MLA/TileLang for GLM. Native is the default. Model family is resolved from
loaded configuration and carried through spawned-worker publication; unsupported
Direct models/profiles fail before pool allocation. Internal Adapter and Triton
reconstruction fixtures retain their original benchmark mode names.

The merge candidate is `fix/glm53-flash/ds41-integration` in parent and SGLang,
with GLM/V4.1 published heads preserved as merge ancestors. Shared parent
metadata is owned by `scripts/model_profiles.py`; launch normalization and named
environment transport by `scripts/local/launch-config.sh`; compiled-image
verification by `scripts/sources.py`; pure report parsing by
`scripts/modal/qualification.py`. Model-specific evidence policies remain
separate. Shared numerical timing statistics live in the fork's `timing.py`.

The current merge-validation status and $26 phase ledger are recorded in the
existing GLM [qualification record](../benchmarks/microbench/glm53-packed/qualification.json).
Historical qualification and prefill parity misses stay bound to their measured
sources. The 64-row prefill candidate remains benchmark-only.

The combined runtime at parent `c23b32f` / SGLang `e0d6d50` passes CPU checks,
GLM numerical/allocation/focused three-tool checks, and Native/Direct full-model
smokes on four H200s. V4.1 passes 39 numerical/allocation cases and all 15 focused
memcheck/synccheck cases. Full focused racecheck fails during Native C1 setup,
before the FP4 store callback; an isolated synchronized case passes but does
not resolve the batch failure. The first V4.1 serving attempts loaded weights,
then rejected a 0.86 memory fraction. The corrected 0.90 profile subsequently
passed Native and Direct smokes concurrently on four H200s per mode, using
parent `f720d20` and image `im-SXCseSK7MhV0kyLb7TPdbJ`. Both loaded all ranks,
captured decode graphs and returned the exact expected chat answer with `stop`.
The separate $15 phase spent $6.24 reported / $6.68 conservatively accounted;
all workers exited. The full-batch racecheck gap prevents complete merge acceptance;
the historical 34-case qualification above does not qualify this merge.

The upstream-decode experiment adds only benchmark/provenance code. It loads
the exact upstream v0.5.22 attention module alongside production Direct on one
H200, with 16-head TP4 geometry and 24 decode cases. It does not establish
full-model serving performance. See the GLM experiment README and ledger.

The source-pinned 24-case upstream decode comparison passed on H200: Direct
adds +1.28947% batch-weighted complete-call overhead, with all numerical and
changing-input graph checks passed. These are attention-only results; see the
[per-case table](../benchmarks/microbench/glm53-packed/README.md#completed-upstream-decode-result).
