# Fork integration changelog

Author: Winston Cai.

Meaningful integrations, newest first. Dates are **parent integration commit
dates**, not benchmark dates or releases. Each entry records the parent commit
and its exact SGLang/FlashMLA pin pair (unique abbreviated SHAs). Entries summarize
committed changes, not automatic validation or performance promotion.

See the [fork guide](README.md) for the current file map, runtime contracts and
source-bound validation evidence. Rejected tuning attempts remain in Git/run
records rather than this milestone log.

## Unreleased

- Parent baseline `1b19ad9`, SGLang `99499fb02c86` / FlashMLA
  `4e72722671d8`: add a source-verified pristine upstream FP8 attention baseline
  and a separate 24-case TP4 decode protocol on one H200. Production kernels,
  persistent formats and serving defaults are unchanged. Fork CPU contracts:
  134 passed, 103 GPU skips; independent review passed. Paid preparation and
  timing outcomes remain pending in the GLM ledger's new $5 phase.

- Main integration from parent `cbfeeba` and main `dac0205`, retaining SGLang
  `e0d6d500d6dc` / FlashMLA `4e72722671d8` and both v0.5.18 pins. Preserve main's
  build locks, split guidance and validation tier; reconcile sanitizer subsets
  with both receipt readers. Retain historical replay records and migrate the
  V4 replay caller to Native/Direct, freezing its shared launch configuration.
  CPU checks cover 157 distinct tests under both integration profiles (155 full
  suite plus the final 22 replay checks); fresh-context review passed. No image
  rebuild or GPU execution. User authorized publication with the original V4.1
  full-batch racecheck failure still explicitly unresolved; FlashMLA unchanged.

- Measured parent `f720d20`, SGLang `e0d6d500d6dc` / FlashMLA `4e72722671d8`:
  V4.1 Native and Direct full-model serving smokes pass concurrently, each on
  four H200s, with the corrected 0.90 memory fraction and bounded 16K pool.
  All ranks load and capture batches 1/2/4; health/chat return 200 and the exact
  expected answer with `stop`. New phase cost: $6.24 reported / $6.68 conservative
  under its separate $15 cap. Workers stopped. Preserve earlier failed runs;
  the full-batch racecheck issue remains separate and unresolved.

- Runtime freeze parent `c23b32f`, SGLang `e0d6d500d6dc` / FlashMLA
  `4e72722671d8`: GLM Native and Direct full-model smokes pass on four H200s.
  V4.1 numerical/allocation (39 cases), memcheck and synccheck (15 cases each)
  pass. Full focused racecheck fails in Native C1 setup/projection; the isolated
  case passes under expanded instrumentation but the batch failure remains
  unresolved. Both V4.1 serving modes require reruns after the CPU-tested memory
  profile fix. All workers are stopped; compute is $20.50 reported / $21.57
  conservatively accounted under the $25 phase cap, with its $3 reserve intact.
  Keep merge acceptance pending; no numerical kernels or defaults changed.

- Serving follow-up from parent `a56cb7b`, with SGLang `e0d6d500d6dc` and
  FlashMLA `4e72722671d8`: all four smoke workers loaded their checkpoints.
  GLM Native passed; Direct captured graphs and became healthy, but its cold
  chat request timed out. Both V4.1 modes stopped before KV allocation because
  GLM's 0.86 static memory fraction is below V4.1's observed 0.864 minimum.
  Give the smoke profile its own fraction (V4.1 0.90, GLM unchanged); the
  environment regression and independent review pass. Corrective GPU checks
  remain pending. Preserve the failed runs and all charges in the GLM ledger.

- Parent follow-up from `c0e59ec`, SGLang `e0d6d500d6dc` / FlashMLA
  `4e72722671d8`: add independent Native/Direct serving workers for GLM and
  V4.1-Flash-1017, pinned metadata, per-mode artifact ownership, and CPU binary/
  cache admission. Full parent CPU checks: 116 passed under each integration.
  Preserve all model/kernel ABIs. The focused V4.1 audit remains unqualified
  after an internal sanitizer exception; broaden its filter consistently and
  install the submitting SDK explicitly for legacy CPU image builders.
  No new four-way serving result or broad sanitizer completion is claimed.

- Integration candidate from parent `8a3d5b2` + V4.1 `d385c1c`; combined SGLang
  `e0d6d500d6dc` (descends from `6667e5d` and `c470887`) / FlashMLA
  `4e72722671d8`: merge both model implementations and replace public model-specific
  selectors with `--kv-cache-format native|direct`. Preserve numerical kernels,
  persistent ABIs and v0.5.18 pins. Share pinned metadata, source verification,
  report parsing, launcher normalization/transport and pure timing statistics.
  Merge CPU and GPU validation is tracked separately in the existing GLM ledger;
  historical serving/timing records do not qualify this new source automatically.
  CPU preparation now checks V4.1 Native/Direct allocation coefficients through
  the production selector; shared fixture cases remove stale pre-migration labels.
  This correction changes tests only, preserving all numerical/runtime content.

- Measured parent `d3aa23f`, v0.5.22 SGLang `6667e5d` / FlashMLA
  `3eb8594`: qualify the opt-in 64-row Packed prefill candidate for the full
  80-case paired H200 benchmark. Both head variants passed focused tools and
  numerical/graph checks. A source-bound resume recovered the final two cases.
  Direct was faster in 70 cases, with a 1.11% worst slowdown. Keep the candidate
  benchmark-only. Modal gross metered $4.01 under the $5 cap.

- Measured parent `8a9a939`, serving fix SGLang `d748fe5` / FlashMLA
  `3eb8594`: support standalone GLM text-only loading and use the active MLA
  cache pool from the forward context. Add CPU checkpoint and serving-image
  preflight and bounded Native/Packed Direct smoke. Four H200s at TP4/EP4
  loaded the pinned FP8 checkpoint, captured decode graphs, and returned
  HTTP 200 and a complete response in both modes. The measured source override
  also contained benchmark-only prefill edits. See the qualification record
  for source hashes, image, commands, failure history and costs.

- Parent implementation `0e3f94b`, v0.5.22 SGLang `5174462` / FlashMLA
  `3eb8594`: reject conflicting GLM image modes and resolved non-BF16 Packed
  activations; bind benchmark resume to complete verified source/artifact
  identity and raw-sample statistics; terminate owned compiler process groups
  on failure. Centralize host geometry and document byte expansion. Expand the
  model-free producer/pool/public-dispatch GPU test. CPU startup/benchmark,
  targeted orchestration, formatting and independent reviews passed; the new GPU
  test was not run and historical H100 evidence remains source-bound. Compact
  the experiment ledger while preserving its immutable Git history and a verified
  local archive. Native default and other pins remain unchanged.

- Parent preparation `0639722`, v0.5.22 SGLang `979839b` (runtime-tested
  `9185c6d25115`) / FlashMLA
  `3eb85944fd7d`: resolve fork lint debt, preserve immutable historical fixtures,
  register executable CPU tests and explicit custom-build CUDA ownership, and
  propagate pytest exit codes. Explicit CPU preparation verifies a five-file
  format-only transition with raw hashes, pinned clang-format and Python3.12 AST
  identity before retaining base AOT binaries; ordinary reuse remains strict.
  Independent review and six rebuild regressions pass. Expanded local checks:
  176 passed, 104 GPU skips and 10 subtests; previously failing lint hooks pass.
  CPU preparation and fresh focused H100 Direct/Adapter numerical, graph and
  memcheck/racecheck/synccheck gates pass; broad instrumentation remains incomplete.
  GitHub lint is green, `run-ci` is applied and its gate passes. Seven missing
  flag rows fix the newly exposed CPU docs failure; broad GPU CI lacks runners.
  Canonical evidence preserves both interrupted attempts and all charges.
  The user added $5 and 15 H100 minutes; retain every previous cost and failure.


- Parent integration `4150a42` (runtime fixes `6f83b1f`), v0.5.22 SGLang
  `fbd8a02269f7` / FlashMLA `3eb85944fd7d`: address review findings in Adapter int64 addressing, effective
  attention dispatch, single-mode timing, named container configuration, TP/EP/
  GPU consistency and the CPU budget regression. Move complete SM90 manifests
  to a fork-owned module; separate parent evidence policy from orchestration,
  retain immutable legacy hashes and safely CPU-recompile older verified images.
  Validation: 126 SGLang CPU checks pass, 103 GPU benchmark checks skip; focused
  parent/independent reviews pass. All 108 parent CPU tests pass under each of
  v0.5.18 and v0.5.22 after the committed pin update. No image build or GPU run;
  the new Adapter boundary regression and rebuilt-artifact execution remain
  unverified. Native and other pins unchanged.


- Parent implementation `06f2058` (measured parent `8aea70a`), v0.5.22 SGLang
  `23e2e08` (tree identical to timed `246ee03`) / FlashMLA `3eb85944fd7d`:
  cooperative V4-inspired Packed loader with vector loads/stores, register bitmap
  prefixes, swizzled tile stores and int64 guards. All 32 CPU variants pass SASS/
  resource gates; Native cubins unchanged, Direct POPC34, no spills, matching two-
  CTA occupancy. All 48 decode cases/30 paired rounds complete on one H100:
  batch-weighted overhead 1.6010% (TP4 0.6829%, TP8 2.5192%), no 2× cases.
  Focused one-CTA numerical/graph/memcheck/racecheck/synccheck passes; full-grid/
  combine instrumentation and broad qualification remain incomplete. Verified CPU
  artifact reuse supports harness-only edits. Optional fixture compaction failed
  its runtime goal and was reverted, preserving its source and incomplete reports.
  All workers stopped; additional phase $10.335584 under $15, no pushes, Native
  defaults and other pins unchanged.

- Measured parent `853e4d6`, v0.5.22 SGLang `889edf56b332` / FlashMLA
  `3eb85944fd7d`: saved56 complete Native FP8 TileLang/Packed Direct H100 cases,
  30 paired rounds each; all measured TP4 cases favor Native. Descriptive only:
  72 cases and7 memcheck cases remain unverified. Racecheck/synccheck13 cases
  each reconcile cleanly. All allocations stopped; conservative cost$19.390829
  under user-updated$20 cap. Canonical JSON and flat CSV retain raw samples and
  source-bound resume history. Production kernels/defaults and all pins unchanged.

- Parent based on `5c8b464`; v0.5.22 SGLang `889edf56b332` / FlashMLA
  `3eb85944fd7d`: add explicit exploratory GLM timing when sanitizer coverage
  is incomplete, following user authorization. Default qualification remains
  mandatory; exploratory exports record exact missing cases and an unqualified
  experiment. Production kernels and all fork pins are unchanged. Seven focused
  launcher tests and real saved-evidence reconciliation pass (racecheck and
  synccheck 13 cases each; memcheck unverified). No timing claim yet.
- Review-fix integration parent `cc1b9a8`, CPU runner follow-up `2707db9`:
  SGLang `c47088700085` / FlashMLA `4e72722671d8`. Reject effective Q8KV8
  overrides and misaligned cache bases/page strides; retain startup prefill
  policy, name layouts and reject ignored geometry flags. Pure receipt checks
  reject failed tools and compare recorded runtime/toolchain identities on CPU
  before H100 timing, with a GPU recheck. Existing fixtures own regressions.
  CPU build/precompile, scoped style/registry, 24 + 15-subtest preflight, six
  new guards, 17 launcher and 103 + 45-subtest V4 CPU checks passed. H100 passed
  39 ordinary cases and 34/tool under all three strict sanitizers; independent
  source/artifact review passed. Final v3 timing: ten cases/600 samples, Packed
  latency +1.30% geomean overall (+1.72% C1, +0.87% C2), range −1.02%..+2.71%.
  Fixed-cap allocator gain remains 21.96%; full-history prefill scratch and
  serving/quality limits are explicit. Follow-up compute conservatively $14.87
  before settlement, all allocations stopped. Historical evidence is preserved.
  Rollback uses `auto` or baseline pins; no production promotion.

- Publish the completed V4.1 experiment from parent `81c0777` on matching
  `ds41-flash-packed` branches in all three repositories. Fork pins remain
  SGLang `f5afebd90750` / FlashMLA `e274b715f49a`; only v0.5.22 branch hints
  and integration documentation change. No runtime/kernel or validation change.
  v0.5.18 and the original development branches are preserved.

- SWA page correction from parent `c6c9f0f`: SGLang `f5afebd90750` /
  FlashMLA `e274b715f49a` (unchanged). The paged H100 factory requires
  SWA page256 while attention reads a 128-token window; v2 mistakenly used
  page128. V3 fixes only the fixture/protocol and adds a window-as-page
  rejection check. The two existing contract/V4 matrix tests passed on Modal
  CPU (2 passed in 0.14 s), cached image, no GPU run or native rebuild.
  Allocation measurements/prediction already used page256. Historical receipts
  remain unchanged; v3 GPU timing is unmeasured. CPU app stopped/zero containers.

- Production-geometry fixture correction from parent `60e5bbe`: SGLang
  `a0f7ff18314c` / FlashMLA `e274b715f49a` (unchanged). V4.1 FP4 timing now
  uses H64/SWA128/page128, Q1/K512, five batches and both C1/page256 and
  C2/page128 (ten cases). Protocol v2 rejects historical v1 timing; SWA64/H128
  remain correctness coverage. Existing matrix/timing contract and V4 matrix
  regression passed on Modal CPU (2 passed in 0.08 s). No native rebuild or GPU
  rerun; v2 GPU timing remains unmeasured. Both CPU apps stopped. Rollback uses
  the prior SGLang pin `98991169803f`; historical receipts/CSV remain intact.

- Follow-up to parent `cc0f82f`: SGLang `98991169803f` / FlashMLA `e274b715f49a`.
  Packed FP4 now stages survivors/bitmap prefixes once and reconstructs fragments
  with V4-style selectors. Adds two exhaustive-mask Direct fixtures and paired
  Native/Packed FP4 timing for all 20 H64/H128/batch/SWA cases. Scheduling, Native,
  persistent layouts and v0.5.18 pins remain unchanged. Fresh-context source
  review passed. From parent `8e9e3e9`, Modal CPU startup 24 tests + 15 subtests,
  ordinary 39 cases, all 34 cases under each strict sanitizer, and 20 paired timing
  cases passed. Packed Direct is 2.69%/2.58% slower than dense Native FP4 for
  SWA64/128 (geometric-mean case ratios); record savings stay 22.22%. Follow-up
  compute approximately $7.81 before billing settlement, all allocations stopped.
  Fresh-context source/artifact review passed with no blocking findings.
  No model/serving/production promotion.

- Experimental V4.1 FP4 integration validated from parent `9ec5971`, based on
  `9419d0e`. v0.5.22 pins: SGLang `0a36a662562f` / FlashMLA `91463e004ec9`;
  v0.5.18 and CUTLASS pins unchanged. Adds opt-in Native 288-byte/Packed
  224-byte storage, deterministic post-quantization 256-coordinate selection,
  SM90 mixed-SWA Direct, BF16 reconstruction/prefill, reserved-page accounting
  and model-free Modal fixtures. Separate FP4 prefill synchronization retains
  stock auto behavior. Native CPU builds/preflight, 37 ordinary cases, full
  32-case synccheck/racecheck and fresh-context review passed. **Unqualified:**
  workspace disablement interrupted memcheck after 27/32; no clean memory
  sanitizer summary/exit. Reported task compute $11.91, all allocations stopped;
  a full rerun exceeds the remaining work allowance while preserving the $2
  reserve. Physical 1-GiB allocation fixture: 656384 Native / 800512 Packed
  tokens (+21.96%); record-byte savings 22.22% are separate. No model quality,
  serving, performance or production claim. Source hashes, exact commands,
  failed attempts, costs and the missing gate are indexed by the
  [experiment record](../benchmarks/microbench/v41-packed-fp4/README.md).

- Reconciled GitHub main `ffb11cb` in parent merge `814aaa8`, retaining its
  v0.5.18 SGLang `6687a7c36c0c`, FlashMLA `97f3faa58450`, frontend preparation/
  diagnostics and HiCache launcher options. The v0.5.22 pair remains SGLang
  `7856615f3fdf` / FlashMLA `3eb85944fd7d`; the qualified FlashMLA commit is now
  published on `remnant/flashmla-v0522`. Historical qualification receipts are
  unchanged. Container preparation checks v0.5.22's grouped model fields rather
  than assuming the v0.5.18 `server_args.py` layout. Validation: 72 CPU launcher/
  source tests per profile and 19 replay tests passed, including positive and
  missing/stale argument-layout fixtures. No image build or GPU run; the separate
  lightweight Torch-fixture dependency issue remains unresolved.
- Both root build recipes are version-named (`Dockerfile.v0.5.18` and
  `Dockerfile.v0.5.22`). Local environment setup owns the integration/image-tag
  settings; local and Modal launchers select the matching recipe through
  `REMNANT_INTEGRATION`, retaining v0.5.18 as default. Fork pins are unchanged;
  70 CPU launcher/source tests pass under each profile. This packaging change
  has no image-build or GPU qualification.

## 2026-10-24 — Isolated v0.5.22 integration and versioned paths

Parent integration `0f0dec1`; versioned-path update `a10bc97`, based on local
`main` `6fa18c7`. v0.5.18 pins: SGLang `921cc2e03f61`, FlashMLA
`97f3faa58450` (unchanged). v0.5.22 pins: SGLang `7856615f3fdf`, FlashMLA
`3eb85944fd7d`. Both CUTLASS pins remain `147f5673d0c1`.

- **Integrated:** semantic port of `88e3b1a..62cb99a`, without the archived
  branch's unrelated experiment ancestry. Four independently pinned versioned
  checkouts; v0.5.18 remains default, v0.5.22 requires the explicit selector.
  Existing source/image gates and launchers use the selected pair; container
  installation paths and the v0.5.18 benchmark matrix remain unchanged.
- **Adapted:** V4 allocator/layout, grouped configuration, prefill/decode,
  CUDA store/Adapter/Direct, host-cache routing, capture and profiling interfaces.
  Upstream frontend conversion offload is retained. The V4 synchronization
  corrections include Native; do not call this an untouched Native binary.
- **Evidence:** [core model-free qualification](../benchmarks/microbench/v0522-qualification-20261023/README.md)
  on the unchanged v0.5.22 fork sources passed numerical tests, exact physical
  allocation reconciliation and all three sanitizers. JSON/CSV and original
  source identities are retained unchanged; raw artifacts remain archived.
- **Current parent checks:** 70 CPU launcher/source tests per profile, shell and
  Python syntax/import checks, mocked build arguments and clean recursive pins;
  all 20 training integration tests pass after updating the capture-image fixture.
  No parent image rebuild or GPU rerun. HiCache GPU, full-model/distributed and
  serving validation remain pending. Historical v0.5.18 results are not relabeled.
## 2026-10-23 — Preparation concurrency and queue attribution diagnostic

Parent `34e7339`; SGLang `6687a7c36c`; FlashMLA `97f3faa58450`.

- **Added:** startup-bounded DSV4 text-chat preparation worker counts 1/2/4
  (default 1, async off by default), matching semaphore/executor capacity,
  and opt-in queued/start/finish/API-resume timestamps. Cancellation retains
  the worker slot until its concurrent future finishes. Native kernels and
  packing contracts are unchanged.
- **Integration:** `serve.sh` preflights the selected mounted implementation and
  worker count. The existing frontend diagnostic adds a sequential four-cell
  Native/Packed Kernel C24 W1/W2 GPUQ screen, with exact-ID timing validation,
  source/image checks and owned server cleanup.
- **Validation:** 27 source-isolated model-free CPU checks pass, including
  synchronized shared fast-tokenizer conversion, request isolation, cancellation,
  shutdown and timing export/arithmetic. Scoped Ruff, Python/embedded-Python
  syntax, shell syntax and whitespace checks pass. Actual-image CPU checks run
  before GPU allocation; full-model worker-count qualification is pending.
  No GPU/Modal jobs, image rebuilds or pushes were performed for this change.
- **Rollback:** restore `DSV4_PREPARATION_WORKERS=1` or disable async preparation
  with `DSV4_ASYNC_PREPARATION=0`. Counts above one remain diagnostic candidates.

## 2026-10-22 — Bounded DSV4 chat preparation candidate

Parent `a64b9c0` (preflight correction `8c6e3e1`); SGLang `63291eba61`;
FlashMLA `97f3faa58450`.

- **Added:** default-off DSV4 text-chat conversion worker, bounded admission with
  cancellation ownership, unchanged receive timestamp and explicit shutdown drain.
  The local launcher preflights actual mounted Python source before server stop.
- **Validation:** 20 source-isolated CPU contracts and four local-launcher tests
  pass, including converter parity, native/AnyIO cancellation, errors, queue time
  and cleanup. Fresh-context runtime and launcher reviews found no scoped blocker.
  Actual-image timing 10/10, async preparation 7/7 and replay 19/19 CPU tests
  subsequently passed. C40/TP4/MF0.88/HiCache n=1 evidence transferred in
  `640b852a` shows post-180-second detokenizer→API mean delay fall from 6.183 s
  to 1.96 ms Native, 5.713 s to 20.89 ms Packed. Both now deliver about
  1.729 req/s; this is not a Packed speedup. Both full-run TTFT-p90 values exceed
  10 s and Packed misses 99% response validity (missing-output events). Retain
  default off; mixed endpoints and response failures remain unqualified. Native
  extensions are unchanged; see the frontend experiment record.
- **Scope/rollback:** controlled DSV4 text-chat only; other tokenizer-consuming
  endpoints are not qualified. Unset `DSV4_ASYNC_PREPARATION` to restore sync mode.

## 2026-10-22 — Opt-in first-output attribution

Parent `524207f`; SGLang `7a95ce8231`; FlashMLA `97f3faa58450`.

- **Added:** startup-gated scheduler send, detokenizer receive/forward and API
  receive/handler stamps, with first-packet latching and clock rebasing through
  pickle/msgpack. The replay adapter captures the existing SSE ID for exact
  client/server timing joins. No kernel, scheduling or cache-policy change.
- **Validation:** 13 source-isolated CPU transport/analysis tests and 19 replay
  tests pass. Full local SGLang import is blocked by unavailable Triton on macOS;
  actual VDI-image CPU import and all 10 timing tests subsequently passed,
  together with 19 replay tests. Instrumented full-model coverage remains pending.
- **Scope:** diagnostics are disabled by default. The recorded C40 HiCache
  experiment is attribution evidence, not throughput acceptance or production
  qualification. Existing binaries are unchanged; mounted Python source is
  preflighted before replacing a server. See the frontend experiment record.

## 2026-10-19 — Recorded-request serving benchmark client

Parent `d50974b`; SGLang `921cc2e03f61`; FlashMLA `97f3faa58450`.

- **Added:** one-time raw LongSWE-derived JSONL conversion with original
  messages/tools/order, recorded token caps and a separate hash/provenance
  manifest. Direct CLI commands reuse SGLang's existing OpenAI dataset/backend;
  the random-text `bench-serving.sh` wrapper remains unchanged.
- **Fixed:** optional recorded prompt-length metadata bypasses unavailable HF
  chat templates without changing the server payload. Tool-call-only output
  contributes to TTFT, and requested streaming completion usage must be present.
  Serving runtime and kernels are unchanged by this integration.
- **Validation:** AST syntax, `ruff check --select F401,F821` on changed Python
  files, and `git diff --check` passed. Added client regression cases have not
  completed locally: collection was blocked by missing `orjson`, then
  `pybase64`. Original-file conversion and image/GPU serving remain unverified.

## 2026-10-19 — Prefill-read benchmark and opt-in fusion

Parent `ba4efd8`; SGLang `40d07208195c`; FlashMLA `9f9a2e5da6e8`.

- **Added:** shared prefill-read timing/traffic fixtures and an opt-in fused
  unpack/RoPE reader preserving intermediate BF16 rounding (SGLang `40d07208195c`).
- **Validation:** retained split/fused studies cover numerical parity, graphs,
  sanitizers and traffic. Their submitted dirty snapshots are not the clean pin
  pair above. The candidate reduces preparation cost but does not beat Native;
  split remains the serving default. See the guide for export-recovery limits.

## 2026-10-18 — Capture/replay hardening

Parent `e390f5d`; SGLang `0098b4277fbd`; FlashMLA `9f9a2e5da6e8`.

- **Fixed:** integer indices for empty CSA captures, replay of already-compacted
  Native rows without double masking, and empty-domain LSE comparison
  (SGLang `0e94e2b32b4c`). Removed the unreachable dense-pruning branch.
- **Changed:** prepared-capture validation moved into the existing calibration
  client (SGLang `0098b4277fbd`); parent audit/preflight hardening also landed
  in `cd85909` before this integration.
- **Validation:** focused regression coverage was extended; no retained full
  live-TP4 capture qualification of this pair is claimed.

## 2026-10-17 — Unified measurements and per-layer Triton layouts

Parent `df9f458`; SGLang `6a728bf95c08`; FlashMLA `9f9a2e5da6e8`.

- **Added:** startup per-layer survivor counts 64..448 in steps of 64,
  variable-width pools and summed layer capacity accounting; non-256 layouts
  require Triton and reject HiCache (SGLang `6a728bf95c08`).
- **Integrated:** capture auditing, shared benchmark/numerical contracts,
  quantization-boundary regression coverage and production-sized cases
  (SGLang `98a60b79de`, merged by `1b41d2679fa5`; parent `c03cd2e`).
- **Included fixes:** spawned-worker settings, growing-prefill gather reuse,
  gated scheduler/graph annotations using PyTorch NVTX, tokenizer first-output
  metadata and an independent Packed store selector. These entered fork history
  across October 12–14; this is their parent integration milestone.
- **Validation:** coverage does not establish every non-256 serving/HiCache
  combination. Later evidence retains its own source hashes.

## 2026-10-11 — Explicit serving modes

Parent `0415009`; SGLang `8a520765f729`; FlashMLA `9f9a2e5da6e8`.

- **Changed:** expose Native, Packed Triton and Packed Kernel serving choices
  through startup configuration and parent launchers (SGLang `8a520765f729`).
  Packed modes share storage; their decode implementations differ.

## 2026-10-09 — H64 producer and FP32 store

Parent `ef62f97`; SGLang `b5d42fbc188b`; FlashMLA `9f9a2e5da6e8`.

- **Changed:** H64 eight-warp Packed decode producer (FlashMLA `2c1ef0a`) and
  fused selection/normalization/storage for FP32 serving inputs
  (SGLang `b5d42fbc188b`).
- **Fixed:** divergent H64 barriers and WG1 KV readiness (FlashMLA `0820d54`).
- **Tests:** extreme bitmap fixtures account for BF16 probability operands
  (FlashMLA `9f9a2e5da6e8`). Inclusion of these tests is not a qualification of
  every later build or serving setup.

## 2026-09-28 — Packed HiCache and loader updates

Parent `ba59edd`; SGLang `54ccadbe521c`; FlashMLA `3d7c7f5ef1c7`.

- **Added:** coordinated Packed-plane HiCache L2 transfers and routing
  (SGLang `f23c9b233d`, test transfer correction `54ccadbe521c`).
- **Changed:** replace the rank lookup table with arithmetic and use aligned
  word loads for fragment expansion (FlashMLA `7576073`, `1203a66`).
  The experimental asynchronous survivor staging was reverted (`86be361`).
- **Scope:** these fork changes originated September 24; the pair above is the
  September 28 parent snapshot. Its FlashMLA tip also includes incidental
  workspace metadata, not an additional kernel improvement.

## 2026-09-22 — Shared benchmark and local-source integration

Parent `407fcd5`; SGLang `0b2006980628`; FlashMLA `529c862003ee`.

- **Changed:** consolidate Native/Adapter/Direct microbenchmarks in SGLang and
  reuse FlashMLA's shared fixtures instead of separate Remnant runners
  (SGLang `0b2006980628`, FlashMLA `529c862003ee`).
- **Fixed:** Direct dispatch and local-fork build integration; the then-current
  HiCache integration was removed (SGLang `14b146242a`). It was reintroduced
  in the subsequent milestone, not silently treated as continuous support.

## 2026-09-21 — Fused Native-layout Adapter

Parent `2fb046b`; SGLang `84647d6284ff`; FlashMLA `a4b20774a4d6`.

- **Added:** fused CUDA Packed-to-Native reconstruction before stock FlashMLA
  (SGLang `84647d6284ff`), alongside the Direct Packed path.
- **Changed:** build/binding integration and attribution. Superseded loader
  experiments remain in fork history, not additional supported runtime modes.

## 2026-09-18 — Initial fork integration

Parent `48bcce5`; SGLang `e21dfe4bf417`; FlashMLA `40a1c73eabc9`.

- **Added:** Remnant cache format, Packed allocation, Triton storage/read paths
  and startup integration on SGLang v0.5.18 (SGLang `aac50712c6`).
- **Added:** SM90 Packed decode/API in FlashMLA (`40a1c73eabc9`) and SGLang
  Direct dispatch (`e21dfe4bf417`). Native remains the default.

## GLM Packed candidate

Author: Winston Cai.

Starting parent: 9419d0e18e1e7677c0e48b8e40ad54bbc30010ca. Starting v0.5.22
SGLang: 7856615f3fdf02e993f3b40ecdbbed8cb8ac9bf7; FlashMLA remains
3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8. Current candidate fork pin is recorded
in the parent tree and canonical GLM qualification JSON. Adds the separate raw
320-byte codec, producer/pool lifecycle integration, bounded Adapter and TileLang
Direct, startup guards and model-free GLM fixtures. Local startup contracts pass;
CPU SM90 compilation passed. Tested parent `5f4754e3015d2268df4b39175ce4622cd305589d`,
SGLang `9e42a0d8886dd93944dc6b47ef3bffef4e033b5a`, FlashMLA unchanged
`3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8`: 90 real-H100 numerical tests and
11 synccheck tests passed with zero reported synchronization errors. Racecheck
timed out; memcheck and independent fresh-context review remain unverified.
Physical Direct storage saves 277,364,736 bytes/rank at 131,072 token slots with
matched KDA reservation. The 54.39% capacity ratio remains a projection.
See `benchmarks/microbench/glm53-packed/qualification.json` for exact commands,
source/artifact identities, costs and failed attempts. Retain as an opt-in
candidate; rollback is Native or the starting SGLang pin. No serving/quality claim.

## GLM two-path TileLang attention measurement

Author: Winston Cai.

Based on parent `4521a1abb359c6e7d2354c4bdef3abc9702368b9`, with FlashMLA
unchanged at `3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8`. The parent gitlink
records the benchmark SGLang commit; production runtime/kernel content remains
identical to `9e42a0d8886dd93944dc6b47ef3bffef4e033b5a`. Adds the 128-case
Native/Direct graph protocol, independent references, bounded bank rotation,
shared paired timing and explicit per-tool sanitizer evidence. CPU validation
and independent review precede GPU execution. Exact outcomes, source/artifact
IDs and costs belong to the existing GLM qualification ledger; missing gates
remain incomplete. Rollback omits the GLM benchmark suite; no default changes.


Attention implementation commits: parent `9672503` and `d092f85`; SGLang
`f71841eade02afd1b601d8f1af72786b494bb6cb`, FlashMLA unchanged at
`3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8`. Two CPU image preparations passed;
new H100 smoke passed one graph fixture and failed the other at a negative
reserved-memory delta assertion. Sanitizers and timing did not start. SGLang
`02cbbd92b802e0437a8597b6b14ec45c994cf98b` repairs accounting using graph-pool
segments; 10 focused CPU contracts and independent targeted review pass, GPU
repair validation remains pending. Aggregate conservative cost is $5.25799667,
50.67854 H100 minutes; all containers stopped. The no-corrective-launch rule
stopped further execution. No performance result or serving-default change.
