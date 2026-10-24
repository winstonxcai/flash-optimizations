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

No pending integration milestones recorded.

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
