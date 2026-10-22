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

### 2026-10-22 — Bounded DSV4 chat preparation candidate

Parent follows `b5d340d`; SGLang `63291eba61`; FlashMLA `97f3faa58450`.

- **Added:** default-off DSV4 text-chat conversion worker, bounded admission with
  cancellation ownership, unchanged receive timestamp and explicit shutdown drain.
  The local launcher preflights actual mounted Python source before server stop.
- **Validation:** 20 source-isolated CPU contracts and four local-launcher tests
  pass, including converter parity, native/AnyIO cancellation, errors, queue time
  and cleanup. Fresh-context runtime and launcher reviews found no scoped blocker.
  Actual-image CPU and instrumented full-model validation are pending; this entry
  is not promotion or a throughput claim. Native extensions are unchanged.
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
