<!-- Author: Winston Cai. -->
# V4.1 Packed FP4 on H100

Question: can SM90 preserve Native V4.1 E2M1 codes/scales while retaining
256 coordinates, and consume the 224-byte Packed representation directly?
Baseline: parent 9419d0e18e1e7677c0e48b8e40ad54bbc30010ca;
SGLang 7856615f3fdf02e993f3b40ecdbbed8cb8ac9bf7;
FlashMLA 3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8.
Worktree: codex/v41-packed-fp4, selected integration v0.5.22.

## Contract and acceptance

Quantize after upstream normalization/rounding/RoPE, then retain exactly 256
largest absolute dequantized values, lower coordinate first on ties. Preserve
codes and all 32 E4M3 scales; no renormalization/requantization. Eight MSB-first
uint64 bitmap words, ascending survivors, even rank in low nibble. Native is
288 logical bytes; Packed is 224. Global values already have RoPE applied.
Keep the existing SM90 SWA layout and unpruned indexer input.

Extend the existing fork tests and scripts/modal/kernels.py: independent exact
packing/reference checks, Native/Packed Direct output/LSE, C1/C2 ownership,
prefill reconstruction, changing-input graphs, streams, physical allocation,
focused V4 regressions and strict memcheck/racecheck/synccheck. Acceptance
requires all gates and fresh-context review. Quality and serving are deferred.
No model downloads, weights, full-model loads, serving, GPUQ or SM100 work.

## Execution and stopping rule

Modal only. Total task budget $15 across all accounts, with $2 reserved for
billing lag/cleanup and at most $13 projected build/test work. CPU extension
builds and preflight precede one H100 at a time. Check live profiles/spend/apps
and verified caches before launch. Bounded jobs; no automatic retries or GPU
native builds. Stop on preflight failure or budget exhaustion and report
missing coverage as unqualified. No profiling or performance benchmarks.

Canonical entry point: REMNANT_INTEGRATION=v0.5.22 modal run
scripts/modal/kernels.py::validate_kernels (the scoped FP4 suite is added here).
Use timestamped run IDs, source/binary hashes and JSON evidence. Keep output
artifacts in the runner's existing results volume; this README indexes outcomes.

## Status

Implementation in progress; not qualified. Attempt001: profile
poohthewinniechurchill rejected the launch at its workspace limit, before build
or GPU allocation. Attempt002: profile wpooh1579, app
ap-HbGInp86LfsFYh4OlB3qy9, stopped during base-image initialization following a
source audit finding (empty C4/C128 placeholders). No GPU allocation or tests.
Baseline on wpooh1579: compute $18.51013772; metered total $31.50823044,
including historical storage. Track incremental task charges, not credits.
CPU rate $0.0473/core-hour, memory $0.008/GiB-hour, H100 $3.95/hour.
The default GPU worker (8 cores, 64 GiB) costs up to $4.8404/hour.

Runtime accounting now deducts the physical reserved SWA/main/indexer pages for
both explicit FP4 modes. The allocation fixture checks actual pool storage plus
a fixed BF16 prefill workspace and the measured complete Direct-call temporary/output
peak against a 1-GiB budget; this is not serving/SLO
capacity. Its metadata is synthetic and weights are never downloaded.
Rollback: select auto or restore baseline fork pins. The new modes remain opt-in.


Attempt003: app `ap-cFImi5E8eq4FX9EP0f2KIz`, parent `f4f21a8`, SGLang
`f2ba3a07`, FlashMLA `4ff78bc`. CPU C++/CUDA compilation passed for Native/Packed
H64/H128 and existing kernels, producing base image `im-XUR1C0KC5J1HIUVyDM3i34`.
Initialization then failed because the local runner changed during assembly;
Modal rejected the changed file before CPU preflight/H100 allocation. Preserve
that failed attempt; freeze submitted files for subsequent builds.

Independent fresh-context source review found no confirmed runtime defect.
Its fixture findings are addressed: the reference oracle now constructs bitmap
bytes independently in a vectorized CPU path; allocation checks reconcile the
whole pool with disjoint storage ownership; real C2 split/in-batch state and
Native/Packed consumer dispatch are covered. The empty C2 score sentinel is
`-inf`, not zero. See `qualification.json` for canonical attempt receipts.

Run the frozen snapshot from this worktree:

```bash
MODAL_PROFILE=wpooh1579 REMNANT_INTEGRATION=v0.5.22 \
REMNANT_VALIDATION_SUITE=v41-fp4 REMNANT_VALIDATION_TIMEOUT_S=1800 \
modal run scripts/modal/kernels.py::validate_kernels \
  --run-id v41-fp4-20261025-attempt009 --sanitizer --max-seconds 1600
```

All executable validation is on Modal. Local work is source preparation and
formatting only. No changes to submitted files while a build/run depends on them.

Attempt004: parent `880bf65`, app `ap-tfsbiBLd5DCI5aLYMnkbwy`. Native extensions,
SM90 binary checks and all FP4 CPU precompilation passed. CPU startup checks
stopped qualification before GPU allocation: 22 passed, 3 failed, 13 passing
subtests. Source audit fixes the default `auto` attention backend and the new
flag fallback for legacy startup fixtures; the next attempt verifies these.
Failure logs now suppress warning summaries and retain short tracebacks.
The strengthened attention fixture balances SWA/global magnitudes and requires
zero-global and zero-global-tail negative controls to fail the original tolerance.
Reported cumulative compute is $0.23699321 before billing lag. Attempt005 has the
same bounded resource projection (CPU preflight <=$1.2688, H100 worker <=$2.4202,
plus image builds), comfortably below the $13 projected-work ceiling.

Attempt005: app `ap-i9KtfByyB4cKGE78RhDJcK`, parent `6e02fae`. CPU build,
SM90 precompilation and legacy startup regressions passed; FP4 startup failed
because the upstream layout variable defaults to `v4` even when unset. Corrected
in SGLang `d3e86c63`: reject explicitly set conflicting layouts, accept the
implicit default and explicit `auto`, with focused regression assertions.
22 CPU tests and 15 subtests passed; one failed. No H100 allocation.
Reported cumulative compute $0.46312721 before billing lag. Attempt006 retains
the same bounded projection and $2 reserve; no automatic retries.

Attempt006: app `ap-misaGh3NhWdoxQaOeiFIhD`, parent `c468f81`. Complete CPU
preflight passed (23 tests, 15 subtests), including SM90 native binaries and FP4
precompilation, image `im-7H2YIAf2ocroGo0CEJN7A8`. GPU container startup failed
before tests: Modal cannot mount the persistent Triton volume over the CPU-built
nonempty cache directory. Scoped FP4 qualification now retains that embedded
cache and verifies every precompiled file against CPU-recorded SHA256; legacy V4
keeps its persistent volume. No numerical or sanitizer GPU result was produced.
Reported compute $0.62971763 before billing lag; the same bounded projection and
$2 reserve apply to attempt007. Native sources are unchanged and reusable.

Prelaunch review also caught the matching finalization dependency: FP4 must skip
committing the unmounted persistent Triton volume. Its result volume still commits
normally, preserving success and failure evidence.

Attempt007: app `ap-QlS7WmsuD00uhZRYW63mex`, parent `f39b25f`. CPU preflight
and H100 startup passed, including embedded-cache SHA256 checks and sanitizer
controls. GPU validation: 24 passed, 13 fixture-import failures in 34.98 seconds;
whole GPU function 56.693831632 seconds. Exact Native/Packed packing/reconstruction
(16 cases), C1/C2 producer boundary (4), startup/dispatch (2) and V4 regressions
(2) passed. Attention, graphs/streams and allocation did not execute because
sibling fixtures used bare imports. SGLang `0e1a6b08` makes these package-relative
imports at module scope, so CPU collection now checks them. Sanitizers not run.
Raw evidence: `remnant-stage2a-results/microbench/v41-fp4-20261025-attempt007/validation/`.
Reported compute $0.84529972 before billing lag. Attempt008 uses the same bounded
projection and reserve, with no automatic retries.

Attempt008: app `ap-kjGoAsXjXCwc16fn7ZnBzX`, parent `1f0630c`. CPU passed.
H100: 29 passed, eight Adapter-fixture alignment failures in 49.81 seconds.
All Direct/reference and negative-control assertions passed before the Adapter
call; all graph/stream and allocator cases passed. Stock prefill requires two
64-entry tiles per iteration (`phase1.cuh`, `2*B_TOPK`); SGLang `0872a474`
pads the 576-entry test selection to 640 with invalid -1 entries. Kernel and
reference tolerances are unchanged. Sanitizers did not run after the failed test.
Measured allocator: Native 656384 tokens, Packed 800512, +21.9578783%, within
1 GiB including identical 467251200-byte SWA reservations, 270336-byte compressor
state, 1048576-byte BF16 workspace and 20492288-byte Direct workspace peak.
Indexer storage scales with token capacity; page reserves/padding are included.
This is not serving/SLO capacity. Canonical summary is in qualification.json;
raw allocation.json and XML remain in the existing results volume.
Reported compute $1.13832086 before billing lag. Attempt009 retains the same
bounded projection and $2 reserve, no automatic retries.

## FP4-only prefill synchronization refinement

Question: can the FP4 prefill/Adapter path satisfy strict sanitizer checks without
changing default stock prefill behavior? Attempt009 parent `8d332cc` passed all
37 ordinary tests, but synccheck found 12800 divergent-barrier errors in stock
BF16 prefill. Exact binary SHA256 `06c8dffa8155b3a7e1d2cc5085dd9fd38471838e887c84ea4c9e7a7c5192d606`
has two `BAR.SYNC.DEFER_BLOCKING 0xc,0x100` PCs, `0x5b90` and `0xd020`,
from separate warpgroup calls to reduce_L. CUDA719/Xid43 followed. The allocation
was stopped; no suppression or sanitizer pass is inferred. CPU-only disassembly
apps: `ap-sElVVs92ttleWoKaZcllsN`, `ap-K6z2M5uoTJNFlIfb2pDupI`.

Changed mechanism: separate SM90/D512 FP4 prefill specialization/entrypoint using
CUTLASS unaligned named-barrier APIs for role-divergent warpgroup synchronization.
The default template policy and existing stock dispatch remain unchanged. Two
new instantiation files own the length/no-length variants; existing CMake/setup
build them. No copied numerical harness or replacement reference is introduced.
Route explicit FP4 sparse prefill (including SWA-only layers) and Adapter through
this same path. Keep all numerical tolerances and the strict three-tool matrix;
cover both length variants and prefill graph replay. Accept only complete passes
and fresh-context review. Stop on further preflight failure, sanitizer error or
budget limit. Before the next launch, record pins, exact commands and projected
cost; retain the failed attempt as evidence.


Attempt009 completed CPU preflight (23 tests +15 subtests) and all37 ordinary
GPU tests (46.90s). Strict synccheck then reported12800 barrier errors in the
inherited stock BF16 prefill Adapter kernel (16 cases passed,16 failed,
175.89s), followed by CUDA719/Xid43. App `ap-zYhVLbCTc5uLmFPPVuUAz8` was
stopped and all allocations released; subsequent tools are not qualified.
CPU-only disassembly apps `ap-sElVVs92ttleWoKaZcllsN` and
`ap-K6z2M5uoTJNFlIfb2pDupI` confirmed the exact binary/PCs above.

The reviewed correction adds a separate FP4 BF16 prefill entry, leaving auto
and stock kernels on the original barrier policy. Adapter exercises both
selection-length variants; graph/stream coverage now includes reconstruction
and FP4 prefill. Source review found no blocking issue. Attempt010 will rerun
CPU preflight and all numerical/sanitizer gates without skips or suppression.
Current reported cumulative task compute is $2.00366166 before billing lag.
With no competing allocations, the same bounded CPU build/preflight plus
1800s H100 worker projects less than $4 additional work, below the $13 work cap.


## Independent oracle placement refinement

Attempt010 (`08c75c0`, SGLang `55660d7f`, FlashMLA `91463e0`) passed native
build, CPU preflight and37 ordinary GPU tests in39.40s. Strict synccheck reached
26 completed cases with no reported errors, but CPU-heavy instrumentation also
covered repeated independent GPU FP32 torch/cuBLAS oracle operations. The app
`ap-0KsEd3K9DvX4uXW4fmtzXJ` was stopped at2026-10-24T18:54:27Z and verified
released. This incomplete sanitizer run is not a pass. Native image
`im-O64c12Igt3h0QxPZunBTGU`, CPU image `im-qe56FQ27sUhO3vKpL7ct78`.

Changed mechanism: run only the independent attention oracle on CPU in FP32,
cast expected output toBF16, compare actual CPU copies with unchanged tolerances,
and reuse one expected tuple per replay. Native/Packed Direct, both Adapter
length variants, reconstruction, graph/stream and all32 sanitizer cases remain
on GPU. No sanitizer coverage is removed. Fresh-context source review found no
blocker; CPU/GPU reduction order may differ, so ordinary parity must pass again
without loosening tolerances. The same full matrix and stopping rule apply.

Attempt011 uses the same canonical command with run ID
`v41-fp4-20261025-attempt011`, `REMNANT_VALIDATION_TIMEOUT_S=2700` and
`--max-seconds 2640`.
At reported cumulative compute $4.04479708 before lag, CPU preflight <=$1.2688,
one45-minute H100 worker <=$3.6303 and CPU image build allowance <$0.10 project
under$9.05 cumulative work. This preserves the$13 work cap and$2 reserve.
