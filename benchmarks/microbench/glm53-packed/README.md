# GLM-5.3-Flash Packed qualification

Author: Winston Cai.

Current optimized decode result: **48/48 cases complete, 30 paired rounds each;
Direct adds 1.60% batch-weighted overhead**. See [the decode outcome](#optimized-decode-outcome-2026-10-26),
[canonical evidence](qualification.json) and [all cases](decode.csv). The earlier
unoptimized 56/128 attention comparison below remains historical and incomplete.
Aggregate authorization is $40 ($20 historical plus the extended $20 phase).
Native remains the serving default.

Review follow-up, 2026-10-26: SGLang `9185c6d25115` passed fresh focused
H100 Direct byte/numerical/graph checks and memcheck/racecheck/synccheck. Adapter
reconstruction across the 2 GiB boundary passed numerical and all three sanitizer
tools, joined across identical source/artifact runs. The final `979839b` adds only
missing server-argument documentation. Pinned lint CI is green at final `979839b`;
`run-ci` is applied and its gate passes. Broad GPU CI has no self-hosted runners.
See [current validation](#review-gpu-validation-and-ci-cleanup-2026-10-26).
The 48-case timing result retains its historical source identity.

Question: can eleven MLA layers use 320-byte rows while retaining Native FP8
TileLang arithmetic and unchanged KDA/indexer/tail reservations on H100?

Mechanism: select 256 raw latent coordinates before Native RMSNorm, persist
256 raw E4M3 bytes plus eight MSB-first uint64 bitmap words. Adapter materializes
selected rows in chunks of at most 128 queries; Direct expands within TileLang.
Native remains the default and rollback.

Source: managed worktree `glm53-packed`, branch `codex/glm53-packed`, starting
parent 9419d0e, SGLang 7856615, FlashMLA 3eb8594. Only v0.5.22 SGLang changes.

Acceptance: exact masks/bytes/reconstruction, production pool reconciliation,
attention atol=0.02/rtol=0.02, changing-input graph replay, and unexplained-error-free
memcheck/racecheck/synccheck. Native is FP8 TileLang on H100. No full weights,
serving, model quality or serving-performance claims. Configuration metadata only.

Budget: one aggregate $15 ledger, CPU preparation <=$2, H100 worker <=$10,
$3 reserve; routine scheduling stops at conservative $12. One Function at a time,
no automatic retries, CPU4/RAM32GiB/H100, aggregate H100 <=120 minutes including
startup/cleanup (45+45+one corrective 30). Use caiw first after fresh spend, rate,
container and artifact checks. Release GPUs before diagnosis or builds.

The canonical JSON record in this directory owns source/artifact identities,
commands, coverage, costs and keep/discard/inconclusive decision. Missing gates
remain unverified. Existing `scripts/modal/kernels.py` owns remote orchestration;
GLM runtime and fixtures belong to the SGLang fork. No replacement harness.

The analytic ratio is 5995/3883 = 1.5439093484 at unchanged KDA reservation.
Physical tests must account for pages, indexer, KDA, tails, aliases and workspaces.
This ratio is not measured serving capacity.

## Optimized decode outcome (2026-10-26)

**Decision: keep the cooperative loader.** All 48 decode cases and all 30 paired
rounds passed `remnant-glm-decode-v1` with fresh Native controls. The batch-weighted
paired overhead is **1.6010%**, with **0.6829% for 16 local heads (TP4 geometry)**
and **2.5192% for eight (TP8 geometry)**. Ten cases favor Direct. Case means range
from -3.5576% to +8.6594%; none reaches 2×. These are model-free one-H100 complete
attention calls, including query conversion, partial attention and combine.
They do not establish multi-GPU serving, model quality or prefill performance.

Measured source: parent `8aea70ae58a9`, SGLang `246ee03c1ad5`, unchanged FlashMLA
`3eb85944fd7d`; starting/rollback SGLang `889edf56b332`. Native code and all sixteen
Native cubins remain byte-identical. CPU inspection covers all 32 maintained
SM90 variants: Direct has 34 POPC sites, vector payload loads/stores, no spills
or stack/local traffic, dynamic shared memory at most 34,944 bytes, and the same
two-CTA / 12.5% theoretical occupancy as Native. The arithmetic backend is unchanged.
The helper expands each latent tile once and reuses it for QK and PV.

Focused source-bound qualification passed: exact bytes including real >2 GiB
addresses; Direct partial output/LSE versus stock on independently reconstructed
Packed; H8/H16 × inner1/2; alternating banks/modes, changing-input graphs and owned
streams. Synccheck: 5 tests in 53.91 s, zero errors. Racecheck: 5 in 55.37 s,
zero hazards/warnings. Memcheck: 5 in 139.62 s, zero errors. Separate full attention
numerical/graph smoke: 5 tests in 26.13 s. Instrumentation is deliberately limited
to the byte probe and one production partial CTA per bank. Full-grid/combine and
broad sanitizer qualification remain incomplete; unchanged control operations run
uninstrumented. The timing command enforces this focused scope and all 48 cases.

The timing Function completed in **136.084 s** including its graph preflight;
controller/image resolution/cleanup totaled **246.936 s**. This measured duration
is specific to this decode matrix and prepared image. No additional timing runs
were scheduled to improve the already met soft 15% goal.

Evidence: canonical [JSON ledger](qualification.json), [every flat result](decode.csv),
and timestamped raw outputs in `data/glm53-packed/glm53-optimized-timing-20261026T092628Z/`
and `caiw/remnant-stage2a-results/microbench/glm53-optimized-timing-20261026T092628Z/timing/`.
Raw Native/Direct samples, paired ratios, fixture/source/toolchain/binary hashes,
commands and failures are preserved. Later test-only fixture changes retain this
measured source identity; they do not relabel or rerun these samples.

| Local heads | Batch | Tails | Cache | Native µs | Direct µs | Paired change |
|---:|---:|---:|:---|---:|---:|---:|
| 16 | 1 | 0 | warm | 20.701 | 21.106 | +1.96% |
| 16 | 1 | 0 | rotating | 20.163 | 20.281 | +0.58% |
| 16 | 1 | 3 | warm | 20.650 | 21.021 | +1.79% |
| 16 | 1 | 3 | rotating | 19.888 | 20.052 | +0.82% |
| 16 | 8 | 0 | warm | 23.481 | 24.432 | +4.05% |
| 16 | 8 | 0 | rotating | 23.105 | 23.987 | +3.82% |
| 16 | 8 | 3 | warm | 23.321 | 24.513 | +5.11% |
| 16 | 8 | 3 | rotating | 23.114 | 23.983 | +3.76% |
| 16 | 16 | 0 | warm | 35.136 | 36.156 | +2.90% |
| 16 | 16 | 0 | rotating | 35.992 | 36.456 | +1.29% |
| 16 | 16 | 3 | warm | 35.303 | 36.317 | +2.87% |
| 16 | 16 | 3 | rotating | 35.974 | 36.491 | +1.44% |
| 16 | 24 | 0 | warm | 50.822 | 49.994 | -1.63% |
| 16 | 24 | 0 | rotating | 49.815 | 49.218 | -1.20% |
| 16 | 24 | 3 | warm | 51.134 | 49.315 | -3.56% |
| 16 | 24 | 3 | rotating | 49.852 | 49.333 | -1.04% |
| 16 | 32 | 0 | warm | 66.175 | 65.260 | -1.38% |
| 16 | 32 | 0 | rotating | 64.300 | 63.871 | -0.67% |
| 16 | 32 | 3 | warm | 66.364 | 65.018 | -2.03% |
| 16 | 32 | 3 | rotating | 64.388 | 64.036 | -0.55% |
| 16 | 64 | 0 | warm | 123.770 | 125.643 | +1.51% |
| 16 | 64 | 0 | rotating | 122.369 | 124.410 | +1.67% |
| 16 | 64 | 3 | warm | 123.706 | 126.094 | +1.93% |
| 16 | 64 | 3 | rotating | 122.269 | 124.564 | +1.88% |
| 8 | 1 | 0 | warm | 21.152 | 22.087 | +4.42% |
| 8 | 1 | 0 | rotating | 20.622 | 20.931 | +1.50% |
| 8 | 1 | 3 | warm | 21.013 | 21.924 | +4.33% |
| 8 | 1 | 3 | rotating | 20.446 | 20.807 | +1.77% |
| 8 | 8 | 0 | warm | 21.570 | 23.438 | +8.66% |
| 8 | 8 | 0 | rotating | 22.046 | 23.171 | +5.10% |
| 8 | 8 | 3 | warm | 21.734 | 23.541 | +8.32% |
| 8 | 8 | 3 | rotating | 22.302 | 23.443 | +5.12% |
| 8 | 16 | 0 | warm | 32.413 | 34.743 | +7.19% |
| 8 | 16 | 0 | rotating | 34.131 | 35.002 | +2.55% |
| 8 | 16 | 3 | warm | 32.499 | 34.630 | +6.56% |
| 8 | 16 | 3 | rotating | 34.198 | 35.011 | +2.38% |
| 8 | 24 | 0 | warm | 45.392 | 46.238 | +1.86% |
| 8 | 24 | 0 | rotating | 46.257 | 46.891 | +1.37% |
| 8 | 24 | 3 | warm | 45.826 | 46.109 | +0.62% |
| 8 | 24 | 3 | rotating | 46.244 | 46.765 | +1.13% |
| 8 | 32 | 0 | warm | 60.196 | 59.025 | -1.95% |
| 8 | 32 | 0 | rotating | 59.141 | 59.214 | +0.12% |
| 8 | 32 | 3 | warm | 60.354 | 59.055 | -2.15% |
| 8 | 32 | 3 | rotating | 59.179 | 59.288 | +0.18% |
| 8 | 64 | 0 | warm | 108.765 | 112.569 | +3.50% |
| 8 | 64 | 0 | rotating | 107.455 | 111.462 | +3.73% |
| 8 | 64 | 3 | warm | 108.976 | 112.911 | +3.61% |
| 8 | 64 | 3 | rotating | 107.539 | 111.595 | +3.77% |

## Original runtime qualification (historical, 2026-10-25)

**Decision: retain an opt-in candidate; qualification is incomplete.** Purpose:
replace only the eleven MLA layers' persistent FP8 storage. Risk: the persistent
layout and Direct loader change; full model quality and serving remain unknown.
Native remains the default. FlashMLA and both v0.5.18 pins are unchanged.

Tested parent `5f4754e3015d2268df4b39175ce4622cd305589d`, SGLang
`9e42a0d8886dd93944dc6b47ef3bffef4e033b5a`, FlashMLA
`3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8`. Final parent edits only record evidence.
CPU production builds target SM90; Python 3.12, CUDA 13.0.3, Triton 3.8.0,
TileLang 0.1.12. Native extensions were reused only after matching source hashes.
The final image is `im-XTjWc2irNsaIzoyCmwuyVW` on caiw. Image construction uses
verified base `im-FCIjtGo9a7Z9hNW3o32leM`; changing that base can change cache identity.
Only the pinned configuration metadata was downloaded, never checkpoint shards.

Evidence:
- 24 startup contracts and 74 maintained orchestration checks passed on CPU.
- The initial H100 smoke failed (8 failed, 1 passed): precompiled Triton launches
  needed three-dimensional grids; the model-free backend fixture needed explicit
  rank context. Both were fixed and CPU-preflighted before the single correction.
- Corrective H100 run: **90 passed**, including the 80-shape attention matrix,
  exact codec parity, Native FP8 conversion, partial outputs/LSE, Adapter chunk
  boundaries, changing-input graphs, production constructors, pool lifecycle and
  allocation reconciliation. Attention tolerances remain atol/rtol 0.02.
- Synccheck: **11 passed, ERROR SUMMARY: 0 errors**, 1,252.49 seconds.
- Racecheck: **timed out** at the shared 2,400-second execution deadline, nine
  pass markers and no final summary. Memcheck **not run**. These are unverified.
- Independent fresh-context review was not performed; separate-agent permission
  was requested but not received. Manual source inspection is not that review.

At 131,072 token slots, 32 requests, TP8 rank geometry and identical KDA reservation:

| Mode | Unique backing bytes | Active CUDA block bytes | Workspace bytes |
|---|---:|---:|---:|
| Native | 3,897,341,212 | 3,898,396,672 | 0 |
| Packed Adapter | 3,758,388,508 | 3,759,443,968 | 138,412,032 |
| Packed Direct | 3,619,976,476 | 3,621,031,936 | 0 |

Direct saves **277,364,736 bytes (264.515625 MiB) per rank**; Adapter saves
138,952,704 bytes (132.515625 MiB) after its fixed workspace. Counted backing
allocations include page padding, indexer, tails, KDA, request metadata and
pointer tables once; `allocation.csv` is the flat summary. CUDA reserved bytes
are diagnostic allocator state, not owned tensor bytes. The **54.39%** ratio is
still a token-capacity projection at unchanged KDA reservation, not measured
serving capacity; fixed overhead and page rounding affect finite-budget gains.

Cost: **$3.82152258 observed compute**, before credits; **$4.70330959 conservative
accrual**, including $1.18365307 CPU preparation and $3.51965653 H100 worker wall
time. The three hardware launches total 2,882.86 conservatively counted seconds
(48.05 minutes). No accounts were switched. Billing can lag; the $3 reserve
remains unused. All containers stopped and the idle preparation deployment was
removed. No further launches were made. Sources and failures remain recoverable
in Git; raw logs/manifests are in ignored `data/glm53-packed/` and Modal volume
`caiw/remnant-stage2a-results/microbench/<run-id>/validation/`. Canonical JSON
contains file hashes, exact commands, accounts, source and compiled artifact IDs.

Reproduction uses the maintained entry point (each remote launch requires its
own applicable authorization and preflight):

```sh
export MODAL_PROFILE=caiw REMNANT_INTEGRATION=v0.5.22 REMNANT_MODEL_FAMILY=glm53
export REMNANT_COMPILED_IMAGE=im-FCIjtGo9a7Z9hNW3o32leM
modal run scripts/modal/kernels.py::prepare_kernels --model-family glm53
modal run scripts/modal/kernels.py::validate_kernels --model-family glm53 --stage numerical --run-id <unique-id> --max-seconds 2400
```

The exact corrective case selection and sanitizer command are in
`qualification.json`; do not silently relabel or overwrite completed runs.
Next gate is independent review and completion of racecheck/memcheck with a
separately approved resource/time plan. No active probe or duplicate runner was
retained; maintained production fixtures own reproduction. Roll back with
`--glm5-mla-cache-format native`, or restore SGLang starting pin `7856615f3fdf02e993f3b40ecdbbed8cb8ac9bf7`.
No full-model load, TP8 execution, latency, serving capacity, quality, multimodal,
speculation, CP, unified memory, disaggregation, HiCache or HiSparse qualification.

## Native versus Direct attention timing

Question: does Packed Direct reduce complete FP8 TileLang attention latency
against unpruned Native? Only attention is measured: query conversion, partial
outputs and combine. Producer/store, indexer, metadata, allocator and model
weights are excluded. Native is not replaced by a reconstructed Packed control.
The required protocol is `remnant-glm-attention-v1`: 128 cases, TP4 then TP8,
decode batches 1/8/16/24/32/64, prefill chunks 128/512/1536/4096/8192,
128K prefix, initial tail phases 0/3, independent/shared prefill selections,
warm/rotating policies, seed 20261024, and 30 balanced paired graph rounds.
Acceptance requires exact codec validity, independent output references,
completed outstanding sanitizer gates, and full matrix coverage. Performance is
descriptive; missing gates or resource exhaustion mean incomplete, not a smaller
successful experiment. Preserve the original $15 total, $12 routine cutoff,
$3 reserve, category caps and 120 aggregate H100 minutes. Stop on failed
preflight/validity, budget or deadline; no automatic retry or corrective launch.

The existing `benchmark/remnant/microbench.py` dispatches GLM into a dedicated
fixture/reference module under `python/sglang/benchmark/remnant/`; the DeepSeek
entry point owns incompatible compression and FlashMLA initialization. Shared
measurement primitives have both benchmarks as consumers. These maintained
modules and their contract tests remain after the experiment; transient launch
controllers are retired after their commands/source hashes enter the ledger.


### Initial timing attempt (historical)

Implementation is retained in SGLang `02cbbd92b802e0437a8597b6b14ec45c994cf98b`.
Parent implementation commits are `9672503` and `d092f85`; FlashMLA remains
`3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8`. Production kernel/runtime content is
unchanged from the original GLM qualification. Local checks passed: 81 existing
non-GPU benchmark tests, 24 startup contracts, 78 maintained launcher tests,
and, after the final fix, 10 focused GLM contracts plus five focused launcher
tests. A fresh-context reviewer inspected selection/causality, baseline fairness,
timing, rotation, graph ownership and cleanup; timeout and explicit sanitizer
coverage findings were fixed. The final accounting delta received a targeted
independent review with no concrete finding.

Two CPU preparations succeeded on `caiw`, producing images
`im-vwwDy7UMUgFqIHRwa0O0xK` and `im-LF2iKAP2WKy42lKF67nPlt`. The first new H100
allocation, `glm53-attention-racecheck-20261025T124407Z`, tested parent `d092f85`
and SGLang `f71841e`. Startup contracts passed (24 tests). Its unprofiled graph
smoke passed heads8/banks2/queries128; heads16/banks4/queries512 completed output
and changing-input replay checks but failed its final positive graph-memory
assertion. Global reserved memory fell by 48,234,496 bytes, so subtracting global
allocator totals cannot measure graph ownership. This failed fixture is not a
qualification pass. No sanitizer instrumentation or timing started.

The retained repair sums unique graph-pool snapshot segment sizes, including
inactive storage. It preserves the bank-independent allocation bound and 70%
device-memory cap. CPU regression covers pool ownership, inactive storage and
invalid snapshots. **The repaired accounting has not run on H100.** The plan's
no-automatic-corrective-allocation rule stopped further GPU scheduling; racecheck,
memcheck, new fixture sanitizer coverage and all 128 timing cases remain pending.
No timing CSV or speedup is reported; `allocation.csv` remains historical physical
storage evidence. Per-run logs, XML, manifests, controller snapshot and CPU test
logs are retained and hashed by `qualification.json`; the controller has been
retired from its temporary executable location. Historical passed/failed evidence
is preserved separately from this phase.

Final read-only checks found zero containers and all three task apps stopped.
Aggregate conservative accrual is **$5.25799667**: $1.54562464 CPU preparation and
$3.71237203 H100 worker allocation, including the original phase. Aggregate H100
allocation is **3,040.71244 seconds (50.67854 minutes)**. This phase added
$0.55468708 and 157.84852 H100 seconds. The latest lagging compute reading is
$3.80780462 before credits; the prior $3.82152258 high-water is preserved and
conservative accrual controls the budget. No account switch, weight download or
push occurred. The $3 reserve remains protected.

The single next entry point is the maintained validator below, after explicit
approval of a corrective allocation and fresh source/artifact/budget preflight:

```sh
MODAL_PROFILE=caiw REMNANT_INTEGRATION=v0.5.22 REMNANT_MODEL_FAMILY=glm53 \
  modal run scripts/modal/kernels.py::validate_kernels --model-family glm53 \
  --stage sanitizer --sanitizer-tools racecheck --run-id <unique-id> --max-seconds 1500
```

Repaired-source CPU preparation and matching artifact verification must precede
that GPU launch. Memcheck remains a separate allocation. Timing dispatch is
`kernels.py::benchmark_kernels --model-family glm53 --suite glm-attention` with
qualified run IDs and a remaining-budget deadline; it rejects missing gates.
Rollback is to omit this suite; Native serving remains the default.


### Resumed authorization (2026-10-25)

The user explicitly approved reruns until the microbenchmark/comparison works,
within the same aggregate $15. This supersedes the prior no-corrective-launch
restriction, not the aggregate ledger or qualification requirements. Production
kernels and timed attention paths remain unchanged. Setup now populates only the
sorted union of valid selected physical rows in full-size zeroed backing banks;
Native/Packed share raw latents, all selected Packed bytes are checked, and hashes
include their physical locations. This is a distinct fixture source revision.

Timing can resume a valid ordered checkpoint through maintained
`benchmark_kernels --resume-run <prior-run-id>`. Source/artifacts, seed, rounds and
all completed per-case checks must match exactly. Only an unfinished final case
is discarded; interior gaps and changed fixtures are rejected. Full acceptance
still requires all 128 cases. This avoids repeating completed work after a
bounded deadline and does not change sample counts or combine independent ratios.


Racecheck run `glm53-attention-racecheck-20261025T134651Z` completed the original
11 cases. An intentional SIGINT to its owned pytest process finalized JUnit
(11 passed, zero failures/errors/skips) and RACECHECK SUMMARY with zero hazards,
errors and warnings. The unfinished bank test is absent from completed coverage;
its unnamed JUnit interrupt placeholder is not counted. The raw overall result
remains failed/interrupted. A separate hash-bound checkpoint note records the
interruption without rewriting the raw result. The maintained gate can merge
only finalized clean completed cases; missing identities remain blocking.

Focused graph sanitizer probes explicitly use Q2/Q8, heads8/16, banks2/4 and
forced inner33/66, with JUnit geometry metadata. Their compiled partial/combine
variants match the full Q128/Q512 smoke, but their allocation/grid dimensions do
not. They preserve independent references, all banks, invalid indices, alternating
and changing-input replay. They perform one requested warmup with a full bank
cycle (actual2/4 calls per mode). Full-size ownership/numerical smoke and all128
measured cases retain their original sizes and ten-warmup policy. Original11
sanitizer cases remain unchanged. All13 identities are required per tool across
clean reports before timing; historical synccheck is supplemented by the probes.


Single-case racecheck `glm53-attention-graphs-20261025T143821Z` finalized the
8-head/Q2/two-bank/inner33 probe with one passed case and zero reported hazards.
The preceding short combined probe allocation had only a pass marker before
its hard deadline, no final report; it supplies no sanitizer coverage.

The focused 16-head representative grid is revised from Q8 to Q2, retaining
four banks and inner66. Focused independent attention oracles now execute the
same FP8-Q rounding and FP32 attention math on CPU, returning BF16 to the original
device. This avoids instrumentation of reference-only cuBLAS work without
excluding any production CUDA kernel. A CPU analytic ties/empty-selection check
and an unprofiled full-size CPU/GPU oracle bridge precede instrumented probes.
Full-size ownership smoke and all128 timing fixtures keep their original query
sizes and GPU oracle. New identities/geometry/reference-device metadata distinguish
this revision from earlier attempts. Review found no concrete math/device issue.

The user's renewed rerun authorization makes $15 the controlling aggregate cap;
necessary reruns may exceed the earlier120-minute/category estimates. The
controller conservatively rejects projected totals above$14.75, retaining$0.25
for shutdown/export/storage. Historical caps, attempts and costs remain recorded.
A small CPU preparation reallocation avoids much more expensive reference-only
GPU sanitizer work. No numerical acceptance criterion or timing count changes.

### Authorized timing priority (2026-10-26)

The user selected descriptive Native versus Direct timing with memcheck explicitly
incomplete, then approved an additional $5: the aggregate task cap is now $20.
The controller retains a $0.25 cleanup/export reserve and schedules below $19.75.
Prior costs are preserved. Original time/category estimates do not reset the ledger.
Both focused racecheck and synccheck probes passed, supplementing the original
11-case reports. Memcheck was stopped for timing priority after six pass markers;
without a final XML/summary, it remains unverified.

The maintained `benchmark_kernels` accepts explicit GLM-only
`--allow-incomplete-qualification`. It still verifies all supplied evidence,
source/toolchain identities, fixture validity and the full measurement protocol;
missing sanitizer coverage is returned as `qualification.status=incomplete` with
exact missing case names and `experiment_status=unqualified`. The default still
blocks timing on missing gates. This option enables descriptive data collection,
not acceptance or production promotion. Finish remaining memcheck only after
timing if the same aggregate budget permits.


## Current descriptive attention outcome (2026-10-26)

**Purpose:** measure complete stock Native FP8 TileLang attention against Packed
Direct on one H100. **Decision:** retain a reproducible descriptive comparison;
do not promote Direct. Production kernels, serving defaults, FlashMLA and both
v0.5.18 pins are unchanged. No kernel tuning was undertaken after observing a loss.

**Evidence:** measured parent `853e4d6bc85acfb2ae56fcba92af77e1ae29b853`,
SGLang `889edf56b33226c80b17ddf3472c1206df62c3f0`, FlashMLA
`3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8`, CPU-prepared image
`im-EomZZ0NhII6stVasjzgyo4`. SM90 runtime/compiled content is unchanged from
qualification. All 81 maintained launcher tests and seven focused protocol tests
passed. A fresh-context reviewer inspected the benchmark; targeted follow-ups
covered later fixes and the explicit exploratory option. No unresolved concrete
review finding remains in the implemented benchmark.

There are **56 complete cases, two modes, 30 balanced paired rounds per case**:
all TP4 (16 local heads) decode batches, plus prefill128/512/1536/4096, both tail
phases, both locality policies and warm/rotating buffers. They passed exact Packed
bytes/bitmap checks, full-output finiteness, independent sampled references at
atol/rtol0.02, stock reconstruction controls and graph ownership/replay checks.
The three resumed prefixes (46,51,54 cases) are preserved exactly. The final raw
record deliberately says `incomplete` after a boundary SIGINT; its 56 valid cases
are not relabeled as a complete128-case experiment.

Medians below are milliseconds per complete attention call. Ratios are medians of
paired Direct/Native round averages, not ratios of aggregate workload latencies.

| TP4 case (tail phase0) | Native ms | Direct ms | Paired Direct/Native |
|---|---:|---:|---:|
| Decode batch1, warm | 0.020587 | 0.065171 | 3.16 |
| Decode batch1, rotating | 0.019885 | 0.086035 | 4.33 |
| Decode batch64, warm | 0.123745 | 1.147201 | 9.27 |
| Prefill128, independent, warm | 0.156369 | 2.051651 | 13.12 |
| Prefill512, independent, warm | 0.592497 | 8.171516 | 13.79 |
| Prefill1536, independent, warm | 1.697530 | 24.269038 | 14.30 |
| Prefill4096, independent, warm | 5.042905 | 72.045659 | 14.29 |

Across these completed cases, per-case paired median ratios range3.1644–14.7105;
every measured case favors Native. No workload-weighted headline is reported.
The CSV contains112 mode rows, raw samples, p95 of round averages, paired ratios
and fixture metadata. The canonical qualification JSON embeds all56 case records,
source/artifact identities, commands, hashes and accrued costs. Timestamped raw
exports remain in ignored `data/glm53-packed/` and Modal volume
`caiw/remnant-stage2a-results/microbench/<run-id>/timing/`.

**Risk and limitations:** this comparison is explicitly unqualified. Reconciled
source-bound sanitizer evidence covers racecheck13/13 and synccheck13/13, both
clean; memcheck6/13 completed cleanly before the user requested timing priority.
Its intentional SIGINT produced a finalized six-case XML and zero-error summary,
bound by a separate checkpoint note. Seven memcheck cases remain unverified.
The72 missing timing cases are all eight8192-query TP4 cases and all64 TP8 cases.
No full-model loading, true TP4/TP8 execution, serving capacity, model quality,
serving latency/SLO or causal performance-profile claim is established.

**Budget:** the user increased aggregate authorization from$15 to$20 and chose
timing before unfinished memcheck. Conservative accrued cost is**$19.39082903**,
including CPU/image preparation$2.27645383 and H100-worker upper-rate wall time
$17.11437519. Observed compute high-water is$16.57580973 before credits; a later
snapshot was lower and does not reduce conservative accrual. All allocations are
released and apps stopped. Remaining routine headroom$0.35917097 cannot cover the
next8192-query case, so no further GPU work is scheduled. The requested$35 cap is
**not approved**. Earlier limits, failures and costs remain historical records;
there is no account or budget reset.

**Next entry point and exact source:** the maintained command is
`benchmark/remnant/microbench.py --model-family glm53 --suite glm-attention`.
Remote dispatch is `scripts/modal/kernels.py::benchmark_kernels` with explicit
`--qualification-runs`, `--allow-incomplete-qualification` and
`--resume-run glm53-attention-timing-20261025T173500Z`. Full qualification remains
the default without the exploratory option. Further allocation needs increased
budget and fresh preflight. Strict resume includes the complete compiled manifest,
parent853e4d6 and the original managed worktree path: preserve the latest aggregate
ledger outsideGit before restoring that measured source, and overlay the latest
ledger afterward. Never restore an earlier spent amount from Git. The ignored
`data/glm53-packed/attention-handoff-ledger.json` preserves the current ledger;
the archived `attention-resumed-controller.py` preserves execution controls.
Do not run the archived controller against an old ledger or assume its cap changed.

**Rollback and lifecycle:** omit the GLM benchmark suite; Native serving remains
the default. Maintained GLM fixtures, shared timing, tests and Modal dispatch stay
active. The task-owned temporary controller and repair helper were retired after
archiving the controller and hashes. Historical run evidence was preserved. No
push was performed.


## CPU-only cached-cubin resource inspection (2026-10-26)

Question: what are the compiled registers/shared-memory requirements and theoretical
H100 residency of the exact Native/Direct attention variants? Acceptance requires
all sixteen H8/H16, inner1/2/33/66 variants, verified source/binary identities and
launch shared-memory arguments. Existing kernels remain the baseline; stop on
artifact mismatch, unexpected GPU devices, a failed preflight or controller deadline.
The proposed cooperative loader is not compiled or qualified by this inspection.

**Complete: all sixteen variants inspected.** Two serial CPU-only metadata exports
reused immutable image `im-EomZZ0NhII6stVasjzgyo4` on `caiw`, without kernel imports,
JIT compilation, CUDA contexts or GPU allocation. Each export used one CPU, 1 GiB
RAM, no retries, one worker maximum, a two-second idle timeout and a 180-second
controller ceiling. An initial invalid idle-timeout registration allocated no
worker; its failure is preserved. Both successful apps stopped with zero containers.

Source: SGLang `889edf56b33226c80b17ddf3472c1206df62c3f0`, FlashMLA
`3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8`, tested source parent `853e4d6`.
Generated-source hashes and the installed TileLang 0.1.12 cache-key recipe joined
each variant to its exact cubin. All 421 cached-file hashes were verified.
`cuobjdump --dump-resource-usage` supplied registers/stack/local bytes; generated
host launchers supplied dynamic shared bytes. ELF inspection verified 1,024 bytes
in `.nv.shared.main_kernel` and zero in `.nv.shared.reserved.0`; H100's separate
1,024-byte driver reservation is counted once in the occupancy calculator.

| Local heads | Inner iterations | Native registers/thread | Direct registers/thread | Dynamic shared, both | Charged shared/block, both |
|---:|---|---:|---:|---:|---:|
| 8 | 1 | 176 | 175 | 25,216 B | 27,264 B |
| 8 | 2 / 33 / 66 | 232 | 236 | 26,240 B | 28,288 B |
| 16 | 1 | 175 | 175 | 25,216 B | 27,264 B |
| 16 | 2 / 33 / 66 | 240 | 240 | 26,240 B | 28,288 B |

All report zero stack/local bytes. Loop variants have two additional generated
512-byte reduction workspaces: the earlier 25,216-byte source-level estimate
missed their compiler lifetime/layout choices.

The image's exact CUDA 13.0.3 `cuda_occupancy.h` was compiled locally with ordinary
macOS `clang++`, without CUDA runtime/driver linkage. H100 SM90 inputs use 65,536
registers/SM, 2,048 threads/SM, 228 KiB shared/SM and 128 threads/block. All sixteen
variants predict **two blocks/SM, eight warps/SM, 12.5% theoretical warp occupancy,
limited only by registers**. Default and maximum shared-carveout models agree.
SM90 allocation rounds by 256 registers/warp and 128 shared-memory bytes. These
are theoretical capacities, not measured achieved occupancy or runtime carveout.

Adding 11,264 shared bytes while holding each cached register count unchanged
still predicts two blocks/SM. The hypothetical shared-only ceiling becomes six
blocks for inner1 and five for inner2/33/66; registers still bind first. This does
not establish the uncompiled cooperative candidate's resources or speed.

[resource-usage.csv](resource-usage.csv) exports one row per compiled variant.
`qualification.json` owns compact results, identities, exact commands, artifact
hashes, independent review and aggregate cost. Raw cubins, generated sources,
ELF/resource reports, the exact calculator header and offline analysis are outside
Git under `data/glm53-packed/resource-inspection-20261026T035110Z/`. The recorded local command is
`python3 /private/tmp/glm53-cubin-inspection/calculate_resources.py`; its sources
and inputs are archived together. No new maintained runner was added.

Conservative inspection accrual is **$0.01**, including CPU/egress/billing-lag
margin. Aggregate task accrual is now **$19.40082903 of $20**; GPU seconds are
unchanged. Attention remains incomplete at 56/128 cases and memcheck remains
incomplete. No runtime, kernels, fork pins or serving defaults changed. Rollback
is to revert this evidence-only parent commit; Native remains the default.


## CPU-only Native/Direct SASS comparison (2026-10-26)

Question: with identical theoretical residency, what extra compiled work does
GLM Direct issue versus Native, and which V4 loader mechanisms should replace it?
Compare unchanged hash-matched cubins, preserving Native as the baseline. Require
all sixteen H8/H16, inner1/2/33/66 variants, raw SASS, source/CFG/opcode checks and
explicit separation of static sites from executed work/cycles. Stop on artifact
mismatch, unexpected GPU devices, failed preflight or the 180-second controller
ceiling. No kernel compilation, GPU context or H100 allocation is allowed.

**Complete: Direct's extra code is concentrated in the scalar expansion path.**
One CPU-only Function reused image `im-EomZZ0NhII6stVasjzgyo4` on `caiw`: one CPU,
1 GiB RAM, no retries, one container maximum, two-second idle timeout. All 421 cache
hashes and 16 SASS hashes/code sizes passed; all 8 paired generated-CUDA math suffixes
are identical after renaming compiler-generated identifiers. Independent CFG and
PTX/SASS reviews agree. The worker is stopped, with zero active containers.

Source remains SGLang `889edf56b33226c80b17ddf3472c1206df62c3f0` and FlashMLA
`3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8`. `cuobjdump --dump-sass` inspected all 16
cached partial kernels; `nvdisasm --print-code --print-line-info` also inspected
the H16/inner2 pair. No kernels were imported, JIT-compiled or executed.

Representative H16/inner2 **static instruction sites inside the main selected-tile
loop**, including mutually exclusive predicated paths:

| Item | Native | Direct |
|---|---:|---:|
| Loop instruction sites | 1,618 | 11,567 |
| Loop entry to first HMMA | 634 | 10,586 |
| First HMMA to backedge | 984 | 981 |
| Population counts (`POPC`) | 0 | 2,304 |
| 128-bit global loads | 8 | 0 |
| 64-bit global bitmap loads | 0 | 1,152 |
| 8-bit global payload loads | 0 | 128 |
| 128-bit shared stores | 8 | 0 |
| 8-bit shared stores | 0 | 380 |
| Branches (`BRA`) | 1 | 640 |
| Reconvergence begin/end (`BSSY`/`BSYNC`) | 0/0 | 132/132 |
| Matrix multiplies (`HMMA.16816.F32`) | 64 | 64 |
| CTA barriers | 23 | 23 |

The loop backedges are Native `0x6b80 -> 0x670` and Direct `0x2dae0 -> 0x800`.
Whole-function code is 2,040/12,016 sites (32,640/192,256bytes), including padding.
All 16 Direct variants have 2,304 POPC, 1,152 bitmap-load and 128 payload-byte-load sites.
All 16 variants have 64 HMMA sites; matched CTA-barrier counts are 22 for inner1 or 23 for
loop variants. No LDL/STL sites occur. These facts do not imply executed lane counts,
DRAM transactions, cache misses or a measured stall breakdown.

The generated Native loader assigns eight threads per row and two rows per
thread, moving 16-byte vectors. Direct assigns one coordinate per thread, then
fully unrolls four separate 32-row loops. It repeats bitmap loads, preceding-word
population counts, validity/rank tests and predicated byte loads/stores before
any attention GEMM. The dependent chain is bitmap -> rank -> payload address ->
byte load -> shared store. The post-loader source suffix preserves QK, softmax,
PV, barriers and output; this is a loader/code-generation problem. Direct's byte
loads can still coalesce because compact ranks ascend; scalarization alone does
not prove uncoalesced DRAM traffic. Large code also creates possible front-end
pressure, but instruction-cache misses remain unmeasured.

There is a secondary raw-byte opportunity: H16/inner2 has 256 FP8 zero-constructor
instructions with RZ/RZ/RZ operands, versus 2 Native; the two nonzero quantization
instructions match. These zero constructions belong to dropped/invalid branches,
not retained-code conversion. H8/inner1 has one zero-constructor in both modes, so this
is shape-dependent and does not explain the universal rank/load expansion cost.

**Arithmetic clarification:** both cached PTX programs request FP8 E4M3
`mma.sync.aligned.m16n8k32...f32.e4m3.e4m3.f32`, but final SM90 SASS unpacks FP8 to
FP16 (`F2FP.F16.E4M3.UNPACK_B`) and uses `HMMA.16816.F32`. The representative pair
has 264 unpack and 64 HMMA sites each. Thus Native FP8 describes cache/input and
quantization semantics here; it does not establish native Hopper FP8/WGMMA
execution. This common backend does not explain Direct's differential slowdown.

The V4-inspired candidate should change only the Packed loader in
`python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py`:

1. Explicitly map four cooperating threads to each of 32 rows. Avoid the current
   coordinate-per-thread, 32-row unrolling.
2. Stage each aligned 256-byte compact payload once with vector loads. Cooperatively
   own the eight bitmap words and calculate per-word prefixes once per row.
3. Expand 8/16 coordinates using integer windows, funnel shifts and byte permutations.
   Produce dropped/invalid zeros through raw masks, not numeric FP8 constructors.
4. Store vector fragments into the existing Native-swizzled FP8 tiles, reuse them
   for QK and PV, and retain established completion/synchronization before reuse.
5. Preserve query conversion, arithmetic, partial outputs and combine. CPU-compile
   all 16 variants and inspect actual SASS/resources before any GPU validation; reject
   recurring scalar row-loop growth, hidden local traffic or changed Native code.

Current FlashMLA 3eb source provides cooperative payload/metadata staging and 8/16-code
expansion. A separately identified historical v0.5.18 production128 P4 binary
`2cd69b26...` provides further SASS evidence: its whole-function 5,432 sites include
38 POPC, 125 PRMT, 18 LDGSTS and 6 SHFL.UP sites. Its CPU-disassembly manifest is 1396a71d...;
full hashes are in the ledger. This differs from current 3eb and the user-supplied
V4.1 timing table and is not an instruction budget or speed prediction for GLM.
Omit V4's stored scales, RoPE, BF16 conversion and 328-byte-row misalignment paths
when loading GLM's aligned raw-code planes. Preserve the 320-byte persistent ABI.

Asynchronous next-tile staging is a later mechanism: inner1 has no next tile, while
inner2/33/66 can provide overlap. Do not transplant V4's 512-thread producer quotas
into GLM's 128-thread CTA. A WGMMA/backend rewrite is also separate from this loader
fix and would require its own controlled baseline and numerical qualification.
Static analysis establishes a concrete implementation defect but cannot predict
its cycle share or promise parity. The next GPU step remains a small paired
performance/numerical screen after CPU preparation, not another full matrix.

[sass-summary.csv](sass-summary.csv) contains per-variant static counts; the
canonical qualification JSON links raw SASS, exact commands, CFG/source/backend
reviews and source/binary hashes. Raw evidence and reproducible temporary probes
are archived outside Git under `data/glm53-packed/sass-inspection-20261026T042319Z/`. No new maintained runner was added.
Conservative CPU/egress/billing-lag accrual is $0.01; aggregate task accrual is now
**$19.41082903 of $20**, with GPU seconds unchanged. Attention remains 56/128 and
memcheck incomplete. No runtime, kernels, pins or serving defaults changed;
rollback is to revert this evidence-only parent commit. Native remains default.

## Cooperative loader and decode comparison

Question: can vector staging and register-owned bitmap prefixes remove the scalar
loader overhead while preserving Native arithmetic? The candidate changes only
Packed tile expansion, adding a device helper consumed by TileLang (source text
is embedded in the compilation key). Acceptance requires CPU codegen/resource
gates, fresh independent review, exact bytes and focused numerical/graph/three-tool
sanitizer checks before all 48 decode cases paired with fresh Native controls.
`--phase decode` uses `remnant-glm-decode-v1`; default all retains 128-case coverage.
The soft target is 15% mean paired overhead weighted by batch size, with every
case reported. New phase budget: $15 additional ($3 CPU, $9 GPU, $3 reserve),
routine cutoff $12, at most 120 new H100 minutes; preserve historical $19.41082903.
Stop on correctness, source, codegen, resource or budget failures. No prefill
timing, weights, serving, WGMMA or asynchronous staging changes. Rollback: Native
or SGLang 889edf56. Raw outputs live in timestamped data/glm53-packed directories;
qualification.json remains the aggregate ledger.

Focused decode sanitizer runs can select `--sanitizer-scope glm-decode`. The
filter `kns=main_kernel` covers the byte probe, Native/Direct partial attention
and combine entry points, including inline expansion. Full graphs still execute;
unchanged PyTorch/FlashInfer kernels are excluded from this focused instrumented
scope. The qualified 13.3.75 tool lacks newer launch dumps. A separate byte-probe
racecheck synchronization trace must prove main_kernel warp and CTA checks;
memcheck/synccheck use completed tests, verified cached entrypoints and the same
filter contract. Default instrumentation
remains all kernels, and broad qualification remains incomplete. The initial
unfiltered attempt passed all five numerical/graph cases, then was stopped after
synccheck stalled in CUDA graph replay; it provides no completed sanitizer gate.

The scoped attempt again stalled at first Native graph replay with no GPU
activity; filtering/large-buffer cleanup did not resolve it. Focused decode tests
now capture explicitly on their owned non-default stream, removing Torch's
implicit cross-stream capture handoff. The timing path retains default capture.
Full-trace/120-second Python stack diagnostics are enabled only for focused tools.
This is a harness correction; the loader, arithmetic and cached cubins are unchanged.

Full timeout stacks identified the expensive instrumented call at
`tilelang_sparse_fwd`'s unchanged combine launch. Earlier short interrupt stacks
were insufficient to attribute this to graph handoff. Packed partial and byte
probe now have distinct entrypoint names, verified on CPU; Native code remains
unchanged. Focused tools use `regex=glm_packed_(partial|bytes)`: they check the
changed loader and in-partial synchronization while all Native/combine/framework
operations still execute as uninstrumented numerical/graph controls. This is
Packed-loader qualification, not complete attention-path sanitizer qualification.

A later full stack showed time inside reference-only GPU `torch.unique`, so
combine/stream attribution alone was incomplete. Focused CPU-oracle validation
now performs metadata uniqueness, bitmap scatter, finiteness and close checks
on CPU as well as attention reference math. Production H100 calls and full
graphs remain unchanged. Default timing validation still uses GPU references;
its inputs, 30 rounds and measured calls are unchanged.

## Fast optimized-loader iteration (2026-10-26)

Question: can changed-loader qualification avoid repeatedly instrumenting the
unchanged Torch/reference/combine path? Kernel sources and the 48-case paired
benchmark remain unchanged. CPU contract checks run first. Harness-only overlays
reuse cached artifacts only after whole compiled-source, compile-owner AST,
artifact, toolchain, configuration and sanitizer identities match; changes to
kernel/probe/factory owners force CPU compilation. No compilation occurs on H100.

`validate_kernels --stage corrective --sanitizer --sanitizer-scope glm-decode`
with the existing byte/full-graph cases runs those five full numerical tests
uninstrumented once. The three tools then run the byte probe and four
`test_decode_partial_graph` cases: H8/H16 × inner1/2, one CTA of the exact CPU-built
production cubin, two alternating banks, changing-input graphs on an owned stream,
independent reconstructed Packed control, partition output and LSE checks at
atol=rtol=0.02. The generated ABI, dynamic shared bytes and cubin hashes are bound
to CPU metadata and JUnit properties. Fixture/reference/assertion work runs on CPU.
The byte probe retains real >2 GiB address and malformed/edge bitmap coverage.

This narrower sanitizer evidence excludes later partitions, actual tail columns
2048–2050, full dispatch/combine and full-grid interactions. The tiny partition
fixtures include local 0–3 tail-like rows, invalid entries and page boundaries;
actual tail columns/full graphs remain covered by the uninstrumented full test.
Broad sanitizer qualification remains incomplete. This is an iteration gate,
not a replacement claim for full integration qualification. Each sanitizer is
bounded and failures are retained; no tolerance or 48×30 timing coverage changes.

### Final iteration decision and costs

Keep the one-CTA suite qualified at SGLang `246ee03`. The final fork commit
`23e2e08` restores exactly that Git tree (`1c18c7b20208d6f9c10fdabce006dc912fc5501d`).
An optional seven-row fixture/batched-transfer trial is rejected: synccheck 57.97 s
and racecheck 60.05 s did not improve 53.91/55.37 s; memcheck remained incomplete
at its controller deadlines (latest 4 cases passed before interruption, no final
zero-error summary). Its source is preserved at `dad9c34` and its timestamped
logs remain in the ledger. The restored fixture's complete 139.62 s memcheck and
48-case timing retain their original source identities. Smaller CPU data alone
was not the remaining bottleneck; do not claim an extra speedup from that trial.

Retained fast path: focused production partials and byte reconstruction rather
than full attention under instrumentation; CPU reference checks; explicit
`--sanitizer-tools` selection; reuse completed source-bound groups; verified CPU
binary reuse for test-only edits. Full numerical graphs and the 48-case timing
matrix remain separate gates. Eleven focused launcher checks and 16 benchmark
CPU checks pass after restoration. Full-grid/combine sanitizer coverage remains
unverified, and broad qualification remains incomplete.

Additional-phase conservative spending: **$10.335584 of $15**, including all
failed attempts and rejected fixtures. Prior spending **$19.410829** is preserved;
aggregate conservative total **$29.746413 under $35**. CPU category $1.338441,
H100-worker category $8.638408, retained conservative overcount $0.358735;
conservative time accounting 117.93/120 new-phase H100 minutes. Two provider-timed
CPU/controller intervals were reclassified out of H100 time without reducing
original total dollar charges. No accounts were switched. The $3 reserve remains
protected; all workers stopped. No further scheduling is needed.

Rollback: Native selection or SGLang `889edf56b332`. Code and evidence are committed
without pushing. The maintained entry point for reproduction is
`scripts/modal/kernels.py::benchmark_kernels --model-family glm53 --suite glm-attention --phase decode`;
exact scoped qualification/run IDs, image recipe, commands and costs are in JSON.


## Review GPU validation and CI cleanup (2026-10-26)

Question: do the reviewed Adapter address fix and relocated artifact interface
execute correctly on H100 after resolving the fork's lint debt? Use the maintained
`kernels.py::prepare_kernels` and `validate_kernels` entry points. Run the existing
focused Direct byte/graph/partial gates and all three sanitizers, then the new
Adapter >2 GiB test under all three tools. This is validation, not new timing.

The user added $5 and 15 H100 minutes: this phase now has $20 total, $17 routine
cutoff, $3 CPU, $14 GPU, $3 protected reserve and 135 aggregate H100 minutes.
Previous expenditure and all failures remain counted. Full-model loading and
serving remain outside this task. Stop on failed preparation; no GPU compilation.

Lint cleanup uses the pinned isort7/Ruff0.15.1/clang-format20.1.7 tools. Immutable
`main_packed` historical fixtures and generated API files retain their bytes.
CPU test registrations have executable entry points; custom-build CUDA tests
retain explicit manual qualification ownership through the maintained runner.

The explicit CPU rebuild consumes `format-transition.json` in this task's ignored
artifact directory. It verifies the accepted base image/commits, every existing
binary, and all unchanged compiled sources. Five declared formatting inputs are
bound by exact before/after hashes, pinned C++ formatting, and Python3.12 AST
identity. Exact candidate hashes are checked after overlay; JIT owners are CPU
prepared again. Ordinary source/image reuse checks remain strict. Record all
new artifact identities and results in the existing canonical JSON.

Current outcome: **keep; focused GPU validation passed** at parent `0639722c102a`,
SGLang `9185c6d25115`, FlashMLA `3eb85944fd7d`, CPU image
`im-PO9F017Jyq5riB4Ndegpi9`. All 32 maintained TileLang variants were CPU-prepared;
retained AOT binaries and source identities were verified before allocation.

| Check | Passing executions | Test duration |
|:--|--:|--:|
| Direct complete byte/graph numerical checks | 5 | 25.713 s |
| Direct instrumentation scope audit | 1 | 22.667 s |
| Direct synccheck | 5 | 40.098 s |
| Direct racecheck | 5 | 40.588 s |
| Direct memcheck | 5 | 131.044 s |
| Adapter >2 GiB numerical check | 1 | 18.445 s |
| Adapter synccheck | 1 | 25.978 s |
| Adapter racecheck | 1 | 23.921 s |
| Adapter memcheck | 1 | 34.490 s |

The Direct sanitizer scope is the byte probe and one production partial CTA per
specialization. Complete attention graphs are checked numerically outside that
instrumentation filter. Adapter tools run without that filter. Broad/full-grid
and combine sanitizer qualification, serving, quality and H20 qualification
remain incomplete. Native remains the rollback; no new timing was collected.

Two interrupted Adapter attempts are retained. Modal startup delays consumed
client time before any GPU container existed; authoritative task queue/finish
records plus ten-second shutdown margins bound those allocations. Dollar charges
still include the entire client wall time. No failed attempt was refunded or
relabeled as passed. Each tool's passing result is bound to identical runtime
hashes and compiled artifacts in the canonical ledger.

This follow-up adds **$1.694222** conservatively, bringing the phase to
**$12.029805 / $20**, and all experiments to **$31.440634 / $40**.
Conservatively accounted phase H100 time is **133.171 / 135 minutes**. All workers
are stopped. The $3 reserve remains protected; no more routine runs are needed.

Local checks: 176 SGLang CPU checks and ten subtests pass (104 explicit GPU skips),
110 parent CPU checks pass under each integration profile, and six focused rebuild
regressions pass. Repository-wide pinned lint succeeds at final `979839b` in GitHub run 40352029193.
The newly enabled base CPU suite found seven undocumented fork flags; documentation
commit `979839b` adds the rows without changing runtime code. The `run-ci` label
was created/applied and the gate rerun passed; broader GPU jobs remain queued with
zero registered self-hosted runners. This is not a full-CI success claim.

Reproduce through `scripts/modal/kernels.py::prepare_kernels` and
`validate_kernels`, using the exact environment/commands recorded per run in
`qualification.json`. The latest Adapter retry selects only `racecheck,memcheck`;
its earlier matching-source synccheck evidence is retained separately.
