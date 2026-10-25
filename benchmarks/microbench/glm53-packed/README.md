# GLM-5.3-Flash Packed qualification

Author: Winston Cai.

Current attention result: **56/128 cases measured; descriptive comparison is
incomplete and unqualified**. Direct is slower in every measured case. See
[the current outcome](#current-descriptive-attention-outcome-2026-10-26),
[all per-case samples and provenance](qualification.json), and [the flat CSV](attention.csv).
The current aggregate authorization is $20; earlier budget/outcome sections below
are historical. Native remains the default.

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
serving, model quality or performance claims. Configuration metadata only.

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
