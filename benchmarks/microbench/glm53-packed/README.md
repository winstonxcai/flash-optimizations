# GLM-5.3-Flash Packed qualification

Author: Winston Cai.

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

## Outcome (2026-10-25)

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


### Execution outcome: incomplete

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
