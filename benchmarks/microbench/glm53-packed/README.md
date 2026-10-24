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
