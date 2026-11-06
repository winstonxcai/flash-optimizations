<!-- Author: Winston Cai. -->
# V4.1 Packed FP4 on H100

Published branch: **`ds41-flash-packed`** in the parent, SGLang v0.5.22 and
FlashMLA v0.5.22 repositories. It continues the completed local
`codex/v41-packed-fp4` work; the original branches remain available. Exact
validated source commits and historical measurements below remain authoritative.
The v0.5.22 submodule branch hints follow the new branch; v0.5.18 stays unchanged.

## Review-fix milestone — completed

Question: can the opt-in implementation expose one readable layout/prefill policy
and fail closed at startup, Direct and qualification boundaries? The candidate
now rejects effective Q8KV8 overrides and misaligned cache bases/page strides,
retains one startup prefill policy, names FP4 layout contracts, and rejects
geometry flags incompatible with the production matrix. The pure
`scripts/modal/qualification.py` owns shared receipt checks: failed sanitizer
states block qualification and recorded runtime/toolchain identities must match.
CPU qualification precedes H100 timing, followed by an allocation-time recheck.
Existing fixtures and the maintained runner own numerical checks.

**Outcome:** all scoped model-free gates passed. Keep the candidate experimental
and opt-in. Packed has confirmed storage/allocator benefits, but no uniform
latency benefit. Full-model quality and serving capacity remain unverified.

Frozen sources: parent runner **`2707db9afb073e92046dbda60b18e36f98d11acb`**,
SGLang **`c4708870008528ea68ed783c68add708687d71fd`**, FlashMLA
**`4e72722671d8860e9939df8b85f22dbea600d159`**. The final parent report changes
only evidence/documentation and a comment; it does not alter the validated
runtime, kernels or measured paths. Manifest SHA256:
`6766ad1e588d94970814d566cbdd8cd7ff7701ae0151df7bcf313db50618ebb5`.
The canonical FlashMLA source override records the preserved, untracked CUTLASS
`.DS_Store`; no unrelated dirty work was removed or committed.

| Gate | Evidence |
|---|---|
| CPU build/precompile | Native extensions built on CPU; binary hashes/SM90 targets and core FP4 precompilation verified before H100 |
| CPU contracts | 24 tests + 15 subtests; six new receipt/dispatch tests |
| Shared V4 CPU regression | 103 tests + 45 subtests across startup, pack, backend, compressor and pool contracts |
| Scoped style/registry/launcher | Ruff/isort and six CUDA instantiation format checks passed; Modal-only CI marker verified; 17 launcher tests passed |
| Ordinary H100 | 39 cases passed, including focused V4 regressions |
| Strict sanitizers | 34/34 each: memcheck 1994.71 s, racecheck 886.70 s, synccheck 1083.71 s; zero errors/hazards, no skips |
| Review | Independent source and artifact review found no unresolved finding |
| Final timing | Ten v3 cases, 20 paths, 600 samples; CPU receipt gate 16.21 s, GPU function 64.09 s |

Runs e/f were deliberately stopped after the required complete tool receipts;
their overall cancellation statuses and unused partial tools remain recorded.
The CPU preflight accepted only the complete matching clean tools. Its proof
matched the GPU recheck, identity SHA256
`04f867999d8b3b307e864c0526272890e358591d835091f0cceb9fb550f944de`.
The earlier CPU import failures, stopped preparation and disconnected H100 run
are retained under `review_fixes.attempts` in the canonical JSON. The failed
subprocess approach was replaced by six pure/mock tests in the existing CPU
builder process. No local executable tests, GPUQ, weights, model volumes,
full-model loads, serving or profiling were used.

### Final production-geometry timing

Native is dense, unpruned FP4. Packed uses the independently pruned reference;
outputs are checked separately. Both paths include their Direct call and combine
work. Cases share the producer, query, SWA, selections and sink, with randomized
paired timing. H64, Q1/K512, SWA window128/page256; batches 8/16/24/28/32 and
C1/page256 plus C2/page128. Positive Δ means Packed is slower.

| Global page | Batch | Native mean µs | Packed mean µs | Δ latency | Native graph p95 µs | Packed graph p95 µs |
|---|---:|---:|---:|---:|---:|---:|
| C1 / 256 | 8 | 71.83 | 71.29 | -0.76% | 72.52 | 71.91 |
| C1 / 256 | 16 | 100.21 | 102.52 | +2.31% | 101.51 | 103.81 |
| C1 / 256 | 24 | 129.88 | 132.18 | +1.77% | 131.46 | 133.53 |
| C1 / 256 | 28 | 157.99 | 162.21 | +2.67% | 159.88 | 164.06 |
| C1 / 256 | 32 | 158.95 | 163.17 | +2.66% | 161.77 | 165.59 |
| C2 / 128 | 8 | 71.77 | 71.35 | -0.58% | 72.46 | 72.00 |
| C2 / 128 | 16 | 100.12 | 101.53 | +1.41% | 101.32 | 102.68 |
| C2 / 128 | 24 | 129.89 | 132.33 | +1.88% | 132.12 | 134.42 |
| C2 / 128 | 28 | 157.85 | 162.13 | +2.71% | 159.84 | 164.00 |
| C2 / 128 | 32 | 167.13 | 165.42 | -1.02% | 168.15 | 166.44 |

Geometric mean of per-case mean latency ratios: **+1.30% overall**, **+1.72%
C1/page256**, **+0.87% C2/page128**. Range: **−1.02% to +2.71%**. These are
warm synthetic CUDA graph results; p95 is across 30 timing samples, each averaging
100 replays. They do not establish a repeatable speedup or serving TPOT/TTFT.
[Canonical raw JSON](runs/v41-reviewfix-20261026-timing/timing.json) retains all
samples and source metadata; [compact CSV](runs/v41-reviewfix-20261026-timing/summary.csv)
is the flat summary. The runner's larger raw CSV remains in the results volume.
Historical v1 measurements below are unchanged and are not relabeled as v3.

The allocator fixture again measured **656,384 Native / 800,512 Packed tokens
(+21.96%)** under the same 1 GiB budget, request cap and SWA policy. Native/Packed
global records are **288/224 bytes (22.22% smaller)**. The fixture reserves 1 MiB
prefill scratch and 20,492,288 bytes of measured Direct workspace; it does not
establish large-context serving memory safety. Inherited BF16 prefill scratch
scales with full compressed history: C1 at 32 × 128K is about **4 GiB/GPU** before
SWA/padding, and the high-watermark allocation is retained.

### Reproduction, budget and handoff

Use the frozen parent commit above and its exact pins. Set
`REMNANT_INTEGRATION=v0.5.22`, `REMNANT_VALIDATION_SUITE=v41-fp4`,
`REMNANT_VALIDATION_TIMEOUT_S=5400`, `MODAL_PROFILE=peroace360`, and the explicit
canonical `REMNANT_FLASHMLA_SOURCE_OVERRIDE` recorded in the JSON. Actual commands:

```bash
modal run --detach scripts/modal/kernels.py::validate_kernels --run-id v41-reviewfix-20261026-e --sanitizer --sanitizer-order memcheck,racecheck,synccheck --max-seconds 5340
modal run --detach scripts/modal/kernels.py::validate_kernels --run-id v41-reviewfix-20261026-f --sanitizer --sanitizer-order synccheck,racecheck,memcheck --max-seconds 2640
modal run --detach scripts/modal/kernels.py::benchmark_kernels --run-id v41-reviewfix-20261026-timing --suite v41-fp4-decode --qualification-runs v41-reviewfix-20261026-e,v41-reviewfix-20261026-f
```

Historical IDs cannot be reused. A new run needs new IDs, its own authorization,
and complete matching receipts. The JSON records the stop commands/app IDs,
source/binary hashes, all failures and the temporary CPU invocation source.
The maintained `kernels.py::benchmark_kernels` entry is the single timing path;
it fails on CPU before GPU allocation when qualification is missing or invalid.

The follow-up ceiling remains **$20 cumulative across accounts**, with **$2
reserved** and at most $18 projected work. Only peroace360 incurred this phase's
compute: baseline $13.73409128, conservative peak observed $28.60140100, giving
**$14.87 cumulative follow-up compute** before billing settlement. The latest
snapshot was $28.38463023; estimates can revise. CPU builds, failures, retries,
the disconnected H100 and final timing are included. **All task apps are stopped
and no containers remain.** No performance profiling is deferred into a live job.

GitHub accelerator jobs remain skipped without `run-ci`; broader repository lint
still reports existing whole-file formatting issues (including unchanged base
blobs in calibration, serving and request-time tests). Focused Modal checks above
passed. No non-Modal accelerator validation was launched. Do not interpret the
scoped qualification as an all-repository CI pass.

Active code remains in the two selected forks and the maintained runner. The
one-off CPU invocation and task-local formatter/LFS/publication tools are retired
after recording their evidence; no additional maintained harness is introduced.
Rollback selects `--dsv41-main-kv-cache-format auto` or the recorded baseline pins.

## Production-geometry matrix and fixture-correction history

The active `v41-fp4-decode` suite now uses **H64, SWA128/page256, Q1/K512**,
batches 8/16/24/28/32, and compressed global pages **C1/256 and C2/128**
(ten cases). The V4.1 config has 64 query heads, one KV head and a separate
32-head indexer. The selected H100 runtime pads smaller attention-TP shards to
64 kernel heads. SWA64 and H128 were supplementary timing coverage and are
removed from the active timing matrix; their correctness tests remain.

The window and allocation page are distinct: `kv_cache_configurator.py:1769`
requires 256-token SWA pages in this paged H100 path. The prior v2 fixture used
page128 by mistake and is superseded; it never had a GPU timing run. The actual
allocator measurement and the 128K capacity prediction already used page256.
This correction changes only the benchmark's page geometry and protocol.
Acceptance adds an explicit window-as-page negative check. Reuse the same
Modal CPU tests/image with a 120-second cap and at most $0.05 projected work;
no native rebuild or GPU run is required. Stop on a contract failure.

Question: does the maintained timing matrix match the supported model's H100
attention geometry? Only case selection and its protocol/judge change. Protocol
`remnant-v41-fp4-decode-v3` prevents historical receipts from qualifying this
matrix. Batches remain a synthetic sweep; this does not model serving traffic,
TP padding values or the indexer's execution. Source anchors are
`deepseek_v4_backend.py::SWA_WINDOW`, `MqaAttentionBase::_kernel_num_heads`,
and `deepseek_v4_memory_pool.py::_init_compressed_pools` in the selected SGLang
fork. [Official model config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)
records `num_attention_heads=64`, `num_key_value_heads=1`,
`index_n_heads=32`, `sliding_window=128` and `index_topk=512`.

Acceptance for this fixture correction: the existing matrix/timing contract
and V4 matrix regression pass on Modal CPU; verify rejection of SWA64, H128,
old protocol, missing cases/modes and bad samples/layouts. No native rebuild or
GPU timing is necessary to verify selection. Stop on a failed contract check;
allow at most $0.05 projected CPU work from the existing $20 budget. Existing
GPU/kernel qualification and raw measurements below retain their original
source identities. At that CPU-only correction stage, v3 had not been GPU-timed;
the fresh v3 results are now recorded in the completed review milestone above.
Rollback for that historical fixture stage restores the prior
SGLang pin `98991169803f4ec0b1b2c3f369695dc62d0bf3ec`.

Historical v2 CPU validation: **2 passed in 0.08 s** on Modal CPU (`ap-vTOOo7wo8VWRzuxlSPA0kn`),
using the two existing test functions above, pytest 8.4.2, one CPU/1 GiB,
a 120-second worker timeout and no GPU, network or volumes on the test worker.
The first CLI shell stopped without a usable test report and is not qualified;
the explicit-argv Sandbox retry passed. Both apps stopped; zero containers remain.
The account compute snapshot changed from $21.42800725 to $21.43512718
(about $0.00712 reported delta, subject to settlement). Exact arguments, source
hashes and retry evidence are recorded in `qualification.json` under
`production_geometry_correction`. SGLang commit: `a0f7ff18314c`; no kernel changes.

Fixture-correction CPU evidence: **2 passed in 0.14 s** on Modal CPU, app
`ap-c1tXzPaAYr6DRwi2B458Vm`, cached image `im-VqSQJWWMrtpbdbzBgw2Aum`,
SGLang `f5afebd90750`. One CPU/1 GiB, 120-second cap, no GPU/volumes;
the matrix test explicitly rejects confusing the 128-token window with a
128-token allocation page. Both tests passed; the app stopped and zero
containers remain. Billing snapshot is $21.43365249, unchanged immediately
after that CPU-only run; settlement was pending. Its evidence remains separate
from the fresh v3 GPU timing above.

The paged SWA pool is allocated once at startup from the configured request
cap, not resized when the active batch changes. With no speculation, attention
DP=1, a 128-token window, 256-token pages, eviction interval 128, overlapping
prefill chunk C, and default radix-prefix headroom 4R:

`SWA slots = align256[R*(128+128+256+256) + 4R*(128+256) + 2C + 256]`

For integer R this is `2304R + 2C + 256`; a further dummy page is included in
physical storage. At 40 layers and 585 padded bytes per SWA row per layer, the
request-dependent slope is 51.416 MiB/GPU per additional allowed request.
Only 17.139 MiB is the active-request/margin allowance; 34.277 MiB is default
cached-prefix headroom. With radix disabled or `--swa-prefix-tails 0`, that
headroom is zero. This is an allocator policy, not a requirement that SWA retain
a full context. The capacity prediction already used this 256-page accounting.

## Historical v1 qualification and timing

**Outcome: model-free correctness and final paired timing passed; keep experimental and opt-in.**
All 34 cases passed each strict sanitizer. The final 20-case comparison puts
Packed Direct **2.69% slower than Native for SWA64** and **2.58% slower for
SWA128**, using geometric means of per-case mean-latency ratios. The worst
case is 5.04% slower. Record savings remain 22.22%; no speedup is claimed.
Final fresh-context source/artifact review passed with no blocking findings. All allocations are stopped.

Question: can V4-style staged bitmap/fragment reconstruction replace the scalar
Packed FP4 loader without changing persistent values, and how does the complete
Direct path compare with fresh dense Native FP4 on the V4 decode matrix?
Acceptance requires numerical, graph/stream, allocation, all three strict
sanitizers, independent review, then full paired timing coverage. No minimum
speedup threshold was approved. The implementation changes one mechanism:
Packed reconstruction; producer scheduling and barrier topology stay fixed.

## Source identity and ownership

Managed worktree: `codex/v41-packed-fp4` at
`/Users/poohthewinnie/.codex/worktrees/v41-packed-fp4/flash-optimizations`.
Original dirty checkout, other worktrees and v0.5.18 integration are preserved.
Nothing is pushed.

| Source | Validated identity |
|---|---|
| Parent runner | `8e9e3e93c0fc3dbea86b6e0ca145d34fecf80163` |
| SGLang v0.5.22 | `98991169803f4ec0b1b2c3f369695dc62d0bf3ec` |
| FlashMLA v0.5.22 | `e274b715f49ab6c300f62e0d42c1cd627b8d1bb5` |
| CUTLASS | `147f5673d0c1c3dcf66f78d677fd647e4a020219` |

The follow-up baseline is parent `cc0f82f`, SGLang `0a36a662`, FlashMLA
`91463e00`; the worktree originated at requested parent `9419d0e`.
SGLang owns producer/pools/accounting/dispatch, fixtures and the maintained
microbenchmark. FlashMLA owns Direct reconstruction and the FP4 prefill policy.
The [fork guide](../../../third_party/README.md) maps ownership.

## Numerical contract and reconstruction

`--dsv41-main-kv-cache-format auto|fp4|remnant`: auto preserves stock behavior,
fp4 selects dense Native FP4, remnant selects Packed FP4. Explicit modes retain
the existing guards against HiCache, distributed cache transfer, CP/bounded-SWA
replay, speculation/DSpark, offload and incompatible legacy/prefill choices.

| Record | Value codes | Bitmap | E4M3 scales | Logical total |
|---|---:|---:|---:|---:|
| Native |256 B|0|32 B|288 B|
| Packed |128 B|64 B|32 B|224 B|

Preserve upstream FP32 normalization, BF16 rounding, RoPE and quantization.
Then retain the 256 largest absolute dequantized FP4 coordinates, ties by lower
original coordinate. Keep their codes and all 32 scales unchanged; no second
RoPE, renormalization or requantization. Indexer input is original/unpruned.
Survivors follow coordinate order, even ranks use low nibbles, and eight
uint64 bitmap words are MSB-first. Scales use original coordinate groups of 16.
Pages retain their existing data-row plane followed by the scale plane.

The new Direct loader stages 128 code bytes, bitmap words, exclusive word
population prefixes and scales once per selected row. It reconstructs eight
coordinates with V4 arithmetic rank selectors and register permutations,
adapted to contiguous nibble windows. Sixteen initialized guard bytes protect
aligned terminal-window reads. Dropped entries skip scale multiplication and
become positive zero; retained signed codes keep their original arithmetic.
Existing producer geometry, BF16 WGMMA layout, peer-store byte counts and
publication/reuse barriers remain. A full-warp completion rendezvous protects
scratch reuse between H64 producer rounds. Invalid rows read no global record.

CPU-only compiled sizeof audit before H100 allocation: Native shared memory
208896 B and Packed 225280 B for both H64/H128. Scratch is transient shared
memory, not part of the persistent 224-byte record. Stock Native FP4 and V4
reconstruction remain on their existing paths. Production imports no test oracle.

## Correctness evidence

| Gate | Outcome |
|---|---|
| CPU native build/core FP4 precompile |passed; verified SM90 targets, source/binary hashes|
| CPU startup/timing-contract tests |24 tests + 15 subtests; 172.87 s|
| Ordinary H100 correctness |39 cases; 49.86 s and 51.02 s on the two allocations|
| Strict memcheck |34/34; zero errors; 2736.45 s|
| Strict racecheck |34/34; zero errors/warnings; 1308.45 s|
| Strict synccheck |34/34; zero errors; 2022.15 s|
| Source review |graph-output validation gap corrected; fresh-context source/artifact review passed|
| Final paired timing |20/20 cases; 40 measured paths; 1200 raw samples|

The sanitizer matrix is 16 store/read, 8 attention/Adapter, 4 changing-input
graph/nondefault-stream, 4 C1/C2 producer, and 2 exhaustive-mask Direct cases.
It covers all 256 eight-coordinate masks, odd/even ranks, terminal windows,
E2M1 codes, scale groups, both page sizes 128/256, invalid/partial/duplicate
selections and poisoned padding. Independent CPU references preserve existing
tolerances; zero-global/zero-tail negative controls must fail them. The ordinary
suite also protects touched V4 dispatch/synchronization with two regressions.

Two bounded jobs supply complete individual tools with identical runtime,
fixture, source, binary, precompile, sanitizer and 34-case identities. Each
successful tool has a clean process/XML/log exit before its passing checkpoint.
The jobs were deliberately stopped after needed receipts; their overall result
remains failed/KeyboardInterrupt because the other per-run tools were unfinished.
Attempt001 also has seven interrupted synccheck cases: these do not qualify.
No partial tool is relabeled passed. The timing runner independently verified
all three complete matching receipts before launching measurement.

## Final Native versus Packed Direct timing

The maintained `--suite v41-fp4-decode` preserves all ten default V4 decode
head/batch cases and repeats them with production SWA geometry: H64/H128 ×
B8/16/24/28/32 × SWA64/page64 or SWA128/page128, Q1/K512. Global pages are 256
(the V4.1 C1 geometry); C2 page 128 is correctness-covered but not timed here.
Native is unpruned 288-byte FP4 through Native Direct; Packed is 224-byte FP4
through the changed Direct loader. This is not a historical V4 FP8 comparison
and does not measure old Packed versus new Packed.

Both paths share post-normalization/post-RoPE BF16 source rows, queries,
shuffled physical selections, lengths, SWA writer/layout and sink. Independent
CPU quantization/pruning/attention references validate each path. Actual captured
outputs are checked after capture and after all timing rounds. Timing includes
scheduling, attention and required split/combine; setup, selection/store,
allocation, reference work, compilation and graph capture are untimed.

Protocol: seed 20260919, ten warmups, thirty randomized paired rounds, one
hundred graph replays/sample. There is no sanitizer or profiler instrumentation
during timing. These are warm synthetic graph measurements, not serving TPOT,
TTFT, SLO capacity, quality or full-model working-set measurements. The sample
p95 is a round-average statistic, not request p95.

Means below are microseconds per complete operation. Positive change means
Packed is slower. [timing.csv](timing.csv) retains every case mean/median/p95,
paired ratio and input identity as a compact output artifact. Raw samples and
the larger canonical CSV remain in the existing Modal volume.

| SWA | Heads | Batch | Native µs | Packed Direct µs | Change |
|---:|---:|---:|---:|---:|---:|
|64|64|8|72.000|72.313|+0.43%|
|64|64|16|99.817|101.794|+1.98%|
|64|64|24|128.417|132.417|+3.11%|
|64|64|28|129.605|133.346|+2.89%|
|64|64|32|158.770|163.730|+3.12%|
|64|128|8|56.213|57.550|+2.38%|
|64|128|16|80.465|83.166|+3.36%|
|64|128|24|92.964|96.780|+4.10%|
|64|128|28|95.536|100.293|+4.98%|
|64|128|32|133.009|133.860|+0.64%|
|128|64|8|72.428|72.322|-0.15%|
|128|64|16|102.567|103.429|+0.84%|
|128|64|24|129.824|133.069|+2.50%|
|128|64|28|159.676|166.548|+4.30%|
|128|64|32|159.066|165.442|+4.01%|
|128|128|8|56.716|57.645|+1.64%|
|128|128|16|80.432|83.854|+4.25%|
|128|128|24|83.774|86.448|+3.19%|
|128|128|28|99.841|100.123|+0.28%|
|128|128|32|141.393|148.517|+5.04%|

The unweighted per-case geometric-mean Direct/Native ratios are 1.026912 for
SWA64 and 1.025758 for SWA128. The isolated -0.15% mean change at H64/B8/SWA128
is not evidence of a repeatable speedup. No workload weighting, confidence
interval or production performance claim is inferred.

## Allocation result

The actual physical allocation fixture passed again: identical 1 GiB available
cache/workspace budget, 40 layers, owners 2/8/14 (C2) and 20 (C1), context 131072,
window 128, global page 256, max requests 8 and prefill chunk 512.

| Quantity | Native | Packed |
|---|---:|---:|
| Full-token capacity |656384|800512|
| Total accounted bytes |1073600000|1073726464|
| SWA storage |467251200|467251200|
| Compressor state |270336|270336|
| BF16 prefill workspace |1048576|1048576|
| Complete Direct temporary/output peak |20492288|20492288|

Capacity gain: `(800512/656384-1)*100 = 21.9578783%`. The separate theoretical
record saving is `1-224/288 = 22.2222222%`. [allocation.csv](allocation.csv)
records global/indexer/storage reserves; reserved pages are already included.
This allocator fixture does not establish full-GPU capacity with weights,
activations, additional workspaces or serving workloads.

## Commands, artifacts and budget

Single maintained entry points: `scripts/modal/kernels.py::validate_kernels`
then `::benchmark_kernels`. Exact historical commands and all artifact hashes
are in [qualification.json](qualification.json). Fresh run IDs are mandatory.

```bash
export MODAL_PROFILE=peroace360 REMNANT_INTEGRATION=v0.5.22
export REMNANT_VALIDATION_SUITE=v41-fp4 REMNANT_VALIDATION_TIMEOUT_S=5400
modal run scripts/modal/kernels.py::validate_kernels \
  --run-id v41-fragments-20261025-attempt001 --sanitizer \
  --sanitizer-order memcheck,racecheck,synccheck --max-seconds 5340
modal run scripts/modal/kernels.py::validate_kernels \
  --run-id v41-fragments-20261025-attempt002 --sanitizer \
  --sanitizer-order synccheck,racecheck,memcheck --max-seconds 2640
modal run scripts/modal/kernels.py::benchmark_kernels \
  --run-id v41-fragments-20261025-timing001 --suite v41-fp4-decode \
  --qualification-runs v41-fragments-20261025-attempt001,v41-fragments-20261025-attempt002
```

Commands show the executed source identity; later documentation commits do not
replace the validated parent runner. Receipt reuse requires exact identities;
otherwise rerun qualification on the intended source rather than bypassing gates.
Both validation jobs used one H100 with 2 CPU cores/16 GiB; inner deadlines
bounded work inside the 5400 s platform cap. Final timing had a 900 s platform
cap and completed in 74.14 function seconds. CPU preflight/builds stayed on CPU,
cached artifacts were hash-verified, and no retries were automatic.

Canonical volume: `remnant-stage2a-results`. Raw prefixes:

- `microbench/v41-fragments-20261025-attempt001/validation/`
- `microbench/v41-fragments-20261025-attempt002/validation/`
- `microbench/v41-fragments-20261025-timing001/timing/`

User authorized **$20 additional cumulative compute across accounts**, including
builds/failures/reruns. $2 remained reserved; projected work was capped at $18.
Only peroace360 incurred follow-up compute: baseline $13.73409128, latest observed
$21.53927724, giving **$7.80518596** before billing lag/settlement. Billing
snapshots can revise; the reserve is preserved. Current rates were H100 $3.95/h,
CPU $0.0473/core-hour, RAM $0.008/GiB-hour. No account/settings changes, model
volumes, weights, full-model loads, serving, GPUQ, SM100 execution or profiling.
App/container inspection confirmed all three apps stopped and zero containers.

## History, cleanup and next step

The earlier scalar implementation phase remains historical: attempts 001–012,
prior $11.91 compute, 32-case synccheck/racecheck, interrupted memcheck and its
unqualified decision are retained in the JSON/Git history. Current qualification
belongs only to the new frozen source and 34-case evidence above.

Keep the canonical implementation/tests, two required prefill instantiations,
maintained runner and this single README. No copied source harness was added.
The temporary Modal CPU resource probe was removed. Owned local raw receipt
downloads are removed after the summary commit; canonical raw files remain in
the results volume. Compact timing/allocation CSVs and JSON stay in Git.

No required implementation, measurement or review work remains. Performance tuning is a separate decision; a Native
speedup has not been established. H20-3e, multi-GPU/full-model serving, quality,
C2-page timing and production readiness remain unverified.

Rollback: select `--dsv41-main-kv-cache-format auto`, or restore the recorded
baseline pins using the normal clean-worktree workflow. Native remains default.
