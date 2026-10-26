<!-- Author: Winston Cai. -->
# V4.1 Packed FP4 on H100

## Current production-geometry matrix

The active `v41-fp4-decode` suite now uses **H64, SWA128/page128, Q1/K512**,
batches 8/16/24/28/32, and compressed global pages **C1/256 and C2/128**
(ten cases). The V4.1 config has 64 query heads, one KV head and a separate
32-head indexer. The selected H100 runtime pads smaller attention-TP shards to
64 kernel heads. SWA64 and H128 were supplementary timing coverage and are
removed from the active timing matrix; their correctness tests remain.

Question: does the maintained timing matrix match the supported model's H100
attention geometry? Only case selection and its protocol/judge change. Protocol
`remnant-v41-fp4-decode-v2` prevents historical receipts from qualifying this
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
source identities; **v2 timing has not been run**. C2/page128 timing is new
coverage and has no historical measurement. Rollback restores the prior
SGLang pin `98991169803f4ec0b1b2c3f369695dc62d0bf3ec`.

Validation: **2 passed in 0.08 s** on Modal CPU (`ap-vTOOo7wo8VWRzuxlSPA0kn`),
using the two existing test functions above, pytest 8.4.2, one CPU/1 GiB,
a 120-second worker timeout and no GPU, network or volumes on the test worker.
The first CLI shell stopped without a usable test report and is not qualified;
the explicit-argv Sandbox retry passed. Both apps stopped; zero containers remain.
The account compute snapshot changed from $21.42800725 to $21.43512718
(about $0.00712 reported delta, subject to settlement). Exact arguments, source
hashes and retry evidence are recorded in `qualification.json` under
`production_geometry_correction`. SGLang commit: `a0f7ff18314c`; no kernel changes.

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
