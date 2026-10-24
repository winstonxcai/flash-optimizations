<!-- Author: Winston Cai. -->
# V4.1 Packed FP4 on H100

**Decision: keep the implementation experimental and opt-in; unqualified.**
Numerical, graph/stream, allocation, synccheck, racecheck and fresh-context source
review gates passed. Modal disabled the workspace during memcheck after 27/32
cases. There is no clean memcheck summary/successful exit, so that gate does not
pass. All allocations are stopped. Reported cumulative task compute is $11.91;
only $1.09 of the $13 work allowance remains, with the $2 reserve preserved.
A complete memcheck rerun does not fit that allowance. Do not promote this candidate.

Question: can H100 preserve V4.1 Native FP4 quantization while retaining 256
coordinates and consuming 224-byte Packed global records directly? Baseline
policy: keep stock auto, SWA/indexer behavior and v0.5.18 intact; compare fresh
synthetic Native/Packed numerical references at identical request/SWA policies.
No quality or speed claim is inferred from the engineering tests.

## Source isolation and ownership

Managed worktree: `codex/v41-packed-fp4`, at
`/Users/poohthewinnie/.codex/worktrees/v41-packed-fp4/flash-optimizations`.
Original dirty checkout and other worktrees are preserved. No pushes.

| Source | Baseline | Current candidate |
|---|---|---|
| Parent | `9419d0e18e1e7677c0e48b8e40ad54bbc30010ca` | validated runner `9ec5971da3e117691311e43680d2ff3e6b8d6885` |
| SGLang v0.5.22 | `7856615f3fdf02e993f3b40ecdbbed8cb8ac9bf7` | `0a36a662562ff3aa98ebc5abca12dcc7cff83168` |
| FlashMLA v0.5.22 | `3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8` | `91463e004ec99762f39a173cd687f9ab98d3628d` |
| CUTLASS | `147f5673d0c1c3dcf66f78d677fd647e4a020219` | unchanged |

SGLang owns the flag, producer/store, reconstruction, pools, accounting and
runtime/AOT dispatch. FlashMLA owns separate FP4 Direct and BF16 prefill entries.
The [fork guide](../../../third_party/README.md) maps changed modules. Production
imports no experiment reference. Two FP4 prefill instantiation files own the
length/no-length specializations; existing CMake/setup consume them.

## Numerical and persistent contract

`--dsv41-main-kv-cache-format auto|fp4|remnant`: auto preserves stock behavior;
fp4 selects Native, remnant selects Packed. Explicit modes reject conflicting
legacy choices, HiCache, distributed cache transfer, CP and bounded-SWA replay, speculation,
DSpark, offload and incompatible prefill backends. Direct/prefill fixtures cover
local query heads 64/128, D512 and SM90; other head counts are unsupported.

Preserve upstream FP32 normalization, BF16 rounding, RoPE and FP4 quantization.
Select exactly 256 greatest absolute **dequantized FP4** coordinates, ties by
lower original coordinate. Retain codes and all 32 E4M3 scales unchanged: no
renormalization/requantization. Indexer input remains original/unpruned.

| Record | Codes | Bitmap | E4M3 scales | Logical total |
|---|---:|---:|---:|---:|
| Native |256 B|0|32 B|288 B|
| Packed |128 B|64 B|32 B|224 B|

Packed survivors follow ascending coordinates; even ranks use low nibbles.
Eight uint64 bitmap words are MSB-first. Scale lookup uses original groups of 16.
Pages store all data rows followed by all scale rows, with 256-byte page alignment.
Global records are already RoPE transformed; Direct does not rotate them again.
Existing H100 SWA retains 448 FP8 NoPE bytes, 64 BF16 rotary coordinates and 8 scale
bytes (584 logical bytes; 585 physical bytes/token at the tested page sizes).
The existing 128-dimensional indexer uses 68 bytes/row and unchanged arithmetic.

Software conversion/loading supports both new modes on SM90. Reconstruction
writes caller-owned BF16 buffers for prefill and Adapter. Explicit FP4 modes,
including SWA-only layers, use a separate prefill barrier policy; stock auto
retains its original dispatch. Cache ownership/reuse and C1/C2 pending state
remain with the existing V4.1 pools.

## Evidence and acceptance

Canonical measurement/attempt receipt: [qualification.json](qualification.json).
[allocation.csv](allocation.csv) is its flat two-row allocation summary for
inspection and downstream analysis; it is an artifact, not another runner.
Raw XML/logs/manifests remain in `remnant-stage2a-results` at
`microbench/v41-fp4-20261025-attempt011/validation/` and
`microbench/v41-fp4-20261025-attempt012/validation/`.

| Gate | Outcome | Evidence |
|---|---|---|
| CPU native build/core FP4 precompile |passed|Modal CPU workers; SM90 binaries/hash checks before GPU |
| CPU startup/order guards |passed|23 tests + 15 subtests; order-boundary checks |
| Ordinary correctness |passed|37 cases, 51.58 s on attempt012 |
| Graphs/streams, allocation, V4 regressions |passed|4 changing-input graph/stream cases, physical pools/edges and 2 V4 controls within the ordinary suite |
| Strict synccheck |passed|32 cases, 2063.42 s; zero errors, attempt011 |
| Strict racecheck |passed|32 cases, 1298.37 s; zero errors/warnings, attempt012 |
| Strict memcheck |**incomplete**|27/32 cases, 2506.21 s; KeyboardInterrupt/workspace disabled; no clean summary |
| Fresh-context review |passed|no blocking findings in arithmetic/layout/accounting/synchronization or scoped runner changes |

The 32-case sanitizer matrix comprises 16 store/read, 8 attention/Adapter,
4 graph/stream and 4 C1/C2 producer cases. It covers ties, zeros, rounding edges,
partial/invalid selections, poisoned padding, both page geometries, both query
head counts, and both Adapter length variants. Negative controls require
zero-global/zero-tail loader faults to fail the unchanged 0.02 attention tolerance.
Independent attention expectations execute in FP32 on CPU, cast output to BF16;
production store/read/Direct/Adapter and graph operations execute on GPU.

Attempt011/012 source content, all binary hashes, compiled FP4 artifacts,
coverage and sanitizer identity match exactly. Completed synccheck/racecheck
case IDs also match. Individual successful receipts may be combined only with
these identities and complete clean tool exits; each runner invocation retains
its all-three gate. Neither interrupted run is relabeled a full passing run.
Memcheck is missing `test_fp4_graph_and_stream[128-True]` and all 4 producer cases, and
**the full 32-case memcheck run must be repeated**. Partial cases are not qualification.

All executable validation ran on Modal. No weights, model volumes, full-model
loads, serving, GPUQ, SM100 execution, profiling or performance benchmarks.
Existing mocked launcher unit tests were extended but not run as a standalone
local suite; real CPU boundary guards/reordering/checkpoints ran on Modal.

## Physical allocation result

Synthetic 1 GiB available cache/workspace budget: 40 layers, source owners
2/8/14 (C2) and 20 (C1), context 131072, window 128, global page 256, request cap 8,
chunked prefill 512. Source derives page geometry/reservations. Native and Packed
use identical caps/SWA policies, including both modes' physical dummy pages.

| Quantity | Native | Packed |
|---|---:|---:|
| Full-token capacity |656384|800512|
| SWA storage |467251200 B|467251200 B|
| Global main storage |472891392 B|448516096 B|
| Indexer storage |111646208 B|136147968 B|
| Compressor state |270336 B|270336 B|
| BF16 prefill workspace |1048576 B|1048576 B|
| Complete Direct temporary/output peak |20492288 B|20492288 B|
| Total accounted storage |1073600000 B|1073726464 B|
| Reserved pages, already included above |6346240 B|6280704 B|

Measured physical tensor allocation reconciles whole-pool disjoint ownership
and 32 boundary cases. Capacity gain is `(800512/656384 - 1)*100 = 21.9578783%`.
Record savings are separately theoretical: `1-224/288 = 22.2222222%`.
The indexer stays 68 B/row; the variable compressed bytes/full-token are
`2.5*(record+68)`, giving 890 B Native versus 730 B Packed before page granularity.
The result is a declared allocator fixture, not full-GPU/serving/SLO capacity.
Workspaces outside the declared fixture, model weights and activations are not
established by these model-free tests. Runtime engineering remains unqualified
until memcheck passes.

## Commands, cost and stopping rule

The single maintained entry point is `scripts/modal/kernels.py::validate_kernels`.
Exact final executed commands follow (historical run IDs cannot be reused):

```bash
MODAL_PROFILE=wpooh1579 REMNANT_INTEGRATION=v0.5.22 \
REMNANT_VALIDATION_SUITE=v41-fp4 REMNANT_VALIDATION_TIMEOUT_S=2700 \
modal run scripts/modal/kernels.py::validate_kernels \
  --run-id v41-fp4-20261025-attempt011 --sanitizer --max-seconds 2640

MODAL_PROFILE=wpooh1579 REMNANT_INTEGRATION=v0.5.22 \
REMNANT_VALIDATION_SUITE=v41-fp4 REMNANT_VALIDATION_TIMEOUT_S=4500 \
modal run scripts/modal/kernels.py::validate_kernels \
  --run-id v41-fp4-20261025-attempt012 --sanitizer \
  --sanitizer-order racecheck,memcheck,synccheck --max-seconds 4440
```

CPU preflight is bounded to 900 s for the final scoped runner; FP4 validation uses
one H100, 2 CPU cores/16 GiB, no automatic retries. Native compilation stays on CPU;
hash-verified artifacts precede GPU allocation. The order accepts only complete
permutations of all 3 tools and checkpoints each completed result immediately.
Earlier attempt011 used 8 CPU cores/64 GiB and a 2700 s GPU cap. No new launch followed the
workspace failure. Metadata inspection confirmed zero live containers afterward.

Cumulative budget is $15 across accounts, including failures/builds, with $2
reserved and at most $13 projected work. Baseline compute on wpooh1579 was
$18.51013772; final observed compute $30.42292268 gives $11.91278496 task compute
before billing lag. Historical storage/credits are not mistaken for task compute.
CPU costs $0.0473/core-hour, RAM $0.008/GiB-hour, H100 $3.95/hour; the final requested
worker is $4.1726/hour. The projected final launch bound was $12.82 cumulative work.
Remaining work allowance $1.08721504 cannot fund the interrupted 41m46s memcheck,
its remaining cases and CPU preflight. Stop and retain the unqualified result.

## History, cleanup and next step

Full attempts 001–012, source identities, hashes, costs, failure reasons and raw
artifact locations are preserved in JSON/Git. Two meaningful corrections were
reviewed: isolated FP4 prefill uses CUTLASS unaligned named barriers for divergent
consumer roles; the independent attention oracle moved to CPU. The stock prefill
fault's exact binary/PC diagnosis remains in the receipt/history. No suppression,
tolerance relaxation or reduced sanitizer matrix was used. Control-connection
warnings recovered on the existing job; no replacement allocation was launched.

The canonical implementation/tests and two necessary FP4 prefill instantiations
remain active. No duplicate source harness/probe was retained. README chronology
is consolidated here; historical receipts remain in JSON and the existing Modal
volume. Temporary local receipt downloads are removed after the summary commit.
v0.5.18 and unrelated dirty work are preserved; fork commits/pins are recorded;
nothing is pushed.

Next: obtain a new budget and enabled Modal account, check its live spend/cache/
competing allocations, then use the same entry point with a fresh run ID and
`--sanitizer-order memcheck,synccheck,racecheck`. A full clean 32-case memcheck is
required. Reuse earlier successful tools only when runtime/fixture/binary/tool
hashes and complete case IDs match; otherwise requalify all gates. Quality,
full-model/serving, performance and production readiness remain deferred.

Rollback: select `--dsv41-main-kv-cache-format auto`, or return to the recorded
baseline pins through the repository's normal clean-worktree workflow.
