# Fused Packed prefill unpack and RoPE

Author: Winston Cai.

## Decision and status

Timing, numerical/sanitizer validation and full-matrix traffic coverage passed.
Keep the opt-in candidate as a measured partial improvement; do not promote it
as a Native speedup. The split reader remains the default. The candidate retains
RoPE and removes its separate pass. Both H100 allocations have stopped.

The question is whether this single fusion closes the Packed prefill-read
penalty versus unchanged Native on the same inputs. Acceptance requires exact
split/fused BF16 parity, the independent references, all-count/edge/graph/JIT
coverage, sanitizer success and complete-operation improvement. No tolerance,
layout, producer, Native operation, frontend or attention kernel changed.

## Complete operation: unprofiled mean microseconds

History 143,360; attention top-k 512 records; D512; SWA128; C4 page64.
Packed192/Packed256 count retained coordinates per record. Ten warmups, thirty
deterministically randomized paired rounds, twenty eager calls per sample.

| Heads | Query chunk | Cache policy | Native | Packed192 fused | Packed256 fused |
|---:|---:|---|---:|---:|---:|
| 64 | 512 | warm | 180.98 | 204.60 | 203.18 |
| 64 | 512 | rotating | 180.52 | 203.03 | 202.99 |
| 64 | 1536 | warm | 358.26 | 380.71 | 381.35 |
| 64 | 1536 | rotating | 357.13 | 379.26 | 379.67 |
| 64 | 8192 | warm | 1555.41 | 1588.87 | 1585.34 |
| 64 | 8192 | rotating | 1558.38 | 1597.06 | 1587.31 |
| 128 | 512 | warm | 263.38 | 286.04 | 285.91 |
| 128 | 512 | rotating | 264.36 | 286.54 | 286.31 |
| 128 | 1536 | warm | 624.32 | 649.92 | 651.17 |
| 128 | 1536 | rotating | 627.73 | 652.75 | 652.51 |
| 128 | 8192 | warm | 2940.73 | 2971.58 | 2984.56 |
| 128 | 8192 | rotating | 2948.58 | 2973.33 | 2990.50 |

All 36 case definitions and input hashes match the original
[split-reader study](../prefill-read-20261018/README.md). Fresh Native controls
are measured together with the candidate. Old split versus new fused results
are separate runs, not paired timings. Complete-path timing is authoritative;
isolated stages need not add because cache state differs.

H64/Q512/warm reconstruction: Native 75.34 us, Packed192 96.45 us, Packed256
96.58 us. Original split results were 75.62 / 110.16 / 109.14 us. Fusion removes
part of the penalty but does not beat Native. The complete Packed192 path is
0.86–13.08% slower than its paired Native control; Packed256 is 1.44–12.45%
slower. No serving throughput, TTFT, quality or 15% speedup claim follows.

## What fusion changes

H64/Q512/warm reconstruction counters; decimal MB and million L2 sectors.
Durations here are serialized profiler measurements, separate from unprofiled
latency above. L2 hit rates remain per launch in `traffic.json`.

| Variant | Profiled us | DRAM read MB | L2 read M sectors | L2 write M sectors |
|---|---:|---:|---:|---:|
| native | 75.74 | 21.043 | 1.058 | 1.760 |
| packed192 | 96.83 | 18.684 | 0.994 | 1.753 |
| packed256 | 96.61 | 21.381 | 1.101 | 1.752 |

The new Packed C4 fused kernel takes 89.57 us (K192) / 89.50 us (K256), versus
69.54 us for Native C4 dequantization. The common SWA preparation accounts for
the rest. The separate RoPE launch is absent from every Packed capture. The
previous split C4 unpack+RoPE totals were 102.94 / 102.62 us in the original
capture. Fusion saves about 13 us of that preparation, while roughly 20 us
relative to Native remains inside the C4 reader. Attention-only timing stays
within 0.6% of Native. The residual is localized to reconstruction; these six
traffic counters do not identify which instruction, latency or occupancy cost
dominates it.

The smaller cache record is not the entire prefill read bill. Native already
stores its rotated 64-coordinate tail as 128 BF16 bytes. Packed keeps the
unrotated sparse coordinates and still reads 32 complex FP32 frequencies
(256 logical bytes per valid C4 row) to perform the same rotation. Ignoring
metadata, the ideal inputs are therefore 584 bytes for Native, 264+256=520
for K192 and 328+256=584 for K256. Cache reuse/coalescing affect actual DRAM
traffic. Every path still writes a dense 1,024-byte BF16 row and uses unchanged
sparse-prefill attention over dense BF16 coordinates. Removing the extra pass reduces movement,
but does not make this a 264-versus-584-byte complete operation.

All 108 operation/range exports partition all 360 captured launches exactly:
120 Native, 120 Packed192, 120 Packed256; every launch has duration, DRAM read
and write, L2 read and write, and L2 hit rate. Kernel names and roles, individual
launches, report hashes and all sums are retained in `traffic.json` and flat
CSVs. `comparison.json` retains counter-ratio bounds for every reconstruction
and complete-path case.

K192 reconstruction DRAM reads are 7.4–11.2% below Native across the matrix;
complete-path reads range from 1.9% lower to 0.2% higher. K256 reconstruction
reads are 0.5–1.6% higher; complete reads range from 1.4% lower to 0.7% higher.
The full attention/read path therefore exposes far less traffic reduction than
the persistent record-size ratio suggests. These results do not establish the
15% serving throughput target.

## Export failure and recovery

The GPU job completed all three reports and all 108 capture identities, then
reached its 850-second work deadline after 86 CSV exports. Its failed result
and logs remain unchanged. The attempted CPU-only export worker was refused
before allocation by the Modal workspace billing-cycle limit. No capture was
retried and no second GPU was allocated concurrently.

The installed NVIDIA report API completed all exports locally from the saved
reports. All 86 existing CLI exports independently match the report API's
kernel identities and six counters; CLI hit-rate printing rounds to two decimal
places, while the API retains full precision. This formatting comparison does
not change numerical or performance tolerances. Capture tool: Nsight Compute
2025.3.1. Local reader: 2026.3.1.0. The API's NVTX selections are disjoint and
cover every action in each full report, including the 22 missing CLI exports.

## Evidence and limitations

The H100 suite passed 276 tests and 15 subtests, with zero failures/errors/skips.
Memcheck, racecheck and synccheck each passed 52 tests and 15 subtests, with zero
sanitizer errors or race hazards. CPU image preflight passed 95 tests and 45
subtests with zero skips. Local checks passed 61 benchmark tests (99 GPU skips
subsequently covered on H100) and 44 launcher tests.

Every C4 row/bank in candidate fixtures matches the split reader's BF16 bytes.
Tests cover all counts 64..448, invalid/padded rows, duplicate selections with
different logical positions, independent physical mappings, partial lengths,
pruned RoPE partners, strided/empty inputs, changed planes/indices/frequencies
on graph replay, stable pointers and compile reuse. Native and Packed outputs
also retain their independent references; Packed is not required to equal
unpruned Native. Full attention finiteness is checked, with eight deterministic
queries per shape checked against independent Torch softmax/sink results.

All 3,240 raw samples, mean/median/p95/stdev, paired ratios and 108 CSV rows were
independently checked. Uploaded source hashes and loaded FlashMLA/Native-store
binary hashes match CPU preflight. The old and new builds have different binary
hashes; cross-run comparison is descriptive. Fresh Native complete-path controls
differ from the original by -0.72% to +0.91%.

These are serving-informed synthetic one-request operations, not recorded
LongSWEbench tensors, model replays or a serving-capacity measurement. Four-bank
rotation tests cache sensitivity; it does not reproduce model-weight traffic.

## Ownership and rollback

Runtime candidate: `unpack_gather_bf16(..., fused_rope=True)`; default false.
Benchmark: `--suite prefill-read --prefill-reader fused`; default split.
No production caller opts in. Disable with `--prefill-reader split` or omit
`fused_rope`. Existing historical results remain unchanged. Committing this
study preserves the opt-in candidate; it does not promote the serving default.

The maintained Modal runner was concurrently restored during this work. Per
user instruction, the recorded execution used the verified saved source snapshot
and both explicit fork source overrides. The exact
source archive, command, image, raw artifact location and hashes are recorded
in the run manifest. The snapshot is owned by this experiment and excluded
from normal uploads/builds. `candidate-sglang.patch` records the focused changes
relative to the archived pre-edit state; unrelated dirty changes are excluded.

## Budget, artifacts and next entry point

Conservative observed compute spend is **$1.99 of the additional $4 cap**
(latest settled account delta $1.98; storage and credits are excluded). The
original study's $3.90 high-water record plus this study is approximately $5.90
of the original $8 authorization. CPU builds and the capped export failure are
included. There were no automatic allocation retries or model loads. Timing,
profiling and the blocked CPU-export app all show stopped with zero tasks.

Large reports and copied source remain outside Git. The verified source archive
and local postprocessing bundle are durable in `remnant-stage2a-results`:

```text
microbench/prefill-read-fused-rope-20261018T150514Z/submitted-sources.tar.gz
microbench/prefill-read-fused-rope-20261018T150514Z/postprocess.tar.gz
microbench/prefill-read-fused-rope-20261018T150514Z/profile/prefill-read-{native,packed192,packed256}.ncu-rep
```

The first archive contains the exact uploaded forks, controller and launcher
checks; the second contains the local exporter, all full/filtered counter CSVs,
completion record and retained failure evidence. Every report/capture JSON,
original result and build manifest was independently SHA256-checked against a
remote stream. Both uploaded archives were read back and hash-verified. The
manifest records all hashes and commands.

The maintained numerical entry point is still the SGLang microbench with
`--suite prefill-read --prefill-reader fused`. The maintained Modal runner now forwards this option and batches the same
three variants; its local mocked checks pass. Use
`scripts/modal/kernels.py::benchmark_kernels` followed by `::profile_kernels`,
with the suite, reader and unique run ID explicit. The source-snapshot controller
preserves exact historical reproduction; `export_profiles.py` in the
postprocessing archive owns offline export recovery. Any new Modal execution needs its own applicable
budget authorization. The next unresolved question is the approximately 20 us
C4-reader gap versus Native, using instruction/latency/occupancy evidence;
these traffic counters alone do not establish its exact hardware cause.
