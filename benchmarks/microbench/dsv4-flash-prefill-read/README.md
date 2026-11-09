# Native / Packed192 / Packed256 prefill-read

Author: Winston Cai.

## Decision and status

Unprofiled timing, numerical validation and full-matrix Nsight Compute coverage
passed. Keep the measurements and the benchmark; do not promote the current
Packed prefill reader as a speed improvement. The H100 allocation is released.
Reported total compute spend is **$3.90 of the $8 cap**, including CPU builds and
failed/interrupted attempts. No full model or weights were loaded, and allocation
retries were disabled.

The current Packed reader adds approximately 34–36 µs to reconstruction for K192
and 33–35 µs for K256 in these fixtures. Attention alone changes by less than 0.8%.
The complete K192 path is 1.25–21.05% slower than Native; K256 is 1.77–20.46%
slower (paired round ratios). Reducing the persistent record to K192 does not
produce a prefill-read speedup with this implementation.

## Complete operation: unprofiled mean µs

All cases use history 143,360; top-k 512 cache records; D512; C4 page64; SWA128.
“192” and “256” are retained coordinates per Packed record, not attention top-k.

| Heads | Query chunk | Cache policy | Native | Packed192 | Packed256 |
|---:|---:|---|---:|---:|---:|
| 64 | 512 | warm | 181.39 | 219.58 | 218.50 |
| 64 | 512 | rotating | 181.83 | 218.99 | 217.93 |
| 64 | 1536 | warm | 357.94 | 395.40 | 394.46 |
| 64 | 1536 | rotating | 358.20 | 395.21 | 394.15 |
| 64 | 8192 | warm | 1541.44 | 1595.42 | 1597.33 |
| 64 | 8192 | rotating | 1549.15 | 1595.07 | 1594.80 |
| 128 | 512 | warm | 264.51 | 301.15 | 301.30 |
| 128 | 512 | rotating | 264.39 | 299.97 | 299.29 |
| 128 | 1536 | warm | 622.70 | 662.62 | 662.65 |
| 128 | 1536 | rotating | 624.35 | 663.64 | 662.50 |
| 128 | 8192 | warm | 2920.39 | 2970.70 | 2978.53 |
| 128 | 8192 | rotating | 2934.90 | 2971.11 | 2986.48 |

Thirty deterministically randomized paired rounds after ten warmups; twenty eager
calls per timing sample. Native uses unchanged production records generated from unpruned FP32 inputs.
Packed uses the same raw inputs with the respective sparsification. Shared-input hashes match
across all three variants, stages and cache policies for each shape.

## What the counters show

H64 / Q512 / warm reconstruction, decimal MB. Nsight Compute kernel durations
are shown separately from the unprofiled latency table above.

| Variant | First C4 kernel DRAM read MB | RoPE kernel DRAM read MB | Total reconstruction DRAM read MB | L2 read M sectors | L2 write M sectors |
|---|---:|---:|---:|---:|---:|
| native | 20.668 | 0.000 | 21.049 | 1.057 | 1.747 |
| packed192 | 9.237 | 11.011 | 20.629 | 1.257 | 2.132 |
| packed256 | 11.717 | 11.183 | 23.280 | 1.368 | 2.131 |

Native's C4 dequantization is 69.31 µs in this capture. Packed192 unpacking is
77.92 µs and its separate RoPE pass adds 25.02 µs; Packed256 is 77.34 + 25.28 µs.
Common SWA preparation adds 6.24/7.49/6.98 µs respectively. The additional pass
rereads the BF16 tail and frequency table, then writes the rotated tail.

Across the twelve reconstruction shape/cache-policy cases:

| Counter ratio versus Native | Packed192 | Packed256 |
|---|---:|---:|
| DRAM read | 0.975–1.024× | 1.070–1.119× |
| L2 read sectors | 1.153–1.198× | 1.237–1.300× |
| L2 write sectors | 1.176–1.225× | 1.175–1.223× |

The persistent record is smaller (584 vs 264/328 logical bytes), but all variants
still create a dense 512-coordinate BF16 workspace, then run the same stock sparse
prefill attention. Packed192's saved first-kernel DRAM reads are approximately
cancelled by RoPE reads; additional L2 traffic remains. The complete H64/Q512/warm
operation reads 141.263 MB for Native, 142.818 MB for Packed192 and 144.908 MB for
Packed256. Across the full complete-path matrix, read traffic is 0.989–1.022×
Native for Packed192 and 0.998–1.034× for Packed256. The compact record does not
translate into a substantial full-path read reduction here.

The measured optimization opportunity is to remove unpack/RoPE preparation cost
and extra materialization traffic. Fusing RoPE with unpacking is a focused
candidate for removing a penalty; these results do **not** establish that fusion
alone would beat Native, or deliver a 15% serving throughput gain. Attention-only
work remains dense and nearly unchanged. No frontend change is needed to explain
this local prefill-read cost.

`traffic.json` retains all 408 launches, original kernel-name identities, six
counters per launch, source/input correlation and report hashes. `traffic.csv`
contains 108 operation totals; `kernel-traffic.csv` contains the individual
launches. L2 counters are sectors. Per-launch hit rates remain in the records;
percentages are not averaged across unequal traffic. Durations are serialized
profiler measurements and are not substituted for acceptance timing.

## Evidence and limits

All 208 tests in the H100 validation run passed with zero skips/failures/errors,
followed by all 36 three-variant timing cases. Source and loaded binary hashes match the CPU build
manifest and archived sources. Independent artifact verification recomputed all
3,240 raw samples' summary statistics and paired ratios; CSV agrees with JSON.
Fifteen focused local launcher/contract tests passed against the saved snapshot.
CPU image preflight also passed 139 checks with no failures or skips. The full
profile matrix contains 108 unique NVTX ranges in three nonempty reports: Native
120 launches, Packed192 144 and Packed256 144. Every launch has duration, DRAM
read/write, L2 read/write and L2 hit-rate counters. Case filters are disjoint and
their ID unions cover every report launch. All CSV exports were reparsed locally;
all capture sources, binaries and shared inputs exactly match timing. All 122
remote file sizes match the download, and the three reports, three capture JSONs,
result JSON and build manifest also pass independent remote-stream SHA256 checks.

Reconstruction includes historical C4 dequantization/unpacking and RoPE as required,
plus common SWA preparation. Attention includes stock sparse prefill and both
max/LSE log2 conversion kernels. Initialization, selection/store, metadata and
compilation are untimed. Complete-path timing is authoritative; isolated stages
need not add because cache state differs.

Numerical checking covers the entire reconstructed workspace, all-output
finiteness, and eight deterministic attention queries per shape against an
independent Torch reference; separate GPU tests exercise partial/invalid entries,
duplicate selections, destination mappings and changed-input graph replay.
Packed attention is checked against its own sparsified reference, not numerical
equality with unpruned Native. The log2 max/LSE validation correction changed no
runtime arithmetic and relaxed no tolerance.

These are serving-informed synthetic one-request operations, not recorded LSWB
query/KV tensors or a model replay. Four-buffer rotation tests cache sensitivity;
it does not reproduce model-weight traffic. This benchmark cannot establish
TTFT/TPOT, quality, a 15% serving speedup, or SLO concurrency.

## Ownership, reproduction and rollback

`timing.json` is canonical; `timing.csv` is a compact flat export retaining raw
samples and input hashes. `manifest.json` records sources, binaries, image and
archive identity; `validation.json` records independent verification. Raw build,
validation and launch logs remain under the manifest's local raw path and Modal
results volume. No model weights were mounted or loaded.

The user requested finishing from the verified attempt-6 source snapshot after
concurrent edits reverted the working Modal runner. The workspace runner was left
intact. The recorded image-only overlay pins image `im-TOIm82uTJPtRtXc17pxChK`;
all launcher function ASTs match the archived source. The original archive and
exact overlay patch are retained in `raw/modal/prefill-read-20261018-113435/attempt-6/`.
Do not infer that the current workspace launcher contains this batching logic.

The numerical entry point remains
`third_party/sglang/benchmark/remnant/microbench.py --suite prefill-read`;
existing decode/store defaults are unchanged. The active study index and protocol
are its adjacent README. No production runtime/kernel code changed. Disable the
study by omitting `--suite prefill-read`; no serving rollback is necessary.
The saved launcher can reproduce the study with a new run ID (new compute still
requires authorization):

```bash
MODAL_PROFILE=poohthewinniechurchill modal run --detach \
  raw/modal/prefill-read-20261018-113435/attempt-6/source/scripts/modal/kernels.py::profile_kernels \
  --run-id <unique-id> --suite prefill-read --max-seconds 3500
```

Original timing entry point: `scripts/modal/kernels.py::benchmark_kernels`, run ID
`prefill-read-20261018-113435-r6`. Profiles use that run ID and the saved image-only
overlay. The precise original commands, failed attempts, source archive and
binding patch are recorded in the raw launch ledger. No production source was
patched; the overlay changes only which already-built image is selected.

Active code remains the SGLang benchmark and its shared cases/tests. The extracted
saved launcher is retained only as this run's documented reproduction source;
it is outside maintained builds/uploads. Historical attempts and original
profiles remain outside Git, in the documented raw directory and Modal volume.
Canonical-runner reconciliation is deferred because the user explicitly requested
snapshot continuation while preserving concurrent workspace edits.
The original experiment did not commit or push. This curated collection is now
committed; that does not change the recorded source or validation scope.
