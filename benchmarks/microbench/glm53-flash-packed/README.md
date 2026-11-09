# GLM-5.3-Flash Packed

## Decode performance follow-up

### 2.0× follow-up

Starting point: measured SGLang `29fe520dd457` and evidence commit `36b8ad8`.
The question is whether Hopper WGMMA can cut Packed partial attention time
enough to reach an equal-case Direct/Native ratio of 0.5000 on the same H200
matrix. The prior candidate remains the batch-1 control. For batches 8–64,
test 64 selected rows per tile, transposed QK and PV, and the existing
cooperative expansion and parallel combine. The cache ABI, pristine Native
source, query conversion, fixture inputs and timing boundaries stay fixed.

CPU-compile and inspect the SM90a instructions first. Use the maintained
eight-case diagnostic to check numerical parity, graph replay and stage
latencies before qualifying a candidate. Acceptance requires focused
sanitizers and the unprofiled 24-case, 30-round matrix with fresh paired Native
samples. Preserve the prior result and any failed candidate. Stop scheduling
when a launch would exceed $20, counting the $1.73 already metered.

The first SM90a build emitted 32 FP8 `QGMMA` sites and zero local-memory
instructions per specialization. The generated KV panel offsets matched the
loader and the transpose buffer was disjoint from both QK operands. The
generic transpose emitted 294 static CTA barriers, with 256 inside the
per-tile transpose. Its CPU evidence is under
`data/modal/glm53-decode-speed/wgmma-cpu-artifacts` in the experiment worktree.
The next revision uses a per-lane 16×16 register transpose and verifies the
CTA barrier count before H200 profiling.
It reduced the count to 39 without spilling. The generated batch-8 kernel
overlapped its transpose destination with Q and KV, so that revision remained
CPU-only. The next build disables TileLang shared-memory reuse for this
kernel and rejects overlapping generated buffer ranges during preparation.

The disjoint-buffer build passed all eight H200 diagnostic cases, including
independent numerical and changing-input graph checks. At batch 32, Direct
partial attention fell from 50.17–53.50 µs to 39.78–42.10 µs. At batch 64,
it fell from 93.02–93.79 µs to 81.48–81.85 µs. The equal-case speedup from
the eight sums of traced kernel durations was 1.512×. The complete 24-case
matrix is still pending. See `decode_wgmma_followup` in the
[qualification ledger](qualification.json) for stage timings, commands,
compiled artifact identities and raw trace hashes.

The 39 static CTA barriers include separate max and sum reductions for each
of the four heads owned by a lane. The next candidate batches those
reductions using the pinned TileLang API, keeping the accumulator layout
and softmax recurrence. CPU preparation now checks for at most 24 static
CTA barriers and rechecks the disjoint shared ranges.

The batched reductions emitted 15 CTA barriers with zero spills. The next
eight-case H200 diagnostic passed and reached 1.551× on the summed stage
traces. The batch-64 partial measured 76.84–77.54 µs. The following candidate
delays each tile's PV issue until the start of the next iteration, where it
can overlap expansion into separate K storage. The final PV drains after
the loop. The experiment will keep the same attention recurrence and check
the generated waits before H200 profiling.

The first pipelined build reached artifact binding. Batch 8 produced the same
device code as the previous build, with a second cached host wrapper. The
retry binds each library to the exact TileLang JIT cache entry and checks it
against the adapter's generated host and device code before profiling.

The pipelined candidate passed all eight diagnostics at 1.555× on summed
stage traces, compared with 1.551× for batched reductions alone. Batch-64
partial attention still took 76.83–78.25 µs. The fixture hashes and pristine
Native partial cubins matched the preceding run. Preparation and profiling
bring conservative compute accounting to $2.78.

The next candidate issues aligned asynchronous copies for both 32-row Packed
payload waves before waiting and expanding them. Its shared scratch grows
from 8.5 to 17 KiB so each wave owns its staging area. The bitmap ranking,
swizzled FP8 tiles, PV schedule and baseline stay fixed. Verify the copy and
wait instructions, shared ranges and register usage before profiling.

CPU preparation verified eight async copy sites per variant, no stack or
local traffic and unchanged upstream Native artifacts. All eight H200
diagnostics passed numerical and changing-input replay checks. The traced
speedup fell to 1.542× and batch-64 partial attention rose to
78.52–79.99 µs. Keep the results in the ledger and return to the batched
kernel and its original gather for the next comparison. Conservative compute
accounting through the async diagnostic is $3.07.

The next candidate replaces the per-lane K-to-V transpose with Hopper matrix
loads and stores. Each warp owns 16 history rows. It transposes pairs of FP8
bytes with a 16-bit matrix load, then uses byte permutations to put each
feature's consecutive history bytes into the SW64 PV operand. A CPU address
model checked all 32,768 bytes against the generated SW128 and SW64 layouts.
The Packed ABI and FP8 attention arithmetic stay fixed. Check generated
matrix instructions, register use and output parity before measuring it.

The compiled transpose used 16 transposed matrix loads and 16 non-transposed
matrix stores per specialization, with no spills. Batch-64 register use
fell from 243 to 141 per thread, and shared memory still allowed two CTAs
per SM. All eight H200 diagnostics passed. Batch-64 Direct partial attention
fell from 76.84–77.54 µs to 63.59–65.15 µs. The equal-case speedup on summed
stage traces rose from 1.551× to 1.751×. Keep this image as the control for
the next experiment. Conservative compute spend is $3.38.

Test whether warp-level PV can read FP8 inputs from the same expanded SW128
K tile while keeping warpgroup QK. This removes the 32 KiB V allocation
and the transpose.
The two GEMMs distribute heads across different lanes, so publish the
16 FP32 scale factors through shared memory. CPU preparation must confirm
the QK and PV instruction families, any FP16 conversion, SW128 layout,
register use and actual shared-memory footprint before running the same
H200 diagnostic.

### Original 1.20× target

The goal is at least 1.20× equal-case geometric-mean speedup over untouched
upstream FP8 TileLang on one H200. The score uses the mean of each case's 30
paired Direct/Native ratios and all 24 cases. Preserve the existing fixture,
Packed ABI, numerical tolerances and changing-input replay checks. The $20
Modal budget covers preparation, profiling, candidates and final confirmation.

The maintained upstream decode entry point accepts `--profile-decode` for a
separate CUDA trace of batches 1, 8, 32 and 64 with zero tails under both cache
policies. It exports per-mode graph kernel durations and the raw traces. Use
those traces to decide which stage to optimize. Run without this flag for
acceptance. The timing output reports the equal-case score alongside the
existing batch-weighted statistic.

### Baseline trace on H200

| Batch | Cache | Native Q / partial / combine (µs) | Direct Q / partial / combine (µs) |
|---:|---|---:|---:|
| 1 | warm | 1.02 / 5.41 / 13.18 | 1.06 / 5.31 / 13.15 |
| 1 | rotating | 1.03 / 5.87 / 13.18 | 1.03 / 5.83 / 13.11 |
| 8 | warm | 1.15 / 12.03 / 8.74 | 1.12 / 13.34 / 8.80 |
| 8 | rotating | 1.13 / 13.13 / 8.44 | 1.14 / 14.13 / 8.45 |
| 32 | warm | 1.66 / 48.70 / 12.32 | 1.70 / 49.47 / 10.40 |
| 32 | rotating | 1.75 / 49.09 / 11.81 | 1.72 / 50.05 / 10.50 |
| 64 | warm | 1.92 / 95.31 / 19.36 | 2.03 / 97.61 / 19.75 |
| 64 | rotating | 2.17 / 95.19 / 19.53 | 2.25 / 97.66 / 19.81 |

At batch 1, combine accounts for about two thirds of traced Direct kernel
time. At batch 64, partial attention accounts for about four fifths. Test
parallel combine at low batch and fewer splits at higher batches. These
diagnostic kernel durations come from one graph replay per mode and case.
The profile manifest, trace hashes and exact command are in the qualification
ledger.

### Tuned H200 decode result

Packed Direct achieved **1.3045× geometric-mean speedup** over untouched
upstream FP8 TileLang. The equal-case paired ratio is **0.76657**, a **23.34%**
latency reduction. All 24 case means favor Direct. Ratios range from 0.44283
to 0.96887. Batch 1 gains the most, while batch 16 rotating gains the least.

The tuned kernel distributes combine work over heads and output columns and
reduces the number of partials at higher batches. The tuned diagnostic put
batch 1 Direct combine at 1.92–2.11 µs, down from 13.11–13.15 µs in the baseline
Direct trace. At batch 64, the tuned split policy emits six partials per query
instead of 33, and the Direct combine trace fell from 19.75–19.82 µs to
2.67–2.69 µs. Those traces are separate from the acceptance measurements below.

The acceptance run `glm53-decode-tuned-acceptance-20261110` measured complete
attention calls on one H200 with 16 local heads. It used fresh Native samples,
30 balanced paired CUDA-graph rounds per case, and the same input hashes and
bank counts as the historical matrix. All fixtures passed exact Packed-code
and independent numerical checks at `atol=rtol=0.02` and changing-input graph
replay. H100 memcheck, racecheck and synccheck each passed 13 focused cases,
covering the byte loader, seven partial variants and five combine variants.

Measured parent `66aee8c308cc`, SGLang `29fe520dd457`, FlashMLA `858bb1ccac6f`,
CPU-prepared image `im-GxvdtgbwKar2lKco4w1lAq`. The pristine module is from
`baa9befa381511cd4d56f83980951e5a45fffc6d`, with identical Native source and
cubins across both profiles. The [ledger](qualification.json) contains exact
commands, image/source/artifact hashes, trace and sanitizer receipts, and
the raw output locations. The [tuned CSV](upstream-decode-tuned.csv) contains
both modes and every paired sample. The existing comparison below retains
its original data.

| Batch | Tails | Cache | Native µs | Direct µs | Mean paired Direct/Native |
|---:|---:|---|---:|---:|---:|
| 1 | 0 | warm | 20.388 | 9.990 | 0.49002 |
| 1 | 0 | rotating | 19.938 | 8.829 | 0.44283 |
| 1 | 3 | warm | 20.352 | 9.958 | 0.48930 |
| 1 | 3 | rotating | 19.904 | 8.816 | 0.44289 |
| 8 | 0 | warm | 22.968 | 19.722 | 0.85868 |
| 8 | 0 | rotating | 22.734 | 18.763 | 0.82536 |
| 8 | 3 | warm | 22.918 | 19.498 | 0.85077 |
| 8 | 3 | rotating | 22.762 | 18.920 | 0.83123 |
| 16 | 0 | warm | 34.826 | 33.255 | 0.95489 |
| 16 | 0 | rotating | 35.711 | 34.550 | 0.96749 |
| 16 | 3 | warm | 34.888 | 33.278 | 0.95386 |
| 16 | 3 | rotating | 35.707 | 34.595 | 0.96887 |
| 24 | 0 | warm | 49.194 | 35.250 | 0.71656 |
| 24 | 0 | rotating | 48.578 | 35.725 | 0.73542 |
| 24 | 3 | warm | 49.391 | 35.081 | 0.71028 |
| 24 | 3 | rotating | 48.604 | 35.756 | 0.73564 |
| 32 | 0 | warm | 64.129 | 54.524 | 0.85023 |
| 32 | 0 | rotating | 62.349 | 57.318 | 0.91930 |
| 32 | 3 | warm | 64.093 | 54.705 | 0.85352 |
| 32 | 3 | rotating | 62.382 | 57.371 | 0.91968 |
| 64 | 0 | warm | 117.876 | 98.781 | 0.83800 |
| 64 | 0 | rotating | 116.359 | 98.039 | 0.84255 |
| 64 | 3 | warm | 117.865 | 98.851 | 0.83868 |
| 64 | 3 | rotating | 116.602 | 98.206 | 0.84223 |

The timing worker used 84.29 seconds. Across preparation, diagnostics,
qualification and timing, the observed Modal compute delta was **$1.73** of
the $20 budget. No workers remain. The tuned specialization is selected in
the microbenchmark. Its serving dispatch and full-model performance are
unverified. The earlier Packed path remains available through the existing
GLM attention suite and the starting pin.

## Pristine upstream TP4 decode comparison

Question: what is production Packed Direct's complete attention-call overhead
against the unchanged upstream v0.5.22 FP8 TileLang entry point? Protocol
`remnant-glm-upstream-decode-v1` uses one H200, 16 local heads, batches
1/8/16/24/32/64, tails 0/3 and warm/rotating buffers (24 cases). It reuses the
existing fixtures and paired timing loop: seed 20261024, 30 balanced paired
rounds, ten warmups and at least 100 calls per sample. Packing, selection,
initialization and compilation are untimed. No weights or serving are involved.

The baseline is the unedited attention module from upstream release commit
`baa9befa381511cd4d56f83980951e5a45fffc6d`, imported separately from the fork's
Native specialization. Its source and imported SGLang helper modules are
verified before compilation and execution. Both paths share the same toolchain.
The benchmark-only `glm_upstream.py` owns this provenance and CPU preparation;
`glm_attention.py` remains the sole fixture, timing and result owner.

Use the maintained entry point with `--model-family glm53 --suite
glm-upstream-decode --phase decode`; the Modal image selection is
`REMNANT_GLM_UPSTREAM_DECODE=1`. Production kernels and serving defaults stay
unchanged. Acceptance requires all 24 cases, both paths, all 30 paired rounds,
independent numerical checks and graph replay. Retain every performance miss.
This result cannot complete the older 48/128-case protocols or establish
unchanged-upstream full-model serving performance.

The new phase has a $5 hard cap ($1 CPU, $3 H200 workers, $1 reserve); stop
routine work at $4. CPU preparation and fresh-context review precede paid GPU
work. Use one worker at a time, no automatic retries, and preserve partial runs.
Source worktree: `data/source-snapshots/glm53-upstream-decode`, matching parent
and SGLang branch `exp/glm53-flash/upstream-decode`. Record results and costs in
the existing qualification ledger, separately from all prior authorizations.

### Completed upstream decode result

Run `glm53-upstream-decode-20261106T065615Z` passed all **24 cases**, both paths
and all **30 paired rounds** on one H200. Measured sources: parent `5acc209`,
SGLang `99499fb02c86`, unchanged FlashMLA `4e72722671d8`; prepared image
`im-E77qOCLcBadvS16nBehz0W`.

**Batch-weighted overhead: +1.28947%.** Seven cases favor Direct; the range is
−1.97922% to +6.12250%, with no 2× outliers. These are descriptive complete
attention-call results, not distributed execution or full-model serving.
Packing and selection remain untimed. Native uses all 512 latent coordinates;
Direct uses the production 320-byte Packed records.

Every fixture bank passed exact Packed bytes, independent numerical checks and
changing-input graph replay. Numerical tolerances remain `atol=rtol=0.02`.
Matching focused H100 sanitizer evidence was reused by runtime/cubin/toolchain
identity; no fresh H200 sanitizer or broad qualification is claimed.

The GPU Function took **105.934 seconds** including fixture validation and
measurement; the complete launch took **278.604 seconds** including image
resolution and CPU receipt preflight. All workers stopped. This phase used
**$0.45247 reported compute / $0.78581 conservatively accounted**, including the
failed CPU attempt, beneath the new $5 cap with its $1 reserve intact.

The [flat CSV](upstream-decode.csv) retains all raw samples, statistics and
fixture identities. The canonical ledger's `upstream_decode` entry records
commands, source/artifact identities, per-case results, failures and costs. Raw
JSON, manifests and logs remain in `winstoncai233/remnant-stage2a-results` under
`microbench/glm53-upstream-decode-20261106T065615Z`. Reconcile the exact saved
statistics with Python 3.12, matching the image; Python 3.11's float summation
can differ in the final bits. No comparison was relaxed.

Times below are means per complete call. Changes are means of paired ratios.

| Batch | Tails | Cache | Upstream µs | Direct µs | Paired change |
|---:|---:|---|---:|---:|---:|
| 1 | 0 | warm | 20.355 | 21.021 | +3.27% |
| 1 | 0 | rotating | 20.181 | 20.125 | -0.28% |
| 1 | 3 | warm | 20.263 | 20.966 | +3.47% |
| 1 | 3 | rotating | 19.869 | 19.948 | +0.39% |
| 8 | 0 | warm | 23.139 | 24.555 | +6.12% |
| 8 | 0 | rotating | 22.645 | 23.635 | +4.37% |
| 8 | 3 | warm | 23.143 | 24.351 | +5.22% |
| 8 | 3 | rotating | 22.773 | 23.859 | +4.77% |
| 16 | 0 | warm | 34.846 | 36.114 | +3.64% |
| 16 | 0 | rotating | 35.702 | 36.238 | +1.50% |
| 16 | 3 | warm | 34.943 | 35.988 | +2.99% |
| 16 | 3 | rotating | 35.682 | 36.210 | +1.48% |
| 24 | 0 | warm | 49.211 | 48.236 | -1.98% |
| 24 | 0 | rotating | 48.547 | 48.678 | +0.27% |
| 24 | 3 | warm | 48.624 | 48.422 | -0.41% |
| 24 | 3 | rotating | 48.630 | 48.741 | +0.23% |
| 32 | 0 | warm | 64.246 | 63.118 | -1.76% |
| 32 | 0 | rotating | 62.392 | 62.229 | -0.26% |
| 32 | 3 | warm | 64.166 | 62.958 | -1.88% |
| 32 | 3 | rotating | 62.415 | 62.303 | -0.18% |
| 64 | 0 | warm | 118.074 | 120.646 | +2.18% |
| 64 | 0 | rotating | 116.365 | 119.097 | +2.35% |
| 64 | 3 | warm | 117.921 | 120.871 | +2.50% |
| 64 | 3 | rotating | 116.609 | 119.334 | +2.34% |

Reproduction uses the maintained preparation and benchmark dispatchers, with
fresh source, artifact, spend and allocation checks before any paid launch:

```sh
export REMNANT_INTEGRATION=v0.5.22 REMNANT_MODEL_FAMILY=glm53
export REMNANT_GLM_UPSTREAM_DECODE=1
export REMNANT_COMPILED_IMAGE=<verified-matching-base-image>
modal run scripts/modal/kernels.py::prepare_kernels --model-family glm53
modal run scripts/modal/kernels.py::benchmark_kernels \
  --model-family glm53 --suite glm-upstream-decode --phase decode \
  --qualification-runs <matching-source-bound-receipt-id> \
  --run-id <unique-run-id> --max-seconds 1140
```


Author: Winston Cai.

The cooperative Packed loader completed all **48 decode cases**, with 30 paired
rounds each: **1.6010% batch-weighted overhead** over fresh Native controls.
The result is bound to its measured source below. Native remains the serving
default. [qualification.json](qualification.json) is the compact decision and
budget ledger; [decode.csv](decode.csv) contains every case and raw sample.

## Decision and evidence

| Scope | Outcome | Source or record |
|---|---|---|
| Optimized decode, 48 cases | Complete; +0.6829% for 16 heads, +2.5192% for eight | SGLang `246ee03c1ad5`, parent `8aea70ae58a9` |
| Latest focused H100 review | Direct numerical/graph and Adapter >2 GiB numerical checks passed, with all three sanitizer tools | SGLang `9185c6d25115`; run IDs in ledger |
| Current combined GLM/V4.1 candidate | CPU checks passed; GLM 175 numerical/integration tests and nine smoke tests passed on H100 | Runtime freeze: parent `c23b32f`, SGLang `e0d6d50`; unchanged GLM content qualified on `250415b`/`80d6a47` |
| Earlier unoptimized attention, 128 cases | Incomplete: 56 cases; memcheck incomplete | [Historical attention CSV](attention.csv) |
| Historical full-checkpoint serving smoke | Native and Direct passed on four H200s | Run `glm53-h200-tp4-final-answer-20261104`; source-bound record below |
| Combined-candidate focused instrumentation | GLM passes all three tools; V4.1 memcheck/synccheck pass, full focused racecheck fails | Ledger `ds41_integration`; isolated Native diagnostic passes but does not qualify the failed matrix |
| Combined-candidate full-model serving | GLM and V4.1 Native/Direct pass; V4.1 modes ran concurrently after the memory correction | [Serving attempt summary](serving-smoke.csv); earlier failed attempts retained |

Ten optimized decode cases favor Direct; case means range from -3.5576% to
+8.6594%. No case reaches 2×. The timing Function took 136.084 seconds including
its graph preflight; controller/image resolution/cleanup took 246.936 seconds.
These are one-H100 synthetic complete attention calls, not multi-GPU serving.

The historical CPU inspection covers all 32 SM90 variants. Direct has 34 POPC
sites, vector payload loads/stores, no spills or stack/local traffic, at most
34,944 bytes dynamic shared memory, and the same two-CTA theoretical residency
as Native. Native source and cubins were unchanged at that measured candidate.
See [resource usage](resource-usage.csv) and [SASS summary](sass-summary.csv).

Focused Direct instrumentation covers the byte probe and one production partial
CTA per specialization, with complete numerical/graph checks separately. Adapter
reconstruction across 2 GiB was instrumented without kernel filtering. Broad
full-grid/combine instrumentation remains incomplete. Later changes never
relabel those source-bound results as fresh qualification.

## Implementation and ownership

The maintained branch is `glm53-flash-packed` in the parent and SGLang fork.
The combined candidate uses V4.1 FlashMLA `4e72722671d8`; both v0.5.18 pins
remain unchanged. GLM runtime and kernels remain in the v0.5.22 SGLang fork. See [the fork file map](../../../third_party/README.md).

| Responsibility | Owner |
|---|---|
| Geometry and startup contract | `sglang/srt/remnant/glm_config.py` |
| Selection, normalization, store and Adapter | `sglang/srt/remnant/glm.py` |
| Persistent planes and lifecycle | `sglang/srt/remnant/glm_pool.py` |
| Direct TileLang specialization | `sglang/kernels/ops/attention/dsa/tilelang_kernel.py` |
| Cooperative byte loader | `sglang/kernels/jit/include/sgl_kernel/glm_packed_decode.cuh` |
| Fixtures, measurement and resume | `sglang/benchmark/remnant/glm_attention.py` |
| CPU compilation and artifact identity | `sglang/benchmark/remnant/glm_artifacts.py` |
| Preparation, execution and export | `scripts/modal/kernels.py` |
| Qualification acceptance | `scripts/modal/kernel_qualification.py` |

SGLang paths above are relative to `third_party/sglang-v0.5.22/python/`.
Eleven MLA layers store 256 raw FP8 bytes and eight uint64 bitmap words per
physical row (320 bytes). Selection uses raw magnitudes before Native RMSNorm;
rounding/conversion and all 34 KDA layers, indexer storage, tails, and allocator
geometry remain unchanged. Direct expands each latent tile once and reuses it
for QK and PV. Adapter reconstructs into a shared workspace in 128-query chunks.

Packed requires resolved BF16 model activations and the supported FP8 TileLang
configuration. Native remains unrestricted by the Packed activation check.
The resolved dtype, including `auto`, is validated before pool allocation.

## Measurement protocol

`--model-family glm53 --suite glm-attention --phase decode` selects
`remnant-glm-decode-v1`: heads 16/8 × batches 1/8/16/24/32/64 × tails 0/3 ×
warm/rotating buffers. Default `--phase all` requires the separate 128-case
`remnant-glm-attention-v1` matrix. A decode run never completes that experiment.

Each request has a separate 128K history. Select 512 distinct four-token groups,
expand to 2,048 rows, append live tails and pad to 2,112 columns. Both modes use
identical queries, indices and bank counts. Seed is 20261024. Graph measurements
include query conversion, partial attention and combine, with ten warmups,
30 balanced paired rounds and at least 100 calls per sample. Producer/store,
selection and initialization are untimed.

Warm reuses one buffer bank; it does not promise L2 residency. Rotation uses
disjoint banks whose selected Packed footprint exceeds twice measured L2, while
keeping fixtures below 70% of VRAM. Report all cases. The headline is the mean
of per-round Direct/Native percentage overhead weighted by batch size; it is
not a workload-weighted serving speedup.

Source identity includes the reference, normalization, geometry, fixture and
statistics dependencies. Resume additionally requires a complete verified build
manifest; missing, changed or stale binaries are rejected. Aggregate statistics
are checked against raw samples. Historical records retain their original schema
and identities and are not eligible to resume across changed code.

## Checks and reproduction

Reproduce the local CPU checks with these existing entry points:

```bash
python -m unittest scripts.modal.tests
cd third_party/sglang-v0.5.22
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/test_glm5_mla_cache_format.py \
  benchmark/remnant/test_microbench.py
```

GPU preparation and execution use the existing `prepare_kernels`,
`validate_kernels`, and `benchmark_kernels` functions in
`scripts/modal/kernels.py`. Select `REMNANT_INTEGRATION=v0.5.22` and
`REMNANT_MODEL_FAMILY=glm53`. `REMNANT_COMPILED_IMAGE` and
`REMNANT_GLM_CPU_REBUILD_FROM` are mutually exclusive. Ordinary reuse requires
exact clean pins and matching source/artifacts. Changed compiled-source inputs
require the approved CPU rebuild/verification path before allocating an H100.
Do not reuse the historical image solely because its ID appears in this record.

Focused qualification uses `--stage corrective --sanitizer --sanitizer-scope
glm-decode`. `--sanitizer-tools` can select individual tools; single-tool success
never counts as all-tool qualification. Timing requires matching
`--qualification-runs` and the requested scope. Exact historical commands and
artifact identities remain in the archive below. New runs use unique run IDs
and preserve partial results; no automatic retry follows a timeout.

The expanded production integration test now invokes the actual model producer,
hybrid pool and public prefill/decode dispatch for Adapter and Direct. It checks
nonidentity physical row placement, second-layer mapping, current-token writes,
duplicates and invalids against independently reconstructed Packed input. Index
selection and page-table transforms are outside this test. This expanded GPU
test passed in the combined candidate's 175-test H100 run. Broad qualification,
model quality, serving capacity and
H20 execution remain unverified. See the newer prefill results below and the
[full-checkpoint H200 smoke](../../../third_party/README.md#full-model-h200-smoke-test).

## H200 small-prefill comparison

`remnant-glm-prefill-small-v1` pairs stock Native FP8 with Packed Direct on
one H200. It covers 16 local heads, 128 and 512 queries, shared and independent
history selections, warm buffers, and an initial 128K prefix. Both modes time
the complete attention call. The 64-row candidate expands two 32-row waves
into one KV tile and uses two partial groups. Results record each path's
geometry because Native may choose two groups at Q128.

After CPU preparation, select the verified image and run from the parent root:

```sh
REMNANT_INTEGRATION=v0.5.22 REMNANT_MODEL_FAMILY=glm53 \
REMNANT_GLM_PREFILL_H200=1 REMNANT_COMPILED_IMAGE=<verified-image-id> \
modal run scripts/modal/kernels.py::benchmark_kernels \
  --model-family glm53 --suite glm-attention --phase prefill-small \
  --packed-prefill-tile 64 --run-id <unique-run-id> --max-seconds 1100
```

The runner qualifies one CTA of the selected Packed cubin under memcheck,
racecheck, and synccheck. It then validates full fixtures and times two fresh
captures in separate processes. Each capture uses 30 paired rounds of at least
100 calls per sample. All four per-case median ratios must be at most 1.00 in
both captures. Retain misses alongside the raw samples. The CTA probe covers
the changed partial kernel. Full-grid sanitizer coverage is separate.

This phase has its own aggregate $15 Modal ceiling. The optimized prefill
candidate remains experimental until numerical and paired timing gates pass.

## Full paired H200 prefill comparison

The `remnant-glm-prefill-v1` experiment asks whether the qualified 64-row
Packed loader holds up against stock Native FP8 across the full prefill matrix.
It measures 80 cases on one H200 with local heads 16 and 8, query chunks 128,
512, 1,536, 4,096 and 8,192, tail phases 0 and 3, shared and independent
history, and warm and rotating buffers. Every case gets 30 fresh paired rounds
using the established graph protocol. The decision is per-case parity, with all
slowdowns retained. The run stops at the $5 phase ceiling.

CPU-prepare the missing H8 64-row artifact, then run the maintained command
above with `--phase prefill --packed-prefill-tile 64 --max-seconds 2340` and a
new run ID. Both head variants must pass focused sanitizer checks. A timeout
can resume from the last valid case with `--resume-run <prior-run-id>` and
the same source, artifact and focused sanitizer evidence. The optimized tile
is selected explicitly by the benchmark. Full-model serving uses its production
dispatch.

The full run completed all 80 cases after a same-image resume of the final
two cases. Packed Direct was faster in 70 cases. Ten missed parity, with a
maximum slowdown of 1.11%. Both head variants passed focused memcheck,
racecheck and synccheck. The saved JSON and CSV contain every paired sample.
The phase used $4.01 gross metered on Modal, with $4.02 conservatively
accounted and all workers stopped. Keep the wide tile as a benchmark candidate.
See the `full_prefill` entry in the qualification record for artifact hashes,
commands, exact cases and costs.

## Budget and shutdown

Total authorization is **$40**: $20 historical and the extended $20 optimization
phase. Conservative accrued spend is **$31.440634**, including **$12.029805** in
the optimization phase. Its routine stop is $17 with a $3 reserve, and the H100
limit is 135 minutes (133.171 minutes accounted). The complete category limits,
authorization changes and aggregate H100 accounting remain in the ledger.
No image build, GPU allocation or additional archive upload ran for this review.
All last-checked Modal containers were stopped. Before any future launch, refresh
spend, rate, allocation, artifact, source and remaining-time checks.

## Historical evidence and recovery

The former ledger contained 57,446 lines and duplicated complete timing results,
compiled manifests and attempt records. The current ledger keeps decisions,
budgets, source-bound summaries and references. No historical evidence was lost:

- [Complete original ledger](https://github.com/winstonxcai/flash-optimizations/blob/ee509d1fbbdd823f527388d2ee763d370d65ad0a/benchmarks/microbench/glm53-packed/qualification.json)
- [Complete original narrative](https://github.com/winstonxcai/flash-optimizations/blob/ee509d1fbbdd823f527388d2ee763d370d65ad0a/benchmarks/microbench/glm53-packed/README.md)
- Verified local extracted archive: `data/glm53-packed/ledger-archive-20261026T165850Z/`
- Original remote raw runs: `caiw/remnant-stage2a-results/microbench/<run-id>/`

The local archive has 67 hash-verified files, including 52 individual attempt
records. Original bytes match the immutable Git commit above. It was not uploaded
as a new remote copy. The ledger records hashes and a `git show` recovery command;
clones can recover the complete record even without the local extracted archive.
The existing CSVs and pinned [model metadata](model-config.json) remain unchanged.

Rollback: select Native, or restore the pre-optimization SGLang pin `889edf56`.


## GLM / V4.1 merge validation

The isolated integration candidate merges both PR histories and exposes
`--kv-cache-format native|direct` on v0.5.22. The untouched v0.5.18 pair keeps its
recorded interface; shared launchers translate Native/Direct explicitly.
`qualification.json` → `ds41_integration` owns this phase's **$26** budget,
including the approved $5 and $1 increases, validation status, source identities and receipts. Historical runs and
the original worktree's unstaged ledger are preserved. The 236 unrelated replay
capture additions and replay-only sweep are excluded from the resulting PR diff.

Both integration/suite CPU regressions and independent reviews passed. GLM
numerical, allocation, graph and focused three-tool checks passed on H100. The
V4.1 allocation fixture correction is test-only and now checks the production
selector/sizing on CPU before allocation; its 39 GPU numerical/allocation cases
pass. The subsequent parallel-serving request reordered the earlier sequence:
serving checks ran while the remaining V4.1 tools were pending. Final acceptance
still requires all gates. No new timing or prefill promotion is part of this
merge; Native remains the default.

The merge-only GLM sanitizer group uses `--stage corrective --sanitizer
--sanitizer-scope glm-merge` with the exact `GLM_MERGE_SANITIZER` cases from the
maintained runner. It instruments the byte loader, four decode partial variants
and both 64-row prefill head variants, retaining the Packed filter/source audit.
It never replaces broad timing-qualification requirements. V4.1 corrective cases
are allowlisted; default full 34-case qualification is unchanged.


### Parallel serving follow-up

The requested serving scope is now four independent workers: GLM Native/Direct
on `winstoncai233`, and V4.1-Flash-1017 Native/Direct beside its cached checkpoint
on `winstoncai`. Each worker owns four H200s, one server process group and a
mode-specific result directory. The controller must own exactly two submissions
per account; `max_containers=2` alone does not bound separate ephemeral apps.

Both models use TP4/EP4, an 8K context, four requests, 16,384 token slots,
decode graphs at 1/2/4, offline weights and the Python radix tree. GLM keeps its
2,048-token prefill chunk. V4.1 uses page256/chunk256, explicit FlashMLA,
FlashInfer MXFP4 and V4.1 parsers; disable FlashInfer autotuning and eager
DeepGEMM precompilation for this functional smoke. MHC arithmetic is preserved.
No model-quality or serving-performance claim follows from this short request.

Use the existing `scripts/modal/serve.py::prepare_serving_smoke` CPU entry point,
then `::smoke_serving --run-id <unique-pair-id> --mode native|direct`, with
`REMNANT_INTEGRATION=v0.5.22`, the selected `REMNANT_MODEL_FAMILY=glm53|dsv41`
and the CPU-verified `REMNANT_COMPILED_IMAGE`. The public dispatcher validates
inputs and weights before scheduling. Parallel controllers use the same worker,
with bounded ownership, deadlines and cancellation; no alternate server harness.

The approved phase cap is $26, with a $23 routine stop and $3 protected reserve.
All four initial workers ran in parallel after CPU admission and an aggregate
cost check. GLM Native passed in 511.78 seconds. GLM Direct loaded and captured
graphs, but its cold chat timed out; a lone rerun passed in 525.48 seconds.
Both successful responses were exactly `1, 2, 3, 4, 5`, with finish reason `stop`.
Native used parent `a56cb7b`; the Direct rerun used `c23b32f`. Both used SGLang
`e0d6d50`, identical GLM settings and the same verified compiled base. These
startup intervals are functional-smoke evidence, not a performance comparison.

Both V4.1 modes loaded all 48 shards, occupying 119.12 GiB per rank, then rejected
the inherited 0.86 memory fraction before KV allocation. Their observed minimum
was 0.864. The committed profile uses 0.90 for V4.1 and retains 0.86 for GLM,
with the 16,384-token bound unchanged. The four-mode environment regression and
independent review pass; V4.1 allocation, graph capture and chat after this fix
remain unverified. [serving-smoke.csv](serving-smoke.csv) is the flat summary of
all five attempts; the JSON ledger owns their source, artifact and cost records.

Focused `v41-merge` instrumentation retains all 15 selected numerical fixtures
and all three tools, while filtering to CPU-verified FP4 store/read, FlashMLA
decode/prefill and combine symbols. A positive Packed synchronization audit is
required. It excludes framework/reference kernels and never satisfies the
unchanged full 34-case timing qualification. The interrupted unfiltered attempt
remains in the aggregate ledger with its entire allocation charged.

The first narrow decode synchronization audit failed with an internal sanitizer
hardware exception surfacing during Adapter prefill. Its 15 uninstrumented
fixtures passed; no sanitizer tool qualified. The audit now uses the same full
production-kernel filter as the three tool runs. This is a coverage correction,
not proof that the exception was a checker defect. Preserve the failed run.
Kernel images explicitly install the submitting Modal SDK, matching the serving
runner, so older image builders supply its dependencies before CPU tests.

The corrected full-pattern audit passes. All 15 focused V4.1 cases pass memcheck
and synccheck. Full focused racecheck fails: ten cases pass, including both
graph/stream cases, then Native ratio-1 compressor setup/projection reports an
illegal address before its FP4 store callback. Four later failures are sticky
context cascades. The checker reports an internal hardware exception and XID
13/43; these do not establish a hardware or checker defect. An isolated rerun of
the first failing case passes unfiltered racecheck on a different H100 with
`CUDA_LAUNCH_BLOCKING=1`. That changed context, filter and timing; it does not
resolve or qualify the failed batch. No numerical kernel fix is justified yet.

At the initial merge-phase closeout, all allocations were released. Compute was $20.50 reported / $21.57
conservatively accounted, with the $3 reserve intact and 70.67 H100 minutes
accounted under the 75-minute cap. No further jobs are scheduled. Remaining
gates are the full V4.1 racecheck failure and both corrected V4.1 serving smokes.
The two PRs remain unmerged and validation remains incomplete. Next use the
maintained `serve.py::smoke_serving` entry point above after new authorization
and fresh preflight; diagnose the race failure with existing fixtures first.


### Five-case racecheck follow-up

Run `v41-failed-five-20261106` selects only the four failed C1/C2 compressor
cases and `test_fp4_fragment_masks[64]`. All five numerical checks passed in
33.71 seconds; all five racecheck cases passed in **143.65 seconds**, with zero
hazards, errors or warnings. This used a fresh H100, unfiltered framework
instrumentation and `CUDA_LAUNCH_BLOCKING=1`, matching the isolated diagnostic.
It does not resolve the earlier 15-case batch failure.

The local controller hit a `Connection._transport` collection error after the
worker completed. Persisted worker JSON, exact XML case identities, log hashes
and the zero-hazard summary were verified independently; no GPU rerun was needed.
Sources and input receipts are archived with the run. No kernel code changed.

The approved extra $1 and ten H100 minutes raise the phase caps to $26 and
85 minutes, retaining the $3 reserve. Updated totals are **$20.76 reported /
$22.11 conservatively accounted** and **75.96 H100 minutes**. All workers are
stopped. The original batch failure and corrected V4.1 serving gates remain
pending.

### Parallel V4.1 full-model serving

The separately authorized $15 phase is recorded under `v41_parallel_serving` in
the same JSON ledger. Run `v41-native-direct-serving-20261106` passed both modes
concurrently on `winstoncai`, using image `im-SXCseSK7MhV0kyLb7TPdbJ`, parent
`f720d20`, SGLang `e0d6d50` and FlashMLA `4e72722671d8`.

| Mode | Start to healthy | Complete smoke, including shutdown | Result |
|---|---:|---:|---|
| Native | 534.59 s | 564.47 s | Passed |
| Direct | 497.56 s | 526.50 s | Passed |

Execution overlapped for 526.50 seconds. Each worker owned four H200s with
TP4/EP4, activation dtype `auto`, FP8 KV, FlashMLA, MXFP4 experts, an 8K context, four
requests, chunk256 and 16,384 token slots. Both servers reported memory fraction
0.90. All four ranks completed loading the pinned 48-shard checkpoint and
capturing decode graphs at batches 1/2/4. Health and chat returned HTTP 200;
both answers were exactly `1, 2, 3, 4, 5`, with finish reason `stop`, 22 prompt
tokens and 14 completion tokens. Owned server/rank processes and GPU workers
exited; final account allocation checks were empty.

This phase used $6.24 reported compute / $6.68 conservatively accounted, within
its separate $15 cap and with the $3 reserve intact. Prior phase costs and
failed smokes remain preserved. These are functional smoke durations, not an
attention or serving performance comparison. The earlier full-batch racecheck
failure remains unresolved; no new timing or quality qualification is implied.
