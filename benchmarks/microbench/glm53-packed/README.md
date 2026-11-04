# GLM-5.3-Flash Packed

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
| Current review fixes | CPU checks; expanded integration GPU test not run | Current parent pin; ledger `current_review` |
| Earlier unoptimized attention, 128 cases | Incomplete: 56 cases; memcheck incomplete | [Historical attention CSV](attention.csv) |
| Broad instrumentation / serving | Unverified | No model weights or serving run |

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
Only the v0.5.22 SGLang integration changes; FlashMLA and both v0.5.18 pins remain
unchanged. See [the fork file map](../../../third_party/README.md).

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

The next step is the existing local CPU entry points, requiring no allocation:

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
test has not run. Broad qualification, model quality, serving capacity and
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
