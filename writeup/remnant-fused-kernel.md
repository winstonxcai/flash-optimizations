# Remnant Fused Reconstruction

## Goal

- Preserve the 328-byte packed record and native 584-byte FlashMLA input layout.
- Reduce packed reconstruction overhead while preserving masking, scales, RoPE,
  and the existing attention consumer.
- Measure the actual decode path, not a dense-BF16 diagnostic path.

Hardware for the corrected measurements: one NVIDIA H100 80 GB HBM3, SM90.
The benchmark uses 128k context, `K=512`, and batches 15, 18, and 21.

## Iteration summary

- **Iteration 1:**
  - Staged each packed row's bitmap once.
  - Used aligned 32-bit NoPE stores.
  - Copied the seven native scale bytes with one 64-bit store.
- **Iteration 2:**
  - Cooperatively staged all 256 packed value bytes.
  - Computed the eight bitmap prefixes once per row.
- **Iteration 3:**
  - Assigned each active lane four groups of four coordinates.
  - Used one rank calculation and one 32-bit output store per group.
- **Iteration 4:**
  - Compared output ownership, warp-register prefixes, and RoPE load timing.
  - The earlier RoPE schedule was the best variant.
- **Iteration 5:**
  - Compared byte permutation, parallel-prefix, and earlier-RoPE variants.
  - Only the validated v5C body was retained and renamed `optimized`.

The ordinary `fused` kernel remains the control. The geometry-specialized body
is now promoted behind the normal `optimized` serving dispatch for the active
`K=512`, page-size-16 shape. The explicit `candidate="geometry"` entry remains
available for regression comparison; `early_rope` is benchmark-only.

## Benchmark correction

The earlier reported attention numbers were fixed-selection measurements that
included dense BF16 readback. They do not represent production decode.

The corrected harness:

- Calls `sgl_kernel.flash_mla.flash_mla_with_kvcache` directly.
- Uses the native paged workspace expected by the consumer, with no dense
  readback between reconstruction and attention.
- Uses 16 real V4-Flash query heads per TP4 rank and only ABI-required padding.
- Allocates 32 shuffled selection sets before timing and cycles them between
  graph replays; input copies are outside device events.
- Used one process per mode after a same-process multi-mode run failed. The
  failure was previously attributed to FlashMLA without proof. Inspection found
  that the harness did not retain pre-capture workspace owners across modes;
  owner retention is now explicit. The corrected independent-context consumer
  matrix is reported below.

The completed geometry experiment used this single matrix command:

```text
MODAL_PROFILE=fxcai21 modal run --timestamps mustafar/scripts/modal/app.py::bench_kernels \
  --suite decode --decode-samples 500 --decode-rounds 3 \
  --decode-selection-sets 32 \
  --decode-modes native,packed,fused,optimized,geometry \
  --decode-batches 15,18,21
```

Runtime flags were pinned by the launcher: TopMag50 enabled, `KEEP=0.5`, and
packed storage enabled. The run emitted the native, optimized, and geometry
dispatch markers, and all five modes used the same consumer and graph timing
path. The optimized path emitted
`MUSTAFAR_FUSED_DISPATCH=packed_to_native_optimized`; the control emitted
`MUSTAFAR_FUSED_DISPATCH=packed_to_native`; geometry emitted
`MUSTAFAR_FUSED_DISPATCH=geometry`.

## Historical direct-consumer results (shared pool)

These measurements removed dense readback, but all requests selected from one
32,768-row pool and physical indices equaled raw positions. They did **not**
model independent 128k contexts with shuffled physical pages. The ratios below
are historical observations, not a validated production gap or promotion gate.

Graph replay latency in microseconds, averaged over three rounds. Each cell is
`mean / p50 / p95`; it includes reconstruction plus the direct FlashMLA call and
its combine work.

| Batch | Native | Packed Triton | Fused | Optimized | Early RoPE |
|---:|---:|---:|---:|---:|---:|
| 15 | 21.758 / 21.739 / 22.027 | 58.057 / 57.867 / 59.125 | 39.058 / 39.029 / 39.648 | 30.163 / 30.165 / 30.624 | 29.329 / 29.280 / 29.685 |
| 18 | 24.229 / 24.192 / 24.544 | 66.929 / 66.827 / 67.744 | 43.434 / 43.413 / 44.021 | 32.707 / 32.683 / 33.515 | 31.990 / 32.000 / 32.608 |
| 21 | 24.916 / 24.875 / 25.301 | 75.035 / 74.885 / 75.925 | 47.110 / 47.072 / 47.733 | 34.546 / 34.528 / 35.339 | 33.850 / 33.856 / 34.539 |

Relative mean results:

- `optimized` is 22.8%, 24.7%, and 26.7% faster than `fused` at B15, B18,
  and B21.
- `optimized` remains 38.6%, 35.0%, and 38.7% slower than native at those
  batches.
- `early_rope` recovers a further 2.8%, 2.2%, and 2.0% versus `optimized`.
  This historical result was below the 5% B21 target; it was not promoted.
- Packed Triton is 2.67–3.01× slower than native in this complete-path test.

The native-materialized diagnostic measured direct FlashMLA on a preconstructed
workspace at B15/B18/B21: `21.130 / 23.125 / 23.617 µs` mean. It is a diagnostic,
not a production baseline. It indicates that the optimized adapter's remaining
gap at B21 is about 10.9 µs relative to the best downstream floor, while the
profiled reconstruction kernel itself is about 9.5 µs per launch.

## Corrected reconstruction results

The corrected reconstruction-only harness used independent 128k logical
contexts, shuffled physical pages, 32 changing selection sets, and K=512. It
measured graph replay of reconstruction only; it did not include FlashMLA
attention. Each cell is mean / p50 / p95 in microseconds, averaged over three
rounds of 500 samples.

| Batch | Fused | Optimized | Early RoPE |
|---:|---:|---:|---:|
| 15 | 23.327 / 23.205 / 23.733 | 12.862 / 12.752 / 13.536 | 12.812 / 12.725 / 13.355 |
| 18 | 25.358 / 25.301 / 25.749 | 13.620 / 13.557 / 13.973 | 13.555 / 13.429 / 13.920 |
| 21 | 28.397 / 28.352 / 28.832 | 14.790 / 14.731 / 15.157 | 14.349 / 14.283 / 14.667 |

Relative to `optimized`, earlier RoPE loads change mean latency by -0.39% at
B15, -0.48% at B18, and -2.98% at B21. The B21 improvement is real but far
below what is needed to close the native gap.

## Iteration 6: geometry specialization

The optimized kernel still accepted `selected_k` and `page_size` at runtime.
For the fixed serving geometry (`K=512`, page size 16), the `geometry`
candidate replaces those address calculations with shifts and masks:

The experiment used the uncommitted source tree at `af40cc5` on
`codex/remnant-sparse-kernel`, SGLang revision `71de97b`, and one H100. No model
weights were loaded.

- `row >> 9` / `row & 511` for query and selection slot.
- `row >> 4` / `row & 15` for output page and page offset.
- Runtime byte stride and 64-bit addresses remain unchanged.
- The optimized body, shared-memory layout, four-warps-per-block geometry,
  NoPE expansion, RoPE schedule, and invalid-row behavior are reused unchanged.

The candidate removed the targeted reciprocal instructions without changing
resource usage:

| Variant | Kernel p50 NCU | Dynamic instructions | Static instructions | MUFU.RCP | Registers/thread | Shared bytes |
|---|---:|---:|---:|---:|---:|---:|
| Optimized | 11.04 µs | 6.677M | 760 | 3 | 32 | 2,496 |
| Geometry | 10.52 µs | 6.086M | 600 | 0 | 32 | 2,496 |

The direct-consumer test used `flash_mla_with_kvcache`, 128k context, 32
changing selection sets, and three rounds of 500 graph replays. Results are
mean / p50 / p95 in microseconds:

| Batch | Native | Packed | Fused | Optimized | Geometry |
|---:|---:|---:|---:|---:|---:|
| 15 | 21.706 / 21.653 / 22.091 | 59.018 / 58.981 / 59.733 | 40.497 / 40.331 / 41.397 | 29.505 / 29.435 / 29.973 | 29.175 / 29.099 / 29.749 |
| 18 | 24.919 / 24.843 / 25.451 | 68.026 / 67.925 / 68.971 | 43.849 / 43.808 / 44.619 | 32.044 / 32.000 / 32.811 | 31.146 / 31.104 / 31.936 |
| 21 | 25.366 / 25.184 / 26.421 | 75.662 / 75.509 / 76.832 | 47.840 / 47.627 / 49.205 | 34.844 / 33.931 / 35.915 | 33.634 / 33.419 / 34.869 |

Geometry improves complete decode mean over optimized by 1.12% at B15,
2.80% at B18, and 3.47% at B21. Reconstruction-only mean improvement was
3.2%, 4.7%, and 4.3%, respectively. The complete-path B21 result is below
the original 5% threshold, but that threshold was too high for this isolated
address-specialization experiment. Using the revised rule of a repeatable 2%
B21 gain with no material lower-batch regression, geometry is promoted as the
optimized serving path. Native parity has not been reached.

The geometry validity fixtures passed, including int32/int64 indices, empty
batch, unsupported-geometry rejection, changing graph inputs, and invalid
locations. The narrowed geometry-only memcheck, racecheck, and synccheck run
also passed. The earlier aggregate racecheck timeout was an orchestration
timeout, not a reported kernel error. The complete JSON is in
`writeup/remnant-geometry-results.json`.

## Combined candidate: geometry plus early RoPE loads

This experiment combined the two best isolated changes without changing the
packed format or numerical contract:

- `combined` reuses the optimized kernel body with
  `kFixedSelectedK=512`, `kFixedPageSize=16`, and `kEarlyRopeLoads=true`.
- Geometry uses shifts and masks for the fixed K/page dimensions.
- Early RoPE loads move the valid row's frequency and scale operands ahead of
  packed-value staging and NoPE expansion.
- Four warps per block, shared-memory layout, output ownership, masking, and
  RoPE arithmetic are unchanged.

The H100 validity funnel passed the changing-index, partial/empty, invalid,
duplicate, page-boundary, int32/int64, graph, stream, and allocation checks.
Focused memcheck, racecheck, and synccheck all passed. The run used the same
independent-context fixture as the corrected geometry experiment.

Reconstruction-only graph replay latency, averaged over three rounds of 500
samples, in microseconds:

| Batch | Generic | Optimized (geometry) | Combined | Combined vs optimized |
|---:|---:|---:|---:|---:|
| 15 | 12.776 | 11.846 | 11.647 | 1.68% faster |
| 18 | 13.722 | 13.183 | 13.081 | 0.78% faster |
| 21 | 14.573 | 13.805 | 13.937 | 0.95% slower |

The complete consumer test used
`sgl_kernel.flash_mla.flash_mla_with_kvcache`, independent 128k contexts,
32 shuffled selection sets, and changing CUDA-graph replay. Each value is
mean / p50 / p95 in microseconds over three rounds of 500 samples:

| Batch | Native | Generic | Optimized (geometry) | Combined |
|---:|---:|---:|---:|---:|
| 15 | 22.113 / 21.408 / 23.157 | 29.529 / 29.152 / 30.880 | 28.953 / 28.629 / 30.176 | 28.834 / 28.571 / 29.941 |
| 18 | 24.643 / 24.373 / 25.909 | 31.570 / 31.440 / 32.565 | 31.285 / 31.152 / 32.576 | 31.066 / 30.949 / 32.363 |
| 21 | 25.192 / 24.800 / 26.976 | 34.170 / 33.797 / 36.331 | 33.809 / 33.675 / 35.605 | 33.357 / 33.216 / 34.795 |

Combined is 0.41%, 0.70%, and 1.34% faster than optimized at B15/B18/B21,
respectively. It remains 30.4%, 26.1%, and 32.4% slower than native. The
early-load and geometry gains are therefore not additive enough to justify a
new promoted dispatch; `optimized` remains the serving path, with geometry
promotion intact. This is a useful negative result: address specialization
and earlier operand loads recover only a small part of the remaining gap, so
the next experiment should target expansion/shared-memory work or eliminate
the temporary native workspace.

Command for the complete consumer matrix:

```text
MODAL_PROFILE=fxcai21 modal run --timestamps mustafar/scripts/modal/app.py::bench_kernels \
  --suite decode --decode-samples 500 --decode-rounds 3 \
  --decode-selection-sets 32 \
  --decode-modes native,generic,optimized,combined \
  --decode-batches 15,18,21
```

The compact aggregate is in `writeup/remnant-combined-results.json`. Raw
timing and sanitizer outputs were downloaded locally under
`mustafar/results/reconstruction-20260914-combined/`.

## Profiling findings

The reconstruction target was profiled after warm-up with CUDA graph nodes
visible. Nsight Systems reported 9.348 µs for `optimized` and 9.300 µs for
`early_rope`; Nsight Compute reported 11.008 and 10.944 µs respectively. These
kernel-only profiler timings are not substituted for the unprofiled timing table.

```text
MODAL_PROFILE=fxcai21 modal run --timestamps mustafar/scripts/modal/app.py::profile_reconstruction \
  --phase initial
MODAL_PROFILE=fxcai21 modal run --timestamps mustafar/scripts/modal/app.py::profile_reconstruction \
  --phase profile --profile-mode early_rope
```

The key findings were:

- Both variants use 32 registers/thread, 2,496 bytes of shared memory/block,
  and zero spills. Occupancy is not the limiting resource.
- Earlier RoPE loads improve eligible warps from 4.711 to 4.911 per active
  cycle, increase issue-active from 76.5% to 79.7%, and reduce long-scoreboard
  stalls from 0.631 to 0.497.
- The previously reported “47-way conflicts” was an aggregate metric
  misreading, not a standalone conflict count. Excessive shared-memory
  wavefronts were unchanged at 137,051 in the compared variants; that metric
  does not establish their share of total latency.
- Math-pipe throttle remains the largest scheduler signal (3.077 versus 2.967),
  while DRAM pressure is low. The evidence points to instruction/expansion work
  and shared-memory scheduling, not capacity or raw DRAM bandwidth.

The validity funnel passed exact NoPE/scales, RoPE tolerance, invalid-row
zeroing, changing graph replay, non-default streams, and zero allocations during
replay. A narrowed geometry-only sanitizer run passed memcheck, racecheck, and
synccheck. The earlier aggregate racecheck timeout was an orchestration timeout,
not a reported kernel error.

The complete aggregate artifact is
`writeup/remnant-reconstruction-profile-results.json`. Downloaded profiler
outputs remain outside the report.

## Status

- The reconstruction measurement now uses the independent-context fixture; the
  older complete-path table remains a shared-pool historical result.
- The corrected full-consumer matrix completed on one H100 with native, packed,
  fused, optimized, and geometry modes. Geometry is now the promoted optimized
  serving path under the revised 2% B21 criterion.
- The combined geometry-plus-early-RoPE candidate passed correctness and focused
  sanitizers, but improved the complete-path mean by only 0.41–1.34% over
  optimized; it was not promoted.
- `optimized` is the retained fused implementation and materially improves over
  the original fused adapter.
- `early_rope` remains a schedule diagnostic, not a promoted implementation.
- `geometry` removes the targeted address reciprocals and improves the complete
  path by 3.47% at B21; optimized dispatch now selects it for active serving
  geometry.
- Native parity has not been reached. The next meaningful optimization must reduce
  expansion/shared-memory cost or eliminate the temporary native workspace. The
  evidence supports a focused shared-memory expansion experiment next, with
  unchanged-body warp-count comparison only if that does not move the profile.

Source baseline: `b237987`; implementation commit `b33d2af` on
`codex/remnant-sparse-kernel`; SGLang revision `71de97b`; model-free H100
measurements with no model download or serving run.
