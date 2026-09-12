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

The ordinary `fused` kernel remains the control. The current `optimized` kernel
is the production candidate; `early_rope` is benchmark-only.

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
- Uses one process per mode. A same-process multi-mode run hit FlashMLA graph
  state interference and was discarded; isolated runs are the controlled
  comparison.

Each mode used the same command shape, with one mode substituted:

```text
MODAL_PROFILE=fxcai21 modal run --timestamps mustafar/scripts/modal/app.py::bench_kernels \
  --suite decode --decode-samples 500 --decode-rounds 3 \
  --decode-selection-sets 32 --decode-modes <mode> \
  --decode-batches 15,18,21
```

Runtime flags were pinned by the launcher: TopMag50 enabled, `KEEP=0.5`, packed
storage enabled, and exactly one of native, packed, fused, or optimized dispatch
selected. The optimized path emitted
`MUSTAFAR_FUSED_DISPATCH=packed_to_native_optimized`; the control emitted
`MUSTAFAR_FUSED_DISPATCH=packed_to_native`.

## Corrected decode results

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
  It fails the predefined 5% B21 gate and is not promoted.
- Packed Triton is 2.67–3.01× slower than native in this complete-path test.

The native-materialized diagnostic measured direct FlashMLA on a preconstructed
workspace at B15/B18/B21: `21.130 / 23.125 / 23.617 µs` mean. It is a diagnostic,
not a production baseline. It indicates that the optimized adapter's remaining
gap at B21 is about 10.9 µs relative to the best downstream floor, while the
profiled reconstruction kernel itself is about 9.5 µs per launch.

## Profiling

Command:

```text
MODAL_PROFILE=fxcai21 modal run --timestamps mustafar/scripts/modal/app.py::profile_decode
```

The profile used optimized reconstruction at B21 with one changing-selection
graph sample. Nsight Systems showed the actual sequence:

- optimized reconstruction: 12 launches, 9.52 µs average, 9.31–10.30 µs;
- FlashMLA sparse decode: 12 launches, 17.26 µs average, 15.39–24.61 µs;
- FlashMLA combine: 12 launches, 7.58 µs average, 4.06–8.42 µs.

There was no dense BF16 readback. The attention and combine kernels use separate
streams and partially overlap, so these component times must not be added as an
exact end-to-end total.

Nsight Compute for the optimized reconstruction reported:

- 128 threads/block, 2,688 blocks, 32 registers/thread, and zero spills;
- 5.66 MB global reads and 9.98 MB global writes per launch; DRAM read pressure
  was only 14.9% of peak and write pressure 0.03%;
- shared-memory loads produced about 138k bank conflicts and 137k excessive
  wavefronts;
- 4.12 eligible warps per active cycle, with notable long-scoreboard,
  math-pipe-throttle, and not-selected stalls.

The limiting cost is therefore not DRAM capacity. It is the dependency-heavy
bitmap/rank expansion and shared-memory scheduling, followed by the unavoidable
FlashMLA and combine work. The earlier-RoPE experiment improves scheduling only
slightly; it does not change that dominant structure.

## Status

- The corrected production-style benchmark is valid and reproducible on H100.
- `optimized` is the retained fused implementation and materially improves over
  the original fused adapter.
- `early_rope` is retained only as a measured comparison because it misses the
  5% acceptance gate.
- Native parity has not been reached. The next meaningful optimization must reduce
  expansion/shared-memory cost or eliminate the temporary native workspace; more
  isolated RoPE-load tuning is unlikely to close the remaining gap.

Source baseline: `b237987`; implementation commit `b33d2af` on
`codex/remnant-sparse-kernel`; SGLang revision `71de97b`; model-free H100
measurements with no model download or serving run.
