# E003: asynchronous packed-survivor staging

## Hypothesis and change

Use SM90 asynchronous global-to-shared copies to stage each lane's contiguous
64-byte packed-survivor slice while the existing bitmap-prefix and scale work
runs. Aligned rows use 16-byte copies; rows offset by 8 bytes use aligned
8-byte copies. The record format, bitmap ranks, decode, RoPE, and scheduling
were unchanged. The group fence/wait were made warp-uniform after the first
draft's synccheck failure.

## Source and build provenance

- Parent HEAD: `e291723581d499d2785f51840644478594fc7225`; parent working-tree
  additions at run time: `bbb04911e102` (Dockerfile and Modal runner).
- SGLang: `0b2006980628f915f8a0b54f28669be9160ed7de`.
- FlashMLA candidate: `5dfb3366519366c615bb99d85c9a11fb048ba502`, branch
  `remnant/flashmla-v0518`; E001 base `7576073b6ce5b209e4fa8306f08aba02683f0f62`.
- Corrected source diff SHA-256:
  `3d7a01de19b0a1d327183c3ad5532023b06bdd27977ddd263cb41ba48a2f6a47`.
- Modal image: `im-smondeA6jsG3YhpJ0Nuf5N`; cold image build completed in
  659.74 seconds. Build was SM90-only (`TORCH_CUDA_ARCH_LIST=9.0`,
  `ENABLE_BELOW_SM90=OFF`, FA3 and SM100 FlashMLA disabled). `remnant_ops`,
  `flashmla_ops`, and Remnant H64/H128 variants were present. No model was
  loaded and no SM100 tests ran.
- Modal profile: `caiw`. Compute/ephemeral-app spend was `$27.38515191` before
  the run and `$28.59653646` after it (still below the `$29` selection limit).
  Total metered spend including volumes after the run was `$60.86385322`.

## Validation

- Regular validity suite:
  `test/registered/attention/unittests/dsv4/test_remnant_flashmla_direct.py`.
  Both full runs passed **12/12** cases, covering H64/H128, B8/B16, K512/K317,
  invalid padding, and CUDA graph replay.
- Compute Sanitizer memcheck on
  `-k direct_matches_fused_adapter`: **8 passed, 4 deselected, zero memcheck
  errors**.
- Synccheck is **not clean**. It reported divergent barriers in the Native
  reference path and also in an isolated Direct CUDA-graph replay, so the
  reports cannot be dismissed as Native-only. The Direct H64 SASS location
  `+0x1e90` is `BAR.SYNC.DEFER_BLOCKING 0xa, 0x100`. This remains unresolved;
  ordinary tests and memcheck passing do not establish synccheck safety.

## Native-versus-Direct benchmark

Existing `benchmark/remnant/microbench.py`, 10 warmups, 100 graph replays,
30 rounds, two passes; no replacement benchmark was introduced. Every row
misses the 2% p95 target. The table reports Direct median and p95 relative to
Native; raw samples and full JSON are in `pass1-microbench.*` and
`pass2-microbench.*`.

| Shape | Pass 1 median | Pass 1 p95 | Pass 2 median | Pass 2 p95 |
|---|---:|---:|---:|---:|
| H64 / B8 / K512 | +38.55% | +40.52% | +39.05% | +41.01% |
| H64 / B8 / K317 | +42.84% | +43.24% | +42.89% | +43.15% |
| H64 / B16 / K512 | +58.22% | +59.39% | +58.54% | +59.56% |
| H64 / B16 / K317 | +76.76% | +77.47% | +76.79% | +77.64% |
| H128 / B8 / K512 | +27.02% | +29.26% | +26.83% | +28.81% |
| H128 / B8 / K317 | +32.99% | +33.64% | +31.98% | +36.11% |
| H128 / B16 / K512 | +48.26% | +50.09% | +47.28% | +48.04% |
| H128 / B16 / K317 | +40.24% | +41.28% | +39.68% | +40.13% |

Both runs reproduce the same ranking and large gap; this experiment does not
move Direct toward Native in end-to-end decode timing.

## Profile comparison against saved E001

Direct-only Nsight Compute captures use the existing E001 H100 profile as the
reference. Times are profiler-instrumented kernel times, not microbenchmark
latencies. The geomean NCU time improves 2.49%, short of the required 5%, and
two of four shapes regress.

| Shape | NCU time E001 → E003 | Time Δ | Instructions Δ | Shared wavefronts Δ | Shared conflicts Δ | Registers / spills |
|---|---:|---:|---:|---:|---:|---:|
| H64 / B8 | 25.760 → 24.160 µs | −6.21% | +0.60% | +12.68% | +20.36% | 168 / 0 |
| H64 / B16 | 39.264 → 39.328 µs | +0.16% | −0.04% | +13.71% | +32.34% | 168 / 0 |
| H128 / B8 | 25.280 → 25.376 µs | +0.38% | +0.14% | +9.83% | +20.57% | 168 / 0 |
| H128 / B16 | 37.376 → 35.840 µs | −4.11% | −0.16% | +10.44% | +20.61% | 168 / 0 |
| Geometric mean | — | **−2.49%** | **+0.14%** | **+11.65%** | **+23.37%** | unchanged |

DRAM reads changed by only +0.003% to +0.020%; occupancy stayed about
18.72–18.74%. There are no local spill requests and SASS resource use remains
168 registers/thread, `STACK:0`, `LOCAL:0`. Disassembly confirms the intended
`LDGSTS.E.LTC128B.64/128` global-to-shared copies and dependency waits. Despite
that, shared wavefronts and shared conflicts rose on every shape, while
instruction count and measured latency were nearly flat. Warp-stall samples
were mixed: long-scoreboard share fell 1.3–6.2 percentage points, while
barrier-stall share rose by 3.9–4.7 points on H64/B8 and H128/B16 and fell on
the other two shapes. Nsight reported default GPU clocks and six unavailable
NVLink metrics; treat absolute profile times as diagnostic.

## Artifacts

- Modal corrected-source memcheck/synccheck run:
  [run](https://modal.com/apps/caiw/main/ap-bwlGcaPkeTNr6j1mJNPJ1n)
- Pass 1 benchmark:
  [run](https://modal.com/apps/caiw/main/ap-w86sSCsow00czhVfV5cJXO), local
  CSV/JSON and Modal volume files `microbench-20260923T192200Z.*`.
- Pass 2 benchmark:
  [run](https://modal.com/apps/caiw/main/ap-6aGx8LU9aK10Ap96vfb6P8), local
  CSV/JSON and Modal volume files `microbench-20260923T192427Z.*`.
- Direct-only profile:
  [run](https://modal.com/apps/caiw/main/ap-qOUJD4bzbxNASeAEaXdtTw), volume
  path `remnant-stage2a-results:/20260923T192620Z-profile-flashmla-direct-395c1794/`.
- Local NCU CSVs and SASS are in `profile/`; raw Nsight `.ncu-rep` and
  `.nsys-rep` files are ignored locally and retained in the Modal volume.
- Failed first-draft synccheck:
  [run](https://modal.com/apps/winstoncai233/main/ap-9dqCTKWA0UfP73MNSOE3jE3).

## Decision

**Reject E003 as a performance candidate.** Corrected code builds; regular
validity and memcheck pass, and the async copies are present in SASS. However,
synccheck still reports divergent barriers, microbenchmark p95 misses the 2%
target by 28.8–77.6%, and the profile misses the ≥5% improvement gate
while increasing shared-memory wavefronts/conflicts. Keep the committed source
and artifacts as the experiment record; do not promote this kernel or change
production pins on the basis of E003.
