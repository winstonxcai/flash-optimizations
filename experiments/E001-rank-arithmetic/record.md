# E001: branch-free bitmap rank arithmetic

## Hypothesis

The Remnant loader's constant-memory nibble rank table adds avoidable rank
calculation overhead. Arithmetic prefix construction should preserve every
rank while reducing loader stalls.

## Change

Replaced `kRemnantNibbleRanks[16]` in
`third_party/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh` with masked
integer arithmetic. The packed record layout, bitmap prefixes, survivor loads,
RoPE, public API, and kernel geometry were unchanged.

## Source state

- Parent: `407fcd5b9004bc2409de13eec7399757abbcd1c1`
- SGLang: `0b2006980628f915f8a0b54f28669be9160ed7de`
- FlashMLA: `529c862003ee535db5b1f4b0799cec88cc9c16b7`
- FlashMLA branch: `remnant/flashmla-v0518`
- FlashMLA candidate was checkpointed locally as commit
  `7576073b6ce5b209e4fa8306f08aba02683f0f62` on `remnant/flashmla-v0518`;
  it has not been pushed.

## Validation

The arithmetic implementation was exhaustively compared with the old table for
all 256 bitmap-byte values: **256/256 identical rank packs**.

Modal validation used the existing H64/H128 × B8/B16 × K512/K317 suite:

```text
12 passed
```

Compute Sanitizer memcheck ran 8 direct-vs-adapter cases with **0 errors**.
Racecheck reported the existing TMA/split-KV hazards across Native and Direct
kernels; it did not identify a rank-specific fault and stopped the wrapper
before synccheck.

## Microbenchmark

Configuration: H100, SM90 only, 10 warmups, 100 repeats, 30 rounds.

| Case | Native ms | Adapter ms | Direct ms | Direct p95 vs Native | Status |
|---|---:|---:|---:|---:|---|
| H64/B8/K512 | 0.014832 | 0.019328 | 0.020675 | +42.20% | MISS |
| H64/B8/K317 | 0.013177 | 0.017705 | 0.018919 | +43.98% | MISS |
| H64/B16/K512 | 0.017899 | 0.024729 | 0.028372 | +60.14% | MISS |
| H64/B16/K317 | 0.015315 | 0.022435 | 0.026750 | +75.33% | MISS |
| H128/B8/K512 | 0.016512 | 0.020754 | 0.021137 | +30.00% | MISS |
| H128/B8/K317 | 0.014637 | 0.019012 | 0.019594 | +34.02% | MISS |
| H128/B16/K512 | 0.020747 | 0.028187 | 0.031025 | +49.86% | MISS |
| H128/B16/K317 | 0.017819 | 0.024699 | 0.025069 | +41.20% | MISS |

Full artifacts:

- JSON: `remnant-stage2a-results:/microbench-20260922T162932Z.json`
- CSV: `remnant-stage2a-results:/microbench-20260922T162932Z.csv`

## Profile comparison

| Shape | Baseline Direct | Candidate Direct | Change |
|---|---:|---:|---:|
| H64/B8 | 27.840 µs | 25.760 µs | −7.47% |
| H64/B16 | 43.808 µs | 39.264 µs | −10.37% |
| H128/B8 | 27.712 µs | 25.280 µs | −8.78% |
| H128/B16 | 42.944 µs | 37.376 µs | −12.97% |
| Geometric mean | — | — | **−9.92%** |

The candidate kept 168 registers and zero spills. Direct shared-memory
wavefronts fell by roughly 0.6–0.9%, bank conflicts by 2.8–4.1%, and long
scoreboard stalls by 11.5–13.5%; instruction count increased by roughly
1.6–2.2%.

Profile artifacts:

```text
remnant-stage2a-results:/20260922T163804Z-profile-flashmla-direct-e6cd43cd/
```

## Decision

**Keep as a candidate checkpoint.** The rank replacement is numerically safe
and improves the isolated Direct kernel, but the end-to-end Direct path still
misses the 2% target by a wide margin. Do not promote this change or update
production pins until the next experiment is selected.
