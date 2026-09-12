# Remnant Fused Reconstruction

## Goal

- Keep the 328-byte packed record and reconstruct FlashMLA's 584-byte native
  input without changing masking, scales, RoPE, or attention math.
- Compare the reconstruction adapters with native FlashMLA under changing
  selections and CUDA graph replay.
- Historical source: commits `17555c9..86b5b6d`.

## Iterations

- **Iteration 1:** staged each row's bitmap once, used aligned 32-bit NoPE
  stores, and copied the native scale field with one 64-bit store.
- **Iteration 2:** cooperatively staged all 256 packed value bytes and computed
  the eight bitmap prefixes once per row.
- **Iteration 3:** assigned each active lane four groups of four coordinates,
  calculating one rank and issuing one 32-bit output store per group.
- **Iteration 4:** compared contiguous output ownership, warp-register
  prefixes, and earlier RoPE operand loads. Earlier RoPE loads were best.
- **Iteration 5:** compared byte permutation, parallel prefix calculation, and
  the earlier RoPE schedule on the grouped expansion. The earlier RoPE schedule
  remained best; byte permutation and parallel prefixes regressed.

Only the final 5C design is retained in the refactored tree, under the name
`optimized`. The original fused kernel remains available as its control.

## Historical H100 results

These are corrected changing-selection CUDA graph means, averaged over three
rounds of 500 samples. The fixture uses `K=512`, a full 128k RoPE table, unique
physical selections, and a persistent native pool.

Reconstruction only:

| Batch | Fused | Optimized | Improvement |
|---:|---:|---:|---:|
| 1 | 7.005 µs | 3.757 µs | 46.4% |
| 8 | 11.680 µs | 5.597 µs | 52.1% |
| 16 | 18.348 µs | 8.519 µs | 53.6% |
| 24 | 25.140 µs | 11.496 µs | 54.3% |

Reconstruction plus FlashMLA:

| Batch | Native | Fused | Optimized | Optimized vs fused | Optimized vs native |
|---:|---:|---:|---:|---:|---:|
| 15 | 17.705 µs | 35.051 µs | 24.848 µs | 29.1% faster | 40.3% slower |
| 18 | 19.375 µs | 39.314 µs | 28.019 µs | 28.7% faster | 44.6% slower |
| 21 | 20.450 µs | 43.151 µs | 29.858 µs | 30.8% faster | 46.0% slower |

## Profiling result

- DRAM used 7.9% of peak; global bandwidth was not the limiting resource.
- Instruction throughput and issue-active were both 56.0%, with no register
  spills.
- The kernel averaged 11.8 active warps but only 4.1 eligible warps per cycle.
- Shared-memory loads produced about 138k bank conflicts and 137k excessive
  wavefronts.
- The remaining reconstruction gap is primarily rank-expansion dependency and
  shared-memory issue pressure. Further global-memory tuning is unlikely to
  close it by itself.

The refactored implementation must be revalidated on L4 and rerun on H100 before
these historical measurements are treated as results for the current branch.
