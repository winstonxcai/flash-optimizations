# E002 Direct-only profile comparison

Source: E002 profile run at
`remnant-stage2a-results:/20260923T172828Z-profile-flashmla-direct-ae1da578/`.
Reference: saved E001 candidate profile at
`remnant-stage2a-results:/20260922T163804Z-profile-flashmla-direct-e6cd43cd/`.
Both are H100 Nsight Compute captures of the Direct decode kernel only.

| Shape | NCU kernel time E001 → E002 (Δ) | Instructions (Δ) | Shared wavefronts (Δ) | Shared bank conflicts (Δ) |
|---|---:|---:|---:|---:|
| H64 / B8 | 25.760 → 25.600 µs (−0.62%) | 2.140M → 2.095M (−2.08%) | 472,095 → 451,898 (−4.28%) | 108,934 → 99,502 (−8.66%) |
| H64 / B16 | 39.264 → 39.520 µs (+0.65%) | 3.893M → 3.797M (−2.47%) | 864,498 → 824,740 (−4.60%) | 216,931 → 198,647 (−8.43%) |
| H128 / B8 | 25.280 → 24.032 µs (−4.94%) | 2.469M → 2.404M (−2.61%) | 624,316 → 607,559 (−2.68%) | 128,799 → 122,469 (−4.91%) |
| H128 / B16 | 37.376 → 35.744 µs (−4.37%) | 4.537M → 4.398M (−3.05%) | 1,176,127 → 1,139,276 (−3.13%) | 261,062 → 246,132 (−5.72%) |
| Geometric mean | **−2.35%** | **−2.55%** | **−3.68%** | **−6.94%** |

Additional NCU findings:

- DRAM bytes read changed by −0.01% to −0.09% per shape (−0.07% geomean).
- Registers: 168/thread on every shape in both candidates. Local spill requests
  were zero; SASS resource usage reports `STACK:0`, `LOCAL:0`, `SHARED:1024`.
- Occupancy limits and cluster occupancy were unchanged for all four shapes.
- Warp stall ratios were mixed: long scoreboard +2.10% geomean, short
  scoreboard −3.16%, barrier −1.31%. Per-shape raw ratios are in the NCU CSVs.

`cuobjdump --dump-sass --dump-resource-usage` output for both Remnant
specializations contains the `SHFL.UP` operations at lane offsets 8 and 16,
replicated in the unrolled code as expected from the pair-prefix scan. The
packed survivor vector-load path remains present. No new spills are indicated.

The NCU kernel-time profile improves on H128 but is essentially flat on H64;
it does not reach the required 5% geomean. This is consistent with rejecting
the candidate given the unprofiled benchmark's H64 regressions. The profile
used default GPU clocks, so absolute values are diagnostic; comparisons are
against the saved E001 profile under the same collection procedure.
