# Remnant experiment ledger

Each entry represents one isolated code change. The source SHAs, commands,
validation results, benchmark artifacts, profile paths, and decision are kept
with the entry so later iterations can use an immutable baseline.

| ID | Change | Validation | Direct vs Native | Profile result | Decision |
|---|---|---:|---:|---:|---|
| [E001](E001-rank-arithmetic/record.md) | Replace the bitmap rank lookup table with branch-free arithmetic | 12/12; memcheck 0 errors | +30–75% p95 | Direct kernel −9.92% geometric mean | Keep as candidate; not production-ready |
| [E002](E002-parallel-metadata/record.md) | Parallelize bitmap/scale staging and prefix construction across four lanes | 12/12 twice; memcheck 8/8 clean | All eight p95 cases miss; Direct medians +1.66% slower geomean vs E001 | NCU kernel time −2.35% geomean; instructions −2.55%, shared conflicts −6.94%, no spills | Rejected: H64 timing regressions exceed 2%; improvement below 5% gate |
| [E003](E003-async-survivor-copy/record.md) | Overlap packed survivor transfer with metadata using SM90 `cp.async` | First draft: memcheck passed, synccheck failed; corrected source blocked before retest by Modal spend limit | Not run | Not run | Undetermined; source committed, validation/performance gate pending |

Large Nsight reports remain in the Modal results volume. The entry records
their exact volume paths and the corresponding Modal run URLs.
