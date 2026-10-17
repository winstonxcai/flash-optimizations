# Benchmark evidence

Author: Winston Cai.

Keep date-labeled representative comparisons, not debugging attempts. Aim for
at least three historical reference points per category as verified collections
become available; currently there is one each. Do not create placeholder dates.
Native is the baseline, Triton the reference/control, Kernel the optimization target.

## Layout

```text
results/                                  # raw VDI/Modal outputs; Git-ignored
raw/                                      # local archive; Git-ignored
transfer/                                 # selected GitHub transfers
benchmarks/
├── README.md
├── microbench/representative-YYYYMMDD/
│   ├── README.md
│   ├── manifest.json
│   ├── validation.json
│   ├── timing.json
│   ├── timing.csv
│   └── profiling/<case>/
│       ├── native.ncu-rep
│       ├── triton.ncu-rep
│       ├── kernel.ncu-rep
│       ├── metrics.json
│       └── screenshots/
└── lswb-replay/representative-YYYYMMDD/
    ├── README.md
    ├── manifest.json
    ├── analysis.json
    ├── PROFILE.md
    └── <mode>/                            # native, triton, kernel
        ├── unprofiled/<original-run-id>/
        │   ├── summary.json
        │   └── manifest.json
        └── profiled/<original-run-id>/
            ├── summary.json
            ├── manifest.json
            ├── server.1.nsys-rep
            └── screenshots/
```

## Current references

| Category | Comparison | Evidence |
|---|---|---|
| Microbench | representative-20261014 | 33 timing cases, 114 passing GPU tests, 12 Compute reports |
| LSWB replay | representative-20261013 | Three unprofiled runs, three Systems reports, node-profile analysis |

Screenshots are still outstanding. Folder dates identify collections; original
run IDs, source hashes and any cross-date measurements remain in manifests.

## Evidence rules

- Unprofiled measurements establish latency/throughput; profiles explain work.
- Profile retention requires report availability and documented kernel/NVTX/rank
  coverage, not client success rate. Retain event-loss warnings and limitations.
- Record sources, binaries, settings, seeds, inputs, commands and artifact hashes.
  Individual replay validity does not establish a matched comparison.
- Keep selected original profiler reports with Git LFS; compact JSON/CSV and
  captioned screenshots use ordinary Git. No duplicate SQLite exports.
- Copy selected VDI folders into `transfer/` and push; pull locally, archive in
  `raw/`, then curate into `benchmarks/`. See `transfer/README.md`.
- Do not change serving scripts' `results/` output layout.

Full logs, prompts, tensors, builds, copied sources and retired attempts remain
outside the curated tree. VDI retains the complete archive; local pre-curation
snapshot and `raw/retired-results-20261014/` preserve this reorganization.
Nothing was deleted. No `logs/` artifacts belong in the clean PR.
