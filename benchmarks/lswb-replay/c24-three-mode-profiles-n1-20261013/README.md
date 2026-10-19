# C24 three-mode replay evidence

Author: Winston Cai.

Native is the baseline, Triton the reference/control, Kernel the optimization
target. This is a committed reference collection, not a newly validated paired experiment.
Original run IDs are preserved. Retired local folders remain recoverable in
repo-root `raw/`; the VDI originals are unchanged.

## Unprofiled

| Mode | Original folder | Run ID | Replay valid |
|---|---|---|---|
| Native | native-unprofiled-c24 | 20261013_005530 | Yes |
| Triton | triton-parity-unprofiled-c24 | 20261013_035958 | Yes |
| Kernel | kernel-jit-runtime-unprofiled-c24 | 20261013_031901 | Yes |

Each mode's `unprofiled/<run-id>/` contains the original `summary.json` and
`manifest.json`. Individual validity does not establish matched configurations
or workload equality; preserve their audits and review pairing before reporting.

## Profiled

Existing four-rank C24 node-profile aggregates are copied as `analysis.json`.
The detailed interpretation is retained in `PROFILE.md`, including event loss,
short-window limitations and source-identity uncertainty. These profiles and
the unprofiled rows above are different runs, not matched capture windows.

| Mode | Original replay run ID | Report and client metadata present |
|---|---|---|
| Native | 20261013_202726 | Yes |
| Triton | 20261013_210108 | Yes |
| Kernel | 20261013_203807 | Yes |

All three original `.nsys-rep` files and client summaries/manifests were imported
from `origin/remnant` commit `9da64b0`; report sizes and SHA-256 hashes match
their LFS identities. Retention does not depend on profiled client success rate.
Graph-node/NVTX/rank coverage and event-loss caveats remain in `PROFILE.md`.
The Kernel report opened in Nsight Systems 2026.5.1 on 2026-10-19. Its initial
timeline screenshot failed visual review (collapsed kernel rows and excessive
empty space); refinement encountered viewer/process and computer-use errors.
The rejected draft remains only in local `raw/nsight-viewer-review-20261019/`.
A readable Systems screenshot and GUI checks for Native/Triton remain outstanding.

## Archive

The pre-curation tree and retired folders are in Git-ignored repo-root `raw/`. `manifest.json`
records hashes and original paths for selected copies. No logs, raw prompts,
SQLite duplicates or copied source trees are added to this comparison.
