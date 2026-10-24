<!-- Author: Winston Cai. -->
# V4.1 Packed FP4 on H100

Question: can SM90 preserve Native V4.1 E2M1 codes/scales while retaining
256 coordinates, and consume the 224-byte Packed representation directly?
Baseline: parent 9419d0e18e1e7677c0e48b8e40ad54bbc30010ca;
SGLang 7856615f3fdf02e993f3b40ecdbbed8cb8ac9bf7;
FlashMLA 3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8.
Worktree: codex/v41-packed-fp4, selected integration v0.5.22.

## Contract and acceptance

Quantize after upstream normalization/rounding/RoPE, then retain exactly 256
largest absolute dequantized values, lower coordinate first on ties. Preserve
codes and all 32 E4M3 scales; no renormalization/requantization. Eight MSB-first
uint64 bitmap words, ascending survivors, even rank in low nibble. Native is
288 logical bytes; Packed is 224. Global values already have RoPE applied.
Keep the existing SM90 SWA layout and unpruned indexer input.

Extend the existing fork tests and scripts/modal/kernels.py: independent exact
packing/reference checks, Native/Packed Direct output/LSE, C1/C2 ownership,
prefill reconstruction, changing-input graphs, streams, physical allocation,
focused V4 regressions and strict memcheck/racecheck/synccheck. Acceptance
requires all gates and fresh-context review. Quality and serving are deferred.
No model downloads, weights, full-model loads, serving, GPUQ or SM100 work.

## Execution and stopping rule

Modal only. Total task budget $15 across all accounts, with $2 reserved for
billing lag/cleanup and at most $13 projected build/test work. CPU extension
builds and preflight precede one H100 at a time. Check live profiles/spend/apps
and verified caches before launch. Bounded jobs; no automatic retries or GPU
native builds. Stop on preflight failure or budget exhaustion and report
missing coverage as unqualified. No profiling or performance benchmarks.

Canonical entry point: REMNANT_INTEGRATION=v0.5.22 modal run
scripts/modal/kernels.py::validate_kernels (the scoped FP4 suite is added here).
Use timestamped run IDs, source/binary hashes and JSON evidence. Keep output
artifacts in the runner's existing results volume; this README indexes outcomes.

## Status

Implementation in progress. No Modal jobs launched; task spend $0.
Rollback: select auto or restore baseline fork pins. The new modes remain opt-in.
