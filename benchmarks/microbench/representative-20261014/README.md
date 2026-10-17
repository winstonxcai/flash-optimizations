# Representative three-mode microbench

Author: Winston Cai.

Four cases × Native / Packed Triton / Direct Packed Kernel; one model-free
H100 80GB (SM90). All 114 GPU validity tests passed, with no skips. All 33
unprofiled cases and 12 Nsight Compute captures passed. No production code changed.

## Unprofiled mean latency

Complete operations, µs; 30 rounds after 10 warmups. Decode uses CUDA graphs
(100 replays per sample); prefill store uses eager execution (20 calls per sample).

| Case | Native | Triton | Kernel |
|---|---:|---:|---:|
| H64 / B16 / K512 attention | 18.054 | 62.188 | 23.292 |
| H64 / B16 / K317 attention | 15.487 | 59.260 | 19.644 |
| Decode store / R24 | 3.018 | 38.291 | 5.351 |
| Prefill store / R8192 | 13.096 | 554.496 | 92.955 |

Decode includes reconstruction when applicable, attention and combine. Store
includes selection/normalization/quantization, but excludes compressor projection.
Native is unpruned; Packed retains 256/512 coordinates. This is not serving TPOT.

## Files

- `timing.json`, `timing.csv`: full 33-case matrix, raw round samples and ratios.
- `validation.json`: actual test counts and original validation artifact hash.
- `manifest.json`: source/binary hashes, versions, commands, cost and run identities.
- `profiling/<case>/{native,triton,kernel}.ncu-rep`: original viewer reports.
- `profiling/<case>/metrics.json`: launch-indexed counter summaries for all modes.

All modes and both allocations match fixture/source/binary hashes. Nsight Compute
2025.3.1 captures warmed eager operations with source import and optimized
`-lineinfo` builds. Counter coverage includes duration, DRAM, L2, occupancy,
registers, local-memory traffic and stalls. Timing remains authoritative:
profiler replay perturbs execution; caches/clocks were not controlled. Local
memory traffic alone does not prove compiler spills. No viewer screenshots yet.

## Reproduce / archive

Original invocation (preserved for source provenance):

```bash
MODAL_PROFILE=poohthewinniechurchill modal run scripts/modal/dev.py::profile \
  --run-id <unique-id> --measure-first --max-seconds 1500 \
  --cases h64_b16_k512,h64_b16_k317,decode-r24-all_valid,prefill-r8192-all_valid
```

The current profiling launcher is `scripts/modal/kernels.py::profile_kernels`;
reuse `--cases` and `--max-seconds` with a fresh run ID after budget approval.
Run `benchmark_kernels` separately for unprofiled timing; `--measure-first` was retired.

Timing: `representative-20261014T023127Z`; completed profiles:
`representative-20261014T024200Z`. The first attempt stopped after one valid
capture because the CSV parser expected long rather than wide exports.
The corrected parser passed 24 CPU tests and the real export before retry;
timing was not rerun. Both apps stopped with zero tasks.

Raw logs/exports and the interrupted attempt remain in
`remnant-stage2a-results:/microbench/<run-id>/` and the local
`~/.cache/remnant/modal-results/` archive. Only representative evidence was
copied here; historical result folders are now archived under repo-root
`raw/retired-results-20261014/`. No commit or push. Use Git LFS for
the selected viewer reports before committing.
