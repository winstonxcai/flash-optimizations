# LongSWE C28: Packed-256 Triton / Kernel, three runs per mode

Author: Winston Cai.

TP4 H100, uniform 256 survivors. Six fresh-server replays at **C28 fixed active users**, 1,200-second dispatch windows, no warm-up, flushed cache and drain included in elapsed duration. The selected dataset contains 4,916 requests across 92 sessions; only a subset completes within each measurement.

## Results

Run-level mean ± sample SD (n=3). TTFT p90 is the mean of three run p90s, not a pooled percentile.

| Mode | Requests/s | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) | Mean E2E (s) | Runs below 10 s |
|---|---:|---:|---:|---:|---:|---:|
| Packed Kernel | 1.8075 ± 0.0071 | 268.93 ± 1.24 | 10.442 ± 0.460 | 62.68 ± 1.93 | 15.27 ± 0.05 | 1/3 |
| Packed Triton | 1.7909 ± 0.0316 | 265.71 ± 6.26 | 10.185 ± 0.443 | 65.12 ± 0.83 | 15.38 ± 0.26 | 1/3 |

| Mode | Repeat | Successful / failed | Requests/s | TTFT p90 (s) | SLO |
|---|---:|---:|---:|---:|---|
| kernel | 1 | 2197 / 22 | 1.8112 | 10.835 | FAIL |
| kernel | 2 | 2175 / 17 | 1.7994 | 10.554 | FAIL |
| kernel | 3 | 2201 / 17 | 1.8121 | 9.936 | PASS |
| triton | 1 | 2124 / 19 | 1.7562 | 10.624 | FAIL |
| triton | 2 | 2174 / 21 | 1.7983 | 10.194 | FAIL |
| triton | 3 | 2205 / 17 | 1.8181 | 9.738 | PASS |

## Evidence and limitations

All 334 original transfer checksums passed before curation; this collection contains six measurements. Counts, response token totals, latency means, throughput arithmetic, cache totals, server configuration and timing coverage were checked. All six summaries satisfy the harness's ≥99% success requirement; missing-output-event failures and every SLO miss remain recorded. Neither mode passed the 10-second TTFT-p90 threshold in all three repeats. No numerical parity or live task-quality claim follows from these replays.

The harness audit is candidate-level, not eligible for official comparison: runner and adapter differ from their repository HEAD and lack a profile identity. Their content hashes match across these runs and the earlier Native C24 reference. Server timing coverage is count-based, not a verified client/server request-ID join. Parallel jobs used disjoint TP4 allocations, but shared host, fixed ordering and variation remain limitations. No new GPU jobs were run for curation.

Compare with [Packed C27](../c27-packed256-n3-20261020/README.md) and the [earlier matched C24 triplicates](../c24-native-packed256-n3-20261018/README.md). Native C24 is a different collection/load, not a paired overnight control. Source/image and GPUQ identities are in `manifest.json`.

## Reproduction

With the recorded source pins, image, harness/dataset identities and serving settings, start the selected mode using `scripts/local/serve.sh`, then run `bash scripts/local/bench-lswb.sh "$TAG" "$PORT" 28 1200`. Use uniform `C4_KEEP_COUNT=256` and the matching Triton/Kernel store and decode configuration. Repeat three times with a fresh server and flushed cache each time, exporting request timing as in the stored server settings. Manifests preserve all harness options and original run IDs. The one-time GPUQ launcher was transferred as a hash only; it is not a checked-in executable reproduction script.

## Artifact layout and provenance

`analysis.json` is the canonical per-run and aggregate record, including sample variance; `summary.csv` is its flat export. `manifest.json` maps every moved evidence file to its original transfer path, hash, original VDI identity and run ID. `provenance/gpuq-logs/` retains job logs. Repeat folders preserve original measurement filenames; mode/server or repeat/server folders retain server-side metadata and logs. Historical paths embedded in original artifacts are unchanged.

Curated SHA256SUMS cover every collection file except themselves. Original transfer/source hashes remain in the manifest. The combined transfer analysis is recoverable at `dc90c14`; these collection analyses select the same observations and statistics without pooling older measurements. No measured values were changed. Rollback: revert the curation commit to restore the original transfer layout.
