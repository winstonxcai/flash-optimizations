# LongSWE C27: Packed-256 Triton / Kernel, three runs per mode

Author: Winston Cai.

TP4 H100, uniform 256 survivors. Six fresh-server replays at **C27 fixed active users**, 1,200-second dispatch windows, no warm-up, flushed cache and drain included in elapsed duration. The selected dataset contains 4,916 requests across 92 sessions; only a subset completes within each measurement.

## Results

Run-level mean ± sample SD (n=3). TTFT p90 is the mean of three run p90s, not a pooled percentile.

| Mode | Requests/s | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) | Mean E2E (s) | Runs below 10 s |
|---|---:|---:|---:|---:|---:|---:|
| Packed Kernel | 1.7040 ± 0.1022 | 254.29 ± 15.55 | 10.352 ± 0.227 | 61.76 ± 6.77 | 15.44 ± 1.16 | 0/3 |
| Packed Triton | 1.7496 ± 0.0735 | 262.78 ± 11.28 | 10.549 ± 1.618 | 63.17 ± 0.93 | 15.20 ± 0.60 | 2/3 |

| Mode | Repeat | Successful / failed | Requests/s | TTFT p90 (s) | SLO |
|---|---:|---:|---:|---:|---|
| kernel | 1 | 1910 / 19 | 1.5869 | 10.122 | FAIL |
| kernel | 2 | 2143 / 21 | 1.7750 | 10.356 | FAIL |
| kernel | 3 | 2128 / 13 | 1.7501 | 10.577 | FAIL |
| triton | 1 | 2191 / 18 | 1.8139 | 9.345 | PASS |
| triton | 2 | 2137 / 16 | 1.7655 | 9.913 | PASS |
| triton | 3 | 2024 / 16 | 1.6694 | 12.387 | FAIL |

## Evidence and limitations

All 334 original transfer checksums passed before curation; this collection contains six measurements. Counts, response token totals, latency means, throughput arithmetic, cache totals, server configuration and timing coverage were checked. All six summaries satisfy the harness's ≥99% success requirement; missing-output-event failures and every SLO miss remain recorded. Neither mode passed the 10-second TTFT-p90 threshold in all three repeats. No numerical parity or live task-quality claim follows from these replays.

The harness audit is candidate-level, not eligible for official comparison: runner and adapter differ from their repository HEAD and lack a profile identity. Their content hashes match across these runs and the earlier Native C24 reference. Server timing coverage is count-based, not a verified client/server request-ID join. Parallel jobs used disjoint TP4 allocations, but shared host, fixed ordering and variation remain limitations. No new GPU jobs were run for curation.

Compare with [Packed C28](../c28-packed256-n3-20261020/README.md) and the [earlier matched C24 triplicates](../c24-native-packed256-n3-20261018/README.md). Native C24 is a different collection/load, not a paired overnight control. Source/image and GPUQ identities are in `manifest.json`.

## Reproduction

With the recorded source pins, image, harness/dataset identities and serving settings, start the selected mode using `scripts/local/serve.sh`, then run `bash scripts/local/bench-lswb.sh "$TAG" "$PORT" 27 1200`. Use uniform `C4_KEEP_COUNT=256` and the matching Triton/Kernel store and decode configuration. Repeat three times with a fresh server and flushed cache each time, exporting request timing as in the stored server settings. Manifests preserve all harness options and original run IDs. The one-time GPUQ launcher was transferred as a hash only; it is not a checked-in executable reproduction script.

## Artifact layout and provenance

`analysis.json` is the canonical per-run and aggregate record, including sample variance; `summary.csv` is its flat export. `manifest.json` maps every moved evidence file to its original transfer path, hash, original VDI identity and run ID. `provenance/gpuq-logs/` retains job logs. Repeat folders preserve original measurement filenames; mode/server or repeat/server folders retain server-side metadata and logs. Historical paths embedded in original artifacts are unchanged.

Curated SHA256SUMS cover every collection file except themselves. Original transfer/source hashes remain in the manifest. The combined transfer analysis is recoverable at `dc90c14`; these collection analyses select the same observations and statistics without pooling older measurements. No measured values were changed. Rollback: revert the curation commit to restore the original transfer layout.
