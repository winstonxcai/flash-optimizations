# Random serving: 128k input / 8k output, C24 and capacity

Author: Winston Cai.

## Purpose and status

Single-run study of Native, Packed-256 Triton and Packed-256 Kernel at matched
C24 and each mode's allocator-derived concurrency ceiling. The six cells passed
artifact review. Repeat before treating their speed differences as stable.

Use SGLang's random workload with the local ShareGPT file recorded in each
`config.json`, exactly 131,072 input and 8,192 output tokens, length ratio 1.0,
seed 42 and unlimited offered request rate. TP4 H100 servers use the same
recorded static settings except mode, resulting pool size and per-run port.
Each run flushes cache for a C-request warm-up wave and again for a separate
3C-request measured phase. Three measured waves are **one independent run**.

## Reference measurements

| Mode | Load policy | C | Measured requests | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) | Mean E2E (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Native | Matched | 24 | 72 | 589.9 | 186.00 | 27.71 | 333.12 |
| Packed-256 Triton | Matched | 24 | 72 | 558.1 | 187.25 | 29.96 | 352.13 |
| Packed-256 Kernel | Matched | 24 | 72 | 588.7 | 186.17 | 27.79 | 333.84 |
| Native | Allocator ceiling | 26 | 78 | 597.4 | 202.58 | 29.48 | 356.35 |
| Packed-256 Triton | Allocator ceiling | 32 | 96 | 600.7 | 246.78 | 36.06 | 436.23 |
| Packed-256 Kernel | Allocator ceiling | 32 | 96 | 631.7 | 245.41 | 33.56 | 414.80 |

Allocator ceilings are `floor(pool_slots / (131072 + 8192))`: 3,745,536 Native
slots give C26; 4,536,832 Packed slots give C32. These are storage-derived loads,
not latency SLO limits or demonstrated maximum sustainable request rates.
At each mode's own ceiling, Kernel's observed output throughput is 5.7% above
Native with 16.4% higher mean E2E. These are different loads and n=1 observations.

`analysis.json` and `summary.csv` retain full-precision measurements. Compare
C24 cells for overhead and ceiling cells for capacity/latency tradeoffs.
No run-to-run uncertainty estimate is available for either comparison.

## Evidence and limits

The 2026-10-18 local review verified incoming checksums, all twelve warm-up/measured
phases, completed counts, empty error fields, exact per-request lengths, token
totals, throughput, mean TTFT and saved summaries. Dataset, benchmark script,
client Python tree, image and launcher hashes agree across all six cells;
recorded static server settings match apart from intended differences. Boot and
collection image IDs agree for every run.

The client tree hash is not a full loaded-runtime/binary manifest. Submitted Git
revision fields are unset; do not infer exact source or checkpoint content
identity from paths. GPU telemetry was not retained in this packet. This
synthetic workload cannot establish coding-task quality or a repeatable gain.
The C24 LongSWE collection used a different image and workload.

Compact `warmup.json` / `measured.json` are derived records omitting generated
text and per-token ITL arrays; aggregate statistics and other request details
remain. Original JSONL hashes and omitted fields are retained in `manifest.json`.
The unchanged packet is archived locally at
`raw/transfer-cleanup-20261019/serving-128k-8k-20261018/`; full originals,
logs and omitted arrays remain in VDI `results/`.
`SHA256SUMS` covers current curated files. Rollback uses the manifest's
`previous_curated_root`; original measurements and run IDs are unchanged.
