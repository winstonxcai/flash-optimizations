# LongSWE C24: Native / Packed-256, three runs per mode

Author: Winston Cai.

## Purpose and status

Preferred repeated reference for implementation overhead at matched C24.
Compare Native, Packed-256 Triton and Packed-256 Kernel using three fresh-server
runs per mode. This is an internal serving reference with the provenance limits
below; it does not establish a capacity gain or coding-task quality.

Protocol: TP4 H100, C24 fixed active users, 1,200-second dispatch windows, no
profiling, no replay warm-up, and a flushed prefix cache. The dataset contains
4,916 recorded requests; the window admits a subset and drains in-flight work.
Throughput uses recorded elapsed duration, including drain.

## Reference measurements

Mean ± sample SD across three runs per mode. TTFT p90 is the mean of each run's
p90, not a percentile over pooled requests. TPOT is the mean per successful
request. Failed requests remain recorded and are excluded from latency summaries.

| Mode | Requests/s | TTFT p90 (s) | Mean TPOT (ms) | Runs below 10 s TTFT p90 |
|---|---:|---:|---:|---:|
| Native | 1.8125 ± 0.0507 | 9.339 ± 0.356 | 50.97 ± 1.44 | 3/3 |
| Packed-256 Triton | 1.8037 ± 0.0378 | 8.999 ± 0.468 | 54.64 ± 0.69 | 3/3 |
| Packed-256 Kernel | 1.8056 ± 0.0165 | 9.338 ± 0.185 | 51.37 ± 0.37 | 3/3 |

`analysis.json` is the canonical aggregate; `summary.csv` retains full-precision
run-level measurements. Sample variance is retained in JSON. Original run IDs
and transferred request metadata are mapped in the manifest.

## Evidence and limits

The 2026-10-18 local review verified incoming checksums, all nine completion/timing
validations, client counts and mean latency metrics, run-level throughput, and
aggregate sample statistics. Dataset and adapter/runner hashes, launcher hashes,
image identity and all recorded static server settings match apart from intended
mode/capacity and per-run ports/log paths. An omitted store selector follows the
decode implementation.

All nine meet the harness's 99% success threshold, with 15–21 failures per run.
The saved harness audit classifies them as candidate runs, not official
comparisons. Client/server timing coverage is count-based, not a verified
request-ID join. Full loaded runtime source and extension hashes were not
captured; image identity alone cannot establish them. Three repeats describe
observed variation but do not establish quality equivalence or a stable SLO
capacity ceiling.

The unchanged packet is archived locally at
`raw/transfer-cleanup-20261019/lswb-c24-n3-20261018/`; complete logs
and originals remain in VDI `results/`. Raw prompts are not copied.
`manifest.json` retains original paths, hashes and the curation review;
`SHA256SUMS` covers current curated files. Rollback uses
`previous_curated_root`; original measurements are unchanged.
