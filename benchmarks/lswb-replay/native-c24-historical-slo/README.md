# Historical Native C24 SLO reference

Author: Winston Cai.

## Purpose and scope

Make the original Native C24 run visible as **repeat 1** of the historical
three-run Native aggregate in [the report](../../../docs/report.md#slo-limited-concurrency--historical-ladder).
This collection is a historical baseline record, separate from the newer
[three-mode C24 triplicate](../c24-native-packed256-n3-20261018/README.md).
No serving rerun or runtime change is involved.

## Repeat 1

Original run: `native-slo-c24/20260909_081547`. Saved protocol: C24 active users,
1,200-second dispatch window, followed by drain, on the 4,916-request,
92-session replay cohort. Throughput includes the recorded drain time.

| Measurement | Saved value |
|---|---:|
| Successful / attempted requests | 2,194 / 2,212 |
| Failed requests | 18 (`missing_output_event`) |
| Duration including drain | 1,212.200 s |
| Requests/s | 1.809932 |
| TTFT p90 | 9.356674 s |
| Mean TPOT | 51.618530 ms |
| Mean E2E | 12.966592 s |
| Device cache-hit rate | 97.304630% |

The saved harness marks the run valid at its 99% success threshold and classifies
it as a candidate comparison. TTFT p90 is below 10 seconds. This is one observed
passing run, not proof of a repeatable maximum concurrency or task-quality parity.

## Relationship to the historical aggregate

- [Repeat 1 summary](repeat-1/20260909_081547/client/summary.json) and
  [original client manifest](repeat-1/20260909_081547/client/manifest.json).
- [Repeat 2 transferred summary](https://github.com/winstonxcai/flash-optimizations/blob/ba4efd854067325dd0288408e43dffd7ea22a5f2/transfer/native-c24-repeat2/20261018_134533/client/summary.json).
- [Repeat 3 transferred summary](https://github.com/winstonxcai/flash-optimizations/blob/ba4efd854067325dd0288408e43dffd7ea22a5f2/transfer/native-c24-repeat3/20261018_134534/client/summary.json).

Only repeat 1 is curated in this directory. The report aggregates these three
historical runs; dataset and runner hashes match, but adapter hashes differ.
Do not describe them as a verified identical-build triplicate or substitute them
for the newer matched C24 collection.

## Evidence and retention

The four copied client files (`summary.json`, `manifest.json`, `done.json`,
`REPORT.md`) are byte-identical to the original archive. Review checked saved
completion status, request-throughput arithmetic and the TTFT-p90 threshold;
it did not rerun the benchmark or independently validate all request metrics.

`manifest.json` records original and current paths, hashes of every archived
file, retained files and limitations. `SHA256SUMS` covers curated files except
itself. The complete archive remains unchanged at
`raw/lswb-replay/native-slo-c24/20260909_081547/`; per-request records,
dataset-manifest duplication and GPU samples remain outside this collection.
Keep original measurements immutable. Rollback removes this curated copy and
restores its documentation links; the original evidence remains recoverable.
