# Overnight benchmark analysis — 2026-10-20

Source transfer: `895b1a1`. All 334 original transfer checksums match; 7 GPUQ jobs succeeded, with 9 G32 measurements and 12 LongSWE replay measurements. This analysis retains every repeat, including SLO misses.

## Shared-prefix G32/C16

Same frozen generated workload, order and seed: 32 reusable prefixes, 40 requests per prefix, 1,280 requests total; nominal 128k prefix + 512-token question, exactly 256 generated tokens. Actual shared prefixes occupy 4,288,911 tokens. Native pool: 3,745,536 slots; Packed-256 pool: 4,536,832 slots (+21.13%). Same TP4, memory fraction and scheduler. Cache was successfully flushed before all 9 repeats; each mode reused one server. No warm-up requests.

Values are run-level mean ± sample SD (n=3). TTFT p90 is the mean of the three run p90s, not a pooled request percentile.

| Mode | Requests/s | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) | Mean E2E (s) | Device hit (%) | Uncached prompt (M tokens) |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 0.5567 ± 0.0049 | 142.51 ± 1.26 | 26.159 ± 1.834 | 64.25 ± 3.83 | 28.68 ± 0.25 | 82.99 ± 0.17 | 29.301 ± 0.294 |
| triton | 1.3476 ± 0.0080 | 344.99 ± 2.04 | 2.981 ± 0.279 | 29.40 ± 0.73 | 11.83 ± 0.07 | 95.32 ± 0.12 | 8.067 ± 0.210 |
| kernel | 1.3390 ± 0.0183 | 342.78 ± 4.69 | 4.801 ± 1.338 | 25.52 ± 1.86 | 11.90 ± 0.16 | 95.30 ± 0.10 | 8.101 ± 0.177 |

Packed Triton: **2.421× (+142.1%)** Native request throughput; Kernel: **2.405× (+140.5%)**. Uncached prompt tokens fall 72.5% and 72.4%, respectively. Independently summed scheduler new-prefill logs show the same mechanism (table below). All 11,520 requests complete without reported client errors and generate exactly 256 tokens. Triton versus Kernel request throughput differs by only 0.65%; these three runs do not establish a general implementation ranking.

**Interpretation:** reproducible capacity/retention benefit for this synthetic working set. The working set lies above Native capacity and below Packed capacity. This is a selected capacity-pressure regime, not evidence of a 2.4× gain on arbitrary workloads or coding-agent tasks. The old n=1 sweep is not pooled into n=3.

## LongSWE replay: Packed-256

Same 4,916-request, 92-user dataset and identical runner/adapter hashes across all repeats. Fixed active-user pool C27/C28; per-user requests are sequential. Each repeat boots a fresh server, flushes cache, and dispatches for 1,200 seconds; throughput includes drain. Outputs are not fixed length.

| Mode | C | Requests/s | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) | Mean E2E (s) | TTFT p90 <10 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| kernel | 27 | 1.7040 ± 0.1022 | 254.29 ± 15.55 | 10.352 ± 0.227 | 61.76 ± 6.77 | 15.44 ± 1.16 | 0/3 |
| kernel | 28 | 1.8075 ± 0.0071 | 268.93 ± 1.24 | 10.442 ± 0.460 | 62.68 ± 1.93 | 15.27 ± 0.05 | 1/3 |
| triton | 27 | 1.7496 ± 0.0735 | 262.78 ± 11.28 | 10.549 ± 1.618 | 63.17 ± 0.93 | 15.20 ± 0.60 | 2/3 |
| triton | 28 | 1.7909 ± 0.0316 | 265.71 ± 6.26 | 10.185 ± 0.443 | 65.12 ± 0.83 | 15.38 ± 0.26 | 1/3 |

| Mode | C | Repeat | Success / failed | Requests/s | TTFT p90 (s) | Mean TPOT (ms) | SLO |
|---|---:|---:|---:|---:|---:|---:|---|
| kernel | 27 | 1 | 1910 / 19 | 1.5869 | 10.122 | 69.33 | FAIL |
| kernel | 27 | 2 | 2143 / 21 | 1.7750 | 10.356 | 59.69 | FAIL |
| kernel | 27 | 3 | 2128 / 13 | 1.7501 | 10.577 | 56.26 | FAIL |
| kernel | 28 | 1 | 2197 / 22 | 1.8112 | 10.835 | 60.73 | FAIL |
| kernel | 28 | 2 | 2175 / 17 | 1.7994 | 10.554 | 62.73 | FAIL |
| kernel | 28 | 3 | 2201 / 17 | 1.8121 | 9.936 | 64.58 | PASS |
| triton | 27 | 1 | 2191 / 18 | 1.8139 | 9.345 | 64.10 | PASS |
| triton | 27 | 2 | 2137 / 16 | 1.7655 | 9.913 | 62.24 | PASS |
| triton | 27 | 3 | 2024 / 16 | 1.6694 | 12.387 | 63.18 | FAIL |
| triton | 28 | 1 | 2124 / 19 | 1.7562 | 10.624 | 65.51 | FAIL |
| triton | 28 | 2 | 2174 / 21 | 1.7983 | 10.194 | 64.17 | FAIL |
| triton | 28 | 3 | 2205 / 17 | 1.8181 | 9.738 | 65.67 | PASS |

Earlier Native C24 n=3 reference: **1.8125 ± 0.0507 requests/s**, TTFT p90 9.339 s, 3/3 SLO passes. Overnight Triton C28 is −1.19%, Kernel C28 −0.27% in requests/s relative to that reference. These are different loads and collection dates, not paired measurements. All overnight Packed C27/C28 group-mean p90s exceed 10 s; none qualifies as a consistently passing SLO ceiling. Individual passing repeats must not be selected to claim a stable ceiling.

Device hit rates remain 97.10–97.16% across these groups. No demonstrated retention improvement comparable to G32. All 12 summaries meet the harness ≥99% success threshold, with 13–22 missing-output-event failures per run; they are not zero-failure runs. C27 varies substantially, including Kernel repeat 1 at 1.5869 requests/s and Triton repeat 3 at 1.6694 requests/s. Retain them; no evidence justifies exclusion or attributing the variation to a particular component.

## Cache accounting cross-check

Client uncached prompt totals include successful requests only; scheduler new-prefill totals include server work for unsuccessful requests and page effects. They are separate denominators and must not be equated.

| Workload | Mode | C | Client uncached prompt (M) | Scheduler new prefill (M) |
|---|---|---:|---:|---:|
| gsp | kernel | 16 | 8.101 | 8.248 |
| gsp | native | 16 | 29.301 | 29.449 |
| gsp | triton | 16 | 8.067 | 8.215 |
| lswb | kernel | 27 | 8.320 | 8.599 |
| lswb | kernel | 28 | 8.977 | 9.275 |
| lswb | triton | 27 | 8.482 | 8.767 |
| lswb | triton | 28 | 8.784 | 9.078 |

## Validation, source identity and limits

- Verified original transfer checksums, measured counts, error arrays, exact G32 output lengths and ordered input lengths; independently recomputed request throughput, LongSWE counts/token totals/means/cache totals from response records. Verified actual server mode/store implementation, uniform 256 survivors, TP4, memory fraction, scheduler and graph configuration; no layer overrides.
- Source: parent `1e3b6f6`; SGLang `921cc2e03f61a0105cf50f0d3175a296c5e2fbb6`; FlashMLA `97f3faa58450673c04c2d4a85b395335bac0a3f1`; CUTLASS `147f5673d0c1c3dcf66f78d677fd647e4a020219`. All recorded forks clean and matched their pins. Image `sha256:09dfea18bd2c4168481710dcb1830a2c2efdb8e8f45f9d1ad7bb85a1b7ff48ab`.
- Parallel jobs used disjoint sets of four GPUs. Shared host and fixed run ordering remain possible confounders; three repeats use one dataset seed and are not three independently sampled workloads.
- No Traceback, out-of-memory, CUDA-error or AssertionError text found in measured server logs. This is a log screen, not numerical kernel validation.
- LongSWE audit is candidate-level: runner/adapter are dirty relative to their repository HEAD, but their recorded content hashes match across these runs and the earlier Native C24 reference. Harness says not eligible for official comparison and lacks a profile identity. Request-timing coverage is count-based, not a verified client/server request-ID join.
- No closed-loop task success or quality claim follows from either throughput workload. No GPU jobs were launched for this analysis.

## Decision

Keep the G32 triplicates as evidence of a capacity-sensitive synthetic workload with a repeatable ~2.4× throughput gain. Keep LongSWE as a workload where the larger pool has not produced a robust throughput or SLO-ceiling improvement. Do not present the G32 result as general agentic speedup.

## Evidence and rollback

`analysis.json` is the canonical run/aggregate record, including sample variance, source paths, validation scope and all repeats; `analysis.csv` is the flat per-run export. Statistical calculations use unweighted arithmetic means, sample SD (`n−1`) and sample variance across the three runs. Rate ratios use ratios of group means. Source metrics remain unchanged. Original SHA256SUMS cover the transferred files; analysis files were added locally afterward. Reverting the analysis-only commit removes these summaries without touching raw evidence.
