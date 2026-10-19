# Feature Sparsity Survives Latent Compression

Author: Winston Cai.

Unless otherwise noted, experiments use **DeepSeek-V4-Flash-0731, SGLang v0.5.18, and TP4 on 4× H100 80 GB**.

## Overview

Magnitude pruning retains 256 of 512 coordinates in the model's compressed attention latents. Persistent C4 records shrink **584 → 328 bytes (43.84%)**, increasing the measured KV pool **1.2113×**.

- **Recorded coding-agent replay:** at matched C24, Kernel observes **−0.38% requests/s / +0.77% mean TPOT** versus Native; Triton's TPOT penalty is **7.20%**. These differences do not establish statistical equivalence.
- **Prefix-cache-cold random serving:** Kernel observes **+5.74% output throughput** at its larger allocator ceiling, with **+16.40% mean E2E latency** (n=1).
- **Synthetic shared-prefix serving:** Kernel observes **2.60× / 1.48× Native throughput at G32/G40**, with higher cache-hit rates (one run per cell). G16 is near-neutral.
- **Agentic quality:** small evaluations show mixed task-success differences; quality equivalence is not established.

## Method and configurations

Each of the model's 21 compressed sparse-attention (CSA) layers uses a 512-dimensional shared key/value latent. C4 denotes four-token compression: each compressed record represents four original token positions. Packed retains the 256 largest-magnitude coordinates per record, storing FP8 values, a bitmap and scales in 328 bytes. Sparsity is imposed by pruning, not assumed to exist as zeros beforehand. All studies below use this policy.

Triton and Kernel implement the same pruning policy and intended numerical computation. Kernel changes execution, not the method. Store checks require matching selections and packed bytes; attention checks use numerical tolerances (output/LSE `atol=0.02, rtol=0.02`), not bitwise identity or identical generated trajectories. These checks compare implementations of Packed, not Packed quality against unpruned Native.

| Configuration | Persistent C4 | Store / decode path |
|---|---:|---|
| Native | 584 bytes | Unpruned storage; stock FlashMLA |
| Packed Triton | 328 bytes | Triton store; reconstruct Native records before stock FlashMLA |
| Packed Kernel | 328 bytes | Fused CUDA store; FlashMLA reads Packed records directly |

Current serving settings: `flashinfer_mxfp4` MoE runner, static memory fraction 0.88, FP8 E4M3 KV, context cap 1,048,576, prefill chunks 8,192, full decode graphs through batch 136, prefill graphs disabled, server seed 20260919. An unset store override follows the Packed decode implementation.

Measured pools are **3,745,536 Native / 4,536,832 Packed full-token slots**, expressed in original-token units rather than C4 records. C4 records compress **1.78×**, but other cache costs remain: total KV capacity rises **1.2113× (21.13%)**, not 1.78×. Neither ratio describes total GPU-memory reduction. For fixed prompt/output lengths, `floor(pool_slots / (input_tokens + output_tokens))` estimates storage-limited concurrency. Successful runs at those loads verify admission, not the maximum sustainable concurrency or a latency-SLO limit.

## Evaluation protocol

### Dataset attribution

The recorded coding-agent replay uses the full **LongSWE-Bench** dataset, the code-repair component of **LongCodeBench (LCB)** ([Rando et al., 2025](https://arxiv.org/abs/2505.07897)). It contains **4,916 model requests from 92 task/user sessions**, not 4,916 independent tasks. The frozen [dataset manifest](../benchmarks/lswb-replay/c24-packed256-store-decode-ablation-n1-20261014/kernel-store-kernel-decode/20261014_123504/client/dataset_manifest.json) retains request/session IDs and source hashes.

Our `business-user-replay-v2` adaptation uses a fixed active-user pool, sequential requests within each user, and a 1,200-second dispatch window followed by drain. The full corpus is available, but each window starts and completes only part of it. Runs start fresh servers with flushed radix caches and no replay warm-up; throughput includes drain time. C24 means 24 active users, not a fixed GPU batch or identical completed request sets. This is serving replay, not LCB's original task-quality evaluation.

### Measurement conventions

- **TTFT:** request-to-first-token latency, including waiting and prefill. **TPOT:** mean time per subsequent output token within a request, not isolated GPU decode time. **E2E:** per-request completion latency, not the whole benchmark's wall time. Throughput measures aggregate completed work per second.
- Matched client concurrency compares behavior under the same concurrency limit, not necessarily identical GPU batches or completed work. Allocator ceilings below are calculated storage limits, not independently established runtime maxima. Use **TTFT p90 <10 s** for observed replay SLO passes.
- For serving studies, `n` counts independent runs, not request waves. Repeated results use mean ± sample SD; aggregated TTFT p90 is the mean of run-level percentiles, not a pooled percentile.
- Latencies summarize successful requests; failures remain recorded. Serving results do not establish coding-task success, and aggregate metrics alone do not identify a bottleneck.
- Source and configuration identities are documented per collection. Current records lack complete loaded-runtime/binary identity; different images and historical implementations remain separate experiments.

## Current serving results

### Recorded replay: matched C24, n=3 — 2026-10-18

| Mode | Requests/s | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) |
|---|---:|---:|---:|---:|
| Native | 1.8125 ± 0.0507 | 278.20 ± 6.66 | 9.339 ± 0.356 | 50.97 ± 1.44 |
| Packed Triton | 1.8037 ± 0.0378 | 274.55 ± 6.27 | 8.999 ± 0.468 | 54.64 ± 0.69 |
| Packed Kernel | 1.8056 ± 0.0165 | 275.00 ± 3.90 | 9.338 ± 0.185 | 51.37 ± 0.37 |

Kernel differs from Native by **−0.38% requests/s and +0.77% mean TPOT**; Triton's TPOT penalty is **7.20%**. Mean device-hit rates are **97.30% / 97.38% / 97.38%**: little difference in observed hit rate, not a direct measure of the value of retained prefixes. All nine runs pass the TTFT criterion; this does not locate their maximum passing concurrency or establish performance equivalence.

The runs satisfy the harness's 99% success threshold with **15–21 failures each**. Dataset, harness/launcher hashes, image and recorded static settings match within the collection. Client/server request-ID joins were not captured; the harness classifies these as candidate comparisons. [Canonical results and provenance](../benchmarks/lswb-replay/c24-native-packed256-n3-20261018/README.md).

### Prefix-cache-cold random serving: 128k input / 8k output, n=1 — 2026-10-18

SGLang `bench_serving` uses the same local ShareGPT-backed random dataset, exact **131,072 input / 8,192 output tokens**, seed 42 and unlimited offered rate. **C is the maximum simultaneous requests, not the total request count.** Each cell has C warm-up requests followed by 3C measured requests, with a prefix-cache flush before each phase. At matched C24, **each mode runs 24 warm-up + 72 measured requests**, with at most 24 in flight. Warm-up results are excluded; each cell is one independent run.

| Load policy | Mode | C | Measured requests | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) | Mean E2E (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Matched | Native | 24 | 72 | 589.9 | 186.00 | 27.71 | 333.12 |
| Matched | Packed Triton | 24 | 72 | 558.1 | 187.25 | 29.96 | 352.13 |
| Matched | Packed Kernel | 24 | 72 | 588.7 | 186.17 | 27.79 | 333.84 |
| Allocator ceiling | Native | 26 | 78 | 597.4 | 202.58 | 29.48 | 356.35 |
| Allocator ceiling | Packed Triton | 32 | 96 | 600.7 | 246.78 | 36.06 | 436.23 |
| Allocator ceiling | Packed Kernel | 32 | 96 | 631.7 | 245.41 | 33.56 | 414.80 |

At matched C24, Kernel is **−0.21% output throughput / +0.27% TPOT**, versus Triton's **−5.39% / +8.09%**. At their allocator ceilings (Native C26, Packed C32), Kernel delivers **5.74% more output tokens/s** but **16.40% longer mean E2E**: 414.80 s versus Native's 356.35 s. More aggregate work is completed per second, not faster individual requests; the loads and total request counts differ. All phases completed without errors and with exact output lengths. These n=1 observations do not establish a repeatable gain.

This collection used a different image from the replay triplicate. Dataset/client/launcher identities and recorded settings match internally; checkpoint-content identity is incomplete. [Canonical results and provenance](../benchmarks/serving/random-128k-8k-c24-capacity-n1-20261018/README.md).

### Shared-prefix working sets: C16, n=1 — 2026-10-19

SGLang's `generated-shared-prefix` dataset supplies **1,280 requests per cell**, targeting a 131,072-token group prefix plus a 512-token unique question, with exactly 256 output tokens. A group shares one reusable prefix; **G16/G32/G40 are group counts, not concurrency**. Actual prompts span 134,414–134,693 tokens. Requests are uniformly shuffled with seed 42; the cache is flushed before each cell, with no request warm-up. Each mode keeps its server resident through G16 → G32 → G40.

| Groups | Mode | Requests/s | Output tok/s | Device hits | TTFT p90 (s) | Mean TPOT (ms) | Req/s vs Native |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | Native | 1.5584 | 398.96 | 96.12% | 2.93 | 25.97 | 1.000× |
| 16 | Packed Triton | 1.5544 | 397.91 | 96.37% | 2.49 | 27.76 | 0.997× |
| 16 | Packed Kernel | 1.5329 | 392.43 | 96.02% | 3.25 | 27.39 | 0.984× |
| 32 | Native | 0.5446 | 139.43 | 82.94% | 24.57 | 69.96 | 1.000× |
| 32 | Packed Triton | 1.2800 | 327.68 | 95.45% | 3.08 | 31.21 | 2.350× |
| 32 | Packed Kernel | 1.4173 | 362.82 | 95.49% | 2.92 | 26.80 | 2.602× |
| 40 | Native | 0.3014 | 77.15 | 65.30% | 55.37 | 99.12 | 1.000× |
| 40 | Packed Triton | 0.4406 | 112.80 | 77.89% | 29.03 | 86.63 | 1.462× |
| 40 | Packed Kernel | 0.4469 | 114.40 | 77.81% | 36.24 | 78.24 | 1.483× |

The shared-prefix working sets total approximately **2.145M / 4.289M / 5.361M tokens**: below both pools, between them, and above both. Kernel's gains at G32/G40 accompany higher cache retention; G16 remains near-neutral. This is consistent with avoiding repeated prefill, not proof of causality or faster intrinsic decode. At G32, Native fails the 10-second TTFT criterion, so the 2.60× ratio is a matched-concurrency result rather than a both-pass SLO comparison.

All nine cells completed 1,280 requests without errors. Checksums, input lengths, exact output lengths and cache accounting passed review. GPU allocations were disjoint but overlapped on one host; execution order and host contention were uncontrolled. These single-run results need replication. [Canonical results and provenance](../benchmarks/serving/shared-prefix-c16-n1-20261019/README.md).

### Supporting operation studies

Model-free [split-reader](../benchmarks/microbench/prefill-read-20261018/README.md) and [fused unpack/RoPE](../benchmarks/microbench/prefill-read-fused-rope-20261018/README.md) studies measure additional Packed preparation cost. Fusion reduces the penalty but remains slower than Native. These measurements do not establish the cost's share of serving latency; see the [historical execution profiles](../benchmarks/lswb-replay/c24-three-mode-profiles-n1-20261013/PROFILE.md) for diagnostic evidence and capture limitations.

## Agentic benchmark results

All agentic Packed runs used **Triton**; Kernel was not separately task-evaluated. Its numerical checks do not establish identical generated trajectories. Sangfor-Bench and SWE-bench use the same Claude Code harness against untouched Native: **50 tasks, two runs per mode**. SWE-bench's first pair uses **TP8**, while its second uses the default configuration. DeepSWE-Bench has one run per mode over 60 tasks, with timeouts scoring zero under a 5,400-second budget. Counts are mean solved tasks, not percentages.

| Evaluation | Native | Packed Triton | Difference |
|---|---:|---:|---:|
| Sangfor-Bench (50 tasks; 2 runs/mode) | **23.0** | **23.5** | +0.5 task |
| SWE-bench (50 tasks; 2 runs/mode) | **32.5** | **30.5** | −2 tasks |
| DeepSWE-Bench (60 tasks; 1 run/mode) | **5** | **7** | +2 tasks |

Sangfor's direction changes across runs; SWE-bench consistently favors Native by two tasks. DeepSWE's **35/60 Native and 36/60 Packed timeouts** heavily limit interpretation. These small, partly heterogeneous evaluations establish neither quality equivalence nor a consistent improvement. Paired outcomes and scoring details are retained in [Appendix B](#appendix-b-agentic-outcomes).

## Findings

The supported system benefit is **21.13% more measured persistent KV capacity**. At matched C24, Kernel's observed requests/s and TPOT differences are **−0.38% / +0.77%** in replay; output throughput and TPOT differ by **−0.21% / +0.27%** in random serving. These are small measured differences, not established equivalence. Throughput gains depend on workload: the random capacity-ceiling gain includes higher latency, and the synthetic shared-prefix gains remain single-run observations. Task-quality effects remain uncertain.

Historical passing points below are not the current Kernel's validated SLO ceiling.

## Appendix A: historical serving measurements

These results preserve the original two-mode Native/Packed implementations and pool sizes. They are not reassigned to the current three-mode configurations. Prefer the current studies for implementation comparisons.

### Historical random serving: 2,048 output tokens

Each point used a C-request warm-up and 3C measured requests. Historical pools were **3,730,944 / 4,519,168 slots**; memory fraction, KV dtype, context cap and graph settings match those above.

#### Same concurrency

| Context | Concurrency | Mode | Requests/s | Total tokens/s | Median TTFT (ms) | Median TPOT (ms) | Median e2e (ms) | Δ tokens/s vs Native |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32k | 107 | Native | 0.3401 | 11,840 | 122,674 | 79.3 | 283,198 | — |
| 32k | 107 | Packed | 0.3293 | 11,467 | 114,286 | 89.7 | 336,650 | −3.2% |
| 64k | 55 | Native | 0.1813 | 12,251 | 87,501 | 98.7 | 285,308 | — |
| 64k | 55 | Packed | 0.1816 | 12,271 | 99,291 | 97.1 | 295,598 | +0.2% |
| 128k | 28 | Native | 0.1039 | 13,834 | 122,015 | 71.9 | 269,184 | — |
| 128k | 28 | Packed | 0.1018 | 13,551 | 122,701 | 74.4 | 274,768 | −2.0% |
| 256k | 14 | Native | 0.0473 | 12,490 | 143,886 | 74.2 | 296,208 | — |
| 256k | 14 | Packed | 0.0466 | 12,318 | 144,542 | 76.0 | 300,412 | −1.4% |

#### Each mode's allocator ceiling

| Context | Mode | Allocator concurrency | Requests/s | Total tokens/s | Δ tokens/s vs Native | Median TTFT (ms) | Median TPOT (ms) | Median e2e (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32k | Native | 107 | 0.3401 | 11,840 | — | 122,674 | 79.3 | 283,198 |
| 32k | Packed | 129 | 0.3401 | 11,841 | +0.0% | 152,028 | 93.7 | 328,838 |
| 64k | Native | 55 | 0.1813 | 12,251 | — | 87,501 | 98.7 | 285,308 |
| 64k | Packed | 66 | 0.1874 | 12,666 | +3.4% | 119,464 | 101.1 | 325,486 |
| 128k | Native | 28 | 0.1039 | 13,834 | — | 122,015 | 71.9 | 269,184 |
| 128k | Packed | 33 | 0.1032 | 13,732 | −0.7% | 143,812 | 86.1 | 319,860 |
| 256k | Native | 14 | 0.0473 | 12,490 | — | 143,886 | 74.2 | 296,208 |
| 256k | Packed | 17 | 0.0470 | 12,426 | −0.5% | 173,528 | 91.8 | 361,531 |

Extra capacity raised offered concurrency without a substantial throughput gain; latency generally increased. These prefix-cache-cold long-input screens are n=1 per cell and do not isolate prefill or decode costs. Total tokens/s counts input plus output, unlike the output-only throughput in the current studies.

### Historical recorded replay

The same 4,916-request cohort averages roughly 144k prompt tokens with short decodes. Prompt length is non-decreasing in 4,823/4,823 within-task pairs. Under an **idealized exact-nesting/no-eviction assumption**, lengths imply 98.06% reusable tokens (14.4M uncached of 743.5M); lengths alone do not establish exact prefix nesting.

An earlier study under a different software configuration, **SGLang v0.5.15**, reported Native C12 / Packed C15 and +77.7% same-concurrency throughput. Source, version and execution differences prevent treating it as a controlled upgrade comparison. Historical runs used fresh servers, empty radix caches and 1,200-second dispatch windows; decode graph coverage was extended above C15.

#### Matched C21

One run per mode, both below 10-second TTFT p90:

| Metric | Native (584-byte C4) | Packed (328-byte C4) | Change |
|---|---:|---:|---:|
| Completed in window | 2,248 | 2,187 | −2.7% |
| Prompt-token throughput (k tok/s) | 263.9 | 257.2 | −2.5% |
| Completion tokens (k) | 343.5 | 329.0 | −4.2% |
| Real (uncached) prefill (M tok) | 8.43 | 8.13 | −3.6% |
| Device cache-hit rate | 97.36% | 97.39% | +0.03 pp |

On the 2,184 requests completed by both modes, device hits are 97.39% / 97.40%, mean TTFT/E2E 5.96/11.21 s versus 5.94/11.49 s, and median TPOT 23 versus 25 ms. This matched subset excludes requests completed by only one mode.

Whole-successful-set latency distributions:

| Metric | Leg | min | p50 | p90 | p95 | p99 | mean | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT | Native | 0.76 s | 5.18 s | **9.03 s** | 10.67 s | 16.90 s | 5.96 s | 107.99 s |
| TTFT | Packed | 0.46 s | 5.07 s | **8.74 s** | 11.43 s | 17.81 s | 5.94 s | 107.26 s |
| TPOT | Native | 0 ms | 23 ms | **57 ms** | 91 ms | 596 ms | 45 ms | 2.75 s |
| TPOT | Packed | 0 ms | 25 ms | **61 ms** | 98 ms | 620 ms | 47 ms | 2.74 s |
| e2e latency | Native | 1.16 s | 7.99 s | **18.66 s** | 23.87 s | 74.43 s | 11.21 s | 163.59 s |
| e2e latency | Packed | 1.27 s | 8.01 s | **19.20 s** | 25.62 s | 84.67 s | 11.49 s | 162.82 s |

#### SLO-limited concurrency — historical ladder

Native C24 has three historical passing runs; C25 has one crossing at 12.10 s. Packed C28 passes at 9.90 s and C29 crosses at 10.63 s. The **four-user difference (+17%)** compares observed passing points, not replicated maximum capacity.

Native's historical aggregate combines [repeat 1](../benchmarks/lswb-replay/native-c24-historical-slo/repeat-1/20260909_081547/client/summary.json), [repeat 2](https://github.com/winstonxcai/flash-optimizations/blob/ba4efd854067325dd0288408e43dffd7ea22a5f2/transfer/native-c24-repeat2/20261018_134533/client/summary.json), and [repeat 3](https://github.com/winstonxcai/flash-optimizations/blob/ba4efd854067325dd0288408e43dffd7ea22a5f2/transfer/native-c24-repeat3/20261018_134534/client/summary.json). Dataset and runner hashes match, but adapter hashes differ. The [historical collection](../benchmarks/lswb-replay/native-c24-historical-slo/README.md) preserves repeat 1's original provenance. This aggregate is separate from the newer matched C24 triplicate.

Native values below average three run summaries; Packed is n=1. Deltas use unrounded values.

| Metric | Native @ 24 ★ (n=3 mean) | Packed @ 28 ★ (historical n=1) | Δ P28 vs N24 |
|---|---:|---:|---:|
| Completed requests per run | 2,190.67 | 2,259 | +3.1% |
| Requests/s | 1.8023 | 1.8710 | +3.8% |
| Prompt-token throughput (k tok/s) | 257.24 | 263.98 | +2.6% |
| Completion tokens (k) | 336.70 | 329.49 | −2.1% |
| Real (uncached) prefill (M tok) | 8.429 | 9.155 | +8.6% |
| Device cache-hit rate | 97.3037% | 97.1276% | −0.18 pp |
| TTFT p90 | **9.2654 s** | **9.9022 s** | +0.64 s |
| Mean TPOT | 51.01 ms | 65.91 ms | +29.2% |
| Mean e2e latency | 13.06 s | 14.70 s | +12.6% |

Original single-run distributions (Native 2,194 successful requests; Packed 2,259), separate from the aggregate:

| Metric | Leg | min | p50 | p90 | p95 | p99 | mean | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT | Native @ 24 | 0.53 s | 6.45 s | **9.36 s** | 11.55 s | 26.29 s | 7.27 s | 133.46 s |
| TTFT | Packed @ 28 | 0.73 s | 6.71 s | **9.90 s** | 11.93 s | 25.18 s | 7.62 s | 166.49 s |
| TPOT | Native @ 24 | 0 ms | 22 ms | **60 ms** | 116 ms | 717 ms | 52 ms | 3.42 s |
| TPOT | Packed @ 28 | 0 ms | 31 ms | **81 ms** | 140 ms | 785 ms | 66 ms | 4.29 s |
| e2e latency | Native @ 24 | 2.20 s | 9.19 s | **20.09 s** | 29.38 s | 134.36 s | 12.97 s | 174.57 s |
| e2e latency | Packed @ 28 | 1.95 s | 10.14 s | **22.39 s** | 30.81 s | 167.56 s | 14.70 s | 226.65 s |

Each ladder point is one fresh-boot window except Native C24 † (three-run means); ★ marks the highest observed passing C and ✗ a crossing.

| C | Native TTFT p90 | Native TPOT | Packed TTFT p90 | Packed TPOT |
|---|---:|---:|---:|---:|
| 15 | 6.66 s | 33.8 ms | 5.93 s | 37.5 ms |
| 16 | 6.83 s | 36.3 ms | 7.31 s | 38.1 ms |
| 17 | 7.77 s | 36.9 ms | 7.57 s | 40.9 ms |
| 18 | 7.67 s | 39.4 ms | 7.80 s | 42.8 ms |
| 19 | 8.46 s | 41.0 ms | 7.86 s | 45.1 ms |
| 20 | 8.40 s | 45.0 ms | 8.79 s | 47.2 ms |
| 21 | 9.03 s | 45.0 ms | 8.74 s | 47.2 ms |
| 22 | 9.26 s | 45.9 ms | 8.97 s | 51.1 ms |
| 23 | 9.85 s | 47.7 ms | 8.85 s | 52.9 ms |
| 24 | 9.27 s ★ † | 51.0 ms † | 9.14 s | 54.8 ms |
| 25 | 12.10 s ✗ | 59.6 ms | 9.40 s | 56.6 ms |
| 26 | — | — | 9.97 s | 57.2 ms |
| 27 | — | — | 9.44 s | 61.1 ms |
| 28 | — | — | 9.90 s ★ | 65.9 ms |
| 29 | — | — | 10.63 s ✗ | 68.6 ms |

Interpretation:

- An earlier Packed C25 ceiling was a stop-rule artifact: later C26–C28 windows passed. C26 shared its host with another four-GPU server. Historical TTFT-p90 variation was about 0.4 s, not a confidence interval or universal noise floor.
- Historical C21 cache hits differ by only +0.03 pp. At C24/C28, requests with ≥95% device hits have TTFT p90 8.97/9.51 s; low-hit requests reach 52/76 s. These descriptive strata do not isolate queueing or prefill costs.
- Packed C28 mean TPOT is +29.2% versus the historical Native C24 aggregate; these are different loads. The original matched-C24 single-run median was 24.0 versus 22.3 ms (+8%).
- C≈20–28 windows complete roughly 2,150–2,270 requests (~1.8 requests/s, 250–270k prompt tokens/s). Extra users did not yield a clear throughput gain; the aggregate cannot identify the bottleneck.

## Appendix B: agentic outcomes

Matrices show paired task outcomes, with Native as rows and **Packed Triton** (abbreviated Packed) as columns. Recorded errors and cap-hits count as failures; the Sangfor adjudication is documented below. DeepSWE-Bench requires a full verifier pass (reward 1).

### Sangfor-Bench Hard 50

Run 1: Native 22/50, Packed 24/50.

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 20 | 2 |
| Native fail | 4 | 24 |

Run 2: Native 24/50, Packed 23/50.

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 21 | 3 |
| Native fail | 2 | 24 |

Native run 2's `apex_gpt-train-data-collector_1dbcd396` is adjudicated as a pass: all 109 runnable tests passed, with two uncollectable tests (reported pass_rate 98.2). It failed in the other three runs.

### SWE-bench

Run 1: Native 32/50, Packed 30/50.

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 29 | 3 |
| Native fail | 1 | 17 |

Run 2: Native 33/50, Packed 31/50.

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 29 | 4 |
| Native fail | 2 | 15 |

All four runs share the same three error instances (sphinx-7985/8269/8475), counted as failures. The observed two-task Packed deficit in both runs has not been subjected to a degradation or equivalence test.

### DeepSWE-Bench

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 1 | 4 |
| Native fail | 6 | 49 |

Native solves 5/60, Packed 7/60. Native/Packed hit the 5,400-second cap on 35/36 tasks. Among under-cap completions, success is 5/25 (20.0%) versus 7/24 (29.2%); these differently selected subsets are not a matched estimate of quality improvement. One run and single-digit solve counts limit inference.

## References

- Stefano Rando, Luca Romani, Alessio Sampieri, Luca Franco, John Yang, Yuta Kyuragi, Fabio Galasso, and Tatsunori Hashimoto. **LongCodeBench: Evaluating Coding LLMs at 1M Context Windows.** 2025. [arXiv:2505.07897](https://arxiv.org/abs/2505.07897). [Official code](https://github.com/Zteefano/long-code-bench) and [dataset](https://huggingface.co/datasets/Steefano/LCB).
