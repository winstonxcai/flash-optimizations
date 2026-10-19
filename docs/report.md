# Remnant: Residual Feature Sparsity Survives Latent Compression

Author: Winston Cai.

## Overview

DeepSeek-V4-Flash already compresses its KV cache: the 21 compressed sparse-attention (CSA) layers cache each token's keys and values as a single learned latent — the 584-byte C4 state. Remnant prunes that latent a second time, keeping the largest-magnitude 256 of 512 coordinates per token (TopMag50) and storing the survivors packed — **584 → 328 bytes per record (43.84% smaller)**. In the latest TP4 H100 serving study, the allocator-reported KV pool grew from **3,745,536 to 4,536,832 full-token slots (1.2113×)**. At 128k input plus 8k output, that raises allocator-derived concurrency from **26 to 32**; this is not a latency-SLO ceiling.

The implementation matters. Across three LongSWE-Bench runs per mode at C24, **Packed Kernel mean TPOT is 0.77% above Native**, versus **7.20% for Packed Triton**; Kernel request throughput is 0.38% lower. The separate 128k/8k study likewise observes near-Native Kernel performance at matched C24. At each mode's allocator ceiling, Kernel produces **5.74% more output tokens/s**, with **16.40% higher mean end-to-end latency**. That capacity-load result is one run per configuration, not a repeatable speedup claim. Historical C24/C28 SLO observations remain below and do not establish the current Kernel's SLO ceiling.

On agentic coding, Packed averages **0.5 more solved tasks on Sangfor-Bench and 2 fewer on SWE-bench**, over two runs per mode. On DeepSWE-Bench it solves **7 versus 5** tasks in one run per mode, with most tasks reaching the time cap. These are observed differences, not demonstrated quality equivalence or improvement.

## Scope and configurations

The latest serving studies use **SGLang v0.5.18**, DeepSeek-V4-Flash-0731 and TP4 on **4× H100 80 GB**, with the fp4-native `flashinfer_mxfp4` MoE runner. Both Packed implementations use the same pruning policy and 328-byte record:

| Configuration | Persistent C4 | Store / decode path |
|---|---:|---|
| Native | 584 bytes | Unpruned Native storage; stock FlashMLA |
| Packed Triton | 328 bytes | Triton store; reconstruct Native records before stock FlashMLA |
| Packed Kernel | 328 bytes | Fused CUDA store; FlashMLA reads Packed records directly |

Recorded settings: mem-fraction-static **0.88**, FP8 E4M3 KV cache, **1,048,576** context cap, 8,192-token prefill chunks, full decode CUDA graphs through batch 136, prefill graphs disabled, and server seed **20260919**. An unset store override follows the selected Packed decode implementation. The older study rebased from v0.5.15 to v0.5.18; its original two-mode labels and measurements are retained as historical evidence, not reassigned to the current Kernel.

## Serving results

Throughput and request-level TPOT describe the complete serving workload, not isolated decode kernels. High prefix-cache hit rates do not by themselves identify the bottleneck: uncached suffix length, prefill work, GPU batch sizes and scheduling can still differ. Kernel-time or prefill-bound explanations require a matched profile, not these aggregate metrics alone.

Separate model-free prefill-read studies (`prefill-read-20261018` and `prefill-read-fused-rope-20261018`) measured additional Packed preparation cost. Fusing unpack and RoPE reduced that cost but did not beat Native. This demonstrates an operation-level penalty, not its share of serving latency or proof that prefill dominates the replay workload.

### LongSWE-Bench: matched C24, three runs per mode — 2026-10-18

Three fresh-server, unprofiled runs per mode replay the same **4,916 recorded requests from 92 users**. Each run starts with a flushed prefix cache, admits work for **1,200 seconds**, then drains in-flight requests; throughput includes the recorded drain time. C24 means 24 active replay users, not a fixed GPU batch size. There is no replay warm-up.

Mean ± sample standard deviation across the three runs:

| Mode | Requests/s | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) |
|---|---:|---:|---:|---:|
| Native | 1.8125 ± 0.0507 | 278.20 ± 6.66 | 9.339 ± 0.356 | 50.97 ± 1.44 |
| Packed Triton | 1.8037 ± 0.0378 | 274.55 ± 6.27 | 8.999 ± 0.468 | 54.64 ± 0.69 |
| Packed Kernel | 1.8056 ± 0.0165 | 275.00 ± 3.90 | 9.338 ± 0.185 | 51.37 ± 0.37 |

Kernel is **−0.38% requests/s and +0.77% mean TPOT** versus Native; Triton is **−0.48% and +7.20%**, respectively. Kernel output-token throughput is 1.15% lower. These observations show near-Native Kernel performance at C24, not statistical equivalence or a decode speedup. All nine runs meet TTFT p90 < 10 s; that establishes passing observations at C24, not a maximum concurrency. Mean device-hit rates are **97.30% / 97.38% / 97.38%** for Native/Triton/Kernel, leaving little measured cache-retention advantage at this load.

TTFT p90 is the **mean of run-level percentiles**, not a pooled percentile; TPOT averages successful-request TPOT within each run. All nine runs meet the harness's 99% success threshold but retain **15–21 failed requests per run**, excluded from latency summaries. The harness classifies these as candidate comparisons. Recorded dataset, client/launcher hashes, image and static settings match within the collection; full loaded-runtime/extension hashes and client/server request-ID joins were not captured.

Canonical aggregate and run records: `benchmarks/lswb-replay/c24-native-packed256-n3-20261018/{analysis.json,summary.csv,manifest.json}`.

### Random serving: 128k input / 8k output — 2026-10-18

Official **SGLang `bench_serving`**, random workload backed by the same local ShareGPT dataset, exact **131,072 input and 8,192 output tokens**, client seed 42 and unlimited offered request rate. Each cell uses a C-request warm-up, a cache flush, then a **3C-request measured phase**. This is **one independent run per cell**, not three repeats. The C24 and allocator-ceiling comparisons are separate load policies.

| Load policy | Mode | C | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) | Mean E2E (s) |
|---|---|---:|---:|---:|---:|---:|
| Matched | Native | 24 | 589.9 | 186.00 | 27.71 | 333.12 |
| Matched | Packed Triton | 24 | 558.1 | 187.25 | 29.96 | 352.13 |
| Matched | Packed Kernel | 24 | 588.7 | 186.17 | 27.79 | 333.84 |
| Allocator ceiling | Native | 26 | 597.4 | 202.58 | 29.48 | 356.35 |
| Allocator ceiling | Packed Triton | 32 | 600.7 | 246.78 | 36.06 | 436.23 |
| Allocator ceiling | Packed Kernel | 32 | 631.7 | 245.41 | 33.56 | 414.80 |

At C24, Kernel is **−0.21% output tokens/s and +0.27% mean TPOT** versus Native; Triton is **−5.39% and +8.09%**. At their own allocator ceilings, Kernel is **+5.74% output tokens/s**, with **+13.82% mean TPOT and +16.40% mean E2E**. This is an observed capacity/latency tradeoff at different loads, not a same-concurrency speedup.

Allocator concurrency is `floor(pool_slots / (131072 + 8192))`: **3,745,536 Native slots → C26; 4,536,832 Packed slots → C32**. It is not an SLO limit. All warm-up/measured phases completed without errors and with exact output lengths. Recorded dataset/client/launcher identities and server settings match within this collection, but exact loaded-runtime/binary and checkpoint-content identities are incomplete. This study used a different image from the LongSWE repeats; do not combine them as one identical-build experiment. No run-to-run uncertainty is available.

Canonical records: `benchmarks/serving/random-128k-8k-c24-capacity-n1-20261018/{analysis.json,summary.csv,manifest.json}`.

### Historical serving measurements

The following tables preserve older measurements and their original Native/Packed labels. They are not three-mode comparisons or measurements of the current Kernel's capacity. Prefer the 2026-10-18 studies above for matched-load implementation comparisons.

#### Historical random SGLang serving — 2,048 output tokens

TP4 on 4× H100 80 GB serving DeepSeek-V4-Flash-0731 with mem-frac 0.88, 1,048,576 context cap, fp8 KV cache, and full decode CUDA graphs to `max_bs 136` (prefill graphs off). Each point ran official `bench_serving` at exact 32k/64k/128k/256k inputs with 2048 outputs: one warm-up wave of C, then 3C measured requests in one run. Native ran at its allocator ceiling at each context; Packed at that same Native ceiling (fair) and at its own (max). The historical pools were **3,730,944 / 4,519,168 slots (1.2112×)**; these are distinct from the latest collection's pool counts.

##### Fair serving — same concurrency

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

Observed Packed throughput differs by −3.2% to +0.2% across these contexts. Latency signs mix; each 3C-request measurement is one run, not three independent repeats.

##### Allocator-ceiling serving

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

Packed's +21% pool increases offered concurrency, with observed throughput differences of −0.7% to +3.4%. Median TTFT rose 18–37% and TPOT up to 24% at the deeper Packed loads. These cold, long-input results do not isolate the bottleneck or establish identical prefill cost; they should not substitute for the newer three-mode 8k-output study.

#### Historical LongSWE-Bench (SGLang v0.5.18)

Replays **4,916 recorded Claude-agent business requests** (~144k prompt tokens/request, short decodes) from **92 tasks** over OpenAI SSE. Ordered by `planned_offset_s`, prompt length is non-decreasing in 4,823/4,823 consecutive within-task pairs. This is consistent with growing transcripts but does not prove exact token-prefix nesting. Under an idealized exact-nesting/no-eviction assumption, the recorded lengths imply **98.06%** reusable prompt tokens (14.4 M uncached of 743.5 M); that is a conditional estimate, not a measured cache-hit ceiling.

The v0.5.18 runs replace the earlier v0.5.15 measurements as this report's historical reference. The earlier study reported Native C12 / Packed C15 and a same-concurrency +77.7% throughput difference; those results are not current Kernel measurements. Version, workload execution and source differences prevent attributing their contrast with v0.5.18 solely to the upgrade. In the v0.5.18 observations, both modes achieve approximately 97% token-weighted device-cache hits.

Each historical window starts a **fresh server with an empty radix cache**, dispatches the same replay corpus for 1,200 seconds and uses TP4 H100 hardware (small decode graphs at C≤15, extended above; prefill graphs off). Fixed active-user concurrency does not fix arrival times, completed-request sets or actual GPU batches. The sweep initially stopped at the first TTFT-p90 crossing, but later C26–C28 windows passed. Treat the ladder as observed passes/crossings, not a repeatably established maximum concurrency.

##### Same concurrency — historical C21

Both modes run at **21 active users**, with one fresh-boot 1,200-second dispatch window per mode. Both observed TTFT p90 values are below 10 seconds. This controls configured concurrency and the replay corpus, not the exact executed workload; it does not supply an independent estimate of run-to-run uncertainty.

| Metric | Native (584-byte C4) | Packed (328-byte C4) | Change |
|---|---:|---:|---:|
| Completed in window | 2,248 | 2,187 | −2.7% |
| Prompt-token throughput (k tok/s) | 263.9 | 257.2 | −2.5% |
| Completion tokens (k) | 343.5 | 329.0 | −4.2% |
| Real (uncached) prefill (M tok) | 8.43 | 8.13 | −3.6% |
| Device cache-hit rate | 97.36% | 97.39% | +0.03 pp |

These windows show a **0.03-percentage-point** device-hit difference, **2.7% fewer completed requests** and **4.2% fewer completion tokens** for Packed. They do not establish equivalence or isolate a decode bottleneck. On the **2,184 requests both legs completed (matched by request_id)**, token-weighted device hit is **97.39% vs 97.40%**, mean TTFT/e2e are **5.96 s/11.21 s vs 5.94 s/11.49 s**, and median request TPOT is **23 vs 25 ms** (+2 ms). That conditional matched subset excludes requests completed by only one mode.

Latency distributions — whole completed set per leg (Native n=2,248, Packed n=2,187):

| Metric | Leg | min | p50 | p90 | p95 | p99 | mean | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT | Native | 0.76 s | 5.18 s | **9.03 s** | 10.67 s | 16.90 s | 5.96 s | 107.99 s |
| TTFT | Packed | 0.46 s | 5.07 s | **8.74 s** | 11.43 s | 17.81 s | 5.94 s | 107.26 s |
| TPOT | Native | 0 ms | 23 ms | **57 ms** | 91 ms | 596 ms | 45 ms | 2.75 s |
| TPOT | Packed | 0 ms | 25 ms | **61 ms** | 98 ms | 620 ms | 47 ms | 2.74 s |
| e2e latency | Native | 1.16 s | 7.99 s | **18.66 s** | 23.87 s | 74.43 s | 11.21 s | 163.59 s |
| e2e latency | Packed | 1.27 s | 8.01 s | **19.20 s** | 25.62 s | 84.67 s | 11.49 s | 162.82 s |

##### SLO-limited concurrency — historical ladder

Under the **TTFT-p90 < 10 s** criterion, the historical Native C24 aggregate contains **three passing runs**, with mean run-level p90 **9.27 s**; Native C25 has a single crossing at **12.10 s**. The historical Packed sweep recorded C28 passing (**9.90 s**) and C29 crossing (**10.63 s**). The difference between these observed passing points is **4 active users (+17%)**; it is not a replicated capacity gain or the current Kernel's measured SLO ceiling.

Native C24 aggregates the original report run `native-slo-c24/20260909_081547` (archived under `raw/lswb-replay/`), [repeat 2](https://github.com/winstonxcai/flash-optimizations/blob/ba4efd854067325dd0288408e43dffd7ea22a5f2/transfer/native-c24-repeat2/20261018_134533/client/summary.json), and [repeat 3](https://github.com/winstonxcai/flash-optimizations/blob/ba4efd854067325dd0288408e43dffd7ea22a5f2/transfer/native-c24-repeat3/20261018_134534/client/summary.json). The repeat links preserve the transferred summaries at their historical commit. Every Native C24 value in the aggregate table is an arithmetic mean of the three run summaries; **TTFT p90 is the mean of run-level p90s, not a pooled-request percentile**. Dataset and runner hashes match across these runs, but adapter hashes differ between the original and new runs, so this is a historical-plus-repeat aggregate rather than a verified identical-build triplicate.

Comparison at the historical passing points. Deltas use unrounded values; Packed remains n=1 and is not paired with the three Native runs. This Native aggregate is separate from the newer matched-build C24 triplicate:

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

Historical single-run latency distributions — the original completed sets (Native 2,194 requests; Packed 2,259 requests). These distributions are retained as recorded; they are not the n=3 aggregate above:

| Metric | Leg | min | p50 | p90 | p95 | p99 | mean | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT | Native @ 24 | 0.53 s | 6.45 s | **9.36 s** | 11.55 s | 26.29 s | 7.27 s | 133.46 s |
| TTFT | Packed @ 28 | 0.73 s | 6.71 s | **9.90 s** | 11.93 s | 25.18 s | 7.62 s | 166.49 s |
| TPOT | Native @ 24 | 0 ms | 22 ms | **60 ms** | 116 ms | 717 ms | 52 ms | 3.42 s |
| TPOT | Packed @ 28 | 0 ms | 31 ms | **81 ms** | 140 ms | 785 ms | 66 ms | 4.29 s |
| e2e latency | Native @ 24 | 2.20 s | 9.19 s | **20.09 s** | 29.38 s | 134.36 s | 12.97 s | 174.57 s |
| e2e latency | Packed @ 28 | 1.95 s | 10.14 s | **22.39 s** | 30.81 s | 167.56 s | 14.70 s | 226.65 s |

Ladder (TTFT p90 / mean TPOT; individual fresh-boot 1,200-second dispatch windows except Native C24 †, which uses the historical three-run means above; ★ = highest observed passing C, ✗ = crossing in that window):

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

Readings:

- **The Packed ceiling of 25 was a stop-rule artifact.** Later windows at c26 (9.97 s), c27 (9.44 s) and c28 (9.90 s) passed; c29 recorded 10.63 s. C26 shared the host with a second TP4 server, an uncontrolled condition rather than a guaranteed one-direction latency effect. These are historical passing/crossing observations, not a validated ceiling for the current Kernel.
- **Run variation:** historical repeats moved TTFT p90 by approximately 0.4 s. This is observed variation, not a confidence interval, a universal noise floor or a rule that adjacent rungs are statistically tied. Near-threshold windows require repeats before ranking configurations.
- **Cache behavior:** historical C21 device-hit rates differ by **+0.03 pp** (97.36 → 97.39). Growing transcripts are a plausible explanation for high reuse; these rates alone do not prove exact nesting, explain the full throughput plateau or rule out capacity benefits at other loads.
- **Latency strata:** in the original single-run analysis, requests with ≥95% device hits have TTFT p90 **8.97 s (Native C24) vs 9.51 s (Packed C28)**. Low-hit requests have much larger p90s (**52 s / 76 s**). These are descriptive subsets at different loads, not a causal attribution of queueing, prefill or decode time; subgroup percentiles do not add to the overall p90.
- **TPOT at different occupancy levels:** historical Packed mean 56.6 ms at C25 vs the updated Native C24 mean 51.0 ms is about +11%; historical Packed C28 at 65.9 ms is **+29.2%** over the Native three-run mean. These compare different loads and measurement dates, not an isolated store/load overhead. The original matched-C24 single-run p50 gap remains 24.0 vs 22.3 ms (+8%).
- **Observed throughput plateau:** historical windows across C≈20–28 complete approximately **2,150–2,270 requests** (~1.8 req/s, ~250–270k prompt tokens/s). The extra configured users did not produce a clear rate gain in these windows. Longer decode steps, prefill interference and scheduling are candidate explanations; aggregate throughput does not establish compute saturation or identify the dominant cost.

## Agentic Benchmark results

Sangfor-Bench and SWE-bench each contain **50 tasks, two runs per mode**, comparing untouched Native 0731 with Packed through the same Claude Code harness. Sangfor runs use TP4; SWE-bench has one TP8 pair and one TP4 pair. The table reports mean solved-task counts, not percentages. These small samples and heterogeneous SWE-bench repeats do not establish statistical equivalence. DeepSWE-Bench has one TP4 run per mode over 60 tasks; timeouts score zero under its 5,400-second agent budget.

| Evaluation | Native | Packed | Difference |
|---|---:|---:|---:|
| Sangfor-Bench (n=50, 2 runs each) | **23.0** | **23.5** | +0.5 task |
| SWE-bench (n=50, 2 runs each) | **32.5** | **30.5** | −2 tasks |
| DeepSWE-Bench (n=60, 1 run each) | **5** | **7** | +2 tasks |

### Sangfor-Bench Hard 50

**Run 1 — Native 22/50, Packed 24/50**

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 20 | 2 |
| Native fail | 4 | 24 |

**Run 2 — Native 24/50, Packed 23/50**

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 21 | 3 |
| Native fail | 2 | 24 |

One instance is adjudicated: Native-run-2's `apex_gpt-train-data-collector_1dbcd396` is counted as a pass — its patch passed all 109 runnable tests with 0 failures (2 uncollectable, pass_rate 98.2). The same instance failed tests in the other three runs.

Packed leads by two tasks in run 1 and trails by one in run 2, averaging **+0.5 solved tasks**. The direction changes across runs; two repeats do not establish a reproducible improvement or quality equivalence.

### SWE-bench

**Run 1 — Native 32/50, Packed 30/50**

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 29 | 3 |
| Native fail | 1 | 17 |

**Run 2 — Native 33/50, Packed 31/50**

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 29 | 4 |
| Native fail | 2 | 15 |

Native leads both runs by **2 tasks**. All four runs share the same 3 error instances (sphinx-7985/8269/8475), grouped as fail. This is an observed Packed deficit, not a result that can be dismissed as within measured run-to-run noise; neither a degradation significance test nor an equivalence test is established here.

### DeepSWE-Bench

60 DeepSWE-Bench tasks, one matched run per mode on TP4. A cap-hit (`AgentTimeoutError`) scores zero under the 5,400-second budget: **Native 35/60, Packed 36/60** tasks reach it. The primary result includes those failures, measuring task success under that time budget rather than model capability independent of runtime. Confusion matrix over all 60 tasks, rows = Native, columns = Packed (pass = full verifier pass, reward 1):

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 1 | 4 |
| Native fail | 6 | 49 |

A task **passes** only on a full verifier pass (reward 1); fail includes test failures and cap-hits. Packed solves **7/60 versus Native 5/60**, an observed two-task lead in one run. Among under-cap completions, the descriptive rates are **5/25 = 20.0% vs 7/24 = 29.2%**; those are differently selected subsets, not a matched estimate of quality improvement. Heavy censoring and single-digit solves limit interpretation, and no run-to-run uncertainty is available.

## Conclusion

Remnant provides **43.84% smaller C4 records and approximately 1.21× server KV capacity**. Direct Packed Kernel is close to Native at matched load: **+0.77% mean TPOT** in the three-run C24 replay and **+0.27%** in the single-run 128k/8k study, versus Triton's +7.20% / +8.09%. No meaningful same-load throughput gain is established. At its larger allocator ceiling, Kernel observed **+5.74% output tokens/s**, at **+16.40% mean E2E latency**, requiring repeats before claiming a stable gain. Historical C28 SLO results do not establish the current Kernel's ceiling; the agentic evaluations remain limited by sample size, run variation and time caps.
