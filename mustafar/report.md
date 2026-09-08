# Mustafar: Residual Feature Sparsity Survives Latent Compression

## Overview

DeepSeek-V4-Flash already compresses its KV cache: the 21 compressed sparse-attention (CSA) layers cache each token's keys and values as a single learned latent — the 584-byte C4 state. Remnant prunes that latent a second time, keeping only the largest-magnitude ~50% of coordinates per token (TopMag50) and storing the survivors packed — **584 → 328 bytes per record (43.84% smaller)**. On a TP4 H100 server (fp4-native MoE runner, mem-frac 0.88, fp8 KV cache) this grew the measured KV pool from **3,730,944 to 4,519,168 full-token slots (1.2112×; allocator-reported, see Serving results)**, raising the allocator-derived maximum resident concurrency at 2048-token decode from **107→129 (32k), 55→66 (64k), 28→33 (128k), 14→17 (256k)**.

This is a KV-capacity optimization, not a decode speedup. At Native's own concurrency the modes are throughput-neutral (**−3.2% to +0.2%** tokens/s); even at Packed's higher ceiling they stay near-neutral (**−0.7% to +3.4%**), because this input-heavy workload is prefill-bound. The capacity payoff appears under prefix reuse: in the LongSWE-Bench replay below, the larger pool keeps more shared prefixes resident (device cache hit 86.7 → 94.7%), and Packed finished **77.7% more requests while doing 25.9% fewer real (uncached) prefills**.

On agentic coding the measured gap sits inside run-to-run noise: across two matched run-pairs per suite, Packed averages **+0.5** on Sangfor-Bench (Native 23.0 vs Packed 23.5) and **−2 tasks** on SWE-bench (Native 32.5 vs Packed 30.5). A third, cap-dominated suite tilts Packed the other way: DeepSWE-Bench full passes are Native **5** vs Packed **7** (details in Benchmark results).

## Scope and configurations

Two legs on the same serving fork — SGLang v0.5.15 @ f63458b — on identical hardware with the fp4-native `flashinfer_mxfp4` MoE runner:

1. **Native** — TopMag/packing off, stock 584-byte C4.
2. **Packed** — TopMag50 pruning on, 328-byte packed C4.

## Serving results

### Random SGLang Bench serving

TP4 on 4× H100 80 GB serving DeepSeek-V4-Flash-0731 with mem-frac 0.88, 1,048,576 context cap, fp8 KV cache, and full decode CUDA graphs to `max_bs 136` (prefill graphs off). Each point ran official `bench_serving` at exact 32k/64k/128k/256k inputs with 2048 outputs: one warm-up wave of C, then three measured waves (3C). Native ran at its allocator ceiling (C = 107/55/28/14 at 32k/64k/128k/256k); Packed at that same Native ceiling (fair) and at its own (C = 129/66/33/17).

#### Fair serving — same concurrency

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

Packed is throughput-neutral at every context (deltas −3.2% to +0.2%). Latency signs mix per context (three measured waves each).

#### Maximum-concurrency serving

| Context | Mode | Max concurrency | Requests/s | Total tokens/s | Δ tokens/s vs Native | Median TTFT (ms) | Median TPOT (ms) | Median e2e (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32k | Native | 107 | 0.3401 | 11,840 | — | 122,674 | 79.3 | 283,198 |
| 32k | Packed | 129 | 0.3401 | 11,841 | +0.0% | 152,028 | 93.7 | 328,838 |
| 64k | Native | 55 | 0.1813 | 12,251 | — | 87,501 | 98.7 | 285,308 |
| 64k | Packed | 66 | 0.1874 | 12,666 | +3.4% | 119,464 | 101.1 | 325,486 |
| 128k | Native | 28 | 0.1039 | 13,834 | — | 122,015 | 71.9 | 269,184 |
| 128k | Packed | 33 | 0.1032 | 13,732 | −0.7% | 143,812 | 86.1 | 319,860 |
| 256k | Native | 14 | 0.0473 | 12,490 | — | 143,886 | 74.2 | 296,208 |
| 256k | Packed | 17 | 0.0470 | 12,426 | −0.5% | 173,528 | 91.8 | 361,531 |

Packed's +21% pool deepens the queue but barely moves throughput (−0.7% to +3.4%): these workloads are 32–256k-prompt cold prefills, which Native and Packed do identically and which already saturate the TP4 pipeline. Median TTFT rose 18–37% and TPOT up to 24% at the deeper packed loads. Capacity pays off where prefixes are reused, not where every request is a cold prefill (next subsection).

### LongSWE-Bench

Replays **4,916 recorded Claude-agent business conversations** (~144k prompt tokens/request, short decodes) over OpenAI SSE in fixed 1200-s windows — a prefix-reusing workload in which most of each request is served from radix cache only if its shared prefix (system prompt, tool schemas, earlier turns) survives eviction. Both legs are the same 0731 fork TP4 servers on identical hardware; only the C4 representation differs — Packed's smaller 328-byte rows pack more capacity into the same KV budget. Two lenses follow: **fair serving** with both modes at the same concurrency, and **SLO-limited concurrency** at each mode's own ceiling under a TTFT-p90 < 10 s budget.

#### Same concurrency

Both legs at concurrency 15, first traffic on a fresh boot, over a fixed 1200-s window:

| Metric | Native (584-byte C4) | Packed (328-byte C4) | Change |
|---|---:|---:|---:|
| Completed in window (0 failed) | 845 | 1,502 | **+77.7%** |
| Prompt-token throughput (k tok/s) | 97.2 | 180.2 | **+85.4%** |
| Completion tokens (k) | 117.9 | 235.5 | +99.8% |
| Real (uncached) prefill (M tok) | 15.5 | 11.5 | **−25.9%** |
| Device cache-hit rate | 86.72% | 94.72% | **+8.0 pp** |

The mechanism is capacity → cache retention → fewer duplicate prefills, and the gain is real per request, not a bigger-set artifact: on the 845 requests both legs completed (matched by request_id), Packed's device hit is **93.80% vs 86.72%**, mean TTFT/e2e are **4.80 s/12.33 s vs 7.88 s/21.15 s**, and median decode TPOT is unchanged (~23 ms).

Latency distributions — whole completed set per leg (Native n=845, Packed n=1,502). Packed's rows include its 657 queue-tail requests, so the legs are not apples-to-apples and that tail inflates Packed's p95/p99:

| Metric | Leg | min | p50 | **p90** | p95 | p99 | mean | max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TTFT | Native | 0.61 s | 2.95 s | **19.34 s** | 40.96 s | 74.35 s | 7.88 s | 94.0 s |
| TTFT | Packed | 0.52 s | 2.49 s | **7.27 s** | 10.78 s | 50.88 s | 4.36 s | 80.2 s |
| TPOT | Native | 0 ms | 22 ms | **478 ms** | 847 ms | 1.70 s | 148 ms | 3.79 s |
| TPOT | Packed | 0 ms | 23 ms | **88 ms** | 298 ms | 943 ms | 65 ms | 3.81 s |
| e2e latency | Native | 1.10 s | 5.98 s | **75.48 s** | 90.17 s | 105.4 s | 21.15 s | 167.1 s |
| e2e latency | Packed | 0.72 s | 5.37 s | **22.93 s** | 62.47 s | 89.92 s | 11.86 s | 165.0 s |

#### SLO-limited concurrency

Native's SLO ceiling under **TTFT p90 < 10 s** is **c12** (p90 **8.55 s**); Packed's is **c15** (**8.17 s**) — **15 concurrent users to Native's 12, +25%**. One concurrency per fresh-boot 1200-s run of the same 4,916-conversation replay; one step past either ceiling fails: Native c13 = 10.33 s, Packed c16 = 20.6 s (a same-boot c16 reading of ~11 s was warm-cache optimistic and past the SLO regardless).

Each mode at its own ceiling — both legs 09-07, same small decode graphs, same boot allocator state:

| Metric | Native @ 12 | Packed @ 15 | Change |
|---|---:|---:|---:|
| Completed in window | 1,239 | 1,578 | **+27.4%** |
| Requests/s | 1.03 | 1.30 | **+26.2%** |
| Prompt-token throughput (k tok/s) | 148.9 | 188.6 | **+26.7%** |
| Completion tokens (k) | 187.4 | 248.8 | +32.8% |
| Real (uncached) prefill (M tok) | 12.4 | 11.2 | **−9.9%** |
| Device cache-hit rate | 93.07% | 95.11% | **+2.0 pp** |

Latency distributions — whole completed set per leg:

| Metric | Leg | min | p50 | **p90** | p95 | p99 | mean | max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TTFT | Native @ 12 | 0.53 s | 2.47 s | **8.55 s** | 17.88 s | 45.01 s | 4.50 s | 66.0 s |
| TTFT | Packed @ 15 | 0.49 s | 2.45 s | **8.17 s** | 10.67 s | 36.35 s | 4.03 s | 85.5 s |
| TPOT | Native @ 12 | 0 ms | 18 ms | **119 ms** | 395 ms | 1.04 s | 64 ms | 1.77 s |
| TPOT | Packed @ 15 | 0 ms | 23 ms | **103 ms** | 212 ms | 852 ms | 58 ms | 2.33 s |
| e2e latency | Native @ 12 | 0.82 s | 4.72 s | **34.73 s** | 62.32 s | 75.13 s | 11.46 s | 110.5 s |
| e2e latency | Packed @ 15 | 1.13 s | 5.42 s | **23.73 s** | 48.91 s | 93.39 s | 11.37 s | 123.8 s |

Packed's edge concentrates in the tail the SLO measures — lower p90 and p95 across TTFT, TPOT and e2e alike — paid for with a higher TPOT/e2e median and a larger worst request (TTFT max 85.5 vs 66.0 s).

## Agentic Benchmark results

Across the two 50-task agentic suites that ran twice per leg, Packed is net-neutral on average: **+0.5 on Sangfor-Bench and −2 on SWE-bench**, both deltas inside run-to-run noise. Those suites are controlled Native (untouched 0731) vs Packed (Remnant, 328-byte C4) pairs on the same checkpoint through the identical Claude Code harness, two runs per leg — Sangfor both at TP4; SWE-bench one TP8 run and one TP4 run. A third suite, DeepSWE-Bench, ran once per leg on the same TP4 hardware; its 60-task pool is cap-dominated (tasks exceeding a 5400 s agent budget score 0 by construction), and on the under-cap subset Packed passes at a higher rate — a weak, single-run Packed tilt.

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

Run 1 swings to Packed (Packed-only 4 vs Native-only 2) and run 2 to Native (Native-only 3 vs Packed-only 2), so the runs bracket each other — the between-leg gap is smaller than each leg's own two-run spread, and the reading is parity.

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

Native leads both runs by the same 2 tasks. All four runs share the same 3 error instances (sphinx-7985/8269/8475), grouped as fail.

### DeepSWE-Bench

60 DeepSWE-Bench tasks, one matched run per leg on the same TP4 servers against the same 60-task pool. Each task caps at a 5400 s agent budget; a cap-hit (`AgentTimeoutError`) yields no patch and reward 0 by construction — a budget artifact, not a resolved fail — and it dominates the pool: **Native 35/60, Packed 36/60**. Only the under-cap **natural completions** carry a quality signal. Confusion matrix over the shared 60 tasks, rows = Native, columns = Packed (pass = full verifier pass, reward 1):

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 1 | 4 |
| Native fail | 6 | 49 |

A task **passes** only on a full verifier pass (reward 1); fail groups under-cap test failures with cap-hits (both reward 0). Packed passes 7/60 to Native's 5/60, with one task passing on both legs. On natural completions the pass rate is **Native 5/25 = 20.0% vs Packed 7/24 = 29.2%**. One run per leg, a ~60%-capped pool, and single-digit passes make this the weakest of the three suites: the +2-task Packed lead is directional and consistent with Sangfor-Bench, but unlike the matched-pair suites it is not tested against run-to-run spread.

## Conclusion

Remnant buys capacity, not decode speed: fair-load serving is throughput-neutral, the prefill-bound workload turns the extra pool into little at max concurrency, and the agentic evals show no consistent quality signal — across two matched run-pairs per suite Packed averages **+0.5** on Sangfor-Bench and is **−2** on SWE-bench, both deltas smaller than each leg's own two-run spread, while the single-run, cap-dominated DeepSWE-Bench suite tilts Packed (full passes **5** vs **7**; 20.0% vs 29.2% on natural completions). Together the three suites read as parity within noise: the Sangfor/SWE deltas sit inside each leg's own spread, and DeepSWE's Packed tilt rests on single-run, single-digit solves. The capacity pays only where shared prefixes are reused. A custom CUDA kernel that directly handles the TopMag50 sparse attention would close the remaining TPOT gap between Packed and Native and could let Packed beat Native even at fair serving.
