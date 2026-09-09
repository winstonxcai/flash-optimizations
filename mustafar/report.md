# Mustafar: Residual Feature Sparsity Survives Latent Compression

## Overview

DeepSeek-V4-Flash already compresses its KV cache: the 21 compressed sparse-attention (CSA) layers cache each token's keys and values as a single learned latent — the 584-byte C4 state. Remnant prunes that latent a second time, keeping only the largest-magnitude ~50% of coordinates per token (TopMag50) and storing the survivors packed — **584 → 328 bytes per record (43.84% smaller)**. On a TP4 H100 server (fp4-native MoE runner, mem-frac 0.88, fp8 KV cache) this grew the measured KV pool from **3,730,944 to 4,519,168 full-token slots (1.2112×; allocator-reported, see Serving results)**, raising the allocator-derived maximum resident concurrency at 2048-token decode from **107→129 (32k), 55→66 (64k), 28→33 (128k), 14→17 (256k)**.

This is a KV-capacity optimization, not a decode speedup. At Native's own concurrency the modes are throughput-neutral (**−3.2% to +0.2%** tokens/s); even at Packed's higher ceiling they stay near-neutral (**−0.7% to +3.4%**), because this input-heavy workload is prefill-bound. Where prefixes are reused the payoff is now small: on the current SGLang v0.5.18 stack (LongSWE-Bench, below) both modes retain ~97% of the shared prefix in L1 and Packed's larger pool buys **+1 concurrent user** under the TTFT-p90 < 10 s SLO (Native C24 vs Packed C25), at ~+10% steady-state TPOT.

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

Packed's +21% pool deepens the queue but barely moves throughput (−0.7% to +3.4%): these workloads are 32–256k-prompt cold prefills, which Native and Packed do identically and which already saturate the TP4 pipeline. Median TTFT rose 18–37% and TPOT up to 24% at the deeper packed loads. Capacity pays off where prefixes are reused, not where every request is a cold prefill — measured under prefix reuse in the LongSWE-Bench section below (re-based to SGLang v0.5.18).

### LongSWE-Bench (SGLang v0.5.18)

Replays **4,916 recorded Claude-agent business conversations** (~144k prompt tokens/request, short decodes) over OpenAI SSE in fixed 1200-s windows. *Version note:* the serving-bench legs above ran SGLang v0.5.15 @ f63458b; LongSWE-Bench was re-measured after the Remnant fork re-based to **v0.5.18** (remnant container — Native = stock v0.5.18, Packed = the 328-byte fork). These v0.5.18 numbers **supersede** the earlier v0.5.15 LongSWE runs (SLO ceilings native c12 / packed c15 and the same-concurrency +77.7% reads). The v0.5.15 → v0.5.18 jump roughly doubled the concurrency either mode can hold, and on v0.5.18 both modes retain ~97% of the shared prefix in L1 — the capacity-driven retention gap of the old runs largely disappears. Two lenses follow: **same concurrency** at a shared concurrency both modes hold under the SLO, and **SLO-limited concurrency** at each mode's own ceiling under a TTFT-p90 < 10 s budget.

Every leg below is one **fresh server boot (empty radix** — no leftover prefix cache) serving the same 4,916-conversation replay for a fixed 1200-s window on the same TP4 H100 servers as above (fp4 `flashinfer_mxfp4`, fp8 KV, 1,048,576 context, mem-frac 0.88; small decode graphs at C≤15, extended at C>15; prefill graphs off). The SLO-limited lens climbs +1 C per config and stops at the first window with TTFT p90 ≥ 10 s; the ceiling is the last passing C.

#### Same concurrency

Both modes at **concurrency 21**, a shared C from the SLO sweep chosen so the two legs compare at identical offered load while still inside the TTFT-p90 < 10 s budget (Native p90 9.03 s, Packed 8.74 s) — and, unlike C22–24, clear of the near-ceiling legs that the SLO-limited lens below uses. Each mode contributes one fresh-boot 1200-s window; the other shared C16–24 legs read the same (ladder below):

| Metric | Native (584-byte C4) | Packed (328-byte C4) | Change |
|---|---:|---:|---:|
| Completed in window | 2,248 | 2,187 | −2.7% |
| Prompt-token throughput (k tok/s) | 263.9 | 257.2 | −2.5% |
| Completion tokens (k) | 343.5 | 329.0 | −4.2% |
| Real (uncached) prefill (M tok) | 8.43 | 8.13 | −3.6% |
| Device cache-hit rate | 97.36% | 97.39% | +0.03 pp |

With both modes retaining the shared prefix at the same ~97% rate, Packed's larger pool has nothing left to buy at equal load: completions and uncached prefill sit within one-window noise of Native's (~1.8 req/s either way), and its only consistent cost is decode. On the **2,184 requests both legs completed (matched by request_id)**, token-weighted device hit is **97.39% vs 97.40%**, mean TTFT/e2e are **5.96 s/11.21 s vs 5.94 s/11.49 s**, and median decode TPOT is **23 vs 25 ms** (+2 ms).

Latency distributions — whole completed set per leg (Native n=2,248, Packed n=2,187):

| Metric | Leg | min | p50 | p90 | p95 | p99 | mean | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT | Native | 0.76 s | 5.18 s | **9.03 s** | 10.67 s | 16.90 s | 5.96 s | 107.99 s |
| TTFT | Packed | 0.46 s | 5.07 s | **8.74 s** | 11.43 s | 17.81 s | 5.94 s | 107.26 s |
| TPOT | Native | 0 ms | 23 ms | **57 ms** | 91 ms | 596 ms | 45 ms | 2.75 s |
| TPOT | Packed | 0 ms | 25 ms | **61 ms** | 98 ms | 620 ms | 47 ms | 2.74 s |
| e2e latency | Native | 1.16 s | 7.99 s | **18.66 s** | 23.87 s | 74.43 s | 11.21 s | 163.59 s |
| e2e latency | Packed | 1.27 s | 8.01 s | **19.20 s** | 25.62 s | 84.67 s | 11.49 s | 162.82 s |

#### SLO-limited concurrency

Under the **TTFT-p90 < 10 s** budget Native's last passing concurrency is **24** (p90 **9.36 s**); Packed's is **25** (**9.40 s**) — **one more concurrent user, +4%**. One step past either ceiling fails, but with very different shape: Native c25 blows past at **12.10 s** (+2.1 s over budget, completions fall 2,212 → 2,029); Packed c26 crosses **softly** at **10.35 s** (+0.35 s over budget, throughput and device hit unchanged) — Packed is within a rounding of holding 26.

Each mode at its own ceiling, plus Packed’s first crossing (c26):

| Metric | Native @ 24 | Packed @ 25 | Packed @ 26 ✗ | Δ P25 vs N24 |
|---|---:|---:|---:|---:|
| Completed in window | 2,212 | 2,222 | 2,183 | +0.5% |
| Requests/s | 1.810 | 1.828 | 1.780 | +1.0% |
| Prompt-token throughput (k tok/s) | 258.4 | 260.4 | 252.5 | +0.8% |
| Completion tokens (k) | 337.0 | 333.0 | 323.2 | −1.2% |
| Real (uncached) prefill (M tok) | 8.44 | 8.18 | 8.27 | −3.1% |
| Device cache-hit rate | 97.30% | 97.40% | 97.31% | +0.10 pp |
| TTFT p90 | **9.36 s** | **9.40 s** | **10.35 s** | +0.04 s |

Latency distributions — whole completed set per leg (Native n=2,212; Packed n=2,222 @ 25, n=2,183 @ 26):

| Metric | Leg | min | p50 | p90 | p95 | p99 | mean | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TTFT | Native @ 24 | 0.53 s | 6.45 s | **9.36 s** | 11.55 s | 26.29 s | 7.27 s | 133.46 s |
| TTFT | Packed @ 25 | 0.62 s | 6.50 s | **9.40 s** | 11.84 s | 21.05 s | 7.17 s | 141.32 s |
| TTFT | Packed @ 26 ✗ | 0.55 s | 7.57 s | **10.35 s** | 12.29 s | 19.77 s | 8.09 s | 150.06 s |
| TPOT | Native @ 24 | 0 ms | 22 ms | **60 ms** | 116 ms | 717 ms | 52 ms | 3.42 s |
| TPOT | Packed @ 25 | 0 ms | 26 ms | **70 ms** | 123 ms | 769 ms | 57 ms | 3.63 s |
| TPOT | Packed @ 26 ✗ | 0 ms | 26 ms | **67 ms** | 114 ms | 845 ms | 58 ms | 3.86 s |
| e2e latency | Native @ 24 | 2.20 s | 9.19 s | **20.09 s** | 29.38 s | 134.36 s | 12.97 s | 174.57 s |
| e2e latency | Packed @ 25 | 1.53 s | 9.44 s | **20.92 s** | 27.37 s | 142.33 s | 13.44 s | 197.87 s |
| e2e latency | Packed @ 26 ✗ | 1.71 s | 10.54 s | **20.48 s** | 26.93 s | 151.13 s | 14.33 s | 207.33 s |

Ladder (TTFT p90 / mean TPOT per fresh-boot 1200-s window; ★ = SLO ceiling, ✗ = first crossing):

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
| 24 | 9.36 s ★ | 51.6 ms | 9.14 s | 54.8 ms |
| 25 | 12.10 s ✗ | 59.6 ms | 9.40 s ★ | 56.6 ms |
| 26 | — | — | 10.35 s ✗ | 57.9 ms |

Readings:

- **One concurrent user (24 → 25, +4%) — down from +25% (c12 vs c15) on v0.5.15.** On v0.5.18 both modes already hold ~97% of the shared prefix (device-hit rows above), so the +21% pool is decode headroom, not cache retention — which is why the old retention-gap story is gone.
- The p90 at the wall is set by the ~95% of requests served almost entirely from radix (~143k-token prompts, ≥95% device hit): their TTFT p90 is **8.97 s (Native @ 24) vs 8.92 s (Packed @ 25)**. The ~5% cold-start / low-hit requests sit far above the SLO in both modes (p90 ~45–68 s) and drive the p99/max tails without reaching p90 mass.
- **TPOT penalty at the ceiling ~+10%**: Packed mean 56.6 ms vs Native 51.6 ms (C25 vs C24); at matched C24 the p50 gap is 24.0 vs 22.3 ms (+8%). The packed c4 store/load cost stays modest and widens only slightly as occupancy rises.
- **Throughput is a plateau**: both modes complete ~2,150–2,250 conversations per 1200-s window across C≈20–26 (~1.8 req/s, ~260k prompt-tok/s) — decode-throughput-saturated. Users past ~24 add no completions; they only lift TTFT (past the ceiling, over the SLO).

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

