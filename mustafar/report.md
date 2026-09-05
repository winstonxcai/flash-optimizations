# Mustafar: 1.21× KV Capacity for Long-Context DeepSeek-V4-Flash Decode

## Summary

Mustafar applies TopMag50 sparsity to the 21 compressed sparse-attention (CSA) layers in DeepSeek-V4-Flash and stores the resulting C4 state in a packed representation. Packed reduces each C4 record from **584 bytes to 328 bytes**: a **43.84% reduction** or **1.7805× C4 compression**.

On a TP4 server with four 80 GB H100s running DeepSeek-V4-Flash-0731 with the fp4-native MoE runner (mem-frac 0.88, fp8 KV cache), this increased the measured server KV pool from **3,730,944 to 4,519,168 full-token-equivalent slots**, or **1.2112× total KV capacity**. At the 2048-token decode workload, the allocator-derived maximum resident concurrency increased from **107 to 129 requests at 32k**, **55 to 66 at 64k**, **28 to 33 at 128k**, and **14 to 17 at 256k**.

At the same concurrency (Native’s allocator ceiling), Native and Packed serving were throughput-neutral in this 2048-token decode run: **−2.0%, +1.6%, −1.0%, and −0.4%** total tokens/s at 32k, 64k, 128k, and 256k. Packed should therefore be understood primarily as a **KV-capacity optimization**, not a token-processing speedup.

At each mode’s maximum concurrency, Packed admitted 1.18–1.21× more requests (matching its pool) but delivered only **+1.2%, +4.9%, +0.3%, and +0.5%** higher total-token throughput at 32k–256k, with higher median TTFT and TPOT at the deeper packed loads. This workload is prefill-bound — each request carries a 32–256k prompt and just 2048 output tokens — so the TP4 pipeline is already saturated before the extra capacity can be turned into throughput. The capacity payoff appears where prefixes are reused: in the real-workload replay below, the larger pool keeps more shared prefixes resident, raising the device cache-hit rate and cutting duplicate prefills.

## Scope and configurations

The serving measurements compare two configurations:

1. **Native:** Mustafar runtime features disabled, with the stock 584-byte C4 representation.
2. **Packed:** TopMag50 pruning enabled, with the 328-byte packed C4 representation.

Native versus Packed measures the complete packed-layout effect. The Native leg used the same image with both Mustafar feature flags explicitly disabled, keeping the model, SGLang build, hardware, and MoE runner (fp4-native `flashinfer_mxfp4`) constant.

## Serving methodology

Serving was measured on **4× NVIDIA H100 80 GB HBM3 GPUs with TP4** serving DeepSeek-V4-Flash-0731 on the mustafar fork of SGLang (v0.5.15 @ f63458b) with the fp4-native MoE runner (`--moe-runner-backend flashinfer_mxfp4`), mem-frac 0.88, context cap 1,048,576, and fp8 KV cache. Each point used SGLang’s official `bench_serving` script with exact 32k, 64k, 128k, and 256k inputs and exactly 2048 output tokens: one warm-up wave of C requests followed by three measured waves (3C). The Native leg ran at its allocator-derived ceiling — native pool 3,730,944 slots → C = 107/55/28/14 at 32k/64k/128k/256k; the Packed leg ran one point at that same Native ceiling and one at its own ceiling — packed pool 4,519,168 slots → C = 129/66/33/17. Full decode CUDA graphs were enabled through `max_bs 136` on both legs, covering every admitted batch; prefill graphs were disabled because this is a decode optimization.

## Serving results

Both legs run the same fp4-native 0731 stack (see methodology); Native's pool is 3,730,944 slots and Packed's 4,519,168 (**1.2112×**). Native has one natural operating point per context — its allocator ceiling (C = 107/55/28/14 at 32k/64k/128k/256k). Packed was measured both at that same Native ceiling (the "fair" comparison) and at its own ceiling (C = 129/66/33/17, the "max" comparison).

### Fair serving performance at the same concurrency

This comparison holds concurrency equal at the Native 584-byte mode's allocator ceiling, measuring the complete packed-layout effect without admitting more requests.

| Context | Concurrency | Mode | Requests/s | Total tokens/s | Median TTFT (ms) | Median TPOT (ms) | Median e2e (ms) | Δ tokens/s vs Native |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32k | 107 | Native | 0.3360 | 11,697 | 123,382 | 80.5 | 288,047 | — |
| 32k | 107 | Packed | 0.3293 | 11,467 | 114,286 | 89.7 | 336,650 | −2.0% |
| 64k | 55 | Native | 0.1787 | 12,076 | 87,879 | 100.5 | 294,033 | — |
| 64k | 55 | Packed | 0.1816 | 12,271 | 99,291 | 97.1 | 295,598 | +1.6% |
| 128k | 28 | Native | 0.1029 | 13,693 | 122,740 | 72.9 | 272,146 | — |
| 128k | 28 | Packed | 0.1018 | 13,551 | 122,701 | 74.4 | 274,768 | −1.0% |
| 256k | 14 | Native | 0.0468 | 12,369 | 144,642 | 75.3 | 299,193 | — |
| 256k | 14 | Packed | 0.0466 | 12,318 | 144,542 | 76.0 | 300,412 | −0.4% |

Packed is throughput-neutral at every context (deltas within ±2%). The latency columns are the noisy read: per-context signs mix — 32k shows a lower median TTFT but a higher TPOT and e2e, 64k a higher median TTFT but a comparable e2e — and each point used three measured waves, which does not establish run-to-run variance.

### Maximum-concurrency serving

This comparison runs each mode at its own allocator-derived maximum resident concurrency. Packed's pool exposes **1.2112× as many KV-token slots**, so it admits ~1.2× more resident requests, but those requests share the same TP4 compute.

| Context | Mode | Max concurrency | Requests/s | Total tokens/s | Δ tokens/s vs Native | Median TTFT (ms) | Median TPOT (ms) | Median e2e (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32k | Native | 107 | 0.3360 | 11,697 | — | 123,382 | 80.5 | 288,047 |
| 32k | Packed | 129 | 0.3401 | 11,841 | +1.2% | 152,028 | 93.7 | 328,838 |
| 64k | Native | 55 | 0.1787 | 12,076 | — | 87,879 | 100.5 | 294,033 |
| 64k | Packed | 66 | 0.1874 | 12,666 | +4.9% | 119,464 | 101.1 | 325,485 |
| 128k | Native | 28 | 0.1029 | 13,693 | — | 122,740 | 72.9 | 272,146 |
| 128k | Packed | 33 | 0.1032 | 13,732 | +0.3% | 143,812 | 86.1 | 319,859 |
| 256k | Native | 14 | 0.0468 | 12,369 | — | 144,642 | 75.3 | 299,193 |
| 256k | Packed | 17 | 0.0470 | 12,426 | +0.5% | 173,528 | 91.8 | 361,531 |

Relative to Native at its ceiling, Packed at its own maximum changed total-token throughput by **+1.2% at 32k**, **+4.9% at 64k**, **+0.3% at 128k**, and **+0.5% at 256k** while admitting 1.18–1.21× the requests. Median TTFT rose by 17–36% and median TPOT by up to 22% at the deeper packed loads.

These workloads are prefill-bound: each request is a 32–256k prompt followed by only 2048 output tokens, so nearly all GPU work is prompt prefill, which Native and Packed perform identically. At the deep concurrencies the fp4-native pool enables (C ≥ 14 even at 256k), the TP4 pipeline is already saturated, so a +21% pool cannot be converted into proportional throughput — it only deepens the queue. The capacity advantage shows up where prefixes are reused rather than where every request is a cold prefill (next subsection).

The one re-run to note: the original 256k Packed-max measurement OOM-crashed the server after ~2.8 h of server lifetime (CUDA memory fragmentation, not an intrinsic 256k-depth ceiling — the same absolute pool occupancy had already succeeded at 32k-max earlier in that boot). It was re-measured cleanly on a fresh boot at C = 17.

### Real-workload serving: 1200 s business-conversation replay

The random-input rows above exercise no prefix reuse. This run replays **4,916 recorded Claude-agent business conversations** (~144k prompt tokens/request on average, short decodes) over the OpenAI SSE protocol at fixed concurrency 15 for a fixed 1200 s window. Both legs are the same mustafar fork serving DeepSeek-V4-Flash-0731 (TP4, 4× H100 80 GB, fp4-native `flashinfer_mxfp4` MoE, mem-frac 0.88, 1M context cap, fp8 KV cache) on identical hardware; only the packed representation differs — Native (stock 584-byte C4, pool 3,730,944) versus Packed (TopMag50 328-byte C4, pool 4,519,168, **1.2112×**). Because the workload is heavily prefix-reusing, most of each request can be served from the radix cache only if its shared prefix (system prompt, tool schemas, earlier turns) survives eviction.

| Metric | Native (584-B C4) | Packed (328-B C4) | Change |
|---|---:|---:|---:|
| Completed in window (0 failed) | 845 | 1,502 | **+77.7%** |
| Request throughput | 0.703 req/s | 1.242 req/s | **+76.7%** |
| Prompt tokens served | 116.8M | 217.9M | +86.5% |
| Prompt-token throughput | 97.2k tok/s | 180.2k tok/s | **+85.4%** |
| Completion tokens | 117,859 | 235,518 | +99.8% |
| Real (uncached) prefill | 15.5M tok | 11.5M tok | **−25.9%** |
| Device cache-hit rate | 86.72% | 94.72% | **+8.0 pp** |
| TTFT mean / median | 7.88 s / 2.95 s | 4.36 s / 2.49 s | −45% / −16% |
| TPOT mean / median | 147.8 ms / 22.5 ms | 64.8 ms / 23.0 ms | −56% / +2% |
| e2e latency mean / median | 21.15 s / 5.98 s | 11.86 s / 5.37 s | −44% / −10% |

Both legs completed every request they started (0 failures). Packed finished **1,502 requests in the same window as Native's 845**. The mechanism is capacity → cache-retention → fewer duplicate prefills: the extra pool lets the radix cache keep more shared conversation prefixes resident between interleaved requests, device cache-hit rises from 86.72% to 94.72%, and Packed does **25.9% fewer real (uncached) prefill tokens while serving 77.7% more requests**. Packing is a KV-capacity optimization — it raises how many prefix tokens stay resident rather than reducing per-request attention work.

The cache-hit gain is real at the request level, not a completion-set artifact. On the **845 requests both legs completed** (byte-identical bodies, matched by request_id), Packed's device hit rate is **93.80% vs 86.72%**, mean TTFT **4.80 s vs 7.88 s**, and mean e2e latency **12.33 s vs 21.15 s** on identical requests. Median decode TPOT is unchanged (~23 ms), consistent with the benefit living in prefill/cache retention, not decode speed.

Latency rows in the table are each leg's own completed set — Packed also finished 657 queue-tail requests — so means and tail percentiles are directionally consistent but not apples-to-apples; the matched-subset rows above are the clean request-level read. An official-upstream run at the same concurrency reached 97.46% device hit but was excluded: it ran on a different GPU (H20-3e), so it is not hardware-comparable with these two H100 fork legs.

## Benchmark test results

| Evaluation | Native | Packed | Difference |
|---|---:|---:|---:|
| **RULER 64k (n=850)** | 95.1% | 95.3% | **+0.16 pp** |
| **Longbench v2 (n=100)** | 54/100 | 52/100 | −2.0 pp |
| **Sangfor-Bench (n=50)** | 22/49* | 21/49* | −1 task |
| **SWE-bench (n=50)** | 32/50 | 30/50 | −2 tasks |

Across the benchmark evaluations, quality differences are small but mixed: **RULER** improves by **+0.16 pp**, **Longbench v2** declines by **−2.0 pp**, combined **Sangfor-Bench** changes from **22/49 to 21/49** task passes, and **SWE-bench** changes from **32/50 to 30/50** task passes on the shared 0731 set.

The small negative task-level deltas in Sangfor-Bench and SWE-bench cannot be distinguished from run-to-run variance from these measurements. *Sangfor results are task-level pass/fail counts; the combined total is 9/24 + 13/25 for Native and 8/24 + 13/25 for Packed, with one invalid Native task excluded.*

### Sangfor-Bench

This matrix combines the two 25-task runs: one with a local Native baseline and one with a separate Native baseline. One invalid Native task is excluded.

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| **Native pass** | 15 | 7 |
| **Native fail** | 6 | 21 |

### SWE-bench

Both legs ran the same 50 SWE-bench instances through the Claude Code harness at **TP8** on a shared set using **DeepSeek-V4-Flash-0731**, comparing the **Native** configuration against the **Packed** configuration. Confusion matrix (rows = Native, columns = Packed):

*For this 2×2 view, error and empty outcomes are grouped as **fail**.*

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 29 | 3 |
| Native fail | 1 | 17 |

46 of 50 instances land in the same pass/fail category: 29 passed by both and 17 failed by both. The four disagreements are small and roughly balanced — Native passed 3 that Packed failed, while Packed passed 1 that Native failed. Net: Native **32/50**, Packed **30/50** (−2). This difference cannot be distinguished from run-to-run variance from this measurement alone.

## Conclusion

Packed delivers a clear memory-capacity result: **43.84% fewer C4 bytes** and **1.2112× total server KV-token capacity** on TP4 H100, translating to allocator-derived maximum concurrency of 107→129 at 32k, 55→66 at 64k, 28→33 at 128k, and 14→17 at 256k for 2048-token decode on the fp4-native stack.

Against Native, **Packed fair-load serving is approximately throughput-neutral** (−2.0% to +1.6% across contexts), and its extra capacity at maximum concurrency adds only **+0.3% to +4.9%** throughput because the input-heavy 2048-decode workload saturates the prefill pipeline first. The KV-capacity payoff appears under prefix reuse: in the 1200 s business-conversation replay, the larger pool keeps more shared prefixes resident, and Packed finished 77.7% more requests while doing 25.9% fewer real (uncached) prefills.

The quality evidence supports TopMag50 as a practical pruning policy: RULER is effectively lossless, Longbench v2 is **−2.0 pp**, combined Sangfor-Bench is **21/49 vs 22/49**, and SWE-bench is **30/50 vs 32/50** for Packed versus Native on the shared DeepSeek-V4-Flash-0731 set. These small end-to-end deltas should be interpreted alongside run-to-run variance. The serving measurements use 2048 output tokens, making them decode-oriented.
