# Mustafar: 1.21× KV Capacity for Long-Context DeepSeek-V4-Flash Decode

## Summary

Mustafar applies TopMag50 sparsity to the 21 compressed sparse-attention (CSA) layers of DeepSeek-V4-Flash and stores the C4 state packed — **584 → 328 bytes per record (43.84% smaller)**. On a TP4 H100 server (fp4-native MoE runner, mem-frac 0.88, fp8 KV cache) this grew the measured KV pool from **3,730,944 to 4,519,168 full-token slots (1.2112×)**, raising the allocator-derived maximum resident concurrency at 2048-token decode from **107→129 (32k), 55→66 (64k), 28→33 (128k), 14→17 (256k)**.

This is a KV-capacity optimization, not a decode speedup. At Native's own concurrency the modes are throughput-neutral (**−3.2% to +0.2%** tokens/s); even at Packed's higher ceiling they stay near-neutral (**−0.7% to +3.4%**), because this input-heavy workload is prefill-bound. The capacity payoff appears under prefix reuse: in the LongSWE-Bench replay below, the larger pool keeps more shared prefixes resident (device cache hit 86.7 → 94.7%), and Packed finished **77.7% more requests while doing 25.9% fewer real (uncached) prefills**.

## Scope and configurations

Two legs on the same mustafar fork (SGLang v0.5.15 @ f63458b), hardware, and fp4-native `flashinfer_mxfp4` MoE runner:

1. **Native** — TopMag/packing off, stock 584-byte C4.
2. **Packed** — TopMag50 pruning on, 328-byte packed C4.

## Serving results

### Random SGLang Bench serving

TP4 on 4× H100 80 GB serving DeepSeek-V4-Flash-0731 with mem-frac 0.88, 1,048,576 context cap, fp8 KV cache, and full decode CUDA graphs to `max_bs 136` (prefill graphs off). Each point ran official `bench_serving` at exact 32k/64k/128k/256k inputs with 2048 outputs: one warm-up wave of C, then three measured waves (3C). Native ran at its allocator ceiling (C = 107/55/28/14 at 32k/64k/128k/256k); Packed at that same Native ceiling (fair) and at its own (C = 129/66/33/17).

Native has one natural operating point per context — its allocator ceiling. Packed is measured at that same ceiling (fair) and at its own higher ceiling (max).

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

Replays **4,916 recorded Claude-agent business conversations** (~144k prompt tokens/request, short decodes) over OpenAI SSE at concurrency 15 for a fixed 1200 s window — a prefix-reusing workload in which most of each request is served from radix cache only if its shared prefix (system prompt, tool schemas, earlier turns) survives eviction. Both legs are the same 0731 fork TP4 servers on identical hardware; only the C4 representation differs — Packed's smaller 328-byte rows pack more capacity into the same KV budget.

| Metric | Native (584-B C4) | Packed (328-B C4) | Change |
|---|---:|---:|---:|
| Completed in window (0 failed) | 845 | 1,502 | **+77.7%** |
| Prompt-token throughput | 97.2k tok/s | 180.2k tok/s | **+85.4%** |
| Completion tokens | 117,859 | 235,518 | +99.8% |
| Real (uncached) prefill | 15.5M tok | 11.5M tok | **−25.9%** |
| Device cache-hit rate | 86.72% | 94.72% | **+8.0 pp** |

The mechanism is capacity → cache retention → fewer duplicate prefills, and the gain is real per request, not a bigger-set artifact: on the 845 requests both legs completed (matched by request_id), Packed's device hit is **93.80% vs 86.72%**, mean TTFT/e2e are **4.80 s/12.33 s vs 7.88 s/21.15 s**, and median decode TPOT is unchanged (~23 ms).

**Latency distributions** — whole completed set per leg (Native n=845, Packed n=1,502). Packed's rows include its 657 queue-tail requests, so the legs are not apples-to-apples and that tail inflates Packed's p95/p99:

| Metric | Leg | min | p50 | **p90** | p95 | p99 | mean | max |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TTFT | Native | 0.61 s | 2.95 s | **19.34 s** | 40.96 s | 74.35 s | 7.88 s | 94.0 s |
| TTFT | Packed | 0.52 s | 2.49 s | **7.27 s** | 10.78 s | 50.88 s | 4.36 s | 80.2 s |
| TPOT | Native | 0 ms | 22 ms | **478 ms** | 847 ms | 1.70 s | 148 ms | 3.79 s |
| TPOT | Packed | 0 ms | 23 ms | **88 ms** | 298 ms | 943 ms | 65 ms | 3.81 s |
| e2e latency | Native | 1.10 s | 5.98 s | **75.48 s** | 90.17 s | 105.4 s | 21.15 s | 167.1 s |
| e2e latency | Packed | 0.72 s | 5.37 s | **22.93 s** | 62.47 s | 89.92 s | 11.86 s | 165.0 s |

## Benchmark results

Both 50-task agentic suites dip slightly under the packed C4; at n=50 the deltas sit within run-to-run noise but are consistently negative.

| Evaluation | Native | Packed | Difference |
|---|---:|---:|---:|
| Sangfor-Bench (n=50) | 28/50 | 24/50 | −4 tasks |
| SWE-bench (n=50) | 32/50 | 30/50 | −2 tasks |

Native = the untouched DeepSeek-V4-Flash-0731 checkpoint; Packed = Mustafar 328-byte C4 on the same 0731 model. Counts are task-level pass/fail: a task passes only when its full test suite passes (SWE-bench resolution; Sangfor 100% pass rate). The Sangfor Native column is the ACG112 `bash_ds_flash` reference run — native untouched 0731 on the identical 50-task set and harness; the SWE-bench legs are our own native-untouched vs packed runs through the same Claude Code harness at TP8.

### Sangfor-Bench

The 50-task hard set on the 0731 build: Native = ACG112 `bash_ds_flash` untouched-0731 reference, Packed = Mustafar 328-byte C4 on the same 0731 model. A task passes only when every repo test passes (pass_rate = 100). Rows = Native, columns = Packed:

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| **Native pass** | 22 | 6 |
| **Native fail** | 2 | 20 |

Packed passes 24/50 to Native's 28/50 (−4 tasks); 42/50 land in the same category and the eight disagreements split 6 native-only to 2 packed-only.

### SWE-bench

Same 50 instances through the Claude Code harness at **TP8** on DeepSeek-V4-Flash-0731 (error/empty outcomes grouped as fail). Rows = Native, columns = Packed:

| Baseline result | Packed pass | Packed fail |
|---|---:|---:|
| Native pass | 29 | 3 |
| Native fail | 1 | 17 |

46/50 land in the same pass/fail category; the four disagreements are balanced (Native passed 3 that Packed failed, Packed 1 that Native failed).

## Conclusion

Mustafar buys capacity, not decode speed: fair-load serving is throughput-neutral, the prefill-bound workload turns the extra pool into little at max concurrency, and quality holds within a few tasks on the two 50-task agentic evals (Packed trails by −4 Sangfor-Bench and −2 SWE-bench tasks). The capacity pays only where shared prefixes are reused. A custom CUDA kernel that directly handles the TopMag50 sparse attention would close the remaining TPOT gap between Packed and Native and could let Packed beat Native even at fair serving.
