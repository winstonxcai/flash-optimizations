# Mustafar: 1.21× KV Capacity for Long-Context DeepSeek-V4-Flash Decode

## Summary

Mustafar applies TopMag50 sparsity to the 21 compressed sparse-attention (CSA) layers in DeepSeek-V4-Flash and stores the resulting state in a packed representation. Packed reduces each record from **584 bytes to 328 bytes**: a **43.84% reduction** or **1.7805× compression**.

On a TP4 server with four 80 GB H100s, this increased the measured server KV pool from **332,288 to 402,688 full-token-equivalent slots**, or **1.2119× total KV capacity**. At the 2048-token decode workload, maximum resident concurrency increased from **9 to 11 requests at 32k**, **4 to 5 at 64k**, and **2 to 3 at 128k**.

At the same concurrency, Native and Packed serving had similar long-context throughput: **+2.85% at 32k**, **+1.05% at 64k**, and **+0.74% at 128k** in this 2048-token decode run. Using all additional Packed capacity admitted more requests, but increased latency at the higher resident loads. Packed should therefore be understood primarily as a **KV-capacity optimization**, not a token-processing speedup.

At each mode’s maximum concurrency, Packed delivered **14–23% higher total-token throughput**, but with higher latency at 64k and 128k.

## Scope and configurations

The serving measurements compare two configurations:

1. **Native:** Mustafar runtime features disabled, with the stock 584-byte representation.
2. **Packed:** TopMag50 pruning enabled, with the 328-byte packed representation.

Native versus Packed measures the complete packed-layout effect. The Native leg used the same image with both Mustafar feature flags explicitly disabled, keeping the model, SGLang build, and hardware configuration constant.

## Serving methodology

Serving was measured on **4× NVIDIA H100 80 GB HBM3 GPUs with TP4**, using SGLang’s official `bench_serving` script with exact 32k, 64k, and 128k inputs and exactly 2048 output tokens. Each point used one warm-up wave followed by three measured waves. Full decode CUDA graphs were enabled for batch sizes 1–12; prefill graphs were disabled because this is a decode optimization.

## Serving results

### Fair serving performance at the same concurrency

This comparison holds concurrency equal at the highest load available to the Native 584-byte mode, measuring the complete packed-layout effect.

| Context | Concurrency | Mode | Requests/s | Total tokens/s | Median TTFT (ms) | Median TPOT (ms) | Δ req/s vs Native | Δ TTFT vs Native | Δ TPOT vs Native |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32k | 9 | Native | 0.1967 | 6,847 | 8,092 | 14.96 | — | — | — |
| 32k | 9 | Packed | 0.2023 | 7,042 | 6,398 | 16.22 | **+2.85%** | **−20.93%** | **+8.46%** |
| 64k | 4 | Native | 0.1298 | 8,773 | 7,329 | 11.48 | — | — | — |
| 64k | 4 | Packed | 0.1312 | 8,865 | 6,663 | 11.64 | **+1.05%** | **−9.09%** | **+1.45%** |
| 128k | 2 | Native | 0.0667 | 8,884 | 9,323 | 10.07 | — | — | — |
| 128k | 2 | Packed | 0.0672 | 8,950 | 8,838 | 10.17 | **+0.74%** | **−5.20%** | **+0.91%** |

Packed remains close to throughput-neutral across these loads. The apparent gains should be treated cautiously because each point used only three measured waves and does not establish run-to-run variance.

### Maximum-concurrency serving

This comparison runs each mode at its own allocator-derived maximum resident concurrency. The Packed pool exposes **1.2119× as many total KV-token slots**, but the additional requests share the same TP4 compute resources.

| Context | Mode | Server KV-token capacity | Max concurrency | Requests/s | Total tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32k | Native | 332,288 | 9 | 0.1967 | 6,847 | 8,092 | 14.96 |
| 32k | Packed | 402,688 (**1.2119×**) | 11 (**1.22×**) | 0.2251 (**1.145×**) | 7,837 (**1.145×**) | 7,596 | 15.99 |
| 64k | Native | 332,288 | 4 | 0.1298 | 8,773 | 7,329 | 11.48 |
| 64k | Packed | 402,688 (**1.2119×**) | 5 (**1.25×**) | 0.1463 (**1.127×**) | 9,887 (**1.127×**) | 7,934 | 12.81 |
| 128k | Native | 332,288 | 2 | 0.0667 | 8,884 | 9,323 | 10.07 |
| 128k | Packed | 402,688 (**1.2119×**) | 3 (**1.50×**) | 0.0818 (**1.225×**) | 10,886 (**1.225×**) | 11,974 | 12.09 |

Relative to Native at its maximum concurrency, Packed at its own maximum changed request and total-token throughput by **+14.47% at 32k**, **+12.69% at 64k**, and **+22.54% at 128k**. Median TTFT changed by **−6.1%**, **+8.3%**, and **+28.4%**, respectively. Median TPOT changed by **+7.0%**, **+11.6%**, and **+20.0%**.

At 64k and 128k, the Native mode's concurrency was already sufficient to saturate the available processing pipeline. Packing admitted another resident request, but did not add GPU compute, memory bandwidth, or TP communication capacity. The extra request therefore increased contention and latency rather than producing proportional throughput.

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

Packed delivers a clear memory-capacity result: **43.84% fewer bytes** and **1.2119× total server KV-token capacity** on TP4 H100, translating to maximum concurrency gains of 9→11 at 32k, 4→5 at 64k, and 2→3 at 128k for 2048-token decode.

Against Native, **Packed fair-load serving is approximately throughput-neutral**. The Packed layout increases KV capacity and maximum concurrency, but does not materially improve throughput when the server is compute-saturated.

The quality evidence supports TopMag50 as a practical pruning policy: RULER is effectively lossless, Longbench v2 is **−2.0 pp**, combined Sangfor-Bench is **21/49 vs 22/49**, and SWE-bench is **30/50 vs 32/50** for Packed versus Native on the shared DeepSeek-V4-Flash-0731 set. These small end-to-end deltas should be interpreted alongside run-to-run variance. The serving measurements use 2048 output tokens, making them decode-oriented.
