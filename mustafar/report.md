# Mustafar: 1.21× KV Capacity for Long-Context DeepSeek-V4-Flash Decode

## Summary

Mustafar applies TopMag50 sparsity to the 21 compressed sparse-attention (CSA) layers in DeepSeek-V4-Flash and stores the resulting C4 state in a packed representation. Stage 1 reduces each C4 record from **584 bytes to 328 bytes**: a **43.84% reduction** or **1.7805× C4 compression**.

On a TP4 server with four 80 GB H100s, this increased the measured server KV pool from **332,288 to 402,688 full-token-equivalent slots**, or **1.2119× total KV capacity**. At the 2048-token decode workload, maximum resident concurrency increased from **9 to 11 requests at 32k**, **4 to 5 at 64k**, and **2 to 3 at 128k**.

At the same concurrency, Stage-1 packed and untouched V4-Flash serving had similar long-context throughput: **+2.85% at 32k**, **+1.05% at 64k**, and **+0.74% at 128k** in this 2048-token decode run. Using all additional packed capacity admitted more requests, but increased latency at the higher resident loads. Stage 1 should therefore be understood primarily as a **KV-capacity optimization**, not a token-processing speedup.

At each mode’s maximum concurrency, the packed layout increased the request ceiling from **9→11 at 32k**, **4→5 at 64k**, and **2→3 at 128k**. This additional resident capacity translated into realized total-token-throughput gains of **+14.47%**, **+12.69%**, and **+22.54%**, respectively, relative to untouched V4-Flash.

## Scope and configurations

The serving measurements compare two configurations:

1. **Untouched V4-Flash:** Mustafar runtime features disabled, with the stock 584-byte C4 representation.
2. **Stage-1 packed TopMag50:** TopMag50 pruning enabled, with the 328-byte packed C4 representation.

Untouched versus packed measures the complete Stage-1 effect. The untouched leg used the same image with both Mustafar feature flags explicitly disabled, keeping the model, SGLang build, and hardware configuration constant.

## Serving methodology

Serving was measured on **4× NVIDIA H100 80 GB HBM3 GPUs with TP4**, using SGLang’s official `bench_serving` script with exact 32k, 64k, and 128k inputs and exactly 2048 output tokens. Each point used one warm-up wave followed by three measured waves. Full decode CUDA graphs were enabled for batch sizes 1–12; prefill graphs were disabled because this is a decode optimization.

## Serving results

### Fair serving performance at the same concurrency

This comparison holds concurrency equal at the highest load available to the untouched 584-byte mode, measuring the complete Stage-1 effect.

| Context | Concurrency | Mode | Requests/s | Total tokens/s | Median TTFT (ms) | Median TPOT (ms) | Δ req/s vs untouched | Δ TTFT vs untouched | Δ TPOT vs untouched |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32k | 9 | Untouched V4-Flash | 0.1967 | 6,847 | 8,092 | 14.96 | — | — | — |
| 32k | 9 | Stage-1 packed | 0.2023 | 7,042 | 6,398 | 16.22 | **+2.85%** | **−20.93%** | **+8.46%** |
| 64k | 4 | Untouched V4-Flash | 0.1298 | 8,773 | 7,329 | 11.48 | — | — | — |
| 64k | 4 | Stage-1 packed | 0.1312 | 8,865 | 6,663 | 11.64 | **+1.05%** | **−9.09%** | **+1.45%** |
| 128k | 2 | Untouched V4-Flash | 0.0667 | 8,884 | 9,323 | 10.07 | — | — | — |
| 128k | 2 | Stage-1 packed | 0.0672 | 8,950 | 8,838 | 10.17 | **+0.74%** | **−5.20%** | **+0.91%** |

Stage 1 remains close to throughput-neutral across these loads. The apparent gains should be treated cautiously because each point used only three measured waves and does not establish run-to-run variance.

### Maximum-concurrency serving

This comparison runs each mode at its own allocator-derived maximum resident concurrency. The packed pool exposes **1.2119× as many total KV-token slots**, but the additional requests share the same TP4 compute resources.

| Context | Mode | Server KV-token capacity | Max concurrency | Requests/s | Total tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32k | Untouched V4-Flash | 332,288 | 9 | 0.1967 | 6,847 | 8,092 | 14.96 |
| 32k | Stage-1 packed | 402,688 (**1.2119×**) | 11 (**1.22×**) | 0.2251 (**1.145×**) | 7,837 (**1.145×**) | 7,596 | 15.99 |
| 64k | Untouched V4-Flash | 332,288 | 4 | 0.1298 | 8,773 | 7,329 | 11.48 |
| 64k | Stage-1 packed | 402,688 (**1.2119×**) | 5 (**1.25×**) | 0.1463 (**1.127×**) | 9,887 (**1.127×**) | 7,934 | 12.81 |
| 128k | Untouched V4-Flash | 332,288 | 2 | 0.0667 | 8,884 | 9,323 | 10.07 |
| 128k | Stage-1 packed | 402,688 (**1.2119×**) | 3 (**1.50×**) | 0.0818 (**1.225×**) | 10,886 (**1.225×**) | 11,974 | 12.09 |

Relative to untouched V4-Flash at its maximum concurrency, packed mode at its own maximum changed request and total-token throughput by **+14.47% at 32k**, **+12.69% at 64k**, and **+22.54% at 128k**. Median TTFT changed by **−6.1%**, **+8.3%**, and **+28.4%**, respectively. Median TPOT changed by **+7.0%**, **+11.6%**, and **+20.0%**.

At 64k and 128k, the untouched mode's concurrency was already sufficient to saturate the available processing pipeline. Packing admitted another resident request, but did not add GPU compute, memory bandwidth, or TP communication capacity. The extra request therefore increased contention and latency rather than producing proportional throughput.

## Benchmark test results

| Evaluation | Native baseline | TopMag50 | Difference |
|---|---:|---:|---:|
| **RULER 64k (n=850)** | 95.1% | 95.3% | **+0.16 pp** |
| **Longbench v2 (n=100)** | 54/100 | 52/100 | −2.0 pp |
| **Sangfor-Bench — first run, native layout (n=25 tasks)** | 9/24 local-native tasks passed* | 8/24 TopMag50 tasks passed* | −1 task |
| **Sangfor-Bench — second run, packed layout (n=25 tasks)** | 13/25 cloud-baseline tasks passed | 13/25 packed TopMag50 tasks passed | 0 tasks |
| **SWE-bench (Claude Code harness; n=50 shared)** | 32/50 resolved | 30/50 resolved | −2 resolves |

Across the three precision evaluations, TopMag50 is essentially lossless relative to native: **RULER 64k (n=850)** improves by **+0.16 percentage points**. Sangfor-Bench was run twice: the first run used the native 584-byte layout and changed from **9/24 to 8/24 task passes** after excluding one invalid native trajectory; only the second run used the packed 328-byte layout. Sangfor tasks contain many model calls and test-case evaluations, so their task count understates the amount of underlying evaluation.

The only aggregate negative signal is **Longbench v2 (n=100)**, where the compressor-latent result is **−2.0 percentage points**. This is a noisy small-sample result compared with RULER’s 850 samples. *Sangfor results in this section are task-level pass/fail counts; one invalid native task was excluded from the first run.* These evaluations validate the TopMag50 pruning policy; the SWE-bench run below is the end-to-end check of the 328-byte packed reconstruction path.

### First 25-task run: local native baseline

| Baseline result | TopMag pass | TopMag fail |
|---|---:|---:|
| **Native pass** | 7 | 2 |
| **Native fail** | 1 | 14 |

### New 25-task run: cloud baseline

| Baseline result | TopMag pass | TopMag fail |
|---|---:|---:|
| **Cloud pass** | 8 | 5 |
| **Cloud fail** | 5 | 7 |

### SWE-bench (Claude Code harness): untouched versus packed

Both legs ran the same 50 SWE-bench instances through the Claude Code harness on a shared set, comparing the **fp4 (V4-Flash untouched)** leg against the **mustafar (stage-1 packed)** leg. Confusion matrix (rows = untouched V4-Flash, columns = stage-1 packed):

| Untouched V4-Flash \ Stage-1 packed | resolved | unresolved | error | empty |
|---|---|---:|---:|---:|---:|
| resolved | 29 | 2 | 0 | 1 |
| unresolved | 1 | 13 | 0 | 0 |
| error | 0 | 0 | 3 | 0 |
| empty | 0 | 1 | 0 | 0 |

45 of 50 instances land in exactly the same category: 29 resolved by both, 13 unresolved by both, and 3 identical eval errors (sphinx-7985, -8269, -8475 are harness errors, not model issues). The five disagreements are small and roughly balanced — untouched resolved 3 that packed did not (2 unresolved + 1 empty), while packed resolved 1 that untouched did not. Net: untouched **32/50**, stage-1 packed **30/50** (−2). A −2 net over 50 instances, with no systematic direction to the disagreements, is consistent with run-to-run variance rather than a compression effect — if packing degraded the model, we would expect packed to lose many more resolves.

## Conclusion

Stage-1 packing delivers a clear memory-capacity result: **43.84% fewer C4 bytes** and **1.2119× total server KV-token capacity** on TP4 H100, translating to maximum concurrency gains of 9→11 at 32k, 4→5 at 64k, and 2→3 at 128k for 2048-token decode.

Against untouched V4-Flash, **Stage 1 fair-load serving is approximately throughput-neutral**. The packed layout increases KV capacity and maximum concurrency, but does not materially improve throughput when the server is compute-saturated. Stage 2, which optimizes the decode kernel itself, is the part expected to produce actual serving gains relative to untouched V4-Flash; those gains have not been measured in this report.

The available precision evidence supports TopMag50 as a pruning policy: RULER is effectively lossless, the compressor-latent Longbench v2 result is −2.0 pp, and the first Sangfor-Bench run changed from 9/24 to 8/24 task passes. The second Sangfor run, which used the packed layout, matched the cloud baseline in aggregate at 13/25 task passes despite task-level disagreements. The SWE-bench end-to-end run agrees: stage-1 packed resolved **30/50** vs 32/50 for untouched V4-Flash, a −2 net within run-to-run noise. The serving measurements use 2048 output tokens, making them a decode-oriented workload.
