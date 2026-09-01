# Mustafar: 1.21× KV Capacity for Long-Context DeepSeek-V4-Flash Decode

## Summary

Mustafar applies TopMag50 sparsity to the 21 compressed sparse-attention (CSA) layers in DeepSeek-V4-Flash and stores the resulting C4 state in a packed representation. Stage 1 reduces each C4 record from **584 bytes to 328 bytes**: a **43.84% reduction** or **1.7805× C4 compression**.

On a TP4 server with four 80 GB H100s, this increased the measured server KV pool from **332,288 to 402,688 full-token-equivalent slots**, or **1.2119× total KV capacity**. At the 2048-token decode workload, maximum resident concurrency increased from **9 to 11 requests at 32k**, **4 to 5 at 64k**, and **2 to 3 at 128k**.

At the same concurrency, Stage-1 packed and untouched V4-Flash serving had similar long-context throughput: **+2.85% at 32k**, **+1.05% at 64k**, and **+0.74% at 128k** in this 2048-token decode run. Using all additional packed capacity admitted more requests, but increased latency at the higher resident loads. Stage 1 should therefore be understood primarily as a **KV-capacity optimization**, not a token-processing speedup.

At each mode’s maximum concurrency, the packed layout increased the request ceiling from **9→11 at 32k**, **4→5 at 64k**, and **2→3 at 128k**. This additional resident capacity translated into realized total-token-throughput gains of **+14.47%**, **+12.69%**, and **+22.54%**, respectively, relative to untouched V4-Flash.

## Scope and configurations

The serving measurements cover three configurations:

1. **Untouched V4-Flash:** Mustafar runtime features disabled, with the stock 584-byte C4 representation.
2. **TopMag50 native layout:** TopMag50 pruning enabled, with the 584-byte C4 representation.
3. **Stage-1 packed TopMag50:** the same TopMag50 policy, with the 328-byte packed C4 representation.

Untouched versus packed measures the complete Stage-1 effect. Untouched versus TopMag50 native-layout isolates pruning overhead, while TopMag50 native-layout versus packed isolates packing and reconstruction overhead. The untouched leg used the same image with both Mustafar feature flags explicitly disabled, keeping the model, SGLang build, and hardware configuration constant.

## Serving methodology

Serving was measured on **4× NVIDIA H100 80 GB HBM3 GPUs with TP4**, using SGLang’s official `bench_serving` script with exact 32k, 64k, and 128k inputs and exactly 2048 output tokens. Each point used one warm-up wave followed by three measured waves. Full decode CUDA graphs were enabled for batch sizes 1–12; prefill graphs were disabled because this is a decode optimization. The three legs ran in parallel and consumed **5.63 H100-hours total**: 1.74 untouched, 1.76 native, and 2.12 packed. Elapsed wall time was approximately 33 minutes, including startup.

## Serving results

### Fair serving performance at the same concurrency

This comparison holds concurrency equal at the highest load available to the 584-byte modes. It shows the complete Stage-1 effect against untouched V4-Flash and separates pruning from packing overhead.

| Context | Concurrency | Mode | Requests/s | Total tokens/s | Median TTFT (ms) | Median TPOT (ms) | Δ req/s vs untouched | Δ TTFT vs untouched | Δ TPOT vs untouched |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32k | 9 | Untouched V4-Flash | 0.1967 | 6,847 | 8,092 | 14.96 | — | — | — |
| 32k | 9 | TopMag50 native layout | 0.1945 | 6,772 | 8,240 | 14.99 | **−1.09%** | **+1.84%** | **+0.24%** |
| 32k | 9 | Stage-1 packed | 0.2023 | 7,042 | 6,398 | 16.22 | **+2.85%** | **−20.93%** | **+8.46%** |
| 64k | 4 | Untouched V4-Flash | 0.1298 | 8,773 | 7,329 | 11.48 | — | — | — |
| 64k | 4 | TopMag50 native layout | 0.1298 | 8,772 | 7,171 | 11.57 | **−0.01%** | **−2.15%** | **+0.78%** |
| 64k | 4 | Stage-1 packed | 0.1312 | 8,865 | 6,663 | 11.64 | **+1.05%** | **−9.09%** | **+1.45%** |
| 128k | 2 | Untouched V4-Flash | 0.0667 | 8,884 | 9,323 | 10.07 | — | — | — |
| 128k | 2 | TopMag50 native layout | 0.0669 | 8,903 | 9,121 | 10.11 | **+0.22%** | **−2.16%** | **+0.34%** |
| 128k | 2 | Stage-1 packed | 0.0672 | 8,950 | 8,838 | 10.17 | **+0.74%** | **−5.20%** | **+0.91%** |

Stage 1 remains close to throughput-neutral across these loads. The apparent gains should be treated cautiously because each point used only three measured waves and does not establish run-to-run variance.

### Maximum-concurrency serving

This comparison runs each mode at its own allocator-derived maximum resident concurrency. The packed pool exposes **1.2119× as many total KV-token slots**, but the additional requests share the same TP4 compute resources.

| Context | Mode | Server KV-token capacity | Max concurrency | Requests/s | Total tokens/s | Median TTFT (ms) | Median TPOT (ms) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 32k | Untouched V4-Flash | 332,288 | 9 | 0.1967 | 6,847 | 8,092 | 14.96 |
| 32k | TopMag50 native layout | 332,288 | 9 | 0.1945 | 6,772 | 8,240 | 14.99 |
| 32k | Stage-1 packed | 402,688 (**1.2119×**) | 11 (**1.22×**) | 0.2251 (**1.145×**) | 7,837 (**1.145×**) | 7,596 | 15.99 |
| 64k | Untouched V4-Flash | 332,288 | 4 | 0.1298 | 8,773 | 7,329 | 11.48 |
| 64k | TopMag50 native layout | 332,288 | 4 | 0.1298 | 8,772 | 7,171 | 11.57 |
| 64k | Stage-1 packed | 402,688 (**1.2119×**) | 5 (**1.25×**) | 0.1463 (**1.127×**) | 9,887 (**1.127×**) | 7,934 | 12.81 |
| 128k | Untouched V4-Flash | 332,288 | 2 | 0.0667 | 8,884 | 9,323 | 10.07 |
| 128k | TopMag50 native layout | 332,288 | 2 | 0.0669 | 8,903 | 9,121 | 10.11 |
| 128k | Stage-1 packed | 402,688 (**1.2119×**) | 3 (**1.50×**) | 0.0818 (**1.225×**) | 10,886 (**1.225×**) | 11,974 | 12.09 |

Relative to untouched V4-Flash at its maximum concurrency, packed mode at its own maximum changed request and total-token throughput by **+14.47% at 32k**, **+12.69% at 64k**, and **+22.54% at 128k**. Median TTFT changed by **−6.1%**, **+8.3%**, and **+28.4%**, respectively. Median TPOT changed by **+7.0%**, **+11.6%**, and **+20.0%**.

At 64k and 128k, native concurrency was already sufficient to saturate the available processing pipeline. Packing admitted another resident request, but did not add GPU compute, memory bandwidth, or TP communication capacity. The extra request therefore increased contention and latency rather than producing proportional throughput.

## Benchmark test results

| Evaluation | Native baseline | TopMag50 | Difference |
|---|---:|---:|---:|
| **RULER 64k (n=850)** | 95.1% | 95.3% | **+0.16 pp** |
| **Longbench v2 (n=100)** | 54/100 | 52/100 | −2.0 pp |
| **Sangfor-Bench (n=25 tasks)** | 92.7%* | 92.8%* | **+0.1 pp*** |

Across the three precision evaluations, TopMag50 is essentially lossless relative to native: **RULER 64k (n=850)** improves by **+0.16 percentage points**, and **Sangfor-Bench (n=25 tasks)** is effectively unchanged at **92.7% native versus 92.8% TopMag50**. Sangfor tasks contain many model calls and test-case evaluations, so its task count understates the amount of underlying evaluation.

The only negative signal is **Longbench v2 (n=100)**, where the compressor-latent result is **−2.0 percentage points**. This is a noisy small-sample result compared with RULER’s 850 samples. *Sangfor values are test-case-weighted across 24 non-degenerate tasks; one invalid native task was excluded.* These evaluations validate the TopMag50 pruning policy, not the 328-byte packed reconstruction path end to end.

## Conclusion

Stage-1 packing delivers a clear memory-capacity result: **43.84% fewer C4 bytes** and **1.2119× total server KV-token capacity** on TP4 H100, translating to maximum concurrency gains of 9→11 at 32k, 4→5 at 64k, and 2→3 at 128k for 2048-token decode.

Against untouched V4-Flash, **Stage 1 fair-load serving is approximately throughput-neutral**. The packed layout increases KV capacity and maximum concurrency, but does not materially improve throughput when the server is compute-saturated. Stage 2, which optimizes the decode kernel itself, is the part expected to produce actual serving gains relative to untouched V4-Flash; those gains have not been measured in this report.

The available precision evidence supports TopMag50 as a native-layout pruning policy: RULER is effectively lossless, the compressor-latent Longbench v2 result is −2.0 pp, and Sangfor-Bench is essentially unchanged at 92.7% versus 92.8%. Packed-layout end-to-end quality remains unvalidated. The serving measurements use 2048 output tokens, making them a decode-oriented workload.

## Appendix: Full Sangfor-Bench results

8 hard, 10 medium, and 7 easy tasks were rerun at `--context-length 262144`, leaving approximately 230k effective input tokens. Pass rates are based on one trajectory per task; mean rows are test-case-weighted.

| Task (# test cases) | Native (262k) | TopMag50 (262k) | Δ TopMag−native (pp) |
|---|---:|---:|---:|
| **HARD (8)** | | | |
| `sri_swe-bench_5f5a7df7` (116) | 92.2 | 90.5 | −1.7 |
| `sri_esecgpt_48486b59` (227) | 0.0* | 33.0 | +33.0 |
| `sri_esecgpt_80fa3321` (267), degenerate | 100.0 | 100.0 | 0.0 |
| `sri_s1_cec32c82` (176) | 97.2 | 98.9 | +1.7 |
| `sri_swe-bench_fea293e6` (86) | 76.7 | 70.9 | −5.8 |
| `sri_ap-gpt_2bcf1160` (164) | 82.3 | 79.3 | −3.0 |
| `tw_esecgpt_f291630` (243) | 100.0 | 100.0 | 0.0 |
| `sri_ap-gpt_d7527749` (137) | 54.0 | 78.8 | +24.8 |
| **Mean (8, test-case-weighted)** | **76.4** | **82.1** | **+5.7** |
| **Mean (7, excluding invalid native task)** | **89.4** | **91.5** | **+2.1** |
| **MEDIUM (10)** | | | |
| `gcjs_kube-log-check-recover_5b6a23ad` (254) | 97.6 | 96.9 | −0.7 |
| `sri_s1_00ce55e2` (126) | 89.7 | 93.7 | +4.0 |
| `aiyycp_sales-flow_d7329e44` (74) | 100.0 | 73.0 | −27.0 |
| `sri_chat-agent_86ce36d3` (62) | 96.8 | 93.5 | −3.3 |
| `sri_chat-agent_b2f8ec64` (75) | 98.7 | 100.0 | +1.3 |
| `sri_s1_d060bef0` (131) | 85.5 | 89.3 | +3.8 |
| `sri_esecgpt_cf8ba0fb` (268) | 100.0 | 100.0 | 0.0 |
| `fy_gptanalystagent_fb3d6a3d` (111) | 70.3 | 64.0 | −6.3 |
| `gcjs_go-zero_22ab9e7d` (48) | 100.0 | 100.0 | 0.0 |
| `sri_ap-gpt_0dd68d23` (122) | 98.4 | 94.3 | −4.1 |
| **Mean (10, test-case-weighted)** | **94.0** | **92.1** | **−2.0** |
| **EASY (7)** | | | |
| `gcjs_kube-log-check-recover_c6a12bfe` (122) | 95.1 | 95.1 | 0.0 |
| `gcjs_kube-log-check-recover_fc67bfda` (132) | 99.2 | 99.2 | 0.0 |
| `tw_esecgpt_4966005` (40) | 100.0 | 100.0 | 0.0 |
| `sri_chat-agent_035a16f0` (27) | 77.8 | 88.9 | +11.1 |
| `tw_esecgpt_6741243f` (40) | 100.0 | 100.0 | 0.0 |
| `gcjs_kube-log-check-recover_e04abbb7` (74) | 100.0 | 100.0 | 0.0 |
| `mss_drme-service_2a2095f8` (35) | 100.0 | 100.0 | 0.0 |
| **Mean (7, test-case-weighted)** | **97.2** | **97.9** | **+0.6** |
| **Running full average (24 tasks, excluding invalid native task)** | **92.7** | **92.8** | **+0.1** |

*The native `sri_esecgpt_48486b59` trajectory produced an invalid build and is excluded from the seven-task mean and running aggregate.*
