# Profiled replay comparison

Author: Winston Cai.

Update after inspection of all ten corrected traces: **the runs labelled Packed strongly indicate Native fallback inside spawned GPU workers. Treat the performance table below as requested labels, not verified implementations. Earlier suggestions of a Packed speedup are superseded by this diagnosis.** Replay success remains a separate, passing client gate.

These five runs used `scripts/local/bench-lswb.sh` for 1200 seconds each, with Nsight start/stop requests at 300/360 and 900/960 seconds. Capture was inactive during model loading and graph creation. `remote-run-record.json` indexes the canonical summaries, measured capture API acknowledgment times, source identity, and remote trace locations; `summary.csv` is derived from the canonical summaries.

The five profiled runs missed the 99% success requirement with `missing_output_event` failures. Their cause is unresolved. These are candidate profiled runs, not an acceptance speed comparison. For both Packed Triton and Packed Kernel, c28 reused that implementation’s c24 server/cache; Native restarted with the same token pool size. The earlier ten trace files are saved and nonempty but their contents have not been analyzed. The corrected Native traces are analyzed below.

Canonical small artifacts are in the five earlier profiled run directories, the corrected unprofiled Native directory, and three corrected profiled c24 directories. Raw requests/responses and ten large `.nsys-rep` files remain on the VDI at the paths recorded in JSON. Do not substitute this aggregate for those raw artifacts when diagnosing failures.

Reproduction entry point (after launching the chosen mode with `serve.sh <mode> profile` and configuring the replay environment):

```bash
bash scripts/local/bench-lswb.sh <tag> 30212 <concurrency> 1200 300:360,900:960
```

The corrected Native server was left running after replay, then observed stopped on the VDI after a chat pause. Subsequent profiling is user-operated. The last agent-observed GPU lease renewal expired at 12:42 Shanghai time on October 12, 2026; this is a historical observation, not current lease status.

The Kernel runs used parent source `34dc2e227e4747e4f235157919b8a14ede0a9fdf`, the same fork pins and Docker image as the earlier runs, and GPUs 0–3. Its reports 1/2 belong to c24 and 3/4 to c28. All four were verified nonempty after collection. Capture acknowledgment delay is recorded; trace boundaries remain to be inspected. The existing serving and replay scripts handled the launch and windows without code changes.

Diagnosis after comparison with `native-slo-c24/20260909_081547`: all five recent launches used decode graphs capped at batch 15, so c24/c28 decode fell back to eager. These runs are misconfigured evidence and cannot establish implementation speed. Original Native c24 measured 277.98 output tokens/s and 22.32 ms median TPOT with 99.19% success. Dataset and runner hashes match; the adapter hash changed and remains a comparison limitation. The correction defaults standalone serving to extended graphs and checks resolved server graph coverage before replay. Validation completed Native c24 unprofiled for 1200s using the existing entry point. The server was left running at completion.

Corrected unprofiled Native (`native-graph-c24/20261012_081956`) measured 277.27 output tokens/s, 22.79 ms median TPOT, and 132.44 ms p95 TPOT, with 2188/2209 successful requests (99.05%; valid). Relative to the historical control, throughput changed by −0.26%, median TPOT by +2.13%, and p95 TPOT by +14.41%. Live decode logs confirmed graphs at batch 24; recorded server configuration includes batches 24 and 28 and max batch 136. This supports keeping the graph correction. It does not establish profiling overhead or corrected Packed performance. The historical control used GPUs 4–7 and lacks source/host identity; its adapter hash also differs.

Corrected profiled Native (`native-graph-nsys-c24/20261012_100313`) measured 260.02 output tokens/s, 24.93 ms median TPOT, and 156.59 ms p95 TPOT, with 2037/2055 successful requests (99.12%; valid). Relative to corrected unprofiled Native, throughput is 6.22% lower, median TPOT 9.36% higher, and p95 TPOT 18.24% higher. Runner, adapter, dataset, and replay settings match. The random seed differs and the token pool is 3,735,296 versus 3,738,624. These results are consistent with profiling overhead, without isolating its causal effect. Both decode configurations cover c24/c28. Capture API acknowledgments were 308.80/384.73 and 901.45/985.82 seconds; actual captured intervals require inspection. Two LFS trace objects total 293,692,500 bytes (280.09 MiB). Both traces were downloaded and their LFS hashes verified, then exported locally with Nsight Systems 2026.5.1.161.

The corrected Native traces use graph-level CUDA tracing. There are 9,376 and 9,601 recorded `cudaGraphLaunch` calls across the four ranks. Within the captured intervals, each GPU recorded 2,344 graph executions in the first trace and about 2,400 in the second, averaging about 11.5–11.8 ms per graph. Graph kernels are not individually recorded, so the listed non-graph kernel totals cannot identify decode pack/unpack or attention cost inside graphs. SGLang all-reduce and CUTLASS grouped GEMM lead the separately recorded kernel totals, which sum GPU time across ranks rather than measure critical-path wall time.

Actual report durations are 52.445 and 60.124 seconds, beginning at 10:08:22 and 10:18:14 Shanghai time. Both reports include historical graph entries with negative timestamps; these were excluded from the summaries. Nsight reports 1,162 and 2,742 incomplete CUPTI events dropped, plus possible unflushed events. It used software instrumentation. These coverage and instrumentation limits prevent treating the reports as complete, uninstrumented acceptance measurements. The 1200-second replay success gate remains independently valid.

The canonical `.nsys-rep` files are stored through LFS. Large SQLite exports remain outside Git in the local Remnant analysis cache; compact trace analysis and its interval rules are in `remote-run-record.json`. Re-export with the installed Nsight CLI using `nsys export --type sqlite --output server.1.sqlite server.1.nsys-rep` (repeat for report 2); inspect the exported schema before adapting SQL. No runtime, image, or fork pins changed during this analysis.

Corrected profiled c24 comparison (1200 seconds each):

| Configuration | Output tokens/s | TPOT p50 / p95 (ms) | TTFT p50 / p95 (ms) | Device cache hit | Successful / attempted |
|---|---:|---:|---:|---:|---:|
| Native | 260.02 | 24.93 / 156.59 | 6009.54 / 16463.38 | 97.15% | 2037 / 2055 (99.12%) |
| Packed Triton | 267.21 | 24.39 / 127.81 | 6283.29 / 16216.06 | 97.30% | 2094 / 2114 (99.05%) |
| Packed Kernel | 270.27 | 25.56 / 128.16 | 5797.12 / 15320.64 | 97.35% | 2131 / 2149 (99.16%) |

All three pass the 99% success gate. Packed Triton throughput is 2.77% above Native; Packed Kernel is 3.94% above Native and 1.15% above Triton. Kernel median TPOT is 4.80% higher than Triton, p95 TPOT is 0.28% higher, and median TTFT is 7.74% lower. The aggregate throughput lead is not evidence of faster decode. Runner, adapter, dataset, replay settings, token pool (3,735,296), and extended decode graph configuration match. Requested cache format and implementation flags match the labels. Effective worker dispatch is not verified by those flags; the trace/source diagnosis below indicates Native fallback. Different random seeds and completion-conditioned request subsets remain uncontrolled; one profiled run each gives no repeated-run uncertainty estimate.

The reported token pool is identical in Native and both Packed modes, and hit rates differ by at most 0.20 percentage points. These runs do not establish a Packed cache-capacity benefit. Result commit `3a14925` contained summaries only; trace commit `e9e1fb9` subsequently supplied eight new Packed-labelled reports, now analyzed locally. The earlier Packed traces listed above belong to the misconfigured runs and must not substitute for these runs. Corrected c28 results are also not yet pushed. Preserve these as candidate serving evidence; do not declare a kernel-speed or SLO-capacity win from this comparison.

Trace diagnosis: Native fallback in runs labelled Packed

All eight requested Packed reports contain Native producer kernels and no expected Remnant pack/unpack kernel fingerprints. Decode graph nodes are not instrumented, so their absence alone would not prove a problem; however, eager prefill is separately traced and should execute the Packed producer. Identical pool allocation is supporting evidence. Source commit `8a520765f7295ece6a8989bb7419ffc8637d5703` keeps the mode in process-local globals in `remnant/config.py`, configured by `ServerArgs` resolution in the parent. The engine forces `spawn`. Worker publication through `runtime_context` does not reconfigure those globals, leaving the Native default. A CPU reproduction using that exact configuration source preserves requested flags but returns Native/Kernel defaults in a freshly spawned process for both Packed requests. This is source-and-trace diagnosis; no live GPU worker state or full model startup was queried.

The next fix must initialize mode inside each worker before memory sizing and graph capture, then verify effective mode, pool class, record bytes, and kernel dispatch per rank. Until then, the reported c24 throughput differences cannot establish a Packed or optimized-kernel speedup. Raw measurements, requested labels, success gates, and traces are preserved.

GPU0 decode graph observations (batch sizes inferred from graph creation order, checked against all 29 configured sizes in each of four workers):

| Requested label | Window | Mean captured batch | Mean graph ms | Graphs/s |
|---|---:|---:|---:|---:|
| Native c24 | 300 | 9.64 | 11.60 | 44.69 |
| Native c24 | 900 | 9.55 | 11.73 | 39.93 |
| Triton c24 | 300 | 7.78 | 10.77 | 44.87 |
| Triton c24 | 900 | 9.48 | 11.59 | 37.65 |
| Kernel c24 | 300 | 9.60 | 11.59 | 42.25 |
| Kernel c24 | 900 | 11.43 | 12.40 | 35.53 |
| Triton c28 | 300 | 28.00 | 16.71 | 2.11 |
| Triton c28 | 900 | 28.00 | 17.14 | 0.87 |
| Kernel c28 | 300 | 28.00 | 16.87 | 2.00 |
| Kernel c28 | 900 | 28.00 | 17.21 | 0.47 |

At captured batch 24, mean graph durations are Native 15.42/15.64 ms, Triton-labelled 15.50/15.61 ms, and Kernel-labelled 15.36/15.63 ms in the two windows. The larger average batch mix in Kernel-labelled c24 explains much of its higher unconditioned graph duration; it is not evidence that optimized packing is slower. Context length and request subsets still differ, and all three appear to execute Native.

Both c28 pairs retain batch-28 graphs. Graph executions fall from tens per second to about 0.5–2 per second; graphs occupy only 0.8–3.5% of captured GPU0 intervals, while recorded GPU0 busy time is 96–97%. Non-graph grouped GEMM and attention dominate, consistent with prefill/re-prefill crowding out decode. This rules out wholly disabled c28 graphs as the explanation. The reason for that prefill pressure remains unproven: c28 summaries and server logs are needed to assess prefix misses, cache eviction, and retractions. The eight reports use graph-level tracing and report possible event loss; boundary and historical-record exclusions are recorded in JSON. Full decode node costs cannot be recovered from these traces.

All large SQLite exports are kept in the local Remnant analysis cache, outside Git. The ephemeral CPU export container exited and was removed. No runtime source, image, fork pin, remote server, or GPU allocation was changed by this analysis.
