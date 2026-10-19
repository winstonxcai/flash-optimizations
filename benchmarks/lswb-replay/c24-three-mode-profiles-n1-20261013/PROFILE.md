# C24 node-level profiling: Native, Packed Triton, Packed Kernel

Author: Winston Cai.

## Takeaway

- Triton adds reconstruction **and** selection/store overhead. Stock FlashMLA
  attention is not the main new cost.
- Kernel uses **Direct Packed FlashMLA**, not the fused reconstruction Adapter.
  Its captured decode latency is close to Native; parity is not established.
- Client C24 does not mean GPU B24 continuously. Actual mean decode batch was
  16.85 / 17.43 / 17.60 for Native / Triton / Kernel.
- These traces diagnose costs, not serving TPOT or C28 scaling.

## Coverage and provenance

Downloaded only the three compressed `server.1.sqlite.gz` exports from
`origin/remnant` commit `40a7d7d`; local checkout and dirty work were preserved.
`analysis.json` is the compact canonical aggregate. No GPU runs, commits,
pushes, or runtime edits were made.

| Check | Native | Triton | Kernel |
|---|---:|---:|---:|
| Recorded window | 6.126 s | 6.360 s | 7.838 s |
| Recorded kernels, four ranks | 2,001,342 | 2,033,568 | 2,028,375 |
| Complete four-rank decode steps | 228 | 212 | 226 |
| Scheduler batch ranges | 964 | 904 | 972 |
| Decode graph replay ranges | 912 | 848 | 904 |
| Replay success, observed job logs | 551/560 | 553/560 | 552/557 |
| Replay valid at 99% threshold | No | No | Yes |

- All four workers have scheduler rank IDs and internal graph-node kernels.
  CPU graph-replay payloads have null rank/step; correlate using the enclosing
  `scheduler.batch` range, process identity, and CUDA launch correlation ID.
- All exports warn of potentially missing CUDA/NVTX events. Native and Kernel
  each report 1,023 incomplete CUPTI events dropped; Triton reports
  875 + 894 + 867. Do not certify entire-window completeness.
- Reject one boundary step per mode with missing rank coverage. Retained decode
  steps have all four ranks and 43 attention-combine calls per rank.
- Launch metadata: V4-Flash-0731, TP4 H100, SGLang image v0.5.18, FP8 KV,
  `flashinfer_mxfp4` MoE, static memory 0.88, chunked prefill 8,192,
  context limit 1,048,576, max-running requests 256, full decode graphs,
  disabled prefill graphs. Native/Kernel used GPUs 0–3; Triton used 4–7.
- The image reports build commit `71de97b264b04dcd514cf904003028aefe9775c8`.
  It does not identify the content hashes of mounted dirty runtime code or
  loaded binaries. Native, Triton, and Kernel configuration is independently
  confirmed by arguments and recorded kernels, not by image metadata alone.
- Fresh client summaries/failure records were not pushed with these exports.
  Replay validity is from previously observed GPUQ logs; the request failure
  causes remain unverified.

## Decode at actual B24

Four-rank graph span: earliest graph-kernel start to latest graph-kernel end,
paired by scheduler step. Samples are profiled, correlated within one boot,
and not identical requests across modes.

| Measurement | Native | Triton | Kernel |
|---|---:|---:|---:|
| Complete B24 steps | 12 | 16 | 19 |
| Mean context per request | 137,538 | 137,139 | 137,230 |
| Graph mean | 15.739 ms | 19.197 ms | 15.854 ms |
| Graph p50 | 15.421 ms | 17.321 ms | 15.415 ms |
| Graph p95 | 17.530 ms | 31.482 ms | 20.261 ms |
| Mean delta versus Native | — | +21.97% | +0.73% |

Triton's p50 penalty is 12.32%; its larger mean penalty includes tail events.
The p95 estimates are weak with only 12–19 steps. Kernel's p50 is effectively
equal to Native in this sample, not a demonstrated speedup.

Reweight shared actual batch sizes using Native's 173 observed steps:
Native **14.726 ms**, Triton **16.601 ms (+12.73%)**, Kernel
**14.835 ms (+0.73%)**. This removes batch-distribution differences only;
contexts, expert routing, addresses, and prefill interference remain unmatched.
All actual-batch mean/p50/p95 values and sample counts are in `analysis.json`.

## B24 kernel breakdown

Mean summed kernel durations **per rank per decode step**, across all layers.
Kernels can overlap: these rows are not additive wall-time components.

| Kernel/group | Native | Triton | Kernel |
|---|---:|---:|---:|
| FP4 MoE `device_kernel` | 6.517 ms | 6.047 ms | 6.418 ms |
| `sm90_fp8_gemm_1d2d_impl` | 1.756 ms | 1.770 ms | 1.771 ms |
| All other GEMM/split-K compute | 1.938 ms | 1.981 ms | 1.943 ms |
| FlashMLA attention | 1.304 ms | 1.292 ms | 1.241 ms |
| FlashMLA combine | 0.298 ms | 0.294 ms | 0.356 ms |
| Indexer logits | 0.873 ms | 0.888 ms | 0.929 ms |
| mHC kernels | 1.364 ms | 1.369 ms | 1.365 ms |
| Custom all-reduce | 0.570 ms | 1.108 ms | 0.613 ms |
| Packed gather/dequantize | — | 0.596 ms | No separate kernel |
| Packed RoPE tail | — | 0.184 ms | Inside Direct loader |
| BF16 → Native row | — | 0.663 ms | No separate kernel |
| Extra radix sort + gatherTopK | — | 0.951 ms | — |
| Packed store | — | 0.130 ms | 0.206 ms |

- Triton reconstruction totals **1.443 ms** (~28.3 µs/gather,
  8.8 µs/RoPE, 31.6 µs/Native conversion per CSA-layer launch).
- Extra Triton `radixSortKVInPlace` and `gatherTopK` total **0.951 ms**.
  The inspected explicit-mask store uses Torch TopK/scatter; the Kernel store
  selects and packs together. This is consistent with the observed difference,
  but absent operation-level call attribution it is not a proven exclusive
  assignment of every sort/gather to mask generation.
- Kernel records both FlashMLA `REMNANT=false` and `REMNANT=true` attention
  instantiations. Direct Packed loading is active; the comparison changes the
  store implementation as well as the attention loader.
- Triton's larger all-reduce duration may include peer waits. It does not prove
  higher communication volume or a standalone network bottleneck.
- MoE/GEMM dominates recorded decode kernel time. Compressing C4 does not reduce
  those matrix computations. This limits the throughput benefit of improving
  attention alone.

## Prefill and remaining limits

Scheduler rank-0 markers record **12,198 / 10,394 / 31,490** prefill tokens in
Native / Triton / Kernel, respectively. Kernel's window therefore contains
2.58× Native's prefill tokens and lasts longer. Total kernel seconds cannot be
compared as though these were equal workloads.

These are shortened first capture windows, not the full 420-second replays or
the nominal 15-second requested interval. Correlation matching covers about
99% of recorded kernels; unmatched/boundary activity remains excluded from
step-level attribution. GPU-active intervals include communication waiting and
are not compute-utilization measurements.

Next analysis needs the fresh client summaries, per-request failure reasons,
and captured server/source manifests. Use unprofiled matched serving windows
for TPOT/throughput conclusions. No retry was launched for this analysis.

## Reproduction method

1. Fetch `origin/remnant`; fetch only `logs/node-*/**/server.1.sqlite.gz` via
   Git LFS. Resolve each pointer at `40a7d7d`, copy its verified object into the
   corresponding local path, and decompress without removing the gzip.
2. Open SQLite read-only. Join `NVTX_EVENTS`, `StringIds`,
   `CUPTI_ACTIVITY_KIND_RUNTIME`, and `CUPTI_ACTIVITY_KIND_KERNEL`.
3. Parse `scheduler.batch` JSON. Associate launch API timestamps contained in
   each range on the same thread; key correlation IDs by process, not globally.
4. Associate kernels to those launches. Require four ranks and 43 combine
   calls per rank. Graph-node kernels associated with `cudaGraphLaunch` supply
   the GPU span; CPU NVTX duration is not GPU duration.
5. Aggregate by actual batch, retain contexts and counts, and independently
   inspect demangled attention templates and diagnostic warnings.

## Capture-boundary timestamp audit (2026-10-14)

The original SQLite exports and original report GUI contain the same negative
NVTX ranges. This is not introduced solely by the newer Mac viewer.
`timestamp-audit.json` records every malformed row and an independent B24
recomputation keyed by process and CUDA launch correlation.

| Mode | Negative NVTX rows | Shared end timestamp (s) | Start minus end (ms) | Negative GPU kernels |
|---|---:|---:|---:|---:|
| native | 6 | 6.035447779 | 5.483–7.482 | 0 |
| triton | 6 | 6.296617602 | 5.456–5.631 | 0 |
| kernel | 6 | 7.750954504 | 5.206–6.842 | 0 |

All six rows per mode are the final `scheduler.run_batch` and nested
`scheduler.batch` EXTEND ranges on ranks 1–3. Seven ranges share the endpoint:
those six and rank 0’s zero-length `scheduler.process_input_requests` range.
The Native/Triton endpoint is about 0.3 ms before the recorded rank-0
`CUDA profiling data flush on cudaProfilerStop` overhead begins. Kernel has
the same boundary pattern, although that named stop-overhead row is absent.

The current profiler code calls CUDA profiler start/stop only on the base
GPU worker, with no barrier or all-worker GPU drain around that branch.
Other workers continue dispatching while the shared capture is closing.
The evidence strongly supports a stop-boundary clipping/race in collection or
report finalization: the common closing timestamp predates the last worker
range starts. This is an inference; the files do not expose the profiler’s
internal clipping algorithm. Negative timestamps for graph creation are
expected because graph setup occurred before capture time zero.

Do not set an invalid duration to zero and call it measured. Exclude these
six malformed ranges and incomplete boundary steps. Preserve original reports.
For future captures, coordinate a profiling-only stop boundary across workers:
finish/close current annotations, drain GPU work on every worker, rendezvous
on the CPU group, then stop capture on the designated worker while dispatch
is held. Resume serving afterward. Validate this sequence on a bounded run;
simply adding a barrier without preventing new work does not guarantee closure.

Independent B24 recomputation exactly reproduces the existing aggregate:
12 Native, 16 Triton, and 19 Kernel complete four-rank steps; zero ambiguous
correlation assignments; all saved kernel-duration means match within 1e-9 ms.
Every retained rank/step has 43 attention-combine calls. The malformed rows
are prefill boundaries and do not enter this B24 comparison.

At B24, separate Kernel FlashMLA durations per rank/step are **0.510 ms Direct
Packed** and **0.731 ms stock** (1.241 ms total). Native/Triton use only the
stock instantiation. This independently confirms the selected decode paths.
Attention plus combine is 1.602 / 1.586 / 1.597 ms for Native / Triton / Kernel.
Kernel’s saving over Triton principally removes reconstruction and explicit
selection overhead; this sample does not demonstrate an attention-only
advantage over Native. GPU-duration sums can overlap and are not wall time.

## Serving TTFT/TPOT recheck

These client summaries belong to the earlier 1200-second runs, not the fresh
short node captures above. Fresh node-run client summaries are still absent
from the fetched branch.

| Run pair | Mode | Mean TTFT (s) | p90 TTFT (s) | Mean TPOT (ms) | Output tokens/s | Mean latency after first token (s) |
|---|---|---:|---:|---:|---:|---:|
| jit-runtime-unprofiled | triton | 6.380 | 8.755 | 55.88 | 280.99 | 6.400 |
| jit-runtime-unprofiled | kernel | 7.129 | 9.055 | 52.30 | 280.06 | 5.845 |
| jit-nsys | triton | 7.217 | 9.484 | 58.24 | 258.64 | 6.695 |
| jit-nsys | kernel | 7.849 | 9.905 | 53.86 | 257.38 | 5.993 |

Triton has lower TTFT; Kernel has lower mean TPOT in both pairs. Output
throughput differs by only 0.33% unprofiled and 0.49% profiled in Triton’s
favor. These are single runs; a throughput winner is not established.

The unprofiled mean end-to-end latency is 12.780 s Triton versus 12.974 s
Kernel: Kernel saves 0.555 s after the first token but adds 0.749 s before it.
Profiled: 13.912 s Triton versus 13.842 s Kernel, with the same TTFT/decode
tradeoff but a slightly lower Kernel end-to-end latency.

Dataset selection and adapter/runner hashes agree for the pairs, but server
random seeds differ, successful request sets differ, and mean generated
lengths differ. Fixed selected workload does not force identical execution
under the fixed-active-user-pool, time-limited replay. Aggregate summaries
cannot isolate queueing, prefill, caching, streaming effects, or per-request
service time. The short node captures have different prefill-token loads;
their decode-only graph measurements cannot establish the cause of TTFT.

Do not substitute pre-fix or mislabeled runs for these current-mode pairs.
The current summaries and decode profiles agree that Kernel removes Triton
decode work. The persistent TTFT disadvantage remains a serving-level
question requiring matched request-level records and prefill/queue attribution.

## Prefill, queueing, and interference investigation

Canonical additional measurements: `prefill-interference.json`, read-only
NVTX/runtime/kernel correlation across complete four-rank batches. The original
exports remain unchanged. Shapes below match token count, request count, and
mean context length; this is not proof of identical tensors or expert routing.

| Prefill tokens | Requests | Mean context | Triton GPU span (ms) | Kernel GPU span (ms) |
|---:|---:|---:|---:|---:|
| 223 | 1 | 141535 | 162.413 | 166.388 |
| 283 | 1 | 134171 | 162.758 | 163.393 |
| 283 | 1 | 137499 | 163.700 | 165.259 |
| 454 | 1 | 144326 | 164.851 | 163.392 |
| 461 | 1 | 140237 | 166.685 | 163.918 |
| 2960 | 1 | 134288 | 242.541 | 242.525 |

Six matching-shape prefill batches show no consistent Kernel slowdown.
The 2,960-token/134,288-context batch agrees within 0.016 ms. Packed BF16
reconstruction and sparse-prefill attention times agree closely. In the
223-token case, mean per-rank gather is 1.674/1.675 ms and sparse attention
2.523/2.525 ms for Triton/Kernel. Triton store plus radix sort and gather is
0.814 ms (excluding other selection helpers); Kernel fused store is 0.106 ms.

Code paths: large/sparse prefill in `deepseek_v4_backend.py` uses the same
`unpack_gather_bf16` and `flash_mla_sparse_fwd` for both Packed modes. Direct
Packed dispatch also applies to small extends when they remain in the decode
attention branch, so not every possible prefill case is identical. The
captured matching prefill cases above exercise the shared sparse path.
Compressor selection/store differs: Triton Torch TopK/mask plus pack; Kernel
fused FP32 selection/norm/store. The recorded store work does not exhibit a
Kernel regression. Local arithmetic correction after these captures should
not be conflated with the captured source/binary identity.

The scheduler tries admissible prefill before decode. These runs have
`enable_mixed_chunk=false`, `chunked_prefill_size=8192`, `max_prefill_tokens=16384`,
FCFS, and overlap enabled. `max_total_num_tokens=4532736` is identical.
No cache-mode branch was found in scheduler admission. Overlap hides CPU
scheduling behind forward work; it does not guarantee simultaneous prefill
and decode model execution.

A single prefill commonly stretches successive decode starts to about
180–190 ms, versus roughly 15–19 ms for B24 decode alone. The matched
2,960-token prefill causes a 257.499 ms Triton versus 255.291 ms Kernel
decode-start gap. The 223-token match has 162.413/166.388 ms GPU spans
despite only 34.810/33.871 ms of non-collective summed kernel duration per
rank. CPU annotation spans are roughly 165–171 ms; thousands of eager launch
APIs and collective waits are visible. CPU samples were disabled, so the
remaining interval cannot be assigned exclusively to Python or CPU compute.

Full 1200-second `jit-nsys-c24` logs, event averages (not time-weighted):

| Metric | Triton | Kernel |
|---|---:|---:|
| Prefill records | 2500 | 2494 |
| New prefill tokens logged | 8,043,008 | 8,048,896 |
| Mean queue at prefill records | 2.18 | 2.48 |
| Mean pending tokens at prefill records | 301,554 | 345,035 |
| Mean running requests at prefill records | 10.65 | 9.21 |
| Mean running requests at sampled decode records | 9.82 | 8.27 |

Kernel shows greater pending work/queues and smaller decode batches for
nearly equal total new-prefill volume. This supports a changed admission/
arrival/cache-reuse trajectory, not a measured per-request queue-delay cause.
The fixed-active-user replay advances users after completions; dataset seed
does not fix wall-clock arrivals or completed request sets. Server seeds also
differ. We cannot identify the specific p90-TTFT requests without request-level
records. The newer short node windows and the 1200-second client results are
different runs; the short windows are diagnostic, not p90 attribution.

Conclusion: direct prefill/store regression is not supported for the six
matching cases. The strongest remaining lead is queueing plus eager prefill
interference/admission timing. A causal Kernel-specific explanation is still
unproven. Existing `--enable-request-time-stats-logging` provides initial queue
duration; `SchedulerReqTimeStats` already tracks forward entry and prefill
finish. A matched next study should retain request IDs, actual arrivals,
cached/uncached tokens, these timestamps, and first SSE output, with identical
server seed and alternating mode order. Avoid creating a replacement harness.
Do not enable mixed chunks or change chunk size while claiming kernel-only
parity; treat scheduling changes as a separately controlled mechanism.

No runtime edits, new GPU jobs, commits, or pushes were made.
