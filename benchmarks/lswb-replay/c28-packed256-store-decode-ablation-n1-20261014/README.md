# Packed C28 store/decode ablation — 2026-10-14

Author: Winston Cai.

All four 1200-second replays completed in one sequential GPUQ allocation on
10.72.1.171, GPUs 4–7, seed 20260919. The all-Kernel combination has the largest
TTFT penalty. Changing only decode with Triton store does not reproduce that
large penalty. Further source/timestamp analysis identifies synchronous frontend preparation
blocking the same event loop that records first output as a concrete bottleneck.
The apparent store/decode interaction is not yet a proven causal kernel effect.

## Complete replay results

TPOT is the arithmetic mean per successful request, not the median. Client
percentiles use the harness convention. The SLO comparison includes the entire
replay, including its startup workload.

| Store | Decode | Successful/failed | TTFT mean (s) | TTFT p90 (s) | TTFT p95 (s) | Mean TPOT (ms) | TTFT >10s |
|---|---|---:|---:|---:|---:|---:|---:|
| Triton | Triton | 2206/17 | 8.106 | 10.329 | 11.416 | 63.97 | 12.83% |
| Triton | Kernel | 2191/19 | 8.219 | 10.094 | 12.025 | 66.01 | 10.59% |
| Kernel | Kernel | 2161/19 | 9.158 | 11.140 | 12.566 | 60.22 | 21.84% |
| Kernel | Triton | 2220/20 | 8.238 | 10.403 | 13.529 | 62.65 | 11.67% |

All four full-run TTFT p90 values exceed the 10-second SLO. Kernel/Kernel misses
it on 21.84% of successful requests, versus 10.59–12.83% for the other cells.
Its lowest mean TPOT does not offset this first-token disadvantage.
The 17/19/19/20 client failures are all `missing_output_event`; throughput and
latency summaries above are conditional on successful requests.

## Startup sensitivity

All four runs have 37 successful requests dispatched before 180s. These include
very long initial requests and substantially inflate mean TPOT. Excluding them
is a sensitivity analysis, not a replacement for the full-run SLO result.

| Store | Decode | TTFT mean (s) | TTFT p90 (s) | Mean TPOT (ms) | TTFT >10s |
|---|---|---:|---:|---:|---:|
| Triton | Triton | 7.309 | 10.190 | 36.93 | 11.57% |
| Triton | Kernel | 7.418 | 9.895 | 38.99 | 9.29% |
| Kernel | Kernel | 8.370 | 10.902 | 32.56 | 20.72% |
| Kernel | Triton | 7.461 | 10.083 | 35.89 | 10.40% |

## Matched workload comparison

The dataset and adapter/runner source hashes agree across the four runs. There
are 2,135 successful request IDs shared by all four; every pairwise shared body
hash matches. Prefix-cache hit rates are 97.06–97.16%, with no retractions.
A fixed seed still does not force identical arrivals, completions, prefix-cache
histories, or scheduler batching in a closed-loop replay.

Holding Triton store fixed, the 1,803 matched requests with equal completion and
cache counts, dispatched after 180s in both runs, show Kernel decode **+16.7ms
mean TTFT**, +59.0ms median paired difference, and **+2.04ms mean TPOT**.
The TTFT paired difference changes sign across time windows; there is no large
consistent decode-only penalty in this cell.

Holding Kernel store fixed, the equivalent 1,767-request cohort shows Kernel
decode **+895.0ms mean TTFT**, +984.8ms median paired difference, and **−2.86ms
mean TPOT**. Fifty of 56 users have a worse mean TTFT in that strict cohort.
Same-time-window paired mean TTFT differences are +170ms at 180–360s and roughly
+1.05–1.08s from 360–1200s. This survives workload, output-length, cache-count,
startup, and coarse arrival-window controls.

Holding Kernel decode fixed, changing Triton store to Kernel store adds about
+950ms mean TTFT on 2,109 body-matched requests after 180s, while reducing mean
TPOT by 5.67ms. Holding Triton decode fixed, that store change adds only +155ms
TTFT and reduces TPOT by 1.08ms on 2,152 matched requests after 180s.
These are observations from single sequential runs, not independent repeated
estimates. The descriptive unpaired TTFT interaction is +806ms.

## Where the extra time appears

Mean server spans for requests received after 180s:

| Store | Decode | Receive→forward (s) | Prefill span (s) | Prefill→first output (s) | Scheduler queue (s) |
|---|---|---:|---:|---:|---:|
| Triton | Triton | 1.599 | 0.458 | 4.360 | 1.144 |
| Triton | Kernel | 1.592 | 0.471 | 4.346 | 1.150 |
| Kernel | Kernel | 1.708 | 0.444 | 5.240 | 1.249 |
| Kernel | Triton | 1.688 | 0.449 | 4.350 | 1.256 |

For Kernel store held fixed, Kernel decode adds only ~20ms receive-to-forward,
reduces the measured prefill span by ~5ms, and adds **~891ms between prefill finish
and the tokenizer's first output**. The all-Kernel deficit is concentrated after
the recorded prefill endpoint, rather than in a slower measured prefill span or
higher average scheduler admission queue.

In the queue<10ms cohort, the post-prefill span is 6.579s for Kernel/Kernel,
versus 5.473–5.615s for the other cells. The aggregate admission queue is therefore
insufficient to explain the gap. Supplementary token-count associations find
440 steady-state all-Kernel SLO misses; only 27.95% queued ≥1s. Those associations
are diagnostic, not request-ID-verified joins.

These timestamps cannot isolate GPU interference: `prefill_finished_time` is a
scheduler wall-clock endpoint after result processing, and `first_output_ts` is
the tokenizer's first output event. Their difference includes streaming/gating,
scheduling and IPC/detokenization work; it is not a pure GPU decode duration.
The current records do not attribute the delay to memory bandwidth or a specific
kernel. All GPUs were heavily utilized (~91–95%); utilization alone is not a
bandwidth or kernel-duration measurement.

## Source investigation: the API event loop is blocked by new input preparation

The SGLang source pinned by these runs is
`530b9b0974aed2b7cb3a76b2d148d5398e32ff77`. The checkout at that pin was inspected
separately from current dirty runtime work. These frontend files have no diff
between the run pin and the current local checkout.

The call chain is concrete:

1. `entrypoints/openai/serving_base.py:79` records `received_time` in async
   `handle_request`; line93 calls the regular synchronous
   `_convert_to_internal_request` directly on that event loop.
2. `entrypoints/openai/serving_chat.py:955` calls `_process_messages`. Its DSV4
   path renders `encoding_dsv4.encode_messages` then calls the synchronous
   `tokenizer.encode` at line1268. Message copying/schema preparation and
   scheduler serialization also occur in this receive-to-dispatch interval.
3. This is the same API/tokenizer process whose async output `handle_loop`
   calls `_handle_batch_output` (`managers/tokenizer_manager.py:2203`), which
   records `first_token_time` at line2451. With `tokenizer_worker_num=1`, an
   incoming blocking conversion prevents that output handler from running.
4. `prefill_finished_time` was already recorded after the scheduler received
   the first sampled token (`batch_result_processor.py:266`). The scheduler
   then performs prefix-cache bookkeeping and streams output at line366.
   The downstream API timestamp can be seconds later even when GPU prefill
   and first-token sampling have finished.

For requests received after 180s, intersect each post-prefill span with the
**union** of every incoming API receive-to-dispatch interval. No verified client
join is needed: all timestamps used here are from the exported server records.
The API intervals never overlap one another, and **zero of 8,853 first-output
timestamps** falls strictly inside one (100µs endpoint margin).

| Store | Decode | Post-prefill mean (s) | Overlapping API preparation (s) | Remaining wall time (s) |
|---|---|---:|---:|---:|
| Triton | Triton | 4.360 | 4.033 | 0.328 |
| Triton | Kernel | 4.346 | 4.022 | 0.324 |
| Kernel | Kernel | 5.240 | 4.868 | 0.372 |
| Kernel | Triton | 4.350 | 4.018 | 0.332 |

About 92–93% of post-prefill wall time overlaps preparation of other API inputs
in all four modes. Approximately 95% of the all-Kernel **extra** mean post-prefill
span versus either mixed configuration overlaps this frontend work. This is
strong evidence for frontend event-loop starvation as a major contributor,
not a CPU stack profile establishing that tokenization alone consumed every
covered instant. The receive-to-dispatch bracket also contains validation,
message conversion, serialization and any intervening awaits.

For the more specific cohort received after 180s, queue<10ms and post-prefill
wait>1s, Kernel/Kernel has a 6.905s mean wait, overlapping 6.437s of incoming work
and 17.09 preparation intervals. The other cells wait 5.759–5.981s and intersect
15.22–15.39 intervals. Average input preparation at 180–1200s is 386ms for
Kernel/Kernel, versus362–377ms for the other cells; mean prompt lengths are
nearly identical at 142k tokens. Across requests, input preparation time and
prompt length correlate 0.88–0.92. This supports long-prompt preparation as a
substantial part of the frontend cost.

There is no `kernel store && kernel decode` branch in the scheduler or API.
The two dispatch flags only choose independent compressor-store and attention
implementations. Triton store adds Torch TopK/mask and a separate pack launch;
Triton decode adds reconstruction/RoPE/Native materialization before stock
FlashMLA. Both therefore alter GPU/CPU launch timing and the closed-loop arrival
and completion cadence. Bursts of incoming preparation can then defer output
handling for a different number of event-loop turns. This is a plausible
explanation for the non-additive serving result, **not proof that faster Kernel
execution caused the larger burst in this one run**. Host scheduling variance
and run order remain alternatives. Switching either flag reduces the extra
observed delay; it does not remove the common 4.3s frontend-dominated wait.

This changes the next debugging priority: separate blocking OpenAI request
preparation from output processing before changing the GPU kernels. A bounded
executor for request conversion, with tokenizer/template thread ownership
reviewed, is the focused engineering direction. More tokenizer worker processes
are an existing diagnostic control but also change CPU concurrency. Merely
setting `--enable-dynamic-batch-tokenizer` does not move this OpenAI conversion:
its async executor is downstream in `TokenizerManager._tokenize_texts`, whereas
OpenAI conversion already produced `input_ids`.

The smallest attribution measurement is scheduler output-send time plus
frontend conversion start/end (or event-loop lag), under counterbalanced runs.
That distinguishes a token waiting in IPC from work before scheduler emission.
The existing C28 replays were unprofiled, so Nsight cannot currently verify the
full path for these exact runs. No production code was changed and no new GPU
jobs were launched for this investigation.

## Provenance and validity

GPUQ job `job-20261014-152423-abb88b0e` succeeded with exit0 at
2026-10-14T16:57:12.975978Z. Servers stopped between legs and after the last run;
allocation cleanup was verified with all GPUs free. Monitoring was removed.

Transfer commit `ec24a7592a1d390585c63fff472d247bd3139b6f` was pushed to `remnant`.
All 56 retained files were fetched selectively, verified against their Git blob
identities, then copied here without modifying their contents. Full server logs,
raw request payloads and model data remain outside this record. Original run IDs
are preserved in each configuration directory. See `transfer-provenance.json`
for run order and fork pins and `manifest.json` for file SHA256 hashes.

All 8,853 exported server timing records are valid; each validation reports
`complete=true`, `partial=false`, `errors=[]`, and seed20260919. However
`client_join_verified=false`: completeness is a count lower bound, not proof
that every client request has been joined to its server record. Exported records
lack loaded runtime/binary hashes; fork pins establish launch provenance only.

Local curated analysis remains uncommitted. Historical C24 data and outputs
were preserved. Its analyzer was parameterized to accept this root/concurrency
rather than creating another numerical harness.

## Reproduce

From the repository root:

```bash
python3 benchmarks/lswb-replay/c24-packed256-store-decode-ablation-n1-20261014/analyze.py \
  --root benchmarks/lswb-replay/c28-packed256-store-decode-ablation-n1-20261014 \
  --concurrency 28 --plot
```

`analysis.json` contains full/steady-state distributions, matched comparisons,
server spans and diagnostic associations. `summary.csv` is the flat full-run
summary. `additional-checks.json` records client failure categories, GPU averages
and server finish categories; it is supplementary to the canonical analyzer.
