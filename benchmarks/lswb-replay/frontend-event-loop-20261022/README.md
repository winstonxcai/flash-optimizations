# Frontend event-loop blocking — CPU diagnostic

Author: Winston Cai.

## Result

**Blocking reproduced without a model.** Two independent CPU invocations covered
three cached OpenHands issues, nine prefixes and 216 trials. In all 54
during-preparation trials, the separate process sent its output while the actual
tokenizer was executing; the frontend handler entered only after preparation
finished. Token IDs and received output identity/text passed every check.

Means across both invocations, in milliseconds except token counts:

| Mean prefix tokens | Preparation | Tokenization | Send→delivery during preparation | Idle send→delivery |
|---:|---:|---:|---:|---:|
| 33,109 | 117.73 | 115.38 | 105.92 | 0.53 |
| 55,628 | 197.66 | 193.26 | 185.98 | 0.56 |
| 123,069 | 466.33 | 457.38 | 454.51 | 0.58 |

- Before/after control means: **0.44–0.67 ms**. After preparation ended, the
  delayed handler entered in **0.33–0.34 ms** on average.
- Tokenization accounts for **97.8–98.1%** of the measured preparation. Sends
  were requested 10 ms after starting preparation; actual send timestamps,
  not that requested delay, determine placement.
- Local macOS, x86_64 Python 3.13, uvloop 0.21.0. These are **not Linux-server
  timing estimates**, LongSWE measurements or full-model TTFT results.
- Five CPU contract tests passed; source/sample hashes, identity coverage,
  token stability, timestamps and aggregate values were independently checked.
  GPU use and Modal cost: **zero**. Forks and production dispatch are unchanged.

**Decision:** synchronous input preparation can substantially defer delivery of
an already-sent output. The test does not establish why Kernel and Triton had
different arrival patterns, or quantify this mechanism's contribution to their
production TTFT difference. Next: repeat on the GPU server's CPU with actual
converted LongSWE bodies; production attribution still needs scheduler-send and
frontend conversion/receive boundaries.

## Question and scope

Can synchronous DSV4 chat preparation delay an output already sent by another
process? This is a mechanism test, not a Triton/Kernel comparison or serving
TTFT benchmark. No model weights, GPU, Modal allocation or runtime changes.

`probe.py` is the isolated experiment entry point. Existing replay clients cannot
execute the server's input preparation or observe its output handler, so they
cannot own this diagnostic. It executes selected, unchanged AST definitions from
the checked-out fork, avoiding imports of GPU execution infrastructure on macOS.
It uses the actual request models, DSV4 encoder, 0731 fast tokenizer, output IPC
struct, receive loop and batch-output handler. Source files are not rewritten.
Initialization, optional metrics/tracing and the network sink are test scaffolds;
this is **not a complete HTTP server**. The sink records actual loopback delivery,
not the production OpenAI SSE formatting path.
Measured preparation is `_apply_jinja_template`, including message serialization,
copying, DSV4 rendering and tokenization. Request-model validation, tool-constraint
selection and scheduler dispatch/serialization are outside this measured span.

## Protocol

- One independently spawned process sends the first output over ZMQ, and reads
  the frontend's loopback sink. Its monotonic timestamps bracket the send and
  record receipt; frontend timestamps cover preparation, handler entry and
  notification. Sender startup is outside measurement.
- Compare idle output handling with output sent before, during and after the
  same synchronous preparation. Reject a sample if its actual send interval
  does not satisfy its intended placement.
- Use identical request bodies across placements; do not insert a sleep in the
  frontend to simulate tokenization. Serialize/tokenize real cached coding
  prefixes and vary prefix length. JSON contains hashes/counts, never prompts.
- Acceptance: every output arrives with the expected identity/text, preparation
  produces stable token IDs, during-preparation sends precede its end, and the
  output handler runs only after preparation. Compare with idle/before/after
  controls; report IPC/host scheduling residuals separately.
- Stop after this CPU mechanism test. Production attribution still requires
  matched scheduler-send/conversion/receive timestamps; existing C28 records do
  not have the scheduler-send boundary.

## Reproduce

Use a CPU environment with `tokenizers`, `transformers`, `pyzmq`, `msgspec`,
`pydantic`, `openai`, `pybase64`, `requests`, `numpy`, `IPython`, `tqdm` and
`uvloop`. The latter matches the production event-loop implementation;
`--event-loop asyncio` is an explicit alternate control.
No dependency or model download occurs in the probe. `--requests` accepts a
JSONL OpenAI body or a cached record containing `request` and `instance_id`.
Use converted replay bodies, not the raw LongSWE dataset schema.

```bash
USE_TORCH=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python benchmarks/lswb-replay/frontend-event-loop-20261022/probe.py \
  --requests /path/to/cached-coding-requests.jsonl \
  --tokenizer /path/to/0731-tokenizer-snapshot \
  --output raw/frontend-event-loop/<fresh-run-id>
```

Default: three issues, three prefix fractions, three rounds, four placements.
Raw inputs remain external. The local cached OpenHands cohort can test the
mechanism; it must not be relabeled as LongSWE. A VDI run can instead use the
actual LongSWE OpenAI bodies. See `results.json` for completed local evidence.

CPU checks (no sockets or tokenizer required):

```bash
python -m unittest discover \
  -s benchmarks/lswb-replay/frontend-event-loop-20261022 -p probe.py -v
```

## Next diagnostic: locate first-output residence time

Question: does the C40 Packed Kernel delay accumulate before the scheduler
emits output, inside detokenization, or between detokenizer send and API receipt?
The existing C40 HiCache records cannot distinguish these stages. Overlap with
API preparation is not a causal attribution, especially at high API occupancy.

- **Mechanism changed:** timestamps only, enabled at server startup with
  `FRONTEND_DIAGNOSTICS=1` through the local launcher. Default is disabled.
  No attention, packing, scheduling, cache, request-body or success-policy change.
- **Frozen matrix:** Native and Packed Kernel-256; C40, TP4, MF0.88,
  HiCache ratio 2, write-through/kernel/layer-first, seed 20260919,
  chunk 8192, original session-owned LongSWE replay, one 900-second run each.
  Analyze the whole run and the cohort received after 180 seconds separately.
- **Baseline:** reuse the saved uninstrumented C40 records for context; this
  diagnostic is not another throughput acceptance result or a paired speedup.
  Both diagnostic modes use the same committed Python source and existing
  image; native extensions are unchanged. Record the actual image ID.
- **Source isolation:** parent worktree `frontend-first-output`, based on
  `2b3e325`; fork branch `codex/frontend-first-output`, based on `921cc2e03f61`.
  Shared experiment checkout and its newer microbenchmark changes are preserved.
- **Boundaries:** scheduler first batch send-start (before payload encoding),
  detokenizer receive, detokenizer forward-start, API socket receive, and API
  handler entry. API request state latches the first packet so later/final
  scheduler snapshots cannot overwrite it. These are CPU boundaries, not CUDA
  completion or visible SSE content timestamps. Existing `first_output_ts`
  retains its original meaning.
- **Exact join:** the adapter captures the existing SSE `id`, matching exported
  metrics `id`. Missing/unstable IDs fail diagnostic coverage, without changing
  response-validity rules. Raw prompts are not copied into diagnostic records.
- **Acceptance:** local transport/serialization/adapter checks, then real module
  CPU checks inside the VDI image before GPUQ allocation. GPU records must have
  unique IDs and complete ordered boundary stamps for joined completed requests;
  report missing coverage and failed responses separately. Infer no missing
  boundary. Require at least 200 joined post-180-second records per mode before
  drawing a mode comparison; otherwise the decision is inconclusive.
- **Stop:** two jobs only, each requests four GPUs from GPUQ. No automatic GPU
  retry, no image rebuild, no Modal launch. Jobs own their server and stop it on
  exit. A failed preflight consumes no GPU allocation. GPUQ queue time is not
  part of the run window.
- **Rollback:** unset `FRONTEND_DIAGNOSTICS`; the new timestamp path is disabled.
  Keep the instrumentation until this attribution decision is recorded. Any
  threading/batching optimization needs a separate correctness and timing gate.

`diagnose.sh` owns this experiment's CPU preflight and GPUQ submission and calls
the existing `serve.sh` / `bench-lswb.sh`; it does not replace the replay runner.
Its output stays under ignored `logs/frontend-first-output-*` and normal
`results/lswb-replay/` directories. It requires the existing local dataset and
manifest via `REPLAY_DATASET` and `REPLAY_MANIFEST`, and a committed clean source.

On the VDI, after pulling the committed parent and synchronizing clean forks:

```bash
export REPLAY_DATASET=/home/jovyan/wenyuhong/benchmarks/datasets/h20-dsv4pro/longcodebench_openai
export REPLAY_MANIFEST=/home/jovyan/wenyuhong/benchmarks/cache/official-longswebench/longswebench-openai-v1.json
bash benchmarks/lswb-replay/frontend-event-loop-20261022/diagnose.sh preflight
bash benchmarks/lswb-replay/frontend-event-loop-20261022/diagnose.sh submit
```

Each leg saves `first-output-pipeline.json` next to the normal replay artifacts.
It uses `first_output_pipeline_diagnostics` in the existing store/decode analyzer,
keeps failed-response identity diagnostics, and checks the post-180-second sample
gate separately from response validity and the TTFT SLO. The CPU probe suite now
also covers timestamp placement/transport through both IPC formats and worker
loops; the original mechanism measurements in `results.json` retain their original
source hashes and are not relabeled as instrumented runs.

After **both saved GPUQ jobs are terminal**, prepare their transfer packet:

```bash
bash benchmarks/lswb-replay/frontend-event-loop-20261022/diagnose.sh collect \
  /path/to/repository/logs/frontend-first-output-<batch-id>
```

This creates a fresh `transfer/<batch-id>/` containing whitelisted metadata,
responses, full request timings, logs, source/image identities, saved job records,
and `SHA256SUMS`. Failed jobs and missing artifacts remain explicit; collection
does not retry jobs or declare a benchmark valid. Raw prompts, request payloads,
and binaries are excluded. Inspect the packet before explicitly staging,
committing and pushing it. Do not pull collector changes onto the VDI while a
diagnostic job is queued or running: its launch sources are frozen.

VDI CPU qualification on `6f87576` / SGLang `7a95ce8231`: real serving-image
request-time tests **10/10 passed**, host replay tests **19/19 passed**, with
exact clean fork pins. No image rebuild or GPU allocation was needed. Initial
submission exposed a GPUQ CLI mismatch: this host accepts the command after
`--`, without a `--bash` option. Submission stopped before creating a job;
the launcher now passes an argument vector and saves the CPU log and CLI help
with every batch. Both instrumented full-model legs completed; see the baseline evidence below.

## Evidence and rollback

### Follow-up candidate: keep chat preparation off the receive loop

Both completed baseline legs locate multi-second output residence between
detokenizer forward-start and API receive: post-180-second means are 6.183 s
Native and 5.713 s Packed Kernel. All successful responses join exactly to ordered
server boundaries (1,346 Native; 1,279 Packed). This pair does not reproduce a
Kernel-specific regression. Together with the source-isolated blocking reproduction, it motivates
one reversible candidate rather than an attention-kernel change.

- **Question:** can the API receive already-produced output promptly while a long
  DSV4 chat request is being rendered/tokenized, preserving the exact converted
  request and all preparation/queue time in client TTFT?
- **Changed mechanism:** opt-in `SGLANG_DSV4_ASYNC_PREPARATION`, default off.
  Offload the existing text-chat converter to one owned worker; bound admission
  and retain ownership until its actual work finishes, including cancellation.
  Preserve receive timestamps, errors, token IDs, tools, reasoning and sampling.
- **Baseline policy:** finish and transfer both instrumented baseline legs before
  changing VDI source. Compare the candidate using the identical C40/TP4/MF0.88/
  HiCache-ratio-2/900-second protocol, with source and image identities recorded.
  Apply the same frontend change to Native and Packed. No GPU retry is automatic.
- **Local gates:** exact synchronous/async conversion parity, default-path parity,
  exception handling, queued/running cancellation, no overlapping conversions,
  no late scheduler dispatch, responsive output handling and explicit shutdown.
  Qualify the actual image on CPU before allocating GPUs.
- **Acceptance:** ordered exact-ID diagnostic coverage and the existing sample
  gate; materially reduce the detokenizer-to-API residence time without moving
  receive timestamps or hiding preparation wait. Report failures, client TTFT,
  TPOT, E2E and request throughput together. A stage improvement alone is not a
  throughput claim or validation of all tokenizer-consuming endpoints.
- **Stop/rollback:** reject or retain as inconclusive if correctness/coverage
  fails, or if another stage dominates the remaining discrepancy. Disable the
  startup switch to restore synchronous conversion. Keep the candidate isolated
  until these gates and independent review pass; preserve baseline evidence.

- Canonical aggregate record: `results.json`; includes both invocation commands,
  package versions and source/input/tokenizer hashes. Raw trial records remain
  in ignored `raw/frontend-event-loop/uvloop-20261022-{003,004}/samples.jsonl`.
  Each file includes one explicitly excluded handshake sample.
- Source: parent `21992f0`, clean pinned SGLang `bf3e538c5c`; tokenizer revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`. Full IDs/hashes are in JSON.
  Python dependencies added for this test remain in ignored
  `data/frontend-cpu-deps`; locally prepend that directory to `PYTHONPATH`.
- Earlier setup/preflight failures remain under `raw/frontend-event-loop/`:
  sandbox-blocked localhost sockets, then unnecessary IPC annotation resolution.
  The final probe uses the real schema's fields without resolving GPU-only
  annotations; the actual default pickle send/receive functions are unchanged.
- No production service was started or modified. Removing this study only removes
  reproduction code/evidence; it does not change serving behavior.


### Candidate execution and scope

The startup launcher knob is `DSV4_ASYNC_PREPARATION=1`. The diagnostic freezes
its value in `async-preparation.txt`, forwards it to GPUQ, checks the activation
log, and runs the maintained `AsyncInputPreparationTests` inside the actual
serving image before GPU allocation. The source-isolated probe passes 20 tests.
The cached-tokenizer CPU evidence in `results.json` retains its original source
hashes; subsequent formatting/logging edits do not relabel that evidence.

```bash
DSV4_ASYNC_PREPARATION=1 bash benchmarks/lswb-replay/frontend-event-loop-20261022/diagnose.sh submit \
  /home/jovyan/wenyuhong/benchmarks/datasets/h20-dsv4pro/longcodebench_openai \
  /home/jovyan/wenyuhong/benchmarks/cache/official-longswebench/longswebench-openai-v1.json
```

This candidate is scoped to DSV4 text chat. Other tokenizer-consuming endpoints
share the tokenizer and are outside this gate; mixed-endpoint concurrency is
not qualified. One worker preserves converter serialization and output-loop
responsiveness, not additional tokenization capacity. Cancellation retains the
slot until the actual worker finishes; shutdown drains owned work, and no
cancelled request is dispatched late. Default off creates no worker.

Baseline packet: `transfer/frontend-first-output-20261022_221813-YP2fOa`,
parent `ae822c5`, SGLang `7a95ce8231`, FlashMLA `97f3faa58450`; both legs ran
sequentially on the same assigned four GPUs and stopped their own servers.
The 900-second diagnostic has 14/1,360 Native and 13/1,292 Packed missing-output
responses, narrowly failing the 99% response-validity rule. Timing coverage passes
independently; neither is an accepted throughput benchmark. Preserve these failures
when interpreting the candidate.

### Candidate image preflight — 2026-10-22

VDI parent `8c6e3e1`, SGLang `63291eba61`, unchanged FlashMLA/image:
actual imported request-time tests **10/10**, maintained async preparation tests
**7/7**, and host replay tests **19/19** passed before submission. The first
candidate invocation hit the stdlib `test` namespace; the maintained test is now
invoked by its source-file path. A subsequent host replay preflight failed its
hard-drain wall-clock check (14.7 s versus a 10 s bound); an unchanged full retry
passed. No tolerance or success rule changed, and neither failed preflight
allocated a GPU. Treat this as host scheduling sensitivity, not erased evidence.

Candidate batch `frontend-first-output-asyncprep1-20261022_234403-D4SLp1`:
Native `job-20261022-154455-ccb64602`, Packed Kernel
`job-20261022-154455-611f2faa`. Both request four devices by count through GPUQ;
queue time is excluded from the 900-second window. Both jobs are now terminal;
the packet was transferred in parent `640b852a`.


### Candidate full-model outcome — 2026-10-23

**Keep default off for controlled chat qualification.** Offloading the unchanged
converter largely removes the multi-second API receive stall. This establishes a
shared frontend bottleneck; it does not establish a Packed throughput advantage.
Missing-output responses remain unresolved.

Candidate parent `8c6e3e1`, SGLang `63291eba61`; baseline parent `ae822c5d`,
SGLang `7a95ce8231`. Both use FlashMLA `97f3faa58450`, image
`sha256:09dfea18bd2c4168481710dcb1830a2c2efdb8e8f45f9d1ad7bb85a1b7ff48ab`
and C40/TP4/MF0.88/HiCache-ratio-2/900-second original replay. Candidate legs ran
sequentially on GPUQ-assigned GPUs 2,5,6,7; baseline used 0,1,2,5. These are n=1
runs at different times, not randomized throughput estimates. Both candidate jobs
terminated with teardown signals in their logs. Kernel exit 1 reflects the
response-validity gate, not a server crash during replay.

Whole-run client results include cold requests and final request drain:

| Mode | Preparation | Successful req/s | Output tok/s | TTFT mean / p90 (s) | E2E mean (s) | TPOT mean (ms) | Failed / attempted |
|---|---|---:|---:|---:|---:|---:|---:|
| Native | Sync | 1.4721 | 209.86 | 10.845 / 12.314 | 21.514 | 113.71 | 14 / 1,360 |
| Native | Worker | 1.7289 | 248.47 | 6.166 / 10.130 | 18.167 | 114.01 | 15 / 1,603 |
| Packed Kernel | Sync | 1.4084 | 196.69 | 12.359 / 11.687 | 22.008 | 108.46 | 13 / 1,292 |
| Packed Kernel | Worker | 1.7285 | 243.40 | 6.391 / 10.264 | 18.255 | 114.75 | 17 / 1,596 |

Within-mode requests/s rises **17.45% Native / 22.73% Packed**; output tokens/s
rises **18.40% / 23.75%**. These are descriptive changes, not accepted speedup
claims. With the fix in both, Packed requests/s is **0.026% lower** and output
throughput **2.04% lower** than Native. Both full-run TTFT-p90 values miss 10 s.
Native response success is 99.064%; Packed 98.935%, below the unchanged 99% gate.
A selected post-startup cohort cannot replace the full-run SLO population.

First-output residence, successful exact-ID joins received at least 180 seconds
after each run's own start:

| Mode | Sync mean / p90 | Worker mean / p90 | Worker samples |
|---|---:|---:|---:|
| Native | 6,183 / 10,324 ms | 1.96 / 1.44 ms | 1,556 |
| Packed Kernel | 5,713 / 9,073 ms | 20.89 / 3.09 ms | 1,547 |

Mean above p90 reflects rare delays. Packed still has eight successful first
packets over 1 s, maximum 6.214 s, concentrated in a brief interval. These records
do not identify that residual tail's cause; not every delay is fixed.

**Check displaced waiting.** The following 1,238 dataset IDs are successful in
all four runs, have identical body hashes/prompt counts, and server receive times
after 180 s in every run. Mean seconds:

| Stage | Native sync | Native worker | Packed sync | Packed worker |
|---|---:|---:|---:|---:|
| API receive → dispatch | 0.354 | 2.938 | 0.357 | 3.141 |
| Dispatch → first forward | 1.233 | 1.464 | 2.973 | 1.276 |
| Prefill | 0.430 | 0.409 | 0.560 | 0.436 |
| Detokenizer send → API receive | 6.146 | 0.002 | 5.697 | 0.025 |
| Client TTFT | 8.926 | 4.872 | 10.310 | 4.952 |
| Client E2E | 15.514 | 13.478 | 16.253 | 13.579 |

Worker admission/conversion wait remains charged to TTFT. It rises before
dispatch, but client TTFT and E2E improve on the same requests, so this is not
only a renamed timing stage. API receive→dispatch includes admission, conversion
and dispatch, not isolated tokenizer execution. Earlier first output lengthens
the visible stream interval; increased client TPOT alone does not prove slower
GPU decode. Generated lengths also differ between Native and Packed.

**Failure audit.** All candidate failures are HTTP 200, normal EOS token 1 with
positive completion usage but no observable output. Native: 13×16 tokens, 1×34,
1×36; Packed: 15×16, 1×7, 1×36. Among common attempted IDs, nine Native and ten
Packed failures also fail with sync preparation. Four previously valid requests
fail in each worker run; five Native / three Packed failures become valid.
Offloading does not eliminate the existing response problem. Without raw SSE or
generated token strings, control-only generation and tool-parser suppression
cannot be distinguished. Keep every failure in the validity gate.

**Evidence:** candidate SHA256SUMS **51/51** pass; baseline 50/50. Recomputing
`first_output_pipeline_diagnostics(client JSONL, server JSONL, replay start)`
from the existing store/decode `analyze.py` exactly reproduces all four retained
pipeline records. Successful joins are 1,588/1,588 Native and 1,579/1,579 Packed,
with ordered stamps; all attempted client/server prompt and completion counts
agree. No retractions. All common off/on attempts have equal body hashes/prompt
counts. Exported serving arguments differ only in startup timings and metrics
output paths; logs confirm worker activation (the environment switch is not
an exported argument). Missing packet sidecars are boot source/GC/host-memory
records, not measured requests or pipeline records.

Canonical `results.json` → `candidate_serving_validation` retains source identity,
checks, raw summaries, recomputed stages, off/on and four-way common-ID cohorts,
nearest-rank client percentiles, failure transitions and whole-run deltas.
Cross-run joins use dataset `request_id`; within-run joins use captured SSE ID.
Post-180 cohorts require every compared server receive time ≥ its own start+180.
Shared analysis functions: `first_output_pipeline_diagnostics` and `client_stats`
in `../c24-packed256-store-decode-ablation-n1-20261014/analyze.py`.

**Next diagnostic:** capture full SSE and generated token strings for a small,
frozen failing-ID set, original payload/caps/EOS, explicit logprobs. Preserve the
original validity decision. Do not increase caps or count empty output as success.
This follow-up and mixed-endpoint checks remain unrun. No new GPU jobs or runtime
changes were made during this analysis. Rollback: `DSV4_ASYNC_PREPARATION=0`
(the default).

## Packed C28 Triton/Kernel off/on comparison — 2026-10-23

**The receive stall is removed in both modes; Kernel's TTFT disadvantage is only
partly resolved.** Its remaining delay is principally preparation/admission and
scheduler waiting. This record does not establish slower Kernel prefill or an
SLO throughput improvement.

Protocol: Packed-256, C28, TP4 H100, MF0.88, HiCache off, original session-owned
LongSWE replay, 1,200-second dispatch window followed by request drain, n=1 per
cell. Each mode ran sync preparation then the worker with a fresh server/cache.
The modes ran concurrently on disjoint TP4 allocations on one host: Triton GPUs
0–3, Kernel GPUs 4–7. Shared CPU contention and fixed off/on order are uncontrolled.
Source is parent `d41a30e44e5c`, SGLang `63291eba6137`, FlashMLA `97f3faa58450`,
with image `sha256:51ed908c159139e522c2254b13c2713bf2fa5c51182f21cfec908252d2831972`
(the C40 comparison used a different image). Both store and decode match the
named mode. The pool is 4,536,832 slots in every cell.

### Full-run client results

Canonical replay summaries, including cold requests and request drain:

| Mode | Preparation | Successful req/s | Output tok/s | TTFT mean / p90 (s) | E2E mean (s) | TPOT mean (ms) | Failed / attempted |
|---|---|---:|---:|---:|---:|---:|---:|
| Packed Triton | Sync | 1.8344 | 275.13 | 7.872 / 9.822 | 14.989 | 65.33 | 15 / 2,231 |
| Packed Triton | Worker | 1.9711 | 295.10 | 5.718 / 10.632 | 13.986 | 70.94 | 17 / 2,398 |
| Packed Kernel | Sync | 1.6652* | 246.68* | 8.979 / 11.140 | 15.333 | 60.99 | 20 / 2,186 |
| Packed Kernel | Worker | 1.9896 | 301.02 | 6.302 / 11.551 | 13.837 | 63.90 | 18 / 2,439 |

Only Triton sync passes the **full-run TTFT-p90 <10 s** requirement. All four
pass the unchanged 99% response-success gate, but their missing-output failures
remain. Device hit rates are nearly identical, **97.14–97.20%**.

With the worker enabled in both, Kernel delivers **0.93% more requests/s** and
**2.01% more output tokens/s** than Triton. Its mean TTFT is 584 ms higher, while
mean E2E is 148 ms lower. These are descriptive n=1 results, not established
kernel speedups. Earlier observable first output also changes the interval used
by client TPOT; TPOT is not an isolated GPU decode measurement.

*Kernel sync has an anomalous denominator: summary duration is **1,300.719 s**,
but its last response completes at **1,208.544 s**. The unexplained **92.176 s**
post-response tail is not a long request drain. Other cells have only 8–28 ms
between last response and summary duration. Preserve the canonical results;
do not headline the apparent +19.48% requests/s / +22.03% output improvement for
Kernel without this qualification. As a separate sensitivity calculation,
dividing each cell's counts by its own last response time gives Kernel sync
**1.7922 req/s / 265.49 output tok/s** and worker **1.9896 / 301.03**: changes of
**+11.01% / +13.38%**. These alternate rates do not replace the replay protocol.
The cause of the tail is unresolved; its activity JSONL is absent from the pulled
packet. Fixed-1,200-second completion counts/rates are also retained in JSON as
a separately labeled sensitivity, excluding drain.

### What changed in the first-output path?

Successful exact-ID joins, received after each run's own first 180 seconds:

| Mode | Sync detokenizer-send → API-receive mean | Worker mean | Sync / worker samples |
|---|---:|---:|---:|
| Packed Triton | 3,837 ms | 2.62 ms | 2,188 / 2,342 |
| Packed Kernel | 4,954 ms | 1.77 ms | 2,138 / 2,380 |

Kernel's additional receive residence disappears. On the **same 2,111 dataset
requests** successful after 180 seconds in all four cells, the mean client TTFT
gap shrinks from **1.095 s** with sync preparation to **0.615 s** with the worker
(about 44%). Their body hashes and prompt counts match. Generated lengths and
arrival times are not held fixed by this closed-loop replay.

A stricter worker-only cohort contains **1,917** successful request IDs with
identical bodies, prompt counts, generated token counts and cached token counts,
received after 180 seconds in both runs. Mean seconds:

| Stage | Triton worker | Kernel worker | Kernel − Triton |
|---|---:|---:|---:|
| API receive → dispatch | 3.349 | 3.800 | +0.451 |
| Dispatch → first forward | 1.039 | 1.255 | +0.216 |
| Prefill | 0.384 | 0.385 | +0.001 |
| Prefill end → API first output | 0.026 | 0.028 | +0.001 |
| Client/server first-output residual | 0.039 | 0.034 | −0.005 |
| **Client TTFT** | **4.837** | **5.501** | **+0.665** |
| Client E2E | 10.395 | 10.411 | +0.016 |

API receive→dispatch includes worker admission, conversion and dispatch, not
isolated tokenization. Scheduler queue means are 0.986 / 1.202 s, already inside
dispatch→forward; do not add them again. The measured prefill difference is only
1.4 ms on this cohort. After-first-output→server-finish is 650 ms shorter for
Kernel, offsetting its higher TTFT. A closed-loop trace still does not identify
whether the remaining upstream wait is caused by mode-dependent arrivals,
shared CPU contention or another scheduling effect.

### Evidence, decision and next diagnostic

- Packet: `transfer/packed-c28-frontend-parallel-20261023_141754-kEe2l9`, commit
  `70347da19726`; **100/100 SHA256 checks pass**, no missing collection artifacts.
  Both GPUQ jobs succeeded and stopped their servers.
- Recomputed first-output pipelines exactly reproduce all four saved records.
  Every successful response joins to complete ordered server boundaries; all
  attempted client/server prompt and completion counts agree. No duplicates or
  retractions. Dataset, runner, adapter and measurement policies match.
- All failures are HTTP 200 with normal EOS token 1 and positive completion
  usage but no observable output event. This intervention does not resolve them.
  Candidate audit still lacks `profile_identity`; no official comparison claim.
- Canonical JSON: `results.json` → `packed_c28_frontend_comparison`. It retains
  original summaries, exact source/image identities, joins, cohort ID hashes,
  nearest-rank client statistics, stage distributions and denominator sensitivity.
  It reuses `first_output_pipeline_diagnostics`, `client_stats` and `stats` in
  `../c24-packed256-store-decode-ablation-n1-20261014/analyze.py`.
- **Decision:** retain the opt-in preparation worker as the validated remedy for
  receive-loop blocking. Keep default off pending broader qualification. The
  Triton/Kernel TTFT discrepancy is partly addressed, not closed.
- **Next:** split preparation admission wait, actual worker execution and
  dispatch; control CPU contention and request arrivals before attributing the
  remaining mode difference. Resolve the client duration anomaly before a
  throughput acceptance rerun. No new GPU jobs or runtime edits were made for
  this local analysis. Rollback remains `DSV4_ASYNC_PREPARATION=0`.


## Native C24 with preparation worker — 2026-10-23

**Native reaches 2.0156 successful requests/s, but narrowly misses the full-run
10-second TTFT-p90 SLO.** The worker-enabled Packed C28 runs remain near the same
throughput plateau. Their extra offered concurrency does not demonstrate more
throughput than this Native reference.

Source: transfer commit `988884b8`, packet
`transfer/native-c24-mf088-worker-20261023_155916`. Original run:
`results/lswb-replay/native-c24-mf088-worker-20261023_155916/20261023_162509`.
Native, TP4 H100, MF0.88, HiCache off, original fixed-active-user-pool replay,
C24, 1,200-second dispatch window, no warm-up, fresh cache and frozen GC, n=1.
Worker activation is present in the boot log. Parent `70347da19726`, clean pinned
SGLang `63291eba6137`, FlashMLA `97f3faa58450`; image `sha256:51ed908c159139e522c2254b13c2713bf2fa5c51182f21cfec908252d2831972`.
The Native and Packed C28 worker runs use the same recorded image, fork pins,
dataset hash, adapter/runner content hashes and payload/measurement options.
Parent commits differ because of evidence-only additions. Loads and run times differ.

### Full-run comparison

All cells have the preparation worker enabled; each is n=1:

| Mode / load | Successful req/s | Output tok/s | TTFT mean / p90 (s) | E2E mean (s) | TPOT mean (ms) | Device hits |
|---|---:|---:|---:|---:|---:|---:|
| Native C24 | **2.0156** | **307.88** | **4.762 / 10.287** | **11.746** | **56.71** | 97.305% |
| Packed Triton C28 | 1.9711 | 295.10 | 5.718 / 10.632 | 13.986 | 70.94 | 97.178% |
| Packed Kernel C28 | 1.9896 | 301.02 | 6.302 / 11.551 | 13.837 | 63.90 | 97.160% |

Relative to Native C24, Packed Triton requests/s is -2.21% and output tok/s
-4.15%; Packed Kernel is -1.29% / -2.23%. These small descriptive differences
are not a statistically established Native advantage. This is also not a
matched-concurrency overhead comparison. **All three full-run p90s exceed 10 s.**
Native exceeds the threshold by 286.5 ms; 11.20% of successful requests exceed
10 s. Do not round the result into a passing SLO point.

Historical Native C24 n=3 means versus this worker n=1:

| Metric | Historical mean | Worker | Descriptive change |
|---|---:|---:|---:|
| Successful req/s | 1.8125 | 2.0156 | +11.21% |
| Output tok/s | 278.20 | 307.88 | +10.67% |
| TTFT mean | 7.356 s | 4.762 s | -35.27% |
| TTFT p90 | 9.339 s | 10.287 s | +10.14% |
| E2E mean | 13.051 s | 11.746 s | -10.00% |
| TPOT mean | 50.97 ms | 56.71 ms | +11.25% |

Historical values come from `../c24-native-packed256-n3-20261018/summary.csv`;
the p90 entry is a mean of run p90s, not a pooled percentile. Their recorded
image matches, but source/run dates differ; this is not a paired attribution of
all changes to the worker. Lower mean TTFT with higher p90/TPOT is compatible
with changed closed-loop arrivals and first-output visibility; no isolated GPU
decode regression follows from client TPOT.

### Validation and pipeline

- 2,459 attempts, **2,437 successful / 22 failed**, response success **99.1053%**:
  passes the unchanged 99% gate. All failures are HTTP 200, normal EOS token 1,
  positive completion usage, no observable output. They remain failures.
- Client counts, token totals, throughput and latency distributions recompute.
  All 2,459 attempts join to unique server IDs with equal prompt/completion
  counts. All 2,437 successful first-output pipelines are complete and ordered;
  no retractions. The existing analyzer establishes stronger coverage than the
  original count-based validation sidecar.
- Post-180 cohort: 2,403 successful exact-ID joins. Detokenizer-send to API-receive
  mean **2.11 ms**, p90 **1.14 ms**; total post-prefill to frontend first output
  mean **25.13 ms**. The shared multi-second receive stall is absent here.
- Duration **1,209.063 s**; last response at 1,209.011 s, last activity at
  1,209.038 s. Only **51 ms** follows the final response, so the anomalous
  92-second tail seen in Kernel sync is not present.
- The three worker runs share 2,280 successful body/prompt-matched request IDs
  received after 180 s. A stricter 1,773-request cohort also matches output and
  cached counts; Native mean TTFT/E2E is 4.154/8.590 s, Triton C28 4.801/10.044 s,
  Kernel C28 5.463/10.118 s. These remain different-load, different-arrival
  comparisons and cannot establish per-operation overhead or accepted SLO goodput.
- Minimal transfer includes no original SHA256 manifest or terminal GPUQ snapshot.
  Local artifact hashes are saved for future reproduction, not relabeled as
  verification against VDI originals. The saved GPUQ job file is a queued
  submission snapshot; client completion, replay exit 0 and shutdown logs supply
  run-completion evidence. Candidate audit lacks `profile_identity`.

Canonical record: `results.json` → `native_c24_worker_validation`, reusing the
same existing analysis functions as C28. **Decision:** evidence supports a shared
frontend improvement and near-neutral relative Packed throughput in these runs;
no SLO throughput advantage is established. For the next SLO probe, reduce Native
concurrency with worker/source/image/protocol frozen; matched C24 Packed worker
controls remain unrun. No runtime edits or GPU jobs were made during analysis.
Rollback remains `DSV4_ASYNC_PREPARATION=0`.
