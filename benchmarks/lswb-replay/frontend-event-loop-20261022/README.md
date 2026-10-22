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
with every batch. Instrumented full-model coverage remains pending.

## Evidence and rollback

### Follow-up candidate: keep chat preparation off the receive loop

The first completed Native diagnostic locates its multi-second output residence
between detokenizer forward-start and API receive. The Packed leg and exact joined
comparison remain pending; this observation does not establish a Kernel-specific
regression. Together with the source-isolated blocking reproduction, it motivates
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
