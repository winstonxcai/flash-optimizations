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

## Evidence and rollback

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
