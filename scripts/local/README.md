<!-- Author: Winston Cai. -->

# Local serving and replay commands

Run from the repository root on the GPU node. Modes: `native` = Native;
`triton` = Packed with Triton packing/reconstruction; `kernel` = Packed with
optimized kernels. All three use the same Docker image.

**GPUQ is only for queueing jobs on shared GPU servers (including the VDI host).**
It reserves exclusive GPUs; it is not required for local CPU checks, dedicated
machines or Modal. On shared machines, run GPU commands inside your GPUQ job or
active lease. Bare serving/capture examples below assume you already hold an
allocation. Build the image and prepare inputs before queueing GPU work.

## Isolated latent-KV capture

Capture reuses the **same `remnant:v0.5.18` image as serving**, but starts a
separate, short-lived container; it never runs inside or replaces the production
`remnant` container. No capture-specific image, tag or label is required.
The shell entry point owns all Docker operations: local image/component checks,
mounts, GPU selection/availability, container launch, timeouts and exact-container
cleanup. It does not source production environment defaults. It requires Docker,
`nvidia-smi` and GNU `timeout`.

### Prepare inputs on CPU

Ask Winston for the frozen, audited OpenHands cohort: `requests.jsonl`,
`manifest.json` and `audit.json` in one directory. These contain request/tool
payloads and stay outside Git; a clone does not include them. Use the original
files together, not a newly sampled cohort or manually edited receipts. The
local checkpoint must include all 48 shards, tokenizer/configuration files and
the original Hugging Face download metadata for revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

Set absolute **host** paths below. The capture parent may exist, but its `scratch`
child must be new. This GPU-free command uses the serving image's existing
`calibrate.preflight` to validate inputs and generate `capture-config.json`,
`replay.jsonl` and the serving-tokenization audit; it does not load model weights.
The two-file source record is required by the existing shared-capture audit,
not by training/evaluation compatibility checks.

```bash
python3 scripts/sources.py .
CAPTURE_IMAGE=$(docker image inspect --format '{{.Id}}' "${IMAGE:-remnant:v0.5.18}")
CAPTURE_MODEL=/data/models/DeepSeek-V4-Flash-0731
CAPTURE_COHORT=/data/pruning/cohort
CAPTURE_BASE=/data/capture
mkdir -p "$CAPTURE_BASE"
docker run --rm -i --pull never --network none \
  --mount "type=bind,src=$CAPTURE_COHORT,dst=/cohort,readonly" \
  --mount "type=bind,src=$CAPTURE_MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$CAPTURE_BASE,dst=/capture-data" \
  --entrypoint python3 "$CAPTURE_IMAGE" - <<'PY'
import hashlib
import json
from pathlib import Path
from sglang.benchmark.remnant.calibrate import preflight

root = Path("/capture-data/scratch")
preflight(Path("/cohort"), Path("/model"), root, storage_schema=2, audit_shared=True)
source = Path("/sgl-workspace/sglang-remnant/python/sglang")
names = ("srt/remnant/capture.py", "benchmark/remnant/calibrate.py")
hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
(root / "source-manifest.json").write_text(json.dumps({"source_sha256": hashes}, indent=2) + "\n")
PY
```

### Capture on four allocated GPUs

Keep the same resolved image ID and prepared paths. On a shared GPU server,
execute this inside the GPUQ allocation, using its assigned GPU IDs:

```bash
CAPTURE_GPUS=0,1,2,3 bash scripts/local/capture-latent.sh run \
  "$CAPTURE_IMAGE" "$CAPTURE_MODEL" \
  "$CAPTURE_BASE/scratch" "$CAPTURE_BASE/published" 3600 30312
```

Preparation above selects schema 2, audited capture and concurrency two. For a
later shared-only rollout (`audit_shared=false`), obtain a matching
`shared-audit-proof.json` from a completed audited pilot; do not disable auditing
or copy an unrelated proof to bypass this requirement. Scratch
and publication destinations must be distinct and writable. Scratch must retain
50 GiB free; existing client staging/publication limits and resume checks remain.
Do not use a read-only mount such as the server's `/mnt/public_data` as output.

The shared Dockerfile copies the fork's capture observer, runtime hooks and
calibration client (`calibrate.py`) alongside the compiled extensions. Use an
up-to-date local image from the normal [image workflow](#update-the-image);
rebuild before capture if those sources changed, not inside a running GPU job.
The shell verifies these components in a GPU-free container, then calls the client's
data preflight against the actual mounts. Missing prerequisites fail;
capture never builds, pulls, patches or updates an image. Container launches use
`--pull never`; no kernel build or production image/configuration change occurs.
Capture stays Native, TP4/concurrency two, seed 42, with prefix reuse, overlap
scheduling and CUDA graphs disabled for this capture process only. Sharing the
image does not mean sharing production server settings or GPU allocations.

`3600` is the **total container budget**, including model startup, readiness,
replay and writer drain—not an additional hour after loading. The client gets
only the remaining time. Timeout/interruption stops the owned server process
group and exact capture container, with bounded cleanup; it never stops another
server. Logs and completed publication/progress artifacts remain available after
failure. Selected GPUs must already be allocated to you and unoccupied; the
launcher does not acquire GPUQ resources or terminate competing processes.

`bash scripts/local/capture-latent.sh --help` is safe locally and launches
nothing. The same shell script runs inside the container: it rechecks mounted
data, starts Native, polls readiness, and gives `calibrate.py capture-client`
only the remaining budget. It owns and cleans up the server and client process
groups. Configuration, audit proof, writable storage, free-space and port checks
live in `calibrate.py capture-preflight`; tensor collection and publication are
unchanged. No separate Python orchestration module is required.

Using the shared image requires no capture-specific kernel build.
Shared-only rollout proofs are source-bound and must match its calibration
client and collector. A stale proof fails rather than being bypassed.

After successful capture, check the real publications with the existing integration
test in a **GPU-free** container using the same image. For this test the image must also
include the training test dependencies (Torch, NumPy and Matplotlib). This verifies
every configured issue/layer, quotas, all TP heads and Native output/LSE; audited
captures additionally verify exact cross-rank sharing. A missing or corrupt
publication fails. Audited capture requires at least two issues for the existing
shared-audit proof, which this check writes into the publication directory.
No model is loaded by this check:

```bash
docker run --rm --pull never --network none \
  --mount "type=bind,src=$PWD,dst=/capture-repo,readonly" \
  --mount "type=bind,src=$CAPTURE_BASE/published,dst=/published" \
  -e PYTHONPATH=/opt/sglang-runtime-fixes:/sgl-workspace/sglang-remnant/python:/capture-repo \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e REMNANT_CAPTURE_VALIDATION_ROOT=/published \
  -e REMNANT_CAPTURE_VALIDATION_RUNNER=/sgl-workspace/sglang-remnant/python/sglang/benchmark/remnant/calibrate.py \
  --entrypoint python3 "$CAPTURE_IMAGE" \
  -m unittest training.tests.test_integration.PublishedCaptureTests -v
```

## Update the image

Update outside running GPUQ jobs. Preserve dirty fork work before synchronizing:

```bash
git pull --ff-only origin remnant
git submodule update --init --recursive --checkout
bash scripts/local/container.sh rebuild
```

Rebuild after Dockerfile or fork source changes. The tag stays
`remnant:v0.5.18`; successful rebuilds replace the `remnant` container and stop
its servers. Host script changes need only a pull. To start an existing image:

```bash
bash scripts/local/container.sh up
```

Local container, serving and benchmark entry points reject dirty or pin-mismatched
forks (including nested submodules) before external work. Stop commands still
work. For a manual Docker build, run `python3 scripts/sources.py .` first. These
checks validate host source, not an already-built image; rebuild after source
changes. The direct in-image benchmark client relies on its launcher's preflight.

The `typing_extensions` overlay installs offline from the small Git-tracked
wheel in `third_party/wheels/`, with its upstream SHA-256 enforced. This build
step needs no PyPI or mirror access; a normal Git pull supplies the wheel.
The base image must still be available locally or through its registry.

## Start, check and stop a server

Choose available GPUs and distinct ports. These examples use GPUs 4–7;
for GPUs 0–3 use `GPUS=0,1,2,3`, `PORT=30212`, `MASTER_PORT=29638`.
Acquire a GPUQ allocation before serving; the queued example below handles it.

```bash
export GPUS=4,5,6,7 PORT=30213 MASTER_PORT=29639
export MODE=triton                    # native | triton | kernel

bash scripts/local/serve.sh "$MODE"
curl -fsS --max-time 10 "http://127.0.0.1:$PORT/health"
tail -n 80 "logs/serve_${MODE}.log"

# Stop later:
bash scripts/local/serve.sh "$MODE" stop
```

Startup waits for model loading and graph creation. Starting another mode on
this port replaces the server owned by these scripts. Server logs are overwritten
on the next launch of that mode; save them with the replay results first.

## Store/decode ablation

`MODE` selects Packed decode; optional `STORE_IMPLEMENTATION=triton|kernel`
selects storage independently. Unset it to retain the existing mode defaults.
The override is ignored for Native. With updated host Python source selected:

```bash
STORE_IMPLEMENTATION=kernel bash scripts/local/serve.sh triton
# Original store with Direct Packed decode:
STORE_IMPLEMENTATION=triton bash scripts/local/serve.sh kernel
```

Use distinct replay tags for mixed modes. `server-info.json` records the explicit
store override; scheduler startup logs record the resolved store and decode modes.

## Random-text serving benchmark

Like `bench-lswb.sh`, `bench-serving.sh` is a client for an **already-running**
server. Start and stop the server separately. The client never changes its
cache mode, launches a server, or stops one after failure.

```bash
bash scripts/local/serve.sh native
bash scripts/local/bench-serving.sh native-128k "$PORT" 131072 2048 28
bash scripts/local/serve.sh native stop

bash scripts/local/serve.sh triton
bash scripts/local/bench-serving.sh packed-128k "$PORT" 131072 2048 28
bash scripts/local/serve.sh triton stop
```

Arguments are `<tag> [port] [input_tokens] [output_tokens] [concurrency|auto]`.
Defaults are the configured port, 32768 input tokens, 2048 output tokens and C8.
The tag labels artifacts; the actual mode and survivor counts come from saved
`server-info.json`. Use the same C for a fair comparison. Replace C with `auto`
to derive it from that server's allocator pool divided by input + output
length; this is a capacity measurement, not an SLO limit. The old `fair`/`max`
orchestration forms are removed.

One warm-up wave of C requests precedes 3C measured requests, using exact
random lengths, `SEED` (default 42), and a cache flush before each wave.
Both waves must complete with the requested output lengths and no errors.
The 3C requests are one measurement, not three independent runs. Decode graph
coverage, context length and request limits are checked before client execution.

The `random` workload samples ShareGPT text, so SGLang may download its backing
dataset even though prompt lengths are synthetic. On restricted-network hosts,
prepare the real ShareGPT JSON before reserving GPUs and pass its local path:

```bash
export BENCH_DATASET_PATH="$PWD/results/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
bash scripts/local/bench-serving.sh native-smoke "$PORT" 32768 256 2
```

In Docker mode this is an absolute **host** path under `HOST_REPO`; the client
maps it into the existing container mount. The script checks the JSON and records
its SHA-256 in `config.json`, and both waves receive `--dataset-path`. Pass
`BENCH_DATASET_PATH` explicitly through GPUQ `--env` when submitting jobs. Keep
the file outside Git and identical across modes. Without this option, SGLang's
default Hugging Face download/cache behavior is preserved.

The VDI default (`BENCH_EXECUTION=docker`) runs the official SGLang client in
the existing container. Inside a prepared environment, including Modal, use:

```bash
BENCH_EXECUTION=direct bash scripts/local/bench-serving.sh native-128k 30211 131072 2048 28
```

`BENCH_TIMEOUT_S` bounds each client wave (default 3600 s). The caller owns the
server/allocation deadline and cleanup. Use a dedicated idle server; parallel
clients against the same port would invalidate the comparison.
Results use unique directories under `$RESULTS_HOST/serving/<tag>/`, including
the resolved configuration, script/client source hashes and both wave records.
Docker image identity is recorded **at collection time**, not asserted as boot
identity; preserve the server boot's image/source record separately.

Modal's `benchmark_serving` wrapper owns its server boot, readiness, total job
budget and process-group cleanup, then calls this same client. The separate
capacity and image scripts remain retired.

## LSWB replay without profiling

### Repository-owned replay client

`bench-lswb.sh` now defaults to the two local production files:
`business_replay/runner.py` (dataset/manifest loading, fixed active-session
scheduler, audit and reports) and `business_replay/adapters/openai_sse.py`
(OpenAI SSE requests, streaming timing, usage and cache details). Both paths
are under `scripts/local/`. No external code checkout or package initializers
are required. Tests, fixtures and corpus are not copied. The host needs
Python 3.10+ and `requests`, but no model/tokenizer packages. Local integration author:
Winston Cai; this is not a copied-code authorship claim.

Keep the large raw file at `data/longswebench.jsonl` on the GPU server; it need
not be downloaded locally. Prepare it once on CPU, then reuse the resulting
records and manifest for all modes. This creates an additional on-disk copy of
the records without modifying the raw file or running a server:

```bash
python3 scripts/local/convert_lswb.py --format replay \
  --input data/longswebench.jsonl --output-dir data/lswb-replay
# After starting the desired server separately:
bash scripts/local/bench-lswb.sh "${MODE}-unprofiled-c24" "$PORT" 24 1200
```

`eval-env.sh` resolves absolute local runner/adapter defaults independently of
`REPLAY_DIR`; explicit nonempty `REPLAY_RUNNER` and `REPLAY_ADAPTER` overrides
still work. Data defaults are repo-root `data/lswb-replay/records/` and
`data/lswb-replay/manifest.json`; a clone does not supply them. Preparation
preserves original record bytes, IDs, sessions, timestamps, messages and tools;
the existing runner validates records and sorts them by its existing rules.
It refuses overwrites and publishes only after validation. New filenames and
record serialization can change workload hashes relative to older manifests;
reuse one prepared collection across new comparisons rather than claiming
historical hash identity. Optional `REPLAY_DIR` retains legacy data defaults
at `datasets/h20-dsv4pro/longcodebench_openai` and
`cache/official-longswebench/longswebench-openai-v1.json` beneath that root;
it need not contain code or be the original directory. Explicit data overrides
take precedence. Relative overrides retain their meaning relative to
`REPLAY_DIR` when set, otherwise the caller's working directory. A relative
`REPLAY_DIR` is anchored to the caller's working directory. The launcher no
longer changes directory, and missing/unreadable paths fail with the variable
and path to fix. The runner still validates manifest content.
Raw `data/longswebench.jsonl` must pass through `--format replay` first;
the bench-serving conversion below is a different format. The reader is unchanged.

This remains a client of an **already-running server**: no weight loading or
server lifecycle changes. C24 means **up to 24 active sessions**, each sending
requests strictly serially and retaining its slot until exhausted. The time
window stops new dispatch and drains in-flight work, so wall time can exceed
1,200 seconds. Preflight, cache flush, flags and artifact collection are unchanged.

**Original import provenance:** initially copied byte-for-byte from
`/home/jovyan/wenyuhong/benchmarks/harnesses/business_replay/` (historical source
only, not a runtime dependency). Checkout HEAD was
`d2b0db9bb8d11c248e6be8054c25ce2f573fc94d`; both production files had uncommitted
modifications, so the hashes identify the actual working-tree snapshot, not a
pristine export of that revision. Original docstrings/notices are retained.
No explicit source license/author was found; no license or authorship is invented.
The unchanged `source_identity()` records this repository and local file hashes
when run.

```text
76ca1e4c25452e1fd163c957a4b262169fd1bba0d8bc3d9ccd3f45b232068bc4  runner.py
363d6adffde63bfe54c483d629cca780d43689ec101a778f25fc110060f00b13  adapters/openai_sse.py
```

**Historical import validation:** static file/path review, byte/hash comparisons,
`bash -n` on the two modified shell files and `git diff --check` only. No copied
Python execution/imports, `--help`, compilation, unittest, smoke, network,
server or GPU checks were performed. Executable configuration checks were also
skipped. Runtime correctness and artifact production remain unverified. Per the
initial minimal-file scope, no tests or fixtures were included. The hashes above
identify that original import; subsequent queue-timing changes have new source
hashes recorded automatically in run manifests.

### Optional request-slot FIFO replay

**Question:** with the same participating conversation population, does Packed
reduce repeated prefill and increase successful requests/s under a queue-inclusive
first-output SLO? Native may tune outstanding concurrency without reducing that
population. This is a new, closed-loop workload protocol, not a production arrival
trace or a live-agent quality test. No throughput advantage is assumed.

Use the existing runner and adapter with `--scheduler request-slot-fifo` and
`--session-population P`. `--max-concurrency C` now limits outstanding requests;
require `P >= C > 0` and at least P selected sessions. Sessions enter in recorded
first-turn order. Each completion returns its session's next recorded turn to
the FIFO tail; exhausted sessions are replaced from the remaining cohort. One
request per session may be outstanding. Actual dispatch order depends on response
completion order, so equality of the global dispatch sequence is not promised.

The initial version requires `--arrival-mode immediate`, a positive fixed window,
and no warmup. Immediate readiness after each response means zero tool think time.
Timestamp pacing and independent offered-rate/backlog experiments are deferred.
Pending sessions outside the participating population have not yet arrived and
are not included in its SLO. Messages, tools, output caps, EOS and SSE parsing
retain their existing definitions; generated answers do not alter later prompts.

**Measurement:** the new protocol is `business-request-slot-replay-v1`. HTTP
`ttft_ms` and `latency_ms` retain their historical meanings. New `ready_ttft_ms`
and `ready_latency_ms` include FIFO waiting, worker dispatch, file reading,
payload preparation and HTTP time, starting at admission or preceding response
observation. The SLO uses ready-to-first-output **< 10,000 ms**, including invalid
responses as violations. Ready requests unresolved at cutoff are right-censored:
the summary reports conservative pass/fail bounds, or `inconclusive`; it does not
silently drop them. `lswb-row.sh` uses this SLO status for the new protocol.

Requests/s and token throughput count successful responses completed inside the
fixed window divided by that window. `counts.successful_in_window` is the
throughput numerator; `counts.successful`, `failed`, `attempted` and `success_rate`
cover every dispatched request, including drain. Cache aggregates and token totals
cover successful window completions. Per-request records also preserve drain
responses. No new requests or sessions are admitted after cutoff. A CPU process
pool initializes all workers and imports adapters before the measurement clock;
workers are terminated and joined before runner exit. `--drain-timeout` defaults
to 120 seconds and bounds collection of dispatched requests. Timeouts are failures,
not excluded samples. Server lifecycle remains owned by the existing launcher.

With an **already-running server inside your GPUQ allocation**, for a first
P24/C20 screen at the server's configured memory fraction:

```bash
REPLAY_SCHEDULER=request-slot-fifo REPLAY_SESSION_POPULATION=24 \
  bash scripts/local/bench-lswb.sh kernel-mf082-p24-c20 "$PORT" 20 1200
```

Optional launcher overrides: `REPLAY_DRAIN_TIMEOUT=120` and
`REPLAY_TTFT_SLO_MS=10000`. Policy settings are validated on CPU before server
health/cache operations. Changing the SLO must be explicit in comparisons.
Population is validated against the selected dataset by the runner. Keep the
population, cohort, memory budget and readiness policy equal across modes. Start
with Native and Packed Kernel at P24/C20, then consider P20/P32 controls and
independent C tuning. Repeat promising cells to n=3; stop expansion if retention
does not improve or queue-inclusive latency exceeds the SLO. Preserve negative
results alongside existing fixed-session replay evidence.

**CPU checks:**

```bash
python3 -m unittest discover -s scripts/local/business_replay -p 'test_*.py' -v
python3 -m unittest scripts.modal.tests.LocalScriptTests scripts.modal.tests.ServingClientTests
bash -n scripts/local/bench-lswb.sh scripts/local/lswb-row.sh
```

Tests use deterministic FIFO cases and controlled fake CPU senders, including the
real CLI/spawn/termination path. They cover ordering, population/concurrency,
failure cleanup, fixed-window/drain accounting, queue timing, SLO censoring and
the existing default policy. They do not establish H100 performance, real SSE
integration, production traffic realism, or cache benefits. GPU qualification is
a separate gate. The changed implementation lives in `runner.py`, optional policy
module `request_slots.py`, and the adapter's opt-in timing metadata; tests live
beside the owning client rather than adding another benchmark harness.

**Risk/rollback:** FIFO rotation changes locality and can increase latency for
both modes. The new SLO includes client waiting and is distinct from historical
HTTP-only SLO results. Keep result tags/protocols separate. To disable the candidate,
unset `REPLAY_SCHEDULER`, `REPLAY_SESSION_POPULATION`, `REPLAY_DRAIN_TIMEOUT` and
`REPLAY_TTFT_SLO_MS`; the default remains `business-user-replay-v2`. No SGLang,
FlashMLA, allocator, model or image changes are involved.

### One-time conversion for SGLang bench-serving

Question: how does Native versus Packed throughput change when the same recorded
LongSWE-derived requests use request-level concurrency instead of the replay's
fixed active sessions? Convert the frozen cohort once, then reuse identical
messages, tools, ordering and recorded output caps for all three modes at C24.
Respect EOS and use temperature zero. This is a serving comparison, not a task
quality evaluation or the 1,200-second replay.

`convert_lswb.py` only owns the one-time CPU conversion. It hashes the
original raw JSONL and writes `requests.jsonl` plus a provenance `manifest.json`
under ignored `data/`. It preserves full structured messages and tools, rejects
malformed rows and duplicate IDs, and refuses to overwrite an existing
collection. Historical responses, usage and request/session identity
are excluded from API payloads. Identity, hashes and recorded lengths remain in
the provenance file. The original archive is unchanged.

```bash
python3 scripts/local/convert_lswb.py \
  --input "$PWD/data/longswebench.jsonl" \
  --output-dir "$PWD/data/lswb-bench-serving"
```

Rows use SGLang's `openai` loader: `messages`, optional `tools`, `max_tokens`,
`temperature`, `ignore_eos`, streaming usage and cache-reporting options. An
additional `prompt_len` metadata field supplies recorded `usage.prompt_tokens`
estimates to the fork's loader because V4's tokenizer has no Hugging Face chat
template. Missing/zero prompt counts use an explicit zero estimate, with their
count recorded in the manifest. Output caps use `usage.completion_tokens`; an
absent/zero value defaults to 256, matching the replay adapter. Every fallback is
recorded. Request IDs and session IDs come from `id` and `token_name`.
This metadata is stripped by the loader before sending the original chat to the
server. Prompt-token throughput and client cache-hit fractions are consequently
estimates; prefer requests/s and reported completion-token throughput.

The loader compatibility fix and tool-call-only TTFT/streaming-usage fixes live
in the SGLang benchmark client. They are independent of the converter and do not
change model serving. Before using the setup on another host, publish the fork
commit, update its parent pin and synchronize the checkout. An older image's
benchmark client does not support the recorded length metadata; select the
updated mounted fork explicitly. The existing `bench-serving.sh` remains the
random-text client and is not used for this experiment.

Before reserving GPUs, exercise the actual loader in the serving image on CPU:

```bash
export SGLANG_ROOT_CT="/mnt/host_root$PWD/third_party/sglang"
export SGLANG_PY="$SGLANG_ROOT_CT/python"
. scripts/local/env.sh
docker exec -i -e CUDA_VISIBLE_DEVICES= \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT" \
  "$CONTAINER" python3 - "$REPO_CT/data/lswb-bench-serving" "$MODEL_CT" <<'PY_LOADER'
import hashlib, json, sys
from pathlib import Path
from sglang.benchmark.datasets.openai_dataset import sample_openai_requests
from sglang.benchmark.utils import get_tokenizer
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
path = root / "requests.jsonl"
assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["dataset_sha256"]
rows = sample_openai_requests(str(path), manifest["requests"], get_tokenizer(sys.argv[2]))
assert len(rows) == manifest["requests"]
for row, record in zip(rows, manifest["records"]):
    assert row.output_len == record["max_tokens"]
    assert row.prompt_len == (record["recorded_prompt_tokens"] or 0)
    assert "prompt_len" not in row.extra_request_body
print(f"CPU loader PASS: {len(rows)} requests; no weights loaded")
PY_LOADER
```

Invoke SGLang's CLI directly against an already-running, dedicated server:

```bash
mkdir -p "$PWD/results/serving"
RUN_ROOT=$(mktemp -d "$PWD/results/serving/lswb-c24-XXXXXX")
LSWB_REQUESTS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["requests"])' \
  "$PWD/data/lswb-bench-serving/manifest.json")
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" > "$RUN_ROOT/server-info.json"
docker exec -i -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT" \
  "$CONTAINER" timeout --signal=TERM --kill-after=5s 10800s \
  python3 -m sglang.benchmark.serving \
  --backend sglang-oai-chat --host 127.0.0.1 --port "$PORT" \
  --model "$MODEL_CT" --served-model-name "$MODEL_NAME" --tokenizer "$MODEL_CT" \
  --dataset-name openai --dataset-path "$REPO_CT/data/lswb-bench-serving/requests.jsonl" \
  --num-prompts "$LSWB_REQUESTS" --max-concurrency 24 --request-rate inf \
  --warmup-requests 0 --flush-cache --disable-ignore-eos --cache-report \
  --output-file "$(to_ct "$RUN_ROOT/measured.jsonl")" --output-details --seed 42 \
  > "$RUN_ROOT/client.log" 2>&1
```

First smoke all three modes with
`--num-prompts 16 --max-concurrency 2` in separately labeled result folders.
Check mode/counts, decode graph coverage, request limits and context lengths
against the running server. Stop on loader, source or configuration failures;
do not drop requests, flatten history or change output policy to pass. Require
at least 99% successful requests in a full comparison; retain all failures and
inspect reported usage, output caps and TTFT before interpreting performance.
The CLI's exit code alone does not validate a result. CPU checks cannot establish
GPU correctness or agent quality.

For GPUQ, reserve four GPUs per mode, use its assigned IDs, and use distinct
ports and boot/result directories. Two TP4 modes may run concurrently on eight
GPUs. The job caller owns server startup and cleanup on every exit; neither the
converter nor the CLI stops a server. Reuse the one converted collection for
all subsequent comparisons. Do not rerun conversion between modes.

Against the running server, run for 1,200 seconds at concurrency 24 or 28:

```bash
bash scripts/local/bench-lswb.sh "${MODE}-unprofiled-c24" "$PORT" 24 1200
bash scripts/local/bench-lswb.sh "${MODE}-unprofiled-c28" "$PORT" 28 1200
```

Each replay checks decode graph coverage, waits for idle and flushes the prefix
cache. Results go to `results/lswb-replay/<tag>/<timestamp>/`. Replay completion
leaves the server running.

## Request-level TTFT diagnostic: Triton and Kernel C24

Use the updated Python fork on the host mount; the launcher probes the
`first_output_ts` field before replacing a server. These Python-only changes
do not require rebuilding native extensions. Transfer the reviewed local
changes to the VDI before using this command. Keep pulls outside the GPUQ job.

One job reserves GPUs 4–7 for two sequential, unprofiled 1,200-second runs.
Both use seed 20260919, the same replay manifest, and the existing cache flush.
GPUQ waits for the specified GPUs; choose another block/port if needed.
For shared service-account execution, the examples scope Git trust to the parent,
both forks and nested CUTLASS for the source preflight. No global Git setting or
wildcard trust is added.

```bash
cd /home/jovyan/winstonxcai/flash-optimizations
mkdir -p logs
gpuq run --user winstonxcai --project flash-optimizations \
  --gpus 4 --gpu-ids 4,5,6,7 --cwd "$PWD" \
  --env GPUS=4,5,6,7 --env PORT=30213 --env MASTER_PORT=29639 \
  --env SERVER_SEED=20260919 \
  --env REPLAY_DATASET="$PWD/data/lswb-replay/records" \
  --env REPLAY_MANIFEST="$PWD/data/lswb-replay/manifest.json" \
  --output "$PWD/logs/ttft-diagnostic-$(date +%Y%m%d_%H%M%S)" \
  --workload-key remnant-ttft-c24 --expected-duration 50m \
  --timeout 5400 --detach -- bash -lc '
set -euo pipefail
export GIT_CONFIG_COUNT=4
export GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1="$PWD/third_party/sglang"
export GIT_CONFIG_KEY_2=safe.directory GIT_CONFIG_VALUE_2="$PWD/third_party/flashmla"
export GIT_CONFIG_KEY_3=safe.directory GIT_CONFIG_VALUE_3="$PWD/third_party/flashmla/csrc/cutlass"
export SGLANG_ROOT_CT="/mnt/host_root$PWD/third_party/sglang"
export SGLANG_PY="$SGLANG_ROOT_CT/python"
export LOG_HOST="$PWD/logs/ttft-c24-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_HOST"
docker exec -e "PYTHONPATH=$SGLANG_PY" "${CONTAINER:-remnant}" \
  python3 "$SGLANG_ROOT_CT/test/registered/unit/observability/test_req_time_stats.py"
mode=
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [ -n "$mode" ]; then
    bash scripts/local/serve.sh "$mode" stop || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
trap "exit 130" INT
trap "exit 143" TERM
for mode in triton kernel; do
  export REQUEST_METRICS_DIR="$LOG_HOST/request-metrics-$mode"
  bash scripts/local/serve.sh "$mode"
  bash scripts/local/bench-lswb.sh "$mode-timing-c24" "$PORT" 24 1200
  bash scripts/local/serve.sh "$mode" stop
  mode=
done
'
```

Each replay saves `request-timings.jsonl`, raw `request-metrics/*.log`,
`request-timing-validation.json`, `collection-config.json`, `replay-window.json`,
and the replay slice of `server.log`. Partial exports survive interruption.
Collection waits up to 30 seconds for asynchronous writes and fails on bad
timestamps, duplicate IDs, exporter errors, or insufficient record counts.
Count coverage is a lower bound; it does not verify the client/server ID join.
Keep the client's raw request records to match OpenAI response IDs when available.

With metrics enabled, the existing exporter supplies `queue_time`,
`forward_entry_time`, `prefill_finished_time`, `request_received_ts`,
`request_finished_ts`, and token/cache counts. `first_output_ts` records the
tokenizer's first output batch, before client SSE delivery. In analysis:

- Initial scheduler wait: `queue_time` (seconds).
- Prefill elapsed: `prefill_finished_time - forward_entry_time`, including
  waits between chunks; this is not pure GPU execution time.
- Post-prefill delay: `first_output_ts - prefill_finished_time`.
- Server first-output latency: `first_output_ts - request_received_ts`.

Client p90 TTFT remains the ≤10-second SLO. Compare components for the same
slow requests; component p90s cannot be added. Same server seed does not fix
closed-loop arrival times or output/request sets. No automatic retries are used.

## All five Nsight Systems runs

Use the exports above. Capture is inactive during loading and graph creation.
Each replay captures seconds 300–360 and 900–960 of its 1,200-second window.
Run this block in Bash:

```bash
set -e
export NSYS_GRAPH_TRACE=graph          # lower-overhead whole-graph controls
for MODE in native triton kernel; do
  bash scripts/local/serve.sh "$MODE" profile
  CONCURRENCIES=24
  [ "$MODE" = native ] || CONCURRENCIES="24 28"
  for C in $CONCURRENCIES; do
    TAG="${MODE}-graph-nsys-c${C}"
    bash scripts/local/bench-lswb.sh "$TAG" "$PORT" "$C" 1200 300:360,900:960
    RUN=$(find "results/lswb-replay/$TAG" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
    cp "logs/serve_${MODE}.log" "$RUN/server.log"
  done
done
```

Reports are separate `.nsys-rep` files under `logs/nsys-<mode>-*/`;
`nsys-windows.jsonl` records capture acknowledgements in each replay directory.
The final Kernel server stays running. Use `serve.sh kernel stop` when finished.

## Short graph-node captures at C24

`serve.sh <mode> profile` now defaults to `NSYS_GRAPH_TRACE=node` and enables
scheduler/operation NVTX before SGLang imports. Startup probes PyTorch's built-in
`torch.cuda.nvtx` and the updated hooks before replacing a server. No external
package, package mirror or Docker change is needed. Node mode exposes graph kernels;
`NSYS_GRAPH_TRACE=graph` retains lower-overhead controls.

`scheduler.batch` records step/rank, mode, actual request counts, prefill tokens
and CPU context statistics. Nested `decode.graph_replay` records the resolved
raw and padded batch sizes. Use range nesting and CUDA API correlation to join
them; CPU range duration is not GPU-step duration. Missing metadata is marked
null/unavailable, with no GPU scalar reads or added synchronization. Normal
non-profile serving remains unchanged.

Synchronize parent scripts and fork edits to the GPU host first. Either rebuild
the image, or use the explicit checked-out Python-source override below, which
reuses installed dependencies and compiled extensions. The import preflight
prints the active hook path. This override is appropriate for these Python-only
hooks, not for unbuilt native extension changes.

GPUQ reserves the same four physical GPUs for three sequential legs. Choose
GPU IDs and an unused port; GPUQ waits for exclusive availability. Do not submit
against an existing overlapping lease. `--output` is a directory.

```bash
cd /home/jovyan/winstonxcai/flash-optimizations
mkdir -p logs
read -rp 'Four physical GPU IDs for GPUQ to reserve (comma-separated): ' GPUS
gpuq run --user winstonxcai --project flash-optimizations \
  --gpus 4 --gpu-ids "$GPUS" --cwd "$PWD" \
  --env GPUS="$GPUS" --env PORT=30213 --env MASTER_PORT=29639 \
  --env REPLAY_DATASET="$PWD/data/lswb-replay/records" \
  --env REPLAY_MANIFEST="$PWD/data/lswb-replay/manifest.json" \
  --output "$PWD/logs/nsys-node-c24-$(date +%Y%m%d_%H%M%S)" \
  --workload-key remnant-node-c24 \
  --expected-duration 45m --timeout 5400 --detach -- bash -lc '
set -euo pipefail
export GIT_CONFIG_COUNT=4
export GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1="$PWD/third_party/sglang"
export GIT_CONFIG_KEY_2=safe.directory GIT_CONFIG_VALUE_2="$PWD/third_party/flashmla"
export GIT_CONFIG_KEY_3=safe.directory GIT_CONFIG_VALUE_3="$PWD/third_party/flashmla/csrc/cutlass"
export NSYS_GRAPH_TRACE=node
export SGLANG_ROOT_CT="/mnt/host_root$PWD/third_party/sglang"
export SGLANG_PY="$SGLANG_ROOT_CT/python"
run_id=$(date +%Y%m%d_%H%M%S)
mode=
cleanup() {
  rc=$?
  trap - EXIT
  if [ -n "$mode" ]; then
    bash scripts/local/serve.sh "$mode" stop || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
trap "exit 143" TERM
trap "exit 130" INT
sha256sum scripts/local/{serve,run-server,bench-lswb}.sh \
  third_party/sglang/python/sglang/srt/utils/nvtx_utils.py \
  third_party/sglang/python/sglang/srt/managers/scheduler.py \
  third_party/sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py \
  > "logs/nsys-node-c24-source-$run_id.txt"
for mode in native triton kernel; do
  bash scripts/local/serve.sh "$mode" profile
  tag="node-$mode-c24-$run_id"
  bash scripts/local/bench-lswb.sh "$tag" "$PORT" 24 420 300:315
  runs=(results/lswb-replay/"$tag"/*)
  test "${#runs[@]}" = 1
  cp "logs/serve_$mode.log" "${runs[0]}/server.log"
  bash scripts/local/serve.sh "$mode" stop
  mode=
done
'
```

Each leg flushes the prefix cache, replays for 420 seconds and requests a
15-second capture after 300 seconds. These are diagnostics, not full 1200-second
SLO comparisons. Resolved seeds/settings remain in the server artifacts; do not
claim random seeds match without checking them. Normal error/termination cleanup
stops the owned server; hard kills require operator verification.

Before interpreting timing, verify four-rank coverage, graph-node/kernel
events, both batch annotations, and diagnosed collection warnings. Node tracing
adds overhead; use unprofiled controls for serving acceptance.

## Queue an unprofiled control with GPUQ

Ensure the image is ready first. Change `MODE` and `CONCURRENCY` to select the
control. The job stops its server on exit and saves its server log. `--output`
is a directory. Git trust is scoped to the parent, both forks and nested CUTLASS.

```bash
mkdir -p logs
gpuq run --user winstonxcai --project flash-optimizations \
  --gpus 4 --gpu-ids 4,5,6,7 --cwd "$PWD" \
  --env GPUS=4,5,6,7 --env PORT=30213 --env MASTER_PORT=29639 \
  --env REPLAY_DATASET="$PWD/data/lswb-replay/records" \
  --env REPLAY_MANIFEST="$PWD/data/lswb-replay/manifest.json" \
  --env MODE=native --env CONCURRENCY=24 \
  --output "$PWD/logs/native-unprofiled-c24" \
  --workload-key remnant-native-unprofiled-c24 \
  --expected-duration 25m --timeout 7200 --detach -- bash -lc '
set -euo pipefail
export GIT_CONFIG_COUNT=4
export GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1="$PWD/third_party/sglang"
export GIT_CONFIG_KEY_2=safe.directory GIT_CONFIG_VALUE_2="$PWD/third_party/flashmla"
export GIT_CONFIG_KEY_3=safe.directory GIT_CONFIG_VALUE_3="$PWD/third_party/flashmla/csrc/cutlass"
cleanup() { bash scripts/local/serve.sh "$MODE" stop; }
trap cleanup EXIT
bash scripts/local/serve.sh "$MODE"
TAG="${MODE}-unprofiled-c${CONCURRENCY}"
bash scripts/local/bench-lswb.sh "$TAG" "$PORT" "$CONCURRENCY" 1200
RUN=$(find "results/lswb-replay/$TAG" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
cp "logs/serve_${MODE}.log" "$RUN/server.log"
'

gpuq status
gpuq jobs --all --user winstonxcai
gpuq logs JOB_ID
```

Finished jobs appear in `jobs --all`; allocations use `allocated_gpu_ids`.
If logs appear empty, inspect `stderr.log` in the job's reported `log_dir`.

## Inspect and push compact results

These commands are for the **GPU server (VDI)** checkout, which retains its
legacy `results/` folder and direct-staging workflow for now. They are not
instructions for the local curated checkout, where raw `results/` is ignored
and retained evidence belongs in `benchmarks/`. No server folder migration is required.

Set `RUN` to the exact completed run directory. The summary contains **mean**
TPOT in milliseconds and completion-token throughput in tokens/second.

```bash
RUN=results/lswb-replay/kernel-unprofiled-c24/REPLACE_WITH_TIMESTAMP
cat "$RUN/client/summary.json"
bash scripts/local/lswb-row.sh "$RUN"

# Save before restarting this mode; GPUQ and the profiling loop already do this.
cp logs/serve_kernel.log "$RUN/server.log"

git add -- "$RUN/client/summary.json" "$RUN/client/manifest.json" \
  "$RUN/client/done.json" "$RUN/client/REPORT.md" \
  "$RUN/client.log" "$RUN/cache-flush.log" "$RUN/server-info.json" \
  "$RUN/gpu-samples.csv" "$RUN/server.log"
# For a profiled run, also stage its capture-window record:
# git add -- "$RUN/nsys-windows.jsonl"
git diff --cached --stat
# Commit only these result paths; preserve unrelated staged work.
git commit --only -m "Record LSWB replay results" -- "$RUN"
git push origin remnant
```

Keep raw prompts and large profiles outside ordinary Git. Transfer traces
separately using the established artifact storage or Git LFS workflow.

## Per-layer Packed Triton survivor counts

Packed defaults to 256 survivors and 328 logical bytes per C4 row. Triton mode
also accepts 64, 128, 192, 320, 384 and 448 survivors; 512 is rejected. Counts
are selected once at startup, not changed between requests or cache blocks.

```bash
# Uniform 192 survivors: 192 FP8 values + 64 bitmap bytes + 8 scale bytes.
C4_KEEP_COUNT=192 bash scripts/local/serve.sh triton

# Defaults apply to all C4 layers; keys are absolute, zero-based model layer IDs.
C4_KEEP_COUNT=192 C4_LAYER_KEEP_COUNTS='{"2":128,"4":320}' \
  bash scripts/local/serve.sh triton
```

Direct server equivalents are `--dsv4-c4-cache-keep-count` and
`--dsv4-c4-cache-layer-keep-counts`. Overrides must identify C4 layers in the
loaded model; their order is independent of pipeline-stage/C4-pool indexing.
The launcher checks mounted-source support and resolves the profile before
stopping an existing server. Values are passed as separate quoted arguments.

Every count retains the largest raw magnitudes across all 512 coordinates.
It does not retain entire groups of 64. Bitmap/scales remain eight entries
per row, with separate physical planes and `count + 72` logical bytes.
The allocator sums the actual C4-layer sizes; smaller layers are not padded
back to 256 values. Native, SWA and C128 layouts remain unchanged.

Any effective non-256 count requires Triton storage **and** reconstruction,
with HiCache disabled. The optimized CUDA store/Direct reader and HiCache
retain their existing 256-only ABI. Learned scoring is not integrated here.

CPU references/configuration and allocation accounting have local coverage.
Non-256 Triton execution, graph replay and memory safety require model-free
H100/SM90 qualification before production use. No serving throughput or task
quality gain follows from allocation size alone.

After separate resource authorization, qualify the existing model-free workflow:

```bash
modal run scripts/modal/kernels.py::validate_kernels --run-id <unique-id> --sanitizer
```

This runs the extended existing tests and sanitizer selections, including mixed
layer graph replay and every supported store/reconstruction count. It is a
validation gate; the unchanged timing matrix remains a 256-survivor comparison.
