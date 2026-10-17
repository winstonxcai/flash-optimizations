<!-- Author: Winston Cai. -->

# Local serving and replay commands

Run from the repository root on the GPU node. Modes: `native` = Native;
`triton` = Packed with Triton packing/reconstruction; `kernel` = Packed with
optimized kernels. All three use the same Docker image.

## Isolated latent-KV capture

Capture uses a **prepared capture-only image and separate container**, never the
running production container. Image setup is a prerequisite, not part of capture.
The shell entry point owns all Docker operations: local image/label checks,
mounts, GPU selection/availability, container launch, timeouts and exact-container
cleanup. It does not source production environment defaults. It requires Docker,
`nvidia-smi` and GNU `timeout`. These examples are commands for an explicitly
allocated GPU-server session, not automatic launch steps:

```bash
CAPTURE_GPUS=0,1,2,3 bash scripts/local/capture-latent.sh run \
  remnant-capture:local /data/models/DeepSeek-V4-Flash-0731 \
  /data/capture/scratch /data/capture/published 3600 30312
```

Provision `capture-config.json` and `replay.jsonl` in scratch beforehand. Schema 2,
an explicit `audit_shared` policy, and concurrency two are required; shared-only
capture also requires the existing matching `shared-audit-proof.json`. Scratch
and publication destinations must be distinct and writable. Scratch must retain
50 GiB free; existing client staging/publication limits and resume checks remain.
Do not use a read-only mount such as the server's `/mnt/public_data` as output.

The prepared image must already contain the capture observer, runtime hooks,
calibration client and compiled extensions. The shell verifies these components
once in a GPU-free preflight container, then calls Python's data preflight against
the actual mounts. Missing prerequisites fail;
capture never builds, pulls, patches or updates an image. Container launches use
`--pull never`; no kernel build or production image/configuration change occurs.
Capture stays Native, TP4/concurrency two, seed 42, with prefix reuse, overlap
scheduling and CUDA graphs disabled through the existing capture-only settings.

`3600` is the **total container budget**, including model startup, readiness,
replay and writer drain—not an additional hour after loading. The client gets
only the remaining time. Timeout/interruption stops the owned server process
group and exact capture container, with bounded cleanup; it never stops another
server. Logs and completed publication/progress artifacts remain available after
failure. Selected GPUs must already be allocated to you and unoccupied; the
launcher does not acquire GPUQ resources or terminate competing processes.

`bash scripts/local/capture-latent.sh --help` is safe locally and launches
nothing. `training/capture.py` runs inside the prepared environment and has no
Docker/image/GPU-selection logic. Its `preflight` action validates configuration,
audit proof, writable storage, free space and the server port. Its `run` action
rechecks those data conditions, starts Native and the existing calibration client,
and cleans up only its server process group. The shell supplies an absolute
deadline; Python converts it once to monotonic time for startup and replay.
The old `build` action is removed; provision images through your separate
image-setup workflow.

## Update the image

Pull outside `foo.txt` and outside GPUQ jobs:

```bash
git pull --ff-only origin remnant
git submodule update --init --recursive
bash scripts/local/container.sh rebuild
```

Rebuild after Dockerfile or fork source changes. The tag stays
`remnant:v0.5.18`; successful rebuilds replace the `remnant` container and stop
its servers. Host script changes need only a pull. To start an existing image:

```bash
bash scripts/local/container.sh up
```

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
export REPLAY_DIR=/home/jovyan/wenyuhong/benchmarks
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

## LSWB replay without profiling

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

```bash
cd /home/jovyan/winstonxcai/flash-optimizations
mkdir -p logs
gpuq run --user winstonxcai --project flash-optimizations \
  --gpus 4 --gpu-ids 4,5,6,7 --cwd "$PWD" \
  --env GPUS=4,5,6,7 --env PORT=30213 --env MASTER_PORT=29639 \
  --env SERVER_SEED=20260919 \
  --env REPLAY_DIR=/home/jovyan/wenyuhong/benchmarks \
  --output "$PWD/logs/ttft-diagnostic-$(date +%Y%m%d_%H%M%S)" \
  --workload-key remnant-ttft-c24 --expected-duration 50m \
  --timeout 5400 --detach -- bash -lc '
set -euo pipefail
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
  --env REPLAY_DIR=/home/jovyan/wenyuhong/benchmarks \
  --output "$PWD/logs/nsys-node-c24-$(date +%Y%m%d_%H%M%S)" \
  --workload-key remnant-node-c24 \
  --expected-duration 45m --timeout 5400 --detach -- bash -lc '
set -euo pipefail
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1="$PWD/third_party/sglang"
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
is a directory. Git trust is scoped to the job's two repositories.

```bash
mkdir -p logs
gpuq run --user winstonxcai --project flash-optimizations \
  --gpus 4 --gpu-ids 4,5,6,7 --cwd "$PWD" \
  --env GPUS=4,5,6,7 --env PORT=30213 --env MASTER_PORT=29639 \
  --env REPLAY_DIR=/home/jovyan/wenyuhong/benchmarks \
  --env MODE=native --env CONCURRENCY=24 \
  --output "$PWD/logs/native-unprofiled-c24" \
  --workload-key remnant-native-unprofiled-c24 \
  --expected-duration 25m --timeout 7200 --detach -- bash -lc '
set -euo pipefail
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PWD"
export GIT_CONFIG_KEY_1=safe.directory GIT_CONFIG_VALUE_1="$PWD/third_party/sglang"
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
