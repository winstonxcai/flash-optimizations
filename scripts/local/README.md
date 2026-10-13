<!-- Author: Winston Cai. -->

# Local serving and replay commands

Run from the repository root on the GPU node. Modes: `native` = Native;
`triton` = Packed with Triton packing/reconstruction; `kernel` = Packed with
optimized kernels. All three use the same Docker image.

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

## LSWB replay without profiling

Against the running server, run for 1,200 seconds at concurrency 24 or 28:

```bash
bash scripts/local/bench-lswb.sh "${MODE}-unprofiled-c24" "$PORT" 24 1200
bash scripts/local/bench-lswb.sh "${MODE}-unprofiled-c28" "$PORT" 28 1200
```

Each replay checks decode graph coverage, waits for idle and flushes the prefix
cache. Results go to `results/lswb-replay/<tag>/<timestamp>/`. Replay completion
leaves the server running.

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
