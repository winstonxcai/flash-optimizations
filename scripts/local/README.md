<!-- Author: Winston Cai. -->

# Production-fork driver scripts (local)

These scripts run the same reviewed SGLang fork used by the Modal production
image. The local path uses Docker for isolation; the image installs the fork at
`/sgl-workspace/sglang-remnant`. The image copies the checked-out forks in
`third_party/`; rebuild it after changing their source. Builds use host networking
so they also work on GPU nodes without Docker's `docker0` bridge. A rebuild keeps
the existing container until the image builds successfully, then replaces it.

| Serving mode | Cache format | Implementation |
|---|---|---|
| `native` (default server format) | Native, stock 584-byte C4 | Stock producer and FlashMLA |
| `triton` | Packed, 328-byte C4 | Torch selection, Triton packing and Native-page reconstruction, stock FlashMLA |
| `kernel` | Packed, 328-byte C4 | Current fused producer and optimized decode dispatch (Direct on SM90) |

`packed` remains a compatibility alias for `kernel`. Direct SGLang invocation
selects Packed with `--dsv4-c4-cache-format remnant` and its implementation with
`--dsv4-c4-cache-implementation triton|kernel` (default `kernel`). The
implementation flag has no effect on Native. Both Packed implementations retain
256 coordinates and share the same allocation and HiCache layout.

## Layout

- `container.sh [prep|recreate|up|rebuild]` — build/start the reproducible image and
  verify that its production fork supports the cache-format flag.
- `env.sh` — shared configuration and helpers. The repository root is derived
  from the script location; override `HOST_REPO`, `MODEL_PATH`, `GPUS`,
  `PORT`, or `CONTAINER` when moving to another node.
- `eval-env.sh` — optional replay/remote-evaluation settings. It has no
  embedded host, password, or credential defaults; evaluation scripts require
  the relevant variables explicitly.
- `run-server.sh` — shared foreground launcher used by both local-container
  and Modal serving. It is normally called by the other scripts.
- `serve.sh <native|triton|kernel> [boot|profile|stop]` — boot or stop a server using the same
  production fork and explicitly select its cache format and implementation.
- `bench-serving.sh` — compatibility dispatcher preserving both public command
  forms below.
- `bench-serving-capacity.sh <fair|max> <ctx> [C_fair]` — local Docker dual-leg
  comparison. `fair` measures Native vs Packed at the same concurrency; `max`
  measures each layout at its own allocator ceiling.
- `bench-serving-image.sh <native|triton|kernel> <in> <out> <concurrency>` — standalone
  measurement used by Modal. It self-boots a server directly in the production
  image and is separate from the local Docker capacity driver. Each run's
  `config.json` records the resolved mode, cache format and implementation.
- `bench-lswb.sh <tag> [port] [C] [dur]` — LongSWE-Bench replay client against a
  running server (prefix-reuse workload).
- `lswb-row.sh <run_dir|tag>` — print the 7 recorded SLO-run fields from a
  finished `bench-lswb.sh` run's `summary.json`.
- `bench-lb2.sh <tag> [port] [out_dir]` — official lm-eval LongBench v2
  multiple-choice benchmark against a running server via `/v1/completions`.
- `bench-agentic.sh <sangfor|swe> <instance-list> [run-id]` — agentic benchmark against
  a running server (list of task ids, one per line) run on the remote YJYBench
  box: `sangfor` = Sangfor-Bench, `swe` = SWE-bench_Verified.
- Model-free FlashMLA validity and timing live in the pinned fork under
  `third_party/flashmla/tests/` and `third_party/flashmla/benchmark/`.
- `bench-lb2.sh` requires the official `lm-eval[longbench]` CLI in the host
  environment and stores its results, logged samples, and SQLite request cache
  under the selected output directory.
- `config/` — decode-graph settings and tracked Sangfor/SWE-bench instance lists.
  Remote evaluation credentials/config are supplied through `EVAL_CFG` and are
  intentionally kept outside this directory and out of the repository.

## Usage pattern

```sh
# 0) first time (or after a container recreate): build and verify the image
./container.sh up
# use this after changing Dockerfile or the production fork revision
git submodule update --init --recursive
./container.sh rebuild

# 1) boot a server (native, triton or kernel), leave it running
./serve.sh triton
./serve.sh triton stop          # later

# 2) attach any eval / bench to the running server
./bench-lb2.sh triton-0731               # official LongBench v2
./bench-lswb.sh triton                   # LSWB replay c15 @ 1200s
./bench-agentic.sh sangfor config/sangfor-bench-hard50.txt  # Sangfor hard-50
./bench-agentic.sh swe    config/swe_instances_50_sweb_verified_mini.txt  # SWE-bench 50

# serving capacity measurements boot their own legs per point (extended decode
# graphs, warm-up + 3 measured waves, official sglang.bench_serving)
./bench-serving.sh fair 32768
./bench-serving.sh max  65536

# standalone single-config (Modal-style image environment)
MODEL_PATH=/models/DeepSeek-V4-Flash-0731 ./bench-serving.sh kernel 32768 2048 8
```

Benchmarks that attach assume the server on `$PORT` (from `env.sh`). `sangfor`/`swe`
clients run on the remote YJYBench box and require `EVAL_SSH`, `EVAL_SCP`,
`EVAL_YJY`, `EVAL_VENV`, and `EVAL_CFG` to be exported before launch. A
`BASE_URL` override may be supplied for a per-run config copy.

LongSWE-Bench requires `REPLAY_DIR` and its derived client paths in
`eval-env.sh`; no cluster-specific replay path is assumed.
Eval results land under `results/` and server logs under
`logs/serve_<mode>.log`.

## Notes

- Servers run on `$GPUS` (default 0,1,2,3) inside the `remnant` container,
  using the Dockerfile-installed fork, fp4-native MoE runner, mem-frac 0.88,
  1M ctx, fp8 KV, and DeepSeek reasoning/tool parsers.
- `serve.sh` and serving capacity drivers default to the extended decode graph
  config, covering c24/c28 and batches through 136. Set `DECODE_CFG` explicitly
  to use a smaller config for agentic workloads. `bench-lswb.sh` saves the
  resolved `/server_info` configuration and refuses to start if full decode
  graphs do not cover the requested concurrency. The shared `env.sh` default
  remains small for other callers.

After updating the worker cache-mode fix, rebuild with `container.sh rebuild`:
the installed runtime is copied into the image. Verify all three modes without
loading model weights or allocating KV pools:

```sh
docker exec remnant python3 -m unittest discover \
  -s /sgl-workspace/sglang-remnant/test/registered/unit \
  -p test_dsv4_c4_cache_format.py -v
```

The spawn tests pass resolved `ServerArgs` through multiprocessing serialization
and use the same publication hook as `ModelRunner`, before KV pool sizing.
On the next server startup, each scheduler logs
`C4 worker cache: pid=... format=... implementation=...`. Native should report
`format=native`; Packed should report `format=remnant` and the selected
`implementation=triton` or `implementation=kernel`. Both Packed modes should
also log `Remnant packed pool` with `logical_row_bytes=328`. Check these worker
and pool messages before running a replay; `/server_info` flags alone do not
prove the worker dispatch mode.

## Nsight Systems capture

Run `./serve.sh triton profile` to launch Packed Triton under Nsight Systems.
The container must already provide `nsys`; launcher edits need only a Git pull,
not an image rebuild. Reports go to a unique host directory under
`logs/nsys-triton-*/server*.nsys-rep` and survive container recreation.
CUDA/NVTX tracing waits for `cudaProfilerStart`, so model loading and CUDA-graph
creation are excluded. CPU sampling and context-switch tracing are disabled.
The launcher allows 40 minutes for cold-start kernel compilation and graph
creation before cleaning up a failed startup. After it reports healthy,
explicitly bracket the workload:

```sh
curl -fsS -X POST "http://127.0.0.1:30212/start_profile" \
  -H 'Content-Type: application/json' -d '{"activities":["CUDA_PROFILER"]}'
# Send the requests to measure here.
curl -fsS -X POST "http://127.0.0.1:30212/stop_profile"
```

Each start/stop pair captures a separate range. Nsight exports reports
asynchronously after each range, before server teardown. The repeat limit is
999 ranges per server launch.
Stopping capture leaves the server running and armed for the next range. Use the configured
`PORT` in the URLs if overridden. `./serve.sh triton stop` tears down the owned
server and profiler process group.

For timed benchmark captures, pass ordered start:end offsets as the fifth
argument to the existing replay script, for example:

```sh
./bench-lswb.sh triton-c24 30212 24 1200 300:360,900:960
```

The offsets start when the client launches. Capture API request and acknowledgement
times are saved in the run's
`nsys-windows.jsonl`. API calls allow up to 180 seconds for Nsight trace export;
export time does not shift the next scheduled window. Capture API failures or an
early client exit before all windows complete make the run fail. Launch the
server with `profile` first; use the same windows for Native comparisons.
