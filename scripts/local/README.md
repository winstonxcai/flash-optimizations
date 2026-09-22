# Production-fork driver scripts (local)

These scripts run the same reviewed SGLang fork used by the Modal production
image. The local path uses Docker for isolation; the image clones the fork at
`/sgl-workspace/sglang-remnant`. Native and packed are cache-format variants
of that one runtime. The old repository-mounted `third_party/sglang` tree is
not used for serving.

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
- `serve.sh <native|packed> [stop]` — boot or stop a server using the same
  production fork. Native uses the default layout; packed passes
  `--dsv4-c4-cache-format remnant`.
- `bench-serving.sh` — compatibility dispatcher preserving both public command
  forms below.
- `bench-serving-capacity.sh <fair|max> <ctx> [C_fair]` — local Docker dual-leg
  comparison. `fair` measures Native vs Packed at the same concurrency; `max`
  measures each layout at its own allocator ceiling.
- `bench-serving-image.sh <native|packed> <in> <out> <concurrency>` — standalone
  measurement used by Modal. It self-boots a server directly in the production
  image and is separate from the local Docker capacity driver.
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
- `config/` — eval inputs and env config: `sangfor-bench-hard50.txt` and
  `swe_instances_50_sweb_verified_mini.txt` (tracked instance lists — Sangfor &
  SWE-bench_Verified), plus `config_deepswe.json` and `config_swe_sangfor.json`
  (env/auth configs — carry the live token, git-ignored).

## Usage pattern

```sh
# 0) first time (or after a container recreate): build and verify the image
./container.sh up
# use this after changing Dockerfile or the production fork revision
./container.sh rebuild

# 1) boot a server (native or packed), leave it running
./serve.sh packed
./serve.sh packed stop          # later

# 2) attach any eval / bench to the running server
./bench-lb2.sh packed-0731               # official LongBench v2
./bench-lswb.sh packed                   # LSWB replay c15 @ 1200s
./bench-agentic.sh sangfor config/sangfor-bench-hard50.txt  # Sangfor hard-50
./bench-agentic.sh swe    config/swe_instances_50_sweb_verified_mini.txt  # SWE-bench 50

# serving capacity measurements boot their own legs per point (extended decode
# graphs, warm-up + 3 measured waves, official sglang.bench_serving)
./bench-serving.sh fair 32768
./bench-serving.sh max  65536

# standalone single-config (Modal-style image environment)
MODEL_PATH=/models/DeepSeek-V4-Flash-0731 ./bench-serving.sh packed 32768 2048 8
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
- Decode CUDA-graph config default = small (agentic benchmark concurrency). The
  serving drivers and C>15 legs override with the extended config so decode
  stays on-graph up to the packed allocator ceiling.
