# mustafar driver scripts (local)

Small, reusable entry points for serving DeepSeek-V4-Flash-0731 on this GPU node
and running the mustafar (Stage-1 packed 328-B C4) evals. Everything runs on the
local H100 box inside the `ruler-eval` SGLang container; the only machine-specific
knowledge lives in one file, [`env.sh`](env.sh).

## Layout

- `env.sh` — shared config + tiny helpers. **Porting to a new node = edit the
  MACHINE CONFIG block at the top** (container name, repo paths, model path,
  remote eval box, default GPUs/port). `serve.sh` is unchanged.
- `serve.sh <native|packed> [stop]` — boot the server (native untouched / packed
  328-B), wait `/health`, print pool + packed guard. Leave running or `stop` it.
- `bench-serving.sh <fair|max> <ctx> [C_fair]` — dual-leg serving comparison:
  `fair` measures Native vs Packed at the same concurrency (Native's allocator
  ceiling unless `C_fair` given); `max` measures each leg at its **own**
  allocator ceiling. Each leg boots its own report-config server.
- `bench-serving.sh <native|packed|fused> <in> <out> <concurrency>` — standalone
  single-config measurement, self-boots one report-config server with no
  container — the Modal / `tests/test_bench_serving.py` entrypoint.
- `bench-lswb.sh <tag> [port] [C] [dur]` — LongSWE-Bench replay client against a
  running server (prefix-reuse workload).
- `eval-lb2.sh <tag> [port] [out.json]` — LongBench v2 **full** eval against a
  running server (473 feasible of 503; 30 samples exceed the 1M ctx cap).
- `agentic-eval.sh <sangfor|swe> <instance-list> [run-id]` — agentic eval against
  a running server (list of task ids, one per line) run on the remote YJYBench
  box: `sangfor` = Sangfor-Bench, `swe` = SWE-bench_Verified.
- `lb2_serve_eval.py` — the LongBench v2 HTTP client (threaded, resumable).
- `config/` — eval inputs and env config: `sangfor-bench-hard50.txt` and
  `swe_instances_50_sweb_verified_mini.txt` (tracked instance lists — Sangfor &
  SWE-bench_Verified), plus `config_deepswe.json` and `config_swe_sangfor.json`
  (env/auth configs — carry the live token, git-ignored).

## Usage pattern

```sh
# 1) boot a server (native or packed), leave it running
./serve.sh packed
./serve.sh packed stop          # later

# 2) attach any eval / bench to the running server
./eval-lb2.sh packed-0731                # LongBench v2 full
./bench-lswb.sh packed                   # LSWB replay c15 @ 1200s
./agentic-eval.sh sangfor config/sangfor-bench-hard50.txt  # Sangfor on the hard-50 set
./agentic-eval.sh swe    config/swe_instances_50_sweb_verified_mini.txt  # SWE-bench_Verified on 50 instances

# serving capacity measurements boot their own legs per point (extended decode
# graphs, warm-up + 3 measured waves, official sglang.bench_serving)
./bench-serving.sh fair 32768
./bench-serving.sh max  65536

# standalone single-config, no container (Modal/tests interface; MODEL_PATH set)
MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731 ./bench-serving.sh packed 32768 2048 8
```

Evals that attach assume the server on `$PORT` (default 30212). `sangfor`/`swe`
clients run on the remote YJYBench box and reach this server through the
`docker_env_config` base URL (see `env.sh`: `EVAL_*`, `BASE_URL` override).
Eval results land under `mustafar/results/` (local) or the eval box's
`results/<run-id>/` (agentic); server logs under `mustafar/logs/serve_<mode>.log`.

## Notes

- Servers run on `$GPUS` (default 4,5,6,7) inside the `ruler-eval` container
  (`docker exec`), fp4-native MoE runner, mem-frac 0.88, 1M ctx, fp8 KV, DeepSeek
  reasoning/tool parsers.
- Decode CUDA-graph config default = small (agentic-eval concurrency). The
  serving drivers override with extended graphs so decode stays on-graph up to
  the packed allocator ceiling.
