# Modal Stage-1 workflow

Run from the repository root. Modal builds `mustafar/Dockerfile` on CPU; the model stays in the persistent `deepseek-v4-flash-fp8` Volume and is not baked into the image.

```bash
modal profile activate <profile>

# One time, CPU only: download the pinned model revision.
modal run mustafar/scripts/modal/app.py::download_model

# One H100: compile and validate the Triton kernels.
modal run mustafar/scripts/modal/app.py::kernel_validation

# Four H100s: run packed then native TP4 on the same allocation.
# Detach so a local-client disconnect cannot cancel the paid job.
modal run --detach mustafar/scripts/modal/app.py::tp4_capacity_ceiling

# Four H100s: short graph-enabled decode A/B at 64k, concurrency 1 and 2.
modal run --detach mustafar/scripts/modal/app.py::tp4_graph_decode_ab
```

The TP4 function uses the official [`sglang.bench_serving`](https://lmsysorg.mintlify.app/docs/developer_guide/bench_serving) entry point at exact 32k, 64k, and 128k input lengths with 16 output tokens, CUDA graphs off, `mem-fraction-static=0.93`, and `max-running-requests=64`. Keep `--random-range-ratio 1.0`; `0` samples lengths from 1 to the target.

Results are committed after every point to `mustafar-stage1-results`:

```bash
modal volume get mustafar-stage1-results tp4-capacity-ceiling.json ./results/
modal volume get mustafar-stage1-results official-capacity ./results/
modal volume get mustafar-stage1-results capacity-topmag50_packed-server.log ./results/
modal volume get mustafar-stage1-results capacity-native-server.log ./results/

modal volume get mustafar-stage1-results tp4-graph-decode-ab.json ./results/
modal volume get mustafar-stage1-results official-graph-decode ./results/
modal volume get mustafar-stage1-results graph-decode-topmag50_packed-server.log ./results/
modal volume get mustafar-stage1-results graph-decode-native-server.log ./results/
```

Measured no-OOM admission ceilings on TP4 H100 SXM were:

| Mode | KV tokens | 32k | 64k | 128k |
|---|---:|---:|---:|---:|
| Packed TopMag50 | 402,688 | 12 | 6 | 3 |
| Native | 332,288 | 10 | 5 | 2 |

All six exact-length runs completed with return code 0. The packed pool provided `1.2119×` total token capacity; this is a capacity/no-OOM benchmark, not a quality evaluation.

The complete paired TP4 Modal function took `1,590.36 s` (`26m 30s`) on four
H100s, equivalent to approximately `1.77 H100-hours`. Server initialization
took `925.67 s` for packed and `197.11 s` for native. The six official
`bench_serving` subprocesses took `443.07 s` including client setup and result
handling; their measured request-workload durations totaled `231.36 s`.

## Graph-enabled decode smoke

The short TP4 run used exact 64k inputs, 128 output tokens, full decode CUDA
graphs for batch sizes 1 and 2, and the official `sglang.bench_serving` client.

| Mode | C1 median ITL | C2 median ITL | Server KV tokens |
|---|---:|---:|---:|
| Packed TopMag50 | 8.258 ms | 8.696 ms | 504,832 |
| Native | 8.079 ms | 8.596 ms | 416,768 |

Packed decode was `2.22%` slower at concurrency 1 and `1.17%` slower at
concurrency 2, while exposing `1.2113x` as many server KV tokens. All 12
requests completed with exactly 128 output tokens, empty error fields, and
graph replay confirmed in both server logs. The paired function took
`873.06 s` (`14m 33s`) on four H100s, or approximately `0.97 H100-hours`.
Packed and native server startup took `593.32 s` and `123.05 s`, respectively.
This is a graph/no-error performance smoke, not a quality evaluation.

## Stage-1 report serving benchmark (2048-token decode)

This is the reproducibility record for the fair-load and maximum-concurrency
tables in `mustafar/report.md`. Three parallel TP4 legs were run: untouched
V4-Flash, TopMag50 with the 584-byte native C4 layout, and Stage-1 TopMag50
with the 328-byte packed C4 layout. The runs used separate Modal profiles with
the model already present in each profile's persistent volume; `winstoncai233`
was not used.

### Setup

| Item | Configuration |
|---|---|
| Model | `sgl-project/DeepSeek-V4-Flash-FP8` |
| Model revision | `ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17` |
| Mustafar Stage-1 commit | `abd18764701eb4db9586a664a1b042c0acb89cd1` |
| SGLang revision | `f63458b5beaceabbd9d749b9fc956370e1b649e6` |
| Hardware | 4× NVIDIA H100 80 GB HBM3 |
| Parallelism | Tensor parallelism 4 (TP4) |
| Server memory fraction | `0.93` |
| Server context limit | 135,168 tokens |
| Maximum running requests | 64 |
| Chunked prefill | 4,096 tokens |
| Output length | Exactly 2048 tokens |
| Decode CUDA graphs | Full graphs for batch sizes 1–12 |
| Prefill CUDA graphs | Disabled |
| Sampling | Exact 32k, 64k, and 128k random inputs; range ratio 1.0 |
| Repetitions | One warm-up wave, followed by three measured waves |
| Validation | Exact output lengths, no errors, intended simultaneous residency, and decode graph replay required |

The completed legs took the following GPU wall times. H100-hours are wall
seconds × 4 GPUs ÷ 3,600 and include server startup and benchmark execution:

| Mode | Server ready (s) | GPU wall (s) | H100-hours | Modal profile |
|---|---:|---:|---:|---|
| Untouched V4-Flash | 907.64 | 1,566.62 | 1.74 | `winstoncai` |
| TopMag50 native | 893.83 | 1,586.77 | 1.76 | `caiw` |
| Stage-1 packed | 717.41 | 1,909.34 | 2.12 | `poohthewinniechurchill` |
| **Total** | — | **5,062.73** | **5.63** | three parallel jobs |

The three jobs ran in parallel for approximately 33 minutes of elapsed wall
time. All three used the preinstalled V4-Flash model, so no model download time
was included.

### Commands used

The paired benchmark was launched on Modal with:

```bash
modal run --detach mustafar/scripts/modal/app.py::tp4_stage1_report_tables --mode all

# Separate untouched V4-Flash behavior in the same image and configuration.
modal run --detach mustafar/scripts/modal/app.py::tp4_stage1_report_tables \
  --mode untouched

# The three production runs were launched concurrently with explicit profiles:
MODAL_PROFILE=winstoncai modal run --detach mustafar/scripts/modal/app.py::tp4_stage1_report_tables --mode untouched
MODAL_PROFILE=caiw modal run --detach mustafar/scripts/modal/app.py::tp4_stage1_report_tables --mode native
MODAL_PROFILE=poohthewinniechurchill modal run --detach mustafar/scripts/modal/app.py::tp4_stage1_report_tables --mode packed
```

This entrypoint came from the temporary report benchmark harness layered on
Stage-1 commit `abd1876`; it is not part of that runtime commit itself.

The untouched leg explicitly used:

```bash
SGLANG_OPT_TOPMAG=0 \
XKV_TOPMAG_KEEP=1.0 \
SGLANG_OPT_TOPMAG_PACKED_C4=0
```

For each mode, the server command was equivalent to the following.
`SGLANG_OPT_TOPMAG_PACKED_C4=0` selected the native C4 layout and `1` selected
Stage-1 packing.

```bash
SGLANG_OPT_TOPMAG=1 \
XKV_TOPMAG_KEEP=0.5 \
SGLANG_OPT_TOPMAG_PACKED_C4=<0-or-1> \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4-Flash-FP8 \
  --served-model-name deepseek-v4-flash \
  --tp 4 \
  --trust-remote-code \
  --mem-fraction-static 0.93 \
  --context-length 135168 \
  --max-running-requests 64 \
  --chunked-prefill-size 4096 \
  --fp8-gemm-backend triton \
  --host 0.0.0.0 \
  --port 30211 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":12,"bs":[1,2,3,4,5,6,7,8,9,10,11,12]},"prefill":{"backend":"disabled"}}' \
  --skip-server-warmup \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --watchdog-timeout 1800
```

Each warm-up and measured point used the official SGLang serving benchmark.
`CONTEXT`, `CONCURRENCY`, `NUM_PROMPTS`, and `SEED` were substituted for the
point being measured:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30211 \
  --model /models/DeepSeek-V4-Flash-FP8 \
  --tokenizer /models/DeepSeek-V4-Flash-FP8 \
  --dataset-name random \
  --random-input-len <CONTEXT> \
  --random-output-len 2048 \
  --random-range-ratio 1.0 \
  --num-prompts <NUM_PROMPTS> \
  --max-concurrency <CONCURRENCY> \
  --request-rate inf \
  --warmup-requests 0 \
  --flush-cache \
  --tokenize-prompt \
  --output-file <RESULT.jsonl> \
  --output-details \
  --seed <SEED>
```

Warm-up used `NUM_PROMPTS=CONCURRENCY`. Measurement used
`NUM_PROMPTS=3*CONCURRENCY`, meaning three measured waves at the configured
concurrency. At 2048 output tokens, the native maximums were C9/C4/C2 and the
packed maximums were C11/C5/C3 at 32k/64k/128k respectively.

| Context | Fair concurrency | 584-byte mode maximum | Packed maximum | Measured requests at each maximum |
|---:|---:|---:|---:|---:|
| 32k | 9 | 9 | 11 | Each 584-byte mode 27; packed 33 |
| 64k | 4 | 4 | 5 | Each 584-byte mode 12; packed 15 |
| 128k | 2 | 2 | 3 | Each 584-byte mode 6; packed 9 |

All measured requests completed with exactly 2048 output tokens, no request
errors, the intended concurrency simultaneously resident, and decode graph
replay confirmed in the server logs.

The untouched result and logs are stored as:

```bash
modal volume get mustafar-stage1-results stage1-report-untouched.json ./results/
modal volume get mustafar-stage1-results stage1-report-untouched-raw ./results/
modal volume get mustafar-stage1-results stage1-report-untouched-untouched.log ./results/
```

Final result files:

```bash
modal volume get mustafar-stage1-results stage1-report-native.json ./results/
modal volume get mustafar-stage1-results stage1-report-packed.json ./results/
```

All 192 measured requests completed with exactly 2048 output tokens, no request
errors, intended simultaneous residency, and decode graph replay confirmed in
the server logs. The raw JSONL remains under
`stage1-report-{untouched,native,packed}-raw`.
