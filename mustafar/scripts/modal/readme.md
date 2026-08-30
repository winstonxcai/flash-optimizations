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
