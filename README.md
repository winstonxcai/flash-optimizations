# Flash Optimizations

Experimental KV-cache optimizations for DeepSeek-V4-Flash serving with SGLang.
The main runnable workflow is **Mustafar**: TopMag50 pruning with a persistent
packed cache and optional fused CUDA reconstruction. The repository also contains
an experimental low-rank implementation, `xkv`, and experiment reports.

## Requirements

- For serving: a Linux NVIDIA GPU host with CUDA, Docker, and NVIDIA Container
  Toolkit. The benchmark defaults target **four H100 GPUs (TP4)**.
- Official `deepseek-ai/DeepSeek-V4-Flash-0731` weights. The Modal downloader pins
  revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
- Current Dockerfiles use **SGLang v0.5.18 / CUDA 13.0**. Mustafar patches match
  specific upstream source anchors; run the drift and verification checks before
  using a different source tree. Some older reports describe earlier versions.
- Standalone benchmarks require Bash, Python, `curl`, `jq`, and `setsid`.
  Kernel execution additionally needs PyTorch, Triton, and the SGLang attention
  dependencies supplied by the GPU image.

Run Python commands from the repository root, or put the root on `PYTHONPATH`.
There is no root-level Python package installer or dependency manifest.
macOS can be used to edit the code or launch Modal jobs; GPU serving runs on Linux.

```bash
git clone git@github.com:winstonxcai/flash-optimizations.git
cd flash-optimizations
# HTTPS alternative:
# git clone https://github.com/winstonxcai/flash-optimizations.git
```

## Quick start: benchmark on a Linux GPU host

The self-contained server image clones SGLang, builds the fused extension,
applies Mustafar, and verifies the patch. Model weights are mounted separately.

```bash
# Run from the repository root.
docker build -t flash-optimizations:0731 -f mustafar/docker/modal.Dockerfile .
mkdir -p results

docker run --rm -it --gpus all --shm-size 120g \
  -v /absolute/path/to/DeepSeek-V4-Flash-0731:/model:ro \
  -v "$PWD/results:/results" \
  -e MODEL_PATH=/model \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e RESULTS_DIR=/results \
  flash-optimizations:0731 bash
```

Inside the container:

```bash
# The Modal app installs these tools separately from the Dockerfile.
apt-get update && apt-get install -y curl jq util-linux coreutils
cd /opt/mustafar/flash-optimizations

# Arguments: mode, input tokens, output tokens, concurrency.
bash mustafar/scripts/local/bench-serving.sh native 32768 2048 8
bash mustafar/scripts/local/bench-serving.sh packed 32768 2048 8
# Optional fused reconstruction, using the extension built into this image:
bash mustafar/scripts/local/bench-serving.sh fused 32768 2048 8
```

Each command starts its own server, waits for health, runs one warm-up wave and
three measured waves, validates completion/output lengths, and stops its server
on exit. These are serving performance measurements; they do not evaluate
model-answer quality. Concurrency must be a positive integer at most 136 and must
fit the available cache capacity.

Each run produces a unique directory containing `server.log`, `warmup.log`,
`warmup.jsonl`, `measured.log`, and `measured.jsonl`. The bind mount above preserves
these under the host's `results/`. Read throughput, TTFT, and TPOT from the
official SGLang benchmark output. Without `RESULTS_DIR`, standalone results go to
`mustafar/logs/bench-serving/`. Graph replay and simultaneous residency require
separate log inspection.

Useful overrides include `PORT` (standalone default `30211`), `TP` (4),
`MEM_FRAC` (0.88), `CTX_LEN` (1048576), `MAX_RUN` (256), `CHUNK` (8192),
`SEED` (42), `PYTHON`, and `DECODE_CFG`. When using a prepared environment outside
Docker, set `SGLANG_ROOT` to the patched SGLang checkout root and `MODEL_PATH`
to the weights directory before running the same script.

## Modes and configuration

| Mode | Persistent cache | Reconstruction |
| --- | --- | --- |
| `native` | Original 584-byte row, no TopMag pruning | Original SGLang path |
| `packed` | 328-byte row retaining 256 of 512 coordinates | Triton path into the existing attention workspace |
| `fused` | Same 328-byte packed row | CUDA adapter into native FlashMLA rows |

The row-size reduction applies to the compressed cache component, not all GPU
memory. TopMag pruning is lossy. Quality and capacity results depend on the model,
workload, and serving configuration; see [Mustafar report](mustafar/report.md).

The benchmark script sets these switches for each mode automatically. For a
custom server, set them **before allocating caches or capturing CUDA graphs**:

| Variable | Meaning |
| --- | --- |
| `SGLANG_OPT_TOPMAG=1` | Enable magnitude pruning |
| `KEEP=0.5` | Retain exactly half the coordinates; required for packed storage |
| `SGLANG_OPT_TOPMAG_PACKED=1` | Enable persistent packed storage |
| `SGLANG_OPT_TOPMAG_FUSED=1` | Select CUDA reconstruction; requires packed storage and the extension |

Fused mode fails if the extension is unavailable. To build it manually in a
compatible CUDA/PyTorch environment, run from the repository root:

```bash
MAX_JOBS=4 python3 mustafar/cuda/setup.py build_ext --inplace
python3 -m mustafar.tests.fused_cpu
```

See the [packed ABI and CUDA notes](mustafar/cuda/README.md).

## Patch an existing SGLang checkout

Use a dedicated v0.5.18 checkout for the patched server. `SG_LOWRANK_SRC` points
to its **python directory**, whereas the benchmark's `SGLANG_ROOT` points to the
checkout root.

```bash
export SG_LOWRANK_SRC=/absolute/path/to/sglang-lowrank/python
python3 -m mustafar drift       # Read-only source-anchor compatibility report
python3 -m mustafar patch
python3 -m mustafar verify

# Restore originals when finished:
python3 -m mustafar unpatch
```

Patching validates targets before writing and keeps `.mustafar.orig` backups.
Verification compares the full expected patch with those originals. Unpatching
requires verified backups; unexpected edits or incompatible older patches need
explicit reconciliation.

## Persistent local serving and evaluations

The local drivers manage separate pristine and patched SGLang trees in a Docker
container. First adapt the **MACHINE CONFIG** block in
[`mustafar/scripts/local/env.sh`](mustafar/scripts/local/env.sh), especially
`HOST_REPO`, `REPO_CT`, `MODEL_CT`, and any external dataset/client paths. Some
paths are fixed assignments, so environment overrides alone are insufficient.
The container helper mounts the host filesystem at `/mnt/host_root` and `/home`
at `/home`; review these mounts when adapting it to your node.

```bash
bash mustafar/scripts/local/container.sh up
bash mustafar/scripts/local/container.sh patch
bash mustafar/scripts/local/serve.sh packed

curl -fsS http://127.0.0.1:30212/health

# Stop the persistent server before launching a benchmark on the same port.
bash mustafar/scripts/local/serve.sh packed stop

# Compare native and packed at a shared concurrency, or their own ceilings.
bash mustafar/scripts/local/bench-serving.sh fair 32768
bash mustafar/scripts/local/bench-serving.sh max 65536
```

Local serving defaults to GPUs `0,1,2,3` and port `30212`. Logs go to
`mustafar/logs/serve_<mode>.log`; dual-leg results go under
`mustafar/results/serving/`. LongBench v2, LongSWE-Bench, Sangfor, and SWE-bench
drivers need datasets, harnesses, or remote configuration not bundled here.
See the [local driver guide](mustafar/scripts/local/README.md) for those workflows.

## Modal

Install the Modal client in your client environment and authenticate your chosen
profile. Run these commands from the repository root. GPU functions allocate
Modal resources; serving uses four H100s.

```bash
python3 -m pip install modal
export MODAL_PROFILE=your-profile

# Download the pinned weights to the persistent model volume.
modal run mustafar/scripts/modal/app.py::download_model

# Validate kernels before serving benchmarks.
modal run mustafar/scripts/modal/app.py::validate_packed
modal run mustafar/scripts/modal/app.py::validate_fused
modal run mustafar/scripts/modal/app.py::bench_kernels --suite fused

modal run --detach mustafar/scripts/modal/app.py::bench_serving \
  --mode packed --input-tokens 32768 --output-tokens 2048 \
  --concurrency 8 --timeout-minutes 60
```

Run separate calls with `--mode native` or `--mode fused` for comparisons.
Weights live in `deepseek-v4-flash-0731`; run artifacts live under `/results`
in the `mustafar-stage2a-results` volume and are committed even on failure.
See [Modal benchmark details](mustafar/scripts/modal/readme.md); its historical
sections preserve older commands that are not current entry points.

## Checks

With PyTorch installed, the numerical references can run on CPU:

```bash
python3 -m mustafar selftest
python3 -m mustafar packed_selftest
```

In a development environment with PyTorch, Modal, and Ruff installed:

```bash
python3 -m unittest discover -s mustafar/tests -t .
ruff check mustafar
ruff format --check mustafar
```

GPU-dependent checks skip when their required hardware or dependencies are
absent. Set `MUSTAFAR_TEST_FETCH_SGLANG=1` to opt into tests fetching upstream
SGLang sources. For explicit GPU validation in the prepared environment:

```bash
python3 -m mustafar.tests.gpu_packed
python3 -m mustafar.tests.gpu_fused
```

## Repository map

- [`mustafar/`](mustafar/): packed storage, pruning references, source patches,
  Triton/CUDA kernels, validation, and serving drivers.
- [`xkv/`](xkv/): experimental rank-192 low-rank KV store. It uses separate
  patch hooks and per-layer basis inputs (`A_<layer:03d>.pt`) not bundled here;
  consult the [low-rank experiment writeup](writeup/xkv-lowrank-concurrency.md)
  before using it. The Mustafar quick start does not enable XKV.
- [`writeup/`](writeup/): experiment methodology, results, and limitations.
- [`log/`](log/): chronological research notes.

## Troubleshooting

- **Patch/drift failure:** check `SG_LOWRANK_SRC` and the SGLang revision. Do not
  apply the current patch blindly to a different version or an edited tree.
- **Server fails to start:** inspect the run's `server.log`; check model mounts,
  GPU visibility, and available memory. Reduce concurrency for cache exhaustion.
- **Port in use:** choose a free `PORT` or stop the server using that port.
- **Missing tools or CUDA extension:** use the prepared GPU environment and the
  tool installation/build commands above.
- **Local drivers use another machine's paths:** update `env.sh`, including
  container-visible paths and external evaluation settings.
