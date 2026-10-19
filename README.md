<!-- Author: Winston Cai. -->

# Remnant

Training-free sparsification and packed storage of compressed attention states
for **DeepSeek-V4-Flash-0731**, using **SGLang v0.5.18 on TP4 / 4× H100 80 GB**.
Default Packed retains 256 of 512 coordinates: **584 → 328 bytes per C4 record
(43.84% smaller)** and **1.2113× measured KV capacity**. C4 compresses four original
token positions into one record; record compression is not total GPU-memory savings.
Throughput benefits depend on workload, and task-quality equivalence is not established.

## Understand the modes

Native is the default. The two Packed paths use the same default 328-byte format.

| Mode | Storage | Decode |
|---|---|---|
| `native` | Native | Stock FlashMLA |
| `triton` | Packed | Reconstruct Native records, then stock FlashMLA |
| `kernel` | Packed | Direct Packed attention |

These are implementations, not different default pruning policies. Kernel remains
256-only. Triton also supports fixed per-layer retention of 64–448 coordinates in
steps of 64, without HiCache; these profiles need separate serving validation.

## Read first

1. [Project report](docs/report.md): motivation, measured gains and limitations.
2. [Fork guide](third_party/README.md): source ownership, integration map and validation limits; [changelog](third_party/CHANGELOG.md) for commit-based milestones.
3. Follow your research track:
   - **Kernel optimization:** [maintained microbenchmark protocol](third_party/sglang/benchmark/remnant/README.md) and [curated measurements](benchmarks/README.md).
   - **Attention-aware latent KV compression:** [method and findings](docs/attention-aware-latent-kv-compression.md) and [training/evaluation commands](training/README.md). Learned scoring is an offline research extension, not the serving default.

Use the maintained runners below; historical measurements retain their own source identities and limitations.

## Run your first check

Requires Python 3.11+ and Git LFS installed. From the repository root, initialize
the pinned forks and run the local CPU launcher checks. No weights, GPU or remote
image build is needed.

```bash
git lfs install --local
git submodule update --init --recursive --checkout
python3 scripts/sources.py .
python -m pip install modal
python -m unittest scripts.modal.tests
```

The checkout command is for a clean clone and may detach fork HEADs. Preserve any
existing fork work first. Preflights require clean forks at the exact parent pins,
including nested dependencies; they report mismatches rather than repairing them.

Training's local CPU checks may use the closest available CPU Torch version to
the shared image; see [training environments](training/README.md#environment).
GPU-server and Modal work uses the image's installed Torch.

## Run on GPUs

### Local: four H100s

Requires a Linux host with Docker/NVIDIA GPU support, Python, curl, GNU `timeout`
and a local 0731 checkpoint. On shared nodes, reserve four GPUs through GPUQ first;
select unused GPU IDs and ports. The container does not automatically start a server.

```bash
export MODEL_PATH=/absolute/path/to/DeepSeek-V4-Flash-0731
export GPUS=0,1,2,3 PORT=30212 MASTER_PORT=29638
export MODE=native                    # native | triton | kernel
bash scripts/local/container.sh up      # builds the image if missing
bash scripts/local/serve.sh "$MODE"
curl -fsS --max-time 10 "http://127.0.0.1:$PORT/health"
# Optional serving smoke: 32k input, 256 output, concurrency 2.
bash scripts/local/bench-serving.sh "${MODE}-smoke" "$PORT" 32768 256 2
bash scripts/local/serve.sh "$MODE" stop  # also stop after a failed benchmark
```

The benchmark client uses the running server; it never starts or stops one.
Random-text serving uses C warm-up requests and 3C measured requests, flushing the
prefix cache before each phase. C limits simultaneous requests, not total work;
3C requests are one run, not three independent runs. It may download ShareGPT
text; restricted-network hosts should prepare `BENCH_DATASET_PATH` first.
See [local workflows](scripts/local/README.md#random-text-serving-benchmark) for
dataset setup and [recorded-request workflows](scripts/local/README.md#lswb-replay-without-profiling)
for the distinction between request-concurrency benchmarks and fixed-active-user
LongSWE replay. GPUQ is only needed on shared GPU servers.

Latent-KV capture also uses this image, in an isolated short-lived container with
capture-specific Native settings—not a separate image or the live serving process.
See [capture setup](scripts/local/README.md#isolated-latent-kv-capture).

### Modal

Select the intended Modal profile and check its budget first: CPU builds and GPU runs are billable.
This model-free three-mode benchmark builds/preflights on CPU, then uses one H100:

```bash
modal setup
modal run scripts/modal/kernels.py::benchmark_kernels --run-id "onboarding-$(date +%Y%m%d-%H%M%S)"
```

See [kernels.py](scripts/modal/kernels.py) for `validate_kernels`, `profile_kernels`
and artifact downloads. These model-free checks do not establish serving TPOT.
Full-model serving uses four H100s and the account's pinned 0731 model volume:

```bash
modal run scripts/modal/serve.py::inspect_model_cache
# Only if the checkpoint is absent:
# modal run scripts/modal/serve.py::download_model
modal run scripts/modal/serve.py::benchmark_serving --mode native
```

[serve.py](scripts/modal/serve.py) owns server startup, cleanup and the same local
benchmark client. Defaults: 32k input, 2,048 output, C8; override with
`--input-tokens`, `--output-tokens`, `--concurrency`. Use `--mode kernel` or
`--mode triton` for Packed. The default 60-minute work limit is not an ETA.

## Make your first change

Read [AGENTS.md](AGENTS.md) before editing. Work in the owning repository:

- `third_party/sglang/`: runtime, packing, allocation and dispatch.
- `third_party/flashmla/`: Direct Packed attention kernels.
- Parent repository: scripts, experiment records and reports.

After fork source or Dockerfile changes, run `bash scripts/local/container.sh rebuild`;
this recreates the container and stops its servers. Host script changes need no rebuild.
Run focused correctness tests before timing; CPU launcher checks do not validate CUDA.
Commit/push changed forks first, then commit/push the parent submodule pins.

For learned pruning, use [training/README.md](training/README.md) for fitting,
six-method comparisons and retention sweeps. Cache/checkpoints are not in a clone;
obtain them from Winston. The cache belongs in ignored `data/training-cache-001/`.
Current results reuse ten development issues for evaluation and checkpoint
selection; they are not a fresh held-out test or a deployed learned policy.

## Find results

Start with the [benchmark index](benchmarks/README.md):

- [LongSWE C24, three runs per mode](benchmarks/lswb-replay/c24-native-packed256-n3-20261018/README.md): repeated Native/Triton/Kernel serving comparison.
- [128k input / 8k output](benchmarks/serving/random-128k-8k-c24-capacity-n1-20261018/README.md): matched C24 and allocator-ceiling loads; one run per cell.
- [Shared-prefix C16](benchmarks/serving/shared-prefix-c16-n1-20261019/README.md): G16/G32/G40 working sets; observed cache-retention gains in a single-run synthetic screen, not replicated production gains.
- [H100 decode/store microbench](benchmarks/microbench/decode-store-h100-three-mode-20261014/README.md): operation timing and Nsight Compute reports, not serving latency.
- [Prefill-read studies](benchmarks/microbench/prefill-read-20261018/README.md) and [fused-reader experiment](benchmarks/microbench/prefill-read-fused-rope-20261018/README.md): reconstruction cost and profiling; not end-to-end serving gains.

JSON is canonical; each collection records provenance and limits. Git LFS stores
large reports; download curated artifacts with `git lfs pull --include="benchmarks/**" --exclude=""`.
Use `transfer/` to move selected results from the GPU server (VDI) to local:
copy into it on the server, push/pull through GitHub (LFS for large files), then
curate locally into `benchmarks/`. Full raw outputs stay in ignored `results/` or `raw/`.
