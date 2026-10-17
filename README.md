<!-- Author: Winston Cai. -->

# Remnant

Training-free sparsification and packed storage of compressed attention states
for **DeepSeek-V4-Flash-0731**. Packed retains 256 of 512 coordinates:
**584 → 328 bytes per C4 record (43.8% smaller)**. Capacity savings are established;
throughput benefits depend on the workload. Current target: H100/SM90, TP4 serving.

## Understand the modes

Native is the default. The two Packed paths use the same 328-byte format.

| Mode | Storage | Decode |
|---|---|---|
| `native` | Native | Stock FlashMLA |
| `triton` | Packed | Reconstruct Native records, then stock FlashMLA |
| `kernel` | Packed | Direct Packed attention |

## Read first

1. [Project report](docs/report.md): motivation, measured gains and limitations.
2. [Fork change map](third_party/changelog.md): where Remnant modifies SGLang and FlashMLA.
3. Follow your research track:
   - **Kernel optimization:** [E010 decisions](experiments/E010/record.md).
   - **Attention-aware latent KV compression:** [method and findings](docs/attention-aware-latent-kv-compression.md) and the [maintained training pipeline](training/README.md). Historical study protocols remain under the ignored `experiments/attention-pruning/` workspace.

Use the maintained runners below; historical experiment commands may be archived.

## Run your first check

From the repository root, initialize the forks and check the launchers locally.
No model weights, GPU or billable Modal run is needed.

```bash
git submodule update --init --recursive
python -m pip install modal
python -m unittest scripts.modal.tests
```

## Run on GPUs

### Local: four H100s

Requires Docker with NVIDIA GPU support and a local 0731 checkpoint. On shared
nodes, reserve GPUs through GPUQ first; select unused GPU IDs and ports.

```bash
export MODEL_PATH=/absolute/path/to/DeepSeek-V4-Flash-0731
export GPUS=0,1,2,3 PORT=30212 MASTER_PORT=29638
bash scripts/local/container.sh up      # builds the image if missing
bash scripts/local/serve.sh native
curl -fsS "http://127.0.0.1:$PORT/health"
bash scripts/local/serve.sh native stop  # when finished
```

Replace `native` with `kernel` or `triton` in both start/stop commands.
See [local workflows](scripts/local/README.md) for serving benchmarks and replay.

### Modal

Check the selected account and budget first: CPU builds and GPU runs are billable.
This model-free three-mode benchmark builds/preflights on CPU, then uses one H100:

```bash
modal setup
modal run scripts/modal/kernels.py::benchmark_kernels --run-id "onboarding-$(date +%Y%m%d-%H%M%S)"
```

See [kernels.py](scripts/modal/kernels.py) for validation, profiling and artifact downloads.
Full-model serving uses four H100s and requires the checkpoint cached on that account:

```bash
modal run scripts/modal/serve.py::inspect_model_cache
modal run scripts/modal/serve.py::benchmark_serving --mode native
```

If missing, download the checkpoint using [serve.py](scripts/modal/serve.py) first.
Use `--mode kernel` or `--mode triton` for Packed.

## Make your first change

Read [AGENTS.md](AGENTS.md) before editing. Work in the owning repository:

- `third_party/sglang/`: runtime, packing, allocation and dispatch.
- `third_party/flashmla/`: Direct Packed attention kernels.
- Parent repository: scripts, experiment records and reports.

After fork source or Dockerfile changes, run `bash scripts/local/container.sh rebuild`;
this recreates the container and stops its servers. Host script changes need no rebuild.
Run focused correctness tests before timing; CPU launcher checks do not validate CUDA.
Commit/push changed forks first, then commit/push the parent submodule pins.

## Find results

[experiments/](experiments/INDEX.md) holds plans and decisions;
[benchmarks/](benchmarks/README.md) holds curated evidence. Raw outputs stay in
Git-ignored `results/` or `raw/`; [transfer/](transfer/README.md) handles selected VDI exports.
