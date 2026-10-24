<!-- Author: Winston Cai. -->

# Attention-pruning training

Fit layer-local sensitivity scorers from cached latent KV states; evaluate with
exact Packed attention replay. No model weights are loaded.

## Cache and checkpoints

The frozen cohort is **40 training / 10 development issues**, with four / ten
sampled decode positions per issue/layer, respectively. Development evaluation
uses the same **2,100 query-layer examples** across all **21 CSA layers (2,4,…,42)**
and **64 heads**. These are reused development data, not a fresh held-out test.

Data and checkpoints are outside Git. Ask Winston for the cache and selected
policy/checkpoints; a clone alone does not include the study data.
Use repo-root `data/training-cache-001/` for the cache (Git-ignored). Other
artifact paths below are relative to `~/.cache/remnant/attention-pruning/`:

| Artifact | Location / required contents |
|---|---|
| Cache | Repo-root `data/training-cache-001/`: `config.json`, `index.json`, `progress.json`, all 21 `layer-XX.json` receipts and `layer-XX.pt` payloads |
| Policy / checkpoints | The selected policy JSON and every checkpoint named in its `selected` entries; the best mix uses `continued-all-results-001/continued-all-001/` and `anchored-pilot-001/outputs/` |
| Historical comparisons | `evaluator-extension-NSETUJ/comparison.json` and `retention-sweep-9ULSmw/sweep.json`; existing measurements are not overwritten |

When relocating, update checkpoint paths in a **copy** of the policy; relative
paths resolve beside it. Layer IDs and complete, finite model parameters remain
required; old optional checkpoint fingerprints are ignored.

Early iterations use the selected current calibration/reference code, without
source-fingerprint or Torch-version gates. Cache payload checksums, tensor
geometry/finiteness, query identities and Native denominators are still checked.
New runs save settings and measurements, not source/checkpoint provenance.
Keep historical caches/results unchanged and compare all methods in the same new
run; version changes do not guarantee identical historical scores. Training stays
at **256 survivors**; only evaluation sweeps **64–448 in steps of 64**.

## Environment

- **GPU server or Modal:** use the shared image and its installed Torch. Do not
  replace Torch inside the image or create a separate training/capture image.
- **Local CPU tests:** use the closest available CPU Torch version to the image's
  version that supports your platform, with NumPy and Matplotlib. Exact version
  equality is not required; no Docker image is needed for these checks.

Compare methods within one environment. Runs record Torch/device/thread settings;
local CPU checks are not bit-exact reproductions of image-based runs or GPU
validation.

## Run in the shared image

Use the existing `remnant` container from the shared `remnant:v0.5.18` image;
no separate training image or host Python installation is needed. From the
repository root on the host, enter the mounted checkout:

```bash
export TRAIN_GPU=0  # Replace with an allocated, idle GPU visible in the container.
docker exec -it -e CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  --workdir "/mnt/host_root$PWD" "${CONTAINER:-remnant}" bash
```

Run the following **inside the container**, from the mounted repository root.
The cache and calibration paths use that checkout; replace the policy and output
examples with your own paths. The image needs Torch, NumPy and Matplotlib.
Host paths are available under
`/mnt/host_root`; output paths must be fresh, persistent and **outside the checkout**.

```bash
export PRUNING_CACHE="$PWD/data/training-cache-001"
export PRUNING_POLICY=/mnt/host_root/data/pruning/policy.json
export PRUNING_RUNNER="$PWD/third_party/sglang-v0.5.18/python/sglang/benchmark/remnant/calibrate.py"
export PRUNING_OUTPUT=/mnt/host_root/data/pruning/new-fit
export PRUNING_RESULTS=/mnt/host_root/data/pruning/new-evaluation

# Train; the generated policy retains untouched layers.
python -m training.train --cache "$PRUNING_CACHE" --policy "$PRUNING_POLICY" \
  --runner "$PRUNING_RUNNER" --output "$PRUNING_OUTPUT" \
  --recipe continuation --layers 2,4 --device cuda

# Diagnostic evaluation of the generated policy; standalone evaluation is CPU-only.
python -m training.evaluate --cache "$PRUNING_CACHE" \
  --policy "$PRUNING_OUTPUT/policy.json" --runner "$PRUNING_RUNNER" \
  --output "$PRUNING_RESULTS/diagnostic.json" --layers 2,4 --threads 4

# CPU integration checks.
python -m unittest training.tests.test_integration.TrainingIntegrationTests -v
```

Training uses the selected GPU, staging one cached batch at a time; no serving
process is started. Standalone evaluation and integration checks use CPU.
Continuation, relaxed and anchored recipes require rank-64 checkpoints; the
rank64 recipe expands rank-16 checkpoints. Use `--help` for options. Capture setup
and commands are in [the local launch guide](../scripts/local/README.md#isolated-latent-kv-capture).

Real Docker checks use the existing shell launcher and publication test described
in the local launch guide. Capture reuses the same shared serving image in its own
short-lived container; only process settings, mounts and outputs differ. It does
not change the running production container or the saved cache.

For local CPU checks, run directly from the repository root in the CPU environment:

```bash
python -m unittest training.tests.test_integration.TrainingIntegrationTests -v
```

Local evaluation uses the same commands below; set `PRUNING_CACHE` to
`$PWD/data/training-cache-001` and use local policy, runner and external output
paths rather than `/mnt/host_root` container paths.

## Compare masks and sweep retention

Use the **frozen** `PRUNING_POLICY`; neither command trains or selects checkpoints.
Both run on CPU with four threads, loading one layer at a time. Omit `--layers`
for all 21 layers; `--layers 2` with a separate output name is a partial preflight,
not a completed full-cohort table.

```bash
# Six rows: magnitude, structured and learned masks at 256 and 192 survivors.
python -m training.evaluate --compare-masks \
  --cache "$PRUNING_CACHE" --policy "$PRUNING_POLICY" --runner "$PRUNING_RUNNER" \
  --output "$PRUNING_RESULTS/comparison.json" --threads 4

# Three curves at each budget: 64, 128, 192, 256, 320, 384, 448.
# Include all 50 issues (40 training + 10 development).
python -m training.evaluate --retention-sweep \
  --cache "$PRUNING_CACHE" --policy "$PRUNING_POLICY" --runner "$PRUNING_RUNNER" \
  --output "$PRUNING_RESULTS/sweep.json" --threads 4 --split all \
  --figure-dir "$PRUNING_RESULTS/sweep-figures"
```

- **Magnitude:** largest raw magnitudes globally (MUSTAFAR adaptation).
- **Structured:** largest two per consecutive four coordinates at 256, or three
  per eight at 192 in the six-row comparison. The sweep always uses **N:8**,
  where N = survivors / 64; its 256 point is **4:8, not 2:4**.
- **Learned:** identical frozen, 256-trained layer scorers at every budget;
  Learned-192 is **trained at 256, evaluated at 192**.

Comparison commands default to the ten development issues. `--split all` includes
all 50 issues and 5,460 query-layer examples (four positions per training issue,
ten per development issue, across 21 layers). Mean error weights issues equally;
p95 pools all examples. Combined curves include training data and do not measure
held-out generalization.

Named adaptations are masks, not complete upstream reproductions. At 256, replay
uses the existing codec; non-256 budgets use a validated dense numerical codec
simulation, not a persistent-format or serving-kernel test. RoPE, SWA, sink,
Native targets and saved denominators are shared across methods.

Each command writes canonical **JSON**, sibling **CSV/Markdown**, and a
`.partial.json` progress record. The sweep also writes `attention-error-mean.png`
and `attention-error-p95.png`. Existing output/figure paths are rejected. JSON/CSV
errors are fractions; Markdown and plots show percentages. Identity/coverage and
numerical/data checks must pass before a completed table is published.

## Mean, p95 and uncertainty

- **Per example:** Frobenius relative output-L2 error across all 64 heads, after
  inverse output RoPE, using the saved Native denominator (formula below).
- **Mean (primary):** average queries within each issue/layer, then layers and
  issues equally; issues with more queries receive no extra weight.
- **p95 (tail diagnostic):** NumPy's linearly interpolated 95th percentile of
  individual query-layer errors, not issue means or layer percentiles.
- **95% intervals:** 10,000 whole-issue bootstrap draws, seed 42; identical draws
  across methods preserve all queries/layers and give paired mean-difference
  intervals versus magnitude at the same budget. These are in JSON; p95 is not
  an uncertainty interval. Reused development data cannot establish fresh-test
  generalization, full-model task quality or serving performance.

## Math by file

Here $x\in\mathbb{R}^{512}$ is one raw latent row. Each layer learns its own
parameters; scoring uses only that row, not future queries.

### `models.py` — sensitivity and selection

For rank-64 encoder $W$ and odd/even decoders $D_o,D_e$:

$$
a=(\|x\|_2^2/512+\epsilon)^{-1/2},\qquad z=W^\top(ax),
$$

$$
h(x)=\theta+\tfrac12\tanh\!\left(D_o^\top\tanh(z)+D_e^\top\tanh(z^2-1)\right).
$$

$$
P=I-\frac{xx^\top}{\|x\|_2^2+512\epsilon},\qquad
s_i=\log|x_i|+\tfrac12\log\left([P^\top\operatorname{diag}(e^{2h(x)})P]_{ii}\right).
$$

Retain the 256 largest scores per row. The implementation clamps logarithm
inputs for stability. This is learned local sensitivity, not an exact Hessian.

### `losses.py` — output error and gradient proxy

Exact replay includes masked RMSNorm, the Packed codec, RoPE, SWA and the sink:

$$
e=\frac{\|O_{\mathrm{Packed}}-O_{\mathrm{Native}}\|_F}
{\|O_{\mathrm{Native}}\|_F}.
$$

Training uses fixed-mass soft gates:

$$
g_i=\sigma((s_i-\lambda)/\tau),\qquad \sum_i g_i=256.
$$

Straight-through uses exact forward values with surrogate gradients; relaxed
uses continuous attention; anchored adds frozen warm-model codec residuals.
Only exact Packed replay judges checkpoints.

### `train.py` / `evaluate.py` — objective and reporting

With TopMag50 error $e_{\mathrm{TM}}$ on the same state:

$$
\mathcal L=\operatorname{mean}_{\mathrm{issue}}
\operatorname{mean}_{\mathrm{query}}\left[e+2\max(e-e_{\mathrm{TM}},0)\right]+R.
$$

Here $e$ follows the selected training mode. $R$ is $10^{-3}$ times the sum of
mean-square penalties on $\theta$ and each odd/even decoder parameter block.
Select a checkpoint only when exact development mean improves and both mean
and p95 remain no worse than TopMag50. Evaluation averages queries within
issue/layer, then layers and issues equally; p95 pools query-layer errors.

See [the mathematical breakdown](../docs/attention-aware-latent-kv-compression.md)
for derivations, codec details and experimental results.
