<!-- Author: Winston Cai. -->

# Attention-pruning training

Fit layer-local sensitivity scorers from cached latent KV states; evaluate with
exact Packed attention replay. No model weights are loaded.

## Run

Use the existing `remnant` container from the shared `remnant:v0.5.18` image;
no separate training image or host Python installation is needed. From the
repository root on the host, enter the mounted checkout:

```bash
export TRAIN_GPU=0  # Replace with an allocated, idle GPU visible in the container.
docker exec -it -e CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
  --workdir "/mnt/host_root$PWD" "${CONTAINER:-remnant}" bash
```

Run the following **inside the container**. Replace the example paths with your
existing cache, warm policy and hash-matching calibration source tree. Host
paths are available under `/mnt/host_root`; output paths must be fresh and
persistent. The container's Torch version must match the cache's source version.

```bash
export PRUNING_CACHE=/mnt/host_root/data/pruning/cache
export PRUNING_POLICY=/mnt/host_root/data/pruning/policy.json
export PRUNING_RUNNER=/mnt/host_root/data/frozen-sglang/python/sglang/benchmark/remnant/calibrate.py
export PRUNING_OUTPUT=/mnt/host_root/data/pruning/new-fit

# Train; the generated policy retains untouched layers.
python -m training.train --cache "$PRUNING_CACHE" --policy "$PRUNING_POLICY" \
  --runner "$PRUNING_RUNNER" --output "$PRUNING_OUTPUT" \
  --recipe continuation --layers 2,4 --device cuda

# Evaluate the generated policy (this standalone command is currently CPU-only).
python -m training.evaluate --cache "$PRUNING_CACHE" \
  --policy "$PRUNING_OUTPUT/policy.json" --runner "$PRUNING_RUNNER" \
  --output /mnt/host_root/data/pruning/evaluation.json --layers 2,4

# CPU integration checks.
python -m unittest training.tests.test_integration.TrainingIntegrationTests -v
```

Training uses the selected GPU, staging one cached batch at a time; no serving
process is started. Standalone evaluation and integration checks use CPU.
Continuation, relaxed and anchored recipes require rank-64 checkpoints; the
rank64 recipe expands rank-16 checkpoints. Use `--help` for options. Capture setup
and commands are in [the local launch guide](../scripts/local/README.md#isolated-latent-kv-capture).

Real Docker checks use the existing shell launcher and publication test described
in the local launch guide.

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
