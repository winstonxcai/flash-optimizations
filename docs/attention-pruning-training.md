# Attention-aware pruning: method and current results

## Summary

We train a small, layer-specific scorer to choose which **256 of 512 latent coordinates** to retain. Unlike magnitude-only TopMag50, it also estimates each coordinate's downstream attention sensitivity. Model weights remain frozen, and the Packed cache remains **328 bytes per row**.

The learned rank-64 policy reduces mean development attention-output error from **8.58% to 7.18%**, a **16.38% relative reduction**. This is an offline replay result, not yet evidence of improved coding-task quality or serving performance. Development data were also used for checkpoint selection.

This method is a learned extension to Remnant's training-free TopMag50 baseline; only the pruning scorers are trained.

## 1. Data and experimental scope

- Capture untouched DeepSeek-V4-Flash-0731 on TP4: 50 OpenHands issues, one per repository, naturally 100–121k-token prompts. Generate to EOS/tool-call completion; do not execute tools.
- Sample decode states across all 21 CSA layers (2, 4, …, 42), assembling all 64 heads from four ranks. Save FP32 raw latents, norm parameters, queries, selected SWA/CSA rows, logical positions, RoPE, sink, and native outputs.
- Freeze a seed-42 **40 train / 10 development** issue split. Cache four queries per training issue/layer and ten per development issue/layer. Verify hashes and native replay before fitting; reuse captures without loading the model again.

| Split | Issues | Queries per issue per layer | Query-layer pairs |
|---|---:|---:|---:|
| Training | 40 | 4 | 3,360 |
| Development | 10 | 10 | 2,100 |

## 2. Mathematical motivation

Magnitude alone does not describe attention sensitivity: deleting equally large coordinates can produce different output errors. We therefore rank coordinates by magnitude and a learned sensitivity estimate.

Let $x$ be a raw 512-dimensional latent row and let $d = 512$. Before applying learned normalization weights, RMS normalization maps $x$ to $a x$, where

$$
a = \left(\frac{\lVert x \rVert_2^2}{d} + \epsilon\right)^{-1/2}.
$$

Its Jacobian is

$$
J = a P,
$$

$$
P = I - \frac{x x^{\top}}{\lVert x \rVert_2^2 + d\epsilon}.
$$

Here, $I$ is the identity matrix and $\epsilon$ is the RMSNorm stability constant. Let $A$ be the downstream output Jacobian with respect to the normalized row, including normalization weights and RoPE. Deleting coordinate $i$ gives the local approximation

$$
\lVert \delta O \rVert_2^2
\approx x_i^2 \left[J^{\top} A^{\top} A J\right]_{ii}.
$$

### Learned sensitivity model

Approximate downstream sensitivity with a positive diagonal matrix $H(x)$ in place of $A^{\top}A$. Each layer has its own parameters:

| Parameter | Shape | Role |
|---|---|---|
| $W$ | 512 × 64 | Encode the normalized row |
| $D_o$ | 64 × 512 | Decode signed projection features |
| $D_e$ | 64 × 512 | Decode squared projection features |
| $\theta$ | 512 | Coordinate biases, clamped to [−0.5, 0.5] |

Compute the projection and sensitivity in separate steps:

$$
z = W^{\top}(a x).
$$

$$
v_o = D_o^{\top}\tanh(z), \qquad v_e = D_e^{\top}\tanh(z^2 - 1).
$$

$$
h(x) = \theta + \frac{1}{2}\tanh(v_o + v_e).
$$

$$
H(x) = \operatorname{diag}\left(\exp(2h(x))\right).
$$

Squaring, exponentiation, and nonlinearities act element-wise. The diagonal entries estimate sensitivity; they are not exact attention derivatives.

### Coordinate ranking

Keep the **256 highest-scoring coordinates**:

$$
s_i = \log |x_i| + \frac{1}{2}\log\left([P^{\top} H(x) P]_{ii}\right).
$$

The common factor $a^2$ cancels in ranking. This is **local quadratic motivation**, not an exact Hessian or a guarantee for deleting half the coordinates simultaneously. The training loss evaluates full Packed attention-output error. Scoring uses only the row and learned parameters, never future queries.

## 3. Training procedure

### Optimization schedule

- Freeze model weights; fit layer scorers on a model-free L4. Warm-start fitted rank-16 scorers; add 48 features with zero-initialized decoders. Fit rank-64 for 40 updates ($\tau:0.015\to0.008$), then continue the best incumbents for 80 ($0.008\to0.004$).
- Adam: encoder LR $0.0005$, other LR $0.0015$; reset moments for continuation, clip gradient norm to 1, clamp $\theta$ to ±0.5. Each update accumulates all 160 training queries in microbatches of eight; temperatures decay geometrically.

### Forward computation

The hard-mask forward pass follows the Packed path:

1. Retain 256 coordinates, then apply masked RMSNorm and the prescribed BF16 rounding.
2. Quantize and pack using E4M3 values and UE8M0 scales.
3. Reconstruct the latent row and apply RoPE.
4. Compute joint SWA + CSA attention, including the attention sink.
5. Round to BF16, apply inverse RoPE, and round to BF16 again.

The method changes which coordinates survive, not the **328-byte record layout**.

### Backward surrogate

Use a straight-through surrogate for hard masking and codec rounding. The soft gates satisfy

$$
g_i = \sigma\left(\frac{s_i - \lambda}{\tau}\right),
$$

$$
\sum_{i=1}^{512} g_i = 256.
$$

Here, $\sigma$ is the sigmoid, $\tau$ is the temperature, and $\lambda$ is the threshold chosen to enforce the retention constraint. Differentiate through the threshold too. This is **not the exact hard-mask gradient**.

### Loss and checkpoint selection

Let $e$ be the learned policy's error and $e_{\mathrm{TopMag}}$ the TopMag50 error on the same sample. Minimize the following loss, averaged equally across issues:

$$
\mathcal{L} = \operatorname{mean}\left[e + 2\max(e - e_{\mathrm{TopMag}}, 0)\right] + R.
$$

The additional penalty discourages errors worse than TopMag50. The regularizer is

$$
R = 0.001\left(
\operatorname{mean}(\theta^2)
+ \operatorname{mean}(D_o^2)
+ \operatorname{mean}(D_e^2)
\right).
$$

Select a new per-layer checkpoint only when development mean improves over the incumbent and development mean and p95 do not exceed TopMag50. Otherwise, retain the incumbent.

## 4. Evaluation and current result

For each query/layer, after inverse RoPE:

$$
e = \frac{\lVert O_{\mathrm{Packed}} - O_{\mathrm{Native}} \rVert_F}
{\lVert O_{\mathrm{Native}} \rVert_F}.
$$

The norm includes all 64 heads. Compare against magnitude-only TopMag50 using the same codec, selections, native targets and frozen denominators. Average equally across issues, queries and layers: **2,100 development query-layer pairs**.

| Policy | Mean relative attention-output error | Relative reduction vs TopMag50 |
|---|---:|---:|
| TopMag50 | 8.5848% | — |
| Learned rank-64 | 7.1786% | 16.38% |

Best audited all-layer policy: `experiments/attention-pruning/continued-best-policy.json`. Measurements: `experiments/attention-pruning/continued-result.json`.

## 5. Interpretation and limitations

The current result supports the hypothesis that a row-only learned scorer can reduce frozen attention-replay error without increasing the Packed record size.

- The **25% relative error-reduction target has not been met**.
- Development data were reused for selection; fresh-test significance remains unvalidated.
- Frozen replay does not measure error propagation through the full model or closed-loop coding-task quality.
- Scorer overhead and end-to-end serving performance remain unvalidated.

The next evidence needed is a fresh held-out evaluation, followed by separately approved task-quality and serving measurements.
