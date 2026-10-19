<!-- Authored by Winston Cai. -->

# Attention-Aware Latent KV Compression

## Summary

We train a small, layer-specific scorer to choose which **256 of 512 latent coordinates** to retain. Unlike magnitude-only TopMag50, it also estimates each coordinate's downstream attention sensitivity. Model weights remain frozen, and the Packed cache remains **328 bytes per row**.

The learned rank-64 policy reduces mean development attention-output error from **8.5848% to 7.1786%** (**16.38% relative reduction**); the best per-layer mix reaches **7.1727%** (**16.45% reduction**). These are offline replay results, not evidence of improved coding-task quality or serving performance. Development data were also used for checkpoint selection.

This method is a learned extension to Remnant's training-free TopMag50 baseline; only the pruning scorers are trained.

Maintained implementation and CPU checks: [training pipeline](../training/README.md).
See [data/checkpoint access](../training/README.md#cache-and-checkpoints) and
[shared-image capture setup](../scripts/local/README.md#isolated-latent-kv-capture).
Historical mechanism studies below retain their original findings; their archived
sources are separate from the maintained training/evaluation commands.

## 1. Data and experimental scope

- Capture untouched DeepSeek-V4-Flash-0731, revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`, on TP4: 50 OpenHands issues, one per repository, naturally 100–121k-token prompts. Generate to EOS/tool-call completion; do not execute tools.
- Sample decode states across all 21 CSA layers (2, 4, …, 42), assembling all 64 heads from four ranks. Save FP32 raw latents, norm parameters, queries, selected SWA/CSA rows (128 SWA + 512 CSA selections), logical positions, RoPE, sink, and native outputs.
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

The best mix adds 32 updates to layers 8/12 ($\tau:0.008\to0.004$): layer 8 differentiates continuous soft attention; layer 12 adds frozen warm codec residuals $\Delta K=K_{\rm Packed}-K_{\rm continuous}$ before attention and a final rounding residual after inverse RoPE. Compute both at the warm hard mask; the output residual compares rounded/unrounded versions of the same warm Packed attention output. These training-only offsets stay fixed. Optimizer, loss and exact Packed selection rules remain unchanged; the other 19 layers remain unchanged.

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

For straight-through training, $e$ uses Packed error; soft/anchored variants use their proxy error. The TopMag50 baseline is always Packed error. The additional penalty discourages errors worse than TopMag50. The regularizer is

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

| Policy | Mean relative L2 | Mean reduction | p95 relative L2 | p95 reduction |
|---|---:|---:|---:|---:|
| TopMag50 | 8.5848% | — | 13.0989% | — |
| Best uniform run | 7.1786% | 16.38% | 11.0497% | 15.64% |
| Best per-layer mix | 7.1727% | 16.45% | 11.0298% | 15.80% |

Reductions are versus TopMag50. p95 is across the 2,100 query-layer errors, not issue means.

Historical result/policy summaries are supplied with the external data bundle.
Checkpoints remain under `~/.cache/remnant/attention-pruning/`: the uniform run
uses `continued-all-results-001/continued-all-001/selected-layer-XX.json`; the
per-layer mix substitutes `anchored-pilot-001/outputs/relaxed-layer-08.json` and
`anchored-layer-12.json`. The [training guide](../training/README.md#cache-and-checkpoints)
lists retained comparison/sweep records and policy relocation instructions.
Development was repeatedly used for selection; fresh-test significance, task
quality and serving overhead remain unvalidated.

### Retention-budget comparisons

The maintained [comparison and sweep commands](../training/README.md#compare-masks-and-sweep-retention)
evaluate magnitude, structured and frozen learned masks on the same 2,100
development query-layer examples. Training remains at 256 survivors; evaluation
sweeps 64–448 in steps of 64. The sweep's structured mask is **N:8**, including
**4:8 at 256**, whereas the separate six-row comparison uses **2:4 at 256**.
Non-256 replay is an offline numerical simulation, not serving validation.

- [Mean relative output-L2 curves](figures/attention-error-mean.png): average
  queries within issue/layer, then layers and issues equally.
- [p95 relative output-L2 curves](figures/attention-error-p95.png): the pooled
  95th percentile of query-layer errors, not an uncertainty interval.

## 5. Interpretation and limitations

The current result supports the hypothesis that a row-only learned scorer can reduce frozen attention-replay error without increasing the Packed record size.

- The **25% relative error-reduction target has not been met**.
- Development data were reused for selection; fresh-test significance remains unvalidated.
- Frozen replay does not measure error propagation through the full model or closed-loop coding-task quality.
- Scorer overhead and end-to-end serving performance remain unvalidated.

The next evidence needed is a fresh held-out evaluation, followed by separately approved task-quality and serving measurements.

## 6. Mathematical limits and mask representability

For a smooth attention pipeline, let $J_q$ map a perturbation of the latent
state to the output for query $q$. Its normalized Gauss–Newton metric is

$$
G_q = \frac{J_q^{\top}J_q}{\lVert O_{\mathrm{Native},q}\rVert_F^2}.
$$

At an expansion point $x_0$, retain the existing residual
$r_q=F_q(x_0)-O_{\mathrm{Native},q}$:

$$
e_q(x_0+\delta x)^2\approx
\frac{\lVert r_q+J_q\delta x\rVert_F^2}
{\lVert O_{\mathrm{Native},q}\rVert_F^2}.
$$

This reduces to $\delta x^\top G_q\delta x$ only when $r_q=0$.
A pruned warm start generally has nonzero residual. Even a smooth unpruned
proxy can differ from the quantized Native target. Dropping that residual
would discard the first-order direction for reducing its existing error.
Gauss–Newton also omits residual-weighted second derivatives of the output;
it is not the full loss Hessian.

This includes the full normalization and attention derivatives, including the
sink. Hard mask selection and codec rounding are not smooth; the expression is
a local model, not an exact formula for finite Packed error.

For a row-only predictor, the conditional average

$$
G^{\star}(x) = \mathbb{E}[G_q \mid x]
$$

is the optimal matrix prediction under squared Frobenius prediction loss, when
$J_q$ denotes the corresponding row block. It averages query-dependent
sensitivity using only information available in that row. This does **not**
make its diagonal ranking the globally optimal pruning policy. Joint deletions
include cross-coordinate and cross-row terms, and pruning half the coordinates
changes normalization, attention probabilities, and quantization scales.
Our evaluation target remains mean relative L2, rather than mean squared L2.

The actual problem is to minimize mean exact Packed error subject to 256
survivors per row and a shared row-only scoring rule. A query-aware oracle has
more information and fewer sharing constraints. Any feasible mask it finds
establishes an attainable error for that oracle; a local search cannot certify
the minimum error or the maximum possible improvement. A continuous relaxation
would provide a rigorous bound only if it contained the actual feasible Packed
outputs and was itself solved with an appropriate optimality certificate.

### A problem we can solve: can the sensitivity represent a target mask?

For a fixed row and target mask, define $h_k=H_{kk}$ and

$$
C_{ik}=x_i^2 P_{ki}^2,
\qquad q_i=(Ch)_i.
$$

Here $q_i$ is the squared ranking score: taking its square root or logarithm
preserves its order. The implementation uses the same small absolute-value
floor as the scorer. Because the current log-weights lie in $[-1,1]$, their
positive diagonal entries satisfy $e^{-2}\le h_k\le e^2$.
For a desired retained set $S$, solve the linear program

$$
\begin{aligned}
\max_{h,b,t}\quad &t\\
\text{subject to}\quad
&(Ch)_i\ge b+t &&(i\in S),\\
&(Ch)_j\le b-t &&(j\notin S),\\
&e^{-2}\le h_k\le e^2 &&(k=1,\ldots,512).
\end{aligned}
$$

A positive optimum supplies a diagonal that strictly realizes the mask.
A negative optimum rules it out within this interval. The implementation
checks a separately computed score gap, primal feasibility, dual gap,
stationarity and complementarity; these are floating-point certificates,
not exact-rational proofs. Near-zero cases remain unresolved. Independent
row diagonals are more expressive than our shared neural scorer, so feasibility
does not guarantee that one network can realize all masks simultaneously.

This solves a useful part of the sensitivity question, but it does not find
the mask with the lowest attention error. Even one row has
$\binom{512}{256}\approx4.73\times10^{152}$ possible masks. Their exact Packed
loss includes changing RMSNorm, quantization and softmax, as well as interactions
with other rows. An inverse Hessian does not solve that finite selection problem.
For example, [Optimal Brain Surgeon](https://proceedings.neurips.cc/paper_files/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf)
solves a local quadratic deletion problem while adjusting the remaining values.
That compensation is an additional degree of freedom beyond our mask-only
selection and prescribed normalization/codec.

To certify a maximum reduction, fix the same data, codec and feasible policy
class, and establish bounds $E_{\mathrm{lower}}\le E^*\le E_{\mathrm{best}}$.
Then, for a positive TopMag baseline $E_0$,

$$
1-\frac{E_{\mathrm{best}}}{E_0}
\;\le\; R^*
\;\le\;1-\frac{E_{\mathrm{lower}}}{E_0}.
$$

A feasible learned policy supplies the first side. The trivial lower error
bound of zero only gives a 100% upper bound on reduction. We currently lack
a useful globally certified lower error bound for the exact Packed objective.
Oracle results and neural-policy results must use their own feasible classes;
neither a local optimum nor a training improvement proves a development ceiling.

## 7. Evidence about the remaining headroom

The existing all-layer replay reports **26.43% training reduction** and
**16.45% development reduction**. Its training-gap audit flags one historical
development TopMag mismatch of approximately 0.00000596 in relative-L2 units;
the report is not fully compatible with every historical control. The gap
nevertheless motivates studying generalization as well as optimization.

A separate CPU pilot on two training issues, one query per layer, searched
feasible masks from both TopMag and the selected learned scorer:

| Policy on the same 42 query-layer samples | Mean relative L2 | Reduction vs pilot TopMag |
|---|---:|---:|
| TopMag50 | 7.5616% | — |
| Current learned scorer | 5.3684% | 29.01% |
| Learned scorer + up to 16 query-aware swaps | 5.1878% | 31.39% |

Every accepted swap was verified using full Packed replay. The search was
still improving at the final step in 36 of 42 learned-start cases. Thus
**31.39% is a demonstrated oracle improvement on this small training pilot,
not a mathematical maximum or a development result**. No weights were changed.

A completed shared-mask experiment tied identical rows across training queries
and tested additional weight updates on layers 8/12 using four issue folds.
The shared oracle reduced fitting error by **1.57% / 0.36%**, respectively,
but no student checkpoint improved exact fitting mean. All guards retained
the incumbent, with **0% additional improvement** on the fold holdout.
This tested incremental transfer; the starting scorer had already seen the
original training issues.

The failure exposed a mismatch between the preference objective and actual
TopK selection. The objective rewarded improvement relative to the original
score gap without requiring the added coordinate to outrank the removed one.
Its loss fell sharply, but none of 2,646 desired add/remove pairs reached the
correct absolute order. The final rejected students recovered 18/552 desired
additions and no desired removals, while altering many other rows. Counts
include repeated occurrences across overlapping fitting folds.

An absolute-margin follow-up completed the same eight folds. It recovered
489/552 desired additions but only 15/552 intended removals; fitting mean
regressed by 1.34–2.79% at the final updates. All checkpoint guards retained
the incumbent. The LP also found 14 of 43 unique changed row/mask targets
numerically infeasible within the current bounds; 29 had positive witnesses.
These selected targets do not estimate a population frequency or an error
ceiling. Source-bound records preserve both the failed update and LP evidence.

A second follow-up taught the whole shared 256-coordinate mask with the
existing fixed-budget logistic loss. It recovered 77/552 desired additions
and 20/552 intended removals. Its best fitting-mean improvement was only
0.0381%; all mean-improving checkpoints worsened p95. Both follow-ups therefore
failed the frozen gate and retained the incumbent in every fold. Together they
took 192.01 seconds of local CPU execution, with no Modal allocation.

| Incremental training objective | Final additions recovered | Final removals recovered | Selected fold-held-out improvement |
|---|---:|---:|---:|
| Relative score-gap preference | 18 / 552 | 0 / 552 | 0% |
| Absolute score-margin preference | 489 / 552 | 15 / 552 | 0% |
| Complete shared-mask logistic loss | 77 / 552 | 20 / 552 | 0% |

Recovery counts describe final rejected updates on overlapping fitting folds.
Held-out metrics describe the selected checkpoints, which all fell back to the
warm model. Do not interpret the unchanged selected result as proof that the
rejected weights were harmless.

The next useful mechanism is a minimum-change constrained weight update:
realize feasible full-mask constraints while controlling collateral changes to
other rows. Its feasibility and actual Packed error need testing before another
training sweep. We have not implemented or measured that update. Simply widening
sensitivity bounds, repeating pair losses or adding more queries has not shown
a substantial gain in prior experiments. Keep the final development target at
**6.4386% mean relative L2** (10.24% below the incumbent), with existing p95
guards and a subsequent fresh-test evaluation. Neither these failures nor the
LP infeasibility of particular masks proves that 25% is impossible.

Historical mask-search evidence is archived under
`~/.cache/remnant/attention-pruning/`: `mask-headroom-20261012-001/`,
`shared-mask-transfer-20261012-001/`, `absolute-margin-20261012-001/` and
`full-mask-20261012-001/`. Request the corresponding source snapshots with these
records; the maintained evaluator does not rerun the retired oracle/student probes.

## 8. Query geometry across layers and heads — 2026-10-12

Analyzed all **5,460 cached query-layer samples**: 260 query positions across
50 issues and 21 CSA layers, with all 64 heads. Query tensors are the captured
post-RoPE vectors. Statistics average equally across issues, then queries;
pooled spectra also average equally across heads. Raw magnitudes retain head
scale. Directional statistics use $u=q/\lVert q\rVert_2$.

| Figure | What it compares |
|---|---|
| [Query magnitudes](figures/query-magnitude-by-layer-head.png) | Mean absolute value over all 512 coordinates, separately for each head/layer |
| [Coordinate energy](figures/query-energy-by-layer-head.png) | Mean squared unit-normalized query coordinates; each head profile sums to one |
| [Head cosine](figures/query-head-cosine-by-layer.png) | Signed cosine between heads of the same query, then averaged over issues |
| [Head energy-profile cosine](figures/query-head-energy-cosine-by-layer.png) | Similarity of heads' average coordinate-energy profiles |
| [Direction and split stability](figures/query-direction-and-stability.png) | Cross-issue directions, inverse query RoPE, train/development profiles and top-32 overlap |
| [Query subspaces](figures/query-subspace-spectrum.png) | Second-moment spectra, effective rank, pooled-mean removal and training-subspace transfer |

**Coordinate preferences are stable across the split.** Across the 1,344
layer/head profiles, train/development energy cosine averages **0.99154**
(range 0.96624–0.99947), Spearman correlation has median **0.95292**, and top-32
coordinate overlap averages **88.76%**. This is stability of average profiles,
not a claim that each query has an identical coordinate ranking.

**Directions vary more than those energy profiles.** Same-head signed cosine
averages **0.36692** across distinct issues, versus **0.45339** within issues.
After inverse query RoPE, cross-issue cosine increases to **0.52345**.
The cross-issue statistic weights issue pairs equally and includes all their
query pairs; these are means, not similarity quantiles or significance tests.
Removing query RoPE diagnoses phase effects but does not replace the actual
query-to-key relative rotation used by attention.

**RoPE and head differences matter.** Coordinates 448–511 are 12.5% of the
dimensions but contain **30.65%** of normalized head energy on average.
Layer 22 coordinates 474/475 alone account for **21.40%** of head-pooled energy.
Large NoPE coordinates also occur: layer 4 coordinate 290 contains **4.67%**.
Head maps show heterogeneous groups, particularly in layers 24, 34 and 40;
these similarities describe query geometry rather than interchangeable heads
or identical attention-output sensitivity.

**A strong leading subspace coexists with a long tail.** Uncentered participation
rank ranges from **6.52 to 50.95**, but capturing 90% directional energy takes
**151–311 components**. Removing the pooled mean raises participation rank to
**25.02–161.16**. This centering removes one overall mean across heads and
queries; it does not separately remove each head's mean. A training top-64
subspace captures **65.05%** of development directional energy on average.
These findings support a low-rank-plus-diagonal prior over a pure rank-64 model.

### How this can inform a sensitivity experiment

Save both the raw query second moment $C=\mathbb E[qq^\top]$ and the normalized
directional moment $\mathbb E[uu^\top]$. The plotted spectra use the latter.
For a fixed perturbation $\Delta k$ in the attention key's rotated coordinates,
the raw second moment gives the exact mean squared change of the scaled logit:

$$
\mathbb E[(\mathrm{scale}\,q^\top\Delta k)^2]
=\mathrm{scale}^2\Delta k^\top C\Delta k.
$$

This is a logit-error metric. Attention-output sensitivity also contains
attention probabilities, the value path, output residuals and the sink.
For a raw latent perturbation, map the query metric through the key's RoPE
and normalization Jacobians before using it. Inverse query RoPE alone does
not perform that mapping.

A controlled follow-up can initialize or regularize the learned metric with
a **training-only query second moment**, preserving its mean contribution and
representing covariance as a low-rank term plus a diagonal tail. Fit residual
corrections with actual Packed output-L2 guards, and compare against the same
incumbent on issue folds. The current analysis suggests this mechanism; it
does not measure a pruning improvement or meet the 25% goal.

Evidence is in
`~/.cache/remnant/attention-pruning/full-cache-001/query-geometry-20261012-094900/`:
JSON provenance/spectra, per-head and coordinate CSVs, and per-layer moment
arrays. Six diagnostic tests passed and all-layer numerical invariants were
checked. This CPU analysis took 108.18 seconds; no weights, kernels, pins or
Modal allocations changed.

## 9. TopMag before and after key RoPE — 2026-10-12

Compared all 5,460 cached query-layer samples at 256/512 survivors on CPU.
Pruning after RoPE follows normalization, whereas production prunes raw latents
before masked RMSNorm. A second pre-RoPE control therefore keeps the unpruned
normalization fixed. Its comparison with post-RoPE pruning isolates the pruning
and codec coordinate basis; the production comparison also changes selection
and normalization placement.

| TopMag50 path | Development mean output L2 | Development p95 output L2 |
|---|---:|---:|
| Production: raw mask → masked RMSNorm → pack → RoPE | 8.5848% | 13.0989% |
| Control: unpruned RMSNorm → mask → pack → RoPE | 9.4178% | 14.1031% |
| Post: unpruned RMSNorm → RoPE → mask → pack | 9.5141% | 14.2925% |

All paths use global magnitude selection, the same BF16/E4M3/UE8M0 codec core,
unchanged queries/SWA/sink and the original Native output denominator. The two
controls omit masked renormalization. Post-RoPE storage would require a changed
decode interpretation; it is an offline format simulation.

Post-RoPE pruning is **1.0221% relatively worse** than normalized-pre pruning
and **10.8248% worse** than production pre on development mean. The controlled
mean difference is +0.09625 percentage points, with an issue-bootstrap 95%
interval of [+0.08020, +0.10983]. It worsens 70.67% of development samples.
Across the full cohort, only layers 30/40 improve controlled mean. Before FP8,
the BF16-row control errors are 9.3494% pre versus 9.4457% post, so the trend
is already present without that quantization stage.

Post-RoPE TopMag is optimal for mask-only Euclidean reconstruction in its own
basis. It cannot dominate all other pruning bases: rotating a pre-RoPE sparse
vector can give more than 256 nonzero coordinates. Attention-output error adds
query direction and softmax/value interactions, so latent magnitude optimality
does not guarantee the best output error. **Keep production pre-RoPE pruning.**

[Layer-by-layer mean/p95 plot](figures/topmag-rope-comparison.png).
Evidence is in
`~/.cache/remnant/attention-pruning/full-cache-001/topmag-rope-20261012-001/`.
Five focused checks passed; execution took 225.02 CPU seconds with no Modal
allocation. The production reference matches the existing pack/replay path,
but historical CPU sweep errors are not bit-exact: Torch 2.8.0 versus 2.10.0,
maximum scalar difference 0.00144004 percentage points. That discrepancy is
recorded separately without relaxing the historical comparison threshold;
the new policies use paired replay in one environment. No weights or production
code were changed.

## 10. Training-only query-prior pilot — 2026-10-12

Completed a matched CPU experiment on layers 8/12: eight seed-42 original
training issues, four folds with six fitting/two held-out issues each, four
queries per issue. Each prior used only its fitting queries, expressed in
the selected keys' pre-RoPE frame. Calibration preserved the mean outer product,
32 covariance directions and the diagonal residual, included norm weights,
and added a 1e-6 ridge after trace normalization.

The two arms started from the same incumbent. Both froze encoders and trained
the existing bias/decoders for 16 Adam updates, with the established soft loss
at layer 8 and warm codec anchor at layer 12. The prior arm added lambda=0.01
times the squared disagreement of row-centered log sensitivity scores. The
control had lambda=0. Both used exact fitting mean/p95 guards for selection.
The scorer parameter schema and forward operations were unchanged; the prior
matrices were consumed only by training.

| Selected checkpoint | Incremental-holdout mean output L2 | p95 |
|---|---:|---:|
| Incumbent | 8.1500% | 12.3711% |
| Ordinary continuation | 8.4015% | 12.5618% |
| Continuation + query prior | 8.4143% | 12.6543% |

**Reject this query-prior regularizer.** Its holdout mean is 3.2428% worse than
warm and 0.1527% worse than continuation. It beats continuation in two of eight
folds but never improves over warm. Fitting improves by 2.51% at layer 8 and
3.44% at layer 12, similar to the control's 2.54%/3.47%. Thus the updates learn
the fitting cohort without improving incremental transfer. The warm scorer
already saw the original training issues; this is not an independent test.

Nearly all held-out rows change at least one retained coordinate, but the
changes are small: average replaced survivors are 3.13 per row for control
and 3.19 for prior, out of 256. Mean survivor overlap remains 98.78% / 98.75%.
The magnitude of mask changes and their output-error consequences should be
reported separately.

This pilot does not show that all query priors fail. It tests a particular
global logit-based prior, rank, strength and update recipe. Query magnitude
stability alone does not supply the attention probabilities, value contribution
or conditional sensitivity needed to predict output-L2 error. The full-layer
incumbent remains at 16.45% development improvement; the 25% target is unmet.

Evidence: `~/.cache/remnant/attention-pruning/query-prior-20261012-001/`.
Eight folds completed in 415.27 CPU seconds. Eighteen focused tests passed;
the audit recomputed every prior and exactly replayed all selected fitting
and holdout metrics. Source archives, calibration receipts and all checkpoints
are preserved. No Modal job, production change or weight promotion occurred.

Use the [training guide](../training/README.md) for maintained fitting,
comparison and retention-sweep commands. Historical diagnostic records and
their source snapshots remain in the external archives named above; obtain
them from Winston when reproducing a retired study. A current-code rerun is a
new measurement, not a replacement for those historical results.
