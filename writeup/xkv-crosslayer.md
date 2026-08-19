# DeepSeek-V4-Flash: xKV cross-layer feasibility on CSA (pre-RoPE compressed latents)

**Question.** Does xKV-style cross-layer low-rank KV compression apply to V4-Flash? I.e. if `W`
adjacent CSA layers share one rank-`r` basis, does that beat compressing each layer alone at the
*same memory*? Measured on the pre-RoPE RMS-normalized compressed latent (`[T, 512]`) of the 21
CSA layers `L ∈ {2,4,…,42}` (`compress_ratio=4`, row-aligned). Metric: relative Frobenius
reconstruction error (lower = better).

---

## Part 1 — SWE-bench probe (n=1 per context, compared at fixed `b = r/W`)

Matched per-layer budget `b = r/W` held constant across W (treated as "equal memory").

**`b = 64` dims/layer** (W1@r64, W2@r128, W3@r192, W4@r256):

| context | T | CKA off-diag | W1 | W2 | W3 | **W4** | W1→W4 gain |
|---|---|---|---|---|---|---|---|
| 16k  | 4096  | 0.46 | 0.552 | 0.523 | 0.501 | **0.482** | 12.6% |
| 32k  | 8171  | 0.42 | 0.563 | 0.537 | 0.518 | **0.502** | 10.8% |
| 64k  | 16384 | 0.40 | 0.559 | 0.535 | 0.519 | **0.506** | 9.6% |
| 128k | 32768 | 0.36 | 0.543 | 0.517 | 0.503 | **0.490** | 9.6% |

**`b = 128` dims/layer** (W1@r128, W2@r256, W3@r384):

| context | W1 | W2 | **W3** | W1→W3 gain |
|---|---|---|---|---|
| 16k  | 0.433 | 0.400 | **0.376** | 13.1% |
| 32k  | 0.445 | 0.417 | **0.399** | 10.5% |
| 64k  | 0.443 | 0.419 | **0.404** | 8.9% |
| 128k | 0.428 | 0.405 | **0.391** | 8.6% |

**Analysis.** Grouping monotonically beats per-layer (`W4 < W3 < W2 < W1`) at every context and
budget — a positive go signal. Two caveats: (a) absolute error is high in the aggressive regime
(`b=64` is 8:1 compression, err ≈ 0.48–0.56); (b) the gain *shrinks* with length (12.6%→9.6%),
tracking the CKA off-diagonal decline (0.46→0.36) — the opposite of xKV's usual "better at long
context" story. CKA shows no strong block structure, so a fixed adjacent W=3–4 window is a
reasonable default with no obvious data-driven group boundary.

---

## Part 2 — RULER re-run (exact baseline + true memory `M_w = Tr + Wrd`, supersedes Part 1)

Part 1's flaw: n=1 is not a measurement, and `b = r/W` is only the *asymptotic* per-layer cost —
it ignores the shared basis, so matched-`b` ≠ matched-memory. This re-run fixes both on RULER, adds
an exact uncompressed baseline (error = 0), and reports true stored memory. Memory in MB at
2 B/elem. **CSA mem** = compressor factor bytes; **total KV** adds the untouched per-layer
sliding-window + indexer bytes. Error = mean over windows × samples ± std across samples.

**8k (qa_2, n=5):**

| method | W | rank | CSA mem | total KV | error |
|---|---|---|---:|---:|---|
| exact CSA     | — | full | 30.9 | 44.7 | 0.0000 |
| per-layer SVD | W1 | 64  | 5.2  | 19.0 | 0.526 ± 0.006 |
| cross-layer   | W2 | 128 | 6.8  | 20.6 | 0.490 ± 0.007 |
| cross-layer   | W3 | 192 | 8.0  | 21.8 | 0.461 ± 0.008 |
| cross-layer   | W4 | 256 | 9.9  | 23.7 | 0.434 ± 0.009 |
| per-layer SVD | W1 | 128 | 10.5 | 24.2 | 0.405 ± 0.007 |
| cross-layer   | W2 | 256 | 13.6 | 27.4 | 0.358 ± 0.008 |
| cross-layer   | W3 | 384 | 16.0 | 29.7 | 0.322 ± 0.010 |

**32k (niah_multikey_2, n=5):**

| method | W | rank | CSA mem | total KV | error |
|---|---|---|---:|---:|---|
| exact CSA     | — | full | 134.8 | 181.6 | 0.0000 |
| per-layer SVD | W1 | 64  | 18.2 | 65.0 | 0.409 ± 0.001 |
| cross-layer   | W2 | 128 | 20.4 | 67.2 | 0.396 ± 0.001 |
| cross-layer   | W3 | 192 | 21.0 | 67.8 | 0.378 ± 0.001 |
| cross-layer   | W4 | 256 | 24.8 | 71.6 | 0.361 ± 0.001 |
| per-layer SVD | W1 | 128 | 36.5 | 83.3 | 0.310 ± 0.001 |
| cross-layer   | W2 | 256 | 40.8 | 87.6 | 0.288 ± 0.001 |
| cross-layer   | W3 | 384 | 42.0 | 88.8 | 0.271 ± 0.001 |

**64k (niah_multikey_2, vt, fwe, qa_2; n=32):**

| method | W | rank | CSA mem | total KV | error |
|---|---|---|---:|---:|---|
| exact CSA     | — | full | 269.3 | 360.1 | 0.0000 |
| per-layer SVD | W1 | 64  | 35.0 | 125.9 | 0.348 ± 0.141 |
| cross-layer   | W2 | 128 | 38.0 | 128.8 | 0.334 ± 0.138 |
| cross-layer   | W3 | 192 | 37.8 | 128.6 | 0.321 ± 0.135 |
| cross-layer   | W4 | 256 | 44.0 | 134.8 | 0.310 ± 0.131 |
| per-layer SVD | W1 | 128 | 70.1 | 160.9 | 0.265 ± 0.123 |
| cross-layer   | W2 | 256 | 76.0 | 166.9 | 0.250 ± 0.116 |
| cross-layer   | W3 | 384 | 75.6 | 166.4 | 0.239 ± 0.111 |

### Main analysis

**1. Grouping still wins (`W4 < W3 < W2 < W1`) at every context and budget** — the SWE result
reproduces on RULER. V4 CSA carries exploitable cross-layer low-rank structure.

**2. But matched-`r/W` ≠ matched-memory, so Part 1 overstated the win at short context.** A rank-`r`
window stores `U ∈ [T, r]` (amortized `r/W`) **+ `V ∈ [r, W·d]`** (codebook, cost `r·d/T`, `d=512`),
so true per-layer cost is `b_eff = r/W + r·d/T`. Widening W forces rank up and ships a larger
codebook that doesn't amortize with length. At 8k, W1@r64 costs **5.2 MB** vs "equal-budget" W4@r256
at **9.9 MB** (1.9× the memory). The gap shrinks as `T` grows:

| context | W1@r64 | W4@r256 | gap |
|---|---|---|---|
| 8k  | 5.2  | 9.9  | 1.90× |
| 32k | 18.2 | 24.8 | 1.36× |
| 64k | 35.0 | 44.0 | 1.26× |

Break-even (`r/W = r·d/T`) is `T = W·d` → ~8k context for W4; below that the codebook outweighs the
payload. Honest comparison is at equal `b_eff`, not equal `r/W`.

**3. Structure is strongly prompt-family-dependent** — invisible to n=1. At 64k, W1@r64 error by
task: **vt 0.20, fwe 0.23, niah_multikey_2 0.41, qa_2 0.55**. The ±0.14 pooled std is almost
entirely between-task (within-task < 0.003). Retrieval-style prompts compress far better than QA —
deployment must be conditioned on workload, not a pooled number.

**4. Absolute error stays high in the aggressive regime.** `b≈64` is 8:1 per-layer compression; even
with grouping, error is 0.31–0.53. Whether the ~5–13% relative gain moves end-task accuracy needs a
downstream eval, not Frobenius error alone.

**5. No length-widening story** — as in Part 1, relative gain does not grow with context (CKA
off-diag mean 0.44/0.55/0.46 at 8k/32k/64k, no bright diagonal blocks). Do not expect the advantage
to widen at 256k+; a fixed adjacent W=3–4 window is a reasonable default.

_Notes: exact baseline stores the full T×d latent per layer (ceiling); compressed uses
`M_w = T·r + W·r·d` tiled over `⌈21/W⌉` non-overlapping groups. Error uses overlapping windows,
memory uses tiled groups (deployment cost). RULER on disk has only 8k/32k/64k (no 16k/128k
counterpart to Part 1); 8k/32k are single-task n=5 (secondary), only 64k is 4-task._

---

## Part 3 — RULER end-task accuracy: native CSA vs W3 cross-layer @ b=64

Parts 1–2 measure the *proxy* (Frobenius recon error of the pre-RoPE latent). This measures the
*end task*: greedy RULER decoding, scored on the real answer. `native CSA` = the model's built-in
path — per-layer compressed latents (`compress_ratio=4`, full 512-dim each, indexer dense); this
is the KV compression V4-Flash runs on its own. `W3` = cross-layer
low-rank: adjacent groups of 3 CSA layers share a rank-192 basis (b = 64 dims/layer, the same
`W3@r192` cell as Part 2), the reconstruction injected back into the KV store (2-pass:
capture → joint SVD → inject). Tasks are the five hardest from xKV's Table 1 — qa_2, qa_1, fwe,
vt, niah_multivalue — n=100 fresh samples/task at 32k. The 8k leg (n=100) uses the original
4-task set (qa_2, fwe, vt, niah_multikey_2), which predates qa_1/niah_multivalue being generated.
64k is deferred (memory-blocked; see notes).

**8k (n=100/task):**

| task | native CSA | W3@b64 | Δ |
|---|---:|---:|---:|
| niah_multikey_2 | 1.000 | 0.980 | −0.02 |
| vt | 1.000 | 0.940 | −0.06 |
| fwe | 0.993 | 0.997 | +0.00 |
| qa_2 | 0.740 | 0.710 | −0.03 |
| **mean** | **0.933** | **0.907** | **−0.03** |

**32k (n=100/task, the five hardest from xKV Table 1):**

| task | native CSA | W3@b64 | Δ |
|---|---:|---:|---:|
| qa_2 | 0.740 | 0.730 | −0.01 |
| qa_1 | 0.880 | 0.870 | −0.01 |
| fwe | 0.987 | 0.983 | −0.00 |
| vt | 0.994 | 0.998 | +0.00 |
| niah_multivalue | 1.000 | 1.000 | 0.00 |
| **mean** | **0.920** | **0.916** | **−0.00** |

_(The earlier 32k niah_multikey_2 cell, from the pre-Table-1 run, is 1.000/1.000 — consistent.)_

**Analysis.**

**1. At 32k the cross-layer compression is free — now on the actual five hardest tasks.** Macro
mean −0.004 (0.920 vs 0.916), per-task deltas all within ±0.01 — sampling noise, not signal. The
shared rank-192 basis recovers native-CSA accuracy at long context while cutting the CSA latent to
**21.0 MB vs 134.8 MB** (Part 2 `W3@r192`, ~6.4×).

**2. At 8k the cost is small and no longer dominated by a single outlier.** Macro −0.03
(0.933 vs 0.907). The earlier vt −0.25 was a single-run outlier; at n=100 it's −0.06. The 8k
penalty still tracks the higher 8k recon error (0.461 at 8k vs 0.378 at 32k, Part 2): short
context is codebook-dominated (Part 2 §2), so a wider effective rank is needed to hold the same
per-token fidelity.

**3. The niah tasks sit at the ceiling.** niah_multikey_2 (8k, −0.02) and niah_multivalue
(32k, 0.00) both stay 1.000 under the low-rank recon — retrieval-style tasks are the most robust
to latent perturbation, consistent with Part 2's "structure is strongly prompt-family-dependent."

**4. qa_2 is baseline-limited, not compression-limited.** Its native-CSA ceiling is 0.740 at both
8k and 32k; the W3 delta (−0.03 / −0.01) is small relative to that. qa_2 headroom comes from the
base model, not from KV fidelity.

_Notes: the baseline column is the model's native CSA path (already `compress_ratio=4`), so W3
measures the *additional* cost of cross-layer low-rank on top of native CSA, not the full
uncompressed-KV baseline. Scoring: niah = needle, vt/fwe = multi-word recall, qa_2 = string match. **64k is
deferred**: the capture pass materializes ~2.8 GB of fp32 latents on rank 0, and neither viable
config leaves room for it — at tp=8 the 0.95-mem-fraction KV pool leaves only ~0.5 GiB free; at
tp=4 the weights (~8.7 GB/rank) plus SGLang's minimum pool (0.89 × 80 GB) fill the card outright.
The long-context result is therefore the 32k leg, where cross-layer SVD is free._
