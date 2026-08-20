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

---

## Part 4 — Cross-layer low-rank on the CSA indexer (64k end-task)

Part 3 showed W3 cross-layer SVD is *free at 32k* on the **compressor latent** (`[T,512]`). This part
runs the same transfer on the **CSA indexer** — the per-query top-512 token selector whose per-layer
score inputs are the 128-dim indexer keys (`index_head_dim=128`, `index_n_heads=64`, fp32
`[B, S_q, 64, S_kv/4]` score tensor). The indexer is likely the largest non-MoE compute component in
V4-Flash, so the win metric here is **compute (dims/token in the indexer)**, not the memory metric of
Part 3. W3@b64 on 128-dim keys is a **2:1** compression — adjacent groups of 3 CSA layers share a
rank-192 basis (b = 64 dims/layer), gentler than Part 3's 8:1 on the 512-dim latent — but the object
is a *selector*, whose top-512 set is a hard threshold on the score tensor.

**Method / environment.** Same 2-pass harness as Part 3, retargeted with `--recon-target indexer`:
pass 1 captures the normed 128-dim indexer keys (`on_compress`, indexer-only gate), computes the
joint SVD, pass 2 injects the `[T,128]` reconstructions back into the indexer path. Only-indexer
proof (smoke, n=3, `XKV_DEBUG=1`): 1008 `compress` + 504 `inject` events all `d=128,
is_indexer=True`; **zero** compressor / `d=512` events. Model served in SGLang 0.5.15
(`ruler-eval` container), tp=4, mem-fraction 0.95, **two parallel legs on 8 GPUs** (qa_2, qa_1, fwe
on GPUs 0–3; vt, niah_multivalue on GPUs 4–7; distinct MASTER_PORT + ctrl-dir + out per leg).
**64k**, the five hardest tasks × n=50 (250 samples), matching the sibling TopMag-indexer 64k run
(Section 4 of the indexer-prune writeup) and Part 3's 32k leg. Date 2026-08-19. 64k is feasible here
where Part 3 deferred it because the indexer capture is `[T,128]` — 4× smaller than the `[T,512]`
latent that OOM'd Part 3's rank 0.

**64k RULER (n=50/task):**

| task | native indexer | W3@b64 indexer | Δ pts |
|---|---:|---:|---:|
| qa_2 | 0.760 | 0.760 | 0.0 |
| qa_1 | 0.820 | 0.800 | +2.0 |
| fwe | 0.867 | 0.827 | +4.0 |
| vt | 0.992 | 0.996 | −0.4 |
| niah_multivalue | 1.000 | 1.000 | 0.0 |
| **mean** | **0.888** | **0.877** | **+1.1** |

The dense column agrees with the TopMag-indexer 64k dense baselines (qa_2 0.76 vs 0.74, qa_1 0.82
vs 0.80, fwe 0.87 vs 0.85, vt 0.99 vs 0.99, niah 1.00 vs 1.00) — the run is measuring the native
indexer path.

**Analysis.**

**1. Not free at 64k — the indexer is more sensitive than the latent.** Macro mean −1.1 pts
(0.888 vs 0.877). Part 3's latent W3 was −0.004 at 32k; the same W3@b64 on the indexer keys costs
~1 pt even at 64k. The indexer's output is a *selection* — a hard top-512 threshold on the score
tensor — so even a small basis error moves tokens across the selection boundary and changes the
attended set wholesale. Averaged Frobenius error on the latent (Part 2) does not capture
selection-boundary sensitivity; this is the end-task cost of that gap.

**2. The cost is workload-shaped: retrieval/needle free, word-recall penalized.** niah_multivalue
(0.00) and vt (−0.4, w3 ≥ dense) sit at the ceiling; qa_2 is untouched (0.0). The penalty
concentrates in **fwe (+4.0)** and **qa_1 (+2.0)** — the multi-word extraction/recall family. This
runs counter to Part 2 §3's "retrieval compresses best": here the *selection* the indexer performs
matters most for tasks whose answer is a specific short word list.

**3. The fwe magnitude is soft but directionally consistent.** fwe settled at 0.867 vs 0.827 at
n=50; the gap ran ~10→4 pts across the run as the dense column drifted down to its (~0.85) baseline
while w3 sat flat near 0.83. At n=50 the per-column binomial SEM is ≈4.8 pts (difference SEM ≈6.8),
so the 4-pt gap and the 2-pt qa_1 gap are individually within ~1 SEM — the family pattern is the
signal, the individual magnitudes are not resolved.

**4. The compute win is real but kernel-gated.** Indexer key-dims/token halve (128→64) at 2:1.
Whether that moves wall-clock depends on the fused indexer kernel skipping the dropped dims — the
same "savings materialize only in the kernel" caveat as the TopMag-indexer sparse-store note. This
run bounds the accuracy ceiling: ~1 pt macro at 64k, worse on word-recall.

_Notes: same caveat as Part 3 — the native-indexer column already runs the model's built-in
`compress_ratio=4`; W3 measures the *additional* cost of cross-layer low-rank on the indexer path.
n=50/task at 64k (RULER on-disk cap + scheduling match with the sibling TopMag-indexer run); no
8k/32k indexer legs yet. Artifacts:
`transferibility/out/ruler_csa_idx_w3_64k{,_a,_b}.json`, `transferibility/par_idx_w3_64k_{a,b}.log`,
launcher `transferibility/sg_idx_w3_64k_par.sh`; smoke
`transferibility/out/ruler_csa_idx_w3_smoke.json` + `transferibility/sg_ctrl_idx_w3_smoke/debug.log`
(only-indexer proof)._

---

## Part 5 — Composed: xKV W3 recon → TopMag50 → indexer (64k end-task)

Part 4 found W3 cross-layer low-rank on the indexer keys is **not free** at 64k (−1.1 pts). This part
composes it with the magnitude lever that *was* free (Part 4 of the Mustafar writeup, TopMag50 on the
same 128-dim keys, +0.82 pts): apply TopMag50 **after** the cross-layer reconstruction, to the
*reconstructed* keys, then let the indexer select on those.

```
K^I →(xKV W3@r192)→ K̂^I →(TopMag 50%)→ K̃^I →(indexer)→ Top-512
```

Part 4 is now a **baseline column** of this experiment's report table. Same 2-pass harness as Part 4,
with `--prune-target indexer --prune-keep 0.5` added: pass 2 reconstructs then TopMag's each
`[T,128]` reconstructed key (mask on `|R_slice|`, zero the raw `inv` coords so the store renormalizes
into `TopMag_50%(K̂^I)`). 64k, 5 hardest tasks × n=50, single tp=4 leg on 4 GPUs (container
`ruler-eval`, 2026-08-20).

**64k RULER (n=50/task, 250 smp):**

| task | native | W3-only | TopMag50 | composed | Δcomp pts |
|---|---:|---:|---:|---:|---:|
| qa_2 | 0.720 | 0.760 | 0.740 | **0.800** | −0.08 |
| qa_1 | 0.780 | 0.800 | 0.780 | 0.780 | +0.00 |
| fwe | 0.853 | 0.827 | 0.860 | 0.833 | +0.02 |
| vt | 0.992 | 0.996 | 0.992 | 1.000 | −0.01 |
| niah_multivalue | 1.000 | 1.000 | 1.000 | 1.000 | +0.00 |
| **mean** | **0.869** | **0.877** | **0.874** | **0.883** | **−0.014** |

**Analysis.**

**1. Composition does NOT compound the W3 error — it is free.** Composed mean 0.883 vs paired native
0.869 (Δ −0.014, i.e. composed 1.4 pts above its own dense on the same samples), vs the Part-4/§7
native range 0.883–0.888 within 0.5 pt. The ≤1–2 pt bar is met with room.

**2. Composed ≥ both single levers in aggregate** (0.883 vs W3-only 0.877, TopMag50 0.874). The
suggestive mechanism: SVD-truncation error lives in the reconstruction's smallest-|·| coords, and
TopMag50 zeroes exactly those — pruning the *reconstruction* partially cleans the cross-layer error
instead of adding to it. At n=50 the 0.6–0.9 pt gaps are within binomial SEM (≈4.8 pts/column), so
"not worse" is the defensible claim; the direction is consistent with the mechanism.

**3. Very mild treatment.** 11/250 samples flip under composition (+7/−4 net). R(0.5) = 0.968 on the
reconstruction ≈ native TopMag's 0.967 — the same energy fraction is kept whether the mask is applied
to native or reconstructed keys.

**4. Workload shape is preserved.** fwe still carries the small word-recall penalty (0.833, 3
down-flips — Part 4's fwe cost persists); retrieval/needle free (niah_multivalue 1.000, vt → 1.000).

**5. The 4× compute ceiling is kernel-gated.** xKV halves the dims scored (128→64) and TopMag zeroes
half the remaining coords → 4× fewer per-position indexer score FLOPs at ≤1 pt of accuracy cost,
conditional on a fused indexer kernel that skips dropped/zeroed dims.

_Notes: `native` = composed run's own pass-1 dense (capture-mode generation is native), the same
paired reference used throughout; its qa_2/qa_1 drew low vs Part 4's (0.72/0.78 vs 0.76/0.82) —
shared-gt dense matches 234/242 across runs, so the native path is stable and the difference is n=50
sample draw + fp8 nondeterminism. Artifacts: `transferibility/out/ruler_csa_idx_w3_tm50_64k.json`,
launcher `transferibility/sg_idx_w3_tm50_64k.sh`, smoke
`transferibility/out/ruler_csa_idx_w3_tm50_smoke.json` (compose proof: 504 `inject` ↔ 504
`compose_inject`, zero `prune_inject` / `dim=512`)._
