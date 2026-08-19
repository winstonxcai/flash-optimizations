# Experiments — V4-Flash KV-compression survey

The arc: **ShadowKV** (bolt-on within-layer SVD) → no headroom on V4; **xKV** (cross-layer low-rank) →
proxy says grouping wins but needs end-task proof; **AsymKV** premise (adjacent-key homogeneity) →
mostly holds, barely; **Mustafar** (magnitude pruning of the compressed cache) → strong go;
**xKV RULER end-task** → free at 32k (latent); **xKV on the CSA indexer** → not free at 64k (−1.1 pts).

**Headline numbers:**

| experiment | verdict | headline metric |
|---|---|---:|
| ShadowKV (within-layer SVD) | ❌ no headroom | frac_95 0.47–0.66 (~4× ShadowKV's 0.16); KV traffic 0.3–3.6 % (≤64k) |
| xKV cross-layer Frobenius | grouping wins, overstated | W1→W4 gain 9.6–12.6 %; err 0.31–0.53 @b≈64; codebook gap 1.9× @8k |
| AsymKV adjacent-key homogeneity | mostly yes, barely | ρ1 ≈ 0.46–0.61 vs paper's 0.80; niah ρ4 > ρ1 (✗ monotone) |
| Mustafar TopMag pruning | STRONG GO | 50%: −0.2 pts; 70%: +0.4–0.6 pts except QA (qa_2 −4.5 @64k) |
| xKV cross-layer end-task | free @32k | −0.004 @32k, −0.03 @8k; 6.4× CSA-latent cut (134.8→21.0 MB) |
| xKV cross-layer on CSA indexer | ⚠ not free @64k | −1.1 pts macro (5 hardest × n=50); retrieval/needle free, fwe +4.0, qa_1 +2.0; 2:1 indexer compute cut is kernel-gated |
| TopMag (Mustafar) on CSA indexer | STRONG GO | 5 hardest @64k × n=50: Δ50 0.82 pt, Δ70 0.55 pt; qa_2 @70% −2.0 (vs −4.5 latent) |

---

## 1. ShadowKV-on-V4-Flash probe

**Question.** Would a ShadowKV-style bolt-on SVD + sparse KV compression make V4-Flash faster than
stock? Headroom exists only if (1) the *retained* KV is still strongly low-rank **and** (2) KV traffic
is a non-trivial share of decode.

### Verdict: NO meaningful speedup

| leg | gate | measured (executable) | pass |
|---|---|---:|---|
| retained KV low-rank | frac_95 ≪ ~0.16 | raw-K **0.59**, CSA **0.47** (4k); SWE 32k–128k: raw-K 0.64–0.66, CSA 0.63 | ❌ ~4× above |
| KV traffic in decode | > ~10 % | **0.3–3.6 %** (≤64k); 6.95 % @128k; crosses 10 % only past ~192k | ❌ |

- Decode is **MoE-weight-bound**: 12.219 GB weights/token vs ≤ 0.46 GB KV @64k — zeroing KV moves
  ≤ ~3.6 % of per-token traffic up to 64k.
- V4's retained KV is already **~16 % of a dense MQA-512 cache** — the ~6× memory win ShadowKV would
  provide is native; what remains is near full-rank.
- **MLA follow-on** (DeepSeek-Coder-V2-Lite proxy for the MLA+DSA family): the within-layer latent is
  *also* not extra-low-rank (frac_95 0.64–0.67 — MLA's `kv_lora_rank` already spent it), but
  **cross-layer redundancy is real and GLOBAL** — xkv_gain ≈ 2.3× @G8, ≈4× over all 27 layers, stable
  16k–64k (strided ≥ adjacent ⇒ one shared basis, not xKV's per-group ones). Realized accuracy: a
  single global SVD basis beats xKV's adjacent group-of-4 at equal memory — budget-lo **0.509 vs
  0.064** avg, budget-hi ≈ dense (0.554 vs 0.558) at 7.7× compression.

*Caveats:* qa_2 pass has a real spectrum only at 4k (16k/64k OOM'd — eager fp32 indexer transient;
the chunked SWE pass runs 32k/64k/128k clean); decode timings are a single-request naive-inference
artifact — trust the roofline; the served `swe_bench_arena` benchmark is still blocked (no image / no
reachable endpoint).

---

## 2. xKV cross-layer on CSA — Frobenius proxy

**Question.** Does a rank-`r` basis shared across adjacent CSA layers beat per-layer SVD at the same
memory? Proxy = relative Frobenius recon error of the pre-RoPE latent (`[T,512]`, 21 CSA layers).

### Verdict: grouping wins monotonically — but the proxy overstates it

- **SWE-bench probe** (n=1, matched `b = r/W`): W4 < W3 < W2 < W1 at every context/budget, W1→W4 gain
  9.6–12.6 %. But the gain *shrinks* with length (12.6→9.6 %), tracking the CKA off-diagonal decline
  (0.46→0.36) — the opposite of xKV's usual "better at long context." No strong block structure → a
  fixed adjacent W=3–4 window is the default.
- **RULER re-run** (exact baseline, true memory `M_w = Tr + Wrd`): grouping still wins everywhere,
  but matched-`r/W` ≠ matched-memory — the shared codebook `V ∈ [r, W·d]` costs `r·d/T`, so at 8k
  W4@r256 is **1.9× the memory of W1@r64** (9.9 vs 5.2 MB). Gap closes with T (1.90→1.36→1.26× at
  8k/32k/64k); break-even `T = W·d` ≈ 8k for W4.
- **Error is prompt-family-dependent, not a pooled number:** at 64k W1@r64 error by task — vt 0.20,
  fwe 0.23, niah_multikey_2 0.41, qa_2 0.55 (between-task std dominates; within-task < 0.003).
- **Absolute error stays high** in the aggressive b≈64 regime (0.31–0.53) → Frobenius alone cannot
  decide go/no-go; needs the downstream end-task eval.

*Caveats:* the SWE probe is n=1 (illustrative); the RULER re-run 8k/32k are single-task n=5
(secondary), 64k is 4-task n=32; error uses overlapping windows, memory tiled groups.

---

## 3. AsymKV sanity check on V4-Flash CSA keys

**Question.** AsymKV assumes *adjacent* cached keys are locally homogeneous (cos-sim ρ(1) ≫ ρ(2) > ρ(4) > ρ(8)). Does that survive V4-Flash's native CSA compression, which is what AsymKV-on-CSA would actually merge over? Measured as ρ(Δ) = mean cos(C_j, C_{j+Δ}) on the pre-RoPE CSA latent (21 layers, RULER tasks, 8k/32k/64k).

### Verdict

**Mostly yes, but barely — and niah breaks it.**

| context | ρ1 | ρ2 | ρ4 | ρ8 | monotone? |
|---|---|---|---|---|---|
| 8k | 0.542 ± 0.13 | 0.485 | 0.494 | 0.460 | ✓ (ρ4≈ρ2) |
| 32k (niah) | 0.489 | 0.429 | **0.520** | 0.435 | ✗ ρ4 > ρ1 |
| 64k (niah) | 0.500 | 0.439 | **0.529** | 0.445 | ✗ ρ4 > ρ1 |

- Per-task @8k: vt 0.572 ✓, fwe 0.608 ✓, qa_2 0.528 ✓, **niah 0.461 — ρ4 (0.501) > ρ1 (✗)**. Reproducible period-4 peak at all depths.
- Homogeneity is real but *moderate*: ρ1 ≈ 0.5, far from the >0.9 the premise wants. ρ1>ρ2 and clean decay hold for 3/4 tasks; niah's ρ4-peak breaks the monotone assumption.

### vs. the paper (Llama-2-7B-chat, arXiv:2506.05410)

| | paper | this work |
|---|---|---|
| adjacent-key cos | μ ≈ **0.80** | **0.46–0.61** (task-dependent) |
| keys vs values | ~0.8 vs ~0 (heterogeneous values) | **K ≡ V** — V4 is shared-KV, so values are exactly as homogeneous as keys |

**Two implications:** (1) CSA keys are locally homogeneous at roughly **half the paper's margin** — adjacent-key merging has less error headroom. (2) The paper's "heterogeneous values" half of the premise **doesn't exist in V4** — the value-side lossless-compression machinery is unnecessary; a plain merge of the shared K=V inherits the key-side homogeneity. The key-side margin is the only binding quantity.

*Caveats:* 64k is niah-only (other RULER 64k prompts exceed 4-GPU KV pool); 32k capped at n=5; measured pre-RoPE (only 64/512 dims rotated, so ≈ same); niah's ρ4-peak mechanism is unexplained (candidate: period-4 structure in the m=4 compression gate).

---

## 4. TopMag pruning of the CSA compressed cache (Mustafar-style)

**Question.** Can Mustafar-style magnitude pruning transfer to V4-Flash's native CSA compressed cache
(`C^Comp ∈ ℝ^512`, 21 CSA layers, Shared-KV) — zero the smallest-|·| coordinates of each stored
compressed vector in place, keep ratio `s`, let the fused store renormalize — and does end-task
RULER accuracy survive at 50% and 70% sparsity? Measured at **32k (4 hardest tasks × n=50)** and
**64k (all 13 RULER tasks, 850 samples/config)**, same scoring as the end-task RULER study (Section 5).

### Verdict

**STRONG GO at both 32k and 64k** — 50% sparsity is free, 70% is nearly free in aggregate, with one
consistent exception: **the QA family degrades at 70% and the penalty grows with context**
(qa_2: +3.0 pts @32k → +4.5 pts @64k, n=100 both).

| leg | dense | pr50 | pr70 | d50 pts | d70 pts | R(0.5) | R(0.7) |
|---|---|---:|---:|---:|---:|---:|---:|
| 32k — 4 tasks × n=50 | 0.933 | **0.935** | 0.927 | **−0.23** | +0.60 | 0.955 | 0.850 |
| 64k — 13 tasks, 850 smp | 0.951 | **0.953** | 0.947 | **−0.16** | +0.39 | 0.954 | 0.845 |

Go rule satisfied at both lengths (50% mean drop ≤ 2 pts **and** R(0.5) > 0.90; 70% mean drop ≤ 2 pts);
retained energy is stable 32k→64k.

- **50% is free.** Mean −0.23 pts @32k, −0.16 pts @64k. Retrieval/needle tasks never move — every
  niah task stays 1.000 @64k at both sparsities; vt actually improves to 1.000 at pr70.
- **70% is nearly free in aggregate except the QA family.** qa_2 drops 3.0 pts @32k and **4.5 pts
  @64k** (0.735→0.690); qa_1 drops 2.0 pts @64k. The 70% QA penalty *grows with context* — consistent
  with the cross-layer studies (Sections 2 and 5) showing retrieval-style prompts compress far
  better than QA.
- **Retained energy does not flag QA.** R(0.7) is uniform across tasks (0.83–0.86 @64k) while qa_2
  alone falls 4.5 pts — the R(s) > 0.90 bar is not a sufficient end-task safety guarantee at 70%.
  Adopting 70% should be workload-conditioned (safe for retrieval/needle, risky for QA-family) or
  capped per-task.
- **No bytes saved yet.** Coordinates are zeroed in place but the store still writes full 512-dim
  vectors, so bytes saved are not measured here; the win materializes only with a sparse store
  (skip zeroed-coordinate writes → ~s× compressed-cache bytes). This go/no-go sets the accuracy
  ceiling; sparse-store bytes are the deployment step.
- **Orthogonal to the cross-layer SVD** (Sections 2 and 5) — low-rank and TopMag could be composed.

*Caveats:* 64k is n=50 for 9 of 13 tasks (RULER on-disk data cap), and qa_1 @70% is n=50 — but
qa_2 @70% is n=100 at both lengths, so its penalty is not noise; no 8k leg; a 1–2 pt mean drop is
borderline vs SEM on the harder tasks (e.g. qa_2 ~0.74).

---

## 5. Cross-layer low-rank on CSA: RULER end-task accuracy

**Question.** The Frobenius proxy study above (Section 2) measured only the reconstruction error of
the pre-RoPE latent. Does the cross-layer low-rank compression actually move **end-task** RULER
accuracy? `W3` = adjacent groups of 3 CSA layers share a rank-192 basis (b = 64 dims/layer,
reconstruction injected back into the KV store) vs the model's **native CSA** path (full 512-dim
per-layer, `compress_ratio=4`). 8k = 4 tasks × n=100; 32k = the five hardest xKV Table-1 tasks ×
n=100; 64k deferred (memory-blocked).

### Verdict

**Free at 32k, small cost at 8k** — cross-layer low-rank recovers native-CSA accuracy at long context
while cutting the CSA latent ~6.4× (21.0 vs 134.8 MB).

| leg | native CSA | W3@b64 | Δ |
|---|---:|---:|---:|
| 8k — 4 tasks × n=100 | 0.933 | 0.907 | **−0.03** |
| 32k — 5 hardest × n=100 | 0.920 | 0.916 | **−0.00** |

- **32k is free.** Macro mean −0.004 (0.920 vs 0.916); every per-task Δ within ±0.01 — sampling
  noise, not signal. The shared rank-192 basis recovers native accuracy at long context while cutting
  the CSA latent 134.8 → 21.0 MB (~6.4×).
- **8k costs a little and tracks recon error.** −0.03 (0.933 vs 0.907); the earlier vt −0.25 was a
  single-run outlier — at n=100 it's −0.06. The penalty tracks the higher 8k recon error (0.461 vs
  0.378 at 32k): short context is codebook-dominated, so it needs a wider effective rank for
  the same per-token fidelity.
- **Retrieval is the most robust.** niah_multikey_2 stays 1.000 at 8k (−0.02) and niah_multivalue
  1.000 at 32k (0.00) under low-rank recon — consistent with the proxy study's finding that
  structure is strongly prompt-family-dependent.
- **qa_2 is baseline-limited, not compression-limited.** Its native-CSA ceiling is 0.740 at *both*
  8k and 32k; the W3 deltas (−0.03/−0.01) are small relative to that. qa_2 headroom comes from the
  base model, not from KV fidelity.

*Caveats/notes:* the baseline column is the model's native CSA path (already `compress_ratio=4`), so
W3 measures the *additional* cost of cross-layer low-rank on top of native CSA — not full
uncompressed-KV accuracy. **64k is deferred**: the capture pass materializes ~2.8 GB of fp32 latents
on rank 0 and neither tp=8 (~0.5 GiB free in the 0.95-mem pool) nor tp=4 (weights + min pool fill the
card) leaves room; the long-context result is the 32k leg.

---

## 6. xKV cross-layer low-rank on the CSA indexer

**Question.** Does a rank-`r` basis shared across adjacent CSA layers preserve the CSA indexer's
*selection* (the top-512 token set → end-task RULER score) while cutting its compute? The indexer —
per-query top-512 token selection over 64 index heads, fp32 `[B, S_q, 64, S_kv/4]` score tensor — is
likely the largest non-MoE compute component in V4-Flash, so the win metric is **compute (dims/token
in the indexer)**, not memory. Same transfer as Section 5 but retargeted from the 512-dim compressor
latent to the **128-dim indexer keys**: W3@b64 (adjacent groups of 3 CSA layers share a rank-192
basis, b = 64 dims/layer) = **2:1** compression on the indexer, 2-pass (capture → joint SVD → inject).

### Verdict: NOT free at 64k — the indexer is more sensitive than the latent

**64k, 5 hardest tasks × n=50 (250 smp), two tp=4 legs on 8 GPUs:**

| task | native indexer | W3@b64 indexer | Δ pts |
|---|---:|---:|---:|
| qa_2 | 0.760 | 0.760 | 0.0 |
| qa_1 | 0.820 | 0.800 | +2.0 |
| fwe | 0.867 | 0.827 | +4.0 |
| vt | 0.992 | 0.996 | −0.4 |
| niah_multivalue | 1.000 | 1.000 | 0.0 |
| **mean** | **0.888** | **0.877** | **+1.1** |

- **Not free at 64k.** Macro mean −1.1 pts vs Section 5's latent W3 at −0.004 @32k. The indexer
  outputs a hard top-512 *selection*, so small basis errors move tokens across the selection boundary
  and change the attended set wholesale — averaged recon error (Section 2) can't see this.
- **Workload-shaped: retrieval/needle free, word-recall penalized.** niah_multivalue 0.00, vt −0.4,
  qa_2 0.0; the cost concentrates in **fwe (+4.0)** and **qa_1 (+2.0)**. Individual deltas are within
  ~1 binomial SEM at n=50 (≈4.8 pts/column); the family pattern is the signal.
- **The 2:1 compute win is real but kernel-gated** — indexer key-dims/token halve (128→64), but the
  wall-clock win only materializes if the fused indexer kernel skips the dropped dims (same caveat as
  Section 4's sparse store).
- **64k feasible here** where Section 5 deferred it: indexer capture is `[T,128]` — 4× smaller than
  the `[T,512]` latent that OOM'd rank 0. Only-indexer capture/inject proven (0 compressor events).

*Caveats: n=50/task at 64k (RULER on-disk cap + scheduling match with the sibling TopMag-indexer run);
the dense column is the model's native indexer (already `compress_ratio=4`); no 8k/32k indexer legs.
Artifacts: `transferibility/out/ruler_csa_idx_w3_64k{,_a,_b}.json`, launcher
`transferibility/sg_idx_w3_64k_par.sh`. Detailed writeup: `writeup/xkv-crosslayer.md` Part 4.*

---

## 7. TopMag (Mustafar) sparsity on the CSA indexer

*Status: **done — STRONG GO @64k** (5 hardest tasks × n=50). Full write-up: appended to
`writeup/mustafar-sparse.md`.*

**Motivation.** Same compute motive: the indexer is likely the largest non-MoE compute component in
V4-Flash. The latent TopMag study (Section 4) was a strong go (50% free, 70% near-free except QA).
Applying the same zero-smallest-|·|-in-place + renormalize to the indexer's working vectors converts
that into **skipped compute** — zeroed coordinates accumulate no score — which is the win Section 4
could not claim in bytes (the store still wrote full 512-dim vectors).

**Method.** Transfer of Section 4's harness (`run-acc --prune-keep {0.5,0.3} --prune-target indexer`),
targeting the indexer's 128-dim `kv_norm` keys: per-row keep top-k by |RMSNorm(raw)·weight|, zero the
rest, fused recompute renormalizes; measure retained energy + end-task RULER. 64k, the 5 hardest tasks
× n=50, two tp=4 legs in parallel on 8 GPUs (container `ruler-eval`, 2026-08-19).

### Verdict: STRONG GO — and the QA caveat does NOT recur on the indexer

| task | dense | pr50 | pr70 | d50 pts | d70 pts | R(0.5) | R(0.7) |
|---|---|---:|---:|---:|---:|---:|---:|
| qa_2 | 0.760 | 0.740 | 0.740 | 2.0 | 2.0 | 0.965 | 0.854 |
| qa_1 | 0.810 | 0.780 | 0.800 | 3.0 | 1.0 | 0.968 | 0.859 |
| fwe | 0.853 | 0.860 | 0.853 | −0.7 | 0.0 | 0.967 | 0.860 |
| vt | 0.992 | 0.992 | 0.992 | 0.0 | 0.0 | 0.971 | 0.879 |
| niah_multivalue | 0.998 | 1.000 | 1.000 | −0.3 | −0.3 | 0.966 | 0.853 |
| **mean** | **0.883** | **0.874** | **0.877** | **0.82** | **0.55** | **0.967** | **0.861** |

- **STRONG GO** — mean 50% drop 0.82 pts (≤2 ✓) and R(0.5) = 0.967 (>0.90 ✓); 70% drop 0.55 pts (≤2 ✓).
- **The qa_2 caveat does not recur.** On the latent, qa_2 @70% fell **4.5 pts** @64k (Section 4, n=100);
  on the indexer it's **−2.0 pts @70%**. The 128-dim indexer working vectors survive 50/70% TopMag more
  gracefully than the 512-dim latent. qa_1 is within n=50 noise (−3.0 @50 vs −1.0 @70; dense baseline
  varies 0.81–0.82 between configs).
- **Retrieval/needle is free again** — vt 0.0 and niah_multivalue −0.3 pts at both sparsities.
- **Win is compute, realized only with a sparse score kernel.** Zeroed coords don't change the dense
  GEMM's FLOPs as executed today; this run sets the accuracy ceiling, the skipped-score-work speedup
  needs a sparse-aware indexer kernel (mirror of Section 4's "no bytes saved yet").
- **Orthogonal to §6's cross-layer low-rank** — TopMag prunes within-dimension coords; xKV cuts the
  dims scored; the two multiply the effective per-position score cost.

*Caveats:* per-task n=50 → ~±7 pt binomial noise (qa_1's sign flip is that noise); 5 hardest tasks
only — no full-13 or short-context indexer legs; R(0.7)≈0.86 uniform and non-diagnostic (same
limitation as Section 4); compute claim contingent on a sparse kernel.
