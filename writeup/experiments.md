# Experiments — V4-Flash KV-compression survey

The arc: compressibility probes (ShadowKV, xKV Frobenius, AsymKV) → end-task accuracy of
cross-layer low-rank and magnitude pruning (RULER, then the CSA indexer) → the lowrank KV-decode
serving build (fixed-basis → windowed self-fit), evaluated end-to-end on Sangfor-Bench. Each section
is one experiment, numbered sequentially; every metric is stated against the baseline used in that run.

**Headline numbers**

| # | experiment | headline metric |
|---|---|---:|
| 1 | ShadowKV-on-V4 probe | frac_95 0.47–0.66 (~4× ShadowKV's 0.16); KV traffic 0.3–3.6 % (≤64k) |
| 2 | xKV Frobenius proxy | W1→W4 +9.6–12.6 %; matched-memory W4 1.9× W1 @8k → 1.26× @64k |
| 3 | AsymKV homogeneity | ρ1 0.46–0.61 (paper 0.80); monotone ✓ @8k, ✗ niah @32k/64k |
| 4 | TopMag on CSA latent | 50 %: −0.23/−0.16 pts @32k/64k; 70 %: qa_2 −4.5 @64k |
| 5 | xKV W3 end-task RULER | 32k −0.004, 8k −0.03; latent 6.4× cut |
| 6 | xKV W3 on indexer | 64k −1.1 pts (fwe +4.0) |
| 7 | TopMag on indexer | 64k −0.82 (50 %) / −0.55 (70 %) pts; qa_2 −2.0 @70 % |
| 8 | Composed xKV+TopMag | composed 0.883 ≥ native 0.869; 4× indexer cost ceiling |
| 9 | Lowrank serving (pre-store) | req/s 0.79–0.90×; TTFT 1.64× @4k; 0 bytes saved |
| 10 | Lowrank concurrency (200 B/store) | +35–40 % C_max; KV 2.92× (584→200 B/token) |
| 11 | Fused triton recon | kernel 0.10 ms p50; tp=8 ITL 180 vs 17–19 ms native (~10–13×) |
| 12 | Fixed-basis Sangfor-Bench | 10/10 garbage (0–1 tool calls); retention 0.77–0.87 @r320 |
| 13 | Windowed self-fit (n=1) | 19/29 vs native 29/29; 528 vs 584 B/token (9.6 %) |
| 14 | Windowed self-fit (n=10) | terminated: 0/10 done @70 min, 3/3 agents 0 tool calls |
| 15 | TopMag 50 % on native c4 (n=1) | 29/29 vs native 29/29 (windowed 19/29); 0 memory saved |
| 16 | TopMag 50 % on native c4 (n=7) | 7× 29/29 (100 %, σ=0) on the windowed-failure task |

---

## 1. ShadowKV-on-V4-Flash probe — week of July 27, 2026

**Methodology.** Bolt-on within-layer SVD + sparse KV compression, testing whether V4-Flash has
headroom to speed up decode. Headroom requires (1) retained KV still strongly low-rank
(frac_95 ≪ ~0.16, ShadowKV's measured value) and (2) KV traffic a non-trivial share of decode.
Measured on raw-K and native CSA latents, 4k–128k, plus an MLA-family follow-on on
DeepSeek-Coder-V2-Lite.

**Metrics.**

| leg | gate (to justify a speedup) | measured |
|---|---|---|
| retained KV low-rank | frac_95 ≪ ~0.16 | raw-K 0.59, CSA 0.47 (4k); SWE 32k–128k 0.63–0.66 |
| KV traffic in decode | > ~10 % of per-token traffic | 0.3–3.6 % (≤64k); 6.95 % @128k; crosses 10 % only past ~192k |

Weights 12.219 GB/token vs ≤0.46 GB KV @64k. MLA follow-on: cross-layer redundancy is real and
global — xkv_gain ≈2.3× @G8, ≈4× over all 27 layers, stable 16k–64k; a single global SVD basis beats
xKV's adjacent-group-of-4 at equal memory (budget-lo 0.509 vs 0.064 avg; budget-hi 0.554 vs 0.558 at
7.7× compression).

**Takeaways.**
- Decode is MoE-weight-bound: zeroing KV moves ≤ ~3.6 % of per-token traffic up to 64k.
- V4's retained KV is already ~16 % of a dense MQA-512 cache — the ~6× win ShadowKV would provide is
  native; what remains is near full-rank.
- Cross-layer redundancy (the MLA family) is the real target, but it needs a low-rank *store*, not
  within-layer SVD.

*Caveat:* qa_2 leg OOM'd at 16k/64k (eager fp32 transient); decode timings are single-request
naive-inference artifacts — trust the roofline.

---

## 2. xKV cross-layer on CSA — Frobenius proxy — week of August 3, 2026

**Methodology.** Proxy for cross-layer low-rank on the pre-RoPE CSA latent (`[T,512]`, 21 layers):
relative Frobenius recon error of a rank-`r` basis shared across adjacent groups of W layers vs
per-layer SVD at the same memory. SWE-bench probe n=1; RULER re-run at 8k/32k/64k with true memory
`M = Tr + Wrd`.

**Metrics.**

| leg | result |
|---|---|
| grouping gain (SWE probe, n=1) | W1→W4 +9.6–12.6 %, gain shrinking with length (CKA off-diag 0.46→0.36) |
| matched-memory codebook cost | W4@r256 = 1.9× W1@r64 @8k (9.9 vs 5.2 MB) → 1.36× @32k → 1.26× @64k; break-even `T = W·d` ≈ 8k |
| error by task @64k W1@r64 | vt 0.20 · fwe 0.23 · niah_multikey_2 0.41 · qa_2 0.55 (between-task std dominates; within-task < 0.003) |
| absolute error @ b≈64 | 0.31–0.53 |

**Takeaways.**
- Grouping wins monotonically but is overstated: matched-`r/W` ≠ matched-memory — the shared codebook
  `V ∈ [r, W·d]` costs `r·d/T`.
- Error is prompt-family-dependent, not a pooled number (task spread 0.20–0.55 @64k).
- Frobenius alone can't decide go/no-go in the aggressive b≈64 regime — needs the end-task eval (exp 5).

*Caveat:* SWE probe n=1 (illustrative); RULER 8k/32k are single-task n=5 (secondary), 64k is 4-task
n=32.

---

## 3. AsymKV sanity check on V4-Flash CSA keys — week of August 10, 2026

**Methodology.** AsymKV premise: adjacent cached keys are locally homogeneous (ρ(1) ≫ ρ(2) > ρ(4) >
ρ(8)). Measured ρ(Δ) = mean cos(C_j, C_{j+Δ}) on pre-RoPE CSA latents (21 layers, RULER tasks,
8k/32k/64k), vs the paper's Llama-2-7B-chat ρ1 ≈ 0.80.

**Metrics.**

| context | ρ1 | ρ2 | ρ4 | ρ8 | monotone? |
|---|---|---|---|---|---|
| 8k | 0.542 ± 0.13 | 0.485 | 0.494 | 0.460 | ✓ |
| 32k (niah) | 0.489 | 0.429 | **0.520** | 0.435 | ✗ ρ4 > ρ1 |
| 64k (niah) | 0.500 | 0.439 | **0.529** | 0.445 | ✗ ρ4 > ρ1 |

Per-task @8k: vt 0.572 ✓, fwe 0.608 ✓, qa_2 0.528 ✓, niah 0.461 — ρ4 (0.501) > ρ1 (✗).

**Takeaways.**
- Homogeneity is real but moderate: ρ1 ≈ 0.5, ~half the paper's 0.80 margin → less error headroom
  for adjacent-key merging.
- niah breaks the monotone premise at 32k/64k (reproducible period-4 ρ peak at all depths).
- V4 is shared-KV (K ≡ V): values are exactly as homogeneous as keys, so the paper's value-side
  compression machinery is unnecessary — the key-side margin is the only binding quantity.

*Caveat:* 64k is niah-only (other RULER 64k prompts exceed the 4-GPU KV pool); 32k capped at n=5; the
niah period-4 peak is unexplained.

---

## 4. TopMag pruning of the CSA compressed cache (latent) — week of August 17, 2026

**Methodology.** Mustafar-style magnitude pruning on the stored native CSA vector (`C^Comp ∈ ℝ^512`,
21 layers): zero the smallest-|·| coordinates in place at keep-ratio 0.5/0.7, fused store renormalizes.
RULER 32k (4 hardest tasks × n=50) and 64k (13 tasks, 850 samples), scored like exp 5.

**Metrics.**

| leg | dense | pr50 | pr70 | Δ50 pts | Δ70 pts | R(0.5) | R(0.7) |
|---|---|---|---|---|---|---|---|
| 32k — 4 tasks × n=50 | 0.933 | 0.935 | 0.927 | −0.23 | +0.60 | 0.955 | 0.850 |
| 64k — 13 tasks, 850 smp | 0.951 | 0.953 | 0.947 | −0.16 | +0.39 | 0.954 | 0.845 |

qa_2 (n=100): −3.0 pts @32k → **−4.5 pts @64k** (0.735 → 0.690).

**Takeaways.**
- 50 % sparsity: −0.23 pts @32k, −0.16 pts @64k; every retrieval/needle task stays 1.000 (vt improves
  to 1.000 at pr70).
- 70 % sparsity costs ≤0.55 pts mean except the QA family — and qa_2's penalty *grows with context*
  (3.0 → 4.5 pts), unlike every other task.
- Retained energy does not flag QA: R(0.7) is uniform 0.83–0.86 across tasks while qa_2 alone falls
  4.5 pts → R(s) > 0.90 is not an end-task safety guarantee at 70 %.
- No bytes saved yet: coords are zeroed in place but the store still writes full 512-dim vectors; the
  win materializes only with a sparse store (skip zeroed writes → ~s× cache bytes).

*Caveat:* 64k is n=50 for 9/13 tasks (on-disk data cap); qa_2 @70% is n=100 at both lengths, so its
penalty is not noise.

---

## 5. Cross-layer low-rank on CSA: RULER end-task accuracy — week of August 17, 2026

**Methodology.** `W3` = adjacent groups of 3 CSA layers share a rank-192 basis (b = 64 dims/layer),
reconstruction injected back into the KV store, vs the model's **native CSA** path (full 512-dim,
`compress_ratio=4`). RULER 8k (4 tasks × n=100) and 32k (5 hardest tasks × n=100).

**Metrics.**

| leg | native CSA | W3@b64 | Δ |
|---|---|---|---|
| 8k — 4 tasks × n=100 | 0.933 | 0.907 | −0.03 |
| 32k — 5 hardest × n=100 | 0.920 | 0.916 | −0.004 |

Latent storage 134.8 → 21.0 MB (**~6.4×**).

**Takeaways.**
- 32k is within noise: mean −0.004, every per-task Δ within ±0.01.
- 8k costs −0.03, tracking the higher short-context recon error (0.461 vs 0.378 @32k) — short context
  is codebook-dominated.
- Retrieval is the most robust: niah_multikey_2 stays 1.000 @8k (−0.02), niah_multivalue 1.000 @32k
  (0.00).
- qa_2 is baseline-limited, not compression-limited: its native-CSA ceiling is 0.740 at both lengths.

*Caveat:* the baseline column is native CSA (already `compress_ratio=4`), so Δ is the *additional*
cost of cross-layer low-rank; 64k deferred (the fp32 latent capture OOM's rank 0 at tp=4 and tp=8).

---

## 6. xKV cross-layer low-rank on the CSA indexer — week of August 17, 2026

**Methodology.** Same transfer as exp 5, retargeted from the 512-dim compressor latent to the 128-dim
indexer keys: W3@b64 (adjacent groups of 3 CSA layers share a rank-192 basis) = 2:1 indexer compression,
2-pass (capture → joint SVD → inject). Win metric is **compute** (dims/token scored), not memory.
64k, 5 hardest tasks × n=50 (250 smp).

**Metrics.**

| task | native indexer | W3@b64 indexer | Δ pts |
|---|---|---|---|
| qa_2 | 0.760 | 0.760 | 0.0 |
| qa_1 | 0.820 | 0.800 | +2.0 |
| fwe | 0.867 | 0.827 | +4.0 |
| vt | 0.992 | 0.996 | −0.4 |
| niah_multivalue | 1.000 | 1.000 | 0.0 |
| **mean** | **0.888** | **0.877** | **+1.1** |

**Takeaways.**
- The indexer is more sensitive than the latent: −1.1 pts @64k vs exp 5's −0.004 @32k — small basis
  errors move tokens across the hard top-512 selection boundary, changing the attended set wholesale.
- Workload-shaped: retrieval/needle free (niah 0.0, vt −0.4, qa_2 0.0); the cost concentrates in fwe
  (+4.0) and qa_1 (+2.0).
- The 2:1 compute win is real (128→64 dims scored) but kernel-gated — it materializes only if the fused
  indexer kernel skips the dropped dims.

*Caveat:* n=50/task → ±4.8 pts/column binomial noise (individual deltas are within it; the family
pattern is the signal); no 8k/32k indexer legs.

---

## 7. TopMag (Mustafar) sparsity on the CSA indexer — week of August 17, 2026

**Methodology.** Transfer of exp 4 to the 128-dim indexer keys: per-row keep top-k by
|RMSNorm(raw)·weight|, zero the rest, fused recompute renormalizes. 64k, 5 hardest tasks × n=50.

**Metrics.**

| task | dense | pr50 | pr70 | Δ50 pts | Δ70 pts | R(0.5) | R(0.7) |
|---|---|---|---|---|---|---|---|
| qa_2 | 0.760 | 0.740 | 0.740 | 2.0 | 2.0 | 0.965 | 0.854 |
| qa_1 | 0.810 | 0.780 | 0.800 | 3.0 | 1.0 | 0.968 | 0.859 |
| fwe | 0.853 | 0.860 | 0.853 | −0.7 | 0.0 | 0.967 | 0.860 |
| vt | 0.992 | 0.992 | 0.992 | 0.0 | 0.0 | 0.971 | 0.879 |
| niah_multivalue | 0.998 | 1.000 | 1.000 | −0.3 | −0.3 | 0.966 | 0.853 |
| **mean** | **0.883** | **0.874** | **0.877** | **0.82** | **0.55** | **0.967** | **0.861** |

**Takeaways.**
- Mean drops 0.82 pts (50 %) and 0.55 pts (70 %) @64k, both within the 2-pt bar, with R(0.5) = 0.967.
- The exp-4 qa_2 caveat does **not** recur on the indexer: qa_2 @70% is −2.0 pts (vs −4.5 on the
  512-dim latent); qa_1's sign flip (−3.0 @50 vs −1.0 @70) is n=50 noise.
- Retrieval/needle free again: vt 0.0 and niah_multivalue −0.3 at both sparsities.
- Win is compute, realized only with a sparse score kernel — zeroed coords don't change the dense
  GEMM's FLOPs as executed today.

*Caveat:* n=50/task → ±7 pt binomial noise; 5 hardest tasks only (no full-13 or short-context legs);
R(0.7)≈0.86 uniform and non-diagnostic.

---

## 8. Composed: xKV W3 cross-layer recon → TopMag50 → CSA indexer — week of August 17, 2026

**Methodology.** Stack the two indexer compute levers end-to-end: native 128-dim keys → xKV W3@r192
(reconstruction across groups of 3 CSA layers, halves dims scored 128→64) → TopMag50 (zeroes half the
remaining coords, *on the reconstruction*) → ordinary top-512 selection. Win metric: 4× per-position
indexer score cost, if the fused kernel skips dropped/zeroed dims. 64k, 5 hardest tasks × n=50.

**Metrics.**

| task | native | W3-only | TopMag50 | composed | Δcomp pts |
|---|---|---|---|---|---|
| qa_2 | 0.720 | 0.760 | 0.740 | **0.800** | −0.08 |
| qa_1 | 0.780 | 0.800 | 0.780 | 0.780 | +0.00 |
| fwe | 0.853 | 0.827 | 0.860 | 0.833 | +0.02 |
| vt | 0.992 | 0.996 | 0.992 | 1.000 | −0.01 |
| niah_multivalue | 1.000 | 1.000 | 1.000 | 1.000 | +0.00 |
| **mean** | **0.869** | **0.877** | **0.874** | **0.883** | **−0.014** |

`native` = this run's own pass-1 dense (paired, same samples); Δcomp = native − composed, positive =
drop. The exp-6/exp-7 dense baselines were 0.888/0.883 — this run's dense drew low on qa_2/qa_1
(0.72/0.78), within n=50 sampling noise.

**Takeaways.**
- Composition does **not** compound the errors: composed 0.883 vs its paired native 0.869
  (Δ = −0.014, i.e. 1.4 pts *above* native on the same samples).
- Composed ≥ both single levers in aggregate (0.883 vs W3-only 0.877, TopMag50 0.874) — the SVD
  truncation error lives in the reconstruction's smallest-|·| coords, which TopMag50 zeroes, so pruning
  the reconstruction partially **cleans** the W3 error.
- The treatment is very mild: only 11/250 samples change under composition (+7/−4);
  R(0.5) = 0.968 ≈ exp-7's native 0.967.
- Workload shape preserved: fwe (word-recall) still carries the small penalty; retrieval/needle free
  (niah 1.000, vt 1.000).

*Caveat:* n=50/task → ±4.8 pts/column (the qa_2 −0.08 "gain" is within it); the composed run's dense
column drew low (0.869 vs 0.883–0.888); no 8k/32k composed legs; the 4× compute claim is kernel-gated.

---

## 9. Lowrank serving benchmark — compressor-only, no store — week of August 24, 2026

**Methodology.** xKV W3 cross-layer low-rank (rank-192 fixed basis, single-pass projection before the
fused store) on the CSA compressor latent only; the indexer untouched. bench_serving, 64 concurrent
requests, random inputs, 128-token outputs, tp=4, real serving loop. This is the **pre-store** form:
the latent is reconstructed to full 512 dims before storing → stored bytes unchanged.

**Metrics.**

| Input | req/s (comp → nat) | TTFT p50 (ms) | ITL p50 (ms) | E2E p99 (ms) |
|---|---|---|---|---|
| 4k | 7.41 / 9.44 (**0.79×**) | 2661 / 1627 | 16.5 / 16.6 | 12451 / 9734 |
| 8k | 7.54 / 8.78 (**0.86×**) | 1963 / 2220 | 16.7 / 16.8 | 12817 / 10090 |
| 32k | 2.16 / 2.68 (**0.81×**) | 5650 / 5394 | 17.6 / 17.5 | 51617 / 40450 |
| 64k | 1.22 / 1.36 (**0.90×**) | 9336 / 9149 | 18.7 / 18.7 | 93234 / 82527 |

**Takeaways.**
- Accuracy unchanged by construction (stored key = rank-192 projection of the same latent); decode is
  neutral — ITL p50 within ±0.14 ms everywhere, compression runs entirely in prefill.
- Request throughput drops 0.79–0.90×, the gap shrinking with context (64k → 0.90×) as the per-token
  projection overhead amortizes over longer prefill.
- Short-context TTFT is worst (4k: 1.64×): a fixed projection cost over a tiny prefill.
- **No memory win**: recon to 512 dims before the store → 0 bytes saved; compression without a
  low-rank store is pure added cost.

*Caveat:* 8k TTFT is a radix-cache prefix artifact (not real); 4k–64k legs are otherwise cache-free.

---

## 10. Lowrank concurrency ceiling (200 B/token store) — week of August 24, 2026

**Methodology.** W3 xKV cross-layer low-rank on the CSA compressor latent (rank-192 fixed basis,
512→192 dims), stored as 200 B/token in the patched low-rank store (`SGLANG_OPT_LOWRANK_KV_STORE=1`).
SGLang 0.5.15 real serving loop (bench_serving, 16-token outputs, N=C concurrent, mem-frac 0.88), tp=8.
Measure: max served concurrency C_max and pool ceiling vs native (584 B/token).

**Metrics.**

| L | native C_max | lowrank C_max | gain |
|---|---|---|---|
| 32k | 146 | 197 | **+35 %** |
| 64k | 72 | 98 | **+36 %** |
| 128k | 35 | 49 | **+40 %** |

KV 584 → 200 B/token (**2.92×**); DSV4 pool ceiling 4,172,032 → 5,650,432 ctx tokens (**+35.4 %**).

**Takeaways.**
- Capacity gain = pool-ceiling gain, exactly: +35–40 % C_max from the 2.92× smaller stored KV;
  original-xkv (recon latent, still 584 B/token) measures the same ceiling as native.
- Measured C_max exceeds the theoretical (146/72/35) at every L — the scheduler queues past the pool
  ceiling instead of rejecting (all 27 legs reach `completed == N`).
- The win is memory-bound *capacity*, not decode speed; the decode-side cost is covered in exp 11.

*Caveat:* this is capacity (requests served), not throughput; the eager-torch ITL numbers that
accompanied the run predate the fused recon and are superseded by exp 11.

---

## 11. Fused Triton recon kernel — week of August 24, 2026

**Methodology.** The low-rank store must re-expand 192-dim coeffs to the 512-dim latent on read; v1 did
it eagerly in torch (gather → dequant → fp32 GEMM → bf16 copy). Build a fused on-chip kernel (gather →
per-tile ue8m0 dequant → bf16 GEMM → tail-RoPE → bf16 store, one launch); A/B vs eager at 32k; clean
batch-1 profiling to locate where decode time actually goes.

**Metrics.**

| measure | value |
|---|---|
| kernel p50 | **0.10 ms** (53,088 sync'd calls, flat across n=512–24,576; max 6.45 ms = 8 JIT-warmup outliers) |
| tp=8 serving ITL p50 | fused 180 ms vs native 17–19 ms (**~10×**); 64k 196 vs 15–16 (**~12×**); 128k 174 vs 12–14 (**~13×**) |
| fused vs eager @ tp=8 | 180 vs 185 ms — null (ordering flips per context) |
| clean batch-1 (tp=4) | native+graph 6.7 · native no-graph 147–149 · lowrank no-graph 183 ms/step |
| break-even ITL_lr | ≤23–26 ms @32k · ≤21–22 @64k · ≤17–20 @128k (vs ~180 ms today) |

**Takeaways.**
- The recon kernel is not the cost: 0.10 ms vs a ~183 ms step (~0.05 %), flat across a 50× n range —
  launch-bound, not compute- or bandwidth-bound.
- The decode slowdown is **missing cuda-graph**: ~148 ms/step is the eager no-cuda-graph forward floor
  (native-no-graph measures the same), the whole low-rank path adds only ~35 ms/step, and the kernel is
  ~0.05 % of that.
- Break-even requires graph capture → ~35–40 ms/step (~1.5–2× over break-even), plus metadata slim →
  ~15–25 ms, inside break-even at 32k/64k.
- Prefill-bound workloads break even *today*: at 128k lowrank req/s ≈ native (0.47–0.54 vs 0.51) even
  at 13× ITL — the +40 % capacity nearly cancels the decode penalty.

*Caveat:* the fused-vs-eager A/B ordering flips between batch-1 and C=8 bench (scheduler jitter
dominates the ~11 ms/layer kernel difference); validation also fixed a pre-existing ue8m0 quant bug
(fp8 overflow → NaN) that had made every earlier low-rank output garbage.

---

## 12. Fixed-basis lowrank Sangfor-Bench — week of August 24, 2026

**Methodology.** Low-rank store with a **frozen** basis fit on corpus latents
(`SGLANG_OPT_LOWRANK_KV_STORE=1`, `XKV_RECON_TRITON=1`), evaluated on Sangfor-Bench. Retention
(fraction of latent energy captured) measured on a failing task's runtime latents; native baseline
52.1 % (99/190).

**Metrics.**

| basis | rank | retention mean / min |
|---|---|---|
| fixed merged (10-task prefill + prior sessions) → decode | 320 | 0.863 / **0.769** |
| same-session prefill-fit → decode | 320 | 0.733 / **0.659** |
| early-decode-fit → late decode | 320 | 0.867 / **0.750** |
| **self-fit (fit on the latents being compressed)** | **128** | **1.000** |

Fixed-rank ceiling (merged basis → sri decode): r320 0.863/0.769 → **garbage** · r384 0.910/0.848 ·
r448 0.952/0.916 · r480 0.975/0.956 · r512 (native) 1.000.

**Takeaways.**
- **10/10 instances produce complete garbage** (babble, `end_turn`, 0–1 tool calls) vs native 52.1 %.
- The c4 latent subspace is content-dependent and drifts during decode: even a basis fit on the same
  session's own prefill doesn't span its decode latents (0.66 min retention @r320).
- Sensitivity is sharp: ~0.95 min retention is borderline-clean, 0.77–0.87 → total garbage — ~5 %
  per-layer energy loss × 21 layers corrupts output.
- The decode latents are intrinsically rank-128-spanable (self-fit = 1.0 at r128) — the information is
  there; the fixed basis just can't find it.

*Caveat:* a prior "clean A/B" for a fixed basis was overfitting (basis fit on that exact session's
latents); the viable design is windowed self-fit (exp 13).

---

## 13. Windowed self-fit (1-sample) — week of August 24, 2026

**Methodology.** Windowed self-fit low-rank store: the newest W=4096 c4 tokens stay **native** (528
B/slot, full rank); at each window boundary a rank-192 basis is SVD-fit on the window's *own* normed
latents and re-encoded in place (self-fit retention ≈ 1.0). Sangfor-Bench cc agent, single instance,
same task as the native CSA baseline run.

**Metrics.**

| | native CSA | windowed self-fit |
|---|---|---|
| run_agent tests | 29/29 (100 %) | **19/29 (65.5 %)** |
| resolved | true | false |

Memory: **528 B/token vs native 584 (1.11× = 9.6 %)**; fixed-basis xkv was 200 B/token (2.92×).

**Takeaways.**
- First low-rank eval with a coherent, task-appropriate patch: the agent read the right file, planned
  the correct fix, applied the core edit — all 9 core-logic tests pass.
- The 10 failures are exact-string gaps, not coherence: 8× a missing 2-char token (`间隔`) in one
  Chinese log message, 2× an edit that failed to apply.
- Honest memory win is small: uniform 528-B slots must hold the newest window native and re-encode in
  place, so a compressed window fills only ~196 B of its slot — the saving is slot *width* (9.6 %),
  not the compression.
- A packed two-pool design (native ring + compact ~200 B history) would restore ~2.2–2.9×, but stays
  **unvalidated** — see exp 14.

*Caveat:* n=1 — diagnostic, not a ranking.

---

## 14. Windowed self-fit (10-sample gate) — terminated — week of August 24, 2026

**Methodology.** Gate for the packed two-pool design: 10 instances sampled (seed=42) from the native
baseline's 190, Sangfor-Bench cc agent, max_workers=3, 18k-s timeout. Run **terminated after ~70 min**
with 0/10 completed.

**Metrics.**

| | 10-sample gate | 1-sample reference (exp 13) |
|---|---|---|
| done / live | 0 / 3 @ 70 min | 1 / 1 @ 32 min |
| tool calls | **0 across all 3 agents** | 4 |
| agent action onset | none — gcjs: single continuous ~7,300-est-token thinking stream, 0 resets · aiyycp: cycling ~1,700-token runs · apex: frozen at 720 lines | first Read at 67 thinking tokens |
| controlled probes (short / 13k / 26k ctx) | all emit clean `tool_use` in ≤138 tokens, terminate normally | — |

**Takeaways.**
- All 3 agents streamed pure thinking with **zero tool calls** for ~70 min — a hard regression vs the
  1-sample's immediate action at 67 tokens / 4 calls / 32 min.
- The server and compression are ruled out as direct causes: identical controlled probes (short, 13k,
  and 26k-token, the last with compression triggered) all terminate normally with clean tool calls; no
  serve/launch errors, no request cycling.
- The failure is specific to the agentic Claude-Code path under this run — **indeterminate**, no root
  cause isolated.
- Implication: windowed-build behavior under a real agentic workload is **unproven**, and the packed
  two-pool design stays gated (red, not green).

*Caveat:* no root cause; the server had ~4 h uptime and 3 concurrent workers when the stall began,
whereas the 1-sample acted on a freshly-started server at max_workers=1 — concurrency or server state
are confounders, not ruled out.

---

## 15. TopMag 50 % on the native c4 latent (n=1)

**Methodology.** Store-time magnitude pruning, native build: each c4 latent has its smallest-|·| 256
of 512 coords zeroed in place right before the stock fused store (`mustafar` package, hook injected
into `compressor_v2.py`). Memory pool (584 B/token), decode, and every other path are **stock
DeepSeek-V4** — no lowrank KV, no basis. Sangfor-Bench cc agent, single instance, same task as exp
13/14 and the native CSA baseline.

**Metrics.**

| | native CSA | windowed self-fit | TopMag 50 % (native) |
|---|---|---|---|
| run_agent tests | 29/29 (100 %) | 19/29 (65.5 %) | **29/29 (100 %)** |
| resolved | true | false | **true** |

Memory: **unchanged** (native 584 B/token) — the point is pure fidelity, not savings.

**Takeaways.**
- TopMag 50 % on the native c4 latent is **indistinguishable from native on this agentic task** —
  all 29 tests pass, including the exact-string Chinese log assertions the windowed build missed
  (8× sleep-path, one missing `间隔` token). Agent: 00:42:38, 9 coherent tool calls, complete patch.
- Direct contrast: windowed self-fit re-encodes windows through a fitted basis and drifted one string
  + dropped one edit (19/29); zeroing the smallest-|·| half of each latent concentrates loss in the
  least-important directions and preserves the kept coords bit-exactly.
- Zero bandwidth saved — this answers *"can the model tolerate 50 % per-latent pruning at
  store-time?"* (yes, on n=1), not *"how much memory can TopMag save?"*. A packed store that drops
  the zeroed coords is the untested follow-up.

*Caveat:* n=1 — diagnostic, not a ranking.

---

## 16. TopMag 50 % on the native c4 latent (n=7)

**Methodology.** Same build as exp 15 (store-time magnitude pruning on the native c4 latent,
`mustafar` package, 256 of 512 coords zeroed pre-store; memory pool, decode, and everything else stock
DeepSeek-V4). 7 independent Sangfor-Bench cc-agent runs of the **same** instance
(`gcjs_kube-log-check-recover_2cadb18b`), each a separate single-instance eval with a unique run_id
(`dsv4-topmag50-20-01..07_20260827`), 2 concurrent. The initial 20-samples-in-one-run attempt hit
`docker 409 Conflict` (harness names containers `task_id__instance_id__MMDDHHMMSS` with no uniquifier)
— rebuilt as a 10×2 wave launcher with unique run_ids.

**Metrics.**

| | native CSA | windowed self-fit | TopMag50 n=1 | **TopMag50 n=7** |
|---|---|---|---|---|
| run_agent tests | 29/29 (100 %) | 19/29 (65.5 %) | 29/29 (100 %) | **7 × 29/29 (100 %)** |
| pass_rate (each) | — | — | 100.0 | **100.0 × 7, error=None each** |
| run-to-run variance | — | — | — | **σ = 0 (7/7)** |

**Takeaways.**
- 7/7 independent runs reproduce native's full 29/29 on the exact task where windowed self-fit failed
  (19/29) — the n=1 diagnostic is upgraded to a zero-variance statement on this instance.
- Live server evidence during the run: 750 k+ `prune zeroed 256` / 384 k+ `prune_skip dim 128`,
  0 `prune_error` — the hook is clean, and the served build was confirmed latent-only
  (no `XKV_TOPMAG_TARGET`).
- **Run was truncated at n=7 by server throughput, not result quality**: decode ran ~2–4 tok/s on
  94–150 k-token agent contexts (`--fp8-gemm-backend triton` + `--disable-cuda-graph`; prefill ~8 k
  tok/s), so 2 concurrent samples took 2 h 10 m – 3 h 29 m vs the n=1's 42 min. 13 remaining samples
  would have needed ~10 h.
- Ops: `XKV_DEBUG=1` grew a 938 MB ctrl/debug.log (~380 store-path appends/s) on a host mount,
  worsening the throttled server — large-n TopMag runs should use `XKV_DEBUG=0` and revisit
  cuda-graphs / fp8 backend.

*Caveat:* all 7 runs are one task (run-to-run variance of the same instance), not 7 distinct tasks —
generalization across the 190-instance pool remains unmeasured; σ=0 is per-task, not per-task-family.
