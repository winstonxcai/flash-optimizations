# TopMag pruning of the CSA compressed cache (Mustafar-style) — condensed

**Question.** Can Mustafar-style magnitude pruning transfer to V4-Flash's native CSA compressed cache
(`C^Comp ∈ ℝ^512`, 21 CSA layers, Shared-KV) — zero the smallest-|·| coordinates of each stored
compressed vector in place, keep ratio `s`, let the fused store renormalize — and does end-task
RULER accuracy survive at 50% and 70% sparsity? Measured at **32k (4 hardest tasks × n=50)** and
**64k (all 13 RULER tasks, 850 samples/config)**, same scoring as `xkv-crosslayer.md` Part 3.

## Verdict

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
  with retrieval-style prompts compressing far better than QA (`xkv-crosslayer.md` Part 2 §3).
- **Retained energy does not flag QA.** R(0.7) is uniform across tasks (0.83–0.86 @64k) while qa_2
  alone falls 4.5 pts — the R(s) > 0.90 bar is not a sufficient end-task safety guarantee at 70%.
  Adopting 70% should be workload-conditioned (safe for retrieval/needle, risky for QA-family) or
  capped per-task.
- **No bytes saved yet.** Coordinates are zeroed in place but the store still writes full 512-dim
  vectors, so bytes saved are not measured here; the win materializes only with a sparse store
  (skip zeroed-coordinate writes → ~s× compressed-cache bytes). This go/no-go sets the accuracy
  ceiling; sparse-store bytes are the deployment step.
- **Orthogonal to the cross-layer SVD** (Parts 1–3 of `xkv-crosslayer.md`) — low-rank and TopMag
  could be composed.

*Caveats:* 64k is n=50 for 9 of 13 tasks (RULER on-disk data cap), and qa_1 @70% is n=50 — but
qa_2 @70% is n=100 at both lengths, so its penalty is not noise; no 8k leg; a 1–2 pt mean drop is
borderline vs SEM on the harder tasks (e.g. qa_2 ~0.74).

Full writeup (tables per task, per-layer R_l(s), reproduce): [mustafar-sparse.md](mustafar-sparse.md)
