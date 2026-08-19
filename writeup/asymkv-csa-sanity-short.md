# AsymKV sanity check on V4-Flash CSA keys — condensed

**Question.** AsymKV assumes *adjacent* cached keys are locally homogeneous (cos-sim ρ(1) ≫ ρ(2) > ρ(4) > ρ(8)). Does that survive V4-Flash's native CSA compression, which is what AsymKV-on-CSA would actually merge over? Measured as ρ(Δ) = mean cos(C_j, C_{j+Δ}) on the pre-RoPE CSA latent (21 layers, RULER tasks, 8k/32k/64k).

## Verdict

**Mostly yes, but barely — and niah breaks it.**

| context | ρ1 | ρ2 | ρ4 | ρ8 | monotone? |
|---|---|---|---|---|---|
| 8k | 0.542 ± 0.13 | 0.485 | 0.494 | 0.460 | ✓ (ρ4≈ρ2) |
| 32k (niah) | 0.489 | 0.429 | **0.520** | 0.435 | ✗ ρ4 > ρ1 |
| 64k (niah) | 0.500 | 0.439 | **0.529** | 0.445 | ✗ ρ4 > ρ1 |

- Per-task @8k: vt 0.572 ✓, fwe 0.608 ✓, qa_2 0.528 ✓, **niah 0.461 — ρ4 (0.501) > ρ1 (✗)**. Reproducible period-4 peak at all depths.
- Homogeneity is real but *moderate*: ρ1 ≈ 0.5, far from the >0.9 the premise wants. ρ1>ρ2 and clean decay hold for 3/4 tasks; niah's ρ4-peak breaks the monotone assumption.

## vs. the paper (Llama-2-7B-chat, arXiv:2506.05410)

| | paper | this work |
|---|---|---|
| adjacent-key cos | μ ≈ **0.80** | **0.46–0.61** (task-dependent) |
| keys vs values | ~0.8 vs ~0 (heterogeneous values) | **K ≡ V** — V4 is shared-KV, so values are exactly as homogeneous as keys |

**Two implications:** (1) CSA keys are locally homogeneous at roughly **half the paper's margin** — adjacent-key merging has less error headroom. (2) The paper's "heterogeneous values" half of the premise **doesn't exist in V4** — the value-side lossless-compression machinery is unnecessary; a plain merge of the shared K=V inherits the key-side homogeneity. The key-side margin is the only binding quantity.

*Caveats:* 64k is niah-only (other RULER 64k prompts exceed 4-GPU KV pool); 32k capped at n=5; measured pre-RoPE (only 64/512 dims rotated, so ≈ same); niah's ρ4-peak mechanism is unexplained (candidate: period-4 structure in the m=4 compression gate).
