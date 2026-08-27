# Lowrank KV-decode: fixed basis fails Sangfor-Bench (10/10 garbage)

Date: 2026-08-27 · DeepSeek-V4-Flash (21 c4 latent layers, compress_ratio=4)
Build: `SGLANG_OPT_LOWRANK_KV_STORE=1` + `XKV_RECON_TRITON=1` (fp8 coeffs + per-64-tile scale + RoPE pos)

## Result
Sangfor-Bench cc agent, 10 sampled instances → **10/10 complete garbage** (babble text,
`end_turn`, 0–1 tool calls). Direct A/B reproduces it. Baseline native DeepSeek-V4-Flash-Local
= 52.1% (99/190).

## Why it failed
The c4 latent subspace is **content-dependent AND drifts during decode**. A fixed/precomputed
basis fit on one content (long-QA, 10-task prefill, even the same task's own prefill) does not
span an arbitrary session's decode latents. This is the fatal deviation from real
[xKV](https://arxiv.org/abs/2503.18893): xKV fits an **online per-sequence SVD** on the
current sequence's own KV; our `lowrank_store.py` uses a frozen calibration basis.

## Metrics — retention (fraction of latent energy captured) @ rank 320
on a failing task's runtime latents:

| basis | rank | retention mean / min |
|---|---|---|
| fixed merged (10-task prefill + prior sessions) → decode | 320 | 0.863 / **0.769** |
| same-session prefill-fit → decode | 320 | 0.733 / **0.659** |
| early-decode-fit → late decode | 320 | 0.867 / **0.750** |
| **self-fit (fit on the latents being compressed)** | **128** | **1.000** |

Sensitivity: ~0.95 min retention is borderline-clean (one session at 0.95–0.98 was clean); a
task at 0.77–0.87 min → total garbage. Even ~5% per-layer energy loss × 21 layers corrupts output.

## Fixed-rank ceiling (merged basis → sri decode)
No fixed basis reaches the ~0.99 needed for clean output at any compression-relevant rank:

| rank | B/token | compression | retention mean / min |
|---|---|---|---|
| 320 | 332 | 1.76x | 0.863 / 0.769 → **garbage** (confirmed) |
| 384 | 396 | 1.47x | 0.910 / 0.848 |
| 448 | 460 | 1.27x | 0.952 / 0.916 |
| 480 | 492 | 1.19x | 0.975 / 0.956 |
| 512 | 584 | 1.00x (native) | 1.000 |

The decode latents ARE intrinsically rank-128-spanable (self-fit = 1.0) — the information is
there; the fixed basis just can't find it. Prior "clean A/B" was overfitting (basis fit on that
exact session's latents).

## Implication
Fixed-basis lowrank KV-decode cannot solve long agentic tasks by construction. The viable
design is **windowed self-fit**: keep the most recent W tokens native, periodically fit a
per-window basis on them, compact older tokens under that basis (self-fit → ~1.0 retention).
Requires multi-basis index + mixed-layout pool (in progress: minimal W=4096 trial).

## Artifacts
- `transferibility/FINDING_fixed_basis_garbage.md` (full evidence trail)
- `transferibility/{fixed_rank_ceiling,early_late_test,prefill_decode_general}.py`
- latents: `out/longdecode/{prefill,decode}/X_*.pt`, `out/basis_merged_320/A_*.pt`
