# Cross-layer low-rank on CSA: RULER end-task accuracy (xkv Part 3) — condensed

**Question.** Parts 1–2 of `xkv-crosslayer.md` measured the *proxy* (Frobenius recon error of the
pre-RoPE latent). Does the cross-layer low-rank compression actually move **end-task** RULER
accuracy? `W3` = adjacent groups of 3 CSA layers share a rank-192 basis (b = 64 dims/layer,
reconstruction injected back into the KV store) vs the model's **native CSA** path (full 512-dim
per-layer, `compress_ratio=4`). 8k = 4 tasks × n=100; 32k = the five hardest xKV Table-1 tasks ×
n=100; 64k deferred (memory-blocked).

## Verdict

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
  0.378 at 32k, Part 2): short context is codebook-dominated, so it needs a wider effective rank for
  the same per-token fidelity.
- **Retrieval is the most robust.** niah_multikey_2 stays 1.000 at 8k (−0.02) and niah_multivalue
  1.000 at 32k (0.00) under low-rank recon — consistent with Part 2's "structure is strongly
  prompt-family-dependent."
- **qa_2 is baseline-limited, not compression-limited.** Its native-CSA ceiling is 0.740 at *both*
  8k and 32k; the W3 deltas (−0.03/−0.01) are small relative to that. qa_2 headroom comes from the
  base model, not from KV fidelity.

*Caveats/notes:* the baseline column is the model's native CSA path (already `compress_ratio=4`), so
W3 measures the *additional* cost of cross-layer low-rank on top of native CSA — not full
uncompressed-KV accuracy. **64k is deferred**: the capture pass materializes ~2.8 GB of fp32 latents
on rank 0 and neither tp=8 (~0.5 GiB free in the 0.95-mem pool) nor tp=4 (weights + min pool fill the
card) leaves room; the long-context result is the 32k leg.

Full writeup (Part 1 SWE probe, Part 2 exact-baseline recon + memory model): [xkv-crosslayer.md](xkv-crosslayer.md)
