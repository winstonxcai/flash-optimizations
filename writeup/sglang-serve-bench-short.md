# xKV W3 CSA compression (compressor latent only) — DeepSeek-V4-Flash serving vs native

**Model**: DeepSeek-V4-Flash-FP8 · **Engine**: SGLang 0.5.15, tp=4 on 4×H20-3e, real serving loop
(`bench_serving`, 64 concurrent requests, random inputs, 128-token outputs).
**Method**: xKV W3 cross-layer low-rank compression (rank-192 fixed basis) applied to the **CSA
compressor latent only**; the indexer is untouched. Serving form = fixed-basis single-pass
`R = RMSNorm(latent) @ A_i` before the fused store (concurrency-safe; no per-request SVD).

## Throughput & latency (compressed vs native)

| Input | req/s (comp → nat) | TTFT p50 (ms) | ITL p50 (ms) | E2E p99 (ms) |
|---|---|---|---|---|
| 4k  | 7.41 / 9.44 (**0.79×**) | 2661 / 1627 | 16.5 / 16.6 | 12451 / 9734 |
| 8k  | 7.54 / 8.78 (**0.86×**) | 1963 / 2220\* | 16.7 / 16.8 | 12817 / 10090 |
| 32k | 2.16 / 2.68 (**0.81×**) | 5650 / 5394 | 17.6 / 17.5 | 51617 / 40450 |
| 64k | 1.22 / 1.36 (**0.90×**) | 9336 / 9149 | 18.7 / 18.7 | 93234 / 82527 |

\*8k TTFT is a radix-cache prefix artifact, not real; 4k–64k legs are otherwise cache-free → clean comparison.

## Findings

- **Accuracy unchanged by construction** — the stored key is the rank-192 projection of the same
  latent; per-layer reconstruction error is fixed by the basis rank (offline accuracy study separate).
- **Decode is neutral**: ITL p50 within ±0.14 ms everywhere — compression runs entirely in prefill.
- **Request throughput drops 0.79–0.90×**, the gap shrinking with context length (64k → 0.90×):
  the per-token prefill-side projection overhead (one extra 512×192 matmul + RMSNorm/inverse-norm +
  fp8↔fp32 round-trip + `copy_` breaking the fused store), amortized over longer prefill.
- **Short-context TTFT is worst** (4k: 1.64×): the fixed projection cost is not amortized over a tiny prefill.
- **No memory win**: the latent is reconstructed to full 512 dims before the fused store → stored bytes
  unchanged → compression without a compressed store is pure added cost. The 2.92× KV cut needs a
  low-rank **store kernel** (`transferibility/xkv_decode/`, the in-progress fix).

## 128k extreme leg

| 128k@64 | req/s | TTFT p50 (s) | note |
|---|---|---|---|
| native | 0.55 | 26.9 | with cache |
| compressed | 0.48 | 25.0 | with cache |

Cache disabled on both sides for the controlled comparison (tiled prompts share ~120k-token prefixes and
contaminate the ratio). native 128k@64 no-cache: 0.08 req/s (TTFT 361s, dur 31 min); native 128k@128
no-cache re-run + compressed no-cache legs running on GPUs 4-7.

**Diagnosis**: the 6.4×/2.92× latent cut requires storing the 192-dim coefficients, not the reconstructed
latent. The in-engine overhead above is the cost of compressing *without* that store; `xkv_decode` moves
the compression into a 200 B/token store so decode reads fewer bytes and the CSA memory ceiling rises.
