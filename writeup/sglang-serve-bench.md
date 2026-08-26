# SGLang serving stress benchmark: xKV CSA compression (W3) vs native — DeepSeek-V4-Flash

**Date**: 2026-08-24 · **Setup**: SGLang 0.5.15.post1, real serving engine, tp=4 across 4×H20-3e
(`bench_serving` via HTTP, 64 concurrent requests, random inputs, 128-token outputs).

**Scheme under test**: xKV W3 cross-layer low-rank compression on the **CSA compressor latent only**
(the indexer is untouched). Serving form = **fixed-basis single-pass inject**: per-layer rank-192
projection `A_i = VrᵀVr` (512×512, calibrated from one 32k prompt) applied row-locally as
`R = normed @ A_i` inside `on_compress`, before the fused norm+rope+store. This is the
concurrency-safe, deployment-realistic form of the 2-pass per-request W3 scheme; no cross-request
state, no per-request SVD.

## Results — serving throughput & latency (64 concurrent, input lengths 4k–64k)

| Input len | req/s native→comp | TTFT p50 (ms) | ITL p50 (ms) | ITL p99 (ms) | E2E p99 (ms) |
|---|---|---|---|---|---|
| 4k  | 9.44 → 7.41 (**0.79×**) | 1627 → 2661 | 16.59 → 16.52 | 35.2 → 35.2 | 9734 → 12451 |
| 8k  | 8.78 → 7.54 (**0.86×**) | 2220 → 1963\* | 16.77 → 16.68 | 46.0 → 36.7 | 10090 → 12817 |
| 32k | 2.68 → 2.16 (**0.81×**) | 5394 → 5650 | 17.45 → 17.59 | 63.4 → 79.2 | 40450 → 51617 |
| 64k | 1.36 → 1.22 (**0.90×**) | 9149 → 9336 | 18.68 → 18.64 | 83.2 → 89.1 | 82527 → 93234 |

\* 8k TTFT is a radix-cache prefix-reuse artifact of the tiled dataset prompts, not a real win; the
4k–64k legs are otherwise cache-free (input throughput × duration ≈ full token count), so they are a
clean compression-vs-native comparison.

- **Decode is neutral**: ITL p50 within ±0.14 ms everywhere; the compression runs entirely in
  prefill and adds nothing to per-token decode latency (the reconstructed latent is stored in the
  same fp8 format the native store emits).
- **Request throughput drops 0.79–0.90×** (compressed/native). The gap shrinks with context length
  (64k → ~10%), consistent with a per-token prefill-side projection overhead (one extra 512×192
  matmul + RMSNorm/inverse-RMSNorm per CSA layer in `on_compress`) that is amortized over longer
  prefill.
- **Tail latency worsens slightly**: ITL p99 +0.0 / −9.3 / +15.8 / +5.9 ms; E2E p99 1.13–1.28×.
  Short-context TTFT is worst (4k: 1.64×) because the fixed projection cost is not amortized over a
  tiny prefill.

## Memory (honest caveat)

In this serving form the CSA latent is reconstructed to **full 512 dims before the fused store**, so
the stored KV bytes are unchanged from native — **no serving memory reduction is realized**. The
6.4× latent cut of the offline scheme (from the cross-layer writeups) needs a low-rank/sparse store
kernel, which is kernel-gated and out of scope for this benchmark. This test therefore measures the
latency/overhead of the compression *path* under load, not a memory win.

## Note on methodology vs the mentor's CAKE-style comparison

Unlike CAKE (a transformers-forward monkeypatch run standalone on HF transformers), xKV runs as an
in-engine `on_compress` hook inside SGLang's real serving loop, so these are engine-realistic
numbers (tp=4, continuous batching, chunked prefill, cuda graphs, fp8_e4m3 KV).

## Pending: 128k extreme leg

128k input is running as a controlled comparison (radix cache **disabled on both sides**, because the
tiled dataset prompts share ~120k-token prefixes and native otherwise serves most prefill from cache
— up to 1.23M cached tokens/batch — while the compressed server showed `#cached-token: 0` in the same
window, contaminating any with-cache 128k ratio). Results to be appended on completion.

| 128k@64 | req/s | TTFT p50 (s) | ITL p50 (ms) | ITL p99 (ms) | E2E p99 (s) |
|---|---|---|---|---|---|
| native (no-cache) | _running_ | — | — | — | — |
| compressed (no-cache) | _queued_ | — | — | — | — |
