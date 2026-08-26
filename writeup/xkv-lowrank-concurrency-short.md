# W3 xKV low-rank compression on DeepSeek-V4-Flash: concurrency ceiling

Applying W3 cross-layer low-rank KV compression on the DeepSeek-V4-Flash CSA compressor latent
(rank-192 fixed basis, 512-dim latent → 192-dim coeffs). SGLang 0.5.15, tp=8 on 8×H100-80, real serving
loop (`bench_serving`, random inputs, 16-token outputs, N=C concurrent requests, mem-fraction 0.88).

## Concurrency capacity (memory-bound ceiling)

| L (synthetic) | store | bytes/token | C_max (theoretical) | C_max (measured, completed == N) | gain |
|---|---|---|---|---|---|
| 32k | native (unpatched) | 584 | 127 | **146** | — |
| 32k | original xkv (fixed-basis inject, recon latent) | 584 | 127 | **146** | — |
| 32k | **lowrank decode** (192-dim coeffs) | **200** | 172 | **197** | **×1.35 (+35%)** |
| 64k | native (unpatched) | 584 | 63 | **72** | — |
| 64k | original xkv | 584 | 63 | **72** | — |
| 64k | **lowrank decode** | **200** | 86 | **98** | **+36%** |
| 128k | native (unpatched) | 584 | 31 | **35** | — |
| 128k | original xkv | 584 | 31 | **35** | — |
| 128k | **lowrank decode** | **200** | 43 | **49** | **+40%** |

- KV size: 584 → 200 B/token (**2.92×**); DSV4 pool ceiling 4,172,032 → 5,650,432 context tokens
  (**+35.4%**);
- All legs pass at 0.8 / 1.0 / 1.15× C_max: 32k nat/orig 101/127/146 · lr 137/172/197 · 64k nat/orig
  50/63/72 · lr 68/86/98 · 128k nat/orig 24/31/35 · lr 34/43/49 (all 27 legs ✓);
- Decode cost: eager low-rank recon ITL ~322–561 ms vs ~15 ms native. **Update (2026-08-26):** the fused
  on-chip recon is built, validated, and fast (0.10 ms p50, 53k in-server calls), yet a triton-vs-torch
  recon A/B at 32k is a **null result** (ITL 282–418 ms both ways) — clean batch-1 probes show the
  ~183 ms/step is ~148 ms of eager no-cuda-graph forward (native-no-graph measures the same) plus only
  ~35 ms of low-rank decode path. See `writeup/xkv-lowrank-fused-recon.md`.

The concurrency gain is exactly the pool-ceiling gain (+35.4%): compression buys capacity only via the
2.92× smaller stored KV (original-xkv stores the reconstructed latent → identical to native, 584 B/token;
a native unpatched leg measures the same 146/72/35 ceiling as original). Measured exceeds theoretical at
every L because the scheduler queues past the pool ceiling rather than rejecting (`completed == N` holds
at all 27 legs).

## Throughput & latency — original xkv

| L | C | req/s | TTFT p50 | TTFT p99 | ITL p50 | E2E p99 | wall |
|---|---|---|---|---|---|---|---|
| 32k | 101 | 4.15 | 18.6s | 23.9s | 16.2ms | 24.3s | 24s |
| 32k | 127 | 13.20 | 8.9s | 9.2s | 17.0ms | 9.6s | 10s |
| 32k | 146 | 13.99 | 9.6s | 10.0s | 18.9ms | 10.4s | 10s |
| 64k | 50 | 1.24 | 31.2s | 40.0s | 14.4ms | 40.4s | 40s |
| 64k | 63 | 6.43 | 9.1s | 9.3s | 15.2ms | 9.7s | 10s |
| 64k | 72 | 6.47 | 10.4s | 10.7s | 16.2ms | 11.1s | 11s |
| 128k | 24 | 0.32 | 26.9s | 72.8s | 12.6ms | 74.2s | 74s |
| 128k | 31 | 0.37 | 49.3s | 84.5s | 13.8ms | 84.9s | 85s |
| 128k | 35 | 0.40 | 57.6s | 84.9s | 13.4ms | 87.0s | 87s |

## Throughput & latency — native (unpatched)

| L | C | req/s | TTFT p50 | TTFT p99 | ITL p50 | E2E p99 | wall |
|---|---|---|---|---|---|---|---|
| 32k | 101 | 4.63 | 19.7s | 21.6s | 5.8ms | 21.8s | 22s |
| 32k | 127 | 12.57 | 9.2s | 9.7s | 17.1ms | 10.0s | 10s |
| 32k | 146 | 12.56 | 10.7s | 11.1s | 19.0ms | 11.5s | 12s |
| 64k | 50 | 1.44 | 18.3s | 34.4s | 14.4ms | 34.7s | 35s |
| 64k | 63 | 5.69 | 10.3s | 10.6s | 15.5ms | 11.0s | 11s |
| 64k | 72 | 6.21 | 10.9s | 11.2s | 16.1ms | 11.5s | 12s |
| 128k | 24 | 0.34 | 34.7s | 69.6s | 12.6ms | 69.9s | 70s |
| 128k | 31 | 0.42 | 30.5s | 73.2s | 12.6ms | 73.5s | 74s |
| 128k | 35 | 0.51 | 20.5s | 66.6s | 13.7ms | 68.3s | 68s |

Native matches original within run-to-run noise on the warm legs (saturated ITL 17.1/19.0 vs 17.0/18.9
@32k) and is noisier on the cold / 128k legs — validating original-xkv as the native proxy for the
ceiling conclusion.

## Throughput & latency — lowrank decode (fused triton recon)

**Update (2026-08-26):** re-measured at tp=8, mem-frac 0.88 (same protocol), **fused
on-chip recon** (`XKV_RECON_TRITON=1`), clean config (XKV_DEBUG=0), single cold run per
context at max served concurrency (1.15× C_max). The v1 eager-torch rows (322–561 ms ITL)
below are superseded — they predate the fused kernel (eager-torch, now `XKV_RECON_TRITON=0`,
was the only recon), ran with the pre-fix ue8m0 quant (all-NaN KV → garbage outputs; the
latency numbers are unaffected), and with `XKV_DEBUG=1` debug `.item()` syncs that inflated
the ITLs. Only the debug syncs materially moved the ITLs.

| L | C | req/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | wall |
|---|---|---|---|---|---|---|---|---|
| 32k | 197 | 7.62 | 22.0s | 22.9s | **180 ms** | 8.3s | 25.7s | 26s |
| 64k | 98 | 2.30 | 32.2s | 39.6s | **196 ms** | 20.2s | 42.6s | 43s |
| 128k | 49 | 0.54 | 62.7s | 87.5s | **174 ms** | 66.8s | 90.2s | 90s |

ITL p50 is flat at ~180 ms across all three contexts (matching tp=4 clean batch-1 183 ms).

### Decode ITL — native vs lowrank

tp=8, N=C, 16-token outputs (native at its own max served C = 146/72/35; lowrank at
197/98/49):

| config (ITL p50) | 32k | 64k | 128k |
|---|---|---|---|
| native CSA (cuda-graph ON) | 17–19 ms | 15–16 ms | 12–14 ms |
| **lowrank, fused triton** (clean) | **180 ms** | **196 ms** | **174 ms** |
| lowrank, eager torch (clean) | 185 ms | 183 ms | 185 ms |
| lowrank, old eager + `XKV_DEBUG=1` | 561 ms | 322 ms | 466 ms |

Slowdown vs native: **~10× / ~12× / ~13×**. The clean torch and fused rows agree to within
run-to-run noise (the ordering flips per context), so the recon kernel is a **null at tp=8
serving concurrency**

**Why the massive slowdown — cuda-graph, not the recon.** Native serves with cuda-graph ON
(6.7 ms/step batch-1, 13–19 ms at concurrency). The lowrank path is forced
`--disable-cuda-graph` because its dynamic `torch.unique` token-set metadata can't be
captured, so every decode step pays the **eager no-cuda-graph forward floor of ~148 ms/step**
(native-no-graph measures the same 147–149 ms). The low-rank method adds only ~35 ms/step
(183 − 148); the recon kernel is 0.10 ms and irrelevant. So **~82% of the ~180 ms is the
missing cuda-graph**, ~18% is the low-rank decode path. The fix is cuda-graph capture of the
decode path (→ ~35–40 ms/step), not the recon.

### Break-even ITL — when is the decode penalty worth it

Lowrank buys **+g concurrent requests** (g = 1.35/1.36/1.40 at 32k/64k/128k) at the cost of
**r× slower decode**. On a decode-bound workload (long outputs), throughput ≈ C/ITL, so
lowrank breaks even iff **ITL_lr ≤ g × ITL_nat**:

| L | g | native ITL (saturated) | **break-even ITL_lr** |
|---|---|---|---|
| 32k | +35% | 17–19 ms | **≤ 23–26 ms** |
| 64k | +36% | 15–16 ms | **≤ 21–22 ms** |
| 128k | +40% | 12–14 ms | **≤ 17–20 ms** |

At this threshold the +35–40% concurrency exactly pays for the slower decode. **Fused today
is 180 ms (~7–10× over break-even); graph capture → ~35–40 ms (~1.5–2× over); graph capture
+ metadata slim → ~15–25 ms, inside break-even at 32k/64k, borderline at 128k.**

The threshold is workload-shaped, not universal. The full condition for lowrank to win on
throughput (O = output tokens/request, TTFT = prefill-dominated E2E slice):

```
ITL_lr < ITL_nat·g + (g·TTFT_nat − TTFT_lr)/O
```

- **Short outputs / prefill-bound** (O small — the regime lowrank targets): the decode term
  is diluted by the prefill slice, so the ITL penalty can sit far above g× and still break
  even. Measured at 128k, lowrank req/s ≈ native (0.47–0.54 vs 0.51) even at 13× ITL — the
  +40% capacity nearly cancels the decode penalty *today*.
- **Long outputs / decode-bound** (O large): the condition collapses to r < g — the strict
  17–26 ms threshold above, which needs the full graph-capture + metadata-slim program.

## Note

Measured in SGLang 0.5.15 on the real serving loop (the deployment-realistic form), not a
transformers/python-forward harness. The low-rank store is a patched `DeepSeekV4LowRankPool`
(`SGLANG_OPT_LOWRANK_KV_STORE=1` in a pristine clone); the concurrency numbers above are post-fix — the
original lowrank probe crashed on its first decode batch (two address-math bugs in the recon gather:
`-1`-padded topk slots reaching the recon, and `c4_sparse_page_indices` being flat pool locs re-expanded
as pages → device-side assert at C=137).
