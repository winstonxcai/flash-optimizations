# Memory-bound concurrency ceiling at 32k/64k/128k: original-xkv vs lowrank decode

**Date**: 2026-08-25 · **Model**: DeepSeek-V4-Flash-FP8 · **Engine**: SGLang 0.5.15, **tp=8 on 8×H100-80**
(`bench_serving` random inputs, 16-token outputs, N=C concurrent requests, mem-fraction 0.88).
**Question (mentor)**: *"How much concurrency can the original plan run? After using your compression,
how much concurrency can be achieved?"* — the memory-bound concurrency ceiling of the DSV4 KV pool for
three store variants of the same W3 cross-layer low-rank compression (rank-192 fixed basis on the CSA
compressor latent).

| store variant | bytes/token | c4 pool slots | pool ceiling (logical tokens) | KV reduction |
|---|---|---|---|---|
| native (unpatched) | 584 | 1,043,008 | 4,172,032 | 1.00× |
| original xkv (fixed-basis inject, reconstructed latent) | 584 | 1,043,008 | 4,172,032 | 1.00× |
| **lowrank decode (192-dim coeffs)** | **200** | **1,412,608** | **5,650,432** | **2.92× / +35.4%** |

Original-xkv reconstructs the latent to 512 dims before the fused store → same 584 B/token as native →
same ceiling. Low-rank keeps the 192-dim coeffs (200 B/token) → 1.354× more pool slots → +35.4% ceiling.

## Answer

| L | theoretical C_max (native / orig / lr) | **measured largest C fully served (native / orig / lr)** | gain |
|---|---|---|---|
| 32k | 127 / 127 / 172 | **146 / 146 / 197** | +35% |
| 64k | 63 / 63 / 86 | **72 / 72 / 98** | +36% |
| 128k | 31 / 31 / 43 | **35 / 35 / 49** | +40% |

The original plan runs **~127 concurrent 32k / ~63 at 64k / ~31 at 128k**; after low-rank compression
**~172 / ~86 / ~43** (measured 197 / 98 / 49). A **native (unpatched)** leg run under the same protocol
measures **146 / 72 / 35** — identical to original, confirming the fixed-basis store adds no capacity
headroom (both store 584 B/token). Measured exceeds theoretical because the scheduler queues past the
pool ceiling rather than rejecting (`completed == N` still holds at all legs). The gain equals the
pool-ceiling gain (+35.4%); compression buys concurrency only via the 2.92× smaller stored KV.

## All concurrency legs (completed == N)

| L | 0.80× C_max | 1.00× C_max | 1.15× C_max |
|---|---|---|---|
| 32k | nat 101 ✓ / orig 101 ✓ / lr 137 ✓ | nat 127 ✓ / orig 127 ✓ / lr 172 ✓ | nat 146 ✓ / orig 146 ✓ / lr 197 ✓ |
| 64k | nat 50 ✓ / orig 50 ✓ / lr 68 ✓ | nat 63 ✓ / orig 63 ✓ / lr 86 ✓ | nat 72 ✓ / orig 72 ✓ / lr 98 ✓ |
| 128k | nat 24 ✓ / orig 24 ✓ / lr 34 ✓ | nat 31 ✓ / orig 31 ✓ / lr 43 ✓ | nat 35 ✓ / orig 35 ✓ / lr 49 ✓ |

All 27 legs (9 per side) pass clean.

## Throughput & latency — original xkv

| L | C | req/s | in_tok/s | TTFT p50 | TTFT p99 | ITL p50 | E2E p99 | wall |
|---|---|---|---|---|---|---|---|---|
| 32k | 101 | 4.15 | 136k | 18.6s | 23.9s | 16.2ms | 24.3s | 24s |
| 32k | 127 | 13.20 | 433k | 8.9s | 9.2s | 17.0ms | 9.6s | 10s |
| 32k | 146 | 13.99 | 458k | 9.6s | 10.0s | 18.9ms | 10.4s | 10s |
| 64k | 50 | 1.24 | 81k | 31.2s | 40.0s | 14.4ms | 40.4s | 40s |
| 64k | 63 | 6.43 | 421k | 9.1s | 9.3s | 15.2ms | 9.7s | 10s |
| 64k | 72 | 6.47 | 424k | 10.4s | 10.7s | 16.2ms | 11.1s | 11s |
| 128k | 24 | 0.32 | 42k | 26.9s | 72.8s | 12.6ms | 74.2s | 74s |
| 128k | 31 | 0.37 | 48k | 49.3s | 84.5s | 13.8ms | 84.9s | 85s |
| 128k | 35 | 0.40 | 53k | 57.6s | 84.9s | 13.4ms | 87.0s | 87s |

## Throughput & latency — native (unpatched)

| L | C | req/s | in_tok/s | TTFT p50 | TTFT p99 | ITL p50 | E2E p99 | wall |
|---|---|---|---|---|---|---|---|---|
| 32k | 101 | 4.63 | 152k | 19.7s | 21.6s | 5.8ms | 21.8s | 22s |
| 32k | 127 | 12.57 | 412k | 9.2s | 9.7s | 17.1ms | 10.0s | 10s |
| 32k | 146 | 12.56 | 412k | 10.7s | 11.1s | 19.0ms | 11.5s | 12s |
| 64k | 50 | 1.44 | 94k | 18.3s | 34.4s | 14.4ms | 34.7s | 35s |
| 64k | 63 | 5.69 | 373k | 10.3s | 10.6s | 15.5ms | 11.0s | 11s |
| 64k | 72 | 6.21 | 407k | 10.9s | 11.2s | 16.1ms | 11.5s | 12s |
| 128k | 24 | 0.34 | 45k | 34.7s | 69.6s | 12.6ms | 69.9s | 70s |
| 128k | 31 | 0.42 | 55k | 30.5s | 73.2s | 12.6ms | 73.5s | 74s |
| 128k | 35 | 0.51 | 67k | 20.5s | 66.6s | 13.7ms | 68.3s | 68s |

Native and original agree within run-to-run noise on the warm (radix-cached) legs — saturated-leg ITL
matches to ±0.3 ms (17.1/19.0 vs 17.0/18.9 @32k, 15.5/16.1 vs 15.2/16.2 @64k, 12.6/13.7 vs 12.6/13.4
@128k) — validating original-xkv as a native proxy for the concurrency-ceiling conclusion. The cold first
leg of each L and the 128k legs are heavily load-sensitive and noisy (e.g. 128k C=35 TTFT 20.5s vs 57.6s
on the original leg).

## Throughput & latency — lowrank decode (fused triton recon)

**Update (2026-08-26):** re-measured at tp=8, mem-frac 0.88 (same protocol as the rows
above), **fused on-chip recon** (`XKV_RECON_TRITON=1`), clean config (XKV_DEBUG=0,
XKV_DECODE_TIMING=0), single cold run per context at the max served concurrency
(1.15× C_max). The v1 eager-torch rows (322–561 ms ITL) below are superseded — they
predate the fused kernel (the triton kernel did not exist; the eager-torch path now kept
behind `XKV_RECON_TRITON=0` was the *only* recon), ran with the pre-fix ue8m0 quant
(all-NaN KV — irrelevant to latency, garbage outputs), and with the debug `.item()` syncs
that `sg_serve_bench.py` forced via `XKV_DEBUG=1` and that we later gated. Only the
debug syncs materially inflated the ITLs; the capacity rows are unchanged.

| L | C | req/s | in_tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | E2E p99 | wall |
|---|---|---|---|---|---|---|---|---|---|
| 32k | 197 | 7.62 | 250k | 22.0s | 22.9s | **180 ms** | 8.3s | 25.7s | 26s |
| 64k | 98 | 2.30 | 151k | 32.2s | 39.6s | **196 ms** | 20.2s | 42.6s | 43s |
| 128k | 49 | 0.54 | 71k | 62.7s | 87.5s | **174 ms** | 66.8s | 90.2s | 90s |

ITL p50 is flat at ~180 ms across all three contexts (matching the tp=4 clean batch-1
183 ms): the recon kernel is 0.10 ms and the method adds ~35 ms/step over the native
no-cuda-graph floor of ~148 ms — so decode is dominated by the eager forward that
`--disable-cuda-graph` forces, and cuda-graph capture of the decode path is the next
lever (`writeup/xkv-lowrank-fused-recon.md`).

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
run-to-run noise (ordering flips per context — torch lower at 64k/128k, fused lower at 32k),
so the recon kernel is a **null at tp=8 serving concurrency**, matching the tp=4 A/B.

**Why the massive slowdown — cuda-graph, not the recon.** Native serves with cuda-graph ON
(6.7 ms/step batch-1, 13–19 ms at concurrency). The lowrank path is forced
`--disable-cuda-graph` because its dynamic `torch.unique` token-set metadata can't be
captured, so every decode step pays the **eager no-cuda-graph forward floor of ~148 ms/step**
(native-no-graph measures the same 147–149 ms, context-independent). The low-rank method adds
only ~35 ms/step (183 − 148); the recon kernel is 0.10 ms and irrelevant. So **~82% of the
~180 ms is the missing cuda-graph**, ~18% is the low-rank decode path (the dynamic
`torch.unique` metadata + store hook). The fix is cuda-graph capture of the decode path
(→ ~35–40 ms/step), not the recon.

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

## Runtime (measured wall-clock)

| leg | native | original | lowrank (eager) |
|---|---|---|---|
| server boot | ~160 s | ~95–100 s | ~93–99 s |
| 32k × 3 | ~1.5 min | ~45 s | ~1.5 min |
| 64k × 3 | ~1.6 min | ~60 s | ~2 min |
| 128k × 3 | ~4.2 min | ~4 min | ~4.5 min |
| full probe (9 legs + boot) | ~10 min | ~9 min | ~13 min |

## Output-QA (garbled-output check)

Ran on native / original / lowrank (10 prompts: 8 basic + 2 long-context). native and original outputs
are byte-identical. The probe's `GARBAGE_SYMBOLS` heuristic flags most outputs on every side (native
6/10, lowrank 8/10) because the chat template elicits JSON+citation text it miscalls as garbage, so the
flag is not a reliable low-rank defect signal; long-context is OK on all three and 8,904 lowrank stores
ran with 0 errors. See `writeup/xkv-lowrank-fused-recon.md`.

## Crash fix (lowrank decode)

The lowrank conc probe originally crashed at its first leg (C=137, 32k) with a device-side assert at the
recon GEMM; all 18 legs above are post-fix. Two address-math bugs, both in
`lowrank_store.py::decode_lowrank`, both masked until the pool is heavily populated:

1. **`-1` padding reached the recon** — empty top-k slots are `-1`-padded while `c4_sparse_topk_lengths`
   is clamped ≥1, so a zero-c4 query carried a "length-1 −1 page"; its garbage RoPE position OOB'd
   `freqs_cis`, poisoning the stream and surfacing as `CUBLAS_STATUS_INTERNAL_ERROR` at the next cuBLAS
   call. Fix: exclude negative locs from the masks, clamp the RoPE position.
2. **`c4_sparse_page_indices` are FLAT pool locs, not page indices** — the topk kernel packs
   `(page_table[block]<<page_bits) | in_page` per slot, and `c4_sparse_topk_lengths` counts tokens, not
   pages. Decode was re-expanding flat locs as pages (`loc*64+63 ≈ 78M` at C=137, c4 pool ~86% full →
   OOB). A lone 32k request passed only because its 512 locs were small/contiguous. Fix: consume them as
   flat locs (no expansion).

## Files

- Probe: `transferibility/sg_conc_cap.py` (reads `DSV4 pool sizes: full=` from boot log, brackets ceiling
  at 0.8/1.0/1.15×) · results: `transferibility/out/conc_cap/{native,original,lowrank}_L*_C*.json`
- Low-rank store kernel: `transferibility/xkv_decode/lowrank_store.py` (patched into a pristine 0.5.15
  clone, `PYTHONPATH=/sgl-workspace/sglang-lowrank/python`, env `SGLANG_OPT_LOWRANK_KV_STORE=1`)
- QA probe: `transferibility/sg_qa_probe.py`
- `DSV4 pool sizes` from boot logs: `transferibility/logs/conc_{native,original,lowrank}.log`
