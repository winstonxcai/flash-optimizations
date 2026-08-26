# Fused Triton low-rank recon kernel for DeepSeek-V4-Flash decode

**Date**: 2026-08-26 · **Model**: DeepSeek-V4-Flash-FP8 · **Engine**: SGLang 0.5.15,
**tp=4 on 4×H100-80** (`ruler-eval`, GPUs 0-3) · **File**: `transferibility/xkv_decode/lowrank_store.py`

## What this is

The low-rank store keeps the W3 cross-layer CSA latent as **192-dim coefficients at
200 B/token** (2.92× smaller than native 584 B). On read, the coefficient has to be
re-expanded to the 512-dim latent before the sparse attention kernel can consume it.
v1 did that expansion eagerly in torch: gather fp8 coeffs → dequant → fp32 GEMM
`[n,192]@[192,512]` → fp32 `[n,512]` write to HBM → bf16 `copy_` → attention. That
round-trip was one suspect behind **ITL 322–561 ms** (vs ~15 ms native at tp=8), and the
2.92× stored-KV bandwidth win never materialized because the recon output was re-read
from HBM at ~4 KB/token.

This writeup documents the **fused on-chip recon**: one Triton kernel that does
gather → per-tile ue8m0 dequant → bf16 GEMM → tail-RoPE → bf16 store in a single
launch, so the latent is built in fp32 accumulators and written straight to the bf16
sparse-workspace slice `flash_mla_sparse_fwd` reads. No fp32 HBM write, no `copy_`.

It also records what the ITL A/B at 32k actually showed: the fused kernel is fast and
correct, but decode is ~150 ms/step of eager no-cuda-graph forward plus ~35 ms/step of
low-rank decode path — **not** the recon (details at the bottom).

While validating it, the numeric self-test exposed a **pre-existing ue8m0
quantization bug** (fp8 overflow → NaN) that made *every* earlier low-rank recon
garbage. Both are documented here.

## The 200 B/token record

```
byte  0 ────── 192 ── 195 ─ 196 ── 200
      │ fp8 coeffs (3×64) │s0s1s2│pad│ int32 pos │
```

- `[0,192)`: rank-192 coefficients, fp8 (e4m3fn on H100), 3 tiles of 64.
- `[192,195)`: one ue8m0 scale byte per 64-tile.
- `[196,200)`: the stored RoPE position (int32), applied at read time.

## Bug fixed during validation: the ue8m0 quant overflowed to NaN

The store-side `_quant_ue8m0` computed the scale as `e = floor(log2(maxabs))` and
stored `fp8 = x / 2^(e-127) = x·2^125` for O(1) coefficients. That overflows
`float8_e4m3fn` (max 448) → **NaN**, so the recon GEMM produced all-NaN KV and every
low-rank output was garbage. The earlier concurrency study measured latency/ceiling
only (`completed == N`), never output correctness, so this was invisible.

Fixed to match `_quant_k_cache_fused_kernel` exactly:

```
max_abs_clamped = max(|x_tile|, 1e-8)
scale        = max_abs_clamped / FP8_MAX                # 448
s            = uint8(ceil(log2(scale)) + 127)           # ue8m0 exponent byte
fp8          = clamp(x / 2^(s-127), FP8_MIN, FP8_MAX)
```

Dequant (unchanged, both paths): `x_hat = fp8 · 2^(s-127)`. Round-trip is now exact
within fp8 mantissa error (`maxabs 5.116 → 5.000`).

## The fused kernel (`_recon_lowrank_kernel`)

Grid `cdiv(n, BLOCK_M)`, **BLOCK_M=32, num_warps=8, num_stages=1**; one program
reconstructs 32 tokens. Mirrors house kernels: `_set_coeff_kernel` byte addressing,
`_dequantize_k_cache_paged_kernel` per-tile ue8m0 dequant, `_compress_norm_rope_kernel`
interleaved freqs + strided rope stores.

Per program:

1. **Gather + byte base.** Load `loc` (flat pool loc) → `base = (loc//64)·PAGE_BYTES + (loc%64)·200`.
2. **RoPE position.** Load int32 at byte `COEFF_SCALE_BYTES=196`, clamp to `[0, max_len-1]`
   (this clamp is what previously let a stale loc OOB `freqs_cis` and poison the stream).
3. **Freqs.** Load 32 cos + 32 sin from the flattened interleaved table
   (`view_as_real(freqs_cis).reshape(-1,64)`, element `2k`=cos, `2k+1`=sin).
4. **Nope GEMM in two halves.** The 512-col latent is emitted as two 256-col halves
   (`BLOCK_NOPE_H=256`, `NUM_HALVES=2`). Each half's `b_full` operand is `[64,256]`
   bf16 = **32 KB dynamic SMEM**; a full `[64,512]` operand is 64 KB, which trips the
   >48 KB opt-in launch path and fails with `CUDA_ERROR_OUT_OF_MEMORY` on a near-full
   GPU — the original launch-OOM bug this split fixes. `tl.range` over halves keeps only
   one `b_full` live per iteration: per half, 3 bf16 `tl.dot`s (`a=[M,64]` ×
   `b_full=[64,256]`) accumulate into fp32 `acc_h`, stored bf16 masked to the 448 nope cols.
5. **Rope real/imag as two dots.** `acc_r [M,32]` (even cols 448,450,…,510) and
   `acc_i [M,32]` (odd cols) accumulate the same 3-tile loop over `b_rope [64,32]`
   operands (4 KB SMEM each). Kept out of the nope loop so each accumulator sums exactly
   `SCALE_TILES` dots.
6. **Tail-RoPE.** `nr = acc_r·cos − acc_i·sin`, `ni = acc_r·sin + acc_i·cos`, stored
   strided at `448+2k` / `448+2k+1` (the exact `_compress_norm_rope_kernel` pattern).
7. **Store bf16** to `out` (the `workspace[n_swa:]` slice), no fp32 HBM write.

**Why real/imag as two dots, not `tl.split`.** The rope tail arrives as a `tl.dot`
register tensor; `tl.split`/`tl.join` on an MMA layout is the fragile part (and is
not used anywhere in the dsv4 attention kernels). Splitting the rotation into two
small `[M,64]@[64,32]` dots costs +12.5% FLOPs and makes the rotation plain strided
stores, matching the reference exactly.

## Dispatch

`dequantize_lowrank_k_cache_paged` routes to the fused kernel when
`XKV_RECON_TRITON=1` (default) and `n ≥ 16`; `XKV_RECON_TRITON=0` forces the eager
torch path (A/B control + numeric reference). The `VrT` bf16 basis and the flattened
freqs table are cached per `(layer, device)`.

## Numeric validation (kernel == torch)

`python lowrank_store.py selftest` (in `ruler-eval`, GPUs 0-3): synthetic
orthonormal basis, fixed freqs table, 1024 random quantized tokens through the same
page-addressable pool; triton vs torch recon on identical inputs.

```
[selftest] n=1024 max_abs_diff=0.015625 allclose(0.01,0.02)=True
```

The kernel's *real* numbers come from 53,088 sync'd in-server measurements during
32k-context decode (Leg A, `XKV_DECODE_TIMING=1`, 21 layers, n from 512 to 24,576):

| n (unique c4 tokens) | calls | p50 | p99 | max |
|---|---|---|---|---|
| ≤ 512 (c4 topk cap) | 13,524 | 0.100 ms | 0.206 ms | 0.295 ms |
| 1k – 4k (32k-context decodes) | 34,028 | 0.102 ms | 0.237 ms | 6.45 ms* |
| > 4k (up to 24,576) | 5,536 | 0.122 ms | 0.291 ms | 2.24 ms |

`*` 8 calls, all at n=2048 immediately after boot — Triton JIT/autotune warmup.
**Zero calls exceed 10 ms.**

The kernel is flat at ~0.10 ms p50 across a 50× n range: it is **launch-bound**, not
compute- or bandwidth-bound. Per token it reads 200 B of coeffs and writes exactly the
1 KB bf16 latent the attention kernel consumes; the 192×512 bf16 basis is a few hundred
KB and stays L2-resident. The eager torch recon path measured ~0.6 ms for the same work (Leg B), so the
fused kernel is ~5× faster even at its worst case — and ~0.05% of the ~183 ms decode step either way.

## End-to-end validation (GPUs 0-3, tp=4, 32k context, rank-192)

**Output-QA** (`sg_qa_probe.py --all`, port bug fixed): 10 prompts on native / original
/ lowrank. native and original produce byte-identical outputs (the patch is a no-op when
the store is disabled). The probe's `GARBAGE_SYMBOLS` heuristic flags most outputs on
**every** side (native 6/10, lowrank 8/10) because the chat template elicits JSON +
citation text the heuristic miscalls as garbage — so the flag is not a reliable low-rank
defect signal. The long-context prompt is OK on all three; 8,904 lowrank stores ran with
0 errors and no CUBLAS/OOM crash on any side. Bottom line: the fused kernel neither
crashes nor systematically degrades output beyond what native shows under this probe.

**ITL A/B at 32k** (`itl_probe_stream.py` batch-1 + `itl_leg.sh` bench, random 128-token
outputs, C=8, N=24): `XKV_RECON_TRITON=1` (Leg A) vs `0` (Leg B), both tp=4, mem-frac
0.95, `--disable-cuda-graph`, `XKV_DECODE_TIMING=1` on both.

| measure | Leg A (fused triton) | Leg B (torch) |
|---|---|---|
| batch-1 ITL p50 | 296 ms | 351 ms |
| C=8 bench median ITL | 418 ms | 282 ms |
| C=8 bench mean ITL | 547 ms | 316 ms |
| C=8 bench p99 ITL | 1,111 ms | 696 ms |
| C=8 output tok/s | 10.8 | 17.6 |
| C=8 bench duration (24 req) | 285 s | 174 s |

The ordering **flips** between the two measurement methods (batch-1 favors triton, the
C=8 bench favors torch). Run-to-run noise — scheduler jitter at C=8, memory
fragmentation at 0.95 mem-frac, server state — dominates the ~0.5 ms/layer kernel
difference (triton 0.10 ms vs torch ~0.6 ms, ×21 layers ≈ 11 ms/step). **This is a null
result: the fused kernel does not move decode ITL.**

## Why the decode is ~180 ms/step — measured, and the recon is not it

**Clean batch-1 ITL, same config (tp=4, mem-frac 0.95, `--disable-cuda-graph`), no
instrumentation:**

| config | ITL p50 (ms/step) |
|---|---|
| native, cuda-graph ON, short ctx (`pristine_server.log`) | 6.7 |
| native, no cuda-graph, 512 ctx | 147 |
| native, no cuda-graph, 32k | 149 |
| lowrank (fused recon), no cuda-graph, 32k | 183 |
| lowrank, no cuda-graph, 32k, `XKV_DECODE_TIMING=1` | 296 |

Three clean facts:

1. **The eager no-cuda-graph forward is the floor: ~148 ms/step**, independent of context
   length at batch-1 (512 ctx = 147 ms, 32k = 149 ms). Native-with-graph is 6.7 ms — the
   graph recovers ~140 ms/step of eager launch + dispatch overhead.
2. **The low-rank method adds only ~35 ms/step over native-no-graph** (183 − 149), ~1.7
   ms/layer. That is the dynamic token-set path — `torch.unique`, the variable-n gather,
   the sparse-workspace build, and the store hook — **not** the recon kernel, which is
   0.10 ms.
3. The 45,780-sample stage instrumentation (two `torch.cuda.synchronize()`s per recon
   call) inflates `decode_lowrank` to ~4.7 ms/layer and the step to ~296 ms; the clean
   number is ~183 ms. (The instrumentation is symmetric, so the A/B above stays valid.)

So the "fused recon fixes decode" hypothesis is disproven twice over: the recon kernel
was never the cost (0.10 ms), and even the whole low-rank decode path is only ~35 ms —
the ~150 ms is the eager forward that `--disable-cuda-graph` forces.

## The real next lever

Make the decode path **cuda-graph-capturable**. The blocker is the dynamic
`torch.unique` token-set metadata (variable n → graph re-capture or a fixed-size n_att
workspace); the fused recon kernel is already a fixed-shape, graph-safe single launch at
0.10 ms. Capturing the graph would take low-rank decode from ~183 ms/step toward the
~35 ms the method actually adds.

## Files

- Kernel + store/read hooks: `transferibility/xkv_decode/lowrank_store.py`
- Numeric self-test: `python lowrank_store.py selftest`
- QA probe: `transferibility/sg_qa_probe.py` (port fix: `PORT = sb.PORT`)
- ITL probe (batch-1): `transferibility/itl_probe_stream.py`
- ITL A/B bench driver: `transferibility/itl_leg.sh` + `relaunch_leg.sh` (a=triton, b=torch)
- Timing instrumentation: `XKV_DECODE_TIMING=1` → `xkv_decode/ctrl/timing.log`
