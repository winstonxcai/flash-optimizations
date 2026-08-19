# Results — ShadowKV-on-V4-Flash low-rank probe

**Run:** `test/lr_probe.py --model v4 --input qa_2` (image `shadowkv:v4`, 2026-07-30)
**Input:** real RULER `qa_2` HotpotQA (seed 42), truncated per context.
**Source:** `results/lr_probe/v4_qa_2.json`, `results/lr_probe/v4_qa_2_rank.png`.
**Full analysis:** see `log.md`.

## Verdict: NO meaningful speedup from adding ShadowKV to V4-Flash

| Decision-rule leg | Threshold | Measured (executable ≤64k) | Pass |
|---|---|---|---|
| Retained KV strongly low-rank | frac_95 ≪ ~0.16 | raw-K **0.59**, CSA **0.47** | ❌ |
| KV traffic non-trivial in decode | > ~10 % | **0.3 %–3.6 %** | ❌ |

> **Confirmed on real `swe_bench_arena` traces at 32k/64k/128k** (see the SWE section below): raw-K
> `frac_95` ≈ 0.64–0.66, CSA ≈ 0.63, KV traffic ≤ 6.95 % — the verdict holds across the full length
> range on the competition's own data, not just the short `qa_2` point.

> The script's auto-printed "plausible" is a **false positive**: `has_residual_lowrank` was tripped
> by a 32-row HCA rank artifact, and `kv_traffic_nontrivial` only by the analytic-only 1M point.
> Both corrected here. (See `log.md` → "Why the script's verdict is wrong".)

## Measurement 1 — SVD headroom (context 4k, folded, ambient D=512)

| Tensor | rows | n_singular | frac_90 | frac_95 | frac_99 | stable_rank | energy@r160 |
|---|---|---|---|---|---|---|---|
| Raw K=V (pre-RoPE, 43 L) | 4096 | 512 | 0.419 | **0.594** | 0.845 | 3.40 | 0.851 |
| CSA pooled (21 L)        | 1024 | 512 | 0.332 | **0.474** | 0.728 | 4.99 | 0.885 |
| HCA pooled (20 L)        | 32   | 32 ⚠ | —     | 0.009 ⚠ | —     | 2.47 | 1.0 ⚠ |

⚠ HCA rank is capped by its 32 rows at 4k — not structural; excluded from the verdict.
**Reference:** ShadowKV exploits frac_95 ≈ **0.16** (rank-160/1024) on Llama-class keys. V4's
retained key is ~3.7× *less* low-rank → little for a bolt-on SVD to remove.
16k / 64k SVD passes **OOM'd** (eager `[heads,S,S]` transient); only 4k has real spectra.

## Measurement 2 — Roofline (decisive)

Active 284.16 M params/layer → **weight read 12.219 GB/token** (fp8). KV vs. that:

| Context | V4 KV (GB) | dense-MQA-512 (GB) | V4/dense | KV traffic frac |
|---|---|---|---|---|
| 4k   | 0.034 | 0.180  | 0.188 | **0.28 %** |
| 8k   | 0.062 | 0.361  | 0.173 | **0.51 %** |
| 16k  | 0.119 | 0.721  | 0.165 | **0.96 %** |
| 32k  | 0.232 | 1.443  | 0.161 | **1.87 %** |
| 64k  | 0.459 | 2.886  | 0.159 | **3.62 %** |
| 128k | 0.913 | 5.771  | 0.158 | **6.95 %** |
| 192k | 1.366 | 8.657  | 0.158 | **10.06 %** ← crosses 10 % |
| 256k | 1.820 | 11.543 | 0.158 | 12.96 % *(analytic)* |
| 384k | 2.727 | 17.314 | 0.157 | 18.24 % *(analytic)* |
| 512k | 3.634 | 23.085 | 0.157 | 22.92 % *(analytic)* |
| 768k | 5.448 | 34.628 | 0.157 | 30.84 % *(analytic)* |
| 1M   | 7.262 | 46.171 | 0.157 | 37.28 % *(analytic)* |

Decode is MoE-weight-bound; zeroing KV removes ≤3.6 % of traffic up to 64k. **KV traffic stays
below the ~10 % decision threshold through ~192k, then climbs: 13 % @256k, 23 % @512k, 37 % @1M.**
So the roofline leg independently kills any KV compressor only up to ~192k; past that the verdict
rests on Leg 1 (retained KV is not low-rank). V4's retained KV is a flat **~15.7 % of a dense
MQA-512 cache at every length** — the ~6× memory win ShadowKV would provide is already native, so
even where KV traffic is large the compressible residual on top is small. (Full sweep computed by
`results/lr_probe/roofline_sweep.py`, config-driven; anchor rows match the probe JSON to the digit.)

> **256k SVD attempt (2026-07-31): OOM, not measured.** A `qa_2` 256k run to put a *measured*
> Leg-1 point past the ~192k threshold OOM'd at the prefill (GPU 1: needed 4.23 GiB, 3.14 GiB free —
> the 43-layer eager prefill with layers 39-42 CPU-offloaded is at the memory ceiling at 256k, and a
> concurrent small-model probe on the node tipped it over). The 256k+ rows therefore remain
> **analytic**. Real measured Leg-1 coverage tops out at **128k** (SWE section: raw-K `frac_95`
> ≈ 0.64), and the flat 32k→128k trend is what anchors the 256k→1M extrapolation. A clean 256k
> retry on an idle node (larger per-GPU cap, no coexisting job) is the way to convert it.

**Decode timing (4k):** 7481 ms/tok, 0.13 tok/s — a **single-request naive-inference artifact**:
`device_map="auto"` runs the 43 layers sequentially across the 8 GPUs with no tensor-parallel
overlap, on offline fp8→bf16-dequant math, batch=1. (The dequantized model is **≈316 GB** — 158 B
params × 2 B — which fits on 8×80 GB, so this is *not* CPU-streaming/memory pressure; an earlier
"569 GB" figure was a unit error, 284.16 M active-params-per-**layer** misread as 284 B total.) Not
a real V4 decode speed; trust the analytic roofline. Real throughput ⇒ the `swe_bench_arena`
serving run.

## Harness sanity (Llama-3.1-8B)

Folded key spectrum recovers ShadowKV's rank ≈ **160/1024** (frac_95 ≈ 0.16) → SVD harness measures
real structure. ⚠ Exact Llama JSON lost (`--rm`, non-mounted path); only the qualitative
confirmation carries forward.

## `swe_bench_arena:0.3.2` dataset — SVD probe at 32k / 64k / 128k (real agent traces)

**Run:** `test/lr_probe.py --model v4 --input swe --oneshot_cap 8192 --prefill_chunk 512`
(2026-07-31, image `shadowkv:v4`, 8×H100). **Input:** 9600 real `swe_bench_arena:0.3.2` requests
(agent conversation + tools schema, chat-template rendered); the request whose rendered length best
matches each target is used. **Source:** `results/lr_probe/v4_swe.json`, `v4_swe_rank.png`.

**This is the decisive confirmation on real competition data** — the earlier `qa_2` run only had a
real 4k spectrum (16k/64k OOM'd). Here the chunked prefill (`oneshot_cap 8192`, `prefill_chunk 512`)
executed **32k, 64k, AND 128k without OOM**, so all three lengths have real SVD spectra.

### SVD headroom (folded, ambient D=512) — mean over layers

| Context | Tensor | rows | n_sing | frac_90 | frac_95 | frac_99 | stable_rank | E@r160 |
|---|---|---|---|---|---|---|---|---|
| **32k**  | Raw K=V (43 L) | 32686  | 512   | 0.483 | **0.658** | 0.875 | 3.92 | 0.825 |
|          | CSA pooled (21 L) | ~8171  | 512   | 0.470 | **0.630** | 0.839 | 6.35 | — |
|          | HCA pooled (20 L) | 255    | 255 ⚠ | 0.008 | 0.012 ⚠   | 0.029 | 2.03 | — |
| **64k**  | Raw K=V (43 L) | 65536  | 512   | 0.481 | **0.658** | 0.876 | 4.10 | 0.826 |
|          | CSA pooled (21 L) | ~16384 | 512   | 0.475 | **0.636** | 0.844 | 6.43 | — |
|          | HCA pooled (20 L) | 512    | 512 ⚠ | 0.009 | 0.013 ⚠   | 0.033 | 1.86 | — |
| **128k** | Raw K=V (43 L) | 131072 | 512   | 0.460 | **0.641** | 0.870 | 3.98 | 0.837 |
|          | CSA pooled (21 L) | ~32768 | 512   | 0.459 | **0.625** | 0.842 | 6.19 | — |
|          | HCA pooled (20 L) | 1024   | 512 ⚠ | 0.008 | 0.013 ⚠   | 0.033 | 1.64 | — |

⚠ HCA `frac_95` is again a **row-count artifact** (S/128 pooled rows: 255→512→1024), not structure —
excluded from the verdict, exactly as in the `qa_2` run.

**Reading.** Raw K=V `frac_95` is **0.64–0.66** and CSA is **0.63–0.64** at *every* SWE length, and
**both are flat-to-declining with context** (128k is not more compressible than 32k). ShadowKV
exploits `frac_95 ≈ 0.16`; V4's retained key is **~4× less low-rank** on real agent traces — the same
result the `qa_2` 4k pass gave (raw-K 0.594 / CSA 0.474), now confirmed across the full length range
on the competition's own data. Leg 1 of the decision rule fails decisively.

### Roofline (SWE lengths; formula validated against the JSON's 4k/64k rows)

Active 284.16 M params/layer → **12.219 GB/token** weight read (fp8), unchanged (config-driven).

| Context | V4 KV (GB) | dense-MQA-512 (GB) | V4/dense | KV traffic frac |
|---|---|---|---|---|
| 32k   | 0.232 | 1.443 | 0.161 | **1.87 %** |
| 64k   | 0.459 | 2.886 | 0.159 | **3.62 %** |
| 128k  | 0.913 | 5.771 | 0.158 | **6.95 %** |

Even at 128k, KV is **< 7 %** of per-token traffic; zeroing it entirely moves decode time by that
much at most. Decode stays MoE-weight-bound. Leg 2 fails through 128k (only past 256k does KV reach
~13 %, and that remains analytic). **Decode timing** (single-request artifact, as before): 7.3–7.7
s/tok across all three lengths — flat, confirming it's a fixed sequential-pipeline floor, not
context-scaling.

### Verdict on SWE data: **unchanged — no meaningful ShadowKV speedup**

Both legs fail on real `swe_bench_arena` traces at 32k/64k/128k: retained KV is not strongly
low-rank (raw-K/CSA `frac_95` ≈ 0.63–0.66, ~4× above ShadowKV's 0.16), and KV traffic is ≤ 7 % of
weight-bound decode. The script again auto-prints "plausible" (`headroom=true`) — the **same false
positive**: `has_residual_lowrank` tripped by the HCA row-count artifact (pulls the pooled mean to
0.329 < 0.5) and `kv_traffic_nontrivial` by the analytic-only 1M row (0.373 > 0.10). Corrected here;
see `log.md`.

## `swe_bench_arena:0.3.2` serving benchmark — BLOCKED (not run)

- Arena image **not loaded**; no `swe_bench_arena-0.3.2.tar.gz` on disk.
- No live `deepseek-v4-flash` endpoint: router `:60000` → **HTTP 503 "No models available"**
  (worker `10.72.1.171:11666` on another node, unreachable); other containers are idle shells.

Need from user: path to the tarball + a reachable `URL` serving `deepseek-v4-flash`. Ready command
in `log.md`.

## MLA family — within-layer vs. CROSS-LAYER (xKV's axis) on DeepSeek-Coder-V2-Lite

> Related: the dedicated xKV cross-layer feasibility study on V4 CSA lives in
> [`writeup/xkv-crosslayer.md`](xkv-crosslayer.md).

**Run:** `test/lr_probe.py --model mla --input qa_2` (2026-07-31, image `shadowkv:v4`, 8×H100).
**Model:** `DeepSeek-Coder-V2-Lite-Instruct` — native `DeepseekV2` MLA, 27 L, `kv_lora_rank=512`,
the small sibling of xKV's own **DeepSeek-Coder-V2** benchmark (so this doubles as a harness check
against xKV's published ~3× on that arch). **Input:** real RULER `qa_2`, 4k + 16k.
**Source:** `results/lr_probe/mla_qa_2.json`, `mla_qa_2_rank.png`.

**Why this run exists.** GLM-5.1/5.2 are `GlmMoeDsa` = **MLA + DSA**, the same latent-KV family.
The V4 verdict (both legs fail) does *not* transfer to MLA unchanged, because MLA keeps a
**token-aligned, full-length** latent — so the one axis still open on compressed models,
**cross-layer** redundancy (xKV, arXiv 2503.18893), is *measurable* here in a way it is not on V4
(V4's per-layer CSA/HCA pooling uses different strides per layer → rows aren't the same tokens →
the cross-layer stack is ill-defined). Coder-V2-Lite is the runnable MLA proxy on one node.

**Metric.** `xkv_gain = Σ_layer eff_rank_95 / joint_eff_rank_95` for a group of `G` layers,
where `joint` is the eff-rank-95 of the `[S, G·512]` feature-concatenated stack. `≈1` ⇒ no
cross-layer redundancy (nothing for xKV beyond per-layer SVD); `≈G` ⇒ layers share one subspace.
`joint_frac_95 = joint_eff_rank_95 / (G·512)` is the stack's own compression ratio. Two groupings
isolate WHERE the redundancy lives: **adjacent** = contiguous blocks (`[0..G-1],[G..2G-1],…`, probes
LOCAL redundancy) vs **strided** = evenly-spaced layers spanning the full depth (probes GLOBAL
redundancy). If adjacent ≫ strided the shared subspace is local (neighbors only); if adjacent ≈
strided it is global (any layers share). `global` = one stack of all 27 layers = the maximal bound.

### Extended sweep (2026-08-03): grouping strategy + length scaling

**xkv_gain, adjacent (ADJ) vs strided (STR), mean over groups:**

| Ctx | within `frac95` | G2 ADJ/STR | G3 ADJ/STR | G4 ADJ/STR | G8 ADJ/STR | G16 ADJ/STR | GLOBAL(27) |
|---|---|---|---|---|---|---|---|
| 4k  | 0.642 | 1.37 / 1.53 | 1.62 / 1.83 | 1.95 / 2.01 | 2.83 / 2.85 | 4.30 / 4.30 | **6.59×** (jf95 0.097) |
| 16k | 0.669 | 1.32 / 1.50 | 1.50 / 1.76 | 1.77 / 1.86 | 2.35 / 2.37 | 3.18 / 3.18 | **4.44×** (jf95 0.151) |
| 32k | 0.671 | 1.31 / 1.49 | 1.48 / 1.75 | 1.74 / 1.83 | 2.27 / 2.28 | 2.99 / 2.99 | **4.01×** (jf95 0.167) |
| 64k | 0.641 | 1.33 / 1.49 | 1.52 / 1.78 | 1.80 / 1.90 | 2.36 / 2.39 | 3.17 / 3.17 | **4.17×** (jf95 0.154) |

**Three findings (each decides part of the method design):**

1. **The redundancy is GLOBAL, not local.** Strided ≥ adjacent at *every* G, and they converge to
   identical at G8/G16. A local-redundancy world (neighbors share, distant layers don't) would give
   ADJ ≫ STR — the opposite of what we see. Under a shared-global-subspace model,
   `gain ≈ Σr/(Σr−(G−1)c)` depends only on the group's rank composition, not on *which* layers are
   picked → "ADJ≈STR, converging to identical" is its exact signature (the small-G STR edge is a
   second-order composition effect: strided pairs a low-rank early layer with a mid layer). **⇒ one
   basis spanning all layers is justified; xKV's per-adjacent-group bases are not required.**

2. **Gain grows with G to a global ceiling of ~4× (16k–64k).** All-27 joint stack needs
   `joint_frac95 ≈ 0.10–0.17` (≈1300–2300 dims at 95% energy vs 8900 summed). Monotone growth with
   G = one shared subspace plus small per-layer deltas.

3. **Length: mild decay then a stable plateau — not a short-context artifact.** G8: 2.83×(4k) →
   2.35×(16k) → 2.27×(32k) → 2.36×(64k); global 6.59→4.44→4.01→4.17. The 4k value is inflated (few
   tokens); the 16k–64k regime is stable (~2.3×@G8, ~4× global) and does **not** trend toward 1 with
   length → extrapolates to GLM's 200k–1M regime. Within-layer `frac95` stays flat ~0.64–0.67 at all
   lengths (the within-layer axis stays dead regardless of context).

### Original 2-length result (2026-07-31, adjacent only) — kept for reference

| Context | within-layer latent `frac_95` (mean) | G2 gain | G4 gain | G8 gain |
|---|---|---|---|---|
| 4k  | **0.642** (min 0.219, max 0.738) | 1.37× | 1.95× | **2.83×** |
| 16k | **0.669** (min 0.229, max 0.785) | 1.32× | 1.77× | **2.35×** |

**Two distinct findings, and they cut opposite ways from the V4 result:**

1. **Within-layer: the latent is NOT extra-low-rank — same story as V4.** The MLA latent's own
   `frac_95 ≈ 0.64–0.67` (mean over 27 layers), i.e. ShadowKV-style *single-layer* SVD would keep
   ~65 % of the 512 dims for 95 % energy — ~4× worse than ShadowKV's 0.16 target, essentially
   identical to V4's raw-K 0.64–0.66. **ShadowKV's idea #1 is already spent by MLA itself**
   (`kv_lora_rank=512` *is* the low-rank projection), so a bolt-on within-layer SVD is as dead on
   MLA as on V4. (Wide spread — min 0.22 — means a few layers are genuinely low-rank, but the mean
   is high.)

2. **Cross-layer: redundancy is real and grows with group size — xKV's axis is open.** Stacking 8
   adjacent layers needs only ~1/2.8 the dims their per-layer sum would (918 vs 2598 at 4k). The
   gain rises monotonically 1.4×→2.0×→2.8× with G, exactly the signature of a **shared cross-layer
   subspace**, and is in line with xKV's published ~3× on full DeepSeek-Coder-V2 → **harness
   validated**. This is the redundancy ShadowKV's *single-layer* SVD structurally cannot see.

**Consequence for the verdict.** The compressible structure on MLA models has **moved axes**: it is
no longer within-layer (MLA ate that), it is **cross-layer** (xKV's target, measured here at ~2.3×
at G8 and ~4× across all 27 layers in the stable 16k–64k regime). So on GLM-5.1/5.2:
- **ShadowKV** (within-layer SVD + sequence sparsity) → little to add: idea #1 is native to MLA,
  idea #2 (sparse selection) is native to DSA. Same conclusion as V4, now *measured* on the family.
- **xKV** (cross-layer aligned SVD) → a real, measured opportunity (~4× global on the Lite proxy),
  because MLA's full-length token-aligned latent is exactly the object it stacks. This is the one
  transfer with headroom, and it is **orthogonal to DSA** (DSA prunes the sequence axis; xKV the
  layer/feature axis). **New (2026-08-03): the redundancy is GLOBAL** (strided ≥ adjacent, converging
  to identical) — so the effective method is not xKV's per-group basis but a *single shared basis
  across all layers*; this directly motivates the step-3 method sketch below.

**Caveats (defensibility).** (a) Measured on Coder-V2-Lite (16B), the MLA *proxy*, not GLM-5.1
itself — validates the mechanism and the harness, not GLM's exact magnitude. (b) `qa_2` at ≤16k;
the cross-layer gain is stable 4k→16k (2.83→2.35 at G8, mild decay), so it's not a short-context
artifact, but GLM's 200K–1M regime is extrapolated. (c) `xkv_gain` uses uncentered eff-rank-95, the
same energy proxy discussed for `frac_95` — an upper bound on what a real aligned-SVD compressor
achieves, not a guaranteed compression ratio. To pin GLM exactly, the next step is the same probe on
`GLM-5.1-FP8` (H100-native FP8 + offload) or the exact-arch `DeepSeek-V3.2-Exp` (MLA **+** DSA).

---

## xKV realized RULER accuracy: adjacent group-of-4 vs. ONE global group (measured 2026-08-04)

The `joint_frac95` spectral evidence above is an *energy-95 upper bound*. This section converts it to
**realized task accuracy** by running xKV's own RULER harness (`evaluate/eval_acc.py`) on
**DeepSeek-Coder-V2-Lite-Instruct** (MLA, 27 layers, `kv_lora_rank=512`), comparing xKV's default
**adjacent group-of-4** against **one global group spanning all 27 layers**, at **equal KV memory**.

**Design.** Per-token stored basis ≈ (#groups × rank_k). 27 layers → group-4 = 7 groups (6×4 + ragged 3),
global = 1 group. Equal memory ⇒ `rank_k(global) = 7 × rank_k(group-4)`. Only the non-RoPE latent
(width 512) is compressed (`--merge_k`); the 64-dim RoPE key is untouched in both configs, so it
cancels. 4 diverse RULER tasks @ **64k**, **n=96 samples/task** (full, verified), exact
`torch.linalg.svd`. Runs isolated one-config-per-GPU on 8×H100.

### Accuracy table (avg over n=96)

| Config | groups | rank_k | stored dims/tok | niah_mk2 | vt | fwe | qa_2 | **avg** |
|---|---|---|---|---|---|---|---|---|
| Uncompressed (dense) | — | — | 13 824 | 0.292 | 0.675 | 0.889 | 0.375 | **0.558** |
| Budget-Hi · group-4 | 7 | 256 | 1 792 | 0.062 | 0.367 | 0.785 | 0.167 | **0.345** |
| Budget-Hi · **global** | 1 | 1 792 | 1 792 | 0.281 | 0.690 | 0.892 | 0.354 | **0.554** |
| Budget-Lo · group-4 | 7 | 128 | 896 | 0.000 | 0.000 | 0.174 | 0.083 | **0.064** |
| Budget-Lo · **global** | 1 | 896 | 896 | 0.250 | 0.631 | 0.872 | 0.281 | **0.509** |

### Δ(global − group-4) at equal memory — global wins every task, both budgets

| task | Hi Δ | Lo Δ |
|---|---|---|
| niah_mk2 | +0.219 | +0.250 |
| vt | +0.323 | +0.631 |
| fwe | +0.108 | +0.698 |
| qa_2 | +0.188 | +0.198 |
| **avg** | **+0.209** | **+0.444** |

**Headline.** At equal KV memory, the single global basis beats adjacent group-of-4 on **all 4 tasks
at both budgets** — never once loses. The margin **widens as the budget tightens** (avg Δ +0.209 → +0.444),
exactly the spectral prediction: global retains more energy per stored dim, so it degrades gracefully
where group-4 collapses.

- **Budget-Hi global essentially matches dense** (0.554 vs 0.558) at **7.7× KV compression** — the
  cross-layer basis is nearly lossless at this rank. Group-4 at the same memory loses 0.213 absolute.
- **Budget-Lo is the sharpest split.** Group-4 has **collapsed** (avg 0.064; niah and vt at 0.000 —
  the 7-group basis is starved at rank-128/group), while global holds **0.509** at **15.4×
  compression** — still 91% of dense. On `fwe`, group-4 0.174 vs global 0.872.

This is the accuracy confirmation of the spectral claim: **the compressible KV structure on MLA is
global, not local.** xKV's own default (adjacent group-of-4) leaves large accuracy on the table
versus pooling the whole model into one shared SVD basis at the same memory.

### One-time SVD build cost (xKV's own efficiency methodology, `bench_svd_overhead_mla.py`)

Global's advantage costs a larger prefill SVD, but it is **sub-second, one-time, and paid once at
prefill** (not per decode step). Measured on H100 @ seqlen 65 536, `torch.svd_lowrank` (xKV's
deployment path), total SVD build across all groups:

| budget | group-4 total | global total | global / group-4 |
|---|---|---|---|
| Hi(1792) | 279 ms | 839 ms | **3.0×** |
| Lo(896) | 130 ms | 366 ms | **2.8×** |

So the price of the global basis is ~3× a sub-second one-time build — negligible against multi-second
prefills and dwarfed by the accuracy gain (up to +0.70 on a task). (Note: the accuracy harness uses
exact `torch.linalg.svd`, ~11× slower than this `svd_lowrank` path, but that is an eval-harness choice,
not the deployment cost.)

**Caveats.** (a) Coder-V2-Lite (16B) is the MLA proxy, not a frontier MLA model — validates the
mechanism at 27 layers. (b) n=96 (not the paper's 500); the deltas here (+0.1 to +0.7) are far larger
than n=96 noise (~±0.10 at p=0.5), so the direction is unambiguous. (c) Depth-scaling to 60 layers
(DeepSeek-V2-236B) is the next test — theory predicts the global advantage *grows* with depth.


---

# Full analysis log

> This is the complete analysis narrative for the low-rank probe summarized above.
> Previously kept as the separate top-level `log.md`; merged here so the study is one document.


**Date:** 2026-07-30
**Question:** Would applying a ShadowKV-style low-rank + sparse KV compression to
DeepSeek-V4-Flash make it *faster than running stock V4 unchanged*? (A self-comparison,
not a transfer-accuracy study.)
**Short answer:** **No meaningful speedup.** The evidence points the other way on both legs of
the decision rule. The probe script's auto-printed verdict ("plausible") is a **false positive**;
this log explains exactly why and gives the honest reading of the numbers actually collected.

Source of every number below: `results/lr_probe/v4_qa_2.json` (+ `v4_qa_2_rank.png`), the completed
run of `test/lr_probe.py` (`--model v4 --input qa_2`, contexts 4k/16k/64k) in image
`shadowkv:v4`. Input = real RULER `qa_2` (HotpotQA, seed 42), 8 samples, each ~262k tokens,
truncated per context.

---

## The decision rule

> ShadowKV headroom on V4 exists **only if** (1) V4's *retained* KV still has strong residual
> **low-rank** structure across the sequence **AND** (2) KV traffic is a **non-trivial fraction**
> of the decode step. If either fails → "no meaningful speedup."

ShadowKV's whole mechanism is relieving **KV-memory** pressure on a *dense-KV GQA* cache
(SVD the pre-RoPE keys to rank ~160/1024, offload values, keep a sparse top-k). Both legs must
hold for that to buy anything. Below, **both fail** on the data we can actually measure.

---

## Why the script's verdict is wrong (read this first)

The script printed:

```json
"verdict": {
  "residual_pooled_frac95_mean": 0.247,
  "max_kv_traffic_fraction": 0.373,
  "has_residual_lowrank": true,
  "kv_traffic_nontrivial": true,
  "headroom": true,
  "statement": "Meaningful ShadowKV headroom on V4 is plausible."
}
```

Both booleans are artifacts:

1. **`has_residual_lowrank=true` is a row-count artifact.** The gate is
   `residual_pooled_frac95_mean < 0.5`. That mean is **0.247** only because it averages CSA and
   HCA together, and **every HCA entry has `n_singular = 32`** — at 4k the HCA cache holds just
   S/128 = 32 pooled rows, so its rank is capped by *having 32 rows*, not by low-rank structure.
   HCA mean `frac_95 = 0.0086` mechanically drags the average under 0.5. Remove that artifact and
   look at the tensors with a real number of rows:
   - **Raw pre-RoPE K=V** (the true ShadowKV analog, 4096 rows, D=512): mean `frac_95 = 0.594`.
   - **CSA pooled** (1024 rows, D=512, `n_singular=512`): mean `frac_95 = 0.474`.
   Neither is "strongly low-rank." A ShadowKV-worthy signal would be `frac_95` well under ~0.16
   (that's the rank-160/1024 ShadowKV exploits on Llama-class keys). V4 sits **3–4× higher.**

2. **`kv_traffic_nontrivial=true` only fires at a context we cannot run.** The gate is
   `max_kv_traffic_fraction > 0.10`, and the max (0.373) comes from the **1M** roofline row, which
   is *analytic only*. At every context length that actually executes on this box (≤64k), KV
   traffic is **0.3 % → 3.6 %** of per-token memory traffic. Decode is overwhelmingly
   **MoE-weight-read-bound** (12.22 GB of weights read per token vs. ≤0.46 GB of KV at 64k).

So the honest verdict flips to **no meaningful headroom**: retained KV is *not* strongly low-rank,
and KV traffic is negligible in the regime we can measure. (Action item: fix the script's gate to
(a) use raw-K / CSA only — exclude any tensor whose `n_singular < ambient_dim`, i.e. row-count
limited — and (b) evaluate KV-traffic only at executable contexts, not the analytic 1M point.)

---

## Measurement 1 — Headroom (SVD of what V4 retains), context = 4k

Folded formation (V4 has `num_key_value_heads=1`, so folded == `[S, 512]`; ambient D = 512).
Aggregates over layers:

| Tensor | rows | `n_singular` | frac_90 | frac_95 | frac_99 | stable_rank | energy@rank160 |
|---|---|---|---|---|---|---|---|
| **Raw K=V** (pre-RoPE, 43 layers) | 4096 | 512 | 0.419 | **0.594** | 0.845 | 3.40 | 0.851 |
| **CSA pooled** (21 layers) | 1024 | 512 | 0.332 | **0.474** | 0.728 | 4.99 | 0.885 |
| **HCA pooled** (20 layers) | 32 | **32** ⚠ | — | 0.009 ⚠ | — | 2.47 | 1.0 ⚠ |

⚠ HCA is rank-limited by its 32 rows at 4k; its "low rank" is not structural. Ignore it for the
verdict.

**Reading.** To retain 95 % of the energy in V4's raw pre-RoPE K=V you need on average **~304 of
512 directions** (`frac_95 = 0.594`; layer 21 = 0.623, i.e. 319 dims). ShadowKV's premise is that
you need only ~160 of 1024 (`frac ≈ 0.16`) on a Llama-class key cache. V4's retained key is
**~3.7× less low-rank in fractional terms** than the structure ShadowKV was built to exploit.
Even `energy@rank160 = 0.851` (mean) means a hard rank-160 truncation of V4's *raw* K throws away
~15 % of the energy — and V4 doesn't even *store* the raw K; it stores the already-pooled CSA/HCA
entries, which are what a real deployment would have to compress further.

This is expected from V4's architecture, and the SVD confirms it: V4 **already spent** the
low-rank headroom by design. It has **no MLA latent**, uses **shared K=V MQA** (one 512-dim head
is both K and V, broadcast to 64 query heads), and natively compresses the cache via CSA (mean-pool
stride 4 + indexer top-512) and HCA (mean-pool stride 128). What's *left* in the retained cache is
close to full-rank residue — there is little for a bolt-on SVD to remove.

Per-layer detail (raw K=V, `frac_95`): most compressible layers are the shallow/boundary ones
(layer 4 = 0.215, layer 30 = 0.426, layers 0/1/38/42 ≈ 0.44–0.49); deep layers are near
full-rank (layers 31/35/37/39/40/41 ≈ 0.72). No layer approaches the ShadowKV regime.

---

## Measurement 2 — Roofline (the decisive half)

Active params/layer (top-6 experts + shared + attn proj) = 284.16 M → **weight read = 12.219 GB
per decoded token** (fp8 = 1 B/param). Against that, V4's retained KV footprint:

| Context | V4 KV (GB) | dense-MQA-512 (GB) | V4/dense | **KV traffic fraction** |
|---|---|---|---|---|
| 4k    | 0.034 | 0.180 | 0.188 | **0.28 %** |
| 16k   | 0.119 | 0.721 | 0.165 | **0.96 %** |
| 64k   | 0.459 | 2.886 | 0.159 | **3.62 %** |
| 256k  | 1.820 | 11.543 | 0.158 | 12.96 % *(analytic)* |
| 1M    | 7.262 | 46.171 | 0.157 | 37.28 % *(analytic)* |

**Decisive line:** at every executable context (≤64k), KV is **< 4 %** of per-token memory
traffic. ShadowKV shrinks *KV*; even shrinking V4's KV to **zero** removes at most ~3.6 % of the
decode traffic at 64k. The MoE weight read (12.22 GB/token, fixed) dominates. There is no decode
speedup to be had from KV compression here — **this holds regardless of Measurement 1.**

Two secondary facts the roofline table also states:
- V4's retained KV is already only **~16 %** of a dense MQA-512 cache (`v4_vs_dense` ≈ 0.16–0.19
  across all lengths) — the compression ShadowKV would provide is *already applied natively*.
- Only past **256k** does KV even approach 13 %+ of traffic — and V4 cannot eager-prefill that far
  on this box (see OOM below), so it's analytic, not a measured operating point.

**Decode timing** (measured, 4k): `7481 ms/token`, `0.13 tok/s`. **This is not a real V4 decode
speed** — it is a **single-request naive-inference artifact**: with `device_map="auto"` the 43
layers are pipeline-sharded across the 8 GPUs and executed **sequentially with no tensor-parallel
overlap**, on offline-dequantized bf16 math, for a batch of one. That is a floor of this diagnostic
setup, not a property of V4, so the **analytic roofline, not this number, is the trustworthy latency
signal.** (Real throughput requires a proper served deployment — which is exactly what the
`swe_bench_arena` run is meant to measure; see status at the end.)

---

## What OOM'd, and why that is itself a finding

- **16k and 64k headroom passes OOM'd** (recorded as `{"error":"OOM"}`), so **only 4k produced
  real SVD data.** Cause: V4 is **eager-attention only** (`head_dim=512` exceeds FlashAttention's
  256 cap; SDPA lacks the attention-sink term), and its **indexer** builds a single fp32 score
  tensor `[B, S_q, index_n_heads=64, S_kv/4]` (`modeling_deepseek_v4.py:456`,
  `scores = matmul(q.float(), compressed_kv.float()...)`). This transient is quadratic in sequence
  × 64 heads × 4 bytes and is **not** FlashAttention-tiled. At 16k one-shot it is ~17 GB in a single
  allocation; at 64k it grows *larger* even when the query axis is chunked, because the compressed
  key axis `S_kv/4` tracks the full sequence, not the chunk (chunk=2048 → ~34 GB). Either exceeds
  the free memory left per card, so it OOMs. This is a **prefill-compute transient**, not KV-memory
  or weight-memory pressure.
- **"Cannot dense-eager-prefill V4 past ~4–16k on 8×H100 without its sparse machinery" is itself a
  roofline result:** V4's own long-context path *requires* CSA/HCA sparsity to be feasible at all.
  A ShadowKV layer would not relieve this (the fp32 indexer score tensor is a *prefill compute*
  transient, not a *KV memory* problem), so it does not create headroom here either.

To get real 16k/64k/256k SVD would need a re-run with the prefill forced through the chunked live
cache at every length (lower `oneshot_cap` below the smallest context, and drop `prefill_chunk` to
≤512 to bound the fp32 indexer transient) — but the 4k spectrum + roofline already resolve the
decision rule, so this is optional.

---

## Why decode is slow here (the 7.5 s/token artifact, corrected)

V4-Flash ships as an **fp8** checkpoint (**159.6 GB** on disk; the index's `total_size` is
159,609,485,896 bytes). Native fp8 matmul in `transformers 5.x` routes through DeepGEMM (disabled
by the cross-device guard on multi-GPU) or a Triton kernel fetched from the HF Hub at runtime — and
this host is **network-blocked**. The offline-safe path is `FineGrainedFP8Config(dequantize=True)`,
which converts fp8→bf16 at load using HF's validated blockwise math.

Measured from the safetensors headers, the checkpoint is **≈158 B params** (141.7 B of it is
int8-packed fp8 MoE experts, plus 6.0 B F8_E4M3, 8.86 B F8_E8M0 block-scales, 1.42 B bf16,
0.04 B f32). Dequantized to bf16 that is **≈316 GB** (158 B × 2 B) — **not** the 569 GB stated in an
earlier draft of this log. (That figure came from mis-reading `284.16` — which is *active params
per layer in **millions*** — as 284 **billion** total params; corrected here.)

**316 GB fits comfortably** on 8×80 GB = 640 GB, and even the conservative `V4_GPU_GIB=66` cap
(528 GB budget) exceeds it, so **size-forced CPU offload is not established** — an earlier claim
that the deepest 8 modules stream from host RAM every forward is **not supported** by these numbers
(and the run's per-layer placement was printed to stdout but never saved to JSON, so it cannot be
verified after the fact). The honest explanation for `7481 ms/token` is the **single-request
naive-inference artifact** described in Measurement 2: `device_map="auto"` runs the 43 layers
**sequentially across GPUs with no tensor-parallel overlap**, on bf16-dequant math, batch=1. That is
a diagnostic-setup floor, not a V4 property. The trustworthy latency signal is the analytic
roofline; a real throughput number requires the served `swe_bench_arena` run.

---

## Harness sanity (Llama-3.1-8B reference)

The Llama-3.1-8B baseline (task #15) ran and confirmed the harness recovers ShadowKV's headline:
the **folded** key spectrum reaches ~95 % energy at **rank ≈ 160 of 1024** (`frac_95 ≈ 0.16`),
matching ShadowKV's rank-160 result and proving the SVD harness measures real low-rank structure
(not noise). ⚠ **Caveat:** that run's JSON was written to a non-mounted in-container path under
`--rm` and was lost, so exact per-layer Llama numbers are not on disk — only the qualitative
rank-160 confirmation carries forward. The V4-vs-ShadowKV contrast above uses ShadowKV's *published*
0.16 fraction as the reference, which is the sourceable comparison. (Re-running Llama with
`--out_dir` on the host mount would restore the exact reference table.)

---

## Verdict (honest, applied to the data)

| Decision-rule leg | Gate | Measured (executable regime) | Pass? |
|---|---|---|---|
| Residual low-rank in retained KV | `frac_95` well below ~0.16 | raw-K 0.59 / CSA 0.47 | **NO** |
| KV traffic non-trivial in decode | > ~10 % of per-token traffic | 0.3 %–3.6 % (≤64k) | **NO** |

**Both legs fail. Verdict: no meaningful speedup from adding ShadowKV to V4-Flash.**
V4 already solves ShadowKV's problem natively (shared K=V MQA + CSA/HCA pooling → retained KV is
~16 % of dense and near full-rank in what remains), and decode is MoE-weight-bound, so compressing
KV — even to zero — moves per-token time by < ~4 % up to 64k. The script's "plausible" print is a
false positive driven by (a) a 32-row HCA rank artifact and (b) an analytic-only 1M context; see
the fix noted above.

---

## SWE-bench-arena dataset probe — 32k / 64k / 128k on real agent traces (2026-07-31)

The `qa_2` run above produced a real spectrum only at **4k** (16k/64k OOM'd on the one-shot eager
prefill). To confirm the verdict on the competition's own data *at long context*, the probe was
re-run against **9600 real `swe_bench_arena:0.3.2` requests** (each an agent conversation +
tools-schema, rendered through a chat-template role framing), with the prefill forced through the
**chunked live cache** at every length (`--oneshot_cap 8192 --prefill_chunk 512`) to bound the fp32
indexer transient described below. This time **32k, 64k, and 128k all completed without OOM** — so we
have real SVD spectra across the full length range, not just a single short point.

Source: `results/lr_probe/v4_swe.json` (+ `v4_swe_rank.png`). Load placement identical to the `qa_2`
run (5 layers/GPU on 8 cards, layers 39–42 + norm + heads on CPU under the `V4_GPU_GIB=66` cap — a
cap-driven offload, not size-forced; the ≈316 GB bf16 model fits the 528 GB GPU budget).

### Headroom (SVD of retained KV), mean over layers

| Context | Tensor | rows | `n_singular` | frac_90 | frac_95 | frac_99 | stable_rank | E@r160 |
|---|---|---|---|---|---|---|---|---|
| **32k**  | Raw K=V (43 L)   | 32686  | 512     | 0.483 | **0.658** | 0.875 | 3.92 | 0.825 |
|          | CSA pooled (21 L)| ~8171  | 512     | 0.470 | **0.630** | 0.839 | 6.35 | — |
|          | HCA pooled (20 L)| 255    | **255** ⚠| 0.008 | 0.012 ⚠  | 0.029 | 2.03 | — |
| **64k**  | Raw K=V (43 L)   | 65536  | 512     | 0.481 | **0.658** | 0.876 | 4.10 | 0.826 |
|          | CSA pooled (21 L)| ~16384 | 512     | 0.475 | **0.636** | 0.844 | 6.43 | — |
|          | HCA pooled (20 L)| 512    | **512** ⚠| 0.009 | 0.013 ⚠  | 0.033 | 1.86 | — |
| **128k** | Raw K=V (43 L)   | 131072 | 512     | 0.460 | **0.641** | 0.870 | 3.98 | 0.837 |
|          | CSA pooled (21 L)| ~32768 | 512     | 0.459 | **0.625** | 0.842 | 6.19 | — |
|          | HCA pooled (20 L)| 1024   | **512** ⚠| 0.008 | 0.013 ⚠  | 0.033 | 1.64 | — |

⚠ **HCA is again row-count-limited** (S/128 pooled rows: 255 → 512 → 1024). At 64k/128k its
`n_singular` saturates at the ambient 512, but its `frac_95 ≈ 0.013` still reflects *having few
distinct pooled rows relative to a huge stride*, not exploitable low-rank structure — the same
artifact called out for the `qa_2` HCA at 4k. Excluded from the verdict.

**The key finding: the spectrum is flat across 4×–128× the context.** Raw pre-RoPE K=V `frac_95`
sits at **0.658 / 0.658 / 0.641** for 32k / 64k / 128k; CSA at **0.630 / 0.636 / 0.625**. It does
**not** trend toward the ShadowKV regime as context grows — if anything it is marginally *less*
compressible at 32k than the `qa_2` 4k pass (0.594), and holds ~0.65 out to 128k. On real agent
traces, to keep 95 % of the raw-K energy you still need ~330 of 512 directions. `energy@rank160`
(0.82–0.84) means a hard rank-160 truncation — ShadowKV's operating point — discards ~16–18 % of the
energy at *every* SWE length. There is no length at which V4's retained KV becomes "ShadowKV-shaped."

Per-layer spread (raw-K `frac_95`) is wide and stable — `[0.18–0.78]` at 128k — with the shallow/
boundary layers most compressible and deep layers near full-rank, exactly as in `qa_2`. No layer
reaches ShadowKV's 0.16.

### Roofline at SWE lengths (decisive half)

The roofline is config-driven and input-independent, so the JSON re-emits the same 4k/16k/64k/256k/1M
rows as the `qa_2` run. Reproducing its formula on the host (validated: it returns the JSON's 4k =
0.28 % and 64k = 3.62 % exactly) gives the SWE-specific operating points:

| Context | V4 KV (GB) | dense-MQA-512 (GB) | V4/dense | **KV traffic fraction** |
|---|---|---|---|---|
| 32k   | 0.232 | 1.443 | 0.161 | **1.87 %** |
| 64k   | 0.459 | 2.886 | 0.159 | **3.62 %** |
| 128k  | 0.913 | 5.771 | 0.158 | **6.95 %** |

Active params/layer = 284.16 M → **12.219 GB weight read per token** (fp8), which dominates. At 128k —
the longest length the SWE traces reach — KV is **6.95 %** of per-token traffic; compressing it to
zero saves at most that. Through the entire executable regime the decode is **MoE-weight-bound**, so
Leg 2 fails just as it did for `qa_2`. (V4's retained KV is also still only ~16 % of a dense MQA-512
cache — `v4_vs_dense` = 0.158–0.161 — i.e. the compression ShadowKV would add is already native.)

**Decode timing** (measured, single-request): **7663 / 7188 / 7342 ms/token** at 32k / 64k / 128k.
The flatness across a 4× context increase is itself informative: it confirms the ~7.5 s/token figure
is a **fixed sequential-pipeline floor** (43 layers run in series across the 8 GPUs with no
tensor-parallel overlap, bf16-dequant math, batch=1), **not** a context-length or KV effect. A real
throughput number still requires the served `swe_bench_arena` run below.

### Why 128k succeeded here where 64k OOM'd for `qa_2`

The `qa_2` 16k/64k OOMs came from a **one-shot** eager prefill building the indexer's fp32
`[B, S_q, 64, S_kv/4]` score transient in a single allocation. Forcing `oneshot_cap=8192` (below the
smallest SWE context) routes *every* SWE length through the chunked live-cache path with
`prefill_chunk=512`, which caps the query axis of that transient at 512 and empties the cache between
chunks. That bounds the fp32 transient enough that even 128k fits — confirming the earlier claim that
the OOM was a **prefill-compute transient**, addressable by chunking, not a KV/weight-memory wall.
(Note this does not change the verdict: it makes V4's *own* long-context prefill feasible; it is not
something a ShadowKV layer would provide, since the transient is prefill compute, not KV memory.)

### Script verdict block (raw) and why it is again a false positive

```json
"verdict": {
  "residual_pooled_frac95_mean": 0.329,
  "max_kv_traffic_fraction": 0.373,
  "has_residual_lowrank": true,
  "kv_traffic_nontrivial": true,
  "headroom": true,
  "statement": "Meaningful ShadowKV headroom on V4 is plausible."
}
```

Identical failure mode to `qa_2`:
1. `has_residual_lowrank` (gate: pooled `frac_95` mean < 0.5) is tripped only because the **HCA
   row-count artifact** (`frac_95 ≈ 0.013`) drags the CSA+HCA average to 0.329. The honest,
   row-complete tensors — raw-K 0.65 and CSA 0.63 — are both well above 0.5.
2. `kv_traffic_nontrivial` (gate: max KV frac > 0.10) fires only on the **analytic 1M** row (0.373).
   Every executable SWE length is ≤ 6.95 %.

Corrected verdict: **both legs fail; no meaningful ShadowKV headroom on V4-Flash**, now confirmed on
9600 real competition traces across 32k/64k/128k — the strongest form of the result.



Requested as a follow-on ("once this current v4 flash has concluded"). It is a **client** that
pounds an OpenAI-compatible endpoint; it needs two things this host currently lacks:

1. **The arena image is not loaded** and no `swe_bench_arena-0.3.2.tar.gz` is on disk
   (`/home/jovyan`, repo dir, `/tmp`, `/data`, `~` all checked). `docker load -i ...` cannot run
   without the tarball.
2. **No live `deepseek-v4-flash` endpoint exists.** The only OpenAI server up is the sglang router
   on `:60000`, which returns **HTTP 503 "No models available"** — its worker
   (`http://10.72.1.171:11666`, a *different* node) is unreachable from here. The other containers
   (`sglang-0515`, `zyh`, `vllm-nightly-0706`, `swift4.1-torch210-cu130`) are idle `bash` shells,
   not serving.

To run the arena I need (from you): the **path to `swe_bench_arena-0.3.2.tar.gz`** to load, and a
**reachable `URL` serving `deepseek-v4-flash`** (either bring the router's worker online, or point
me at the host:port of a live V4 server). The exact command, ready to fill in:

```bash
docker load -i /path/to/swe_bench_arena-0.3.2.tar.gz     # once the tarball is provided
mkdir -p /home/jovyan/winstonxcai/perf-result
docker run --rm -u 0:0 --name perf-run1 --network=host \
  -e URL=http://<V4_HOST>:<PORT> \
  -e MODEL=deepseek-v4-flash \
  -e CONCURRENCY=20 \
  -v /home/jovyan/winstonxcai/perf-result:/data/output \
  swe_bench_arena:0.3.2
```

Its output (Mean Decode tok/s, TTFT, E2EL P90/P99) would be the **real** decode-throughput number
that the offline single-request probe here could not produce — and would independently confirm the
roofline conclusion that V4 decode is weight-bound, not KV-bound.
