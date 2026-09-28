# STAR-CSA: low-rank compression of the CSA latent

**Status: the redundancy measurement has run on the LongSWE-Bench replay; the
logit metric is blocked and the stability ladder has not run.** Retention
numbers in *Measurement results* are measured, on a real capture, CPU-only. The
attention-logit-error table is empty because the join that produces it is known
to be broken (coverage 0.002%-0.005%) — the diagnosis is written out in that
section rather than papered over. Every remaining empty table names the command
that fills it. The serving comparison and the stability ladder are outstanding.

## Overview

DeepSeek-V4-Flash already compresses its cache: each of the 21 compressed
sparse-attention (CSA) layers stores one learned 512-dimensional latent per
4 tokens — the 584-byte C4 record. STAR-CSA compresses that latent a second
time, under the observation that its 512 dimensions are not of equal kind:

- 448 dimensions are the **NoPE** payload — learned, dense, and the part a
  low-rank projection can actually shrink;
- 64 dimensions are the **RoPE** tail — a positional rotation that a low-rank
  projection in latent space cannot represent, because rotation is not a
  change of basis the projection can absorb.

So the tail is stored **exact**, in bf16, at its native offset, and only the
448 NoPE dimensions are replaced by a rank-`r` coefficient vector `z`:

```
native C4 record   [ NoPE: 448 fp8 ][ scales: 7 u8 ][ pad: 1 ][ RoPE: 64 bf16 ]
                   = 448 + 8 + 128 = 584 B per token

STAR-CSA record    [ z: r fp8 ][ scales: ceil(r/64) u8 ][ RoPE: 64 bf16 ][ pad to 4 ]
```

with `x̂ = D_l z` for a per-layer basis `D_l ∈ R^{448×r}`. Two consequences
matter more than the byte count:

1. **The read path does not have to reconstruct the 448 dims to score.** Because
   `q_Nᵀ (D z) == (Dᵀ q_N)ᵀ z`, a query projected once per layer into rank space
   scores directly against the stored `z`. The store stays small and the read
   stays short.
2. **The 448 dims can be absorbed into the grouped output projection.** Because
   `W_o,a (D z̄) == (W_o,a D) z̄` for the aggregated values `z̄`, the value path
   never materialises `x̂` either.

Both identities are exact in exact arithmetic and differ only by fp accumulation
order; both are pinned by `starkv/tests/test_reference.py`, along with the
`r = 448` round trip, which is `torch.equal` — bit-exact — because at full rank
there is nothing to truncate. That is the entire claim of this study: *the
latent is a linear object, so compressing and reading it in a different basis is
algebraically free — the only question is how much energy a rank-`r` basis
keeps.*

| rank `r` | bytes/token | vs native 584 B | capacity multiple |
|---:|---:|---:|---:|
| 64 | 196 | 0.336 | 2.98× |
| 128 | 260 | 0.445 | 2.25× |
| 192 | 324 | 0.555 | 1.80× |
| 256 | 388 | 0.664 | 1.51× |
| 320 | 456 | 0.781 | 1.28× |
| 384 | 520 | 0.890 | 1.12× |
| 448 | 584 | 1.000 | 1.00× |

The capacity multiple is the *record* ratio. It is not a serving claim: whether
it converts into slots, and whether it converts into anything else, is measured
in the serving section, which is not run yet.

## Scope and configurations

Two legs on the same v0.5.18 serving stack, TP4, fp4-native `flashinfer_mxfp4`
MoE runner, mem-frac 0.88, 1,048,576 context cap, fp8 KV cache:

1. **Native** — the unmodified tree (the image's own `/sgl-workspace/sglang`),
   stock 584-byte C4.
2. **STAR-CSA** — the patched fork (`/home/jovyan/winstonxcai/sglang-starkv`)
   with `SGLANG_OPT_STARKV=1` and a fitted basis at rank `r`.

The CSA compressor, the Lightning Indexer, the SWA pool and the c128 pool are
**not** modified. c4 entries are indexed at `token // compress_ratio` in both
the indexer's Top-`k` output and the main cache, so the indexer selects exactly
the same positions as in the unmodified model — the compression is invisible to
what gets attended, and only changes how a selected entry is stored.

### How the legs are run

Every GPU leg is scheduled by **GPUQ**, one allocation per job:

```sh
gpuq run --project starkv --gpus 4 --timeout 60m \
  --output /home/jovyan/gpuq-results/starkv \
  --cwd /home/jovyan/winstonxcai/flash-optimizations \
  -- bash starkv/scripts/local/capture.sh <tag> [duration_s] [concurrency]
```

Each leg is a **throwaway container pinned to exactly the granted devices**
(`--gpus "\"device=$CUDA_VISIBLE_DEVICES\""` — the inner quotes keep the comma
list from being read as a device *count*); nothing here ever runs `--gpus all`,
and the device list is never set by hand. The long-lived `starkv` container is a
CPU-only control box created *without* `--gpus` — it runs the CPU checks and
cannot reach a GPU at all.

### What "basis" means here

`D_l ∈ R^{448×r}` is the per-layer matrix that projects the normed NoPE latent
down (`z = D_lᵀ x`) and back up (`x̂ = D_l z`). Two providers are measured, and
the gap between them is the study's central question:

- **frozen global** — *one* `D_l` per layer, computed once offline (SVD of a
  calibration capture), then held fixed for every token, every sequence, every
  session, at every decode step. Nothing about the running sequence changes it.
- **self-fit** — `D_l` fit online on the very latents being compressed. This is
  the oracle ceiling, not a deployable scheme; it is measured to bound how much
  a trained-but-still-frozen basis could recover.

The measurement below reaches both through proxies, since neither is deployable
as stated: **held-out** (fit on half the latents offline, scored on the other) is
what frozen global buys on latents it never saw, and **self-fit** is the ceiling
above it. The gap between the two is the room training has to work in — and the
`drift` split asks the held-out question across a session rather than across a
random split, which is where a frozen basis is actually exposed.

Out-of-scope for this pass: **basis training** (soft-thresholding, distillation,
a compression penalty). A `train/` module is added only if the measurement says
a frozen basis is viable at all.

## Measurement results

Everything here is **CPU-only**, from `python -m starkv.analysis.spectrum
starkv/captures/cap2`; the per-layer tables are in
`starkv/results/spectrum-cap2.{md,json}`. The capture is 1400 decode steps per
layer over the LongSWE-Bench replay, which yields **2,708 rows per layer** on
each of the 21 CSA layers (202/204 replay requests succeeded). The store was off
for the capture leg, so every recorded latent is native.

Three retention flavours, because they fail differently and only one of them is
the question a frozen basis actually faces:

- **self-fit** — basis fit on the sampled rows, scored on those same rows. The
  oracle ceiling at that rank: no offline procedure can beat it.
- **held-out** — fit on one half, scored on the other. What a basis frozen
  *before* it saw these latents would have bought. This is the situation a
  trained-then-frozen basis is in.
- **drift** — fit on the earliest quarter of the decode window, scored on the
  latest. The same question asked across a session, on the axis a long decode
  actually travels. Drift is what a *frozen global* basis is exposed to that a
  held-out split does not model: the split is random over the whole window, the
  drift gap is the whole window between fit and score.

### Decision rule (stated before the run)

Two conditions, both required before committing to STAR-style basis training:

1. **A frozen basis must be viable at a rank where the memory win is real.**
   The frozen-global proxy must stay close enough to self-fit that the gap is
   plausibly closable by training, at **r ≤ 320** — above that the record is
   within 1.3× of native and the capacity argument is gone.
2. **The logit error on the selected entries must stay inside budget.**
   `eps_score` at that rank, on the entries the model actually attends, must be
   small enough not to move top-`k` selection. This is the condition the plan
   argues is the meaningful one, and the one an energy proxy does not predict.

If (1) holds but (2) fails, no rank below identity is usable on this workload.
If (2) holds while (1) does not, the study adopts the self-fit variant of its own
basis provider and keeps STAR's adaptive per-layer rank.

The rule is recorded here **in advance** so the verdict cannot be fitted to the
numbers after the fact. Condition (1) is readable from the spectrum below;
condition (2) is not, because the join that produces it is blocked — which is
why the section below ends in *no verdict* rather than a verdict.

### Retention spectrum

Cross-layer mean over the 21 CSA layers:

| r | bytes/token | self-fit | held-out | held-out min | drift | drift min | capacity multiple |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 260 | 0.8672 | 0.7878 | 0.7118 | 0.7271 | 0.3026 | 2.25× |
| 192 | 324 | 0.9242 | 0.8572 | 0.8013 | 0.8074 | 0.4555 | 1.80× |
| 256 | 388 | 0.9581 | 0.9071 | 0.8662 | 0.8688 | 0.5705 | 1.51× |
| 320 | 456 | 0.9794 | 0.9456 | 0.9179 | 0.9199 | 0.7187 | 1.28× |
| 384 | 520 | 0.9927 | 0.9767 | 0.9620 | 0.9635 | 0.8549 | 1.12× |
| 448 | 584 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.00× |

Per layer:

| layer | r=128 self-fit | r=128 held-out | r=192 held-out | r=256 held-out | r=320 held-out | r=384 held-out | early→late drift (r=320) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.8317 | 0.7500 | 0.8310 | 0.8904 | 0.9360 | 0.9722 | 0.9166 |
| 4 | 0.9059 | 0.8470 | 0.9052 | 0.9437 | 0.9702 | 0.9905 | 0.9551 |
| 6 | 0.8768 | 0.8048 | 0.8704 | 0.9177 | 0.9537 | 0.9813 | 0.9323 |
| 8 | 0.8758 | 0.8029 | 0.8666 | 0.9119 | 0.9475 | 0.9766 | 0.9245 |
| 10 | 0.8420 | 0.7500 | 0.8303 | 0.8901 | 0.9369 | 0.9744 | 0.9057 |
| 12 | 0.8554 | 0.7539 | 0.8292 | 0.8863 | 0.9320 | 0.9702 | 0.8998 |
| 14 | 0.8543 | 0.7558 | 0.8300 | 0.8854 | 0.9308 | 0.9687 | 0.8995 |
| 16 | 0.8461 | 0.7421 | 0.8186 | 0.8772 | 0.9245 | 0.9654 | 0.8907 |
| 18 | 0.8625 | 0.7633 | 0.8372 | 0.8917 | 0.9359 | 0.9724 | 0.9015 |
| 20 | 0.8470 | 0.7470 | 0.8247 | 0.8836 | 0.9299 | 0.9682 | 0.8956 |
| 22 | 0.8703 | 0.7931 | 0.8628 | 0.9113 | 0.9484 | 0.9773 | 0.9231 |
| 24 | 0.8315 | 0.7416 | 0.8270 | 0.8885 | 0.9343 | 0.9704 | 0.9083 |
| 26 | 0.8812 | 0.8018 | 0.8669 | 0.9130 | 0.9491 | 0.9775 | 0.9212 |
| 28 | 0.9044 | 0.8473 | 0.8989 | 0.9364 | 0.9654 | 0.9878 | 0.9430 |
| 30 | 0.8870 | 0.8145 | 0.8764 | 0.9210 | 0.9550 | 0.9819 | 0.9286 |
| 32 | 0.8418 | 0.7513 | 0.8287 | 0.8861 | 0.9311 | 0.9686 | 0.8998 |
| 34 | 0.8546 | 0.7624 | 0.8373 | 0.8914 | 0.9347 | 0.9706 | 0.9009 |
| 36 | 0.9259 | 0.8874 | 0.9257 | 0.9536 | 0.9748 | 0.9907 | 0.9590 |
| 38 | 0.8517 | 0.7748 | 0.8559 | 0.9119 | 0.9540 | 0.9860 | 0.9288 |
| 40 | 0.8058 | 0.7118 | 0.8013 | 0.8662 | 0.9179 | 0.9620 | 0.8895 |
| 42 | 0.9594 | 0.9404 | 0.9768 | 0.9911 | 0.9963 | 0.9989 | 0.9936 |

### Rank profile

Smallest rank per layer meeting **0.99 held-out** retention, and under the
stricter **0.99 drift** criterion, with the record size that rank implies:

| layer | rank at held-out ≥ 0.99 | rank at drift ≥ 0.99 |
|---:|---|---|
| 2 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 4 | **384** (520 B, 0.9905) | **448** (584 B, 1.0000) |
| 6 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 8 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 10 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 12 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 14 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 16 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 18 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 20 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 22 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 24 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 26 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 28 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 30 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 32 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 34 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 36 | **384** (520 B, 0.9907) | **448** (584 B, 1.0000) |
| 38 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 40 | **448** (584 B, 1.0000) | **448** (584 B, 1.0000) |
| 42 | **256** (388 B, 0.9911) | **320** (456 B, 0.9936) |

Held-out reaches 0.99 only at the identity rank for 18 of 21 layers: L4 (0.9905)
and L36 (0.9907) clear it at r=384, and L42 at r=256 (0.9911). Those margins are
thin — L4 and L36 clear it by 0.0005 and 0.0007 — and under the drift criterion
only L42 does better than the identity rank. The per-layer heterogeneity STAR
relies on is visible (L42 is genuinely more compressible than L40), but at a 0.99
target it does not buy a smaller record on this sample.

### Attention-logit error — BLOCKED

The capture cannot support this metric as built, so no value is reported here.
`eps_basis` / `eps_quant` are defined on the entries the model actually
selected, joined to a sampled store row through the c4 slot. That join is
failing: across all 21 layers the store capture holds only **17 distinct `loc`
values** for 2,708 rows (range `INT64_MIN`..65216, repeating with period 8),
against **162,458 distinct** values in the attention capture's index tensor. The
keys the hook records are therefore not the slots the attention selected, and
the join matched 109–295 of ~5.5M selected entries per layer — coverage
**0.0020%–0.0053%**.

`spectrum.py` carries a `MIN_COVERAGE` floor and returns `None` below it rather
than emitting a number over a sliver; the columns render `--` and the counts are
in the JSON (`score[r].entries`, `.unmatched_entries`, `.coverage`) if the join
is ever fixed. Diagnosing it means reading the vendor's `out_loc` semantics in
the decode store path (`_get_out_loc`; `compressor_v2.py` notes "for decode:
store ALL tokens. Non-boundary tokens have out_loc=0 (safe)"), which is where
the degeneracy most likely enters — and this needs another capture, i.e. GPU
time, so it is not in this pass.

One thing is already clear from the counts and does not depend on the fix: the
store hook samples only the rows being *written* during the window, while each
decode step selects 512 slots spanning the entire sequence history. Even with a
correct `loc`, coverage would be bounded by (rows written in the window) /
(entries selected), so a usable `eps` needs read-path capture, not a repair of
the write-path join.

### No verdict yet

The retention ladder above is measured; the logit metric that the decision rule
turns on is not. Retention is a proxy for it, and the plan is explicit that a
Frobenius-energy proxy does not predict the score metric — so the honest reading
of this section is *the spectrum is in, the verdict is blocked*, not "the
scheme fails at r ≤ 320".

With that said, two features of the spectrum are worth flagging before the
metric exists, because they bound what any fix can show:

1. **The oracle ceiling itself misses the 0.99 target at r ≤ 320.** Self-fit
   retention is 0.9794 at r=320 and 0.9581 at r=256. Not even a basis fit on the
   very rows being scored reaches the target at those ranks, so no amount of
   basis *training* closes that gap — training can move held-out toward
   self-fit, and self-fit is already short.
2. **Drift costs more than held-out at every rank.** At r=320 the cross-layer
   drift mean is 0.9199 against a held-out mean of 0.9456, and the per-row tail
   is heavy (worst-single-row retention 0.7187 at r=320, 0.3026 at r=128). The
   latents move within a session by more than a random half-split suggests, which
   is exactly the failure a globally frozen basis is exposed to.

## Long-decode stability

Retention loss does not announce itself as a wrong answer; it announces itself
as babble after a few thousand decode steps. A single end-of-run score cannot
tell you where the divergence started, so these instrumentation runs read the
*shape* of it. Run as a two-phase ladder: the reference continuation is
generated once on the native leg, then scored teacher-forced on both.

Filled by: `bash starkv/scripts/local/stability.sh reference <tag>` then
`stability.sh compare <tag>`, against the long replay prompts in
`starkv/data/long-prompts.jsonl`.

### Teacher-forced logprob drift

| decode length | mean \|Δlogprob\| | max \|Δlogprob\| | position bucket | n | mean \|Δlogprob\| | max \|Δlogprob\| | top-1 agreement |
|---:|---:|---:|---|---:|---:|---:|---:|
| 2k | | | 0–2048 | | | | |
| 8k | | | 2048–8192 | | | | |
| 32k | | | 8192–16384 | | | | |
| | | | 16384–32768 | | | | |
| | | | 32768+ | | | | |

The failure signature is a curve that **grows with position**. A flat curve
means the basis is holding regardless of session length.

### Greedy degeneration

| leg | tokens | distinct-1 | distinct-2 | distinct-3 | longest repeated n-gram |
|---|---:|---:|---:|---:|---:|
| native | | | | | |
| star-csa | | | | | |

### In-runtime retention telemetry

Per-store-call per-layer retention (`‖Dz‖² / ‖x‖²`) over a live session, one
summary line per store call, at `starkv/ctrl/retention.jsonl`. Read as a curve:
flat means the basis tracks the distribution; decaying means it does not.

## Serving results

Not yet run. Nothing here is a speed claim; the payoff under test is capacity.

| context | mode | max concurrency | requests/s | total tokens/s | median TTFT (ms) | median TPOT (ms) |
|---:|---|---:|---:|---:|---:|---:|
| | | | | | | |

## Fused kernel

`starkv/triton/` is a **documented stub**, `IMPLEMENTED = False`. It records the
contract the fused kernel has to satisfy — pre-project `q` once per layer,
dequantise `z` in-register, keep the two score terms separate, aggregate values
in rank space — and the two exact identities that make it correct. Writing it
before the rank regime is chosen would be premature: the block shapes are a
function of `r`.

## Reproducing

No GPU, no server. `drift` is pure text and runs on host python3; `selftest`
imports torch, so it runs in the CPU-only control container:

```sh
python3 -m starkv drift                        # anchor census vs a tree
docker exec starkv bash -c \
  'cd /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations && python -m starkv selftest'
```

Container setup (no server booted):

```sh
docker build -t starkv:v0.5.18 -f starkv/docker/local.Dockerfile .
bash starkv/scripts/local/container.sh          # image + CPU-only container + prep
bash starkv/scripts/local/container.sh patch    # patch the fork tree + verify
```

Measurement. The two `gpuq run` steps each take one allocation; the analysis
between them is CPU-only but imports torch, so it runs in the control container:

```sh
# capture: native ground-truth latents over the LongSWE-Bench replay.
# argv is <tag> [duration_s] [concurrency] [decode_steps] [row_cap]; the 4th
# argument is what makes the rank sweep measurable, so it is passed explicitly
# rather than left at its 256-step default.
gpuq run --project starkv --gpus 4 --timeout 60m \
  --output /home/jovyan/gpuq-results/starkv \
  --cwd /home/jovyan/winstonxcai/flash-optimizations \
  -- bash starkv/scripts/local/capture.sh cap2 600 8 1400 65536

# measure: retention + rank profile. CPU-only, but imports torch, so it runs in
# the control container. eps_basis/eps_quant are emitted as null -- see the
# blocked section above; the join counts are still written to the JSON.
docker exec starkv bash -c 'cd /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations && \
  python -m starkv.analysis.spectrum starkv/captures/cap2'

# long-decode drift: generate the reference on native, then score it on star-csa
gpuq run --project starkv --gpus 4 --timeout 60m \
  --output /home/jovyan/gpuq-results/starkv \
  --cwd /home/jovyan/winstonxcai/flash-optimizations \
  -- bash starkv/scripts/local/stability.sh reference <tag> 1
gpuq run --project starkv --gpus 4 --timeout 60m \
  --output /home/jovyan/gpuq-results/starkv \
  --cwd /home/jovyan/winstonxcai/flash-optimizations \
  -- bash starkv/scripts/local/stability.sh compare <tag>
```

The two capture legs are `job-20260916-064508` (`cap1`, the 256-step default)
and `job-20260916-071441` (`cap2`, `1400` steps). Both held one 4-GPU
allocation and ran their full window; the spectrum afterwards took no GPU.

The two stability phases are separate jobs because one leg at mem-frac 0.88
already claims the devices it was granted, and GPUQ grants one allocation per
job; the reference continuation is written to `$GPUQ_OUT` so the second phase
scores byte-identical text. `gpuq wait <job-id>` is the way to follow a job.

## Verification status

| check | result |
|---|---|
| Reference identities + record layout (`python -m starkv selftest`) | **47/47 CPU** |
| Anchor census vs v0.5.18 in the fork tree | **14/14 intact** |
| Patch applied + verified (`container.sh patch`) | **4 files, 143 insertions** |
| Patched sources compile (`py_compile`) | **pass** |
| Container created, no server booted | **yes — CPU-only, `DeviceRequests` empty** |
| Fork tree patched in the container | **yes — 4 files, verified in-tree** |
| Inference (either leg) | **yes — `cap1`/`cap2` capture legs ran their full 256- and 1400-step windows per layer; health checks passed; replay client 221/223 and 202/204, both `valid: true` at the 0.99 floor** |
| Capture + spectrum | **yes — `cap2`: 2,708 rows/layer × 21 layers; spectrum CPU-only, `results/spectrum-cap2.{md,json}`** |
| Attention-logit error | **BLOCKED — join coverage 0.002%-0.005%; see Measurement results** |
| Stability ladder | **not run** |
| Serving comparison | **not run** |

The capture legs ran as `gpuq run` jobs (`--gpus 4`, throwaway container pinned
to the granted devices, one allocation per job), never `docker run --gpus all`.
The spectrum runs consumed no GPU: they are CPU-only reads of the capture under
`starkv/captures/`, which is gitignored.
