# Windowed self-fit: the pivot for lowrank KV-decode

Date: 2026-08-27 · DeepSeek-V4-Flash (21 c4 latent layers, compress_ratio=4)
Build: `SGLANG_OPT_LOWRANK_KV_STORE=1` + `XKV_WINDOWED=1` + `XKV_WINDOW=4096` + `XKV_COEFF_DIM=192`

## Why we need it

Fixed-basis lowrank KV-decode is dead by construction: the c4 latent subspace is **content-dependent
and drifts during decode**, so a precomputed basis retains only 0.77–0.87 of per-layer energy @ r320 →
**10/10 garbage** on Sangfor-Bench (babble, 0–1 tool calls). Real xKV avoids this with an **online
per-sequence SVD**; we now do the same, per window: the newest W=4096 c4 tokens stay **native** (full
rank), and at each window boundary a rank-192 basis is SVD-fit on that window's **own** normed
latents and re-encoded in place (self-fit retention ≈ 1.0). This is the **correctness-first pivot**:
it trades most of the memory win for coherence.

## Initial 1-sample result (Sangfor-Bench cc agent, same task)

| | native CSA | windowed self-fit |
|---|---|---|
| run_agent tests | 29/29 (100 %) | **19/29 (65.5 %)** |
| resolved | true | false |

First lowrank eval that produced a **coherent, task-appropriate patch**: the agent read the right
file, planned the correct fix, applied the core edit — all 9 core-logic tests pass. The 10 failures
are exact-string gaps, not coherence — 8× a missing 2-char token (`间隔`) in one Chinese log message,
2× an edit that failed to apply. **n=1: diagnostic, not a ranking** — a 10-sample paired gate is
running (2026-08-27).

## Honest memory saving

| build | B/token (c4 pool) | vs native |
|---|---|---|
| native CSA | 584 | — |
| fixed-basis xkv (r192) | 200 | 2.92× |
| **windowed self-fit (r192)** | **528** | **1.11× = 9.6 %** |

Only the c4-latent pool is touched (SWA/full + c128 stay native). Slots are uniform **528 B** so the
newest window can be stored native and re-encoded in place; a compressed window fills only ~196 B of
its slot, but the pool cannot reclaim the rest — **the saving is slot width, not the compression.**
The future packed build (native ring + compact ~200 B history pool) would restore ~2.2–2.9× and is
**gated on the 10-sample result**.
