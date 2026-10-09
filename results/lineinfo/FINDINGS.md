# H64 Remnant vs Native Sparse-Decode — Verified Findings (rev 2)

Fixes and supersedes part of [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md): the timing
methodology (§5b) and the source-correlation status (§5c) are corrected here from
direct evidence. Everything below is reproducible from the branch artifacts.

- Image: `remnant:v0.5.18` = `sha256:4bc8c844f4401592bbb5aa88bc61f942000fd92f276f4824c38955e80a38760e`
- Container (disposable): `remnant-pilot-20261009`. The long-lived `remnant`
  container and the `weave-repro` job (GPUs 4-7, port 30018) were never touched.
- Fixture: h64, batch 32, q1, swa64 / context 4512, seed 20260919, MODEL1
  (`d_qk=512`), `sm_scale = d_qk**-0.55`. `third_party/flashmla/tests/lib.py`
  md5 `7bbfefa9b7f851f8f8f1f5bc32e5f179` (== container `/tmp/pilot/lib.py`).
  Submodule flashmla `9f9a2e5da6e82d5f7730e1515d21fa313fb34604`.

## 1. Timing — corrected methodology

CUDA events around **one replay of a captured graph that contains 100 complete
decode calls**; reported value = elapsed GPU time / 100. Both graphs warmed; 10
matched **A/B/B/A** rounds (native, direct, direct, native). One replay = 100
complete calls; each complete call = splitkv **+ combine**, because the combine
kernel is launched inside the C++ op right after the splitkv kernel
(`csrc/api/sparse_decode.h`, container `/opt/flashmla-remnant`).

Harness: [pilot/mb_parity_timing.py](pilot/mb_parity_timing.py) (sha256 in JSON),
driver [mb_run.sh](mb_run.sh). Loaded module during the run = the lineinfo build
sha256 `b44d6826…`; the release `.so` sha256 `23497e35…` was restored afterward.

| path | median us/call | mean | min | max |
|------|---------------:|-----:|----:|----:|
| native  | **21.69** | 21.67 | 21.39 | 21.91 |
| direct  | **29.14** | 29.15 | 28.76 | 29.32 |

Parity (finite out/lse, atol=rtol=0.02): `out_max_abs_diff=0.001953125`,
`lse_max_abs_diff=1.9073486328125e-06`; graph parity identical.

**Why this differs from earlier numbers.** The prior `pilot.py` `timed()` divided a
50-replay *batch* by 50 (23.2/30.7 us — cheap in-batch submission). The first
`mb_parity_timing.py` draft timed a **single** replay bounded by
`torch.cuda.synchronize()` on both sides, which folds in launch/sync-tail overhead
and read 52/47 us. Neither is the per-call GPU cost. The corrected method
(event-timed 100-call graph / 100) is what the medians above report.

## 2. Negative result — both paths launch the SAME kernel; direct is slower

At this fixture both spellings dispatch the **same** sm90 sparse-FP8 MLA decode
kernel; they differ only in the extra-K page layout, not in the kernel body:

- native  cubin: `KernelTemplate<ModelType1, 64, false>` (REMNANT=false) — REG:168, 6447 instrs
- remnant cubin: `KernelTemplate<ModelType1, 64, true>`  (REMNANT=true)  — REG:128, 5480 instrs

The lineinfo build's profiled kernel is SASS-identical to the release cubin
(5480 instrs, identical resource record). The **direct** (remnant) path is
~34% slower end-to-end here (29.14 vs 21.69 us), the **opposite direction** from
the isolated `ncu` main-kernel durations in [FINAL_ANALYSIS.md](FINAL_ANALYSIS.md)
§4 (which showed the wide-producer remnant kernel faster on its lone tile). No
claim is made that either is faster in general; the two measurements time
different scopes (whole op incl. host/stream overhead and combine vs. one kernel
launch under ncu).

## 3. ncu source correlation — RESOLVED (the prior "BLOCKING" verdict was wrong)

[FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) §5a/§5c claimed source correlation was
"BLOCKING" because `ncu --page source` showed zero `.cu:<line>` matches. That was
an **invocation artifact**, not missing debug info.

**Fix:** source correlation requires `--print-source cuda,sass` (the default
`--print-source` view emits SASS only). With the correct flag, both saved release
captures resolve real `.cu` lines:

```
ncu --import <rep> --page source --print-source cuda,sass
```

Evidence (saved): [source_correlation_remnant.txt](source_correlation_remnant.txt),
[source_correlation_native.txt](source_correlation_native.txt). 33 `File Path:`
source files resolve; the kernel's own TU `csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`
contributes ~289 instruction rows. Hottest lines by executed threads (remnant):

| file:line | source |
|-----------|--------|
| splitkv_mla.cuh:419 | `cur_rO(i) *= scale_for_old;` |
| splitkv_mla.cuh:787 | `cur_rO(i) *= cur_scales[local_row_idx];` |
| splitkv_mla.cuh:37-44 | remnant nibble/rank unpack (`remnant_nibble_rank_pack`, `__popc`) |
| splitkv_mla.cuh:82 | `remnant_window_fp8x4(...)` |
| splitkv_mla.cuh:426 | `cur_rS(i) = (bf16)cur_rP(i);` |

So per-source-line attribution for this kernel **is** available; no re-capture is
needed. The prior conclusion should be struck.

## 4. nsys timeline — what is and is not representable (not marked complete)

Working from the DBs (read-only) rather than `nsys stats` aggregates:

- Per-kernel graph correlation is **absent**: in both the packed and native
  captures every kernel row is `launchType=0` (eager) with `graphNodeId=0` and
  a single `graphId=0`. Graph structure must come from
  `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` (`graphId`,`graphExecId`) and
  `CUDA_GRAPH_EVENTS`.
- `CUDA_GRAPH_EVENTS`: 348 `Graph Creation` (eventClass 95) + 116
  `GraphExec Creation` (eventClass 94), the 116 split evenly across exactly the 4
  TP-rank `globalTid`s.
- `GRAPH_TRACE`: packed 29582 rows / 16 distinct `graphExecId` / 4 ranks; native
  26162 rows / 15 `graphExecId` / 4 ranks. Spans begin negative because the graph
  API timeline starts before the first kernel.
- A **steady-state window boundary** and per-rank profiler API entry/exit
  boundaries were **not** independently reconstructed from the raw event rows and
  are therefore **not** marked complete here.

## 5. Artifacts and provenance

- Timing: [pilot_out/mb_timing.json](pilot_out/mb_timing.json),
  [pilot_out/mb_parity.json](pilot_out/mb_parity.json).
- Source correlation: `source_correlation_{remnant,native}.txt` (this dir).
- Emission JSON (shas, rounds, paths): `FINDINGS.json` (this dir).
- Repo/commit and the exact pushed paths are recorded in `FINDINGS.json` at
  publish time.
