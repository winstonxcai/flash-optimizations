# Decode cost attribution in the mustafar c4 path (TP4, per-rank)

## Run / environment

- **Date:** 2026-09-17. Wall clock ~2.9 h for the matrix, 24/24 points valid, 0 failures.
- **Hardware:** 4x NVIDIA H100 80 GB, TP4 (GPUs 0-3, one NV18 full-mesh NUMA-0 node),
  held under a `gpuq` lease for the duration.
- **Serving:** container `remnant`, SGLang v0.5.18, model `DeepSeek-V4-Flash-0731`,
  port 30212. One boot per (mode, pass): 4 modes x 3 passes = 12 boots.
- **Modes:** `native` (stock tree, no mustafar on `PYTHONPATH`),
  `packed` (`TOPMAG=1 PACKED=1`), `optimized` (same + fused load),
  `topmag` (`TOPMAG=1 PACKED=0` -- the pruning-without-layout control).
  Mode order rotates by one each pass so drift decorrelates from mode.
- **Client:** `sglang.bench_serving`, random 32768-in / 2048-out, `--request-rate inf`,
  `--flush-cache`, `--seed 42`, a warm-up wave at C then a measured wave of 3C prompts
  at `--max-concurrency C`. C in {8, 25}; C=25 is the report's ladder point.
- **Profiler:** torch profiler via `POST /start_profile`, `num_steps=20`,
  `profile_stages=["decode"]`, `record_shapes=true`, `merge_profiles` unset
  (unsupported in v0.5.18 -- per-rank files are the only option). Fired 30 s into
  the measured wave so the captured steps run at the pinned batch C.
  192 traces, 7.1 GB.

This is a **synthetic matched** measurement at a pinned client config, not a
reproduction of the LongSWE-Bench SLO legs. Its job is per-kernel attribution.

## Verdict

1. **The +2.31 ms/step cutlass MoE GroupedGEMM penalty does not exist.** Across all
   four modes the GroupedGEMM total is flat to within the pass-to-pass noise
   (<=1.011x). The 09-15 attribution that put 46% of packed's penalty on it was
   wrong. No MoE-routing perturbation to chase.
2. **packed costs +2.291 ms/step over native at C=25**, split into
   **pruning +0.840** and **c4 record layout +1.451**; the fused load gives back
   **-1.173**. The layout term is real and is the larger half.
3. **The fused-load saving replicates and is rank-uniform** -- 1.168-1.173 ms/step
   on every rank at C=25, matching the window delta exactly.
4. **The all-reduce corruption recurs at 3 of 24 points (12%), all at C=8**, and
   **not at all at C=25** (every C=25 point is 1.000x rank spread).

## Measurements

### Decode window, ms/step (20 profiled steps, mean over clean points)

| mode | C=8 | C=25 | C=25 rank spread | C=25 pass spread |
|---|---:|---:|---:|---:|
| native | 10.849 | 15.707 | 1.000x | 1.002x |
| topmag | 11.781 | 16.547 | 1.000x | 1.001x |
| optimized | 12.868 (corrupt pt) | 16.825 | 1.000x | 1.002x |
| packed | 12.278 | 17.998 | 1.000x | 1.003x |

Delta vs native (ms/step):

| term | C=8 | C=25 |
|---|---:|---:|
| pruning (topmag - native) | +0.932 | +0.840 |
| c4 layout (packed - topmag) | +0.497 | +1.451 |
| fused load (optimized - packed) | -0.589 | -1.173 |
| **packed total** | **+1.429** | **+2.291** |

Excluding the one corrupt `optimized` C=8 point: native 10.765, topmag 11.803
(+1.037), packed 12.278 (+0.476), optimized 11.822 (-0.456).

Per-step wall clock is the **trace window** (first kernel start -> last kernel
end), never sum-of-durations. Decode is gap-free and the all-reduce overlaps
compute, so Sigma-duration overcounts by ~16% (sum/window ~1.16). Both are
reported by the tool; they are not interchangeable.

### Q1 -- the GroupedGEMM penalty is absent, in every mode

cutlass MoE GroupedGEMM, ms per 20 decode steps (3 passes each):

| mode | C=8 r1/r2/r3 | C=25 r1/r2/r3 |
|---|---|---|
| native | 78.89 / 78.33 / 78.90 | 152.37 / 153.11 / 152.58 |
| topmag | 77.90 / 78.72 / 78.79 | 151.58 / 151.96 / 151.42 |
| optimized | 78.38 / 78.90 / 78.40 | 152.39 / 152.84 / 152.44 |
| packed | 78.89 / 78.92 / 78.83 | 151.64 / 151.90 / 152.46 |

At C=25 the four mode means are 152.686 / 151.650 / 152.556 / 151.997. The
largest mode difference (1.011x at C=8) is **smaller than the pass-to-pass
spread of a single mode**, so the measurement floor is above any mode effect.
Measured delta vs the plan's hypothesis: **-0.03 ms/step, against a predicted
+2.31**. The penalty is not there to explain.

The consequence is that pruning does not perturb MoE routing at a measurable
level, so the choice between "pruning perturbs routing" and "the c4 layout is the
cause" is not a live one -- the second framing never had a denominator.

### Q2 -- the fused-load saving replicates and is rank-uniform

Three unfused reconstruction kernels (`_bf16_to_native_kernel`,
`_unpack_gather_bf16`, `_rope_tail_complex_inplace`) vs the single fused load,
kernel totals in ms per 20 steps:

| C | rank | unfused (packed) | fused (optimized) | saving ms/step | window saving |
|---|---:|---:|---:|---:|---:|
| 8 | 0 / 1 / 2 / 3 | 9.80 / 9.81 / 9.84 / 9.79 | 2.21 / 2.22 / 2.21 / 2.21 | 0.380 / 0.380 / 0.381 / 0.379 | 0.365 / 0.456 / 0.636 / 0.387 |
| 25 | 0 / 1 / 2 / 3 | 28.40 / 28.38 / 28.26 / 28.28 | 4.93 / 4.93 / 4.91 / 4.92 | 1.173 / 1.172 / 1.168 / 1.168 | 1.173 / 1.173 / 1.173 / 1.173 |

At C=25 the kernel-level saving and the window saving agree to the last digit on
every rank. The C=8 window column is noisier than the kernel column because the
C=8 traces carry the corrupt-collective artifact (below); the kernel totals are
the trustworthy reading there.

### Q3 -- the all-reduce corruption recurs, only at the smaller batch

Per-rank `all_reduce_kernel` spread within a point, flag threshold 2x:

| point | all-reduce range (ms) | spread |
|---|---|---:|
| native-c8-r1 | 8.46 - 21.34 | **2.52x CORRUPT** |
| topmag-c8-r1 | 9.90 - 20.34 | **2.06x CORRUPT** |
| optimized-c8-r2 | 10.01 - 114.30 | **11.42x CORRUPT** |
| every C=25 point (12/12) | 9.87 - 12.49 | 1.000x window spread |

**3 of 24 points, 12%, all at C=8.** The worst instance is `optimized-c8-r2`:
TP0/TP2/TP3 at 114.30/108.71/65.53 ms against TP1 at 10.01 ms -- 11.4x on the
same kernel with identical launch counts (1740) and identical kernel counts
(43220 events on every rank). This is the same signature as the 09-15 `native`
non-graph capture: the profiler dumping host-side launch-bound idle into the
collective on a subset of ranks. It is a capture artifact, not a workload
property.

**No C=25 point is affected**, including the `optimized` point whose window
spread the 09-15 analysis found at 1.082x. The plan's 2x exclusion rule was
applied; C=8 conclusions that survive it are quoted from the kernel totals.

### Q4 -- window spread does not track batch size positively

| C | min | median | max |
|---|---:|---:|---:|
| 8 | 1.006x | 1.013x | 1.026x (1.432x incl. corrupt) |
| 25 | 1.000x | 1.000x | 1.000x |

NCCL holds the ranks in lockstep, so a spread above ~1.02x on a 4-rank point is
a capture artifact. C=25 is at the measurement floor; C=8 is not. The prediction
that a *larger* batch would show *more* skew is not supported -- the reverse
ordering is what the data shows, and the C=8 tail is the corrupt point.

### Where packed's +2.291 ms/step goes (C=25, per-kernel totals)

| kernel group | native | topmag | optimized | packed | packed delta |
|---|---:|---:|---:|---:|---:|
| `radixSortKVInPlace` | 0.000 | 9.934 | 9.979 | 9.949 | +0.497 |
| `sbtopk::gatherTopK` | 0.000 | 9.596 | 8.324 | 8.242 | +0.412 |
| `_bf16_to_native_kernel` | 0.000 | 0.000 | 0.000 | 12.230 | +0.611 |
| `_unpack_gather_bf16` | 0.000 | 0.000 | 0.000 | 11.993 | +0.600 |
| `_rope_tail_complex_inplace` | 0.000 | 0.000 | 0.000 | 4.107 | +0.205 |
| `_pack_fp8_kernel` | 0.000 | 0.000 | 2.526 | 2.514 | +0.126 |
| fused load | 0.000 | 0.000 | 4.923 | 0.000 | -- |
| cutlass GroupedGEMM | 152.686 | 151.650 | 152.556 | 151.997 | -0.034 |
| MLA sparse | 22.926 | 22.768 | 22.088 | 22.495 | -0.022 |

Units are ms per 20 decode steps; divide by 20 for ms/step. Sigma-duration
deltas (topmag +0.925, packed +2.395) reconcile with the window deltas
(+0.840, +2.291) within ~5%.

`topmag` confirms the sort cost is a property of pruning, not of the c4 layout:
it carries `radixSortKVInPlace` + `sbtopk::gatherTopK` at the same magnitude as
`packed` (19.53 vs 18.19 ms/20 steps) while carrying none of the c4 record
kernels. That is the pre-registered `_pack_fp8_kernel` absence check from the
plan, and it holds: 0 events in `topmag`, 420 per point in `packed`/`optimized`
(case-insensitive substring match on the trace text).

Reproduction of the earlier decomposition, at the corrected operating point:

| term | 09-15 table | this run |
|---|---:|---:|
| cutlass MoE GroupedGEMM | +2.31 (46%) | ~0.00 |
| c4 kernels | +1.30 (26%) | +1.54 |
| sort | +0.94 (19%) | +0.91 |
| everything else | +0.46 (9%) | -0.16 |
| **total** | **+4.34** | **+2.291** |

The c4 and sort terms replicate closely. The total differs almost entirely
because the GroupedGEMM term -- the single largest term in the old table -- is
not real.

## Caveats

- **The capture artifact is real and recurs.** 3 of 24 points (12%) carry a
  corrupt `all_reduce_kernel`, all at C=8. Any collective claim must come from
  C=25, where the artifact did not recur, or must exclude the flagged points.
  `trace_ranks.py summary` prints the per-rank all-reduce spread for exactly this
  reason -- check it before quoting a derived number.
- **The 09-15 `optimized` trace is a different kernel.** It measures the
  **in-kernel-mask variant** (`_pack_fp8_kernel` at 20.85 us vs packed's 2.36 us),
  which has since been reverted. It is not a baseline for the current tree. The
  current `optimized` = packed's store path (sort + pack) + the fused load, which
  is what this run measured. See `remnant-topmag-sort-elimination.md`, which
  still documents the in-kernel-mask change and its 0.94 ms/step headline as
  scoping for a change that did not ship in that form.
- **`optimized` did not dispatch `geometry`.** The plan expected the
  `MUSTAFAR_FUSED_DISPATCH=geometry` marker; every 09-17 boot of `optimized`
  emitted `packed_to_native_optimized`. By `mustafar/fused.py:125-139`,
  `candidate` stays `None` only when `geometry_unsupported` is true, i.e. one of
  `physical_indices.dim() != 2`, `shape[1] != 512`, `page_size != 16`. The fused
  call site derives `page_size` from the pool
  (`compressor_v2.py:263`/`:286`), and the live server reports
  `page_size=256, c128_page_size=16`, so the third condition is the likely one --
  but the failing condition was not instrumented, so this is inference. The
  measurement is unaffected (the fused-load saving above is the generic
  `packed_to_native_optimized` body, rank-uniform and window-confirmed), but the
  claim in `remnant-fused-kernel.md` that geometry "is now the promoted optimized
  serving path" does not hold for this serving geometry.
- **`profile-matrix.sh`'s `assert_leg` is unsound for `native`.** It greps the
  boot log for `mustafar` to detect a patched tree; native's 32 hits are all the
  results-directory path echoed into the log. The real guarantee for native is
  `PYTHONPATH` excluding `$REPO_CT` plus the absence of a dispatch marker, both
  of which held. The WARN is a false positive and should not be read as evidence
  of tree contamination.
- **`profile_stages` is a silent no-op on the live path.** The container runs
  `SGLANG_PROFILE_V2` unset, so `srt/managers/scheduler_components/profiler_manager.py`
  serves the request and never stores `profile_stages`; DECODE and EXTEND are
  both captured, and EXTEND can open a session after a point has finished with
  it. `profile-ranks.sh` handles this by clearing a session on entry and on exit.
  The decode traces are unaffected, but a future driver written against
  `srt/utils/profile_utils.py` will silently disagree with what the server does.
- **This is not an SLO measurement.** The client config is pinned for attribution,
  and the in-kernel profiler perturbs timing badly. The wave throughput numbers in
  `RESULT.txt` are validity evidence, not results.
- **C=25 sits above the small decode config's `max_bs`.** `DECODE_CFG_EXT` is
  required or the point falls off-graph and stops being comparable.

## Artifacts

- Traces: `mustafar/results/profile-20260917/` (192 files, 7.1 GB, `*-TP-{0..3}-{DECODE,EXTEND}.trace.json.gz`)
- Per-point client logs: `mustafar/results/profile-run-20260917-full/<mode>-c<C>-r<pass>/`
- Point validity: `mustafar/results/profile-run-20260917-full/RESULT.txt` (24 `OK` lines)
- Per-rank summary: `mustafar/results/profile-20260917-analysis/summary-decode.txt`
- Per-kernel breakdowns: `mustafar/results/profile-20260917-analysis/kernels/` (96 files)
- Aggregate and Q1-Q4 tables: `mustafar/results/profile-20260917-analysis/aggregate.txt`
- Boot logs (dispatch markers): `mustafar/logs/serve_{native,packed,optimized,topmag}.log`
- Archived earlier smoke runs: `mustafar/results/profile-20260917-smoke/`,
  `mustafar/results/profile-20260917-stale/`

## Reproduce

```text
# 4 modes x 3 passes x {C=8, C=25}, ~2.9 h
gpuq lease acquire --project mustafar --gpus 4 --gpu-ids 0,1,2,3 --ttl 240m --keep-idle --detach
gpuq lease exec <lease_id> --timeout 12600 -- \
  bash /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts/local/profile-matrix.sh

# analysis
cd mustafar
python3 tools/trace_ranks.py summary results/profile-20260917/*DECODE* --steps 20
python3 tools/trace_ranks.py kernels results/profile-20260917/packed-c25-r1-TP-0-DECODE.trace.json.gz --top 30
```

`profile-matrix.sh` and `profile-ranks.sh` are new in this study;
`tools/trace_ranks.py` is the committed analysis tool. See
`remnant-fused-kernel.md` for the fused reconstruction kernel under test.

Source: `codex/remnant-sparse-kernel`; container `remnant`, SGLang v0.5.18;
deepseek-v4-flash-0731; TP4 on 4x H100 80 GB.
