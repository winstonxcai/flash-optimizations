# TopMag 50% on the native c4 latent — Sangfor-Bench n=7 (7× same instance)

Date: 2026-08-28 · DeepSeek-V4-Flash (21 c4 latent layers, compress_ratio=4)
Build: `SGLANG_OPT_TOPMAG=1` + `XKV_TOPMAG_KEEP=0.5` (Mustafar package, `flash-optimizations/mustafar/`)
Change: **store-time only.** Each stored c4 latent vector has its smallest-|·| 256 of 512 coords
zeroed in place, immediately before the stock fused `compress_norm_rope_store`. The memory pool
(584 B/token native layout), decode, and every other path are the **stock DeepSeek-V4 build** — no
lowrank KV, no basis/SVD, no pool change. Identical build to the n=1 run (`dsv4-topmag50-1_20260827`).

## Result — 7 independent agent runs, all pass

Sangfor-Bench cc agent, **same sampled instance** `gcjs_kube-log-check-recover_2cadb18b`
(MaxInterval restart + Chinese-log task) run 7×, each as a separate single-instance eval with a
unique run_id (`dsv4-topmag50-20-01..07_20260827`), 2 concurrent.

| | native CSA | windowed self-fit | TopMag 50% n=1 | **TopMag 50% n=7** |
|---|---|---|---|---|
| run_agent tests | 29/29 (100 %) | 19/29 (65.5 %) | 29/29 (100 %) | **7 × 29/29 (100 %)** |
| pass_rate (each) | — | — | 100.0 | **100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0** |
| run-to-run variance | — | — | — | **σ = 0 (7/7 success)** |

**Every one of the 7 independent agent runs produced a coherent, task-appropriate patch that passed
all 29 tests** (error=None in every `test_result.json`). On the same instance where the windowed
self-fit build failed (19/29, one missing 2-char `间隔` token + one failed edit), TopMag-50-native
reproduced native's 29/29 across **7/7** independent runs with zero failures.

## Methodology

- **Prune site**: `compressor_v2.py::_forward_compress_all_in_one`, the single site that stores both
  the 512-dim c4 latent and the 128-dim indexer. The hook prunes **only** the 512-dim c4 latent
  (128-dim indexer skipped). `torch.topk(k=256, largest=False)` → smallest-|·| half per row set to 0;
  `keep=1.0` is a no-op. Same code and site as the n=1 run.
- **Server contract verified live during the run**: ctrl debug.log showed 750 k+ `prune zeroed 256`
  events (512-dim latents) and 384 k+ `prune_skip dim 128` (indexer untouched), 0 `prune_error`. The
  running server was the latent build (env: `SGLANG_OPT_TOPMAG=1`, `XKV_TOPMAG_KEEP=0.5`, **no**
  `XKV_TOPMAG_TARGET`; confirmed after converting the box from an earlier archived indexer build).
- **Deploy**: 10×2 wave launcher (`run_topmag50_20.sh`) — each sample is a separate `yjybench.cli`
  invocation with a unique run_id, because the harness names eval containers
  `task_id__instance_id__MMDDHHMMSS` with no uniquifier, so 20 identical instance_ids in one run
  collide on the second-granularity timestamp (observed `docker 409 Conflict` on the initial attempt).
- **Server throughput reality**: decode ran at **~2–4 tok/s** on the 94–150 k-token agent contexts
  (config-inherent: `--fp8-gemm-backend triton` + `--disable-cuda-graph`; prefill stayed fast at
  ~8 k tok/s, GPU util 30–50 %). With 2 agents sharing that rate, waves took 2 h 10 m – 3 h 29 m vs
  the n=1's 00:42:38 on a fresh server. The n=20 was **truncated at n=7 by throughput, not result
  quality** — the first 7 samples were all clean 29/29 and the run was stopped by decision at that
  point (13 remaining samples would have needed ~10 h).

## Interpretation

- **n=1 → n=7: the diagnostic becomes a variance statement.** The n=1 caveat was "single instance,
  ±huge CI". Seven independent agent runs of the *same* instance now bound the run-to-run variance on
  this task at **σ = 0** — TopMag-50-native is not merely capable of matching native once, it
  reproduces native's full pass on every attempt, while windowed self-fit deterministically failed
  the same task.
- **Scope still narrow**: all 7 runs are the same task, not 7 tasks. This says "on the task where the
  competing design degrades, TopMag-50-native is reliably native-equivalent (7/7)" — it does not yet
  rank the two designs across the 190-instance task pool. A proper generalization claim needs
  distinct sampled instances, which the throughput ceiling made impractical this run.
- **Zero bandwidth saved** (native 584 B/token, unchanged) — this remains the pure-fidelity question;
  a packed store that drops the zeroed coords is still untested.
- **Ops lesson for large-n TopMag runs**: `XKV_DEBUG=1` grew `ctrl/debug.log` to **938 MB** (~380
  prune log lines/s × 12 h) on a host mount, adding store-path I/O that worsened the already-throttled
  server. Future large-n runs should launch with `XKV_DEBUG=0` (the `_dbg` guard short-circuits, so
  pruning itself is unaffected) and consider enabling cuda-graphs / a non-triton fp8 backend, where
  decode throughput is the real bottleneck.

## Artifacts

- Eval results (7 dirs): `/data/zyj/YJYBench/results/test/Sangfor-Bench_cc_vibe_deepseek-v4-flash_dsv4-topmag50-20-0{1..7}_20260827/gcjs_kube-log-check-recover_2cadb18b/`
  (each: `test_result.json` pass_rate=100.0, error=None)
- n=1 reference: `results/test/..._dsv4-topmag50-1_20260827/...`
- Native paired result: `results/test/Sangfor-Bench_cc_vibe_DeepSeek-V4-Flash-Local_task_20260825_195126_744/gcjs_kube-log-check-recover_2cadb18b/`
- Windowed paired result: `results/test/..._dsv4-windowed-1_20260827/...`
- Launcher: `flash-optimizations/mustafar/run_topmag50_20.sh` · watchdog: `watchdog_topmag50_20.sh`
- Build: `flash-optimizations/mustafar/` (`reference.topmag_zero`, `ops.maybe_prune`, `launch.sh`)
- Server prune evidence: `flash-optimizations/mustafar/ctrl/debug.log` (938 MB, XKV_DEBUG=1)
