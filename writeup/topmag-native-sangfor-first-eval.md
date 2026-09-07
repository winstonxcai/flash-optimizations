# TopMag 50% on the native c4 latent — first Sangfor-Bench eval (n=1)

Date: 2026-08-27 · DeepSeek-V4-Flash (21 c4 latent layers, compress_ratio=4)
Build: `SGLANG_OPT_TOPMAG=1` + `KEEP=0.5` (Mustafar package, `flash-optimizations/mustafar/`)
Change: **store-time only.** Each stored c4 latent vector has its smallest-|·| 256 of 512 coords
zeroed in place, immediately before the stock fused `compress_norm_rope_store`. The memory pool
(584 B/token native layout), decode, and every other path are the **stock DeepSeek-V4 build** — no
lowrank KV, no basis/SVD, no pool change.

## Result — single-instance smoke eval

Sangfor-Bench cc agent, 1 sampled instance (`gcjs_kube-log-check-recover_2cadb18b`, MaxInterval
restart + Chinese-log task), max_workers=1 → **29/29 run_agent tests (100 %), resolved=true**.

Paired against the stored native CSA result for the **same instance** (`task_20260825_195126_744`)
and the windowed self-fit build (same instance, `dsv4-windowed-1_20260827`):

| | native CSA | windowed self-fit | TopMag 50% (native) |
|---|---|---|---|
| run_agent tests | 29/29 (100 %) | 19/29 (65.5 %) | **29/29 (100 %)** |
| resolved | true | false | **true** |

Agent loop: 00:42:38, 9 tool calls (Read → Edit×2 → `go build` → `go test` → verify → Read×2 → finish),
a coherent, task-appropriate patch (full diff in Artifacts). TopMag 50 % matches native exactly and
beats windowed self-fit on the same instance.

## Methodology

- **Prune site**: `compressor_v2.py::_forward_compress_all_in_one`, the single site that stores both
  the 512-dim c4 latent and the 128-dim indexer. The hook prunes **only** the 512-dim c4 latent
  (128-dim indexer skipped — verified in the ctrl debug log: 1396 `prune` / 756 `prune_skip`, 0
  errors during smoke). `torch.topk(k=256, largest=False)` → the smallest-|·| half per row is set to
  0; `keep=1.0` is a no-op. Numeric self-test passes: exactly 256/row zeroed, zeroed set == smallest-k
  per row, non-zeroed coords bit-identical.
- **No memory change**: the c4 store keeps its native 584 B/token layout; zeros quantize to exactly 0
  in the native fp8 store. Decode path is byte-identical to native (same pages, same kernels) — only
  the stored values differ.
- **Smoke gate passed** before the eval: short tool-use probe → clean `tool_use` (`bash {"command":
  "pwd"}`), and a ~17.9k-token context probe → clean `tool_use` (`echo OK`), 0 server errors, 0
  garbage.
- **Deploy**: Mustafar patch injected an import block + `_sg_lr.maybe_prune(kv_compressed)` hook into
  the active source (`/sgl-workspace/sglang-lowrank/python`); all old `XKV_LOWRANK` markers cleared
  from the 4 previously-patched files (verified absent).

## Interpretation

- **Fidelity: indistinguishable from native on this task.** The agent read the same file, produced
  the same class of fix, and passed **all 29 tests including the exact-string Chinese log assertions**
  — the 8× B-series sleep-path tests the windowed build failed on one missing token (`间隔`) all PASS.
  TopMag 50 % is the purest test of magnitude-pruning the c4 latent: no memory saved, no decode
  change — and on this agentic workload the pruned latent decoded cleanly enough for a 100 %
  task-appropriate patch.
- **Why TopMag is gentler than windowed lowrank.** Zeroing 50 % of each latent's coords preserves the
  survivor subspace exactly (decode of the kept coords is bit-identical to native; zeros are
  unambiguous). The windowed build instead re-encodes whole windows through a fitted rank-192 basis —
  a *global* recomposition that measurably drifted one exact string and dropped one edit. Magnitude
  pruning of the smallest |·| coords concentrates the loss in directions the model cares least about.
- **Server health**: 743 k `prune` / 381 k `prune_skip` (indexer correctly untouched), 0
  `prune_error` across the full eval — the hook is a clean no-op-risk addition to the native store.
- **n=1.** No resolve-rate estimate; diagnostic value only. Native is 99/190 resolved across the full
  suite; this result says TopMag-50-native can behave like native on a task windowed lowrank degrades
  — it does not rank the two designs globally.

## Caveats

- Single instance; ±huge CI on anything quantitative.
- TopMag 50 % here costs **zero** bandwidth (native 584 B/token, unchanged) — this is the pure
  fidelity question answered, not a memory-savings result. The follow-up question (does a
  packed/bandwidth-saving TopMag store — e.g. dropping the zeroed coords from the fp8 pages — keep
  this fidelity?) is untested.
- This task is a near-minimal agentic workload (read one file, edit one function); does not test
  long-context retention under heavy compaction.

## Artifacts

- Eval results: `/data/zyj/YJYBench/results/test/Sangfor-Bench_cc_vibe_deepseek-v4-flash_dsv4-topmag50-1_20260827/gcjs_kube-log-check-recover_2cadb18b/`
  (`test_result.json`, `test_output.log`, `cc_claude_steps.jsonl`)
- Native paired result: `results/test/Sangfor-Bench_cc_vibe_DeepSeek-V4-Flash-Local_task_20260825_195126_744/gcjs_kube-log-check-recover_2cadb18b/`
- Windowed paired result: `results/test/Sangfor-Bench_cc_vibe_deepseek-v4-flash_dsv4-windowed-1_20260827/gcjs_kube-log-check-recover_2cadb18b/`
- Build: `flash-optimizations/mustafar/` (`reference.topmag_zero`, `ops.maybe_prune`, `launch.sh`)
- Agent patch (`fix.patch`): `NeedRestartBroker` gained the MaxInterval branch with the exact
  `"等待时间达到MaxInterval，重启Pod"` log, the sleep-path log gained `间隔`
  (`"距离上次重启时间间隔达到SleepInterval，重启Pod"`), the broken `maxTsleepTimerimer` timer was
  removed from `RunRestartBrokerController`, `GRACEPERIODSECONDS` → `DEFAULT_GRACEPERIODSECONDS`, and
  `"更新重启事件"` was added to the DeletePod action.
