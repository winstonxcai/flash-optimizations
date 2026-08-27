# Lowrank KV-decode: windowed self-fit — first clean Sangfor-Bench eval (19/29 vs native 29/29)

Date: 2026-08-27 · DeepSeek-V4-Flash (21 c4 latent layers, compress_ratio=4)
Build: `SGLANG_OPT_LOWRANK_KV_STORE=1` + `XKV_WINDOWED=1` + `XKV_WINDOW=4096` + `XKV_COEFF_DIM=192`
Layout: recent W=4096 c4 latents stored **native** (528 B/slot, basis_id=255); at each window
boundary a per-window rank-192 basis is SVD-fit on that window's **own** normed latents and the
window is re-encoded in place (192 fp8 coeffs + 3 u8 scale tiles + basis_id at slot offset 520).

## Result — single-instance smoke eval

Sangfor-Bench cc agent, 1 sampled instance (`gcjs_kube-log-check-recover_2cadb18b`, MaxInterval
restart + Chinese-log task), max_workers=1 → **19/29 run_agent tests (65.5%), resolved=false**.

Paired against the stored native CSA result for the **same instance** (`task_20260825_195126_744`):

| | native CSA | windowed self-fit |
|---|---|---|
| run_agent tests | **29/29 (100%)** | **19/29 (65.5%)** |
| resolved | true | false |

This is the **first lowrank eval instance that produced a coherent, task-appropriate patch** —
the agent read the right file, planned the correct fix, applied the core edit, and finished the
full ~32-min agent loop cleanly (server healthy throughout: 420 win_finalize / 84 early
store_error / 42 win_recon_no_basis, 0 server errors). Contrast: the fixed-basis build was
**10/10 garbage** (babble, end_turn, 0–1 tool calls) — see
`lowrank-sangfor-fixed-basis-fail.md`.

## Failure breakdown (10 of 29) — all root-caused from cc_claude_steps.jsonl

**8× B-series (sleep-path log):** `NeedRestartBroker` returned the **correct** value
(`restart=true`); the test asserts the *exact* sleep-path log string. The agent's live Edit#1
wrote `"距离上次重启时间达到SleepInterval，重启Pod"`; the test expects
`"距离上次重启时间间隔达到SleepInterval，重启Pod"` — one missing token (间隔). Exact-string
mismatch in a Chinese log message → 8 failures.

**2× DeletePod-log:** the agent's Edit#2 **failed to apply** (`String to replace not found in
file` — the case body no longer matched its stale context). The `"更新重启事件"` Info log was
never added.

**Passed (19):** all 9 MaxInterval A-tests, all 4 no-restart C-tests, NoRestartNoLeak, both
SetBrokerRestartStat, DeletePodUpdatesStat, both DefaultVariables. **The core MaxInterval restart
logic is functionally correct.**

## Interpretation

- **Coherence: fixed.** Windowed self-fit yields clean, on-task decode — the fatal deviation of
  the fixed-basis design (content-dependent latent subspace, drift during decode) is resolved by
  self-fitting each window's basis on its own latents (self-fit retention ≈ 1.0).
- **Fidelity: mild real gap.** On the identical task native solves at 100%, the windowed model
  drifted on one exact string and dropped one edit. Correct intent, sloppy exactness — the
  expected signature of compression-induced fidelity loss, not a coherence failure.
- **n=1.** No resolve-rate estimate; diagnostic value only. Do not read 65.5% vs 100% as a
  resolve-rate ranking.

## Caveats

- Single instance; ±huge CI on anything quantitative.
- The earlier OOM'd eval (11:16, NULL result) is not part of this comparison.
- Failed-edit (DeletePod) may be a harness-adjacent failure (stale context), not compression.
- This task is a near-minimal agentic workload (read one file, edit one function); does not test
  long-context retention under windowed compaction.

## Artifacts

- Eval results: `/data/zyj/YJYBench/results/test/Sangfor-Bench_cc_vibe_deepseek-v4-flash_dsv4-windowed-1_20260827/gcjs_kube-log-check-recover_2cadb18b/`
  (`test_result.json`, `test_output.log`, `cc_claude_steps.jsonl`)
- Native paired result: `results/test/Sangfor-Bench_cc_vibe_DeepSeek-V4-Flash-Local_task_20260825_195126_744/gcjs_kube-log-check-recover_2cadb18b/`
- Build: `transferibility/lowrank_store.py` (`_finalize_window` SVD, `_svd_lock`/`_svd_with_retry`,
  windowed store path, 528 B/slot pool sizing)
- Launcher: `transferibility/relaunch_windowed.sh 4096`
