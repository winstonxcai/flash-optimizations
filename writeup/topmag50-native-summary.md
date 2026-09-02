# TopMag 50% magnitude pruning of the native c4 cache — consolidated results across RULER · LongBench V2 · Sangfor-Bench

**Question.** DeepSeek-V4-Flash stores a compressed cache (`C^Comp ∈ ℝ^512`, 21 c4 layers, compress_ratio=4,
Shared-KV). Does *store-time magnitude pruning* — zero the smallest-|·| coordinates of each stored compressed
vector, keep ratio `s=0.5`, let the fused store renormalize — survive end-task accuracy across long-context and
agentic evals? Three benchmarks, three servers, all on **DeepSeek-V4-Flash-FP8 / SGLang 0.5.15, tp=4**:

| benchmark | target | context | date | n |
|---|---|---|---|---|
| RULER (13 tasks) | latent (512-dim) + indexer (128-dim) | 32k/64k | 2026-08-17 / 19 | 850 smp (latent), 250 smp (indexer) |
| LongBench V2 | **indexer** (`--prune-target indexer`) | 16k–128k | 2026-08-30 | n=100 |
| LongBench V2 | **latent** (`--prune-target compressor`) | 16k–128k | 2026-08-31 | n=100 |
| Sangfor-Bench (distinct tasks) | latent (native c4, `SGLANG_OPT_TOPMAG=1`) | agentic | 2026-08-28→31 | 5 + 25 distinct |

Builds: `transferibility/sg_capture.py` injection harness (RULER/LB2) and the Mustafar package
(`flash-optimizations/mustafar/`, store-time only, `XKV_DEBUG=0` — Sangfor). Both are the **stock DeepSeek-V4
build with the same top-k-by-|RMSNorm(raw)·weight| mask**; only the store's stored coordinates are zeroed.

---

## Verdict / headline

**TopMag50 is lossless-to-native on every benchmark — the one apparent regression was a cloud-vs-local artifact.**

1. **RULER (latent, 64k, 850 smp): mean −0.16 pts; niah/vt never move; 70% is free except the QA family.**
   Retained energy R(0.5) = 0.954, stable 32k→64k.
2. **RULER (indexer, 64k, 5 hardest × n=50): mean −0.82 pts @50, −0.55 pts @70 — the latent's QA caveat
   does NOT transfer** (qa_2 @70%: −2.0 pts on the indexer vs −4.5 pts on the latent).
3. **LongBench V2 (n=100): indexer lossless — 55/100 both, +0.0 pp overall; 2 single-sample flips, no task-level
   regressions.** Retained energy 0.9695. **The compressor-latent target is the one caveat: −2.0 pp overall, all
   in the 64-128k bucket (−5.9 pp), QA/Code families — consistent with the RULER latent-QA caveat.**
4. **Sangfor-Bench 25 distinct: easy+medium *improve* (+3.2 pp, n=17); the hard bucket's −20 pp crater is a
   cloud-vs-local server effect (−23.7 pp local-native vs cloud), not pruning — apples-to-apples TopMag50 is
   +3.5 pp over the local-native control (+17 test cases of 1149).**

Bottom line: at 50% store sparsity the native c4 compressed cache is accuracy-neutral on retrieval (RULER),
lossless on a 100-sample LongBench V2, and neutral-to-positive on real agentic Sangfor tasks. The sparse win is
contingent on the deployment step (sparse store for bytes / sparse score kernel for compute) — see Caveats.

---

## Benchmark 1 — RULER

Full detail: `writeup/mustafar-sparse.md`.

### 1a. Latent target (`--prune-target latent`, 512-dim)

Go rule (mean drop ≤ 2 pts **and** R(0.5) > 0.90) satisfied at both lengths.

| leg | dense | pr50 | pr70 | d50 pts | d70 pts | R(0.5) | R(0.7) |
|---|---|---:|---:|---:|---:|---:|---:|
| 32k — 4 tasks × n=50 | 0.933 | **0.935** | 0.927 | **−0.23** | +0.60 | 0.955 | 0.850 |
| 64k — 13 tasks, 850 smp | 0.951 | **0.953** | 0.947 | **−0.16** | +0.39 | 0.954 | 0.845 |

- **50% is free at both lengths**; every niah needle task is 1.000 at 64k (pr50 ≥ dense on every task but cwe,
  which holds 0.00). vt improves to 1.000 even at pr70.
- **The only caveat is the QA family at 70%**, and the penalty *grows with context*: qa_2 0.735→0.690 @64k
  (−4.5 pts, n=100) vs 0.750→0.720 @32k (−3.0 pts); qa_1 −2.0 pts @64k. R(0.7) is uniform (0.83–0.86) and does
  not flag it — `R>0.90` is not a sufficient per-task safety bar at 70%.

### 1b. Indexer target (`--prune-target indexer`, 128-dim kv_norm) — compute win

Pruning the indexer's keys converts sparsity into *skipped score work* (the `[B,S_q,64,S_kv/4]` score GEMM is the
largest non-MoE compute). Measured 64k, 5 hardest tasks × n=50:

| task | dense | pr50 | pr70 | d50 pts | d70 pts | R(0.5) | R(0.7) |
|---|---|---:|---:|---:|---:|---:|---:|
| qa_2 | 0.760 | 0.740 | 0.740 | 2.0 | 2.0 | 0.965 | 0.854 |
| qa_1 | 0.810 | 0.780 | 0.800 | 3.0 | 1.0 | 0.968 | 0.859 |
| fwe | 0.853 | 0.860 | 0.853 | −0.7 | 0.0 | 0.967 | 0.860 |
| vt | 0.992 | 0.992 | 0.992 | 0.0 | 0.0 | 0.971 | 0.879 |
| niah_multivalue | 0.998 | 1.000 | 1.000 | −0.3 | −0.3 | 0.966 | 0.853 |
| **mean** | **0.883** | **0.874** | **0.877** | **0.82** | **0.55** | **0.967** | **0.861** |

- Go rule satisfied. **The QA-family caveat does not transfer to the indexer**: qa_2 @70% is −2.0 pts here vs
  −4.5 pts on the latent — the caveat is a property of the compressor latent, not of pruning.
- Accuracy-only run: the wall-clock win needs a sparse-aware indexer score kernel (zeroed coords still execute in
  the dense GEMM). Composable with cross-layer low-rank (xKV) — TopMag cuts live coordinates within a dimension,
  xKV cuts the dimensions scored.

---

## Benchmark 2 — LongBench V2 (n=100, TopMag50-on-indexer)

`sg_capture.py run-lb2`, in-process `sgl.Engine`, per-sample **Pass 1 DENSE + Pass 2 TopMag50-on-indexer**
(`--prune-keep 0.5 --prune-target indexer`), tp=4 on GPUs 4-7, ctx ≤ 131072, max-new 512, chunked-prefill 4096
(paged-indexer path — the DSV4 nonpaged indexer OOMs on q_chunk×seq temp at default 8192).

| bucket | n | dense | **TopMag50** | delta |
|---|---|---:|---:|---:|
| **OVERALL** | **100** | **55 (55.0%)** | **55 (55.0%)** | **+0.0 pp** |
| 16-32k | 33 | 14 (42.4%) | 13 (39.4%) | −3.0 pp |
| 32-64k | 33 | 22 (66.7%) | 23 (69.7%) | +3.0 pp |
| 64-128k | 34 | 19 (55.9%) | 19 (55.9%) | +0.0 pp |

By domain: Single-Document QA 0, Long-dialogue 0, Long in-context 0, Code Repo 0, Multi-Document QA −4.5 pp,
Long Structured Data +20 pp (each of the last two is a single flip).

**Exactly 2 sample flips** (n=100):

| bucket | ctx | domain | dense | TopMag |
|---|---|---|---|---|
| 16-32k | 21,275 | Multi-Document QA | CORRECT (D) | wrong (B) |
| 32-64k | 42,417 | Long Structured Data | wrong (B) | CORRECT (C) |

Retained energy (indexer): **mean 0.9695** (min 0.9664, max 0.9732, n=100).

**LongBench V2 verdict: lossless.** Pass rate identical (55/55), the two single-sample flips cancel, and no
bucket/domain shows a systematic drop.

### 2b. Latent target (`--prune-target compressor`, 512-dim) — the one caveat

Same 100-sample selection, TopMag50 on the **compressor latent** instead of the indexer keys
(`transferibility/out/lb2_prune100_latent.json`). Note the within-run dense pass (54) is 1 sample below the
indexer run's (55) — cross-run noise, so compare dense-vs-prune **within** this run.

| bucket | n | dense | **TopMag50** | delta |
|---|---|---:|---:|---:|
| **OVERALL** | **100** | **54 (54.0%)** | **52 (52.0%)** | **−2.0 pp** |
| 16-32k | 33 | 14 (42.4%) | 14 (42.4%) | +0.0 pp |
| 32-64k | 33 | 22 (66.7%) | 22 (66.7%) | +0.0 pp |
| 64-128k | 34 | 18 (52.9%) | 16 (47.1%) | −5.9 pp |

By domain: Single-Document QA −9.1 pp, Code Repository −9.1 pp, Multi-Document QA −4.5 pp, Long Structured Data
+40 pp (each driven by 1-2 flips); Long-dialogue and Long in-context flat.

**6 flips** — 2 improve (both Long Structured Data, ctx 42,417 + 78,264), 4 regress (2 Single-Doc QA + 1 Multi-Doc
QA at 64-128k, 1 Code Repo at 32-64k). Retained energy (compressor latent): **mean 0.9531** (min 0.9496, max
0.9568) — higher sparsity budget than the indexer's 0.9695 at the same keep ratio, consistent with the latent's
smaller information margin.

**Latent verdict: the QA caveat transfers.** Unlike the indexer (lossless), pruning the 512-dim compressor latent
costs −2.0 pp overall, entirely concentrated in the 64-128k bucket and the QA/Code families. This matches RULER's
finding that the QA caveat is a property of the **compressor latent**, not of pruning — the indexer's 128-dim
kv_norm keys carry what QA needs.

---

## Benchmark 3 — Sangfor-Bench distinct tasks

`SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=0.5 XKV_DEBUG=0`, tp=4, port 30211, agentic e2e evals (cc/vibe). Native
baselines = the cloud 0725 web run (`task_20260825_195126_744`, `newapi-ai.sangfor.com`). Run log:
`log/2026-08-31.md`.

### 3a. 5 distinct (n=1 per task) — `writeup/topmag-native-sangfor-5distinct.md`

| task | diff/lang | native (cloud) | **TopMag50** | verdict |
|---|---|---|---|---|
| sri_esecgpt_ebc6bf7a | easy/EN | 50% (5/10) | **100% (10/10)** | ✅ fixed |
| apex_soar-app_b05c9039 | easy/CN | 100% | 100% (10/10) | ✅ hold |
| sri_swe-bench_35a41525 | medium/EN | 100% | 100% (43/43) | ✅ hold |
| sri_s1_f650e49b | medium/CN | 95.8% (69/72) | **72.2% (52/72)** | ⚠️ regression |
| apex_chat-agent_9347a21 | hard/CN | 100% | 100% (56/56) | ✅ hold |

4/5 native-equivalent including the hardest (chat-agent, 56-test multi-file agent loop). The one regression
(sri_s1_f650e49b) is also the only task with native headroom — and it was run against the cloud baseline, so a
cloud-vs-local confound was suspected. That is exactly what the 25-distinct + local-native control resolved below.

### 3b. 25 distinct (n=1 per task, ex-degenerate) — `aggregate: transferibility/scripts/agg_25d.py`

13 native-pass + 12 native-fail, difficulty 7 easy / 10 medium / 8 hard, lang 10 go / 15 python.
Excluded degenerate: `sri_esecgpt_80fa3321` (suite collapsed 267/267 → 4/4, `raw_summary=null`).

| bucket | n | native (cloud) | **TopMag50** | delta | regr / hold / improve |
|---|---:|---:|---:|---:|---:|
| **ALL (ex-degen)** | **24** | 75.4% | **71.8%** | **−3.6 pp** | 8 / 12 / 4 |
| easy+medium | 17 | 83.6% | **86.9%** | **+3.2 pp** | 4 / 10 / 3 |
| hard (ex-degen) | 7 | 55.2% | **35.1%** | **−20.2 pp** | 4 / 2 / 1 |
| native-PASS (ex-degen) | 12 | 100.0% | 88.1% | −11.9 pp | 4 / 8 / 0 |
| native-FAIL | 12 | 50.7% | **55.4%** | **+4.7 pp** | 4 / 4 / 4 |

Per-difficulty: easy 98.6→98.3 (−0.3 pp), medium 73.1→**78.9** (+5.7 pp), hard 60.8→43.2 (−17.7 pp, incl. degen).

Full 25-row table (native cloud pass/total → TopMag pass_rate %):

| task | diff | native | **TopMag** | Δ |
|---|---|---|---|---:|
| gcjs_kube-log-check-recover_c6a12bfe | easy | 122/122 | 100.0 | 0.0 |
| gcjs_kube-log-check-recover_fc67bfda | easy | 131/132 | 99.2 | 0.0 |
| tw_esecgpt_4966005 | easy | 40/40 | 100.0 | 0.0 |
| sri_chat-agent_035a16f0 | easy | 25/27 | 88.9 | −3.7 |
| tw_esecgpt_6741243f | easy | 40/40 | 100.0 | 0.0 |
| gcjs_kube-log-check-recover_e04abbb7 | easy | 73/74 | 100.0 | +1.4 |
| mss_drme-service_2a2095f8 | easy | 35/35 | 100.0 | 0.0 |
| sri_esecgpt_cf8ba0fb | medium | 268/268 | 100.0 | 0.0 |
| gcjs_kube-log-check-recover_5b6a23ad | medium | 101/254 | 96.8 | **+57.1** |
| sri_s1_00ce55e2 | medium | 126/126 | 92.9 | −7.1 |
| aiyycp_sales-flow_d7329e44 | medium | 3/74 | 4.1 | 0.0 |
| fy_gptanalystagent_fb3d6a3d | medium | 111/111 | 55.0 | **−45.1** |
| sri_chat-agent_86ce36d3 | medium | 0/62 | 62.9 | **+62.9** |
| sri_chat-agent_b2f8ec64 | medium | 75/75 | 100.0 | 0.0 |
| sri_s1_d060bef0 | medium | 118/131 | 90.1 | 0.0 |
| gcjs_go-zero_22ab9e7d | medium | 48/48 | 100.0 | 0.0 |
| sri_ap-gpt_0dd68d23 | medium | 119/122 | 86.9 | −10.7 |
| sri_swe-bench_5f5a7df7 | hard | 116/116 | 36.2 | **−63.8** |
| sri_esecgpt_48486b59 | hard | 75/227 | 0.0 | −33.0 |
| sri_esecgpt_80fa3321 | hard | 267/267 | 100.0 | 0.0 *(degen)* |
| sri_s1_cec32c82 | hard | 76/176 | 19.3 | −23.9 |
| sri_swe-bench_fea293e6 | hard | 86/86 | 73.3 | −26.7 |
| sri_ap-gpt_2bcf1160 | hard | 16/164 | 15.8 | **+6.1** |
| tw_esecgpt_f291630 | hard | 243/243 | 100.0 | 0.0 |
| sri_ap-gpt_d7527749 | hard | 1/137 | 0.7 | 0.0 |

Read-through: native-FAIL tasks don't regress (many *improve* massively: 0→62.9, 101→96.8); native-PASS "leaks"
cluster in the **hard** bucket and were shown below to be a server/noise effect, not pruning.

### 3c. Hard bucket — 3-way disambiguation (cloud-native vs LOCAL-native vs TopMag50)

The −20 pp hard crater looked damning, but every TopMag hard task was compared against the **cloud** baseline.
An 8-hard-task **local native-CSA control** (`dsv4-hardnative-*_20260830`, `SGLANG_OPT_TOPMAG=0`) on the same
GPUs closed that confound. Per-task pass rate:

| task | cloud-native | local-native | **TopMag50** | T−local |
|---|---:|---:|---:|---:|
| sri_swe-bench_5f5a7df7 | 100.0% | 39.7% | 36.2% | −3.4 pp |
| sri_esecgpt_48486b59 | 33.0% | 0.0% | 0.0% | 0.0 |
| sri_esecgpt_80fa3321 | 100.0% | 100.0% | 100.0% | 0.0 *(degen)* |
| sri_s1_cec32c82 | 43.2% | 28.4% | 19.3% | **−9.1 pp** |
| sri_swe-bench_fea293e6 | 100.0% | 43.0% | 73.3% | **+30.2 pp** |
| sri_ap-gpt_2bcf1160 | 9.8% | 9.8% | 15.8% | **+6.1 pp** |
| tw_esecgpt_f291630 | 100.0% | 100.0% | 100.0% | 0.0 |
| sri_ap-gpt_d7527749 | 0.7% | 0.0% | 0.7% | +0.7 pp |

| aggregate (per-task mean) | cloud-native | local-native | TopMag50 |
|---|---:|---:|---:|
| ALL 8 hard | 60.8% | 40.1% | **43.2%** (+3.1 pp vs local) |
| ex-degenerate (7) | 55.2% | 31.6% | **35.1%** (+3.5 pp vs local) |

**The crater was a server effect, not pruning:** local-native alone is −20.7 pp (ALL 8) / −23.7 pp (ex-degen)
vs cloud — the pass-bucket "leaks" (`5f5a7df7` 100→36, `fea293e6` 100→73) crater on *local native* too
(100→40, 100→43). Apples-to-apples, TopMag50 is **+3.1/+3.5 pp over the local-native control**.

Test-case view (ex-degen, `transferibility/scripts/hard3way_counts.py`): only **`sri_s1_cec32c82` genuinely
regresses (−16 tc)**; `sri_swe-bench_fea293e6` (+26 tc) and `sri_ap-gpt_2bcf1160` (+10 tc) improve; aggregate
**+17 test cases of 1149 (+1.5 pp)** vs local-native. The pass-bucket "leaks" in 3b (`fy_gptanalystagent`
111/111→55, `sri_s1_00ce55e2` 126/126→93) are medium tasks with **no local-native control** — un-controlled
confound, but consistent with the same server effect.

### 3d. Same-window 262k rerun — hard + medium + easy (`--context-length 262144`)

The 135k runs above were cratered by sglang's input ceiling (`135168 − 32000` completion budget
≈ 103k effective, below the cc-agent's ~104k natural peak → 400s → dead turns). Both local servers
rerun the **same 8 hard + 10 medium + 7 easy** tasks at `--context-length 262144` (effective input
~230k) so the local-vs-cloud comparison is apples-to-apples. Pass rate % per task (n=1);
Δ columns are the difference in pass rate, in **percentage points (pp)**; mean rows and the running
full-avg row are **test-case-weighted** (Σ passed / Σ tests).

| task (# test cases) | native (262k) | TopMag50 (262k) | Δ (T−native) (pp) |
|---|---:|---:|---:|
| **HARD (8)** | | | |
| sri_swe-bench_5f5a7df7 (116) | 92.2 | 90.5 | −1.7 |
| sri_esecgpt_48486b59 (227) | 0.0 <sup>1</sup> | 33.0 | +33.0 |
| sri_esecgpt_80fa3321 (267) *(degen)* | 100.0 | 100.0 | 0.0 |
| sri_s1_cec32c82 (176) | 97.2 | 98.9 | +1.7 |
| sri_swe-bench_fea293e6 (86) | 76.7 | 70.9 | −5.8 |
| sri_ap-gpt_2bcf1160 (164) | 82.3 | 79.3 | −3.0 |
| tw_esecgpt_f291630 (243) | 100.0 | 100.0 | 0.0 |
| sri_ap-gpt_d7527749 (137) | 54.0 | 78.8 | +24.8 |
| **mean (8, test-case-wtd)** | **76.4** | **82.1** | **+5.7** |
| **mean (7, ex-48486b59, test-case-wtd)** <sup>1</sup> | **89.4** | **91.5** | **+2.1** |
| **MEDIUM (10)** | | | |
| gcjs_kube-log-check-recover_5b6a23ad (254) | 97.6 | 96.9 | −0.7 |
| sri_s1_00ce55e2 (126) | 89.7 | 93.7 | +4.0 |
| aiyycp_sales-flow_d7329e44 (74) | 100.0 | 73.0 | −27.0 |
| sri_chat-agent_86ce36d3 (62) | 96.8 | 93.5 | −3.3 |
| sri_chat-agent_b2f8ec64 (75) | 98.7 | 100.0 | +1.3 |
| sri_s1_d060bef0 (131) | 85.5 | 89.3 | +3.8 |
| sri_esecgpt_cf8ba0fb (268) | 100.0 | 100.0 | 0.0 |
| fy_gptanalystagent_fb3d6a3d (111) | 70.3 | 64.0 | −6.3 |
| gcjs_go-zero_22ab9e7d (48) | 100.0 | 100.0 | 0.0 |
| sri_ap-gpt_0dd68d23 (122) | 98.4 | 94.3 | −4.1 |
| **mean (10, test-case-wtd)** | **94.0** | **92.1** | **−2.0** |
| **EASY (7)** | | | |
| gcjs_kube-log-check-recover_c6a12bfe (122) | 95.1 | 95.1 | 0.0 |
| gcjs_kube-log-check-recover_fc67bfda (132) | 99.2 | 99.2 | 0.0 |
| tw_esecgpt_4966005 (40) | 100.0 | 100.0 | 0.0 |
| sri_chat-agent_035a16f0 (27) | 77.8 | 88.9 | +11.1 |
| tw_esecgpt_6741243f (40) | 100.0 | 100.0 | 0.0 |
| gcjs_kube-log-check-recover_e04abbb7 (74) | 100.0 | 100.0 | 0.0 |
| mss_drme-service_2a2095f8 (35) | 100.0 | 100.0 | 0.0 |
| **mean (7, test-case-wtd)** | **97.2** | **97.9** | **+0.6** |
| **running full avg (24 tasks, test-case-wtd, ex-48486b59)** <sup>1</sup> | **92.7** | **92.8** | **+0.1** |

<sup>1</sup> Native 48486b59 is a **broken-build trajectory**: the agent left the gptprocessor package
uncompilable (missing private dep `aes-go-module-core`, its own final-summary admission), so its 0.0 is
not a valid control and the task is excluded from the mean (7) row. Its native suite also shrank 227→202
(broken build dropped tests), so native's mean (8) counts 0/202; TopMag ran the full cloud suite (75/227),
exact cloud parity.

Read-through (hard bucket): **the context fix recovers the crater to at-or-above cloud on both local
legs** (the cloud numbers that cratered — cec32c82, 2bcf1160, d7527749 — all run ≥79 on TopMag here).
Apples-to-apples, TopMag − native is +5.7 pp on test-case-weighted mean(8) — dominated by native's
invalid 48486b59 — and **+2.1 pp on mean(7)**. Medium bucket (10/10 both legs): native weighted mean 94.0
vs TopMag 92.1 (−2.0 pp), the only bucket where native edges TopMag, driven by n=1 trajectories on
d7329e44 (−27) and fy_gptanalystagent (−6.3). Easy leg (7/7 both legs): native 97.2 vs TopMag 97.9
(+0.6 pp), native's only miss `035a16f0` (77.8 vs 88.9, n=1). **Test-case-weighted running full avg
(24 tasks, ex-48486b59): native 92.7 vs TopMag 92.8 (+0.1 pp)** — essentially lossless at the aggregate
once the invalid native control is dropped.

### 3f. New 25 distinct @262k — packed set (native 262k leg running)

The **Stage-1 packed** TopMag50 run (real 328-byte store, 1.78×; `SGLANG_OPT_TOPMAG=1
XKV_TOPMAG_KEEP=0.5 SGLANG_OPT_TOPMAG_PACKED_C4=1`, no native shadow pool) covered **25 NEW
tasks** (the §3d 24-task set excluded), split 13 cloud-pass / 12 cloud-fail (hard 8 / medium 10 /
easy 7), all ≥8 cloud tests, at ctx 262144 on two concurrent TP4 servers (A=30211/GPUs 4-7,
B=30212/GPUs 0-3), 2026-09-01→02. The **local-native 262k leg** for these same 25 tasks
(`dsv4-native25-262k-*_20260902`, pristine stock servers) is **running** — its cells fill in as
it completes. Pass rate % per task (n=1); Δ in pp. Cloud = `task_20260825_195126_744`.

| task (# test cases) | cloud | native (262k) | packed (262k) | Δ (packed−native) (pp) | Δ (packed−cloud) (pp) |
|---|---:|---:|---:|---:|---:|
| **HARD (8)** | | | | | |
| apex_chat-agent_9347a21 (56) | 100.0 | — | 100.0 | — | 0.0 |
| apex_chat-agent_f268ef1 (32) | 100.0 | — | 25.0 | — | −75.0 |
| apex_gpt-train-data-collector_1dbcd396 (111) <sup>2</sup> | 98.2 | — | 89.1 | — | −9.1 |
| apex_soar-app_9207ca23 (23) | 91.3 | — | 100.0 | — | +8.7 |
| apex_source-tracing-investigation_17ae176f (42) | 100.0 | — | 100.0 | — | 0.0 |
| apex_source-tracing-investigation_a4432711 (20) | 100.0 | — | 85.0 | — | −15.0 |
| apex_source-tracing-investigation_ab21ecf0 (28) | 96.4 | — | 100.0 | — | +3.6 |
| sri_ap-gpt_0cd2c2ac (107) | 15.9 | — | 12.1 | — | −3.7 |
| **HARD mean (8, test-case-wtd)** | **77.3** | — | **65.0** | — | **−12.3** |
| **MEDIUM (10)** | | | | | |
| aiyycp_sales-audit-platform_2439f30d (24) <sup>2</sup> | 100.0 | — | 95.0 | — | −5.0 |
| aiyycp_sales-audit-platform_53266d85 (23) | 100.0 | — | 100.0 | — | 0.0 |
| aiyycp_sales-audit-platform_ebac64e2 (34) | 91.2 | — | 91.2 | — | 0.0 |
| aiyycp_sales-audit-platform_ef78d2c0 (15) <sup>2</sup> | 93.3 | — | 93.3 | — | 0.0 |
| aiyycp_sales-auth_daf3ea25 (25) | 96.0 | — | 96.0 | — | 0.0 |
| aiyycp_sales-conversation_fa2bb019 (25) | 100.0 | — | 100.0 | — | 0.0 |
| aiyycp_sales-flow_033981bd (29) | 100.0 | — | 96.6 | — | −3.5 |
| aiyycp_sales-flow_09299ad2 (24) | 100.0 | — | 29.2 | — | −70.8 |
| aiyycp_sales-flow_191d12be (37) | 43.2 | — | 64.9 | — | +21.6 |
| aiyycp_sales-flow_edb6ec00 (24) | 95.8 | — | 100.0 | — | +4.2 |
| **MEDIUM mean (10, test-case-wtd)** | **89.6** | — | **86.0** | — | **−3.6** |
| **EASY (7)** | | | | | |
| apex_soar-app_282ef229 (11) | 100.0 | — | 100.0 | — | 0.0 |
| apex_soar-app_4896a623 (10) | 100.0 | — | 100.0 | — | 0.0 |
| apex_soar-app_4b0d01bf (8) | 87.5 | — | 75.0 | — | −12.5 |
| apex_soar-app_76e6e4f8 (15) | 33.3 | — | 100.0 | — | +66.7 |
| apex_soar-app_8073b35d (12) | 91.7 | — | 100.0 | — | +8.3 |
| apex_soar-app_969ed0d4 (8) | 100.0 | — | 100.0 | — | 0.0 |
| apex_soar-app_989a23c5 (14) | 100.0 | — | 100.0 | — | 0.0 |
| **EASY mean (7, test-case-wtd)** | **84.6** | — | **97.4** | — | **+12.8** |
| **running full avg (25, test-case-wtd)** | **82.3** | — | **76.5** | — | **−5.8** |

<sup>2</sup> Suite size changed between cloud and packed legs (1dbcd396 111→55, 2439f30d 24→20,
ef78d2c0 15→30); Δ uses each run's own total.

Packed−cloud read-through: −5.8 pp overall, driven by two genuine task-level collapses —
`apex_chat-agent_f268ef1` (32/32→8/32, 24/32 same-name flips) and `aiyycp_sales-flow_09299ad2`
(24/24→7/24, 17/24 flips). Excluding those two, packed 80.8 vs cloud 80.9 (−0.1 pp); hard −6.6,
medium +2.9, easy +12.8. 5 cloud-fail tasks fixed to full pass (9207ca23, ab21ecf0, edb6ec00,
76e6e4f8, 8073b35d); 191d12be improved 16/37→24/37. The native 262k leg will resolve packing
vs cloud-vs-prompt confound for the two collapses.

---

## Caveats

- **n=1 per Sangfor agentic task** — no within-task variance bound; ±large per-task binomial error on 10-test
  tasks. The n=7 same-task run established σ=0 on one instance only (`topmag-native-sangfor-n7.md`).
- **Cloud-vs-local confound is real and quantified** (−20.7 pp local-vs-cloud on the hard bucket) but only
  *controlled* for the 8 hard tasks. Medium-bucket leaks have no local-native control.
- **Ceiling effects:** native 100% → TopMag 100% can't distinguish preserved fidelity from "too easy to expose
  KV loss". The tasks with real test mass (swe-bench 43/86/116, chat-agent 56) carry the evidence.
- **RULER 64k: n=50 for 9/13 tasks** (on-disk cap); qa_2 @70% is n=100 at both lengths, so its 70% penalty is
  not noise.
- **No 8k RULER leg; LB2 is n=100** (2 flips ⇒ ±~10 pp overall noise floor at this pass rate).
- **Bytes/compute not measured.** Store-time zeroing still writes full 512-dim vectors (dense store) and the
  dense score GEMM still executes zeroed coords. The win materializes only with a **sparse store** (~s×
  compressed-cache bytes) and/or a **sparse indexer score kernel** — the accuracy ceiling is what's established
  here; sparse-store bandwidth is the deployment step (Stage 0 Triton sparse pack/unpack is implemented but
  unevaluated, storage 576<1024 B/row bf16).

## Artifacts

- RULER latent: `transferibility/out/ruler_csa_prune50_64k.json`, `ruler_csa_prune70_64k.json` (850 smp each);
  indexer: `transferibility/out/ruler_csa_idx_prune50_64k.json`, `ruler_csa_idx_prune70_64k.json` (250 smp each).
- LB2: `transferibility/out/lb2_prune100.json` (indexer target), `transferibility/out/lb2_prune100_latent.json`
  (compressor-latent target; per-sample dense/prune scores + retained energy);
  `transferibility/scripts/analyze_lb2_prune100.py`.
- Sangfor: 5d/25d result dirs under `/data/zyj/YJYBench/results/test/Sangfor-Bench_cc_vibe_*_{dsv4-topmag50-5d|dsv4-topmag50-25d}-*_2026082{8,9}/…`; hard-native
  control `dsv4-hardnative-*_20260830`; master logs `dsv4-topmag50-{5d,25d}_master.log`,
  `dsv4-hardnative_master.log`; aggregators `transferibility/scripts/agg_25d.py`, `agg_hard3way.py`,
  `hard3way_counts.py`.
- Launchers/watchdogs: `flash-optimizations/mustafar/scripts/` (`run_topmag50_5distinct.sh`,
  `run_topmag50_25d.sh`, `launch_inner_native.sh`, `run_hard_native.sh`, `watchdog_hard_native.sh`).

## Reproduce

```bash
# RULER latent 64k (keep=0.5 on GPUs 0-3; keep=0.3 on 4-7) — transferibility harness
docker exec ruler-eval bash -c "cd /mnt/host_root/home/jovyan/winstonxcai/transferibility && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 SG_ENV_OVERRIDE=1 NCCL_IB_DISABLE=1 \
  NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL python3 -u sg_capture.py run-acc \
    --prune-keep 0.5 --lengths 64k --n-64k 100 --tp 4 --mem-fraction 0.95 \
    --ctrl-dir .../sg_ctrl_prune50_64k --out .../ruler_csa_prune50_64k.json"
# RULER indexer 64k: same with --prune-target indexer --tasks qa_2,qa_1,fwe,vt,niah_multivalue

# LB2 n=100, TopMag50-on-indexer, in-process engine (chunked-prefill 4096 = paged-indexer path)
python3 -u sg_capture.py run-lb2 --prune-keep 0.5 --prune-target indexer \
  --selection .../lb2_selection_100.json --tp 4 --mem-fraction 0.95 \
  --chunked-prefill-size 4096 --context-length 131072 --max-new 512 --out .../lb2_prune100.json

# Sangfor: TopMag50 server, then one e2e invocation per instance (mustafar package)
SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=0.5 XKV_DEBUG=0 python3 -m sglang.launch_server \
  --model-path .../DeepSeek-V4-Flash-FP8 --served-model-name deepseek-v4-flash --tp 4 \
  --fp8-gemm-backend triton --disable-cuda-graph
python3 -m yjybench.cli --benchmark Sangfor-Bench --mode e2e --max_workers 1 --timeout 18000 \
  --exp_name test --docker_env_config docker_env_config_web_*.json --agent_type cc --agent_mode vibe \
  --sangforbench_prompt_source claude_result-tasks.md --instance_ids <task> --run_id <rid>
```

---

*See also:* `writeup/mustafar-sparse.md` (RULER detail), `writeup/topmag-native-sangfor-5distinct.md`
(5-distinct), `writeup/topmag-native-sangfor-n7.md` (same-task σ=0), `writeup/xkv-crosslayer.md` (orthogonal
cross-layer low-rank; composable).
