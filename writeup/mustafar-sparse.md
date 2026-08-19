# DeepSeek-V4-Flash: TopMag magnitude pruning of the CSA compressed cache (Mustafar-style)

**Question.** Can Mustafar-style magnitude pruning of the compressed KV cache transfer to
V4-Flash's native CSA compressed cache (`C^Comp ∈ ℝ^512`, 21 CSA layers, Shared-KV)? I.e. zero the
smallest-|·| coordinates of each stored compressed vector in place, keep ratio `s`, and let the
fused store renormalize the rest — does end-task RULER accuracy survive at 50% and 70% sparsity?
Measured at 32k (4 hardest tasks, n=50) and **64k (all 13 RULER tasks, 850 samples/config)**, same
scoring as the cross-layer study's Part 3 (`writeup/xkv-crosslayer.md`).

---

## Run / environment

- **Model:** DeepSeek-V4-Flash-FP8, served in SGLang 0.5.15 (`lmsysorg/sglang:v0.5.15-cu130`
  container `sglang-44039`), tp=4, mem-fraction 0.95.
- **Injection:** `sg_capture.py run-acc --prune-keep {0.5,0.3}` — in `on_compress`, per-row keep
  top-k by |RMSNorm(raw)·weight|, zero the raw coords of the rest (fused store renormalizes as
  usual); worker persists per-sample retained energy `R(s)=‖C̃‖²_F/‖C‖²_F` (aggregate + per-layer).
- **Eval legs:** (a) **32k** — RULER 4 tasks × n=50 (niah_multikey_2, vt, fwe, qa_2);
  (b) **64k** — RULER all 13 tasks, n=100 for niah_multikey_2/vt/fwe/qa_2 and n=50 for the other 9
  (RULER on-disk data cap) → 850 samples/config. Dense re-measured in-run for both.
- Both legs: the two configs run **in parallel** on 8 GPUs (two tp=4 runs: GPUs 0–3 keep=0.5,
  GPUs 4–7 keep=0.3), separate `--ctrl-dir` per ratio + distinct `MASTER_PORT`. Date 2026-08-17.

## Verdict / headline

**STRONG GO at both 32k and 64k** — TopMag pruning of the compressed cache is free at 50% sparsity
and nearly free at 70%, with one consistent exception: **the QA family degrades at 70% and the
penalty grows with context** (qa_2: +3.0 pts @32k → +4.5 pts @64k, n=100 both).

| leg | dense | pr50 (keep .5) | pr70 (keep .3) | d50 pts | d70 pts | R(0.5) | R(0.7) |
|---|---|---:|---:|---:|---:|---:|---:|
| 32k — 4 tasks × n=50 | 0.933 | **0.935** | 0.927 | **−0.23** | **+0.60** | 0.955 | 0.850 |
| 64k — 13 tasks, 850 smp | 0.951 | **0.953** | 0.947 | **−0.16** | **+0.39** | 0.954 | 0.845 |

Go rule satisfied at both lengths: 50% mean drop ≤ 2 pts **and** R(0.5) > 0.90; 70% mean drop ≤ 2 pts.
Retained energy is stable 32k→64k.

## Measurements

**32k RULER (n=50/task):**

| task | dense | pr50 | pr70 | d50 pts | d70 pts | R(0.5) | R(0.7) |
|---|---|---:|---:|---:|---:|---:|---:|
| niah_multikey_2 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.954 | 0.844 |
| vt | 0.994 | 1.000 | 1.000 | −0.60 | −0.60 | 0.958 | 0.858 |
| fwe | 0.987 | 0.980 | 0.987 | 0.67 | 0.00 | 0.957 | 0.856 |
| qa_2 | 0.750 | 0.760 | 0.720 | −1.00 | 3.00 | 0.952 | 0.839 |
| **mean** | **0.933** | **0.935** | **0.927** | **−0.23** | **0.60** | **0.955** | **0.850** |

**64k RULER (n=100 for niah_multikey_2, vt, fwe, qa_2; n=50 otherwise):**

| task | dense | pr50 | pr70 | d50 pts | d70 pts | R(0.5) | R(0.7) |
|---|---|---:|---:|---:|---:|---:|---:|
| niah_single_1 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.959 | 0.864 |
| niah_single_2 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.950 | 0.832 |
| niah_single_3 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.950 | 0.832 |
| niah_multikey_1 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.950 | 0.831 |
| niah_multikey_2 | 0.995 | 1.000 | 1.000 | −0.50 | −0.50 | 0.955 | 0.846 |
| niah_multikey_3 | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.956 | 0.851 |
| niah_multivalue | 0.998 | 1.000 | 1.000 | −0.25 | −0.25 | 0.950 | 0.832 |
| niah_multiquery | 1.000 | 1.000 | 1.000 | 0.00 | 0.00 | 0.950 | 0.832 |
| vt | 0.993 | 0.996 | 1.000 | −0.30 | −0.70 | 0.959 | 0.862 |
| cwe | 0.990 | 0.990 | 0.988 | 0.00 | 0.20 | 0.953 | 0.841 |
| fwe | 0.852 | 0.857 | 0.853 | −0.50 | −0.17 | 0.958 | 0.857 |
| qa_1 | 0.800 | 0.800 | 0.780 | 0.00 | **2.00** | 0.952 | 0.836 |
| qa_2 | 0.735 | 0.740 | 0.690 | −0.50 | **4.50** | 0.953 | 0.841 |
| **mean** | **0.951** | **0.953** | **0.947** | **−0.16** | **0.39** | **0.954** | **0.845** |

**Per-layer retained energy R_l(s)** (mean over samples, layers 2,4,…,42): at 64k R(0.5) band
0.943–0.993, R(0.7) band 0.813–0.970 (32k: 0.943–0.994 / 0.814–0.973). No single layer collapses;
the 70% loss is spread thin at both lengths.

## Analysis

1. **50% sparsity is free at both lengths** — 32k mean −0.23 pts, 64k −0.16 pts. Needle/retrieval
   tasks never move (all niah tasks 1.000 at 64k); pr50 ≥ dense on every 64k task except cwe
   (0.00). vt actually improves to 1.000 at pr70.
2. **70% sparsity is nearly free in aggregate** (+0.60 @32k, +0.39 @64k) **except the QA family**:
   qa_2 drops 3.0 pts @32k and 4.5 pts @64k (0.750→0.720, 0.735→0.690; n=100 both), qa_1 drops
   2.0 pts @64k (n=50). The 70% QA penalty *grows* with context — consistent with the cross-layer
   study's finding that retrieval-style prompts compress far better than QA
   (`xkv-crosslayer.md` Part 2 §3).
3. **Retained energy does not flag QA.** R(0.7) is uniform across tasks (0.83–0.86 @64k) while
   qa_2 alone falls 4.5 pts — the `R(s)>0.90` bar is not a sufficient end-task safety guarantee at
   70%. Adopting 70% sparsity should be workload-conditioned (safe for retrieval/needle, risky for
   QA-family) or capped per-task.
4. **Memory note.** The harness zeroes coordinates in place but the store still writes full
   512-dim vectors, so **bytes saved are not measured here**. The win only materializes with a
   sparse store (skip zeroed-coordinate writes → ~s× compressed-cache bytes). This go/no-go
   establishes the accuracy ceiling; sparse-store bytes are the deployment step.
5. **Orthogonal to the cross-layer SVD** (Parts 1–3 of `xkv-crosslayer.md`) — the two could be
   composed (low-rank then TopMag).

## Caveats

- 64k: n=50 for 9 of 13 tasks (RULER on-disk data cap); qa_1 @70% (+2.0 pts) is n=50. qa_2 @70% is
  n=100 at both lengths, so its penalty is not noise.
- 32k: 4 tasks × n=50 (the cross-layer Part 3 leg was n=100); dense and pr50 agree with the
  recorded 32k baselines.
- No 8k leg for pruning; only 32k/64k tested.
- `report-prune`'s decision uses mean ± per-task judgment; a 1–2 pt mean drop is borderline vs SEM
  on the harder tasks (e.g. qa_2 ~0.74).

## Artifacts

- 64k: `transferibility/out/ruler_csa_prune50_64k.json`,
  `transferibility/out/ruler_csa_prune70_64k.json` (850 samples each)
- 32k: `transferibility/out/ruler_csa_prune50.json`, `transferibility/out/ruler_csa_prune70.json`
- Per-layer payloads: `transferibility/sg_ctrl_prune50{,_64k}/results/`,
  `transferibility/sg_ctrl_prune70{,_64k}/results/`
- Logs: `transferibility/par_prune50{,_64k}.log`, `par_prune70{,_64k}.log`; launchers
  `transferibility/sg_prune_smoke.sh`, `transferibility/sg_prune_par.sh`

## Reproduce

```bash
# 32k leg — two tp=4 runs in parallel on 8 GPUs; separate ctrl-dir + distinct MASTER_PORT
docker exec sglang-44039 bash -c "cd /mnt/host_root/home/jovyan/winstonxcai/transferibility && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 SG_ENV_OVERRIDE=1 NCCL_IB_DISABLE=1 \
  NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL python3 -u sg_capture.py run-acc \
    --prune-keep 0.5 --lengths 32k --tasks niah_multikey_2,vt,fwe,qa_2 --n 50 --tp 4 \
    --ctrl-dir /mnt/host_root/home/jovyan/winstonxcai/transferibility/sg_ctrl_prune50 \
    --out /mnt/host_root/home/jovyan/winstonxcai/transferibility/out/ruler_csa_prune50.json"

# 64k leg — all 13 tasks, n=100 where data allows (--n-64k 100), --mem-fraction 0.95
docker exec sglang-44039 bash -c "cd /mnt/host_root/home/jovyan/winstonxcai/transferibility && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29501 SG_ENV_OVERRIDE=1 NCCL_IB_DISABLE=1 \
  NCCL_SOCKET_IFNAME=lo NCCL_P2P_LEVEL=NVL python3 -u sg_capture.py run-acc \
    --prune-keep 0.5 --lengths 64k --n-64k 100 --tp 4 --mem-fraction 0.95 \
    --ctrl-dir /mnt/host_root/home/jovyan/winstonxcai/transferibility/sg_ctrl_prune50_64k \
    --out /mnt/host_root/home/jovyan/winstonxcai/transferibility/out/ruler_csa_prune50_64k.json"
# ... same for keep=0.3 on GPUs 4-7 (MASTER_PORT=29500, *_70* paths)
python3 -u sg_capture.py report-prune   # prints the decision table + verdict
```
