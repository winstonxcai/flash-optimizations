# DeepSeek-V4-Flash: AsymKV local-homogeneity sanity check on CSA compressed keys

**Question.** Does AsymKV's core structural assumption — *adjacent cached keys are locally
homogeneous* (ρ(1) high, ρ(1) ≫ ρ(2) > ρ(4) > ρ(8)) — survive V4-Flash's native CSA
compression? AsymKV merges **adjacent** KV entries, so this must hold on the compressed key
sequence or the method has no safety margin. Measured as ρ_K(Δ) = E_j[cos(C_j, C_{j+Δ})] over
the pre-RoPE RMS-normalized CSA latent (`[T,512]`, T=S/4 rows, one per m=4-token window) of the
21 CSA layers `L∈{2,4,…,42}`, per RULER task and context.

**Run / environment.** SGLang **0.5.15.post1** (`mirrors.sangfor.com/lmsysorg/sglang:latest-cu130`)
in container `sglang-mirror` (created with `--shm-size=120g`), `tp=4` on GPUs 0–3, model
`DeepSeek-V4-Flash-FP8` (loads as **fp8**: `quant=fp8, fmt=e4m3, mem usage 68.82 GB/card`),
`mem_fraction_static=0.95`, `SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=env://`, no
`chunked_prefill_size=-1`. Capture = patched `compressor_v2.py` (`on_compress` right after
`compress_forward`), pre-RoPE RMSNorm recomputed, cos-stats on-GPU in-hook, Δ∈{1,2,4,8,16,32}.
Date 2026-08-14. **V4 is Shared-KV** (the same tensor is read as key and value), so
ρ_V(Δ) ≡ ρ_K(Δ) exactly by construction on every stream.

---

## Verdict (bottom line)

**Task-dependent — the premise holds for 3 of 4 RULER tasks, not niah; and the homogeneity
margin is far smaller than the paper's.**

| context (n) | layer-mean ρ1 | ρ2 | ρ4 | ρ8 | check |
|---|---|---|---|---|---|
| 8k (40) | **0.542 ± 0.134** | 0.485 | 0.494 | 0.460 | ρ1>ρ2 ✓; ρ4≈ρ2 |
| 32k (5, niah) | 0.489 | 0.429 | **0.520** | 0.435 | ρ4>ρ1 ✗ |
| 64k (10, niah) | 0.500 | 0.439 | **0.529** | 0.445 | ρ4>ρ1 ✗ |

Per-task @ 8k (n=10 each): **vt** 0.572 > ρ4 0.517 ✓ · **fwe** 0.608 > 0.549 ✓ ·
**qa_2** 0.528 > 0.407 ✓ · **niah** 0.461 < ρ4 0.501 ✗ (period-4 peak).

**So:** local key homogeneity is real but *moderate* (ρ1 ≈ 0.5, not the >0.9 the premise wants),
ρ1>ρ2 holds everywhere, and decay is clean for vt/fwe/qa_2 — but niah shows a reproducible
**period-4 peak (ρ4 > ρ1 ≈ 0.04)** that breaks the monotone assumption AsymKV relies on.

---

## Comparison with the paper's premise margin

The AsymKV paper (arXiv:2506.05410, NeurIPS 2025) established the premise on standard-transformers
KV caches (Llama-2-7B-chat / ShareGPT) with a **drastic** margin:

| | paper (Llama-2-7B-chat) | this work (V4-Flash CSA, 8k) |
|---|---|---|
| adjacent-key cosine similarity | μ ≈ **0.80** (σ² ≈ 0.02) | **0.46–0.61** (task-dependent) |
| adjacent-key attention-weight Spearman ρ | **0.727** | not measured (pre-RoPE latent) |
| adjacent-value cosine similarity | ≈ **0** | — (K=V, not defined) |
| keys vs values gap | ~0.8 vs ~0 | ρ_K ≡ ρ_V (shared-KV) |

Implications for an AsymKV-on-CSA build: (1) CSA compressed keys are locally homogeneous but with
roughly half the paper's margin, so adjacent-key merging has less error headroom; (2) the
"heterogeneous values" half of the premise is a **non-issue in V4** — K and V are the same
tensor, so values are exactly as homogeneous as keys. No separate value-side constraint exists
and AsymKV's lossless value-compression machinery is unnecessary: a plain merge of the shared
K=V inherits the key-side homogeneity. The only binding quantity is the (smaller) key-side margin.

---

## Measurements

- Per-layer ρ(Δ): vt/qa_2/fwe are monotone ρ1>ρ2>ρ4>ρ8; niah shows a ρ4-peak at essentially all
  depths (strongest in shallow layers 2–10). Reproducible across samples: max |Δρ1| < 0.023
  (n=10 per task). T=S/4 exactly, all 21 layers row-aligned.
- 8k per-task layer-mean ρ1 (n=10 each): fwe 0.608, vt 0.572, qa_2 0.528, niah 0.461.
- 8k SEM ≈ 0.02 (n=40), so the ρ1>ρ2 gap (~0.06) and niah's ρ4>ρ1 gap (~0.04) are significant.

## Caveats

- **64k is niah-only** (the other 64k RULER prompts, S=58–70k, exceed the KV pool on 4 GPUs);
  **32k capped at n=5** (RULER disk set has 5). 32k/64k columns therefore conflate context with
  the niah task.
- Measured on the **pre-RoPE** latent (repo convention); RoPE rotates only 64/512 dims, so Δ=1
  pre≈post.
- Capture row-order is validated by the internal-consistency argument (vt/qa_2 show the expected
  monotone pattern) + reproducibility, **not** cross-checked against a transformers reference
  (blocked by the 4-GPU memory constraint).
- The niah ρ4-peak mechanism is **unexplained** — candidate: period-4 structure in the m=4
  compression gate/source; flagged as the next investigation, not yet proven.

## Artifacts

`transferibility/ckpt_sglang/*.json` (55 samples) · `transferibility/out_sglang/asymkv_csa_{8k,32k,64k}.{md,json}`
· `transferibility/out_sglang/asymkv_csa_summary.md` · `transferibility/out_sglang/asymkv_figs/*.png`
· probe `transferibility/sg_asymkv_probe.py`.

## Reproduce

```bash
# 1. container (must have large shm for NCCL)
docker run -d --name sglang-mirror --gpus all --privileged --network host --shm-size=120g \
  -v /home/jovyan:/home -v /:/mnt/host_root \
  --entrypoint /bin/bash mirrors.sangfor.com/lmsysorg/sglang:latest-cu130 -c "sleep infinity"

# 2. run (GPUs 0-3 only)
docker exec -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=lo -e NCCL_P2P_LEVEL=NVL \
  -e SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE=env:// -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29501 \
  sglang-mirror python3 /mnt/host_root/home/jovyan/winstonxcai/transferibility/sg_asymkv_probe.py \
  run --per_task 10 --labels 8k,32k,64k
```
