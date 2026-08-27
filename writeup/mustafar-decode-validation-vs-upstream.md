# Mustafar decode-path validation vs upstream (non-kernel) MUSTAFAR

Date: 2026-08-27 · Reference: `mustafar-upstream/` @ `86fa14d` (github.com/dhjoo98/mustafar)
Scope: **non-kernel** Python paths only — `models/llama_mustafar_Kt_Mag_Vt_Mag.py`
(+ `Kt_Mag_Vc_Mag`, `Kt_Mag_Vt_Opa`, `Kt_Opa_Vt_Mag`), **not** `llama_mustafar_kernel.py`.
Question: does our claim "TopMag 50 % on the native c4 latent needs **no decode change**"
hold against the reference implementation's decode?

## Verdict — structurally identical decode

Upstream non-kernel stores **pruned-but-dense** K/V in `past_key_value` and runs **stock
attention** over the zeros. We do the same against the native DeepSeek-V4 c4 pool. Neither
has a decode-time reconstruction, mask re-derivation, or sparse-kernel step. "No decode
change" is exactly what the reference does.

## Upstream non-kernel decode (the reference)

- KV cache = dense `[B, H, T, D]` tensors where pruned coords are plain `0.0`
  (`key_states_full` / `value_states_full`, carried in `past_key_value`, line 1091).
- Prune injection happens **at the cache**: `dh_prune_key` / `dh_prune_value`
  (lines 66–146) zero the smallest-|·| coords of a per-(head, token) vector in place.
  Prefill prunes all but the trailing `residual_length` tokens (lines 1039–1045 K,
  1081–1087 V); decode appends the new token dense, then prunes the oldest token as it
  leaves the residual window (lines 910–928 K, 1008–1026 V).
- Attention is **plain matmul** over the pruned cache (flash variant lines 873–874, 963,
  974):
  `attn_weights = Q · K_prunedᵀ / √d → softmax → attn_output = attn_weights · V_pruned`.
  Zeroed K coords contribute exactly `0` to the QK dot product; zeroed V coords
  contribute exactly `0` to the weighted sum. No bitmap, no mask, no recon — the zeros
  are just values.

## Our decode (native c4, TopMag 50 %)

- The **only** injection in the build is store-time: `compressor_v2.py::
  _forward_compress_all_in_one` calls `_sg_lr.maybe_prune(kv_compressed)` right before
  the stock fused `compress_norm_rope_store` (RMSNorm → RoPE → fp8 quant → paged store).
- Decode is 100 % stock sglang dsv4 flash-MLA over the native fp8 pages: dequantize with
  the stored per-tile scales, attend. Zeroed coords dequantize to exactly `0` (fp8 `0`
  × scale) and flow through attention exactly like upstream's dense zeros.
- No `_sg_lr.*` symbol exists on any decode path; the pool layout (584 B/token) and
  kernels are byte-identical to native.

## Where we deliberately differ (all store-side, none decode-side)

| | upstream MUSTAFAR (non-kernel) | ours (native c4) |
|---|---|---|
| Prune object | per-head, per-token K/V `[B,H,T,D]`, post-RoPE | the 512-dim c4 latent per compress-window (ratio=4 tokens pooled), pre-norm/rope/quant |
| Injection time | prefill + incremental during decode (sliding window) | once, at store time (decode never sees an unpruned latent) |
| Survivor scaling | none (pruned attention-ready states) | RMSNorm after pruning rescales survivors (deterministic, folded into stored fp8) |
| Selection | `kthvalue(k)` + `\|x\| ≥ thr` — off-by-one: keeps D−k+1/row, keeps all ties | `topk(k, largest=False)` — exactly k zeroed/row, deterministic tie-break |
| Sparsity | paper default 70 % (keep 30 %) | 50 % (keep 50 %) |

Selection semantics were cross-checked numerically (`mustafar/upstream_validate.py`, on
`archive/topmag-c4-indexer`): on tie-free data the common kept coords are **bit-identical**
and upstream keeps exactly +1 coord/row (the off-by-one); on ties upstream keeps all ties,
ours prunes exactly k. Survivors are never modified, only zeroed, in both.

## Conclusion

Our decode steps are the DeepSeek-V4-native analog of MUSTAFAR's non-kernel decode:
store pruned-but-dense KV, run stock attention, no reconstruction. The differences are
in *where* pruning is injected (c4 latent vs per-head K/V) and the exact selection
rule, not in the decode path. The 29/29 latent eval is the empirical confirmation that
this design preserves enough signal for an agentic workload.
