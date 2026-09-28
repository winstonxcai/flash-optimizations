# Eliminating the TopMag sort from the packed store path

Scoping for removing `mustafar/reference.py::topmag_keep_mask`'s `torch.topk` from
the live packed decode path, and folding the keep-mask computation into
`_pack_fp8_kernel`.

## Where the 0.94 ms/step comes from

Decode profile, batch 24, TP-0, packed mode
(`mustafar/results/profile-20260915/packed/1789454191.7251894-TP-0-DECODE.trace.json.gz`),
per c4 layer per step, 21 c4 layers over 43:

| kernel | µs | launches | origin |
|---|---|---|---|
| `at::native::radixSortKVInPlace<2,-1,32,32,float,...>` | 25.01 | 420 | **PyTorch ATen** |
| `at::native::sbtopk::gatherTopK<float,unsigned int,2,...>` | 19.75 | 420 | **PyTorch ATen** |
| `_pack_fp8_kernel` (Triton) | 2.36 | 420 | ours |
| `_bf16_to_native_kernel` (load side) | 25.70 | 420 | ours |

44.76 µs × 21 = **0.94 ms/step** — 22% of packed's +4.34 ms/step decode penalty,
and **19× the cost of the pack kernel it feeds**.

Both sort kernels are stock PyTorch, reached from a single line we wrote:

```python
# mustafar/reference.py:35-40
mag = latent.abs().float()
_, idx = mag.topk(prune_k, dim=-1, largest=False)   # prune_k = 256, HEAD_DIM = 512
mask.scatter_(dim=-1, index=idx, value=False)
```

PyTorch selects its sort-based path rather than a selection because `prune_k` is
exactly half of `HEAD_DIM`; its small-sort heuristic engages when `k` is a large
fraction of `n`, so it sorts all 512 elements to return the smallest 256. The
cost is a shape coincidence, not a design choice. Nothing downstream consumes the
ordering — only the resulting mask, which is 256 bools.

## Which paths are live

One call site, in the fork's `_forward_compress_all_in_one`
(`sglang/srt/layers/attention/dsv4/compressor_v2.py:198-218`):

```python
if compress_ratio == 4 and not is_indexer and _sg_lr.topmag_enabled():
    keep_mask = _sg_lr.topmag_keep_mask(kv_compressed, _sg_lr.topmag_keep())
    if _sg_lr.packed_enabled():
        _sg_lr.validate_packed_static_config()
        _sg_lr.pack_rows(kv_compressed, keep_mask, norm.weight, ..., packed_pool, ...)
        return
    _sg_lr.topmag_zero_from_mask(kv_compressed, keep_mask)
```

Reached from both `forward_compress*` and `forward_unified:235`; both converge on
this one block. Mode matrix from `mustafar/scripts/local/serve.sh`:

| mode | TOPMAG | PACKED | branch taken |
|---|---|---|---|
| native | 0 | — | block skipped entirely |
| packed | 1 | 1 | `pack_rows` + `return` |
| optimized | 1 | 1 | `pack_rows` + `return` |

**In every mode we run, `packed_enabled()` is true and the block returns early.**
So `topmag_zero_from_mask` is dead on our serving path, and the mask's only live
consumer is `pack_rows`, whose only use of it is `_pack_fp8_kernel`.

`topmag_zero_from_mask` and `reference.topmag_keep_mask` must therefore stay —
they are the non-packed baseline and the numerical reference the tests compare
against — but neither is on the hot path.

## Design

### Constraint: the mask is needed before the norm

`_pack_fp8_kernel` ([mustafar/triton/kernels.py:46-55](../mustafar/triton/kernels.py#L46-L55))
already loads both the mask and the raw row, and applies the mask *before* the
RMS norm:

```python
bits = tl.load(keep_mask_ptr + row * HEAD_DIM + offs, ...).to(tl.int1)
x    = tl.load(latent_ptr + row * HEAD_DIM + offs, ...)
x    = tl.where(bits, x, 0.0).to(tl.float32)
inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + norm_eps)
```

The norm runs over kept coordinates only, and the mask must come from the
*unmodified* latent. So the threshold has to be computed in-program, from the row
the kernel already holds in registers, before the `tl.where`. No extra global
traffic is needed — only a reorder (load `x` first, derive `bits`, then proceed).

### Constraint: comparator fidelity is achievable

`topmag_keep_mask` compares `latent.abs().float()`. bf16→fp32 is lossless, so
loading the latent row and casting to fp32 reproduces the comparison inputs
**exactly**. Bit-exact tie behaviour is therefore reachable, not merely
approximated — which matters because the tests are adversarial about ties.

### Constraint: exact 256, ties by index

- `test_harness.py:220` — `mask.sum(-1) == rows * PACKED_KEPT_VALUES`
- `validity.py:206` — `torch.equal(decoded_mask, mask)`
- `validity.py:196-198` — constructs fully-tied rows (`x[0,:300]=0`, `x[1].fill_(1)`)
- `harness.py:345-352` — mask patterns deliberately "re-select *which* 256
  coordinates survive and never change the count"
- `pack_rows_ref` raises if any row keeps a different count

So the count is a hard invariant and the tie-heavy cases are deliberate.

### Algorithm: 40-round in-register radix select

For non-negative floats the IEEE bit pattern is monotonic in value, so an integer
compare on the magnitude bits *is* a magnitude compare. Build a composite key that
carries the index in the low bits:

```
key = (bitcast_fp32(|x|) & 0x7FFFFFFF) << 9 | lane    # 31 magnitude bits + 9 index bits
```

Then select the 256 smallest keys by radix descent, most-significant bit first:

```
remaining = 256
prefix = 0
for b in range(39, -1, -1):                      # fixed trip count
    zeros = sum(1 for lane if (key >> b) == (prefix << 1))   # lanes matching prefix with bit b = 0
    if zeros >= remaining:
        prefix = prefix << 1
    else:
        prefix = prefix << 1 | 1
        remaining -= zeros
bits = key <= (prefix << 1)                      # ties already resolved by index
```

Because the index is part of the key, the comparison is total: no two lanes share
a key, and equal magnitudes resolve in ascending lane order — exactly
`topk(largest=False)`'s index-order tie-break.

`<` vs `<=` at the end needs care: the final `remaining` may be 0, in which case
the boundary must exclude the prefix-equal lanes. Handle by descending one extra
step and using a strict key compare against the discovered threshold.

Cost estimate: 40 iterations, each a 512-lane masked reduction. `tl.sum` over 512
lanes with `num_warps=8` is a cross-warp tree using shared memory, so ~40 barriers
per program. Rough order 5-10 µs against 44.76 µs removed. **This is an estimate,
not a measurement** — it rides entirely inside a launch that already exists, so
the win is bounded below by zero extra launches either way.

### Fallback if the fixed 40 rounds measure badly

Two-level narrowing: a `tl.histogram` over the top 8 magnitude bits locates the
bucket containing the 256th element in ~1 reduction plus a 256-lane scan, then the
radix descent runs only over the remaining ~23 bits within that bucket. On the
pathological all-tied row every lane lands in one bucket and it degenerates to the
full descent, but that is a fixed cost, not a data-dependent loop, so it stays
graph-capture safe. Do not build this before measuring the simple version.

### Constraint: graph-capture safety

`reference.py:37-39` documents that `scatter_` replaced `mask[idx] = False`
*specifically* because the latter triggers a CPU→CUDA scalar copy that CUDA graph
capture rejects. The decode path is captured. The replacement must have no host
sync, no `.item()`, no dynamic trip counts, and no device→host readback. The
40-round loop above has a compile-time trip count, which is why it is preferred
over a convergence-based loop.

## Change surface

1. `mustafar/triton/kernels.py::_pack_fp8_kernel` — add the in-program selection;
   drop `keep_mask_ptr` (or make it nullable for the reference path).
2. `mustafar/packed.py::pack_rows` — drop the `keep_mask` parameter and its
   dtype/shape validation.
3. `mustafar/patches/` (fork edit, `compressor_v2.py`) — in the packed branch,
   skip `topmag_keep_mask`; keep the call for the `topmag_zero_from_mask`
   fallback. This touches a patch file, not the sglang version, so it is
   compatible with the frozen-version constraint. Verify the `## MUSTAFAR`
   anchor still applies after the edit.
4. `mustafar/tests/speed.py:333,467` and `harness.py:583,907` — callers of
   `pack_rows` need updating for the signature change.
5. New test: the in-kernel producer must match `reference.topmag_keep_mask`
   bit-for-bit on the tie-heavy patterns in `harness.py` (`IDENTITY`, `no_tail`,
   `half_pair`, and the all-zeros / all-ones rows from `validity.py`).

## Expected gain

0.94 ms/step. Packed decode penalty goes from +4.34 ms/step to ~+3.4. Combined
with the promoted optimized fused kernel on the load side (~0.62 ms), ~+2.8 ms/step
— packed mid-15 ms against native's 12.66.

Both halves are in `mustafar/` and its fork patch. Neither changes the sglang
version, the model, or the routing, so neither carries a quality risk.

## Open questions

- **Measured cost of the 40-round descent.** Unmeasured; the whole scoping rests
  on it being well under 44.76 µs. Prototype on one shape before committing.
- **Does anything compare a packed bitmap against `reference.topmag_keep_mask`
  after the change?** `validity.py:206` compares the decoded bitmap against the
  mask from the same call, so it survives a changed producer. But if any
  end-to-end validity check re-derives the mask independently, the producer must
  be bit-exact — which the composite-key design does provide, so this is a
  verification step rather than a design risk.
- **Whether the non-packed path is worth optimising at all.** It is unreachable in
  every mode we run. Leave it.
