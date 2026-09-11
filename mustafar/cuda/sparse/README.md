# `sparse` — read the 328-byte packed records directly

Extension: `mustafar._sparse` · gate: `SGLANG_OPT_TOPMAG_SPARSE=1`

The other two packed modes still *reassemble* the record before attention —
`packed` into a Triton workspace, [`../fused/`](../fused/README.md) into the
584-byte native layout. This backend skips reassembly entirely. It reads the
packed record in the shared-memory fill, decodes FP8 E4M3 against the UE8M0 tile
scale there, rotates the 64-dim tail in-kernel, and runs QK^T and PV over the
same tile. Because `d_qk == d_v == 512`, one tile is both key and value — there
is no 584-byte row anywhere in this path, and no `flash_mla_*` call on a
reconstructed buffer.

## Two passes, softmax on the host

The softmax stays in PyTorch between two kernel launches, mirroring
dhjoo98/mustafar's own reference wiring (key formulation → host softmax → value
formulation). This module therefore exposes only the two SpMM-shaped passes:

```cpp
void sparse_scores(...);   // S = sm_scale * Q . K^T   -> [queries, heads, 512] f32
void sparse_output(...);   // O = P . V                -> [queries, heads, 512] bf16
```

wired together by `mustafar.sparse.sparse_forward`, which owns the softmax, its
log-sum-exp, and the merge with the native SWA+sink leg. `sparse_scores` returns
scores already scaled by `sm_scale`, matching `flash_mla_sparse_fwd`'s internal
scaling. `P` handed to `sparse_output` must already be the topk-masked softmax
(zero beyond each row's `topk_length`), so invalid slots contribute nothing.

Fusing the softmax into a single online-softmax kernel is a follow-on, worth
doing once this path is bit-comparable to the Triton reference.

## Enable

```bash
SGLANG_OPT_TOPMAG=1
KEEP=0.5
SGLANG_OPT_TOPMAG_PACKED=1
SGLANG_OPT_TOPMAG_SPARSE=1
```

Mutually exclusive with `SGLANG_OPT_TOPMAG_FUSED` — both claim the same c4 decode
call, so setting both is rejected rather than silently resolved. Applies to
**single-token decode only**: a multi-token extend, and any sm120 target, keep
the packed reconstruct path. `attn_sink` stays on the native SWA leg, so it is
counted exactly once across the split-softmax merge.

## Geometry

Pinned by [`../packed_abi.cuh`](../packed_abi.cuh), shared with `fused`:
`HEAD_DIM=512`, `NOPE_DIM=448`, `ROPE_DIM=64`, `BITMAP_WORDS=8`,
`PACKED_KEPT_VALUES=256`, 328 bytes/record, `TOPK=512`. The RoPE tail is bitmap
word 7 — `64*w..64*w+63` is exactly one word and one scale byte, so RoPE never
straddles a tile, and a half-pruned real/imag pair still rotates correctly.

## Build

```bash
cd mustafar/cuda
TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py build_ext --build-lib <repo> --build-temp /tmp/mustafar-build
```

Built as its own `CUDAExtension`, separate from `mustafar._fused`, so a failure
here never takes down the fallback. `sm_90` (not `9.0a`) matches the existing
cubin.

## Vendored files

`sparse_kernel.cu` and `bindings.cpp` are ours; the rest of the directory is
vendored. **[`NOTICE`](NOTICE) is the authoritative list** — in short, five files
come verbatim from dhjoo98/mustafar (itself Flash-LLM-derived, Apache-2.0), and
of those only `MMA_PTX.cuh` is `#include`d by a compiled translation unit. The
other four (`AsyncCopy_PTX.cuh`, `MatMulUtilities.cuh`, `Reduction_Kernel.cuh`,
`TilingConfig.h`) are kept for reference only. What is actually reused is the
MSB-first bitmap-scan convention and the `mma.m16n8k16` FP16/FP32-accumulate
wrapper; the Flash-LLM SpMM tiling, its `K_Global`-dependent word indexing, and
its per-tile `idx` offset table are not — they are tied to a 4096-wide
contraction this workload does not have.

[`PLAN.md`](PLAN.md) is the original approved kernel design — the record of what
was intended before any of it was built. Its identifiers and file paths have
been updated to the current naming, but its *design* is as first proposed, and
the implementation diverged from it in places (the vendored `SpMM_Kernel.cuh`
adaptation it describes was replaced by a purpose-built kernel, and the
single-call Python surface became the two-pass `sparse_forward`). Where the two
disagree, this README is authoritative.
