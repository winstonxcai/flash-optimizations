# Sparse MLA kernel: consume the 328-byte packed layout directly

## Context

`mustafar/` stores the DSV4-Flash CSA latent C4 state packed — TopMag50 keeps 256
of 512 coords as FP8 E4M3 codes + 8×uint64 bitmap + 8 UE8M0 tile scales, 328 B/row
— but nothing consumes it sparsely. The only CUDA path today,
[mustafar/cuda/fused.cu](../fused/fused.cu)
(`packed_to_native_kernel`, `mustafar._fused`), **reassembles** each packed row
back into the 584-byte FlashMLA-native layout and hands that to `flash_mla_*`. So
we pay the packing cost and still pay full-width dense attention on a buffer we
just rebuilt — the compression buys memory, not compute.

`dhjoo98/mustafar` (cloned at
[mustafar/](../../), `main` @ `86fa14d`) is a Flash-LLM-derived
**batched sparse SpMM** with exactly the two primitives we need: a bitmap→dense
smem expansion (`SpMM_DecompressFromRegisterToShared`, `__clzll` scan) and a
double-buffered `mma.m16n16k16` FP16 / FP32-accumulate pipeline
(`PipelinedCoreComputations`). It stores FP16 NZ values with an explicit
per-tile `idx` offset table; we store FP8 + UE8M0 scales with an implicit global
popcount rank. Same MSB-first bitmap convention (`bit 63-lane`), same ascending
coordinate ordering.

**Goal:** a first real sparse MLA kernel that reads the 328-byte records directly
and produces the c4 attention leg's `(output, lse)`, which then merges with the
native SWA+sink leg — no reassembly, no `flash_mla` call on a reconstructed
buffer. `cuda/fused/fused.cu` is left in place as the `_FUSED` fallback so both can be
compared on the same run.

## Contract — what "correct" means

The new kernel must reproduce, for the c4 leg only:

```
flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512)
    where kv = unpack_gather_bf16(values, bitmaps, scales, physical, raw, topk_len, freqs)
```

`unpack_gather_bf16` is the existing Triton reference
([packed.py](../../packed.py#L224),
[triton/kernels.py:104-151](../../triton/kernels.py#L104-L151))
and has a pure-Torch twin, `unpack_rows_ref`
([reference.py:105](../../reference.py#L105)). Decoding it:

1. Decompress **all 512 coords** from the packed record: `fp8_e4m3 × exp2(scale_byte - 127)`,
   pruned lanes → 0. `rank = cumsum(kept) - 1` over the 512 coords, so a lane's
   value sits at `values[row*256 + rank]`.
2. Rotate the tail coords 448..511 in place by `position = raw * 4`
   ([triton/kernels.py:154-182](../../triton/kernels.py#L154-L182)).

Because both legs use the same 512-dim row, `d_qk = d_v = 512` and there is no
576-byte record anywhere in the sparse path. Snapshotted constants:
`HEAD_DIM=512`, `NOPE_DIM=448`, `ROPE_DIM=64`, `TILE_SIZE=64`, `BITMAP_WORDS=8`,
`PACKED_KEPT_VALUES=256`, `FP8_E4M3_MAX=448.0` ([config.py](../../config.py)).

Note the tail is **word 7** — bitmap word `w` covers coords `64w..64w+63`, so
`448..511` is exactly one bitmap word and one UE8M0 scale byte, and RoPE never
straddles a tile. TopMag may prune arbitrary tail coords; that is harmless, since
a zeroed lane rotates to zero and the pairwise real/imag mix handles a
half-pruned pair correctly.

## Design

### Staged — v1 then v2

`dhjoo98`'s own reference wiring does the softmax **in PyTorch** between two
kernel launches (`mustafar_key_formulation` → softmax → `mustafar_value_formulation`,
[models/llama_mustafar_kernel.py:274,314](../../models/llama_mustafar_kernel.py#L274)).
Neither vendored kernel fuses softmax; bolting online softmax into
`PipelinedCoreComputations` means rewriting the K-loop's accumulator rescaling.

- **v1 (this plan):** two SpMM-shaped kernels + a softmax/LSE pass, mirroring
  dhjoo98's structure. Delivers the full `(O_c4, lse_c4)` contract and the LSE
  merge. Smallest delta from vendored code — the point is to prove the direct
  328-byte read is correct before optimizing it.
- **v2 (follow-on, not planned here):** fuse the softmax into one kernel with an
  online-softmax accumulator. Only worth it once v1 is bit-comparable to the
  Triton reference.

### Vendored tree — `mustafar/cuda/sparse/`

Copy `dhjoo98/mustafar/kernel/csrc/*` wholesale, **keeping every Apache-2.0 /
Flash-LLM header**, plus a `NOTICE` naming dhjoo98/mustafar and Flash-LLM:

| vendored file | treatment |
|---|---|
| `AsyncCopy_PTX.cuh`, `MMA_PTX.cuh`, `Reduction_Kernel.cuh` | unchanged (`cp.async` + `mma.m16n8k16` FP16, both sm_90-valid) |
| `TilingConfig.h` | one edit: the per-kernel `TilingConfig` instantiations |
| `MatMulUtilities.cuh` | unchanged; `PipelinedCoreComputations` / `CopyTileFromGlobalToShared_X_64` as-is |
| `SpMM_Kernel.cuh` | the one substantive adaptation (below) |
| `SpMM_API.cu` | adapt the host-side tensor plumbing to our ABI |

New alongside it: `sparse.h` / `sparse_kernel.cu` (host API + softmax/LSE) and
`bindings.cpp` (pybind module `mustafar._sparse`),
`mustafar/sparse.py` (loader, mirroring [fused.py](../../fused.py)),
and a second `CUDAExtension` in [cuda/setup.py](../setup.py)
so `mustafar._fused` and `mustafar._sparse` build independently.
`cuda/packed_abi.cuh` is reused for the constants — do not duplicate them.

### The three real deltas in `SpMM_Kernel.cuh`

`SpMM_DecompressFromRegisterToShared` is kept: the `__clzll` / `pos1 << 6` MSB
scan is byte-identical in convention to ours and needs no change.

1. **Register load → FP8 + scale.** Replace `SpMM_CopyFromGlobalToReg`'s `uint4`
   (8 halves) NZ load with a `uint2` (8 bytes) FP8 load from
   `values[physical*256 + tile_rank + j]`. Decode `decode_e4m3fn(byte)` (already
   written in
   [fused.cu](../fused/fused.cu)) × `exp2(scale_byte - 127)`
   for tile word `w`'s scale byte. **This is done in the smem fill**, so the
   `mma.m16n16k16` pipeline downstream stays FP16 in / FP32 accumulate, untouched.
2. **Rank without an `idx` table.** Drop dhjoo98's `idx` / `NZ_Offset` global
   tensors. Our values for word `w` start at the popcount of words `0..w-1` of
   that row — computed in-register from the 8 bitmap words the kernel already
   loads. Removes two input tensors and a load per tile.
3. **RoPE for one tile.** Tile word 7 is the 64-dim tail and needs rotation by
   `position = raw * 4` after decompress. Reuse the `freq_pairs` layout already
   consumed by `fused.cu` and produced by `get_packed_freqs`; the 64 coords are
   one bitmap word and one scale byte, so this is uniform and boundary-aligned.

### Tiling

dhjoo98 hardcodes `N_Global = 8` (their head count) with
`TilingConfig<4,1,1,1>` for Key and `<2,1,1,1>` for Value. Our `h_q` is 64, so
widen the N dimension — e.g. `TilingConfig<1,4,1,0>` gives `TILE_M=64`,
`TILE_N=64`, `TILE_K=64`, `BLOCK_WARPS=4` / 128 threads, keeping their
`BLOCK_ROW_WARPS × BLOCK_WARPS = 4` assumption. Smem cost per K-tile stays ~8 KB
for A and ~8 KB for B (the densified A tile is 64×64 fp16, not 64×512), so the
existing double-buffered pipeline fits comfortably. `TILE_M` / the A-vs-B tile
assignment per leg are the tuning knobs; pin them by matching dhjoo98's reference
wiring, which already solves that mapping for the same shapes.

### Python surface

`mustafar/sparse.py` exposes `sparse_forward(...)`, doing the whole
two-legged computation in one call so the SGLang patch stays a single insertion:

```
o_swa, lse_swa = flash_mla_with_kvcache(q, swa_k_cache, swa_page_indices,
                                        swa_topk_lengths, attn_sink=attn_sink,
                                        extra_k_cache=None, ...)
o_c4,  lse_c4  = sparse.sparse_forward(q, values, bitmaps, scales,
                                            physical, raw, topk_lengths,
                                            freq_pairs, sm_scale, ...)
o = merge_lse(o_swa, lse_swa, o_c4, lse_c4)      # base-2 log-sum-exp
```

`flash_mla_with_kvcache` already returns `softmax_lse` float32, documented as
2-based, so the split-softmax merge is standard. **Implementation-time check:**
`attn_sink` must be counted exactly once across the two legs — assert it lands in
exactly one of them, since double-counting is silent.

### Gating

New `SGLANG_OPT_TOPMAG_SPARSE=1` in [config.py](../../config.py)
with a `sparse_enabled()` helper (mirroring `fused_enabled()`), validated
static config: requires `SGLANG_OPT_TOPMAG=1`, `KEEP=0.5`,
`SGLANG_OPT_TOPMAG_PACKED=1`, `topk == 512`, and rejects being set together with
`SGLANG_OPT_TOPMAG_FUSED` (both rewrite the same decode call — fail loudly rather
than silently pick one). `cuda/fused/fused.cu` and `_FUSED` are untouched.

[patches/attention.py](../../patches/attention.py) gets one
new branch at the c4 decode call site (the same anchor the packed path already
inserts at), routed through `sparse_forward`; the native path and the packed
reconstruct path are unchanged.

## Files

- **New:** `cuda/sparse/{NOTICE,TilingConfig.h,AsyncCopy_PTX.cuh,MMA_PTX.cuh,MatMulUtilities.cuh,Reduction_Kernel.cuh,SpMM_Kernel.cuh,SpMM_API.cu,sparse.h,sparse_kernel.cu,bindings.cpp}`,
  `mustafar/sparse.py`.
- **Modify:** `cuda/setup.py` (second `CUDAExtension`), `config.py` (gate + validation),
  `patches/attention.py` (one branch), `cuda/README.md`.
- **Do not touch:** `cuda/fused/fused.cu`, `cuda/fused/bindings.cpp`, `cuda/packed_abi.cuh`,
  everything under `triton/`.

## Verification

Extend the existing suites rather than adding a new one — they already have the
harness, the workloads and the tolerances:

- **Validity (the `sparse` leg in [tests/validity.py](../../tests/validity.py)):**
  new kernel vs native, over the existing 3-point context×batch grid plus the
  adversarial index/mask patterns. Row readouts and `(o, lse)` are compared
  separately, since a correct-lse/wrong-o split is the most likely failure and
  the merged output would hide it. The RoPE tail is checked separately from the
  NoPE coordinates, so an in-kernel RoPE bug is distinguishable from an FP8
  decode bug; see the tolerance section of [tests/README.md](../../tests/README.md)
  for which constant bounds which stage. The sparse leg has no dense row output,
  so its `rows` stage uses one-hot probe scores covering all 512 coordinates.
- **Speed:** the `sparse` column of the stage x leg table
  (`native | packed.bf16 | packed.native | fused | sparse`) — the direct-read leg
  must be compared against the reassemble-then-`flash_mla` path on identical
  input, which is the number that justifies the whole exercise. The comparison is
  reported, never asserted, until the softmax moves into the kernel (v2 below): as
  wired, v1 is expected to lose to the reassembling path.
- **CPU:** `python -m unittest mustafar.tests.test_patching` offline (passes
  today, 12 tests); the new gate's validation logic is pure Python and gets a
  CPU test alongside the existing backend-selection tests.
- **Build:** the vendored tree must compile under the `remnant` container's
  nvcc/torch before any GPU work.

**GPU use requires explicit permission** (`ask for permission for any gpu use`).
No GPU job is launched as part of implementation; the plan stops at "builds
clean + CPU tests green", and the GPU validity/speed run is a separate ask.

## Out of scope

- Fusing softmax into the kernel (v2, above).
- Touching the pack/unpack path, the native SWA/sink leg, or `cuda/fused/fused.cu`.
- Any change to the SGLang version or the pinned host tree.
- Landing the vendored tree into the outer repo's gitignore rules — `mustafar/`
  currently shows as untracked; worth a separate one-line decision, not part of
  this change.
