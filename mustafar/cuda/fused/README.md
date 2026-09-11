# `fused` — reassemble, then run the stock FlashMLA kernel

Extension: `mustafar._fused` · gate: `SGLANG_OPT_TOPMAG_FUSED=1`

The packed store keeps the KV latent in 328-byte records; FlashMLA wants the
584-byte native layout. This backend closes that gap in one CUDA pass:

```text
packed values + bitmaps + scales
    -> packed_to_native_kernel
    -> temporary 584-byte FlashMLA-native rows (preallocated workspace)
    -> unchanged flash_mla_* on the reconstructed buffer
```

It is the *reassembly* backend: the attention math is still dense over all 512
coordinates, and still the stock `flash_mla_*` kernel. What it saves is the
Triton round-trip the `packed` mode pays — the reconstruction happens in-kernel,
in the shared-memory fill, rather than through a separate Triton launch that
materializes rows in a workspace.

## Surface

```cpp
void packed_to_native_cuda(values, bitmaps, scales, physical_indices,
                           raw_indices, topk_lengths, freq_pairs,
                           native_out, page_size, bytes_per_page);
```

Exposed to Python as `mustafar._fused.packed_to_native` (see
[`bindings.cpp`](bindings.cpp)). One entry point — the fused leg does not split
into passes, since the softmax and the attention itself are FlashMLA's.

The adapter uses four warps per block and one warp per selected row, launches on
PyTorch's current stream, mutates a preallocated native workspace, and performs
no tensor allocation and no host scalar read. Invalid and truncated rows are
fully zeroed.

## Enable

Requires all four:

```bash
SGLANG_OPT_TOPMAG=1
KEEP=0.5
SGLANG_OPT_TOPMAG_PACKED=1
SGLANG_OPT_TOPMAG_FUSED=1
```

The gate defaults off and fails loudly if the extension is unavailable rather
than silently falling back. It does not change persistent storage.

Mutually exclusive with [`../sparse/`](../sparse/README.md) — both claim the same
c4 decode call site, so setting both is rejected at config-validation time.

## Relations

- Uses the shared ABI constants from [`../packed_abi.cuh`](../packed_abi.cuh);
  no private copy.
- Produces the same output ABI as the `packed`/Triton path, so the two are
  drop-in alternatives behind the same patch anchor.
- Stays in the tree as the `_FUSED` fallback for the direct-read backend. It is
  deliberately frozen: [`../sparse/sparse_kernel.cu`](../sparse/sparse_kernel.cu)
  duplicates `decode_e4m3fn` rather than importing it, so that a change to the
  sparse path can never perturb this one.
