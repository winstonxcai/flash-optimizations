# `fused` — reassemble, then run the stock FlashMLA kernel

Extension: `remnant._fused` · gate: `SGLANG_OPT_TOPMAG_FUSED=1`

The packed store keeps the KV latent in 328-byte records; FlashMLA wants the
584-byte native layout. This backend closes that gap in one CUDA pass. The
extension contains the original adapter and the retained optimized adapter:

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

Exposed to Python as `remnant._fused.packed_to_native` and
`remnant._fused.packed_to_native_optimized` (see
[`bindings.cpp`](bindings.cpp)). The fused leg does not split into passes, since
the softmax and the attention itself are FlashMLA's.

Both adapters use four warps per block and one warp per selected row, launch on
PyTorch's current stream, mutate a preallocated native workspace, and perform
no tensor allocation or host scalar read. Invalid and truncated rows are fully
zeroed. The optimized adapter stages each bitmap and packed-value row once,
reuses one set of bitmap prefixes, and emits aligned 32-bit output stores.

## Enable

Requires all four:

```bash
SGLANG_OPT_TOPMAG=1
KEEP=0.5
SGLANG_OPT_TOPMAG_PACKED=1
SGLANG_OPT_TOPMAG_FUSED=1
```

Add `SGLANG_OPT_TOPMAG_FUSED_OPTIMIZED=1` to select the optimized adapter. It is
explicitly opt-in; ordinary fused dispatch remains unchanged.

The gate defaults off and fails loudly if the extension is unavailable rather
than silently falling back. It does not change persistent storage.

Production direct decode is implemented in the FlashMLA fork; this extension
remains available for comparison only.

## Relations

- Uses the same packed ABI as the production FlashMLA path.
- Produces the same output ABI as the `packed`/Triton path, so the two are
  drop-in alternatives behind the same patch anchor.
- Stays in the tree as the `_FUSED` comparison implementation.
