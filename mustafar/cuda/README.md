# Packed ABI and fused reconstruction

Both packed modes persist three page-major arrays per layer:

```text
values[num_pages, page_size, 256]  uint8
bitmaps[num_pages, page_size, 8]   uint64
scales[num_pages, page_size, 8]    uint8
```

Each logical row is 328 bytes. Bitmap word `w` covers coordinates
`64*w..64*w+63`; coordinate lane `l` uses bit `63-l`. The 256 FP8 E4M3 codes
are ordered by ascending original coordinate. Scale byte `w` is the UE8M0
scale for original coordinates `64*w..64*w+63`.

The `packed` mode reconstructs selected rows into an existing FlashMLA-compatible
workspace with the existing Triton path. `fused` provides one standalone CUDA
adapter with the same output ABI:

```text
packed values + bitmaps + scales
    -> packed_to_native
    -> temporary 584-byte FlashMLA-native rows
    -> unchanged FlashMLA
```

Enable it only with all four settings:

```bash
SGLANG_OPT_TOPMAG=1
KEEP=0.5
SGLANG_OPT_TOPMAG_PACKED=1
SGLANG_OPT_TOPMAG_FUSED=1
```

The `SGLANG_OPT_TOPMAG_FUSED` gate defaults off and fails if its extension is unavailable. It
does not change persistent storage. The adapter uses four warps per block and
one warp per selected row, launches on PyTorch's current stream, mutates a
preallocated native workspace, and performs no tensor allocation or host
scalar read. Invalid and truncated rows are fully zeroed.

The Modal funnel is intentionally ordered by cost:

```bash
MODAL_PROFILE=your-profile modal run \
  mustafar/scripts/modal/app.py::validate_fused
MODAL_PROFILE=your-profile modal run \
  mustafar/scripts/modal/app.py::bench_kernels --suite fused
# Run only after both gates pass, on an account holding the pinned 0731 volume.
MODAL_PROFILE=your-profile modal run \
  mustafar/scripts/modal/app.py::download_model
MODAL_PROFILE=your-profile modal run --detach \
  mustafar/scripts/modal/app.py::bench_serving --mode fused
```

Every run writes to a unique directory below `/results` in the existing
`mustafar-stage2a-results` volume, so prior artifacts are not overwritten.
