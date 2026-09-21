# remnant CUDA backends

The retained fused backend is a comparison implementation for the TopMag
packed store. Production direct decode lives in the FlashMLA fork under
`third_party/flashmla/`.

| extension | `remnant._fused` |
| gate | `SGLANG_OPT_TOPMAG_FUSED=1` |
| strategy | reassemble → 584-byte native rows → stock `flash_mla_*` |
| scope | comparison only; production direct decode uses the FlashMLA fork |

Use `fused` only to compare reconstruction behavior and cost against the
production FlashMLA integration.

## The shared ABI

Both backends and the `packed` mode persist three page-major arrays per layer:

```text
values[num_pages, page_size, 256]  uint8
bitmaps[num_pages, page_size, 8]   uint64
scales[num_pages, page_size, 8]    uint8
```

Each logical row is 328 bytes. Bitmap word `w` covers coordinates
`64*w..64*w+63`; coordinate lane `l` uses bit `63-l`. The 256 FP8 E4M3 codes are
ordered by ascending original coordinate. Scale byte `w` is the UE8M0 scale for
original coordinates `64*w..64*w+63`.

Those constants live in exactly one place, [`packed_abi.cuh`](packed_abi.cuh),
`#include`d by both backends — no private copies. It stays at this level rather
than moving into either subfolder, since neither owns it.

## Build

`setup.py` stays here and builds both extensions from one command:

```bash
cd remnant/cuda
TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py build_ext --build-lib <repo> --build-temp /tmp/remnant-build
```

The extension is retained only as the `_FUSED` comparison path.

## Validation funnel

The production validation funnel is intentionally ordered by cost:

```bash
MODAL_PROFILE=your-profile modal run \
  remnant/scripts/modal/app.py::validate_flashmla_fork
MODAL_PROFILE=your-profile modal run \
  remnant/scripts/modal/app.py::validate_flashmla_direct_decode
# Run only after both gates pass, on an account holding the pinned 0731 volume.
MODAL_PROFILE=your-profile modal run \
  remnant/scripts/modal/app.py::download_model
MODAL_PROFILE=your-profile modal run --detach \
  remnant/scripts/modal/app.py::bench_serving --mode fused
```

Every run writes to a unique directory below `/results` in the existing
`remnant-stage2a-results` volume, so prior artifacts are not overwritten.
