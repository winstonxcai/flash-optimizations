# mustafar CUDA backends

Two compiled backends for the TopMag packed store, each in its own subfolder
with its own README. They share one ABI and one build, and are mutually
exclusive at runtime because both claim the same c4 decode call site.

| | [`fused/`](fused/README.md) | [`sparse/`](sparse/README.md) |
|---|---|---|
| extension | `mustafar._fused` | `mustafar._sparse` |
| gate | `SGLANG_OPT_TOPMAG_FUSED=1` | `SGLANG_OPT_TOPMAG_SPARSE=1` |
| strategy | reassemble → 584-byte native rows → stock `flash_mla_*` | read the 328-byte record directly, both products from one tile |
| attention | dense over all 512 coords | QK^T and PV in-kernel, no reassembly |
| softmax | FlashMLA's own | host-side, between two passes (fusing is a follow-on) |
| scope | all decode shapes | single-token decode; multi-token extend and sm120 stay packed |

Pick `fused` to keep the native kernel's numerics exactly with a cheaper
reconstruction. Pick `sparse` to stop paying full-width dense attention on a
buffer that was just rebuilt — that is the one that turns the compression into
saved *compute* rather than saved memory.

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
cd mustafar/cuda
TORCH_CUDA_ARCH_LIST=9.0 python3 setup.py build_ext --build-lib <repo> --build-temp /tmp/mustafar-build
```

`include_dirs` is `cuda/` plus `cuda/sparse/`, so `fused.cu`'s
`#include "packed_abi.cuh"` and `sparse_kernel.cu`'s `"../packed_abi.cuh"` both
resolve unchanged. The two `bindings.cpp` files share a basename in different
directories, which is fine — `BuildExtension` derives object paths from the full
source path.

The two extensions build independently on purpose: a failure in one never takes
down the other, and `_fused` remains the `_FUSED` fallback.

## Validation funnel

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

On this node, the same GPU suites run under a GPUQ lease via
[`../scripts/local/kernel-run.sh`](../scripts/local/kernel-run.sh).
