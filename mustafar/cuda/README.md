# Packed C4 ABI and Stage 2 boundary

Stage 1 persists three page-major arrays per C4 layer:

```text
values[num_pages, page_size, 256]  uint8
bitmaps[num_pages, page_size, 8]   uint64
scales[num_pages, page_size, 8]    uint8
```

Each logical row is 328 bytes. Bitmap word `w` covers coordinates
`64*w..64*w+63`; coordinate lane `l` uses bit `63-l`. The 256 FP8 E4M3 codes
are ordered by ascending original coordinate. Scale byte `w` is the UE8M0
scale for original coordinates `64*w..64*w+63`.

Stage 1 reconstructs selected rows into an existing FlashMLA-compatible
workspace. No sparse CUDA attention consumer is compiled or activated here.
Stage 2 may include `packed_c4_abi.cuh` and consume these arrays directly, but
must preserve the exact bitmap, coordinate ordering, scale, raw-index, invalid
slot, and RoPE-position conventions validated by Stage 1.
