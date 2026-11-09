# Microbenchmarks

Author: Winston Cai.

| Collection | Operation | Baseline and comparison | Evidence |
|---|---|---|---|
| [dsv4-flash-decode-store](dsv4-flash-decode-store/README.md) | V4 decode and store | Native vs. Packed Triton and Packed Direct | 33 timing cases, 114 validity checks and 12 Nsight Compute captures |
| [dsv4-flash-prefill-read](dsv4-flash-prefill-read/README.md) | V4 prefill reconstruction | Native vs. Packed192 and Packed256 with the split reader | 36 timing cases and 108 profiled operations |
| [dsv4-flash-prefill-read-fused-rope](dsv4-flash-prefill-read-fused-rope/README.md) | V4 fused prefill reconstruction and RoPE | Native and the earlier split-reader measurements | 36 timing cases, 108 profiled operations and sanitizer results |
| [glm53-flash-packed](glm53-flash-packed/README.md) | GLM 5.3 decode and prefill attention | Untouched upstream FP8 TileLang for the 24-case H200 decode comparison. Fork Native for the separate Direct studies. | Paired timing CSVs, allocation and resource records, and qualification |
| [dsv41-flash-packed-fp4](dsv41-flash-packed-fp4/README.md) | V4.1 FP4 attention | Synchronization-corrected upstream Native for the ten-case H100 decode comparison, alongside dense FP4 and Packed Direct | Timing, allocation and source-bound qualification records |

Use lowercase, hyphenated `<model-or-integration>-<experiment>` names for
collections. Keep dates, hardware and protocol versions in run IDs and records
unless they distinguish collections. See the [benchmark index](../README.md)
for serving and replay evidence.

## Previous paths

| Previous directory | Current location |
|---|---|
| `decode-store-h100-three-mode-20261014` | [dsv4-flash-decode-store](dsv4-flash-decode-store/README.md) |
| `prefill-read-20261018` | [dsv4-flash-prefill-read](dsv4-flash-prefill-read/README.md) |
| `prefill-read-fused-rope-20261018` | [dsv4-flash-prefill-read-fused-rope](dsv4-flash-prefill-read-fused-rope/README.md) |
| `glm53-packed` | [glm53-flash-packed](glm53-flash-packed/README.md) |
| `v41-packed-fp4` | [dsv41-flash-packed-fp4](dsv41-flash-packed-fp4/README.md) |
| `v0522-qualification-20261023` | [Retired v0.5.22 qualification record](https://github.com/winstonxcai/flash-optimizations/blob/83530552f0e10bc08c03e1e69a928e3ccdc14d99/benchmarks/microbench/v0522-qualification-20261023/README.md) |

The fused-reader checksum manifest describes the original evidence, including
[the original README](https://github.com/winstonxcai/flash-optimizations/blob/83530552f0e10bc08c03e1e69a928e3ccdc14d99/benchmarks/microbench/prefill-read-fused-rope-20261018/README.md).
The moved README updates its link to the split-reader study.
