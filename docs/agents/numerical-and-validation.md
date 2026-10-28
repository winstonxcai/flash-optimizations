# Numerical and runtime contracts; validation and measurement

- Call the configurations **Native** and **Packed**. Native is the default;
  Packed is selected by `--dsv4-c4-cache-format remnant`. Do not revive the
  historical TopMag environment-variable bundle or stage-1 terminology.
- Default Packed retains 256 of 512 coordinates: 256 FP8 value bytes, a
  64-byte bitmap and eight scale bytes (**328 bytes**). Triton supports startup
  per-C4-layer counts 64..448 in steps of 64, using `count + 72` logical bytes
  in separate value/bitmap/scale planes; 512 survivors are invalid. Non-256
  profiles require Triton store/reconstruction and no HiCache. Optimized CUDA
  store/Direct and HiCache remain 256-only. Native C4 remains 584 logical bytes.
  Configurable counts are not adaptive runtime budgets; new GPU profiles need
  H100 validation before production use.
- Match the current producer's dtype and arithmetic. Its raw compressor
  output and norm weights are FP32. Selection uses raw magnitudes, followed by
  masking, RMSNorm, the prescribed BF16 rounding, and FP8 packing. Historical
  BF16-input store screens are not evidence of exact FP32 production parity.
- Keep raw logical positions distinct from physical cache locations. Preserve
  partial lengths, invalid/padded entries, duplicate selections, RoPE, and the
  sink. Verify page geometry from source/configuration rather than old fixtures.
- Kernel development currently targets H100/SM90. Do not build or run SM100
  tests unless the user explicitly expands the scope. Keep stock Native code
  and unrelated SGLang sparse-attention kernels intact.

Reuse existing fixtures, references, and runners instead of creating replacement
validity or benchmark harnesses. Select checks proportional to the requested
change; preserve independent references and established tolerances.

The entry points below describe the default v0.5.18 pair. The v0.5.22 pair uses
the corresponding versioned checkout and its adapted upstream interfaces; see
`third_party/README.md`. Its unchanged fork sources have passed core model-free
numerical, allocation and sanitizer qualification. HiCache GPU, full-model and
serving validation remain pending. Reorganized parent packaging has not been rebuilt.

| Purpose | Existing entry point |
|---|---|
| FlashMLA Direct output/LSE, edge cases, and graph replay | `third_party/flashmla-v0.5.18/tests/test_flash_mla_remnant_decoding.py`; shared `tests/lib.py` and `tests/ref.py` |
| SGLang integration | `third_party/sglang-v0.5.18/test/registered/attention/unittests/dsv4/test_remnant_*.py` |
| Low-level pack/unpack | `third_party/sglang-v0.5.18/python/sglang/test/kernels/deepseek_v4/test_remnant_*_kernel.py` |
| Startup cache-format validation | `third_party/sglang-v0.5.18/test/registered/unit/test_dsv4_c4_cache_format.py` |
| Native/Adapter/Direct decode timing | `third_party/sglang-v0.5.18/benchmark/remnant/microbench.py` and its README |
| Cached builds, validation, timing and kernel profiles | `scripts/modal/kernels.py` |

- Direct reads Packed records inside FlashMLA. Adapter reconstructs Native
  cache records before stock FlashMLA. Report their complete measured paths
  accurately, including combine/reconstruction where required by the protocol.
- Use the approved experiment's explicit matrix and acceptance rules;
  supplemental shapes do not replace required coverage. Do not turn a
  pruning-quality study into a kernel-speed gate.
- Check numerical validity before interpreting timing. Apply relevant graph,
  stream, and sanitizer coverage for CUDA changes. Missing instrumentation or
  an unexplained sanitizer failure does not count as passing validation.
- Respect the experiment's baseline policy. Reuse saved Native/Adapter results
  when requested; refresh controls only when the approved protocol or a
  separately identified harness correction requires it. Historical ratios are
  not paired measurements. Describe p95 of graph timing samples as such.
- Keep profiling separate from acceptance timing. Reuse prior profiles and
  collect new ones to answer a specific unresolved question. Check SASS,
  registers/spills, and actual instrumentation before attributing a speedup.
- For Nsight Systems (`nsys`), read
  `/Applications/NVIDIA Nsight Systems.app/Contents/skills/nsight-systems/SKILL.md`
  before use on this Mac. On other hosts, use the installed skill's location;
  report if unavailable rather than assuming this macOS path exists there.
- Microbenchmarks and frozen attention replay do not establish serving
  TPOT/TTFT, SLO capacity, or closed-loop agent task quality.
