# Benchmark evidence

Author: Winston Cai.

Keep representative measurements with enough evidence to identify their workload,
implementation and limits. Native is the baseline; Packed Triton and Packed
Kernel are separate measured implementations. Job success and intact checksums
alone do not establish a repeatable speedup.

## Which reference to use

| Question | Collection | Evidence and appropriate use |
|---|---|---|
| Same-concurrency serving overhead | [C24 Native / Packed-256, n=3](lswb-replay/c24-native-packed256-n3-20261018/README.md) | **Preferred repeated C24 reference.** Nine fresh-server, unprofiled 1,200 s replays; three per mode. Recorded settings, image, dataset and harness identities match. All nine TTFT p90 values pass 10 s. Full loaded runtime identity was not captured. |
| Long-input serving at matched load and capacity | [Random 128k input / 8k output, n=1](serving/random-128k-8k-c24-capacity-n1-20261018/README.md) | **Single-run study.** Six cells: C24 and each mode's allocator ceiling. Available source identities and recorded settings match. Repeat before treating a gain as stable; allocator capacity is not an SLO ceiling. |
| Decode/store operation cost and GPU counters | [H100 three-mode decode/store](microbench/decode-store-h100-three-mode-20261014/README.md) | **Validated operation reference.** 33 timing cases, documented 114 passing GPU checks and 12 Compute reports. Timing samples are not independent serving runs. |
| Native / Packed192 / Packed256 prefill-read cost | [Original split reader](microbench/prefill-read-20261018/README.md) | **Model-free operation study.** 36 timing cases and 108 captured operations on shared inputs; Packed reconstruction adds cost. This is not a serving-speed measurement. |
| Fusing Packed prefill unpack and RoPE | [Fused-reader candidate](microbench/prefill-read-fused-rope-20261018/README.md) | **Validated opt-in experiment.** All-count parity, graph/sanitizer checks, 36 timing cases and 108 captures. Fusion removes part of the penalty but remains slower than Native. CSV export recovery and cross-run comparison limits are recorded. |
| Historical C24 execution profiles | [C24 three-mode timing and profiles, n=1](lswb-replay/c24-three-mode-profiles-n1-20261013/README.md) | **Historical diagnostic.** One unprofiled run per mode plus separate Systems captures. Pairing was not established; preserve event-loss and source-identity caveats. Prefer the repeated C24 collection for serving comparisons. |
| Packed store/decode interaction at C24 | [C24 Packed-256 ablation, n=1](lswb-replay/c24-packed256-store-decode-ablation-n1-20261014/README.md) | **Single-run diagnostic.** Four store/decode combinations. Session and source-plumbing differences limit causal conclusions. |
| Packed store/decode interaction at C28 | [C28 Packed-256 ablation, n=1](lswb-replay/c28-packed256-store-decode-ablation-n1-20261014/README.md) | **Single-run diagnostic.** Four sequential combinations; all four full-run TTFT p90 values exceed 10 s. Useful for investigation, not a passing concurrency reference. |

The C24 repeats and random-serving study used different images. Compare modes
within a collection; do not combine them into an identical-build experiment.
Existing additional studies retain their own documented acceptance status.

## Naming and layout

Collection names describe the workload or operation, comparison, repetition
count when applicable, and collection date. `n3` means three independent runs
per configuration; `n1` means one. Warm-up waves, measured request waves, graph
replays and profiler captures do not increase that repetition count.

```text
benchmarks/
├── README.md
├── lswb-replay/
│   ├── c24-native-packed256-n3-20261018/
│   ├── c24-three-mode-profiles-n1-20261013/
│   ├── c24-packed256-store-decode-ablation-n1-20261014/
│   └── c28-packed256-store-decode-ablation-n1-20261014/
├── serving/
│   └── random-128k-8k-c24-capacity-n1-20261018/
└── microbench/
    ├── decode-store-h100-three-mode-20261014/
    ├── prefill-read-20261018/
    └── prefill-read-fused-rope-20261018/
```

Original run IDs, source paths, comparison IDs and archive locations remain
unchanged inside collections. Manifests map the current curated location to
original evidence. Dates identify collections; some contain cross-date runs.
Renaming a collection does not revalidate or relabel its measurements.

## Evidence rules for future references

- State the question, fixed workload/settings, intended differences and accepted
  use. Keep source/binary hashes, image and checkpoint identities, seeds,
  commands, raw measurements and validity results; explicitly identify gaps.
- Check numerical validity before interpreting timing. Replay success alone
  does not establish numerical correctness or closed-loop coding-task quality.
- For repeated serving references, collect at least three independent runs per
  cell under the same protocol. Retain failures and report mean, sample SD and
  sample variance. Label averages of run-level percentiles; do not call them
  pooled percentiles or hide individual SLO misses behind an average.
- Compare equal concurrency for implementation overhead. Compare each mode's
  own capacity separately and report accompanying latency and cache behavior.
- Use unprofiled runs for performance. Profiles explain work; preserve replay,
  event-loss, coverage and source-identity limitations. A representative Compute
  screenshot is retained; a readable Systems screenshot remains outstanding.
- Keep JSON canonical and CSV as its flat export. Verify checksums before
  curation; update current paths and curated-file checksums after documentation
  changes, retaining original source hashes and run identities.
- Preserve measured artifacts. Add a new run directory when a protocol, source
  or configuration changes; do not overwrite a saved reference.
