<!-- Authored by Winston Cai. -->

# Review guidance

Read by the code reviewer and by humans. `AGENTS.md` says how to work; this
file says what a bad change looks like here and how findings must be written.
Failure in this repository is usually **drift**: the source, image, script,
fixture and claimed workload stop describing the same experiment.

## Output rules

- Open the summary with a count of findings ("3 findings" or "No findings").
  Say plainly when there is nothing to fix.
- Every finding needs a `file:line` and a concrete failing input or state, not a
  guess from a name. Drop findings you cannot demonstrate.
- At most 3 nits. Never report formatting that the repo's formatter handles.
- Label each finding Important, Nit, or Pre-existing.
- After three review rounds, ask the author to reconsider the approach. Later
  rounds review only the new diff and do not repeat earlier findings.
- When a bug is a pattern, list every instance, including the sibling GLM and
  V4.1 branches, so it is fixed once.

## Skip

`data/source-snapshots/`, `data/producer-artifacts/`, `**/runs/`, `raw/`,
`transfer/`, `*.patch`, `qualification.json`, `SHA256SUMS`, `.DS_Store`.
Run records are historical evidence; review the code that produced them.

## What failure looks like, by area

**Kernels and layouts** (`third_party/flashmla-v*`, `third_party/sglang-v*`): it races
or misaddresses.
- Unaligned base or stride (16-byte alignment of TMA/vector loads), unguarded
  rows, padding or duplicate selections, a missing barrier or stream hand-off.
- Raw logical positions confused with physical cache locations.
- Packed record offsets (328 bytes at 256 survivors; `count + 72` otherwise)
  that disagree between store, reconstruction and Direct.
- Numerical arithmetic that differs from the producer: raw FP32 compressor
  output, selection on raw magnitudes, then mask, RMSNorm and BF16 rounding.

**Launchers and preflight** (`scripts/modal/`, `scripts/local/`): it drifts.
- A fork HEAD that is not the parent pin, dirty fork files, or a cached binary
  not keyed by source and toolchain hashes.
- A preflight that repairs the checkout, or a check that passes when coverage
  is missing (skipped cases, no sanitizer summary, a timeout treated as success).
- Mocked helpers that never exercise the real launcher boundary.
- Image inputs edited during a build; a module name that shadows an installed
  package (`kernels.py` shadowed the HF `kernels` package).

**Benchmarks and analysis**: it lies.
- The measured path differs from production: dense readback absent from
  production, materialized versus persistent Native addressing, fixed rather
  than changing selections, RoPE tables too short, different arrival or cache
  history between arms.
- Cached scheduler metadata warmed and then mutated, or timing interpreted
  before numerical validity is checked.
- Historical ratios presented as paired measurements.

**Startup and configuration**: it accepts an invalid config or declares ready
early.
- A flag or architecture accepted by one validator and rejected by another
  (for example text-only flags and the architecture allowlist).
- Reading a removed attribute on a v0.5.22 object instead of its accessor.
- Treating a "ready" log line as health. Acceptance is `/health` 200 and a
  successful completion.
- A port left bound after teardown.

**Tests**: test the contract, not the shape. Flag tests that read source text or
ASTs, assert on mock call arguments copied from the implementation, or pin one
experiment's receipt instead of validating against a schema.

## Known incidents

Each has been fixed at least once; new code must not reintroduce it. When this
file changes, rerun the reviewer against these cases and keep the edit only if
it still finds them.

| Incident | Where | Class |
|---|---|---|
| Misaligned 128-bit survivor load: 328 B records alternate 16-byte and 8-byte row alignment | reported at `splitkv_mla.cuh:123` in the V4.1 review; current handling (aligned `uint4` path, else two 64-bit loads) at `third_party/flashmla-v0.5.22/csrc/kernels/sm90/decode/sparse/splitkv_mla.cuh:130-140` | kernel |
| FP4 fixture used page128; production uses 256-token SWA pages | V4.1 attempt ledger | benchmark |
| Tile stride used ceil-to-multiple instead of `ceil_div` (576 vs 18 tiles) | direct-rope-ablation diagnostic | benchmark |
| GPU torch/cuBLAS oracle work instrumented by compute-sanitizer | V4.1 attempt010, then GLM again | validation |
| Sibling fixture imports failing outside package-relative paths | V4.1 attempt007, then GLM validation | validation |
| `kernels.py` shadowed the HF `kernels` package | `raw/modal/prefill-read-20261018-113435/` | launcher |
| Launcher/runner edited during a Modal build | seven sessions, 10-13 to 10-26 | launcher |
| GLM missing from text-only architecture allowlist; wrapper ignored `language_model_only` | GLM startup | startup |
| `ForwardBatch.token_to_kv_pool` removed in v0.5.22 | GLM CUDA graph capture | startup |
| Cached scheduler metadata mutated after warmup | direct-rope-ablation harness | benchmark |
| Bootstrap server port still bound at calibration handoff | LSWB `PORT=30382` | startup |
