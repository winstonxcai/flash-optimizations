# Shared-prefix C16: three modes, one run per cell

Author: Winston Cai.

## Protocol

Synthetic SGLang `bench_serving` generated-shared-prefix workload on
DeepSeek-V4-Flash-0731, TP4. **C16 is concurrent requests; G16/G32/G40 are
distinct reusable-prefix groups, not concurrency levels.** Each cell has 1,280
requests with exactly 256 output tokens. No request warm-up; flush radix cache
before each cell. Each mode keeps its server resident for G16 → G32 → G40.
Client seed 42; uniform shuffled groups, one turn, unlimited offered rate.

Prefix/question targets are 131,072/512 tokens. Actual prompts span
**134,414–134,693 tokens**; use recorded `input_lens`, not generic
`random_input_len` / `random_output_len` defaults, for this dataset.
Servers use mem-fraction 0.88, 8,192-token chunks, full decode graphs through
batch 136, prefill graphs disabled, FP8 KV and `flashinfer_mxfp4` MoE.

## Observations

| Groups | Mode | Requests/s | Output tok/s | Device hits | TTFT p90 (s) | Mean TPOT (ms) | Req/s vs Native |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | Native | 1.5584 | 398.96 | 96.12% | 2.93 | 25.97 | 1.000× |
| 16 | Packed Triton | 1.5544 | 397.91 | 96.37% | 2.49 | 27.76 | 0.997× |
| 16 | Packed Kernel | 1.5329 | 392.43 | 96.02% | 3.25 | 27.39 | 0.984× |
| 32 | Native | 0.5446 | 139.43 | 82.94% | 24.57 | 69.96 | 1.000× |
| 32 | Packed Triton | 1.2800 | 327.68 | 95.45% | 3.08 | 31.21 | 2.350× |
| 32 | Packed Kernel | 1.4173 | 362.82 | 95.49% | 2.92 | 26.80 | 2.602× |
| 40 | Native | 0.3014 | 77.15 | 65.30% | 55.37 | 99.12 | 1.000× |
| 40 | Packed Triton | 0.4406 | 112.80 | 77.89% | 29.03 | 86.63 | 1.462× |
| 40 | Packed Kernel | 0.4469 | 114.40 | 77.81% | 36.24 | 78.24 | 1.483× |

At G16, all modes report about 96% device hits and similar throughput.
At G32, Kernel observes **2.60× Native request/output throughput**, with
**95.49% versus 82.94% device hits**. At G40, its ratio is **1.48×**, with
**77.81% versus 65.30% hits**. These single-run observations are consistent with
a benefit from retaining more reusable prefixes; they do not isolate causality
or establish a repeatable production gain.

Recorded pools: **3,745,536 Native / 4,536,832 Packed full-token slots**.
The input audit records roughly 2.145M / 4.289M / 5.361M shared-prefix tokens
for G16/G32/G40, respectively. These describe the configured working sets,
not measurements proving simultaneous residency of every prefix.

## Evidence and limits

- All **46 transferred checksums**, nine completion validations, ordered input
  lengths across modes, exact output lengths, empty errors, cache totals,
  throughput arithmetic and mean TTFT passed local review.
- **n=1 per cell**: no uncertainty estimate, statistical equivalence, SLO-capacity
  or closed-loop task-quality claim. No new GPU run was performed for curation.
- Native ran on GPUs 0–3; Triton then Kernel on 4–7. Allocation intervals
  overlapped; host contention and mode order were not controlled.
- Source commits, image, launcher hashes and client arguments are retained.
  Full loaded-binary identity, source dirtiness and hardware telemetry are
  unavailable. The original launcher is hash-recorded but not in this packet;
  obtain `run.sh` from the VDI archive for an exact historical rerun.
- Mean TPOT/E2E are saved statistics, not independently recomputed from ITLs:
  generated text, ITLs and embedded server info were omitted before transfer.

## Files

- `analysis.json`: canonical reviewed metrics, definitions and per-cell checks.
- `summary.csv`: full-precision flat export of its nine cells.
- `{native,triton,kernel}/g{16,32,40}/`: unchanged `measured.json`,
  `server-config.json` and `validation.json`.
- `inputs.json`: original tokenizer/input-cache hashes and ordered input lengths;
  no prompts or generated text.
- `manifest.json`: original paths/run ID, transfer/source/image/job identities,
  command arguments, omitted-file hashes and retained-file checksums.
- `SHA256SUMS`: current curated files, excluding the checksum file itself.

Source transfer: `c25eccb`, `gsp-c16-sweep-20261019_112638-zf48HK`.
All 47 incoming files remain unchanged under ignored
`raw/gsp-c16-sweep-20261019_112638-zf48HK/`; full originals remain on the VDI.
Client/server logs are not duplicated in this curated collection.
