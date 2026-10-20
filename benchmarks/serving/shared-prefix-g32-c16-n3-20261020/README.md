# Shared-prefix G32/C16: Native / Packed-256, three runs per mode

Author: Winston Cai.


Same frozen generated workload, order and seed: 32 reusable prefixes, 40 requests per prefix, 1,280 requests total; nominal 128k prefix + 512-token question, exactly 256 generated tokens. Actual shared prefixes occupy 4,288,911 tokens. Native pool: 3,745,536 slots; Packed-256 pool: 4,536,832 slots (+21.13%). Same TP4, memory fraction and scheduler. Cache was successfully flushed before all 9 repeats; each mode reused one server. No warm-up requests.

Values are run-level mean ± sample SD (n=3). TTFT p90 is the mean of the three run p90s, not a pooled request percentile.

| Mode | Requests/s | Output tokens/s | TTFT p90 (s) | Mean TPOT (ms) | Mean E2E (s) | Device hit (%) | Uncached prompt (M tokens) |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 0.5567 ± 0.0049 | 142.51 ± 1.26 | 26.159 ± 1.834 | 64.25 ± 3.83 | 28.68 ± 0.25 | 82.99 ± 0.17 | 29.301 ± 0.294 |
| triton | 1.3476 ± 0.0080 | 344.99 ± 2.04 | 2.981 ± 0.279 | 29.40 ± 0.73 | 11.83 ± 0.07 | 95.32 ± 0.12 | 8.067 ± 0.210 |
| kernel | 1.3390 ± 0.0183 | 342.78 ± 4.69 | 4.801 ± 1.338 | 25.52 ± 1.86 | 11.90 ± 0.16 | 95.30 ± 0.10 | 8.101 ± 0.177 |

Packed Triton: **2.421× (+142.1%)** Native request throughput; Kernel: **2.405× (+140.5%)**. Uncached prompt tokens fall 72.5% and 72.4%, respectively. Independently summed scheduler new-prefill logs show the same mechanism (table below). All 11,520 requests complete without reported client errors and generate exactly 256 tokens. Triton versus Kernel request throughput differs by only 0.65%; these three runs do not establish a general implementation ranking.

**Interpretation:** reproducible capacity/retention benefit for this synthetic working set. The working set lies above Native capacity and below Packed capacity. This is a selected capacity-pressure regime, not evidence of a 2.4× gain on arbitrary workloads or coding-agent tasks. The old n=1 sweep is not pooled into n=3.


## Evidence and limitations

All 334 original transfer checksums passed before curation; this collection contains nine measurements. Exact completions/output lengths, ordered input lengths, cache totals and throughput arithmetic were reviewed. Scheduler new-prefill logs independently support the retention mechanism. No new GPU jobs were run. Fork/image identities and selected GPUQ jobs are retained in `manifest.json`.

Parallel jobs used disjoint TP4 allocations; fixed ordering, shared host and one dataset seed limit generalization. A resident server was reused per mode. This is a synthetic mechanism study, not closed-loop coding-agent task-quality evidence. The original [G16/G32/G40 n=1 sweep](../shared-prefix-c16-n1-20261019/README.md) remains separate.

## Reproduction

Use the recorded fork pins and serving settings, then SGLang's `generated-shared-prefix` dataset with the exact recorded tokenizer and cached input hash. The full client argument record is in each `repeatN/client.log`; the source/input identities are in `provenance/`. Required arguments: `--backend sglang --dataset-name generated-shared-prefix --gsp-num-groups 32 --gsp-prompts-per-group 40 --gsp-system-prompt-len 131072 --gsp-question-len 512 --gsp-output-len 256 --gsp-num-turns 1 --gsp-group-distribution uniform --max-concurrency 16 --num-prompts 1280 --seed 42 --warmup-requests 0 --flush-cache --cache-report`. Preserve the recorded request body and other client options. Run three times against the resident mode server, successfully flushing before every repeat. The original one-time launcher was transferred as a hash only; it is not a checked-in executable reproduction script.

## Artifact layout and provenance

`analysis.json` is the canonical per-run and aggregate record, including sample variance; `summary.csv` is its flat export. `manifest.json` maps every moved evidence file to its original transfer path, hash, original VDI identity and run ID. `provenance/gpuq-logs/` retains job logs. Repeat folders preserve original measurement filenames; mode/server or repeat/server folders retain server-side metadata and logs. Historical paths embedded in original artifacts are unchanged.

Curated SHA256SUMS cover every collection file except themselves. Original transfer/source hashes remain in the manifest. The combined transfer analysis is recoverable at `dc90c14`; these collection analyses select the same observations and statistics without pooling older measurements. No measured values were changed. Rollback: revert the curation commit to restore the original transfer layout.
