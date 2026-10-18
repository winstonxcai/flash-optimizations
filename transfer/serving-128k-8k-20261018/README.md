# Random serving: 128k input / 8k output

Author: Winston Cai.

Question: measure Native, Packed-256 Triton and Packed-256 Kernel at matched C24 and at their allocator-derived concurrency ceilings.

One independent run per cell. Each run uses C warm-up requests followed by 3C measured requests. Three measured waves are not three repetitions.

analysis.json and summary.csv retain full-precision measurements. Per-run config, server settings, client source hashes, image identities and launcher hashes are preserved. Cross-run source consistency must be reviewed before making controlled comparison claims.

Compact warmup.json/measured.json omit generated text and per-token ITL arrays; aggregate statistics and other request details are retained. Original JSONL hashes are recorded in manifest.json. Full originals, logs and omitted arrays remain in VDI results/.

Allocator ceilings are not SLO limits. This synthetic workload does not establish coding-task quality or a repeatable throughput gain.
