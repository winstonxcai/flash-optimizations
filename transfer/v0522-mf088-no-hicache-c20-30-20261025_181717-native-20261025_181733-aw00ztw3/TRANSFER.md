# Selected SLO sweep transfer

Author: Winston Cai.

Source: `results/lswb-replay/v0522-mf088-no-hicache-c20-30-20261025_181717-native-20261025_181733-aw00ztw3`.

Selected summaries, manifests, replay/startup/cleanup logs, per-point reports, GPU samples, collection settings and timing-validation reports are copied without modification and retain the original directory structure. `SHA256SUMS` records their source checksums.

This is not a full raw-run copy. Raw request/response streams, dataset manifests, active-request streams, server logs, server-info dumps and detailed request-metric/timing streams are excluded to keep prompts, payloads and bulk diagnostic data out of Git. Full originals remain at the source path. Source paths embedded in original manifests are intentionally unchanged.

Both runs completed C20–C30, one 1200-second measurement per point. All points fail the configured 99% response-success gate (`missing_output_event`); these artifacts do not establish fully SLO-qualified capacity.
