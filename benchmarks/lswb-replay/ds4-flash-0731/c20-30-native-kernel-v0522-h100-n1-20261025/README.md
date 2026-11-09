# V4-Flash-0731: v0.5.22 Native and Packed Kernel, C20–C30

Author: Winston Cai.

Two TP4 H100 sweeps, each with one 1,200-second measurement per concurrency
and one server boot per mode. HiCache was disabled. All points failed the
99% response-success gate due to `missing_output_event`; these observations do
not establish an SLO-qualified capacity.

The `native/` and `kernel/` directories retain the original selected packets,
including their manifests, per-point summaries and source `SHA256SUMS` files.
Both checksum files passed before curation. Original transfer folders:

- `transfer/v0522-mf088-no-hicache-c20-30-20261025_181717-native-20261025_181733-aw00ztw3`
- `transfer/v0522-mf088-no-hicache-c20-30-build-20261025_195128-kernel-20261025_200815-vc5was12`

The packets are selected artifacts, not full raw runs; see each `TRANSFER.md`
for exclusions, original VDI paths and provenance. Do not infer independent
server starts or production-quality results from the point count.
