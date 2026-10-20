# Overnight results relocated

Author: Winston Cai.

The complete measured payloads have been curated into:

- [Shared-prefix G32/C16, three modes, n=3](../../benchmarks/serving/shared-prefix-g32-c16-n3-20261020/README.md)
- [LongSWE Packed-256 C27, n=3](../../benchmarks/lswb-replay/c27-packed256-n3-20261020/README.md)
- [LongSWE Packed-256 C28, n=3](../../benchmarks/lswb-replay/c28-packed256-n3-20261020/README.md)

All 334 incoming checksums passed before migration. All 331 payload files were copied byte-for-byte and verified before their transfer copies were removed. The three root metadata files covered by incoming checksums (README, jobs and transfer manifest), the checksum file and the combined analysis are superseded by collection-specific records. Complete originals remain recoverable in Git at `dc90c14`; original transfer was `895b1a1`. Each collection manifest retains original paths/hashes, source identities, relevant jobs and original artifact metadata. No repeat or SLO failure was excluded.
