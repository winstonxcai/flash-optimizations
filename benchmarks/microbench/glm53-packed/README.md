# GLM-5.3-Flash Packed qualification

Author: Winston Cai.

Question: can eleven MLA layers use 320-byte rows while retaining Native FP8
TileLang arithmetic and unchanged KDA/indexer/tail reservations on H100?

Mechanism: select 256 raw latent coordinates before Native RMSNorm, persist
256 raw E4M3 bytes plus eight MSB-first uint64 bitmap words. Adapter materializes
selected rows in chunks of at most 128 queries; Direct expands within TileLang.
Native remains the default and rollback.

Source: managed worktree `glm53-packed`, branch `codex/glm53-packed`, starting
parent 9419d0e, SGLang 7856615, FlashMLA 3eb8594. Only v0.5.22 SGLang changes.

Acceptance: exact masks/bytes/reconstruction, production pool reconciliation,
attention atol=0.02/rtol=0.02, changing-input graph replay, and unexplained-error-free
memcheck/racecheck/synccheck. Native is FP8 TileLang on H100. No full weights,
serving, model quality or performance claims. Configuration metadata only.

Budget: one aggregate $15 ledger, CPU preparation <=$2, H100 worker <=$10,
$3 reserve; routine scheduling stops at conservative $12. One Function at a time,
no automatic retries, CPU4/RAM32GiB/H100, aggregate H100 <=120 minutes including
startup/cleanup (45+45+one corrective 30). Use caiw first after fresh spend, rate,
container and artifact checks. Release GPUs before diagnosis or builds.

The canonical JSON record in this directory owns source/artifact identities,
commands, coverage, costs and keep/discard/inconclusive decision. Missing gates
remain unverified. Existing `scripts/modal/kernels.py` owns remote orchestration;
GLM runtime and fixtures belong to the SGLang fork. No replacement harness.

The analytic ratio is 5995/3883 = 1.5439093484 at unchanged KDA reservation.
Physical tests must account for pages, indexer, KDA, tails, aliases and workspaces.
This ratio is not measured serving capacity.
