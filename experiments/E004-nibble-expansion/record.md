# E004 — Four-coordinate packed-byte expansion

Status: candidate committed; H100 validity and performance gates pending.

## Hypothesis and isolated change

Starting from the E001 branch state, explicitly revert E003's asynchronous
survivor staging (`86be361`). E004 replaces per-coordinate shared byte gathers
with one four-coordinate helper: it computes four nibble ranks, reads two
aligned 32-bit shared words, gathers bytes with `__byte_perm`, and masks
pruned coordinates. Invalid and empty fragments return before survivor loads.
The record layout, global staging, prefix calculation, RoPE, barriers,
scheduling, and conversion remain unchanged. E002 is not included.

The source-level expected change is four byte loads to two word loads plus one
byte permutation for each non-empty four-coordinate fragment. This is only a
hypothesis; SASS and H100 measurements decide whether it helps.

## Source revisions

- FlashMLA branch: `remnant/flashmla-v0518`
- E001 rank-arithmetic base: `7576073b`
- E003 async-stage commit, explicitly reverted: `5dfb336`
- E003 revert: `86be361`
- E004 candidate: `1203a668864d1c2773bb686d78f6e6af48b549cb`
- SGLang bitmap-edge integration test: `45053e7bf74182f807ab59fef1a386cafa3631b6`
- Parent HEAD at implementation start: `028197c378b1a52fecbedd89458326a4458ec02a`

The parent Dockerfile and Modal runner have pre-existing local edits. The Modal
runner also has E004-only Direct selection and per-shape profile filenames.
Their worktree diff hashes will be recorded with the run artifacts; no unrelated
parent changes or `.DS_Store` files are included in the fork commits.

## CPU-only preflight

- Python AST parsing passed for the Modal runner and both Remnant decode tests.
- `git diff --check` passed in the parent, SGLang, and FlashMLA trees.
- Independent host-side verification passed for all 16 four-bit masks and all
  256 starting survivor ranks; each aligned eight-byte window stayed within the
  272-byte staged row.
- No CUDA compilation or GPU test has run locally.

## H100 sequence

Use the `poohthewinniechurchill` Modal profile. Its latest compute spend before
the run was `$27.63321362` (headroom `$1.36678638` to the `$29` threshold).
Run only SM90/H100, no model load, in this order:

1. Existing SGLang direct-vs-fused-adapter validity tests, including TopMag and
   bitmap edge patterns, H64/H128, B8/B16, K512/K317, and CUDA graph replay.
2. Compute Sanitizer memcheck and synccheck on the direct/graph tests. Any
   unresolved synchronization error stops the experiment before timing.
3. Exactly one candidate-only `microbench.py --path direct` pass: H64/H128 ×
   B8/B16 × K512/K317, 10 warmups, 100 graph replays, 30 rounds. Do not time
   Native or Adapter; reuse E001 files.
4. Candidate-only Nsight Systems and Compute on H64/H128 × B8/B16 at K512;
   save unique per-shape output files and inspect SASS for word loads/PRMT.

Compare candidate medians with the saved E001 Direct samples. Report those
cross-run ratios separately from per-run samples, and reuse E001 Adapter and
Native measurements for the remaining-gap table. No second timing pass.

## Decision gates

Retain as an accepted optimization only if validity, memcheck, and synccheck
pass; there are no new spills or occupancy loss; the single pass improves the
geometric-mean Direct median by at least 5% versus E001 Direct; and no measured
shape regresses by more than 2%. Report the remaining gap to the frozen E001
Adapter and Native paths. An incremental win does not claim the 2% Native goal.

## Artifacts

Pending Modal run. Record the exact commands, Modal job/result-volume paths,
JSON/CSV samples, sanitizer logs, profile reports, source SHAs, SASS evidence,
and keep/reject decision here after the gated run.
