# E004 — Four-coordinate packed-byte expansion

Status: candidate compiled; numerical checks and memcheck passed; **not accepted**
because Direct synccheck reports divergent barriers. Timing and profiling were
correctly stopped at that gate.

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
- Parent HEAD used for Modal: `44358cc607d9fe922504cb34ed603694707c9f3d`

The parent Dockerfile and Modal runner have pre-existing local edits. The Modal
runner also has E004-only Direct selection and per-shape profile filenames. The
`scripts/modal/app.py` diff SHA-256 was `215c498e6bbc0cc663652a1b9a903747ff4431b2f25caac6bc7b1025a561c1b6`
for the first run and `169d6ad7c85dff45e6cdee3ba0cd5d515dc6bf6b1dbb1127db3995a9da107086`
for the isolated follow-up (which added the test selector and capped sanitizer
output). The `Dockerfile` diff SHA-256 was
`87eb53c6c7a807b78c36d0d68c60ee6d3efa134602b0018ecf73c9f848cdf7c2`. No
unrelated parent changes or `.DS_Store` files are included in the fork commits.

## CPU-only preflight

- Python AST parsing passed for the Modal runner and both Remnant decode tests.
- `git diff --check` passed in the parent, SGLang, and FlashMLA trees.
- Independent host-side verification passed for all 16 four-bit masks and all
  256 starting survivor ranks; each aligned eight-byte window stayed within the
  272-byte staged row.
- No CUDA compilation or GPU test has run locally.

## H100 sequence

Use the `poohthewinniechurchill` Modal profile. It was at `$27.63321362`
compute spend before the run and `$27.77617079` at the final billing check after
the two validity/sanitizer jobs (about `$0.143` additional compute; still under
`$29`). The image build took 649.52
seconds and compiled the extensions in the image; the H100 preflight confirmed
the cached extensions were used, with no runtime compilation. The run was
SM90/H100 only; no model was loaded and no SM100 tests ran.

Run sequence and outcome:

1. Existing SGLang direct-vs-fused-adapter validity tests, including TopMag and
   bitmap edge patterns, H64/H128, B8/B16, K512/K317, and CUDA graph replay.
2. Compute Sanitizer memcheck and synccheck on the direct/graph tests. Any
   unresolved synchronization error stops the experiment before timing.
3. Exactly one candidate-only `microbench.py --path direct` pass: H64/H128 ×
   B8/B16 × K512/K317, 10 warmups, 100 graph replays, 30 rounds. Do not time
   Native or Adapter; reuse E001 files.
4. Candidate-only Nsight Systems and Compute on H64/H128 × B8/B16 at K512;
   save unique per-shape output files and inspect SASS for word loads/PRMT.

The first Modal run's memcheck completed **20 passed**, covering all Direct vs
fused-adapter cases (both TopMag and edge masks, H64/H128, B8/B16, K512/K317)
and graph replay. Synccheck failed with 3,104 divergent-barrier reports; its
first report was in `KernelTemplate<MODEL1,64,false>` while the parity test
executed the Native/fused-adapter reference path.

To avoid attributing that result to the wrong path, a second run isolated only
`test_direct_decode_cuda_graph_replay` under synccheck. It also failed: 4,800
reports in `KernelTemplate<MODEL1,64,true>` (the Direct Remnant specialization),
with the host trace at the direct graph test and PC `+0x1e90`. That confirms the
Direct path itself still has an unresolved divergent-barrier report; it is not
waived as a Native-only or pre-existing issue. The corresponding Modal runs:
[full memcheck + synccheck](https://modal.com/apps/poohthewinniechurchill/main/ap-4PPhKoetKAfpH0W03CO7bN)
and [isolated Direct synccheck](https://modal.com/apps/poohthewinniechurchill/main/ap-42zkQCUcU96UoymD2ZLbT3).

Because synccheck failed, **no timing pass or Nsight profile was run**. No
Native/Adapter measurements were rerun; E001 data remains unchanged. E004's
performance and SASS/resource acceptance gates are unmeasured.

## Decision gates

E004 is **rejected for promotion / not accepted**: Direct synccheck failed, so
the experiment stopped before benchmarking and profiling. Keep the committed
candidate only as a reproducible experiment; do not update the parent production
submodule pin or push it. The performance gates remain unevaluated. A follow-up
must first isolate and fix the Direct barrier divergence without mixing in a
performance change, then rerun validity/synccheck before any timing.

## Artifacts

Exact Modal commands:

```sh
modal run scripts/modal/app.py::sanitize_flashmla_direct_decode \
  --sanitizer-tools memcheck,synccheck \
  --parent-sha 44358cc607d9fe922504cb34ed603694707c9f3d \
  --sglang-sha 45053e7bf74182f807ab59fef1a386cafa3631b6 \
  --flashmla-sha 1203a668864d1c2773bb686d78f6e6af48b549cb

modal run scripts/modal/app.py::sanitize_flashmla_direct_decode \
  --sanitizer-tools synccheck \
  --test-selector direct_decode_cuda_graph_replay \
  --parent-sha 44358cc607d9fe922504cb34ed603694707c9f3d \
  --sglang-sha 45053e7bf74182f807ab59fef1a386cafa3631b6 \
  --flashmla-sha 1203a668864d1c2773bb686d78f6e6af48b549cb
```

The Modal links above retain the complete sanitizer output. No benchmark JSON/CSV,
Nsight reports, or SASS dump exist for E004 because the hard synchronization
gate failed. The saved E001 comparison inputs remain in
`experiments/E001-rank-arithmetic/` and were not modified.
