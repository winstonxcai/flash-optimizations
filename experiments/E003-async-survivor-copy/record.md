# E003: asynchronous packed survivor staging

## Hypothesis

The Direct producer synchronously loads each packed survivor slice into
registers, then writes it into shared memory, before beginning bitmap and scale
processing. Asynchronous global-to-shared copies may overlap that transfer with
the existing metadata work and remove the register-to-shared copy chain.

## Change

In `third_party/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`, the
survivor staging in `prepare_remnant_row` now uses CUTLASS `cp.async` helpers.
Aligned chunks use 16-byte copies; rows offset by 8 bytes use aligned 8-byte
copies. Valid lanes issue copies while the original E001 bitmap-prefix and
scale work runs. Copy-group commit and wait instructions are warp-uniform;
invalid lanes issue no memory copies and wait on an empty group. The first
draft put commit/wait behind the lane-varying validity predicate; initial
synccheck exposed divergent synchronization, so that was corrected before
continuing. The record format, rank arithmetic, decode, RoPE, and scheduling
are unchanged.

## Source state

- Parent HEAD: `407fcd5b9004bc2409de13eec7399757abbcd1c1`
- Parent tracked diff SHA-256 at run time: `9b89d58b9433a32d24667b509a39a2c90dc19e80839e778277b37cb973beb911`
- SGLang HEAD: `0b2006980628f915f8a0b54f28669be9160ed7de`
- FlashMLA E001 base commit: `7576073b6ce5b209e4fa8306f08aba02683f0f62`
- FlashMLA E003 source diff SHA-256: `3d7a01de19b0a1d327183c3ad5532023b06bdd27977ddd263cb41ba48a2f6a47`
- FlashMLA E003 code commit: `5dfb3366519366c615bb99d85c9a11fb048ba502` (`perf: stage Remnant survivors asynchronously`)
- FlashMLA branch: `remnant/flashmla-v0518`
- No full model or SM100 path is part of this experiment.

## Validation and benchmark

The initial Modal image build for the first draft (`4913bcbf…`) succeeded on
SM90 as image `im-3TzEeolsqBmbocqk5wukvW`. Its
existing direct-versus-adapter validity cases passed under Compute Sanitizer
memcheck (8 passed; zero errors). Synccheck then found divergent async
synchronization (3,772 reports) because the first draft guarded group commit
and wait with lane-varying validity. The source was corrected to make both
instructions warp-uniform while invalid lanes issue no memory copies.

The corrected source (`3d7a01de…`, now committed as `5dfb3366519366c615bb99d85c9a11fb048ba502`) has **not** been
rebuilt or tested: the retry was stopped
by the `winstoncai233` Modal account's billing-cycle spend limit during image
construction (`im-kEjt4o6c90Jb99DtCaCiKi`); it was stopped before compilation.
Therefore corrected-source validity, memcheck, synccheck,
benchmark, and profile are all pending. No timing or profile result is claimed.
Modal runs: first draft `ap-9dqCTKWA0UfP73MNSOE3jE3`; corrected retry
`ap-axZwTbRTzTie2zdtanQlAs`.

After spend is available, use the existing SGLang suite
`test/registered/attention/unittests/dsv4/test_remnant_flashmla_direct.py`,
memcheck and synccheck on its direct-versus-adapter cases, and
`benchmark/remnant/microbench.py` with 10 warmups, 100 graph replays, 30 rounds
over H64/H128 × B8/B16 × K512/K317. Reuse E001 Native and Direct baselines.
Then compare E003 Direct against the saved E001 Direct profile; check copy/wait
instructions in SASS, kernel time, instructions, shared-memory
traffic/conflicts, global traffic, registers, spills, occupancy, and warp
stalls. Keep only if validity and sanitizers pass, no spills appear, Direct
improves at least 5% geometrically, and neither timing pass regresses any shape
by more than 2%.

## Results

Blocked before corrected-source validation; performance decision pending.
