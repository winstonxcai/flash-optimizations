<!-- Authored by Winston Cai. -->

# Repository guidance

This repository develops Remnant: training-free sparsification and persistent
storage of pretrained models' compressed attention states. DeepSeek-V4-Flash-0731
is the current integration model. The parent repository contains experiment
records, serving scripts, Modal runners, and the report; production runtime and
kernel code live in two independent Git submodules.

## Where to work

| Area | Location |
|---|---|
| SGLang runtime, packing, allocator, and dispatch | `third_party/sglang/` |
| FlashMLA direct decode kernel | `third_party/flashmla/` |
| Fork file-change map | `third_party/changelog.md` |
| Local serving and evaluation | `scripts/local/` |
| Modal workflows and local CPU checks | `scripts/modal/serve.py`, `kernels.py`, and `tests.py`; read their module docstrings |
| Kernel experiments and decisions | `experiments/E###*/`; read the active experiment's plan/goal and record |
| Maintained pruning data/training/evaluation | `training/`; read `training/README.md` |
| Historical attention-pruning experiments | `docs/attention-aware-latent-kv-compression.md`; ignored `experiments/attention-pruning/`; historical Modal runner `capture.py` |
| Project findings and paper narrative | `docs/report.md` |
| Paper figures | `docs/figures/` |

Read applicable nested `AGENTS.md` files before editing their directories.
Check current source and active experiment records when older documentation
disagrees with them. Do not treat a recorded branch head as the current checkout.

## Git and source ownership

- Inspect the parent and relevant submodules' branch, status, and diff before
  editing. Preserve existing staged, unstaged, and untracked work.
- The established branches are parent `remnant`, SGLang `remnant/v0.5.18`, and
  FlashMLA `remnant/flashmla-v0518`. Do not develop on `main` or switch branches
  over dirty work. `origin/...` names are remote references, not local branches.
- Use the checked-out forks in `third_party/` as the default source for edits and
  builds. Do not substitute a GitHub commit, fetch a newer upstream version, or
  run the legacy runtime-patching/backup workflow without task authorization.
- When the active experiment needs an isolated candidate, use its documented
  worktree or source-snapshot workflow. Record the source path and overrides;
  avoid unexplained `tmp/` copies. Preserve the shared fork checkout.
- A submodule pin records a commit, not its uncommitted files. When asked to
  commit/push, handle each affected fork on its own branch, then update the
  parent pins. Push referenced fork commits before the parent. Stage only the
  requested work; do not sweep unrelated dirty files into a commit.
- Update `third_party/changelog.md` when the set of fork files changed by
  Remnant changes. Preserve upstream licenses and attribution. Add or retain
  `Author: Winston Cai.` in the top docstring/comment for Remnant changes, using
  the file's existing language and format.

## Code quality and engineering practices

Aim for code that another engineer can read, reproduce, and safely extend.
Correctness, clear ownership, and a small maintenance burden come first;
performance improvements must have measured evidence.

- Follow the surrounding SGLang/FlashMLA style, directory structure, and build
  conventions. Use existing formatters and lint configuration; avoid formatting
  unrelated upstream code or introducing another toolchain.
- Keep changes focused on the requested behavior. Prefer small functions with
  clear responsibilities, descriptive names, and explicit inputs/outputs. Add
  abstractions when they remove real duplication or clarify an interface;
  avoid speculative frameworks, compatibility layers, and duplicate runners.
- Share numerical logic and case definitions between consumers. Keep math,
  configuration, artifact handling, and remote orchestration in appropriate
  modules so they can be inspected independently.
- Use type annotations for Python interfaces and structured records for
  configuration/results. Document tensor shape, dtype, device, layout, stride,
  index meaning, and buffer ownership where they form an interface contract.
  Name constants and explain layout offsets or kernel parameters that are not
  obvious from the code.
- Validate assumptions at the boundary that owns them. Raise actionable errors
  containing the failed operation and relevant case/artifact identity. Do not
  swallow exceptions, silently change dtypes/configuration, relax tolerances,
  or turn missing coverage into a successful result.
- Explain numerical choices, synchronization, and non-obvious tradeoffs in
  comments/docstrings. Keep documentation consistent with the implementation;
  avoid comments that only repeat the code or make unsupported speed claims.
- Make CUDA/Triton memory access, row validity, synchronization, and stream
  ownership explicit. Guard invalid accesses, respect alignment and padding,
  and preserve graph replay with changing inputs. Keep unnecessary allocation,
  CPU synchronization, device transfers, and debug instrumentation out of
  measured production paths.
- Manage subprocesses, streams, temporary files, and remote jobs with explicit
  ownership and cleanup on failure. Give external commands bounded timeouts
  and check exit status; do not leave servers or billable allocations running.
- Use existing dependencies and public interfaces where practical. Put
  production behavior in the appropriate fork module rather than runtime
  monkey patches, source-string rewrites, or imports from experiment scripts.
- For behavioral changes, extend existing tests with independent expected
  results and relevant edge cases. Test the contract, not a copy of the
  implementation. Run applicable focused checks and report skipped GPU checks
  accurately. Documentation-only changes need document review, not GPU runs.
- Remove code made obsolete by the change when its consumers are accounted
  for. Preserve historical experiment evidence, and keep unvalidated candidates
  distinct from the production path until their promotion gates pass.

## Goal-oriented experiments and file lifecycle

Keep `experiments/` and `scripts/modal/` organized around explicit goals, not
the chronology of debugging attempts. A new experiment should leave a clear
decision and a reproducible command, not a collection of unexplained scripts.

### Before creating files

- State the experiment's question, changed mechanism, acceptance criteria,
  baseline policy, and stopping condition in its existing plan or README.
  Identify the smallest existing entry point that can answer the question.
- Search existing runners, fixtures, analysis helpers, and tests first. Prefer
  a parameter or focused function in the owning module over a copied runner.
  Do not create another benchmark, reference, or validation harness merely to
  support a new candidate.
- Create a file only when it has a distinct responsibility, a documented
  consumer or runnable command, and a clear lifecycle. Explain why an existing
  module cannot reasonably own it. Avoid `new`, `final`, `v2`, `fixed`, and
  timestamp suffixes for source files; versions belong in Git and run records.

### Placement and ownership

- Put experiment-specific candidates, probes, and analysis under the owning
  `experiments/E###*/` directory, or the established named study directory.
  Keep one plan/README as the index of its entry points, artifacts, and decision;
  do not add parallel status documents or result trees.
- Reserve `scripts/modal/` for maintained remote orchestration and genuinely
  reusable infrastructure: builds, preflight, execution, and artifact transfer.
  Keep numerical code and experiment-specific analysis out of this shared
  directory. Reuse its existing runner interfaces before adding another one.
- Promote an experiment helper into shared infrastructure only when actual
  consumers need it. Keep production runtime and kernels in the appropriate
  fork; production must not import experiment code.
- Use timestamped run directories for outputs, not new source-file copies.
  Store scratch files in an explicitly owned temporary directory and remove
  them on completion or failure. Keep large artifacts outside Git and link
  them from the experiment record.

### Completion and cleanup

- Cleanup is part of completing an experiment. Record the outcome, exact source
  identity, commands, and evidence before removing any candidate or probe.
  A rejected experiment needs reproducible evidence, not an active runner.
- Review every file created or changed for the experiment: retain the canonical
  implementation and necessary reproduction code; consolidate duplication;
  retire superseded probes, debug instrumentation, unused helpers, and wrappers.
  Preserve independent references and useful tests.
- Before moving or deleting a file, check imports, documented commands, build
  inputs, and artifact references. Update affected consumers and documentation
  together. Do not delete unrelated dirty work, historical evidence, or files
  still needed by another experiment. Ask when ownership or scope is unclear.
- Preserve rejected source through a verified Git commit or a recorded source
  patch/snapshot before retiring its active copy. Do not rely on Git to recover
  untracked or ignored experiment files. Confirm a recoverable record exists.
- At handoff, state what remains active, what was consolidated or removed, and
  the single command or entry point for the next step. Flag any deferred cleanup
  explicitly instead of leaving unexplained files behind.

These rules apply to new work immediately. They do not authorize a broad cleanup
of existing files without a scoped request.

## Numerical and runtime contracts

- Call the configurations **Native** and **Packed**. Native is the default;
  Packed is selected by `--dsv4-c4-cache-format remnant`. Do not revive the
  historical TopMag environment-variable bundle or stage-1 terminology.
- The production Packed record retains 256 of 512 coordinates and stores
  256 FP8 value bytes, a 64-byte bitmap, and eight scale bytes: **328 bytes**.
  The native C4 record is 584 logical bytes. Changing survivor count does not
  automatically change the record size; format changes need their own plan.
- Match the current producer's dtype and arithmetic. Its raw compressor
  output and norm weights are FP32. Selection uses raw magnitudes, followed by
  masking, RMSNorm, the prescribed BF16 rounding, and FP8 packing. Historical
  BF16-input store screens are not evidence of exact FP32 production parity.
- Keep raw logical positions distinct from physical cache locations. Preserve
  partial lengths, invalid/padded entries, duplicate selections, RoPE, and the
  sink. Verify page geometry from source/configuration rather than old fixtures.
- Kernel development currently targets H100/SM90. Do not build or run SM100
  tests unless the user explicitly expands the scope. Keep stock Native code
  and unrelated SGLang sparse-attention kernels intact.

## Validation and measurement

Reuse existing fixtures, references, and runners instead of creating replacement
validity or benchmark harnesses. Select checks proportional to the requested
change; preserve independent references and established tolerances.

| Purpose | Existing entry point |
|---|---|
| FlashMLA Direct output/LSE, edge cases, and graph replay | `third_party/flashmla/tests/test_flash_mla_remnant_decoding.py`; shared `tests/lib.py` and `tests/ref.py` |
| SGLang integration | `third_party/sglang/test/registered/attention/unittests/dsv4/test_remnant_*.py` |
| Low-level pack/unpack | `third_party/sglang/python/sglang/test/kernels/deepseek_v4/test_remnant_*_kernel.py` |
| Startup cache-format validation | `third_party/sglang/test/registered/unit/test_dsv4_c4_cache_format.py` |
| Native/Adapter/Direct decode timing | `third_party/sglang/benchmark/remnant/microbench.py` and its README |
| Historical store/read integration timing | `experiments/E010/archive/modal/remnant_integration.py` (archived) |
| Cached builds, validation, timing and kernel profiles | `scripts/modal/kernels.py` and the active experiment's runner |

- Direct reads Packed records inside FlashMLA. Adapter reconstructs Native
  cache records before stock FlashMLA. Report their complete measured paths
  accurately, including combine/reconstruction where required by the protocol.
- Use the active experiment's explicit matrix and acceptance rules. The frozen
  E010 decode judge is H64/H128 × B8/B16 × K512/K317; supplemental shapes do not
  replace it. Do not turn a pruning-quality study into a kernel-speed gate.
- Check numerical validity before interpreting timing. Apply relevant graph,
  stream, and sanitizer coverage for CUDA changes. Missing instrumentation or
  an unexplained sanitizer failure does not count as passing validation.
- Respect the experiment's baseline policy. Reuse saved Native/Adapter results
  when requested; refresh controls only when the approved protocol or a
  separately identified harness correction requires it. Historical ratios are
  not paired measurements. Describe p95 of graph timing samples as such.
- Keep profiling separate from acceptance timing. Reuse prior profiles and
  collect new ones to answer a specific unresolved question. Check SASS,
  registers/spills, and actual instrumentation before attributing a speedup.
- Microbenchmarks and frozen attention replay do not establish serving
  TPOT/TTFT, SLO capacity, or closed-loop agent task quality.

## Modal authorization and cost

- A Modal launch requires user authorization covering the experiment and
  budget, including CPU image builds and full-model runs. Existing approval
  for scoped runs/retries remains valid; do not repeatedly request it.
- Before requesting a new launch, finish local diagnosis/preflight and explain
  the fixed failures, remaining GPU checks, and expected resource use.
- Check the active profile, **compute spend**, cached artifacts/model volumes,
  and competing allocations with the Modal CLI. Follow the active experiment's
  account exclusions and caps; do not assume past account headroom persists.
- Build native extensions on CPU/image-build workers and reuse artifacts keyed
  by source/toolchain hashes. Verify binaries, Python dispatch, and manifests
  before GPU allocation. Stop on a failed preflight instead of rebuilding on
  H100 or blindly relaunching.
- Use the authorized timeout, retry, and concurrency limits. After a monitoring
  timeout, inspect the existing job before starting another. Stop allocations
  promptly after collection and record failures as part of the cost.

## Experiment records

- Work from the last accepted candidate and test one stated mechanism at a
  time. Follow the active plan's commit/snapshot requirements before execution.
  Keep rejected attempts in history/records; do not silently promote them.
- Save unique timestamped run IDs, source content hashes (including dirty
  inputs), artifact identities, commands, fixture/matrix versions, raw samples,
  validity evidence, measured costs, and an explicit keep/discard/inconclusive
  decision. Follow the active ledger rather than creating another logging tree.
- JSON is the canonical measurement record; CSV is the flat summary. Preserve
  historical results and report every shape's gate misses. Do not relabel old
  measurements as a corrected fixture or a new production configuration.
- Keep compact aggregate reports and source patches in Git. Store model
  weights, compiled binaries, large profiles/tensors, raw prompts/tool payloads,
  and credentials outside Git, in the documented cache or Modal volume.
- Before handing off, recheck the requested plan and report what completed,
  what remains unverified, and whether code, tests, pins, or pushes changed.
