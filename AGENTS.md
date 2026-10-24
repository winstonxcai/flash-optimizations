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
| Fork integration and file map | `third_party/README.md` |
| Fork integration history | `third_party/CHANGELOG.md` |
| Local serving and evaluation | `scripts/local/` |
| Modal workflows and local CPU checks | `scripts/modal/serve.py`, `kernels.py`, and `tests.py`; read their module docstrings |
| Maintained pruning data/training/evaluation | `training/`; read `training/README.md` |
| Attention-aware pruning methods and findings | `docs/attention-aware-latent-kv-compression.md` |
| Project findings and paper narrative | `docs/report.md` |
| Paper figures | `docs/figures/` |

Read applicable nested `AGENTS.md` files before editing their directories.
Check current source and active experiment records when older documentation
disagrees with them. Do not treat a recorded branch head as the current checkout.

## GPU servers and shared VDI storage

The VDI has separate VS Code Remote SSH windows for two GPU nodes:

| SSH host | Hardware |
|---|---|
| `10.72.1.171` | 8 × NVIDIA H100 80 GB |
| `10.72.1.175` | 8 × NVIDIA H100 80 GB |

- Confirm the VS Code SSH host and terminal hostname before submitting jobs,
  inspecting processes or stopping servers. GPUQ queues, job status, GPU IDs
  and running processes belong to that node; inspect both nodes when deciding
  where an authorized job can run. Request GPU counts through GPUQ and use its
  assigned devices rather than manually choosing GPU IDs.
- Both nodes access the same shared disk for
  `/home/jovyan/winstonxcai/flash-optimizations` and its repository folders,
  including source, data, logs, results and transfer packets. Files do not need
  copying between these nodes. Git operations, source edits and permission
  changes on one node affect the other. Do not pull, change fork pins or edit
  shared serving sources while a job on either node depends on that checkout.
- Check existing images and containers on the destination node first. Matching
  image/container names and shared repository files do not establish matching
  Docker image IDs or container state. Reuse an existing matching image; do not
  export, rebuild or recreate an image merely because execution moves nodes.
  If the recorded image ID differs, establish the intended baseline before
  aligning it. Preserve unrelated running containers and servers.
- Record the execution node alongside each GPUQ job ID and assigned GPUs. When
  moving a job, confirm cancellation on the original node, stop its old result
  watcher and submit the replacement once. Keep cancelled-job evidence and
  prevent collectors on different nodes from publishing the same run twice.
- GPUQ may execute as `gpuq-runner` (group `gpuq`) rather than the interactive
  root account. Verify that this identity can access prepared inputs and write
  the run's artifacts before queueing. Let GPUQ manage its job logs; use the
  existing serving/replay artifact paths rather than extra logging scaffolding.

## Git and source ownership

- Inspect the parent and relevant submodules' branch, status, and diff before
  editing. Preserve existing staged, unstaged, and untracked work.
- After completing requested edits and relevant checks, commit by default unless
  the user explicitly says otherwise. Include only files and changes made for
  the current task in this conversation. Leave all unrelated dirty work and
  pre-existing staged changes untouched; if a file contains unrelated edits,
  stage only this task's hunks. Never use a blanket add/commit. If changes cannot
  be safely separated, ask before committing. Report the commit; do not push
  unless explicitly requested.
- The established branches are parent `remnant`, SGLang `remnant/v0.5.18`, and
  FlashMLA `remnant/flashmla-v0518`. Do not develop on `main` or switch branches
  over dirty work. `origin/...` names are remote references, not local branches.
- Use the checked-out forks in `third_party/` as the default source for edits and
  builds. Do not substitute a GitHub commit, fetch a newer upstream version, or
  run the legacy runtime-patching/backup workflow without task authorization.
- When the active experiment needs an isolated candidate, use its documented
  worktree or source-snapshot workflow. Record the source path and overrides;
  avoid unexplained `tmp/` copies. Preserve the shared fork checkout.
- A submodule pin records a commit, not its uncommitted files. When committing
  or pushing, handle each affected fork on its own branch, then update the
  parent pins. Push referenced fork commits before the parent. Stage only the
  requested work; do not sweep unrelated dirty files into a commit.
- After an authorized parent pull/checkout or pin change, synchronize the forks
  to the exact recorded commits, not the latest remote branch; ahead is also a
  mismatch. Inspect dirty work first. For a clean reproduction checkout use
  `git submodule update --init --recursive --checkout`; this may detach HEAD.
  Prefer a fast-forward to the recorded pin when preserving a development branch.
  Never force, reset, discard or automatically stash work to synchronize. Stop
  and report if matching the pin would disturb existing changes.
- Before ordinary reproducible builds, serving or benchmark runs, existing
  local/Modal preflights must require each fork's HEAD to equal the parent pin
  and reject staged, unstaged or untracked fork changes. Fail before image builds
  or GPU allocation, reporting expected/actual commits and synchronization steps;
  preflights must not repair the checkout themselves. Deliberate development or
  approved experiments may use an explicit source override with recorded commit
  and content hashes. Commit fork changes and update parent pins before release.
- Update `third_party/README.md` when fork ownership, changed files or contracts
  change. Record meaningful integrations/fixes in `third_party/CHANGELOG.md`
  with the parent commit, both fork pins and validation scope; keep tuning/debug
  chronology in Git and run records. Preserve upstream licenses and attribution.
  Add or retain `Author: Winston Cai.` in the top docstring/comment for Remnant
  changes, using the file's existing language and format.

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

## Review and acceptance

- Summarize nontrivial changes as **Purpose, Risk, Evidence, Limitations,
  Rollback**. Evidence includes source identity, exact commands and outcomes;
  state what the checks cannot establish, not just what passed.
- Independently review benchmark fixtures, measured paths and comparison
  fairness alongside kernel changes. Use a fresh-context review for high-risk
  changes; test success alone is not independent review.
- Scale scrutiny to risk. Prioritize numerical arithmetic, CUDA synchronization,
  persistent layouts, allocator accounting, statistical claims and GPU job
  ownership over cosmetic issues. Require concrete evidence for findings.
- Turn discovered failure modes into focused regression checks where practical,
  rather than repeatedly expanding instructions.
- Review correctness first, simplicity second. Keep simplification behavior-
  preserving. After three review/fix rounds, reassess the approach; subsequent
  reviews should focus on new changes and unresolved findings, not repeat nits.
- Keep changes small and reversible. Identify the known-good baseline and how
  to disable or revert a candidate before promotion; human approval remains
  required wherever the task or repository policy requires it.

## Goal-oriented experiments and file lifecycle

Keep research work and shared runners organized around explicit goals, not
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

- Keep experiment-specific candidates, probes, and analysis isolated from
  production code and shared runners.
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
- Default Packed retains 256 of 512 coordinates: 256 FP8 value bytes, a
  64-byte bitmap and eight scale bytes (**328 bytes**). Triton supports startup
  per-C4-layer counts 64..448 in steps of 64, using `count + 72` logical bytes
  in separate value/bitmap/scale planes; 512 survivors are invalid. Non-256
  profiles require Triton store/reconstruction and no HiCache. Optimized CUDA
  store/Direct and HiCache remain 256-only. Native C4 remains 584 logical bytes.
  Configurable counts are not adaptive runtime budgets; new GPU profiles need
  H100 validation before production use.
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
| Cached builds, validation, timing and kernel profiles | `scripts/modal/kernels.py` |

- Direct reads Packed records inside FlashMLA. Adapter reconstructs Native
  cache records before stock FlashMLA. Report their complete measured paths
  accurately, including combine/reconstruction where required by the protocol.
- Use the approved experiment's explicit matrix and acceptance rules;
  supplemental shapes do not replace required coverage. Do not turn a
  pruning-quality study into a kernel-speed gate.
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
- For Nsight Systems (`nsys`), read
  `/Applications/NVIDIA Nsight Systems.app/Contents/skills/nsight-systems/SKILL.md`
  before use on this Mac. On other hosts, use the installed skill's location;
  report if unavailable rather than assuming this macOS path exists there.
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
