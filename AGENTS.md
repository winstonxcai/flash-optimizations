<!-- Authored by Winston Cai. -->

# Repository guidance

This repository develops Remnant: training-free sparsification and persistent
storage of pretrained models' compressed attention states. DeepSeek-V4-Flash-0731
is the current integration model. The parent repository contains experiment
records, serving scripts, Modal runners, and the report; production runtime and
kernel code live in two fork repositories, checked out as independently pinned
v0.5.18 and v0.5.22 integration pairs.

## Where to work

| Area | Location |
|---|---|
| SGLang runtime, packing, allocator, and dispatch | `third_party/sglang-v0.5.18/` |
| FlashMLA direct decode kernel | `third_party/flashmla-v0.5.18/` |
| Explicit v0.5.22 integration pair | `third_party/sglang-v0.5.22/`, `third_party/flashmla-v0.5.22/` |
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

## Non-negotiables (details in `docs/agents/`)

Read the linked file before working in its area.

| Area | Rule in one line | Details |
|---|---|---|
| Git and pins | Inspect parent and fork status first. Commit only this task's files or hunks, never a blanket add. Do not push unless asked. Do not develop on `main`. Never reset, discard or stash to synchronize. New branches use model-oriented names. | [git-and-sources.md](docs/agents/git-and-sources.md) |
| Source identity | Preflights must require each fork's HEAD to equal the parent pin and reject dirty fork files before any image build or GPU allocation. Never edit image inputs while a Modal build runs. | [git-and-sources.md](docs/agents/git-and-sources.md) |
| GPU servers | Confirm the node (`10.72.1.171` or `.175`) before submitting or stopping jobs. Use GPUQ-assigned devices. The disk is shared between nodes. | [gpu-servers.md](docs/agents/gpu-servers.md) |
| Numerics | Native is the default and Packed is `--dsv4-c4-cache-format remnant`. Match the producer's dtype and arithmetic. Target H100/SM90 only. | [numerical-and-validation.md](docs/agents/numerical-and-validation.md) |
| Validation | Reuse existing fixtures and runners. Check numerical validity before timing. A missing sanitizer summary is not a pass. | [numerical-and-validation.md](docs/agents/numerical-and-validation.md) |
| Modal | A launch needs user authorization covering experiment and budget. Finish CPU preflight first. Stop allocations after collection. | [modal.md](docs/agents/modal.md) |
| Experiments | State question, acceptance and stopping condition before creating files. Clean up and record the decision at the end. | [experiments.md](docs/agents/experiments.md) |

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
