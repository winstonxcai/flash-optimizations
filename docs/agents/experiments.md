# Experiment lifecycle and records

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
