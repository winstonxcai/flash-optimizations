# Git, submodules and source ownership

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
  FlashMLA `remnant/flashmla-v0518`. This isolated integration uses parent
  `codex/sglang-v0.5.22` with fork branches `remnant/v0.5.22` and
  `remnant/flashmla-v0522`. Select the matching pair explicitly with
  `REMNANT_INTEGRATION=v0.5.22`; unset it for v0.5.18. Do not develop on `main` or switch branches
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

### Branch naming

Use model-oriented names for new branches. `<model>` includes the model family
and variant, for example `glm53-flash`.

| Purpose | Pattern | Example |
|---|---|---|
| Maintained model implementation | `<model>-packed` | `glm53-flash-packed` |
| Performance work | `perf/<model>/<topic>` | `perf/glm53-flash/loader` |
| Fixes | `fix/<model>/<topic>` | `fix/glm53-flash/retraction` |
| Experiments | `exp/<model>/<topic>` | `exp/glm53-flash/cache-layout` |
| Shared infrastructure | `<type>/shared/<topic>` | `fix/shared/modal-budget` |

- Use lowercase names and hyphens within each slash-separated component. Do not
  automatically prepend agent or tool names such as `codex/`; honor an exact
  branch name explicitly requested by the user.
- Use matching implementation branch names in the parent and affected forks.
  Each repository still owns its commits; update submodule pins separately.
- Include hardware or backend names only when the implementation differs.
  H100 and H20 can share an implementation branch when they use the same kernels.
- Preserve existing integration branches and published legacy names until an
  explicit migration is agreed. This convention does not authorize renaming or
  deleting existing branches.
- Exact submodule commits remain authoritative for source identity. A branch
  name does not establish which code was built or which validation passed.

## Image inputs during builds

- Do not edit any file copied into a Modal image (`scripts/modal/*.py`,
  fork sources, fixtures) while a build or prepare step is running; Modal
  rejects the build with "modified during build process". The launcher writes
  `.modal-build.lock` at the repository root for the duration of a build, and a
  project hook denies edits to image inputs while it exists. Wait for the build
  to finish or cancel it first.
