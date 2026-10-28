<!-- Authored by Winston Cai. -->

# Remnant

Training-free sparsification and persistent storage of pretrained models'
compressed attention states, integrated into SGLang and FlashMLA forks.

- Working rules: [AGENTS.md](AGENTS.md); detail by topic in
  [docs/agents/](docs/agents/). Review criteria: [REVIEW.md](REVIEW.md).
- Production code lives in `third_party/sglang-v*` and `third_party/flashmla-v*`
  (v0.5.18 and v0.5.22 fork pairs, separate Git repositories); the parent holds runners, records and the report.
- Configurations are **Native** (default) and **Packed**
  (`--dsv4-c4-cache-format remnant`). Target H100/SM90.
- Before reporting work done, run something that proves it and show the
  output: the focused CPU tests, a preflight, or the named validation entry
  point in `docs/agents/numerical-and-validation.md`.
- Before opening a PR, have a fresh-context subagent review the diff against
  `REVIEW.md`. Fill in the PR template, including source identity and rollback.
- Do not edit `scripts/modal/*.py` or fork sources while a Modal build runs
  (`.modal-build.lock` exists).
- Modal launches need explicit authorization covering experiment and budget.
