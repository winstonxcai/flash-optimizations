# starkv tests

Three suites, all CPU-only, all runnable from a bare host with nothing but
torch:

| suite | question |
|---|---|
| `test_reference.py` | do the three identities hold exactly, and is the record layout what `config.py` says? |
| `test_analysis.py` | do the measurement modules separate what they claim to separate, and do the two capture hooks agree on one decode window? |
| `test_patching.py` | does the patch machinery refuse what it should refuse, and restore byte-exactly when it doesn't? |

Run them all:

```sh
python -m starkv selftest
```

or one at a time (`python -m starkv.tests.test_analysis`). Each suite also
exports `TESTS` and a plain `run() -> int`, so the aggregate `selftest` needs no
test runner.

There is no GPU suite. Everything that needs CUDA is a *driver* script under
`scripts/local/`, listed in the plan's verification table and in `report.md`,
not a test here — a test that skips itself on a CPU host mostly reports that it
did not run.

## What each suite is really asserting

**`test_reference.py`** pins the three identities the whole design rests on:

1. `z = Dᵀx`, `x̂ = Dz` — **bit-exact** at `r = 448` with an identity basis, which
   is the only rank at which the round trip is allowed to be lossless.
2. `q_Nᵀ(Dz) == (Dᵀq_N)ᵀz` — the latent read path is algebraically exact, so a
   reconstruct MVE and a latent kernel must agree to fp accumulation order.
3. `W_o,a (Dz̄) == (W_o,a D) z̄` — and that the absorption touches **only** the
   448 NoPE columns, leaving the 64 RoPE columns alone.

Both identities are compared with `torch.allclose` at a tolerance justified by
accumulation, not at a bit level: they are equal in exact arithmetic and are
computed in different orders, so the tolerance is the statement that the
difference is fp noise and nothing else.

The rest of the suite is the record layout (`bytes_for_rank`, monotone in rank,
every cell smaller than native), the pack/unpack and pool store/gather round
trips, and the two primitives — RMSNorm against its definition, RoPE against an
interleaved complex rotation written out longhand. Those two are checked against
their *definitions* rather than against the container's implementation, because
the point is to catch the day the definition and the implementation diverge.

`test_store_zeroes_non_boundary_rows_but_keeps_the_slot` pins the vendor's store
semantics: non-boundary decode rows are zeroed onto `out_loc == 0` but the slot
stays allocated. Getting this wrong changes what the pool *means*, and it is the
one contract that a plausible-looking reimplementation gets wrong silently.

**`test_analysis.py`** builds fixtures with a *known right answer* — a capture
whose latents genuinely live in a low-rank subspace, a basis that is exactly
exact, a stub server whose drift is a linear function of position — because both
analysis modules would otherwise happily produce plausible tables from garbage.
The stub's `text` field carries its own token ids, so `score` can be checked
against arithmetic rather than against itself.

Two checks carry most of the weight:

- `test_score_error_separates_the_basis_cost_from_the_quantization_cost` — with
  an identity basis, `eps_basis == 0.0` **exactly** while `eps_quant > 0.0`. If
  the two were not separable, the whole "is it the rank or is it the basis"
  question the study exists to answer would be unanswerable.
- `test_capture_hooks_share_one_window_and_tag_rows_with_their_call` — feeds the
  files the capture hooks actually wrote back through `spectrum.score_error`. The
  store hook and the attention hook run at different points in the decode step
  and buffer independently; if their windows ever stop covering the same steps,
  the join silently returns nothing and the measurement would report an empty
  table rather than an error.

**`test_patching.py`** runs two ways, and the split matters:

- **synthetic fixtures** — small self-authored files with self-authored anchors.
  Hermetic, runs anywhere, never goes stale.
- **real anchors** — the same operations against copies of the actual v0.5.18
  targets, skipped cleanly when `config.SRC_ROOT` is absent.

Anchor *drift* — whether the anchors still match a given tree — is
`python -m starkv drift`, run by `scripts/local/container.sh`, and is not a test:
it is a census against a tree that lives in a container, and it reports a file
count rather than passing or failing.

## Adding a test

1. Add the function, then add it to that module's `TESTS` tuple. A test that is
   not in `TESTS` does not run — `selftest` has no discovery.
2. Keep it CPU-only and importable without a container. If it needs a real tree,
   skip cleanly on a missing `config.SRC_ROOT` rather than failing.
3. Give the fixture a known answer. Asserting that a function returns a
   `dict` with the right keys is not a test; assert the numbers that would
   be wrong if the function were subtly wrong.
4. Tolerances are named constants. A new numeric literal inline in a test is a
   tolerance that nobody can find later.
