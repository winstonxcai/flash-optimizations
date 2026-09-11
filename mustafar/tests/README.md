# mustafar tests

Two questions, two suites, one shared harness:

- **validity** — does packed match native? `validity.py`
- **speed** — what does packing cost against native? `speed.py`

Neither boots a server. Native is exercised through the *same operators the
patch replaces* (`compress_norm_rope_store`, `dequantize_k_cache_paged`), so
"match native" is literal rather than synthetic.

## File naming — one rule

| file | kind | rule |
|---|---|---|
| `test_<topic>.py` | unittest module | the **only** discoverable files; contain `class *Tests(unittest.TestCase)` and nothing else |
| `<topic>.py` | runner module | everything else; exports one `run_*()` entrypoint, never a `TestCase` |
| `harness.py` | shared library | no entrypoint; the only module other runners import for workloads and timing |
| `fixtures/` | data | pinned baselines |

Current tree:

```
tests/
  test_patching.py          # patching.py machinery + real-anchor integration
  test_backend_selection.py # entrypoint contract (flags pinned, env restored)
  test_harness.py           # the case grid perturbs what it claims to
  test_speed.py             # the speed table's stages, legs and contrasts agree
  test_numerics.py          # registration only; skip guards live here
  test_bench_serving.py     # serving benchmark, unchanged
  test_fused.py             # CPU image-build gate (ABI + E4M3 decode)
  harness.py                # workloads, geometry, native refs, legs, patterns, timing
  validity.py               # run_reference / run_packed_reference / run_validity
  speed.py                  # run_speed
  fixtures/
```

The **legs** are one vocabulary, declared once in `harness.py` so the two suites
cannot drift apart. `native` is the bar — stock SGLang with every
`SGLANG_OPT_TOPMAG*` flag off, no mustafar code, no Triton — and never a
candidate. The candidates are `packed.bf16`, `packed.native`, `fused`, and
`sparse`; `packed` appears as its two real entry points rather than as one column
because they are two Triton operators with different costs (`packed.bf16` renders
dense BF16 for the multi-token-extend call site, `packed.native` the 584-byte
layout for the decode/small-extend one). `--legs` narrows the candidate set on
either entrypoint; `native` is always run.

`validity.py`'s GPU entrypoint is one function, `run_validity`, over four
**stages** — `store`, `rows`, `attention`, `pruning` — asserting each leg against
the bar at each one, so a failure says which leg differs from native and where.

`speed.py`'s is `run_speed`, over `store`, `rows.dense_bf16`,
`rows.native_layout`, and `attention`: validity's single `rows` stage split by
the product each leg renders, because timing "the rows stage" would mean adding
together two different operators. `speed.MATRIX` is the table as data — each
stage names the legs it times with a note and the legs it cannot with the reason,
so a column that is absent is reported as absent rather than silently missing.
`pruning` is not timed: it is a correctness stage with no operator of its own.
Each stage takes its ratios against a named bar, and `speed.CONTRASTS` names the
comparisons the suite exists to answer; the ones that are asserted rather than
merely reported say so, and say why.

## Entrypoint contract

Every `run_*()` a test registers must:

1. Pin the flags for the whole call, overriding whatever the ambient
   environment holds. Both suites pin the **all-flags-off** base — `native` is
   defined as stock SGLang — and each leg turns its own flags back on inside a
   `harness.leg_env(leg)` block. That block also runs
   `config.validate_packed_static_config()`, so the pinned set is proven legal
   rather than assumed, and `harness.OFF_ENV` is the base every other pin is
   written against: a leg states exactly what it turns on, and nothing is
   inherited.
2. Raise `RuntimeError` matching `"requires CUDA"` when
   `torch.cuda.is_available()` is false, **before** any allocation.
3. Restore `os.environ` byte-exactly on success and on failure. (`patch.dict`
   gives you this; do not hand-roll it.)
4. Be registered in `test_numerics.py` under the narrowest correct skip guard
   (`HAS_CUDA` / `HAS_TRITON` / `HAS_SGLANG`). Legs whose CUDA extension is not
   built are left out by `harness.select_legs` when no `--legs` is given (and
   rejected with a message when one is asked for by name), so no guard is needed
   for `_fused` / `_sparse`.

`test_backend_selection.py` enforces 1–3 mechanically over its entrypoint list,
and checks each leg's pin is a legal configuration.

## Adding a test

1. Pick or add a workload in `harness.WORKLOADS` (context × batch), or a case
   **pattern** in `harness.ADVERSARIAL_PATTERNS` (how `physical`/`raw`/`lengths`
   and the keep-mask are perturbed). `TOPK` is fixed at 512 — no smaller select
   is a legal packed configuration.
2. Add the check to `validity.py` (assert it) or `speed.py` (time it), as a stage
   that names its bar. A new **leg** is one entry in `harness.LEGS` plus its pin
   in `harness.LEG_ENV`, and nothing else: both suites, the CSV columns, and the
   contrast names derive from that list.
3. If you added a pattern, assert in `test_harness.py` that it actually perturbs
   what it claims to. A pattern that silently reverts to `identity` adds no
   coverage and nothing else will say so. A stage or leg added to `speed.MATRIX`
   is held to the same rule by `test_speed.py`.
4. Register the entrypoint in `test_numerics.py` and add it to
   `test_backend_selection.py`'s list.

## Tolerances

A tolerance is a **named constant in one place**. Never inline a numeric
literal in a suite — add it to `harness.py` with its justification.

Each stage's budget is a separate pair, because reusing one number for several
jobs means a widening justified by one silently loosens all of them:

- NoPE FP8 codes and UE8M0 scales: **bit-exact** (`torch.equal`). Native and
  packed quantise identically. A pattern whose bitmap is not the input mask
  fails here too — the bitmap is compared, not just the codes.
- RoPE tail: `harness.TAIL_ATOL` / `harness.TAIL_RTOL`. Native keeps the 64
  tail dims in BF16, the 328-byte ABI stores them as FP8. That loss is the
  design, not a defect. Applies to the `rows` stage.
- c4 `(o, lse)`: `harness.ATTN_ATOL` / `harness.ATTN_RTOL`, asserted
  **separately** for `o` and `lse` — a correct-lse/wrong-o split is the
  likeliest real failure and the merged output would hide it.
- TopMag50's own cost: `harness.QUALITY_ATOL` / `harness.QUALITY_RTOL`, used
  only by the `pruning` stage, whose bar is native over the *untouched* latent.
  A failure there is a finding about the compression, not about a kernel — name
  it that way rather than widening the kernel tolerances.

A failure here is **surfaced, never silently widened**. If an observed maximum
exceeds the pinned tolerance, either fix the kernel or write down the wider
value with its justification. Pin observed maxima per case in
`fixtures/validity-baseline.json` (`run_validity(write_baseline=True)`, or the
`--write-baseline` flag) to fail on regression even inside the absolute
tolerance. The pinned values are exact, not padded: the same GPU on the same
inputs is expected to reproduce them, and a gate that flaps is a finding about
non-determinism rather than something to widen away.

## Real anchors

Synthetic fixtures cannot be built from the real anchor list: those anchors are
fragments of one large call expression (bare argument lists, partial `if`
bodies) and `patching._render` compiles what it produces. So:

- `PatchMachineryTests` uses small self-authored fixtures with self-authored
  anchors. Hermetic, runs anywhere, never stale.
- `RealAnchorTests` copies the real pinned tree and runs the same operations
  against the real anchors. Skips cleanly when `config.SRC_ROOT` is absent.
- Anchor *drift* — do the anchors still match a given tree — is
  `patching.drift()`, run by `scripts/local/container.sh`. Not a test.

## Running

Offline, no GPU, no network (works on a bare host):

```sh
python3 -m unittest mustafar.tests.test_patching -v
```

In the `remnant` container (GPU kernels + real anchors):

```sh
docker exec remnant bash -c 'cd /sgl-workspace/sglang-lowrank && \
  PYTHONPATH=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations \
  python3 -m unittest mustafar.tests.test_numerics -v'
```

Direct, prints per-case JSON (and writes `speed.{json,csv}` when
`MUSTAFAR_RESULTS_DIR` is set):

```sh
python3 -m mustafar.tests.validity   # every stage, every available leg
python3 -m mustafar.tests.validity --legs sparse       # the sparse leg only
python3 -m mustafar.tests.validity --write-baseline    # pin the regression gate
python3 -m mustafar.tests.speed      # every stage, every available leg
python3 -m mustafar.tests.speed --legs fused,sparse    # a subset of the candidates
```

Package CLI equivalents: `python3 -m mustafar selftest` and
`python3 -m mustafar packed_selftest`.
