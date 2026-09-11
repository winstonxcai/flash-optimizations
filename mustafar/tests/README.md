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
  test_numerics.py          # registration only; skip guards live here
  test_bench_serving.py     # serving benchmark, unchanged
  test_fused.py             # CPU image-build gate (ABI + E4M3 decode)
  harness.py                # workloads, geometry, native refs, timing, tolerances
  validity.py               # run_reference / run_packed_reference / run_validity
                            #   / run_sparse_t4
  speed.py                  # run_speed
  fixtures/
```

## Entrypoint contract

Every `run_*()` a test registers must:

1. Pin the flags for the whole call —
   `@patch.dict(os.environ, SGLANG_OPT_TOPMAG="1", KEEP="0.5",
   SGLANG_OPT_TOPMAG_PACKED="1", SGLANG_OPT_TOPMAG_FUSED="0")` — re-enabling
   `_FUSED` only inside the leg that needs it.
2. Raise `RuntimeError` matching `"requires CUDA"` when
   `torch.cuda.is_available()` is false, **before** any allocation.
3. Restore `os.environ` byte-exactly on success and on failure. (`patch.dict`
   gives you this; do not hand-roll it.)
4. Be registered in `test_numerics.py` under the narrowest correct skip guard
   (`HAS_CUDA` / `HAS_TRITON` / `HAS_SGLANG` / `HAS_FUSED` / `HAS_SPARSE`).

`test_backend_selection.py` enforces 1–3 mechanically over its entrypoint list.
Add your new entrypoint there.

## Adding a test

1. Pick or add a workload in `harness.WORKLOADS` (context × batch). `TOPK` is
   fixed at 512 — no smaller select is a legal packed configuration.
2. Add the check to `validity.py` (assert it) or `speed.py` (time it).
3. Register it in `test_numerics.py`.
4. Add the entrypoint name to `test_backend_selection.py`'s list.

## Tolerances

A tolerance is a **named constant in one place**. Never inline a numeric
literal in a suite — add it to `harness.py` with its justification.

- NoPE FP8 codes and UE8M0 scales: **bit-exact** (`torch.equal`). Native and
  packed quantise identically.
- RoPE tail: `harness.TAIL_ATOL` / `harness.TAIL_RTOL`. Native keeps the 64
  tail dims in BF16, the 328-byte ABI stores them as FP8. That loss is the
  design, not a defect.
- End-to-end qk logits and attention output: same two constants.

A failure here is **surfaced, never silently widened**. If an observed maximum
exceeds the pinned tolerance, either fix the kernel or write down the wider
value with its justification. Pin observed maxima per workload in
`fixtures/validity-baseline.json` to fail on regression even inside the
absolute tolerance.

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

Direct, prints per-workload JSON (and writes `speed.{json,csv}` when
`MUSTAFAR_RESULTS_DIR` is set):

```sh
python3 -m mustafar.tests.validity   # T1/T2/T3 over the workload grid
python3 -m mustafar.tests.validity --sparse   # T4 only, the direct read
python3 -m mustafar.tests.speed      # native | packed/triton | packed/fused
```

Package CLI equivalents: `python3 -m mustafar selftest` and
`python3 -m mustafar packed_selftest`.
