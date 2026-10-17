<!-- Author: Winston Cai. -->

# Attention-pruning pipeline

Train the accepted row-local rank-64 scorer and evaluate it with exact Packed
attention replay. These commands use cached tensors, not model weights.
Production Native/Triton/kernel code remains in the forks and never imports
this package. Capture is a separate, explicitly launched workflow.

## Start here

Run from the repository root with Python 3.10+, PyTorch and NumPy. Use the
capture's frozen calibration runner and matching numerical sources. Training
also requires the source Torch version; compare results on matching devices
and thread settings.

```bash
python -m training.train --cache /data/pruning/cache \
  --policy /data/pruning/policy.json \
  --runner /data/pruning/frozen-sglang/python/sglang/benchmark/remnant/calibrate.py \
  --output /data/pruning/new-fit --recipe continuation --layers 2,4 --device cpu

python -m training.evaluate --cache /data/pruning/cache \
  --policy /data/pruning/new-fit/policy.json \
  --runner /data/pruning/frozen-sglang/python/sglang/benchmark/remnant/calibrate.py \
  --output /data/pruning/evaluation.json --layers 2,4

python -m unittest training.tests.test_integration -v
```

Training needs a fresh external output directory and a hash-bound warm policy.
A successful fit publishes `<output>/policy.json`; use that file directly for
evaluation or continuation. Failed fits retain diagnostics, not a usable policy.
No command downloads checkpoints, starts Modal, or allocates GPUs implicitly.

## Ownership

| File | Responsibility |
|---|---|
| `artifacts.py` | Source lifetime, hashes, policy/checkpoint reads, common run identity |
| `models.py` | Rank-16 warm construction and rank-64 row-local scoring/restoration |
| `losses.py` | Exact scalar/batched replay, training proxies, preflights and objective selection |
| `train.py` | Optimizer loop, checkpoint selection/serialization and policy publication |
| `evaluate.py` | Policy/comparison replay, aggregation, plotting and report publication |
| `data/batches.py` | State interface, synchronous batch transfers and temporary warm anchors |
| `data/dataset.py` | CaptureReader, captured/cached state validation; explicitly separated historical cache tools |
| `data/storage.py` | Storage accounting, explicit conversion and paired reader measurements |
| `capture.py` | In-environment data preflight, Native server/replay lifecycle and cleanup |
| `tests/test_integration.py` | CPU workflow and failure scenarios using real readers and losses |

Training/evaluation do not import one another or the storage-audit module.
Ignored experiment scripts are historical evidence, not maintained dependencies.
There are no forwarding wrappers, replacement numerical references, or alternate
test harnesses.

## Source and artifact contracts

- The supplied runner is checked **before execution**. Runner and numerical
  primitives execute compiled source bytes rather than timestamp-based bytecode
  caches. One immutable reference revision is owned per Python process. A source
  change or unowned reference import requires a fresh process; no global modules
  are silently replaced.
- Every numerical run records cache config/index/progress hashes, policy and
  runner hashes, requested checkpoint identities, Torch, device and threads.
  Training adds recipe/source hashes; comparisons add numerical-source,
  per-layer, platform and command details.
- Policies retain all input layers. Trained entries use new relative checkpoint
  paths; untouched entries preserve metadata/hashes and use absolute references
  to their original files. Historical scores and audit claims are not inherited.
- Rank64 expansion requires rank-16 odd/even warm parameters; other recipes
  require rank-64 parameters. Checkpoint names and initialization remain fixed.
- Development selection uses exact scalar Packed replay, never a training proxy.
  Recipes retain the recorded 40 expansion, 80 continuation or 32 relaxed/anchored
  updates, batch eight and half-scale learning rates. The recorded mixed policy
  uses relaxed layer 8, anchored layer 12 and nineteen continuation incumbents.
  Applying recipes elsewhere is a new experiment, not reproduction of that mix.
- `fit(mode=...)` accepts only `conditional_expanded`. The argument and legacy
  guard fields are compatibility fields, not additional active architectures.

## Capture, readers and memory

`scripts/local/capture-latent.sh run` owns Docker image checks, mounts, GPU
selection, deadlines and exact-container cleanup. It uses a prepared capture-only
image and never builds or pulls one. `training/capture.py` owns capture inputs,
the existing Native launcher/calibration client and server-process cleanup.
See `scripts/local/README.md` for preparation and launch commands.

`CaptureReader(runner, published_root, issues)` owns one runner and verifies every
required publication before measurement. Call `reader.load_views(scratch, issue,
layers)` to extract one issue for the requested layers. Extraction is removed on
success or exception. Bundle/marker/config changes invalidate cached verification;
extracted member hashes and numerical checks still run. A returned view is CPU
data, not a permanent second dataset copy.

Both ordinary reading and hash-bound preload verify issue, layer and reservoir
ordinal before constructing training states. Four-rank alignment, duplicate
selections, Native byte targets, RoPE, sink and saved denominators are preserved.
Only fully admitted states with the frozen training geometry are supported.

Existing ZIP-serialized training caches are memory-mapped on CPU. Hashes use
8 MiB chunks; complete tensor/coverage/denominator checks still scan the requested
layer. Mapping is not a hard RSS ceiling: tensor pages remain OS-managed.

Fitting stages one ordered batch synchronously on the chosen device, through
backward. No workers, pinned pools, prefetch or extra streams are hidden here.
The unused `target_unrounded` diagnostic remains in cache files but is not staged.
Development and baseline replay remain scalar and state-at-a-time. Singleton
tails of batched fits retain batched arithmetic; the initial STE check remains
at temperature 0.03.

Anchored training computes each frozen-warm residual once and writes a temporary
FP32 mapped bank under the output directory. Current geometry needs 1,179,648
scratch bytes per training state. Matching residuals stage with each batch; the
bank is removed on success or exception. Abrupt process death may leave a private
`remnant-anchors-*` directory; remove it only after confirming the run is inactive.
This is not a persistent cache format or resume mechanism.

Historical cache construction still accumulates one layer's derived CPU states.
Serialized rank bytes are bounded to one four-rank group (256 MiB maximum), with
failure cleanup. It is not a fully streaming tensor-cache builder. The CLI uses
CPU-mapped sources; callers supplying GPU lists retain ownership of those lists.
CUDA transfer parity and peak-device-memory checks require separate authorization.

## Storage and historical tools

```bash
python -m training.data.storage --root /data/capture \
  --runner /data/frozen-sglang/python/sglang/benchmark/remnant/calibrate.py \
  --output /data/new-storage-report
```

Audit alone writes compact JSON/CSV reports. Add `--convert` explicitly to write
and verify a shared-format copy. Physical payload bytes count referenced schema-2
shared files/constants once. Tensor field sizes are **logical extents** and may
count aliased storage more than once; they are not serialization overhead. The
unsupported overhead calculation is removed.

`--reader-check` and `--batch-reader-check` compare the original extracted capture
with converted/published data. They require complete coverage, not an implicit
subset. Timing excludes the initial bundle verification and includes extraction,
member checks and numerical replay. First/middle/last layers are unique even for
one- or two-layer captures. First-read measurements are not cold-cache claims.

The historical section of `data/dataset.py` exposes the original cache build and
coverage extension through its existing CLI. These deliberately require the
frozen 50-issue, four-/ten-query setup; coverage extension is restricted to layers
8/12. CPU data work is permitted on CUDA-equipped hosts. This is **not yet an
arbitrary-cohort cache builder or trainer**. Use `--help` for those reproduction
commands; their formats, saved targets and denominator tolerance are unchanged.

## Comparisons and retention sweeps

```bash
python -m training.evaluate --compare-masks \
  --cache /data/pruning/cache --policy /data/pruning/policy.json \
  --runner /data/frozen-sglang/python/sglang/benchmark/remnant/calibrate.py \
  --output /data/pruning/comparison.json --threads 4
```

The six-method comparison evaluates global magnitude, structured and learned
masks at 256/192 survivors. Structured masks are consecutive pre-RoPE 2:4 at 256
and 3:8 at 192. Names describe adaptations, not full upstream reproductions.
Learned reuses the frozen 256-trained scorer; budget transfer is not retraining.

Use `--retention-sweep --figure-dir docs/figures` instead of `--compare-masks`
for 64–448 survivors in steps of 64. Its structured masks consistently use N:8;
256 is **4:8, not 2:4**. The two PNGs contain three measured curves for mean/p95.
Matplotlib is needed only for plotting. Existing figures are never overwritten.

All methods retain masked FP32 RMSNorm, prescribed BF16/UE8M0/E4M3 rounding,
key/inverse-output RoPE, SWA, sink and saved Native denominators. The offline
simulation must match the 256-value codec exactly before other budgets are
admitted. Non-256 budgets are not deployed record formats.

Mean weights queries within issue/layer, then layers and issues equally. p95
pools query-layer errors; 95% mean/paired intervals use 10,000 whole-issue
bootstrap draws (seed 42). These reused development issues were selected during
tuning: results are not fresh-test significance, task quality or serving speed.
The full comparison covers ten development issues, 21 layers and 2,100 paired
examples per method. `--layers 2` is an explicitly labeled diagnostic.

Reports stage together and use exclusive publication. The canonical JSON is
published last as the completion marker. Ordinary failures roll back only this
attempt's files and retain partial diagnostics; competing writers' files survive.
A crash can leave uncommitted ancillary files without JSON. Inspect those before
cleanup. Output prefixes and figures must be fresh; failed runs are not resumed.

## Verification and historical evidence

The single integration module uses synthetic full-shape CPU states, real cache
readers, two-update fits, generated-policy evaluation/continuation and independent
Packed references. It covers all three gradient modes, batches 1/8 and singleton
tails, missing/corrupt coverage, warm-anchor cleanup, stale-source rejection,
storage accounting and report-publication failures. Capture processes are mocked.
These checks do not rerun the reported quality result or load model weights.

`migration.json` is the original migration receipt, not a current test report.
Large artifacts and recovery snapshots remain outside Git:

- `~/.cache/remnant/attention-pruning/ownership-t3NDTX/`: ownership refactor.
- `~/.cache/remnant/attention-pruning/bounded-oMTsMW/`: bounded reader/staging.
- `~/.cache/remnant/attention-pruning/capture-split-Ued7ce/`: verified capture split.
- `~/.cache/remnant/attention-pruning/review-fixes-mh7gmL/`: first review fixes.
- `~/.cache/remnant/attention-pruning/cleanup-audit-Dxu08m/`: this cleanup and its two audits.
- `~/.cache/remnant/attention-pruning/training-test-backup-Heus0N/training-tests.tar.gz`:
  superseded unit tests; SHA-256
  `ee5783c07f2c95e57130fa3c3604427752e36a70917a46fd97a33eb31eccb9c6`.

The calibration runner, observer and fork calibration tests are now tracked in
SGLang. Reproduction still requires the exact hash-matching source tree recorded
by the capture/cache; a newer fork checkout is not a substitute. No fork files,
pins, commits, pushes or production runtime changes are part of this cleanup.
