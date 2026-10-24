# v0.5.22 model-free qualification

Author: Winston Cai.

**Core qualification passed:** candidate `3eb8594` passes 457 numerical
tests + 60 subtests, exact allocation reconciliation, and all 58 racecheck cases
plus 15 subtests with zero hazards/errors/warnings. Synccheck and memcheck each
pass the same 58-case selection + 15 subtests with zero errors.
Earlier Native barrier/race failures and incomplete attempts are retained below,
not attributed to the corrected candidate. No full-model load or timing;
HiCache remains deferred.

Canonical aggregates: [JSON](qualification.json), [allocation CSV](allocation.csv).

| Current candidate `3eb8594` | Completed coverage | Result |
|---|---:|---|
| Local launcher/source checks | 67 tests | Passed |
| Linux CPU preflight | 134 tests + 56 subtests | Passed |
| H100 core numerical/runtime | 457 tests + 60 subtests | Passed |
| Physical allocation | 56 edges, three complete pools, workspaces | Exact reconciliation |
| Racecheck 2026.2.1 | 58 cases + 15 subtests | Zero hazards/errors/warnings |
| Synccheck 2026.2.1 | 58 cases + 15 subtests | Zero errors |
| Memcheck 2026.2.1 | 58 cases + 15 subtests | Zero errors |

Current numerical coverage omits exactly four deferred HiCache GPU tests from
the historical 461. The frozen sanitizer selection omits HiCache restoration
and adds H128 changing-input graph replay: still 58 base cases per tool, but
not the historical 58-case identity set. All current checkpoints use the same
source, binary and tool manifest; no older-binary pass clears a current gate.

Source: parent runtime `e68f9c4`, SGLang `7856615`, FlashMLA `3eb8594`,
CUTLASS `147f567`; qualification image `im-vOlukc1AW0gozz9mlAmg1m`.
The V4 Native path includes the authorized synchronization corrections: it is
not the untouched historical Native binary. No arithmetic/layout change or
performance claim follows. Revert to the unchanged v0.5.18 profile by unsetting
`REMNANT_INTEGRATION`; the older v0.5.22 binary remains sanitizer-failing.

Reported compute at **2026-10-24 12:17 UTC**: **$8.5113** for the latest $15
continuation; **$26.6918** total on `winston-85885`, below its $28 ceiling.
Billing can lag; the $2 continuation reserve remains intact. No active containers.
HiCache GPU qualification, full-model/distributed execution and performance
remain separate gates. No models were loaded and no changes were pushed.

## Curation into the clean integration branch

This collection was copied from archived parent `62cb99a` without rewriting its
JSON/CSV, hashes, original source paths or historical receipts. The runtime tested
above was parent `e68f9c4`, not this reorganized parent worktree. Historical commands
and budget statements below describe those completed attempts, not current launch
authority or current account balances.

The clean `codex/sglang-v0.5.22` branch starts from local `main` `6fa18c7` and keeps
the same qualified v0.5.22 fork sources (`7856615` / `3eb8594`). It separately
renames the unchanged main v0.5.18 pair. Its parent packaging has only CPU/static
checks: **no image rebuild or GPU rerun was performed after reorganization**.

Raw paths in the records are relative to the archived worktree under the
original repository checkout,
`data/source-snapshots/sglang-v0522-port/`, not this new worktree. Full logs, JUnit,
binary manifests and failed-attempt artifacts remain there under
`raw/v0522-qualification/` and in Modal volume `remnant-stage2a-results`.
Only these three compact files are curated here. HiCache GPU, full-model,
distributed execution and serving remain pending.

## Scope

- Actual `Dockerfile.v0.5.22`, one H100/SM90, no model weights or model load.
- Pinned 0731 configuration; one TP4 rank's geometry, not distributed execution.
- Same **131,072 total original-token capacity**, 32 request slots plus dummy,
  43 target layers including 21 CSA layers. Not 131k tokens per request.
- Native, Packed Triton and Packed Kernel use identical pool capacities.
  Packed Kernel/HiCache remain 256-only; Triton checks retain 64..448 by 64.
- Stop after correctness/storage reconciliation. No serving, timing or profiling.

## Matched-capacity allocation

Attempt005/007/008/010/021 measurements agree exactly, including allocator snapshots.
Unique backing bytes, with aliases deduplicated:

| Mode | Persistent pool bytes | C4 bytes | C4 reduction | Reconstruction workspace MiB, B8 / B24 / B32 |
|---|---:|---:|---:|---:|
| Native | 3,775,862,528 | 405,699,840 | — | 0 / 0 / 0 |
| Packed Triton | 3,597,632,000 | 227,469,312 | 43.93% | 6.301 / 18.902 / 25.203 |
| Packed Kernel | 3,597,632,000 | 227,469,312 | 43.93% | 2.301 / 6.902 / 9.203 |

- **178,230,528 bytes (169.974 MiB) saved per rank:** 4.72% of this complete
  persistent-pool fixture. This does not establish a server-capacity multiplier.
- C4 page64: 512 usable + 4 reserved pages = 33,024 physical rows per layer.
  Padded Native uses **585 bytes/row**, Packed256 328. Logical Native is 584;
  its logical reduction is 43.84%, distinct from physical reduction above.
- Packed planes across 21 layers: values 177,537,024; bitmaps 44,384,256;
  scales 5,548,032 bytes. Native page padding totals 693,504 bytes.
- Unchanged bytes: SWA 2,279,646,720; C128 22,118,400; indexer 91,542,528;
  compressor state 976,855,040.
- All 56 edge allocations reconcile: C4 capacities 0/1/63/64/65/130/32768,
  Native and seven Triton retention counts. Zero capacity still reserves pages.
- Workspaces use K512, one backend reused across layers, excluding common
  attention scratch. Kernel's existing workspace is counted despite Direct
  bypassing reconstruction; it was not removed to improve the result.
- CUDA active-requested bytes match persistent storage exactly. Allocated block
  bytes and reserved segments are separately recorded, not equated to storage.

## Historical baseline validation and reproduction

This section describes the original `0be0962` binary, not current-candidate
qualification. See the October 24 explicit-acquire results below.

Attempt005: Linux CPU preflight **134 passed + 56 subtests**; H100 validity suite
**461 passed + 60 subtests**, no skips. Memcheck's 58 tests passed, but seven
unsuppressed CUDA-Python initialization API errors failed that gate; racecheck
and synccheck did not run. This is not a completed sanitizer qualification.

The vendor image bundled CUDA-Python/bindings 13.4.4. The seven failing call
offsets request ABI 13041 (CUDA 13.4.1), while the toolchain/sanitizer is CUDA
13.0. The shared serving recipe now pins both packages to 13.0.3. Attempt006
stopped on seven pre-existing whole-image dependency conflicts, reproduced in
the unchanged vendor environment; it allocated no GPU. Attempt007 checks the
changed packages' dependencies and all consumers' active base requirements,
then fresh-process import controls and the full suite. Unrelated optional
vision/video/Dynamo/NIXL conflicts remain unmodified. No sanitizer API reporting
is disabled and no errors are suppressed.

Attempt007 passed all validity checks and memcheck (58 tests, zero errors), but
2025.3.1 racecheck reported 21 grouped hazards (9 errors, 12 warnings), including
Native and Packed V4 specializations. Synccheck was not reached. Attempt008
tested a standalone, SHA-pinned CUDA Sanitizer 13.3.75 package without changing
the serving image's AOT binaries, compiler, runtime or kernels. NVIDIA's
[release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
document later Hopper warpgroup and indirect cross-cluster barrier fixes.
The old reports remain failures unless the new strict instrumented run resolves
the gate; numerical passes alone do not establish race safety.

Attempt008 reused the same serving-image/AOT binaries and again passed **134
Linux CPU tests + 56 subtests** and **461 H100-worker validity tests + 60 subtests**,
with zero skips/failures. Compute Sanitizer **2026.2.1** then exhausted the work
deadline during memcheck: 20 progress dots, no final error summary or JUnit.
This is **incomplete**, not a zero-error pass. Racecheck/synccheck did not run.
The diagnostic import controls allow no CUDA initialization and therefore do
not substitute for the full instrumented tests. No tolerances were relaxed,
API/race reports suppressed, or Native attention code changed.

The validity cases require FP32 CUDA-store value/bitmap/scale bytes to match the
reference, exact reconstructed NoPE/seven Native scales, and existing RoPE,
Direct-output/LSE tolerances (`atol=rtol=0.02`). They cover SWA/sink, invalid and
partial rows, page edges, ties/zeros, changing graph inputs, non-default streams
and allocation-free replay. Numerical passes do not replace sanitizer gates.
Attempt007/008 AOT hashes match; their separately CPU-built Native-store JIT
hashes differ and are recorded, not described as byte-identical builds.

| Gate | Historical baseline evidence |
|---|---|
| CPU build, imports, clean pins, SM90 payloads | Passed |
| Numerical/runtime validity, all seven Triton counts, HiCache round trip | Passed |
| Real allocation formulas, 56 edges, three modes and workspaces | Passed |
| Memcheck, 2025.3.1 / 2026.2.1 | Passed 58 tests + 15 subtests / timed out |
| Racecheck, 2025.3.1 | Failed: 21 grouped reports, 9 errors + 12 warnings |
| Racecheck, 2026.2.1, isolated A/B/C | Failed: 116 errors; all three numerical tests pass |
| Synccheck, 2026.2.1, isolated A/B/C | Failed: 5,664 divergent-barrier errors in Native H64; target aborted, no JUnit |
| Packed H64 graph replay alone, 2026.2.1 racecheck | Passed one numerical test, zero hazards; not complete core coverage |

Final local launcher/source-isolation regression: **61 CPU tests passed**;
`REMNANT_INTEGRATION=v0.5.18 python -m unittest scripts.modal.tests`.
JSON/CSV, independent allocation formulas, and all 35 retained artifact hashes
were checked against downloaded results; independent evidence review agreed.

The newer tool's slow progress is observed, not explained; the earlier hazards
have **not** been established as false positives. Diagnose before promotion.

## Attempts and cost

### Follow-up: sanitizer source audit

- Question: do the six reported instruction pairs identify missing synchronization,
  or checker limitations? Map the existing AOT instructions before changing code.
- Baseline: attempt008 serving AOT hashes, unchanged Native/Packed kernels.
- First action: bounded CPU-only inspection of the cached qualification image;
  no extension rebuild, weights, timing, or H100 allocation.
- Stop before any GPU retry unless its conservative projection fits the original
  aggregate limits. A numerical pass or absence of a completed report is not a
  sanitizer pass. Remaining gates and previous failures stay recorded above.

CPU audit009 inspected the **same AOT hash** as attempt007/008, with no rebuild.
Its disassembly maps the 21 grouped reports to six instruction pairs:

| Path | PCs | Shared object | Reports |
|---|---|---|---:|
| Native H64 | `0x230 → 0x780` | Query-barrier init/arrival | 1 warning |
| Packed H64 | `0x1f0 → 0x6e0` | Query-barrier init/arrival | 9 warnings |
| Native H128 | `0x240 → 0x7b0` | Query-barrier init/arrival | 1 warning |
| Packed H128 | `0x240 → 0x7b0` | Query-barrier init/arrival | 1 warning |
| Native H128 | `0x1110 → 0x6770` | Softmax-scale read/reuse | 1 error |
| Packed H64 | `0xaf00 → 0x10a70` | Softmax-scale read/reuse | 8 errors |

- The warnings pair `SYNCS.EXCH.64` with `SYNCS.ARRIVE.TRANS64` on `bar_q`.
  The errors pair `LDS.64` with `STS.64` on shared `sScale`, **not packed KV**.
- Source ownership is inferred from instruction types, offsets and use;
  this binary has no source-line attribution. Ready/free handoff barriers 8/9
  with 256 participants are present in the relevant compiled paths. Their
  presence alone neither proves correctness nor establishes a checker bug.
- No runtime or kernel was changed. The existing launcher now runs v0.5.22
  **synccheck → racecheck → memcheck**. It retains independent tools' results
  after a failure while time remains, but **any failed/not-run tool fails the
  run**. Test selections, tolerances, defaults and deadlines are unchanged;
  v0.5.18 retains its previous tool order.
- Sanitizer gates now require both passing pytest coverage and a clean final
  tool summary. Missing instrumentation, tool warnings or a truncated log cannot
  become a pass. The CPU-only import controls remain diagnostics, not gates.
- Follow-up validation: **65 local CPU tests passed**. Regression fixtures cover
  interleaved warning/progress lines, mixed summaries, malformed JUnit, failure
  propagation and deadlines. Saved attempt007 memcheck is accepted; its
  racecheck and attempt008's truncated memcheck are rejected. Independent review
  verified all six PC mappings and quoted handoff instructions against the dump.
- Raw disassembly is local in `raw/v0522-qualification/sanitizer-audit009/` and
  the result volume; exact instructions and artifact hash are in the JSON.
  CPU audit cost **$0.00169**; reported total **$3.85818**, subject to billing lag.
- **At that point, no further GPU allocation:** two conservative minutes remained, less than the
  last 309-second validity stage alone. A further GPU run needs an increased
  aggregate-time allowance or a separately reviewed, shorter protocol.

### Authorized continuation

The user extended the aggregate H100 ceiling to **60 minutes**, retaining the
**$10 total compute ceiling and $2 reserve**. Attempt010 used the unchanged
kernel sources and stricter launcher `8d041f0`: 900 seconds work, 960 seconds
platform timeout, retries disabled. Previous conservative accounting is 2,580
seconds; this reserves 60 seconds beyond the 960-second cap within the new limit.
Spend before launch was $3.85818, with no active containers. The conservative
$2 projection included CPU preparation, H100 work and export.

- CPU **134 + 56 subtests** and H100 validity **461 + 60 subtests** passed again,
  with no failures/skips. All three compiled binaries match attempt008 exactly;
  the allocation report is byte-identical to attempts005/007/008.
- **2026.2.1 synccheck timed out:** final log has 44 progress dots, no JUnit or
  final error summary. Racecheck/memcheck were not launched. Dots alone do not
  establish completed case coverage or a zero-error result.
- A mid-run nonblocking Python stack sampled input preparation at
  `test_remnant_cuda_graph.py:145`, `raw.copy_(raw.roll(2, dims=1))`, **before
  graph replay**. This localizes that sample, not the underlying cause or the
  whole timeout. The attempted native-stack sample failed because `py-spy`
  rejects `--native` with `--nonblocking`; no native stack was obtained.
- The final launcher exception exposed a sub-second deadline edge: it attempted
  to schedule the next tool with zero whole seconds. Racecheck did not start.
  A focused guard fix and regression preserve the incomplete tool matrix;
  **66 local CPU tests pass**, including expiration between the loop and command
  checks. Independent review passed. No production/kernel/fixture changes or case
  removal; the fix has not been rerun on GPU.
- Next diagnostic: isolate the sanitizer's input-update step before another
  full-suite retry. Kernel changes are not justified by this host sample alone.
  Further GPU work needs new time authorization; no allocation is scheduled.

| Attempt | Outcome | Conservative H100 allocation cap |
|---|---|---:|
| 001 | Modal Docker COPY validation stopped before build | 0 min |
| 002 | CPU C++17/Torch-header failure; fixed to C++20 | 0 min |
| 003 | Allocation passed; validity exposed HiCache/prefill fixture errors | 6 min |
| 004 | CPU image build externally shut down; no compiler error | 0 min |
| 005 | Validity/allocation passed; memcheck failed seven CUDA-Python API calls | 10 min |
| 006 | Whole-image dependency check stopped CPU build; inherited conflicts isolated | 0 min |
| 007 | Validity/allocation/memcheck passed; racecheck failed | 7 min |
| 008 | Validity/allocation passed; newer memcheck timed out | 20 min |
| CPU audit009 | Exact AOT disassembly; no runtime change or H100 allocation | 0 min |
| 010 | Validity/allocation passed; newer synccheck timed out | 16 min |

- Profile `winston-85885`, initially zero compute spend. CLI-reported task/account
  compute **$5.23 as of 2026-10-23 14:09 UTC**, subject to billing lag; this includes
  CPU builds and diagnostic shells. $10 ceiling and $2 reserve were respected.
- **59 of 60 min conservative aggregate allocation accounting**; observed container
  start-to-app-stop intervals total **53 min 55 s**. These include startup/cleanup,
  not only pytest time. No duplicate allocation or automatic retry was launched.
- Attempt010 stopped at **14:05:54 UTC**; container listing subsequently returned
  empty. All qualification apps are stopped. No further launch is scheduled.

Historical attempt010 command from the isolated worktree (not a new launch authorization):

```bash
MODAL_PROFILE=winston-85885 REMNANT_INTEGRATION=v0.5.22 \
REMNANT_VALIDATION_TIMEOUT_S=960 \
python -m modal run --detach scripts/modal/kernels.py::validate_kernels \
  --run-id v0522-qualification-20261023-attempt010 --sanitizer --max-seconds 900
```

Do not reuse this run ID or launch under the remaining one-minute time allowance.
The original 45-minute ceiling was extended to 60; the $10 compute ceiling and
$2 reserve are unchanged. Platform retries remain disabled.

Source pair: SGLang `7856615f3fdf02e993f3b40ecdbbed8cb8ac9bf7`, FlashMLA
`0be0962e1628e27e8f9d8a27995a29ac7dcf8bc2`, CUTLASS
`147f5673d0c1c3dcf66f78d677fd647e4a020219`. Attempt005 parent `a7dfd96`;
attempt006 parent `d63a12c`; attempt007 parent `ff67789`; attempt008 `81705d7`;
attempt010 parent `dbf8a05`, launcher `8d041f0`.
Post-run deadline-reporting fix: `57f210a`.
Original v0.5.18 checkouts and pins are unchanged.

Raw logs, JUnit, allocation and build manifests are retained locally under
`raw/v0522-qualification/attempt-<NNN>/all-results/validation/` and in Modal volume
`remnant-stage2a-results`, `microbench/v0522-qualification-20261023-attempt<NNN>/validation/`.
Only compact evidence belongs in this collection; raw logs are not Git inputs.
The JSON preserves the initial attempts001–008 snapshot and appends the CPU
audit and attempt010 continuation, including hashes, identities, coverage and
costs. Full logs/XML remain local and in the volume.

## Limits and rollback

### October 24 authorized V4 barrier A/B

- **Question:** do contract-correct V4 barriers resolve the completed Native
  synchronization failures and KV/output-union races? The user approved this
  correction and asked to finish on Modal before moving to the VDI.
- **Mechanism:** use unaligned 256-consumer barriers in divergent V4 warpgroup
  paths; use the same unaligned final CTA barrier for Native H64's 384 threads;
  use release/acquire for the final V4 H128 cluster handoff. Keep counts, IDs,
  phases, WGMMA/TMA waits, numerical arithmetic and layouts unchanged. Non-V4
  paths and Packed H64's already-passing instructions are preserved.
- **Baseline:** retain the original `0be0962` image and all failed raw evidence.
  This candidate changes Native synchronization, not the untouched control's
  historical identity. No performance claim follows from a sanitizer pass.
- **Execution:** commit the fork and parent pin, build the exact serving recipe
  and qualification preflight on CPU, then isolate Native H64 synccheck before
  H128, graph replay and the broader core three-tool gates. HiCache remains
  deferred. Require completed pytest/JUnit and clean final tool summaries; no
  suppressions, missing-case passes or tolerance changes.
- **Limits:** use `winston-85885` while eligible (fresh compute $5.98558599,
  no active containers before preparation); preserve exclusions, the $28 total
  ceiling and $2 reserve. Bound every allocation and retain unique attempt IDs.
  Stop on a failed preflight or unresolved gate; do not launch full models as
  a workaround. Candidate results are pending.
- **Latest instruction:** finish all core numerical and sanitizer coverage on
  Modal before VDI testing. The new image passed CPU preparation; binary
  inspection preserves Packed H64 and all four non-V4 sparse functions exactly.
  Core coverage is 457 numerical cases (only the four explicitly deferred
  HiCache GPU cases omitted) and 58 cases per sanitizer, including an additional
  H128 changing-graph case. Keep the $28 total/$2 reserve; reallocate the initial
  resource envelopes to these model-free checks rather than model loads.
  Partition long sanitizer selections into bounded, checkpointed groups if
  needed; their union must equal the complete frozen selection for each tool.
  Incomplete attempt016 is retained. Attempt017 adds visible pytest progress
  and stack dumps, without changing the candidate, checks or tolerances.
- **Budget extension:** the user authorized another $10 on October 24 while
  attempt019 was running: **$38 aggregate**, including prior spend, with $2
  reserved. Existing account exclusions and the $28 per-account ceiling remain;
  switch accounts if necessary, never duplicate a live allocation.
- **Barrier-only result (`31877b4`):** attempt017 passes modern synccheck on
  Native H64/H128 parity fixtures. Attempt018 times out after late numerical
  progress, without JUnit; attempt019 checkpoints **457 numerical tests + 60
  subtests**, all passing, and reconciles the allocation again. However its four
  Direct cases fail modern racecheck with **208 errors**: Native H64 48, Native
  H128 48, Packed H128 112. Numerical passes do not clear those reports.
- **Second isolated mechanism:** give WG1 the same explicit V4 local-ready
  acquire already used by Packed H64, plus peer-ready acquire for H128.
  Reports pair producer KV stores with WG1 output stores in the aliased union.
  Preserve barrier arrivals/counts, phase progression, arithmetic and non-V4
  paths. Independent review finds no dependency cycle or phase mismatch.
  Build on CPU, then rerun Direct racecheck before broad qualification;
  completed results follow. Retain all earlier failures and source identities.
- **Explicit-acquire result (`3eb8594`):** CPU audit020 preserves five
  of eight sparse functions exactly. Attempt021 passes the same four Direct
  racecheck cases with **zero hazards/errors/warnings**, then all **457 core
  numerical tests + 60 subtests** and exact allocation reconciliation. Attempt022
  completes all 49 codec/reference racecheck cases cleanly in 663.4 seconds.
  Its graph group is deliberately interrupted after 360.4 seconds: only three
  of eight subcases in the first test have completed, so the fixed 500-second
  group cap cannot cover all five tests. This is incomplete, not a numerical
  failure or sanitizer pass. Attempt023 checkpoints graph functions separately
  with larger bounded deadlines; no fixture/tolerance or binary changes.
- **Completed racecheck:** attempts021–024 together cover exactly 58 cases +
  15 subtests, with no duplicates, omissions, skips, hazards, errors or warnings.
  Only completed checkpoints are accepted, even when a later group in the same
  allocation times out. All use the same source, three binaries and modern tool.
- **Synccheck progress:** attempt025 passes 49 codec/reference cases.
  Attempt026 passes store/reconstruction, mixed-layer and fused-read graphs
  (three cases + 15 subtests), then its four-case Direct checkpoint times out
  at 450 seconds. Its unfinished Direct progress is not accepted. Attempt027
  retries only Direct and the two remaining CUDA-store graph cases, with
  separate 900/350-second bounds. No source, fixture or tolerance changes.
- **Completed synccheck:** attempt027 passes Direct's four cases and both
  CUDA-store graph cases. Its union with completed checkpoints025/026 is exactly
  58 unique cases + 15 subtests, zero errors/skips, same manifest and binaries.
  Direct takes 540.8 seconds: the previous 450-second bound was insufficient.
  Attempt028 starts checkpointed memcheck on unchanged codec/reference cases.
- **Memcheck checkpoints:** attempts028/029 pass 51 cases + 8 subtests.
  Store/reconstruction's eight subcases take 948.5 seconds. Attempt030 times out
  in Direct's four-case group at 900 seconds, without final JUnit/tool summary;
  none of its partial progress is accepted. Attempt031 checks eligible account
  `poohthewinnie` ($13.28898377), but image access is denied before any sandbox
  allocation. No alternate binaries are substituted. Attempt032 reuses the
  exact image on `winston-85885` ($24.76424133), with a 1,500-second work deadline
  and 1,560-second platform cap. Its $3.10 conservative projection fits below
  the $28 account ceiling and the new allowance; completed groups are separate
  checkpoints. No automatic retries, skipped cases or weakened tolerances.
- **Completed memcheck:** attempt032 passes fused-read (all seven budgets),
  both CUDA-store graph cases, both Direct parity cases and H64/H128 Direct
  graph replay. Its union with attempts028/029 is exactly **58 unique cases +
  15 subtests**, zero errors/skips. The earlier incomplete Direct030 group
  contributes nothing. All three tools have the same full-identity-set hash;
  raw logs, JUnit and manifests are downloaded locally. Only compact aggregates
  and artifact hashes enter Git. This closes core model-free qualification,
  not HiCache, a full-model boot or serving validation.
- **Latest authorization:** the user grants a new **$15 continuation envelope**
  starting from fresh account compute **$18.18047701**, with **$2 reserved**.
  No containers were active. This is an additional bounded allowance, not a
  reason to consume prior unused budgets. Keep every account below $28 and
  switch eligible accounts if necessary; exclusions are unchanged. Reuse image
  `im-vOlukc1AW0gozz9mlAmg1m`; do not rebuild or rerun completed gates.

### Historical October 24 continuation: before barrier corrections

This preserves the earlier authorization and failed-baseline investigation.
The corrected candidate and later budget extensions are recorded above;
full-model verification has not been run.

- Approved total compute ceiling: **$28 including the $5.22929253 already
  reported**, with $2 reserved for billing lag/export/cleanup. Retain account
  exclusions, start below $25 and project each account below $28. No automatic
  retries or duplicate allocations. The earlier 60-minute allowance is replaced
  by this explicitly authorized, bounded continuation.
- First audit the cached image on CPU: exact pins are clean, but a local
  full-tree FlashMLA digest differs from the image receipt. Reconcile individual
  files before GPU reuse. Reuse cached weights and AOT binaries only when verified.
- CPU audits012/013 reconciled the mismatch: every one of the 6,915 shipped
  FlashMLA source files is byte-identical. The local-only nested `cutlass/.git`
  pointer is excluded by Docker, not a changed build input. All 48 checkpoint
  shards have the pinned revision receipts (166,886,535,336 weight bytes).
  Audit011's CLI image-ID syntax failed before allocation; audit012 wrote valid
  evidence but its unnecessary in-sandbox SDK commit lacked credentials. Normal
  shutdown persisted the evidence; audit013 completed successfully without it.
- Then isolate sanitizer slowdown with labeled progress, fresh processes and
  small existing fixtures, preserving all core store/reconstruction/Direct
  cases and strict final summaries. No filters, suppressed hazards or relaxed
  tolerances. HiCache qualification is explicitly deferred by the user.
- Diagnostic014 reuses the verified cached image, one H100, a 600-second
  platform cap and 540-second work deadline. First run a bounded plain-Torch
  roll/copy control under modern synccheck, then the three existing Direct
  cases A/B/C from the race audit under modern racecheck. Preserve full logs,
  verbose pytest, automatic host stacks and nonzero timeout status. This is
  diagnostic coverage only, not a substitute for the complete core gates.
- Diagnostic014 finished in 219.6 seconds: plain Torch roll/copy passed modern
  synccheck in 4.0 seconds; all three Direct numerical tests passed, but modern
  racecheck failed with **116 errors, zero warnings** (100 displayed by its
  default limit). Displayed paths include Native H64/H128 and Packed H128.
  Therefore the earlier reports cannot be dismissed as an older-tool issue.
  Diagnostic015 keeps binaries unchanged and isolates synccheck on A/B/C plus
  racecheck on C alone, with all findings displayed and a five-minute cap.
- Diagnostic015 finished in 183.6 seconds. Synccheck reports 5,664 divergent
  barriers in Native H64 and the target aborts before publishing JUnit. The
  first PC, `0x1ea0`, maps to `BAR.SYNC.DEFER_BLOCKING 0xa, 0x100`: a
  256-consumer named barrier inside partial-CTA branches. The independent
  Packed H64 graph-replay case passes modern racecheck (one test, zero hazards).
- Modern race reports concern the **aliased KV/output shared-memory union**;
  they are not simply repeats of the older query-barrier/softmax-scale reports.
  A minimal A/B candidate would use unaligned consumer/full-CTA rendezvous and
  release/acquire cluster handoff. These have **not been changed or validated**.
  Native correction requires the requested user permission because it changes
  the previously untouched baseline. No full model is launched to bypass it.
- Parent source-identity fix `25c6be9` excludes nested `.git` pointer metadata,
  matching Docker's existing exclusions. **67 local CPU tests pass**, with
  independent review; shipped source hashes now match the cached image directly.
  Runtime/kernel code and fork pins remain unchanged. Raw012–015 evidence is
  downloaded under `raw/v0522-qualification/`; only compact aggregates enter Git.
- Reported compute is **$5.98558599** including prior work; no containers remain.
  New GPU diagnostic function times total 403.2 seconds; conservative platform
  caps reserve 15 of the approved 30 additional single-H100 minutes.
- Only after core memcheck/racecheck/synccheck pass, verify Native, Packed Triton
  and Packed Kernel sequentially on TP4, one model load per mode and one live
  server at a time. Use the planned short/32k/128k, changing-batch and shared-prefix
  requests; preserve exact fixed output lengths and verify actual graph/dispatch
  evidence. Compare Packed implementations, not intentional pruning against
  Native. No performance or task-quality promotion from smoke tests.
- Initial resource envelopes: at most 30 additional single-H100 minutes and
  60 TP4 minutes, reduced whenever fresh spend projections require it. CPU
  image preparation occurs first; checkpoint each result and stop allocations
  on failure. If a gate cannot be resolved within the ceiling, report the
  unresolved blocker; do not run full models to bypass it.

Matching Packed mathematics does not mean matching untouched Native outputs or
task quality. Full-model startup, live server HiCache, usable token/concurrency
ceilings, TPOT/TTFT and throughput remain unqualified. Unset
`REMNANT_INTEGRATION` to retain v0.5.18; never mix either profile's binaries/pins.
