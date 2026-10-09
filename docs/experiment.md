# Attention-aware pruning calibration on Modal

## Goal and scope

Test the original Remnant hypothesis: **equal-sized errors in raw C4 coordinates
can cause different attention-output errors.** Use that evidence to compare
TopMag50 with a layer-specific attention-aware score:

\[
s_{l,i}(c)=|c_i|\sqrt{h_{l,i}}.
\]

The first decision is whether this score reduces held-out attention-output error
while retaining exactly **256 of 512 coordinates**. Only then investigate smaller
survivor budgets.

- Use the current production fork and DeepSeek-V4-Flash-0731.
- Collect real attention states once; run ablations and calibration offline.
- Keep CSA selections, queries, SWA, the indexer, attention math, and model weights
  fixed during replay.
- Do not change pruning/RoPE order, the CUDA kernel, or the 328-byte ABI.
- Do not treat a local attention-error improvement as an end-to-end quality or
  TPOT improvement.

## Execution status — 2026-10-06

### Fresh held-out validation — 2026-10-07

- User-approved first cohort: **50 issues**, selected as the first 50 in the
  existing seeded scan, without examining inference outcomes. The interrupted
  100-issue search and its 54 selections remain preserved separately.
- Capture uses a fresh `validation50` prefix, TP4 with continuously replenished
  concurrency up to ten, natural completions, and per-issue checkpoints.
- Keep total compute below $30: use the existing $28 account ceiling, a $3
  shutdown/export reserve, fresh spend checks, and a rate-derived time cutoff.
  A stopped partial cohort is retained but receives no significance verdict.
- The frozen optimized mask is compared with TopMag50 offline on native
  captures; this is not a live optimized-policy quality benchmark. Bootstrap
  the entire predeclared 50-issue cohort, not a budget-selected subset.

- Account: `fxcai21`, verified **$0 initial compute spend**, with cached pinned
  0731 weights. No GPU launch until the complete fresh cohort and CPU gates pass.
- Freeze the existing derivative-mask weights; do not refit on these issues.
  Use the revised **100,000–135,000 token** input band (minimum lowered from
  120,000 at the user's request), exclude prior repositories, and collect
  all 21 CSA layers / 64 heads through natural completion.
- Use a two-request oracle pilot, then a continuously replenished queue of
  at most ten requests. One model load; compiled extensions built on CPU.
- Commit hash-checked captures after each completed issue. Validate/export on
  CPU while capture continues; download compact per-issue results locally.
  Raw snapshots remain durable in the volume without needing GPU time to export.
- Enforce the $28 account ceiling with a timed allocation, fresh CLI billing,
  shutdown margin and $3 CPU/export reserve. Preserve partial results, but do
  not give a significance verdict for a budget-selected incomplete cohort.
- Current state: `experiments/attention-pruning/validation50-record.json`.
  The superseded 100-issue search is preserved in `validation100-record.json`.
- Local dataset-viewer connections repeatedly reset. A CPU-only Modal transport
  fallback fetches bounded batches of missing pages, verifies the pinned revision
  and complete rows, then downloads them into the immutable local cache. Selection
  and tokenization stay local; seeded order, saved issues and numerical policy
  remain unchanged. Source recovery stops at a $0.75 compute threshold or a failed
  integrity gate; it does not retry GPU allocations. Fetch sequentially with five
  seconds between requests; commit each page. On 429, release the remote CPU
  worker, recover partial pages and wait locally for Retry-After before continuing.

### Original ten-issue capture

The frozen ten-issue cohort, capture hooks, offline runner and Modal entry
points are implemented. Attempt **004** completed its TP4 capture on
`poohthewinnie`, started at 19:37:23 UTC with a 60-minute cap. All 19 H100
tests passed; the two-issue pilot passed the original native-output/LSE
oracle across all 21 CSA layers and 64 heads. All ten requests finished
naturally; Numba generated 5,050 tokens. The TP4 app stopped at 20:14:52 UTC.
Capture took 2,233.48 seconds / 2.4816 H100-hours, including startup.
The initial model-free L4 run reached its cap after 12 complete layers;
the checkpoints are retained. A CPU-tested resumed L4 run started at
20:59 UTC, preloading immutable snapshots and rechecking every reused oracle.
All math/statistics function ASTs match the original runner. The resumed run
completed at 21:10 UTC: all 2,016 native oracle checks passed, all 21 layers
were analyzed, and held-out continuous/packed results are local. H-TopMag
worsens packed issue-balanced mean error by 5.49% and p95 by 15.55%; do not
promote it. Both L4 attempts used 0.6713 L4-hours. The corrected CPU archive
completed at 21:03 UTC. All raw captures and analysis are now local; an 8-MiB
transfer-corrupted block was recovered from the intact remote original and
the repaired 11.1-GiB archive passed full gzip integrity. All 2,016 states
also passed an independent local CPU oracle and coverage/source/payload audit.
Execution is complete; the policy-improvement gate fails. Numerical findings
are in `experiments/attention-pruning/report.md`.

Attempts 001–003 exposed dummy-batch, request-ID/tokenization and TP row-order
assumptions. These were corrected and regression-tested; all 399 saved pilot
states from attempt 003 pass the unchanged oracle. Frozen histories, split and
tolerances remain unchanged. Failure details and costs are in
`experiments/attention-pruning/run-record.json`; raw artifacts stay outside Git.

Account compute was **$12.7840** before the L4 launch, excluding storage;
no other allocation was active. The projected total with offline analysis,
export, download and reserve was below $16. Before the resumed L4 launch,
compute was $13.3365; its conservative updated ceiling including reserve and
download was $19. Latest compute is $13.7186 on `poohthewinnie` and $15.5929
on `peroace360`, excluding storage and before credits. All GPU jobs are
stopped; no further retry is needed. The concise report and compact evidence
are complete; raw captures stay under the local cache. The recurring
follow-up is disabled, with no commit/push or unrelated edits.

## Current code and numerical contract

Paths below are relative to the repository root.

| Responsibility | Current source |
|---|---|
| Exact-global TopMag mask; Torch packing and reconstruction | `third_party/sglang/python/sglang/srt/remnant/reference.py` |
| Persistent packing wrapper and Triton implementation | `third_party/sglang/python/sglang/srt/remnant/packed.py` and `third_party/sglang/python/sglang/srt/remnant/triton/kernels.py` |
| Raw compressor output, before pruning/norm/store | `third_party/sglang/python/sglang/srt/layers/attention/dsv4/compressor_v2.py` |
| Selected physical locations and FlashMLA consumer | `third_party/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py` |
| Query preparation, head slicing, and inverse output RoPE | `third_party/sglang/python/sglang/srt/models/deepseek_v4.py` |
| Native paged-cache readback | `third_party/sglang/python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py` |
| Shared server launch; existing Modal image and volumes | `scripts/local/run-server.sh`; `scripts/modal/app.py`; `Dockerfile` |

### What is actually being pruned

For a raw compressor row \(c\in\mathbb R^{512}\), the packed path currently does:

1. Compute the mask from **unmodified raw** \(|c_i|\).
2. Zero discarded coordinates.
3. Recompute learned weighted RMSNorm over all 512 coordinates, including zeros.
4. Round the normalized values to BF16.
5. Quantize eight 64-coordinate tiles to E4M3 with UE8M0 scales.
6. Store 256 retained FP8 bytes, eight bitmaps, and eight scales: **328 bytes**.
7. Reconstruct and rotate the 64-coordinate tail at attention time.

Important distinctions:

- K and V share the full **512-dimensional** vector: 448 NoPE + 64 RoPE.
- The output's 64-coordinate tail receives inverse RoPE at the query position,
  before `wo_a`/`wo_b`. Measure the output at that point.
- Native storage is **584 logical bytes per row**, with FP8 NoPE and a BF16 RoPE
  tail. Its precision contract is not identical to packed storage.
- The current server selector is `--dsv4-c4-cache-format native|remnant`.
  `run-server.sh native` selects the untouched layout; `packed` selects Remnant.
  Do not use historical `SGLANG_OPT_TOPMAG_*` flags.
- The current metadata requires a 256-token base page, making C4 pages
  **64 compressed rows**. Read actual geometry from the running pool; do not
  reuse the historical page-size-16 fixture.
- On SM90, current Remnant serving reads packed rows through the direct
  FlashMLA path. This experiment is about the pruning policy, not selecting
  an old fused-adapter implementation.

TopMag minimizes raw-coordinate reconstruction error at a fixed survivor count.
It does **not** by itself minimize error after masked RMSNorm, quantization, or
attention. All candidate scores remain in the raw \(c\) coordinate basis.

## 1. CPU preflight before reserving GPUs

### Freeze the experiment

Create one run manifest containing:

- Model `deepseek-ai/DeepSeek-V4-Flash-0731`, revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
- Parent, SGLang, and FlashMLA revisions **plus content hashes of dirty sources**.
  HEAD alone does not identify the current implementation.
- Image identity, extension hashes, software versions, server arguments, and
  the input-data fingerprint.
- Seed **42** for trace selection, train/held-out split, and row sampling.
- **All 21 CSA model layer IDs: 2, 4, …, 42**, checked against
  `config.compress_ratios == 4`. Use 2, 22, and 42 only for representative
  report/figure rows, not to limit analysis coverage.
- Capture quota, sampled decode positions, ablation chunk size, and the
  account-specific wall-time limit.

Do not guess the layer IDs or assume every attention layer is CSA.

### Prepare the trace cohort

Use the **10 selected OpenHands traces**, with one issue per repository and a
frozen **6 calibration / 4 held-out** split before collecting tensors.

- Select a small, fixed set of recorded requests from each trace. Prefer
  representative long-context requests, with a primary band around 128k.
- Store the original messages/tool history, trace ID, request ID, prompt token
  count under the pinned tokenizer/chat template, and replay parameters.
- Replay each recorded request independently using its exact recorded context.
  Do not append newly generated outputs to another recorded request.
- Remove the saved 128-token preparation limit in a derived replay request,
  leaving the frozen original untouched. Generate until natural EOS/tool-call
  completion; reject context-limit termination. Sample the complete decode
  continuation with a seeded bounded reservoir, not just its first 128 tokens.
- No task-specific shared history or near-duplicate issue may cross the split.
  Common OpenHands system instructions and tool schemas remain unchanged.
- Existing replay logs or the 4,916-request workload are **not** tensor captures.
  First verify that the chosen messages and trace boundaries are available.
  They are not currently bundled in the Modal image.

Record cohort size and coverage without putting prompt or tool payloads in this
plan:

| Split | Independent traces | Requests | Query states planned / captured | Prompt token range |
|---|---:|---:|---:|---|
| Calibration | 6 | 6 | 58 / 58, per CSA layer | 121,073–133,203 (serving) |
| Held-out | 4 | 4 | 38 / 38, per CSA layer | 120,584–125,476 (serving) |

#### Completed CPU preparation — 2026-10-05

- Source: `nebius/SWE-rebench-openhands-trajectories`, revision
  `35455389ab51bf5e2306bfd436ef72d0f98bf882`.
- Used the pinned 0731 tokenizer and the current SGLang V4 encoder, including
  complete tool schemas/results; **non-thinking mode, official low effort**.
- Stopped at exactly **10 qualifying issues** after inspecting 4,715 rows.
  Selection band: 120,000–135,000 tokens; median: **123,478.5 tokens**.
- Each saved request ends immediately before an original assistant response.
  No padding, concatenation, message truncation, or unresolved tool calls.
- Re-counted all saved requests and re-fetched all ten original source rows,
  checking the pinned revision and exact message/tool history.
- All four official encoder fixtures and six preparation unit tests passed.
  Issue-description five-word shingles had no overlap across selected issues;
  this is duplicate screening, not semantic decontamination.
- Eight source trajectories were unresolved and two resolved. This is a
  deliberately long-context **frozen-history reconstruction cohort**, not a
  representative traffic sample or V4 coding-success evaluation.
- CPU selection took **1,126.32 s (18m 46s)**, excluding earlier pilot/setup and
  the final audit. **No Modal/GPU usage or model-weight download.**

| Issue | Input tokens | Split |
|---|---:|---|
| `keras-team__keras-nlp-131` | 127,531 | Calibration |
| `ml31415__numpy-groupies-48` | 121,053 | Calibration |
| `aesara-devs__aesara-549` | 120,564 | Held-out |
| `keras-team__keras-cv-1596` | 123,977 | Held-out |
| `BoboTiG__ebook-reader-dict-2149` | 121,342 | Calibration |
| `jackdewinter__pymarkdown-886` | 133,183 | Calibration |
| `numba__numba-5627` | 125,456 | Held-out |
| `gerbenoostra__poetry-plugin-mono-repo-deps-24` | 122,980 | Held-out |
| `spacetelescope__jwst-3629` | 121,903 | Calibration |
| `exo-lang__exo-542` | 127,601 | Calibration |

Local cohort directory:

```text
/Users/poohthewinnie/.cache/remnant/attention-pruning/openhands-0731-cohort10-20261005/
├── requests.jsonl  # exact requests, source IDs, lengths, prompt hashes, split
├── manifest.json  # pinned revisions, encoder/tokenizer hashes, selection record
└── audit.json     # final source verification, token re-counts, duplicate checks
```

Preparation code: `experiments/attention-pruning/prepare_openhands.py`; usage and
CPU tests: `experiments/attention-pruning/README.md`. Raw histories stay outside
Git. Stage these exact requests in Modal for capture; do not reselect or append
generated responses. If reasoning settings change, re-render and re-count.

Frozen preparation settings: `temperature=0`, `seed=42`, `max_tokens=128`,
`reasoning_effort=low`, `chat_template_kwargs.thinking=false`. Capture removes
`max_tokens`/`max_completion_tokens` from its derived replay only. All other
settings, original messages, tools, seed, and split are unchanged.

Natural completion of these independent requests is not a complete agentic
benchmark run: no generated tool actions execute and no new agent turn is fed
back. This experiment freezes states from the untouched model.

### Check image, data, and account readiness

Use the existing image/volumes in `scripts/modal/app.py`:

- `Dockerfile` already compiles the local SM90 extensions during image build.
  Verify their identity and availability on CPU before launch.
- Reject a missing extension instead of spending H100 time on the current
  launcher's runtime-build fallback.
- Check the chosen account's `deepseek-v4-flash-0731` volume for the pinned
  checkpoint and required shards. If missing, use the existing CPU-only
  `download_model` first; model download must finish before GPU allocation.
- Stage the cohort and manifest under a new results prefix. Never overwrite
  earlier experiments.
- Check current **compute** spend and other active allocations. Exclude
  `caiw`, `winstoncai`, and `winstoncai233`; do not start on an account above $25
  or allow projected spend to reach $28.

A missing trace cohort, checkpoint, or usable image stops the experiment here,
before GPU costs.

## 2. Collect real attention states once on TP4

### Capture configuration

Use one **4×H100, TP4** allocation and one untouched native server:

- Launch `scripts/local/run-server.sh native` directly inside the existing
  Modal image. The caller owns one process group, logs, readiness, and cleanup.
- Do not invoke Docker or a second server launcher inside Modal.
- Set `MAX_RUN=10`. Run a two-request pilot concurrently, validate all 21 layers
  and four ranks, then submit the remaining eight together without repeating
  the pilot. The actual scheduler may mix prefill and decode or admit fewer.
- Use `MEM_FRAC=0.88` and `CHUNK=8192` initially, reserving headroom for capture.
- Set `CTX_LEN=1048576` for EOS headroom, without imposing an output-token cap.
- Disable radix-cache reuse and overlap scheduling for capture ownership only.
- Disable both graph phases **only for this instrumented capture**:

```bash
DECODE_CFG='{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}'
```

Python observation hooks are not replayed on every CUDA-graph execution.
Disabling graphs prevents silently capturing only graph construction. This
capture's serving latency is not a performance result.

### Capture point A: raw compressor output

In `CompressorBackendMixin._forward_compress_all_in_one`, observe
`kv_compressed` immediately after `compress_forward(...)` and before pruning,
RMSNorm, RoPE, or quantization.

- Observe all 21 model layers with `compress_ratio == 4` and
  `is_indexer == False`.
- Pass capture identity from `forward_unified`, which knows the actual model
  `layer_id` and `ForwardBatch`. The optional `packed_layer_id` is unset for
  native storage and cannot identify native captures.
- Save the compressor's `norm.weight` and `norm.variance_epsilon`.
- Maintain an observation-only raw-row shadow for the active request.
- Map raw rows using the **same write plan and out locations as the producer**.
    - Decode: a valid write occurs at the compression boundary; use the plan's
      sequence length and the matching destination location.
    - Prefill: follow the plan's ragged-row mapping and write validity.
      The state-buffer write location is not the C4 cache destination.
    - Record the row's raw logical index and RoPE position. For C4, the latter
      is `seq_len - 4`, equivalently `4 × raw_c4_index` for a valid row.
- Track request ownership and row generation so page reuse cannot associate
  an old raw row with a new physical slot.
- Do not infer raw rows by undoing native FP8 quantization: that information
  has already been lost.

Keep the shadow bounded to live cohort requests and the 21 CSA layers, not the
maximum-capacity pool. Store raw rows in pinned CPU memory, not on GPU: the ten
prepared prompts require approximately 6.24 GiB per rank before metadata and
continuations. Stream-ordered copies retain sources until completion, then
release GPU ownership. A bounded two-job writer queue provides backpressure.

### Capture point B: actual decode consumer inputs

In `DeepseekV4AttnBackend.forward`, snapshot the actual consumer inputs and its
output immediately after `flash_mla_with_kvcache`, before output inverse RoPE.

Save:

- Already-prepared/rotated queries and the actual softmax scale.
- CSA physical page indices, raw logical indices/positions, and admitted
  top-K lengths.
- SWA indices, lengths, and selected decoded native SWA rows.
- Selected native C4 bytes/decoded rows, using the real page size and byte stride.
- Attention sink, scheduler metadata, and any padding/head-slice information.

For native mode, raw-index metadata may not be populated. Resolve positions from
the producer shadow's physical-location/generation map and verify the association;
do not fabricate positions from the physical page number.

Mask truncated/invalid indices **before** passing locations to native readback
helpers. Reproduce the consumer's admitted-entry semantics rather than treating
all negative entries as an identical case.

### Capture point C: the output being measured

Save both:

- `bar_o`: attention output before inverse query-position RoPE.
- `O`: the real-head output after inverse RoPE, before `wo_a`/`wo_b`.

V4-Flash has 64 real query heads; TP4 supplies **16 per rank**. Save all four rank
slices and assemble those 64 real heads offline. Exclude padded heads entirely.

Verify that raw compressor rows and shared KV contents agree across ranks before
deduplicating them. A rank-0-only pilot covers 16 heads and must not be presented
as a full-head calibration.

### Snapshot identity and quota

Every snapshot needs:

- Trace/request ID, model layer ID, pool layer ID, TP rank/head range, and decode
  position. Use `ForwardBatch.rids` and request-pool metadata where available.
- Raw \(c\), native decoded C4/SWA, raw positions, physical locations, row
  generations, validity, lengths, RoPE table entries, norm parameters, and sink.
- Captured pre/post-inverse outputs and source/config fingerprints.

Initial target per layer:

- **96 unique query states**, spread across the cohort and decode positions.
  Keep the nine lowest seeded hash priorities for the first two sorted issue
  IDs in each split, and ten for the others. This samples each complete natural
  continuation, using the same reservoir ordinals on all layers and ranks.
  Short continuations retain fewer snapshots and are reported as shortfalls.
- **8 uniformly sampled valid CSA rows per query** for ablation:
  **768 query/target-row instances per layer**.
- Save all selected C4 rows for each query, not only those eight targets; the
  whole selected set is needed for the pruning comparison.

Track capture completeness before running offline analysis:

| CSA layer | Model layer ID | Pool layer ID | Query states planned / captured | Ablation rows planned / captured | All 64 heads assembled | Raw-row mapping verified |
|---|---:|---:|---:|---:|---|---|
| All CSA | 2, 4, …, 42 | resolved by actual pool | 96 per layer / pending | up to 768 per layer / pending | pending | CPU mapping tests pass |

An instance is one real-head query set plus one target row—not a full model run.
Each instance yields 512 coordinate tests. Sample rows uniformly rather than
silently biasing calibration toward high-attention rows. Any attention-mass
stratification is a separately labeled diagnostic.

Copy selected states into owned buffers while their producer/consumer data are
still valid. Use correct stream/event ordering; drain bounded snapshot batches
rather than synchronizing and copying the full cache every token. Disable
cross-request prefix reuse for the capture, or explicitly restore raw shadow
ownership for cache hits. Fail on a missing raw row.

## 3. Validate a faithful offline attention oracle

Use two clearly separated reference spaces:

1. **Continuous raw-coordinate oracle:** full-precision weighted RMSNorm and
   RoPE for CSA, with captured queries and fixed SWA. Use this for derivatives
   and the core sensitivity hypothesis.
2. **Native-byte oracle:** captured native C4/SWA decoded exactly from their
   storage format. Use this to reproduce captured native attention and as the
   untouched baseline for the packed comparison.

Do not mix the continuous oracle's output with the native-byte oracle's error
denominator.

### V4 attention equations

For raw CSA row \(c_t\), define

\[
N(c_t)=a_t Wc_t,\qquad
a_t=\left(\frac{\|c_t\|_2^2}{512}+\epsilon_{\rm norm}\right)^{-1/2},
\qquad u_t=R_tN(c_t).
\]

Here \(W\) is the learned diagonal compressor RMSNorm weight and \(R_t\) is the
actual partial RoPE transform, identity on the first 448 coordinates.

For head \(h\), use the captured, already-rotated \(q_h\). Let \(\mathcal A\)
contain **both admitted SWA and CSA entries**:

\[
z_{h,j}=\gamma q_h^\top u_j,\qquad
Z_h=e^{a_h^{\rm sink}}+\sum_{j\in\mathcal A}e^{z_{h,j}},
\qquad p_{h,j}=\frac{e^{z_{h,j}}}{Z_h},
\]

\[
\bar o_h=\sum_{j\in\mathcal A}p_{h,j}u_j,\qquad
O_h=R_{\rm query}^{\top}\bar o_h.
\]

- Match `gamma` and the sink exactly; the sink contributes to normalization
  with a **zero value vector**.
- Use stable log-sum-exp, not exponentiation of unshifted large logits.
- Do not renormalize the non-sink probabilities to sum to one.
- Preserve duplicate selections and actual admitted lengths.
- Apply inverse RoPE only to the output tail at the captured query position.

In exact arithmetic, this inverse rotation preserves L2 output-error norms.
Include it anyway to verify the real output contract and numerical rounding.

### Replay correctness gates

Before fitting any weights:

- Native-byte oracle output matches the captured consumer output under
  `atol=0.02, rtol=0.02`; investigate systematic failures rather than loosening
  tolerance.
- Compare LSE where available using the same log convention and `atol=0.01`.
  Do not assume an API's LSE includes the sink without checking its definition.
- Test small fixtures for a dominant sink, empty CSA, partial lengths,
  duplicates, page boundaries, invalid entries, and large RoPE positions.
- Match inverse RoPE and real-head slicing separately.
- Spot-check FP32 oracle results against FP64 on small fixtures.

The raw-coordinate oracle is intentionally continuous; differences from captured
native quantization are measured separately and are not failed derivative tests.

Record the oracle checks before proceeding to ablation:

| Check | Cases covered | Maximum output error | Maximum LSE error | Pass / fail | Notes |
|---|---|---:|---:|---|---|
| Native-byte replay vs captured consumer | Early / middle / late CSA | TBD | TBD | TBD | TBD |
| Continuous FP32 vs FP64 fixture | Small fixtures | TBD | N/A | TBD | TBD |
| Sink, empty CSA, and partial lengths | Dominant sink; empty; partial | TBD | TBD | TBD | TBD |
| Index and layout semantics | Duplicates; invalid entries; page boundary | TBD | TBD | TBD | TBD |
| Positional handling | Large RoPE positions; inverse output RoPE; head slicing | TBD | N/A | TBD | TBD |

## 4. Single-coordinate ablation

For each sampled target row and coordinate \(i\), replace the raw row with

\[
c_t^{(-i)}=c_t-c_{t,i}e_i.
\]

Recompute **that row's RMSNorm and RoPE**, then the same attention. All other raw
rows, queries, selections, SWA, and the sink remain fixed.

Measure

\[
E_i=\sum_{\text{64 real heads}}\|O_h-O_h^{(-i)}\|_2^2,\qquad
R_i=\frac{E_i}{c_{t,i}^2+\epsilon_{\rm metric}}.
\]

`epsilon_metric` is a separately recorded numerical floor, not RMSNorm epsilon.
Exclude/report near-zero-coordinate ratios using a calibration-set-defined
threshold, and show how the sensitivity ranking changes with that threshold.
Never interpret a large ratio caused only by a tiny denominator as evidence.

Average samples within each trace, then average traces equally. Report
\(\bar R_{l,i}\) and its within-layer mean-normalized version.

Diagnostics:

- 512-coordinate sensitivity plot with NoPE/RoPE boundaries.
- p90/p10 sensitivity ratio, coefficient of variation, and trace-bootstrap
  uncertainty; flag unstable or near-zero denominators.
- Matched-magnitude bins showing whether equally sized deletions cause different
  output errors.
- Separate NoPE and RoPE summaries and stability across trace groups.

Summarize empirical ablation sensitivity by layer and coordinate region:

| Layer | Coordinate region | Query / row instances | p90 / p10 of \(\bar R_{l,i}\) | CV of \(\bar R_{l,i}\) | 95% trace-bootstrap CI | Matched-magnitude result |
|---|---|---:|---:|---:|---|---|
| Early CSA | All 512 | TBD | TBD | TBD | TBD | TBD |
| Early CSA | NoPE 0–447 | TBD | TBD | TBD | TBD | TBD |
| Early CSA | RoPE 448–511 | TBD | TBD | TBD | TBD | TBD |
| Middle CSA | All 512 | TBD | TBD | TBD | TBD | TBD |
| Middle CSA | NoPE 0–447 | TBD | TBD | TBD | TBD | TBD |
| Middle CSA | RoPE 448–511 | TBD | TBD | TBD | TBD | TBD |
| Late CSA | All 512 | TBD | TBD | TBD | TBD | TBD |
| Late CSA | NoPE 0–447 | TBD | TBD | TBD | TBD | TBD |
| Late CSA | RoPE 448–511 | TBD | TBD | TBD | TBD | TBD |

Non-uniform sensitivity is evidence against uniform coordinate importance.
It is **not alone proof** that a fixed weighted TopMag score improves pruning.

### Avoid 512 full attention recomputations per instance

For a one-row change \(\Delta u_t\), let

\[
\delta_h=\gamma q_h^\top\Delta u_t,\qquad
D_h=1+p_{h,t}(e^{\delta_h}-1).
\]

Then the exact continuous-oracle update is

\[
\bar o'_h=
\frac{\bar o_h+
p_{h,t}(e^{\delta_h}-1)u_t+
p_{h,t}e^{\delta_h}\Delta u_t}{D_h}.
\]

Implement the equivalent log-stable expression and vectorize coordinate tests
in chunks of 32–64. First verify it against complete recomputation, including
the sink and duplicated target entries. If the same physical row appears more
than once, perturb all its occurrences and use their summed baseline mass.

This update includes both the K/logit change and the V change because K=V.
It removes repeated QK work without freezing attention probabilities.

## 5. Compare analytic sensitivity with measured damage

The simplified formula \(p_t[I+(c_t-o)q^\top]\) is insufficient for raw V4 rows.
Include the learned norm, RoPE, attention scale, sink-normalized probabilities,
and all real query heads.

For \(D=512\),

\[
DN(c)=aW-\frac{a^3}{D}(Wc)c^\top.
\]

The Jacobian for head \(h\) and raw target row \(t\) is

\[
J_{h,t}
=p_{h,t}R_{\rm query}^{\top}
\left[I+\gamma(u_t-\bar o_h)q_h^\top\right]R_tDN(c_t).
\]

For \(b_i=R_tDN(c_t)e_i\) and \(v_h=u_t-\bar o_h\),

\[
\|J_{h,t}e_i\|_2^2
=p_{h,t}^2\left[
\|b_i\|_2^2+
2\gamma(q_h^\top b_i)(b_i^\top v_h)+
\gamma^2(q_h^\top b_i)^2\|v_h\|_2^2
\right].
\]

Compute all 512 diagonal entries using the diagonal-plus-rank-one norm
derivative and pairwise RoPE; do not allocate a dense 512×512 Jacobian for every
head/sample.

For example, with \(\tilde q_h=R_t^\top q_h\), compute the vector of dot products
without materializing the \(b_i\):

\[
q_h^\top b_i
=a w_i\tilde q_{h,i}
-\frac{a^3}{D}c_i(\tilde q_h^\top Wc).
\]

Use the same expression with \(R_t^\top v_h\) for \(b_i^\top v_h\).
The column norms follow directly from the diagonal-plus-rank-one derivative.

Fit only on calibration traces:

\[
h_{l,i}=
\mathbb E_{\rm calibration}
\left[\sum_h\|J_{h,t}e_i\|_2^2\right].
\]

- Balance traces as in the ablation aggregation; store the sample counts.
- Normalize each layer's \(h\) to mean one for plotting/export. A uniform
  within-layer scale does not change the pruning mask.
- Require finite, nonnegative diagonals. Clamp only tiny floating-point
  negatives; substantial negatives indicate a formula or numerical bug.
- Validate the analytic derivative with central finite differences on small
  fixtures before comparing it with full-coordinate deletion.
- Report Spearman and Pearson correlation with empirical \(\bar R_{l,i}\) on
  calibration **and held-out** traces, with trace-bootstrap intervals.

Record the analytic-derivative check separately from its predictive value:

| Layer | Finite-difference fixtures | Perturbation step sweep | Maximum relative Jacobian-vector error | Pass / fail |
|---|---:|---|---:|---|
| Early CSA | TBD | TBD | TBD | TBD |
| Middle CSA | TBD | TBD | TBD | TBD |
| Late CSA | TBD | TBD | TBD | TBD |

| Layer | Split | Trace / query / row instances | Spearman (\rho) [95% CI] | Pearson (r) [95% CI] | Finite, nonnegative diagonal fraction |
|---|---|---:|---|---|---:|
| Early CSA | Calibration | TBD | TBD | TBD | TBD |
| Early CSA | Held-out | TBD | TBD | TBD | TBD |
| Middle CSA | Calibration | TBD | TBD | TBD | TBD |
| Middle CSA | Held-out | TBD | TBD | TBD | TBD |
| Late CSA | Calibration | TBD | TBD | TBD | TBD |
| Late CSA | Held-out | TBD | TBD | TBD | TBD |

The Jacobian predicts small continuous perturbations. Deleting a whole feature,
pruning half a row, and FP8 rounding are nonlinear finite changes; correlation
and the held-out packing test must establish whether the approximation is useful.
Do not differentiate through FP8 rounding as if it were an exact smooth map.

## 6. Held-out TopMag vs attention-aware TopMag

Compare only two policies initially:

- **TopMag:** \(s_i=|c_i|\).
- **H-TopMag:** \(s_i=|c_i|\sqrt{h_{l,i}}\).

Both retain exactly **256 coordinates per row**.

The score follows from a diagonal approximation, not an exact pruning theorem:

\[
\mathbb E\|\Delta O\|^2
\approx \Delta c^\top\bar H_l\Delta c
\approx \sum_{i\;\mathrm{pruned}}h_{l,i}c_i^2.
\]

Keeping the largest \(|c_i|\sqrt{h_{l,i}}\) minimizes this last expression.
Off-diagonal interactions and large masked-RMSNorm changes are precisely why
the complete held-out pruning test is required.

### Mask and row reconstruction

- Preserve `topmag_keep_mask`'s lowest-score complement and cutoff-tie behavior.
  Use the same mechanism on the weighted scores, computing each mask once.
- Pass the **original raw row**, not the weighted scores, into packing.
- Prune **all selected CSA rows** for the query, not just the ablation targets.
  Reuse the mask for duplicate occurrences of the same raw row.
- Keep SWA, CSA selection, raw positions, queries, and sink unchanged.
- First compare continuous masked-norm/RoPE rows against the continuous
  unpruned oracle.
- Then compare actual packed round trips against the native-byte oracle:
    - `pack_rows_ref(raw_c, keep_mask, norm_weight, norm_eps)`.
    - `unpack_rows_ref(...)` to recover pre-RoPE BF16 rows.
    - Apply the captured RoPE table/positions using the current reconstruction
      convention, including BF16 rounding.
    - Replay attention and inverse output RoPE.
- Cross-check the Torch packing/reconstruction against the existing production
  packer on a small fixture. Changing the mask must not change raw values or
  the packing ABI.
- An all-ones \(h\) must reproduce the existing TopMag mask and output.
  Include natural zeros and cutoff ties in this regression test.

### Metrics and decision tables

Primary held-out metrics, computed per query and aggregated by trace:

\[
\text{relative L2}=
\frac{\|O-\hat O\|_F}{\max(\|O\|_F,\epsilon_O)},
\qquad
\text{cosine}=
\frac{\langle O,\hat O\rangle}
{\max(\|O\|_F\|\hat O\|_F,\epsilon_{\cos})}.
\]

Also report absolute output error, p95 relative L2, maximum observed error,
and paired H-TopMag-minus-TopMag error by trace. Record the denominator floors
and flag near-zero reference outputs.

Produce one table for each reference space. Paired error change is
H-TopMag minus TopMag, so a negative value favors H-TopMag.

**Continuous-oracle results**

| Layer | Held-out traces / queries | TopMag rel. L2 mean / p95 | H-TopMag rel. L2 mean / p95 | Paired Δ rel. L2 [95% CI] | Absolute L2 mean (TopMag / H) | Max absolute L2 (TopMag / H) | Max rel. L2 (TopMag / H) | Cosine (TopMag / H) |
|---|---|---|---|---|---|---|---|---|
| Early CSA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Middle CSA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Late CSA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Packed-round-trip results**

| Layer | Held-out traces / queries | TopMag rel. L2 mean / p95 | H-TopMag rel. L2 mean / p95 | Paired Δ rel. L2 [95% CI] | Absolute L2 mean (TopMag / H) | Max absolute L2 (TopMag / H) | Max rel. L2 (TopMag / H) | Cosine (TopMag / H) |
|---|---|---|---|---|---|---|---|---|
| Early CSA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Middle CSA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Late CSA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Use 2,000 trace-level paired bootstrap resamples; do not count correlated
heads, coordinates, or adjacent decode positions as independent benchmark
samples. Record \(\epsilon_O\), \(\epsilon_{\cos}\), and the near-zero-output
count alongside these tables.

## 7. Minimal implementation and Modal execution

### Planned code changes

Keep one reusable calibration runner and the existing Modal app.

- **Add `sglang/srt/remnant/capture.py`:** opt-in observer, bounded raw shadow,
  snapshot ownership, and serialization.
    - A new capture setting, e.g. `REMNANT_CAPTURE_DIR`, enables it explicitly.
    - When unset, hooks perform no tensor copies, allocations, or mutation.
    - Hook only the capture sites described above; do not refactor unrelated
      runtime paths.
- **Add `sglang/benchmark/remnant/calibrate.py`:** one runner with preflight,
  capture-client, offline replay, and report subcommands.
    - Own cohort selection, masks, the FP32 oracle, analytic diagonal, and
      compact summaries.
    - Run the same module locally and inside Modal; do not duplicate the math
      in launcher functions.
- **Extend `scripts/modal/app.py`** with thin CPU-preflight, TP4-capture, and
  model-free L4-replay functions.
    - Reuse `server_image`, `model_volume`, and `results_volume`.
    - No second Modal app, Docker layer, or one-off benchmark server.
- **Add `third_party/sglang/test/registered/attention/unittests/dsv4/test_remnant_calibration.py`**
  in the existing attention test tree.
  Test derivatives, one-row updates, sink/duplicate semantics, mask equivalence,
  and packed reconstruction before the capture launch.

The `sglang/...` paths above are inside `third_party/sglang/python/`. They are
implemented additions; ordinary serving leaves `REMNANT_CAPTURE_DIR` unset.

### Execution sequence

1. CPU: validate inputs/math tests, select an eligible account, download/check
   weights if necessary, and build/verify the image.
2. TP4 H100: boot **one native server**, capture a two-request pilot, verify
   alignment and readiness, then collect the fixed cohort.
3. Commit the capture volume and terminate the TP4 allocation immediately.
4. One L4: verify the native-byte oracle against captured outputs, run
   ablations/analytic calibration, and evaluate both held-out policies.
5. CPU: bootstrap, write figures and compact tables, and download the capture
   and results locally for future reuse.

The offline L4 run does not need model weights or Hopper FlashMLA. It uses Torch
references and captured consumer outputs. Validate hardware-specific packing
equivalence during the capture allocation if the reference test requires SM90;
do not quietly allocate another TP4 server.

Implemented launch interfaces (select the eligible account with `MODAL_PROFILE`):

```bash
modal run scripts/modal/app.py::calibration_preflight --run-id <id>
MODAL_PROFILE=<profile> modal run scripts/modal/app.py::capture_attention --run-id <id> --timeout-minutes <cap> --account <profile>
modal run scripts/modal/app.py::replay_attention --run-id <id>
```

The staged manifest controls the data and numerical settings. These entry points
must report the selected account, model/image identity, elapsed allocation time,
snapshot count, and saved result prefix.

### Cost controls

- Plan for one model load, not one load per layer, policy, or ablation.
- Initial **hard ceiling: 60 minutes for TP4 capture**, including startup and
  teardown: at most **4 aggregate H100-hours**. This is a ceiling, not an ETA.
- Set a smaller limit when account headroom requires it. Reserve teardown/billing
  margin and compute the affordable limit from the current rate and compute
  spend:
  \[
  T_{\max}\le
  \frac{28-\text{current spend}-\text{reserve}-\text{offline budget}}
  {4\times\text{H100 hourly rate}+\text{CPU/memory hourly cost}}.
  \]
  \(T_{\max}\) is in hours; include other pending allocations in the spend bound.
- CPU image/model readiness comes first. During the two-request pilot, measure
  startup and capture seconds/request before committing to the remaining cohort.
  On the checked account, requested TP4 resources cost $19.3616/hour including
  CPU/memory; the L4 resources cost $1.5012/hour. Recheck rates before reuse.
- Stop/finalize before the cap. A partial cohort is reported as incomplete, not
  silently promoted. The user authorized automatic retries on 2026-10-06;
  diagnose and fix each failure before retrying, recheck account headroom, and
  preserve every attempt's artifacts and cost. Do not use blind platform retries.
- Give the first offline analysis at most **30 L4 minutes**, processing bounded
  chunks. A failed math/capture gate ends the run rather than launching more
  expensive hardware.
- Prefer one eligible account with cached weights. Modal volumes are account
  scoped; if switching accounts for analysis, transfer only the capture, not
  the checkpoint.
- There is no reliable dollar estimate until account spend, model readiness,
  and current pricing are checked. Do not infer free GPU availability from
  previous VDI access.

## 8. Success criteria, outputs, and next decision

Use this run record to capture actual execution and resource use:

| Phase | Account | GPU allocation | Status | Wall time | H100-hours | L4-hours | Results prefix |
|---|---|---|---|---:|---:|---:|---|
| CPU preflight / image readiness | peroace360 | None | Passed; each source correction subsequently preflighted | 588.14 s initial image build | 0 | 0 | `20261006-attention-pruning-001` |
| Native state capture | peroace360 | 4×H100 TP4 | Failed during dummy-batch autotune; 0 requests | 868.66 s worker | 0.9652 | 0 | `20261006-attention-pruning-001` |
| Corrected native capture | peroace360 | 4×H100 TP4 | Ready; 2 requests completed; no usable states (ID/count integration) | 673.60 s worker | 0.7484 | 0 | `20261006-attention-pruning-002` |
| Third native capture | peroace360 | 4×H100 TP4 | 2 pilot EOS completions; TP ordering corrected; 399 states pass oracle locally | 949.23 s worker | 1.0547 | 0 | `20261006-attention-pruning-003` |
| Fourth CPU image/checkpoint preflight | poohthewinnie | None | Passed; 18 CPU tests, 48 pinned shards, matching source hashes | — | 0 | 0 | `20261006-attention-pruning-004` |
| Fourth native capture | poohthewinnie | 4×H100 TP4 | Complete, all ten requests natural tool-call finishes | 2,233.48 s | 2.4816 | 0 | `20261006-attention-pruning-004` |
| Offline replay / calibration | poohthewinnie | L4 | Complete; initial I/O timeout plus revalidated resume | 1,721.51 + 695.06 s | 0 | 0.6713 | `20261006-attention-pruning-004/analysis` |
| Bootstrap / report | Local | None | Complete; archive integrity and all 2,016 local raw-state oracles pass | — | 0 | 0 | `experiments/attention-pruning/report.md` |

Failure artifacts are downloaded outside Git. `experiments/attention-pruning/run-record.json`
records the exact app/image IDs, revisions, validation and resource use. At the
latest billing check, `peroace360` account-wide compute was **$15.5929** and
`poohthewinnie` was **$13.7186**; storage is excluded. Earlier provisional
views were higher and revised downward. This
is not an isolated invoice for the failed function.

The corrected observer skips only startup dummy batches and requests outside
the frozen cohort. Missing IDs on real sampling batches still fail closed.
Future capture launches set `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` to avoid exhaustive
all-M startup compilation; runtime JIT, FlashInfer autotuning and attention math
remain unchanged. This is capture-only, not a serving benchmark configuration.
Pass `--account <profile>` alongside `MODAL_PROFILE=<profile>` to record account
provenance; local profile environment is not automatically forwarded to workers.
Use a new input/result prefix and CPU preflight after the source correction.

Attempt 002 found that native OpenAI protocol serialization adds tool-schema
defaults, increasing all ten preparation counts by exactly 20 tokens. The
original histories remain frozen. Attempt 003 audits the serialized serving
prompts with the pinned encoder/tokenizer on CPU and retains both prepared and
execution counts/hashes. Explicit body request IDs and enabled header overrides
connect these requests to the capture hooks; response IDs are verified. EOS
receipts are saved before subsequent validation. The final CPU suite passes 21
tests (one CUDA test skipped). The held-out result does not support H-TopMag
promotion: packed mean error +5.49%, p95 +15.55%, with aggregate regression on
all four held-out issues. The layer-38 regression dominates, despite
improvement in 14 of 21 layer means.

Proceed only if all three original gates hold:

1. **Sensitivity is meaningfully non-uniform.**
    - Spread persists across independent traces and matched-magnitude bins,
      rather than arising from tiny-coordinate ratios or one layer/trace.
2. **The analytic diagonal predicts empirical damage.**
    - Positive, stable held-out rank correlation, supported by derivative
      correctness tests and uncertainty estimates.
3. **H-TopMag-256 improves held-out output error at equal storage.**
    - Paired error reduction is repeatable across trace groups; report any
      layer or tail regressions instead of hiding them in the overall mean.

Do not import the earlier kernel experiment's 5% speed threshold here. This is
a quality/sensitivity experiment: effect size and uncertainty decide whether
the extra scoring complexity is justified.

Keep a new append-only result prefix with:

- Frozen manifest and trace split.
- Captured states and capture validation report.
- Layerwise `h[512]` and sensitivity summaries.
- Held-out continuous and packed tables for all 21 layers, correlations,
  confidence intervals, and representative early/middle/late diagnostic figures.
- Commands, hashes, actual counts, wall time, H100-hours, and L4-hours.

Download raw captures locally; track only compact, non-sensitive aggregate
evidence in the repository. No prompt/tool payloads, weights, or large tensor
dumps belong in Git.

If the test succeeds:

- Validate the scoring policy on more independent trajectories before deployment.
- Then study \(K\in\{256,224,192,160,128\}\) **offline** against TopMag50 quality.
- The current packer requires exactly 256 survivors. Smaller \(K\) cannot use
  `pack_rows_ref` unchanged and does not automatically reduce the 328-byte
  persistent record. A new ABI/kernel/storage change is a separate plan.
- Run end-to-end agentic quality tests only after the local signal survives the
  packed round trip. Frozen attention replay cannot establish closed-loop
  task success, altered future queries/selections, or production TPOT.
