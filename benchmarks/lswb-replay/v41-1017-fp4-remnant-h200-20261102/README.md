<!-- Authored by Winston Cai. -->
# V4.1-Flash-1017 FP4 vs Remnant, paired LSWB on 4xH200 (Modal)

Status: **fp4 arm not yet running to completion** (launch approved; see attempts).

## Question

On DeepSeek-V4.1-Flash-1017 (`deepseek-ai/DeepSeek-V4.1-Flash-1017` @
`1bae0e1088993b9b82de08f59de5be5a34e42170`), does Packed
(`--dsv41-main-kv-cache-format remnant`) change replay TTFT, TPOT or cache hit
rate against dense FP4 (`fp4`) at a concurrency near the Native pool limit and
one 25% lower? Experimental and opt-in; not a quality study.

## Design

- **Arms:** `fp4` on profile `poohthewinnie`, `remnant` on profile `winstoncai`.
  Different hosts: ratios are matched-configuration comparisons, not same-host pairs.
- **Hardware/serving:** `H200!:4`, TP4, `MEM_FRAC=0.92`, `MAX_RUN=48`, seed
  20260919, Engram tables on the GPUs, extended decode graphs, no speculation.
- **Workload:** `business-user-replay-v2` (4,916 requests, 92 sessions), fixed
  active-user pool, 900 s per point, caches flushed before each point, one boot
  per arm: C_low then C_high.
- **Sources:** parent branch `ds41-flash-lswb-modal` on `ds41-flash-packed`
  (`f734f0a`); SGLang `4f8fb0912c84`, FlashMLA `858bb1ccac6f`, CUTLASS
  `147f5673d0c1`. These pins are the unqualified follow-up candidate.
- **Entry points:** `scripts/modal/serve.py::prepare_lswb` and `::benchmark_lswb`
  with `REMNANT_INTEGRATION=v0.5.22 REMNANT_MODEL=v41-1017`; shared
  `run-server.sh` (modes `v41-fp4`, `v41-remnant`) and `bench-lswb.sh`
  (`LSWB_EXECUTION=direct`).

## Concurrency derivation

Weights are about 118.4 GiB per GPU at TP4 (checkpoint headers: 502.4 GB without
MTP; Engram tables are part of the 510 GB). MF0.92 leaves about 9.3 GiB of pools.
Native (890 B/token global+indexer from the 1 GiB allocator fixture) holds about
7.87M tokens at `MAX_RUN=48`; Packed (730 B) about 9.60M.
`capacity_plan.py` replays the scheduler's session order and takes the largest C
whose peak live context fits 90% of the pool (C_low = 75% of it); the simulated
C27 peak (4.13M) matches the V4 C27 log (4.04M observed). For 7.87M this yields
about C_high 47 / C_low 35. The Native boot re-derives C from its reported
`max_total_num_tokens`; the Packed arm is launched with those exact values and
fails if they exceed its pool. Token sizes come from recorded `usage` counts
(V4 tokenizer), not the 1017 tokenizer.

## Acceptance

Both arms: boot healthy within 25 min, `server_info` matches the requested arm,
device check is four H200, three sanity replies are non-degenerate, both points
reach success rate >= 0.99 with a valid `done.json`. Matched C across arms.

## Stopping conditions and budget

Stop on: startup log OOM/NCCL/CUDA error, no healthy server in 25 min, wrong
`server_info`, no completed request in 300 s, >5% failures after 50 requests,
or a replay past duration + 900 s. Skip C_high if C_low fails. Hard cap $27 per
profile ($54 total, $3 reserved each; every profile showed $0 for November
before this run). Estimated ~$18 per arm at 8 vCPU / 64 GiB / 4xH200.
Volume storage is not budgeted; delete each 1017 weights volume after collection.

## Known limits

Pool-size prediction is not a gate (recorded in `c-plan.json`/`server-info.json`).
The per-rank weight-memory log line is not parsed. A 100K-token sanity prompt is
not sent; the first replay point covers long-context prefill. Request-timing
export is off. The replay corpus identity is compared across the two profiles
(`prepare-receipt.json` `dataset_sha256`), not to the historical `1e97f17d`, since
reconverting changes record filenames.

## Attempts on profile poohthewinnie (fp4 arm), 2026-11-02

| # | Failure | Cost |
|---|---|---|
| 1 | `H200!` is not a valid GPU type (exact-match suffix is H100-only) | $0 |
| 2 | Modal client dependency (`grpclib`) missing from the serving image | ~$0 |
| 3 | `Mxfp4FlashinferCutlassMoEMethod` needs 128-aligned MoE width; TP4 gives 576 of 2304 | ~$0.2 |
| 4 | Added EP4. "Explicit V4.1 FP4 requires SM90 FlashMLA": `auto` attention backend is never resolved to `flashmla` | ~$3 |
| 5 | Added `--dsv4-attn-backend flashmla`. Rust radix-tree extension absent from the image after CUDA graph capture | ~$5.6 |

Measured at boot (attempts 4-5): weights 120.06 GB per GPU (load 392-429 s),
available pool memory 6.62 GB, 890 B per full token, `max_total_num_tokens`
4,628,480 at `MAX_RUN=48`, graph capture 115 s. The planner on that pool gives
C_high 28 / C_low 21 (the 7.87M estimate above was too high). Account spend after
attempt 5: $9.99 of the $27 cap; a full run is estimated at $13-18 more.
| 6 | Booted, `server_info` and 3 sanity prompts passed (pool 4,765,184 tokens), cache flushed, replay C21 started. CUDA OOM on all ranks at the first long prefill (1 GiB scratch allocation with 0.2-0.7 GiB free of 139.8 GiB; 132 GiB held by PyTorch). 97/97 requests failed, run stopped by the early-stop rule | ~$6.4 |

Fix for attempt 5: `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python` (Python TreeCore).

After attempt 6, `poohthewinnie` is at $16.39 of the $27 cap. Conclusion: 4xH200
at TP4/EP4 with Engram tables on the GPUs leaves no headroom at MF0.92 (weights
111.8 GiB/GPU), and lowering MF to make room shrinks the pool to about 1M tokens.
Needs a memory-layout change (Engram host table, or 8 GPUs) before another launch.
