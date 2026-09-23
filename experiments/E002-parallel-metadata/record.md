# E002: parallel bitmap and scale staging

## Hypothesis

`prepare_remnant_row` had one lane per token load all eight bitmap words and
scales, then build the nine rank prefixes serially. Four lanes already
cooperate on the token's packed survivors. Sharing metadata loads and prefix
construction across those lanes could shorten the producer critical path.

## Change

In `third_party/flashmla/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`, each of
the four cooperating lanes now loads two bitmap words and their scales. A
two-step warp scan computes the preceding pair counts; lanes write their unique
bitmap, scale, and prefix entries. The 328-byte layout, survivor loads, rank
arithmetic, RoPE, and scheduling are unchanged. SASS contains the expected
`SHFL.UP` operations at distances 8 and 16.

## Source state

- Parent HEAD: `407fcd5b9004bc2409de13eec7399757abbcd1c1`
- Parent tracked diff SHA-256 at run time: `ab06801a890538608814b8eb30ec3a60b6afa0eda0ea497e11419bc45390611f`
- SGLang HEAD: `0b2006980628f915f8a0b54f28669be9160ed7de`
- FlashMLA HEAD: `529c862003ee535db5b1f4b0799cec88cc9c16b7` (`remnant/flashmla-v0518`)
- FlashMLA source diff SHA-256 at run time (E001 + E002): `eebc4c15e8c080b2893e98620e5df319cd0f8663f51b2afca1c1d07150817fbb`
- Build image: `im-4Obgx8EgUdMdpNy0RBhslc`; only SM90 was built. No SM100 tests or full model load.
- E001 Native and Direct timings/profiles were reused; Native was not re-profiled.

## Validity

The existing SGLang Direct decode suite passed twice: **12/12** each pass.
Compute Sanitizer memcheck passed **8 cases, 0 errors** (four cases deselected
by the existing direct-versus-adapter test selector). Both runs had only the
existing pytest warning for the unrecognized `asyncio_mode` option.

- Pass 1: [Modal run](https://modal.com/apps/winstoncai233/main/ap-Mjq81iUoAioiLb8ih2MYW4)
- Memcheck: [Modal run](https://modal.com/apps/winstoncai233/main/ap-G34YOEFmrcZuzpFjdSKCSb)
- Pass 2: [Modal run](https://modal.com/apps/winstoncai233/main/ap-gnCQy3BsWlY2WezzOehSRF)
- Direct profile: [Modal run](https://modal.com/apps/winstoncai233/main/ap-iIThygHWiCIh3OibMclUp1)

## Timing

Both passes reused `benchmark/remnant/microbench.py`: 10 warmups, 100 CUDA-graph
replays, 30 rounds; H64/H128 × B8/B16 × K512/K317. All eight cases missed the
2% Direct-versus-Native p95 target in both passes. Pass 2's p95 regressions
were **32.45–102.27%**.

Pass 2 Direct-median changes versus E001:

| Shape | K512 | K317 |
|---|---:|---:|
| H64 / B8 | +3.13% | +3.43% |
| H64 / B16 | +6.44% | +9.00% |
| H128 / B8 | −3.09% | −1.29% |
| H128 / B16 | −1.42% | −2.25% |

Geometric-mean Direct median was **2.26% slower** than E001 in pass 1 and
**1.66% slower** in pass 2. Pass 2 versus pass 1 improved only 0.59%
geometrically, so the unfavorable result is directionally consistent. Detailed
samples and exact runtime source revisions are saved in `pass1-microbench.*`
and `pass2-microbench.*`.

## Profile-backed decision

The Direct-only Nsight Systems/Compute capture compared E002 to the saved E001
Direct profile on the same four H100 shapes. NCU kernel time improved by only
**2.35% geomean**, below the 5% gate; H64/B16 regressed 0.65%. Instructions
fell 2.55% geomean, shared-memory wavefronts 3.68%, and shared bank conflicts
6.94%. DRAM bytes read changed by −0.07%. Registers stayed at **168/thread**,
spills stayed at **zero** (`STACK:0`, `LOCAL:0`), and reported occupancy limits
were unchanged. Stall counters were mixed: long-scoreboard +2.10% geomean,
short-scoreboard −3.16%, barrier −1.31%.

Detailed counters, per-shape deltas, and artifact paths are in
[`profile-summary.md`](profile-summary.md). The profile ran with unmodified GPU
clocks; Nsight reported six unavailable NVLink counters, unrelated to the
decode kernel. The full `.nsys-rep`, four `.ncu-rep` files, CSV exports, and
`sass.txt` are retained under `profile/`.

**Decision: reject E002 as a performance candidate.** Validity and memory
safety pass, and profiling confirms lower instruction/shared-memory counts,
but neither the measured Direct latency nor the ≥5% kernel-time gate improved
enough; both timing passes show >2% H64 regressions. Keep production pins
unchanged. E002's metadata scan was removed from the active source when E003
started; its measured source hash, samples, and profiles remain recorded here.

## Plan completion check

- Existing validity suite and memcheck used; no replacement harness added: done.
- Two requested timing passes used the existing microbenchmark; E001 Native and
  Direct baselines were reused: done.
- Ended with a Direct-only profile and checked SASS for the two intended
  `SHFL.UP` scan instructions: done.
- Saved source revisions, timing samples, and raw profile reports together:
  done.
- No SM100 tests, full-model load, commit, push, or production pin update: done.
