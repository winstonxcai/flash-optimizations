# Triton physically-sparse TopMag — staged plan (Stage 0 = round-trip correctness)

## Context

TopMag 50% pruning on the native c4 latent is proven native-equivalent (29/29, n=7) but prunes **in
place** — the zeroed coords still occupy their 584 B/token slots, so it saves **zero bandwidth**. The
goal is a physically-sparse store.

This revision restructures the earlier plan, which jumped too quickly from "prove pack/unpack
correctness" to "replace the native C4 cache" and led with an FP8 packed layout that **confounded
TopMag error with a new quantization error**. Storage semantics are now staged explicitly (below);
the strongest part — the round-trip invariant — is preserved and is the Stage-0 deliverable. It also
reflects that the "native cache" is not a plain `[N,512]` tensor: it is the **584 B/token hybrid**
(448 fp8 NoPE + 128 bf16 RoPE + 8 B scale), so persisting pre-RMSNorm/pre-RoPE sparse values moves
the cache boundary — a real architectural trade, decided in Stage 1, not assumed here.

**Correctness invariant (Stage 0):**
```
unpack(pack(TopMag(CComp))) == dense-zero TopMag(CComp)
```
with packed values in **`kv_compressed`'s original dtype** (no FP8). At the injection point (before
`compress_norm_rope_store`) `kv_compressed` is pre-RMSNorm/pre-RoPE/pre-quant fp32/bf16, so Stage 0
packed values are exactly that dtype — otherwise TopMag error and FP8 quant error are confounded.

**Empirical finding that drives the packer (verified in the container):** `torch.topk(256,
largest=False)` tie-breaking is **arbitrary-but-deterministic** — not columns 0..255, and not
reproducible by an in-kernel `(mag, idx)` sort. So the exact-global TopMag keep-mask **must** come from
the same host-side `topk` call `topmag_zero` uses.

## Staging — correctness prototype split from storage design

| Stage | What | Proof / gate | Pool touched? |
|---|---|---|---|
| **0** | Triton pack/unpack, **original dtype**, round-trip vs dense TopMag | bit-exact round-trip (bf16+fp32); live shadow-check | no |
| **0.5** | FP8 packed values (UE8M0 survivors) as a numerical experiment | FP8-quant error acceptable on long-decode tasks | no |
| **1** | Persistent sparse C4 pool + fused gather/unpack; pre- vs post-transform decision | real bytes/req + concurrency gain | yes |
| **2** | CUDA sparse consumer (no dense [512,512] intermediate) | ITL improvement | yes |

**This plan implements Stage 0 and designs Stage 0.5** (both compression-only — no memory-pool change,
no decode change). Stages 1–2 are the roadmap; their decisions are flagged now, not locked.

**Deliberately NOT locked in:** *"pre-RoPE sparse cache + re-run RMSNorm/RoPE on every read."* A row is
written once but retrieved up to ~512×/decode token; moving the transform from write-time to read-time
trades HBM savings against repeated norm/RoPE compute. Exact-TopMag semantics make pre-transform the
correct *first* implementation, but it may not be the best *serving* architecture — Stage 1 profiles it.

## Algorithm decision (boxed — explicit before the format is designed)

```
exact global TopMag, host-side mask:
  keep-mask = complement of topk(k=256, largest=False).indices
            = same torch.topk call reference.topmag_zero makes, computed at the store site.
  bit-exact, reproduces the experiment — correct first implementation.
```
- In-kernel `kth_largest` / `tl.sort` over 512 elements is the **hard part** (no cheap Triton
  primitive; tie-break can't reproduce torch's). **Deferred** — only revisit if the host `topk`
  bottlenecks in Stage 1, and even then the tie-break contract must be pinned against the deployment
  torch build.
- Approximate-threshold and tile-local TopK are **production candidates to evaluate later**, not now.
- **Do not assume the Stage-0 packer is the production packer.**

## Stage 0 — deliverable

### Files
- **NEW** `flash-optimizations/mustafar/triton/kernels.py` — all `@triton.jit` kernels
  (`_pack_ccomp_kernel`, `_unpack_ccomp_kernel`), re-exported by `triton/__init__.py`. Kernel code
  only here; no torch host logic in the subfolder.
- **NEW** `flash-optimizations/mustafar/sparse.py` — host wrappers (`pack_ccomp`, `unpack_ccomp`) +
  helpers + torch refs; imports kernels via `from .triton import …`.
- **NEW** `flash-optimizations/mustafar/selftest_sparse.py` — `run()` assertions, CLI `sparseselftest`.
- **EDIT** `flash-optimizations/mustafar/reference.py` — factor `topk_drop_indices(latent, keep)`
  out of `topmag_zero` (lines 11-29); both call it (single mask source, no drift).
- **EDIT** `flash-optimizations/mustafar/__main__.py` — add `sparseselftest`.
- **EDIT** `flash-optimizations/mustafar/config.py` — add `BITMAP_WORDS = HEAD_DIM // 64  # 8` and
  `XKV_SPARSE_SHADOW` flag (default 0).
- **RENAME** `flash-optimizations/mustafar/patched/compressor_v2.py` →
  `patched/compressor_v2_pre_triton.py` (git mv). It is an unreferenced snapshot copy of the patched
  compressor (current dense-zero TopMag, pre-Triton); the name makes clear it's the reference before
  the Triton sparse store. No code references `patched/`, so nothing else changes.

No sglang behavior change when shadow is off (Stage-0 pool untouched).

### Server-file impact (Stage 0) — explicit
| Server file | Stage-0 change |
|---|---|
| `deepseek_v4_memory_pool.py` (memory pool) | **UNCHANGED.** 584 B/token native layout and pool class stay exactly as-is. The sparse persistent pool is Stage 1. |
| `compressor_v2.py` (compressor) | **UNCHANGED in behavior.** The existing TopMag `maybe_prune` hook (already deployed) stays. Stage 0 adds **only** an optional, default-off shadow check (`XKV_SPARSE_SHADOW=1`): pack→unpack→`torch.equal` against the dense pruned tensor, logged, then `compress_norm_rope_store` writes **byte-identical** output. With the flag off (default), Stage 0 adds nothing to this file. |
| `deepseek_v4_backend.py` (decode) | **UNCHANGED.** No decode-path edits in Stage 0. The `top512 → fused gather/unpack → norm/rope` work is Stage 1+. |
| `indexer.py`, kernels (`fused_norm_rope`, FlashMLA), pool config | **UNCHANGED.** |

Stage 0's only edits are inside the `mustafar/` package (`triton/kernels.py`, `sparse.py`,
`selftest_sparse.py`, `reference.py`, `__main__.py`, `config.py`) — a GPU-unit-testable compression
module. The server runs exactly as today unless the shadow flag is explicitly on.

### Kernel A — `_pack_ccomp_kernel` (`triton/kernels.py`), one program per row
```
row = program_id(0); offs = arange(0, 512)
vals = load(x[row, offs])                       # kv_compressed.dtype (fp32/bf16/fp16)
bits = load(mask[row, offs]).to(int1)           # host-precomputed keep-mask
rank = cumsum(bits.to(int32), axis=0) - 1       # flat 512-wide scan → global packed rank 0..255
store(packed[row, rank], vals, mask=bits)       # masked scatter
```
- Grid `(n,)`, `num_warps=4` (128 threads → 4 elems/thread, ≈25-45 regs/thread, sane on sm90).
- **`KEEP_K: tl.constexpr` from day one** — used as row stride / rank bound; **no hard-coded 256**
  anywhere in pointer math or allocation. Variants: keep=0.5→256, 0.375→192, 0.3125→160. (KEEP need
  not be a power of two.)
- Flat cumsum collapses upstream's per-64-tile + host-cumsum machinery: `popcount(mask_row) == KEEP`
  by construction, ranks 0..KEEP-1 hit exactly once → no uninitialized packed slots (so
  `torch.equal(packed, packed_ref)` is a valid full-tensor check).

### Kernel B — `_unpack_ccomp_kernel` (exact inverse of A)
```
row = program_id(0); offs = arange(0, 512)
word = offs // 64;  lane = offs % 64
bm = load(bitmap[row, word])                    # 8 distinct int64 broadcast → coalesced
bits = ((bm >> (63 - lane)) & 1).to(int1)       # MSB = lane 0
rank = cumsum(bits.to(int32), axis=0) - 1
val = load(packed[row, rank], mask=bits, other=0.0)   # masked load = inverse of masked scatter
store(out[row, offs], val)                      # pruned coords land as exactly 0 (== topmag_zero)
```
Grid `(n,)`, `num_warps=4`. **No separate gather kernel** — by design the consumer is
`gather_unpack(sparse_cache, top512_idx)` fused (Stage 1); SEL_VALUES/SEL_BITMAP temps never exist
except transiently inside the kernel.

### Bitmap representation
- Stage 0: **`[n, 8]` int64**, word w covers cols `64w..64w+63`, bit `63-lane` = keep(col `64w+lane`)
  (MSB = lane 0, matches upstream convention). `int64` not `uint64` (torch storage); signed `>>` +
  `& 1` still extracts the right bit (top bit = `-2**63`).
- **NOT locked:** `16 × uint32` is warp32-natural on H100 and costs the same 64 B/row — explicitly
  micro-benchmarked in Stage 1. Host-built bitmap (`_mask_to_bitmap`) keeps the switch to a host +
  `constexpr` change.
- In-kernel bitmap build (`tl.sum(bitmask*shifts)`) deferred to the fused-store milestone (0.5/1).

### Host wrapper API (`sparse.py`)
```python
def pack_ccomp(latent, keep=0.5) -> (packed[n,KEEP] in latent.dtype, bitmap[n,8] int64)
def unpack_ccomp(packed, bitmap, n_rows=None) -> dense[n, 512]
def _keep_count(keep)          # 512 - round(512*keep) == topmag_zero's k; allocation parameterized by keep
def _keep_mask_from_latent(latent, keep)   # int8 [n,512]; kidx=topk(k, largest=False).indices;
                                           # mask.ones().scatter_(-1, kidx, 0)  ← same call as topmag_zero
def _mask_to_bitmap(mask); def _bitmap_to_bits(bitmap)
def pack_ccomp_ref(latent, keep);  def unpack_ccomp_ref(packed, bitmap)   # torch cross-check
```
Zero-row guards return empty tensors without launching. Kernels shaped so a `QUANTIZE: tl.constexpr`
slots in for Stage 0.5 without changing the host API.

### Selftest (`selftest_sparse.py`, `python3 -m mustafar sparseselftest`, on `cuda:0`)
1. **Bit-exact round-trip (bf16 and fp32, n=256, keep=0.5):**
   `torch.equal(unpack_ccomp(*pack_ccomp(x, 0.5)), reference.topmag_zero(x.clone(), 0.5))`. Plus
   shape/dtype asserts, per-row `mask.sum(-1)==256`, kept coords bit-identical, pruned coords exactly 0.
2. **Torch-ref cross-check:** `torch.equal(packed, pack_ccomp_ref(..))`, `torch.equal(bitmap, ..)`,
   `torch.equal(recon, unpack_ccomp_ref(..))`.
3. **Bit-convention pins:** keep col 0 → `bitmap[0,0] == -2**63`; col 63 → `bitmap[0,0] == 1`;
   col 64 → `bitmap[0,1] == -2**63`; `_bitmap_to_bits(_mask_to_bitmap(m))` recovers `m.bool()`.
4. **Tie cases (locks the design):** all-equal magnitudes `full(1.0)` and many-way ties
   `arange(512)%5` — round-trip still equals `topmag_zero` bit-exactly, exactly 256 zeros. Guards
   against a future "simplification" to first-256-columns.
5. **Sparsity sweep via KEEP_K variants:** keep ∈ {1.0, 0.5, 0.375, 0.3125} — round-trip holds,
   packed shapes match `_keep_count`.
6. **Storage-size assert:** `packed.numel()*elem + bitmap.numel()*8 < n*512*elem` (fp32: 1088<2048
   B/row; bf16: 576<1024 B/row) — physical savings is real.
7. Zero-row edge: no kernel launch.

### Live shadow-check (optional, default-off)
`XKV_SPARSE_SHADOW=1`: at the store site, after `maybe_prune`, run pack→unpack and `torch.equal` vs
the dense pruned tensor; log pass/fail; continue to `compress_norm_rope_store` **unchanged**
(byte-identical server behavior). Off (default) → zero server change. Validates the kernels against
**real compressor output distributions**, not just synthetic tensors. This is also the hook point
Stage 0.5 reuses.

## Stage 0.5 — FP8 packed storage feasibility (design, not this plan's deliverable)

Pipeline: BF16/FP16 CComp → TopMag → FP8-quantize survivors (per-64 UE8M0, the native store's
`abs_max / kFP8E4M3Max` formula) → pack → unpack/dequant → native rest of pipeline. **Numerical
experiment only, no pool change.**
- (a) Reconstruction-fidelity measurement on real latent distributions: fp8-recon vs raw Stage-0 recon
  (MSE / max-abs / bit-identical counts) — isolates the **FP8-quant error alone**, on top of TopMag.
- (b) Gate: if (a) is promising, a long-decode Sangfor-Bench eval with the shadow hook in fp8 mode —
  answers whether pre-RoPE FP8 storage is acceptable **before** touching sglang's allocator.

## Roadmap — Stage 1 & 2 (flags, not commits)

- **Stage 1 — persistent sparse C4 pool.** Replace the 584 B/token native slot store with the packed
  sparse pool (`TopMag + FP8 pack` at row creation; rows packed exactly once). Read path:
  `top512_idx → fused gather_unpack(sparse_cache, top512_idx) → native-equivalent downstream rep →
  existing attention`. **Design constraints:** no separate SEL_VALUES/SEL_BITMAP gather kernels; no
  dense `[512,512]` materialization to HBM; bitmap 8×uint64 vs 16×uint32 micro-benchmark. **Profile
  before fusing norm/RoPE:** read-side RMSNorm/RoPE cost on ~512 selected rows vs HBM bytes saved —
  decide design A (sparse **pre-RoPE** CComp + unpack/norm/RoPE at read, preserves exact TopMag
  semantics) vs design B (sparse **post-transform/native** representation, cheaper reads, **changes**
  exact TopMag semantics). Benchmark: bytes/request, max concurrency, TTFT, ITL, req/s, pack cost,
  read-side reconstruction cost.
- **Stage 2 — CUDA sparse consumer.** `top512 → packed CComp → sparse QK → softmax → sparse PV` —
  removes dense reconstruction entirely; where the real ITL improvement lands.

## Verification (Stage 0)

```
docker exec ruler-eval bash -c "cd /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations \
  && CUDA_VISIBLE_DEVICES=0 python3 -m mustafar sparseselftest"
```
All assertions pass (bit-exact round-trip incl. tie cases; KEEP_K sweep; storage-size assert). With
`XKV_SPARSE_SHADOW=1` on the running 30211 server (off by default), shadow-check logs all-equal for a
sampling of real latent rows. Commit + push `flash-optimizations` (Stage-0 pack/unpack + round-trip
validated; server behavior unchanged).
