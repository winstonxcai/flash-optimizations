"""Shared workloads, geometry, native references, and timing for the kernel suites.

No server is booted for these suites, so this module is the single source of
dimensional truth: every CSA dimension is imported from :mod:`mustafar.config`
and the serving-side page geometry is pinned here rather than re-literalised.

Tolerances live here and nowhere else. Do not inline a numeric tolerance in a
suite -- add a named constant below so the value has one home and one
justification.

  * NoPE fp8 codes and UE8M0 scales are **bit-exact** (``torch.equal``): the
    native store and the packed store quantise them identically.
  * The RoPE tail is BF16 on the native path and FP8 on the packed path, so it is
    inherently lossy; ``TAIL_ATOL``/``TAIL_RTOL`` bound it.
  * End-to-end qk logits and attention output inherit that tail loss and use the
    same two constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import torch

from .. import config, reference

# --- serving-side geometry (fixed by DeepSeek-V4's CSA layout) ---------------
PAGE_SIZE = 64  # native tokens per page; packed rows index PAGE_SIZE // 4
TOPK = 512  # selected_k: the patch hard-requires c4_topk == 512
COMPRESS_RATIO = 4
CSA_LAYERS = 21  # compressed sparse-attention layers (see report.md)
HEAD_COUNT = 64  # query heads on a CSA layer; not in config, fixed by the model
SM_SCALE = config.HEAD_DIM**-0.5  # qk softmax scale, shared by both c4 legs

# --- tolerances (single source of truth) -------------------------------------
TAIL_ATOL = 0.02
TAIL_RTOL = 0.02


def native_page_stride(page_size: int = PAGE_SIZE) -> int:
    """Bytes per native page, padded to a multiple of the 576-byte record tile."""
    return ((config.NATIVE_RECORD_BYTES * page_size + 575) // 576) * 576


@dataclass(frozen=True)
class Workload:
    """One point of the context x batch grid.

    ``batch`` query rows each select ``TOPK`` records, and the cache holds
    ``context_rows`` compressed records. ``batch * TOPK`` must not exceed
    ``context_rows``: the suites gather the first ``batch * TOPK`` records, so a
    partial gather exercises "more context resident than selected this step".
    """

    name: str
    batch: int
    context_rows: int

    @property
    def gather_rows(self) -> int:
        return self.batch * TOPK


WORKLOADS: tuple[Workload, ...] = (
    Workload("short", 1, 512),  # batch 1, gather covers the whole cache
    Workload("mid", 16, 8192),  # batch 16, gather covers the whole cache
    Workload("long", 64, 65536),  # batch 64, gather covers half the cache
)


class DecodePlan:
    """Minimal stand-in for SGLang's compressor plan ABI.

    The kernels only ever read ``plan[1]`` (the int32 row block, exposed as a
    uint8 view exactly as the injected pool accessor does) plus ``is_decode`` and
    ``compress_ratio``.
    """

    def __init__(self, rows: torch.Tensor, *, is_decode: bool = True):
        self.rows = rows
        self.is_decode = is_decode
        self.compress_ratio = COMPRESS_RATIO

    def __getitem__(self, index: int):
        if index == 1:
            return self.rows.view(torch.uint8)
        raise IndexError(index)


@dataclass
class Case:
    """Every input both suites need for one workload, built once and reused."""

    workload: Workload
    device: torch.device
    latent: torch.Tensor  # (context_rows, HEAD_DIM) bf16, untouched
    masked_latent: torch.Tensor  # same latent with non-kept coords zeroed
    mask: torch.Tensor  # (context_rows, HEAD_DIM) bool, TopMag50 keep set
    weight: torch.Tensor  # (HEAD_DIM,) bf16 norm weight
    freqs: torch.Tensor  # (max_position + 1, ROPE_DIM // 2) complex64
    locations: torch.Tensor  # (context_rows,) int64 native store targets
    plan: DecodePlan
    physical: torch.Tensor  # (batch, TOPK) int32 cache record ids
    raw: torch.Tensor  # (batch, TOPK) int32 raw indices used for RoPE
    lengths: torch.Tensor  # (batch,) int32

    @property
    def rows(self) -> int:
        return self.workload.context_rows

    @property
    def batch(self) -> int:
        return self.workload.batch


def build_case(workload: Workload, device, seed: int = 20260910) -> Case:
    """Build the untouched latent and every index tensor for one workload."""
    torch.manual_seed(seed)
    device = torch.device(device)
    rows = workload.context_rows

    latent = torch.randn(rows, config.HEAD_DIM, dtype=torch.bfloat16, device=device)
    mask = reference.topmag_keep_mask(latent, 0.5)
    masked_latent = latent.masked_fill(~mask, 0)
    weight = torch.linspace(
        0.75, 1.25, config.HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    # Unit-magnitude frequencies keep the RoPE rotation a pure phase, so a tail
    # mismatch can only come from quantisation or a misaligned position.
    max_position = rows * COMPRESS_RATIO
    angles = torch.randn(
        max_position + 1, config.ROPE_DIM // 2, dtype=torch.float32, device=device
    )
    freqs = torch.polar(torch.ones_like(angles), angles).contiguous()

    locations = torch.arange(rows, dtype=torch.int64, device=device)
    plan_rows = torch.zeros(rows, 4, dtype=torch.int32, device=device)
    # Word 0 of the plan is seq_len, and the native store rotates the tail at
    # position = seq_len - compress_ratio (fused_norm_rope_v2.cuh). The packed
    # gather derives the same position as raw * 4 (triton/kernels.py). Setting
    # seq_len = 4 * (raw + 1) therefore makes the two legs phase-aligned on
    # record i, which is what makes the T2 tail comparison meaningful.
    plan_rows[:, 0] = COMPRESS_RATIO * (
        1 + torch.arange(rows, dtype=torch.int32, device=device)
    )

    gather = torch.arange(workload.gather_rows, dtype=torch.int32, device=device)
    physical = gather.reshape(workload.batch, TOPK).contiguous()
    lengths = torch.full((workload.batch,), TOPK, dtype=torch.int32, device=device)

    return Case(
        workload=workload,
        device=device,
        latent=latent,
        masked_latent=masked_latent,
        mask=mask,
        weight=weight,
        freqs=freqs,
        locations=locations,
        plan=DecodePlan(plan_rows),
        physical=physical,
        raw=physical.clone(),
        lengths=lengths,
    )


# --- native (584-byte) leg ---------------------------------------------------
def native_store(case: Case, source: torch.Tensor, page_size: int = PAGE_SIZE):
    """Run the real native fused norm+RoPE store over ``source`` rows.

    ``source`` is the dense latent actually stored: the untouched latent for a
    quality leg, or ``case.masked_latent`` when the goal is to compare the two
    stores on an identical kept set. Returns ``(kvcache, locations)``.
    """
    from sglang.kernels.ops.attention.dsv4.compress import compress_norm_rope_store

    rows = source.shape[0]
    pages = (rows + page_size - 1) // page_size
    kvcache = torch.zeros(
        pages, native_page_stride(page_size), dtype=torch.uint8, device=case.device
    )
    compress_norm_rope_store(
        source.contiguous(),
        case.plan,
        norm_weight=case.weight,
        norm_eps=1.0e-6,
        freq_cis=case.freqs,
        out_loc=case.locations,
        kvcache=kvcache,
        page_size=page_size,
    )
    return kvcache, case.locations


def native_dense(kvcache: torch.Tensor, locations: torch.Tensor, page_size: int = PAGE_SIZE):
    """Decode native records with the production operator the patch replaces."""
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    return dequantize_k_cache_paged(kvcache, locations, page_size).reshape(
        locations.shape[0], config.HEAD_DIM
    )


# --- packed (328-byte) leg ---------------------------------------------------
def packed_buffers(case: Case, rows: int | None = None):
    """TopMag-pack the case latent into persistent 328-byte buffers."""
    from ..packed import PackedBuffers, pack_rows

    rows = case.rows if rows is None else rows
    buffers = PackedBuffers(
        torch.zeros(
            rows, config.PACKED_KEPT_VALUES, dtype=torch.uint8, device=case.device
        ),
        torch.zeros(
            rows, config.BITMAP_WORDS, dtype=torch.uint64, device=case.device
        ),
        torch.zeros(
            rows, config.PACKED_SCALE_BYTES, dtype=torch.uint8, device=case.device
        ),
    )
    pack_rows(
        case.latent[:rows],
        case.mask[:rows],
        case.weight,
        1.0e-6,
        case.plan,
        case.locations[:rows],
        buffers,
    )
    return buffers


def packed_dense(case: Case, buffers, output: torch.Tensor | None = None) -> torch.Tensor:
    """Reconstruct dense BF16 rows via the Triton gather (the compressed_slice path)."""
    from ..packed import unpack_gather_bf16

    if output is None:
        output = torch.empty(
            case.workload.gather_rows,
            config.HEAD_DIM,
            dtype=torch.bfloat16,
            device=case.device,
        )
    unpack_gather_bf16(
        buffers, case.physical, case.raw, case.lengths, case.freqs, output
    )
    return output


def packed_native(case: Case, buffers, workspace) -> torch.Tensor:
    """Materialise the native page layout the FlashMLA consumer reads.

    Dispatches to the fused CUDA adapter when ``SGLANG_OPT_TOPMAG_FUSED=1``
    (see :func:`mustafar.packed.unpack_gather_native`), so the same call drives
    both the Triton and fused legs.
    """
    from ..packed import unpack_gather_native

    unpack_gather_native(
        buffers, case.physical, case.raw, case.lengths, case.freqs, workspace
    )
    return workspace.native_bytes


def native_workspace(case: Case, *, with_dense: bool = False):
    """Allocate the packed-index native gather workspace for one case.

    ``page_size`` is ``PAGE_SIZE // COMPRESS_RATIO`` because packed indices index
    the compressed record grid, not the native 64-token page.
    """
    from ..packed import NativeWorkspace

    return NativeWorkspace.allocate(
        case.batch,
        TOPK,
        PAGE_SIZE // COMPRESS_RATIO,
        case.device,
        with_dense=with_dense,
    )


def workspace_dense(workspace) -> torch.Tensor:
    """Decode a packed-produced native workspace with the production decoder.

    This is what lets T2 hold the fused/Triton reconstruct to the *native*
    bar: both the 584-byte reference store and the packed reconstruct are read
    back through the same ``dequantize_k_cache_paged`` the patch replaces, so a
    layout error cannot hide behind a bespoke reader.
    """
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    locations = workspace.temporary_indices.reshape(-1).to(torch.int64)
    return dequantize_k_cache_paged(
        workspace.native_bytes, locations, page_size=workspace.page_size
    ).reshape(locations.shape[0], config.HEAD_DIM)


# --- c4 (sparse MLA) leg inputs ----------------------------------------------
def c4_query(case: Case, seed: int = 20260911) -> torch.Tensor:
    """``(batch, HEAD_COUNT, HEAD_DIM)`` bf16 queries for one case.

    Reseeded per call so the query stream is reproducible independently of how
    many cases ran before it in the same process.
    """
    torch.manual_seed(seed)
    return torch.randn(
        case.batch,
        HEAD_COUNT,
        config.HEAD_DIM,
        dtype=torch.bfloat16,
        device=case.device,
    )


def probe_query(case: Case, stride: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot queries and the head dim each one reads.

    Head ``h`` is one-hot at dim ``h * stride``, so ``scores[q, h, j]`` reduces to
    a single KV coordinate, ``kv[q * TOPK + j, h * stride]``. With ``stride=8``
    the 64 heads span ``0, 8, ... 504`` -- 56 NoPE dims (0..440) and 8 tail dims
    (448..504) -- which is what makes an in-kernel RoPE error separable from an
    FP8 decode error.
    """
    dims = torch.arange(HEAD_COUNT, device=case.device) * stride
    q = torch.zeros(
        case.batch, HEAD_COUNT, config.HEAD_DIM,
        dtype=torch.bfloat16, device=case.device,
    )
    q.scatter_(2, dims.view(1, HEAD_COUNT, 1).expand(case.batch, HEAD_COUNT, 1), 1.0)
    return q, dims


def flat_indices(case: Case) -> torch.Tensor:
    """``(batch, 1, TOPK)`` int32 row ids into a flattened ``batch * TOPK`` buffer.

    ``flash_mla_sparse_fwd`` indexes a per-batch KV tensor, while the packed
    kernels use global record ids; this is the reshuffle between them.
    """
    base = torch.arange(case.batch, dtype=torch.int32, device=case.device) * TOPK
    offsets = torch.arange(TOPK, dtype=torch.int32, device=case.device)
    return (base.view(-1, 1) + offsets.view(1, -1)).view(case.batch, 1, TOPK).contiguous()


def c4_bar(q, kv, indices, sm_scale):
    """The bar the direct read is held to: FlashMLA sparse on gathered rows.

    This is the production c4 leg today -- ``unpack_gather_bf16`` into dense BF16
    KV, then ``flash_mla_sparse_fwd``. Imported lazily so this module stays
    importable without SGLang (the CPU tier imports it).
    """
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    return flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=config.HEAD_DIM)


# --- timing ------------------------------------------------------------------
def timed(fn, *, warmup: int = 20, repeats: int = 100) -> dict[str, float]:
    """Median and p95 device time in microseconds for one callable."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return {
        "p50_us": median(samples),
        "p95_us": samples[min(len(samples) - 1, int(0.95 * len(samples)))],
    }


def captured(fn):
    """Warm ``fn`` once, capture it in a CUDA graph, and return a replay callable."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay
