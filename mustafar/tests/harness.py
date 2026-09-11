"""Shared workloads, geometry, native references, legs, and timing.

No server is booted for these suites, so this module is the single source of
dimensional truth: every CSA dimension is imported from :mod:`mustafar.config`
and the serving-side page geometry is pinned here rather than re-literalised.

Tolerances and the leg vocabulary live here and nowhere else. Do not inline a
numeric tolerance in a suite -- add a named constant below so the value has one
home and one justification -- and do not name a leg outside :data:`LEGS`, for the
same reason.

  * NoPE fp8 codes and UE8M0 scales are **bit-exact** (``torch.equal``): the
    native store and the packed store quantise them identically.
  * The RoPE tail is BF16 on the native path and FP8 on the packed path, so it is
    inherently lossy; ``TAIL_ATOL``/``TAIL_RTOL`` bound it.
  * End-to-end qk logits and attention output inherit that tail loss and use the
    same two constants.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from statistics import median
from unittest.mock import patch

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

# The c4 attention stage's budget. The RoPE tail is 64 of the 512 KV dims and the
# softmax mixes all of them, so the output inherits more than the tail's own
# bound. Grounded in the maxima of the last direct-read run
# (results/sparse-mla-20260911-062418/t4.log: o <= 3.91e-3, lse <= 3.28e-4) with
# ~2x margin; re-pin from the first run of the stage/leg suite.
ATTN_ATOL = 8.0e-3
ATTN_RTOL = 1.0e-3

# The pruning stage's budget: how far TopMag50 may move the c4 answer away from
# the uncompressed one. NOT CALIBRATED -- these are sanity ceilings, not measured
# bounds, and they are deliberately loose because the honest bound can only come
# from a run. The tight gate is fixtures/validity-baseline.json, which
# --write-baseline emits from the first GPU run; re-pin these from that same run
# so the constant and the fixture agree. A failure here is a finding about
# TopMag50, not a kernel defect.
QUALITY_ATOL = 1.0
QUALITY_RTOL = 1.0


# --- legs (single source of truth) -------------------------------------------
# One vocabulary for both suites. ``native`` is not a candidate: it is the bar the
# candidates are held to (validity) or measured against (speed), so it is named
# separately from the list of things being compared.
NATIVE = "native"

# Candidate legs, in report order. ``packed`` appears as its two real entry points
# rather than as one column: ``packed.bf16`` renders dense BF16 for the
# multi-token-extend call site, ``packed.native`` renders the 584-byte native
# layout for the decode/small-extend one. Two Triton operators with different
# costs, not two names for one thing.
LEGS = ("packed.bf16", "packed.native", "fused", "sparse")

# Report order: the bar first, then the candidates.
COLUMNS = (NATIVE, *LEGS)

# Every flag a leg needs, and the all-off base they are written against. A leg
# states exactly what it turns on, so nothing is inherited by accident: the
# launcher exports the packed flags for every module it drives, and a leg whose
# config came from the ambient environment would be a different leg under a
# direct run than under the launcher.
OFF_ENV = {
    "SGLANG_OPT_TOPMAG": "0",
    "SGLANG_OPT_TOPMAG_PACKED": "0",
    "SGLANG_OPT_TOPMAG_FUSED": "0",
    "SGLANG_OPT_TOPMAG_SPARSE": "0",
}
_PACKED_ENV = {
    **OFF_ENV,
    "SGLANG_OPT_TOPMAG": "1",
    "KEEP": "0.5",
    "SGLANG_OPT_TOPMAG_PACKED": "1",
}
# Keyed by column, ``native`` included: it is a leg with a configuration like any
# other, and naming it here is what lets a caller dispatch on the leg rather than
# special-casing the bar.
LEG_ENV = {
    NATIVE: OFF_ENV,
    "packed.bf16": _PACKED_ENV,
    "packed.native": _PACKED_ENV,
    "fused": {**_PACKED_ENV, "SGLANG_OPT_TOPMAG_FUSED": "1"},
    "sparse": {**_PACKED_ENV, "SGLANG_OPT_TOPMAG_SPARSE": "1"},
}


@contextlib.contextmanager
def leg_env(leg: str):
    """Pin the flags one leg needs, and prove the pinned set is a legal one.

    ``validate_packed_static_config`` is what rejects the fused and sparse gates
    together -- they rewrite the same c4 decode call site -- so running it here
    rather than trusting the caller makes the mutual exclusion a runtime fact
    instead of a comment.
    """
    with patch.dict(os.environ, LEG_ENV[leg]):
        config.validate_packed_static_config()
        yield


def _fused_available() -> bool:
    from ..fused import fused_available

    return bool(fused_available())


def _sparse_available() -> bool:
    from ..sparse import sparse_available

    return bool(sparse_available())


def leg_available(leg: str) -> bool:
    """Whether a candidate leg's CUDA extension is built.

    ``packed`` is the leg that can never regress and the leg that localises a
    failure, so it has to stay reachable with both extensions absent -- if it
    quietly became conditional on one being built, neither property would hold.
    """
    if leg == "fused":
        return _fused_available()
    if leg == "sparse":
        return _sparse_available()
    return True


def available_legs() -> tuple[str, ...]:
    """The candidate legs whose extension is present, in report order."""
    return tuple(leg for leg in LEGS if leg_available(leg))


def select_legs(legs: tuple[str, ...] | None) -> tuple[str, ...]:
    """Resolve a requested leg subset against what is actually built.

    ``None`` means every available candidate. The two ways a request can fail are
    told apart so a caller does not have to parse one message to find out which:
    an unknown name is a usage error (``ValueError``), an unbuilt one is a build
    error (``RuntimeError``). The result is always in :data:`LEGS` order, and
    ``native`` is never in it -- it is the bar, not a candidate.
    """
    available = available_legs()
    if legs is None:
        return available
    # The bar is always run and never optional, so naming it is a usage error --
    # reported as such rather than as an unknown name, since it is very much known.
    if NATIVE in legs:
        raise ValueError(
            f"{NATIVE} is always run and cannot be selected; candidates: {list(LEGS)}"
        )
    unknown = [leg for leg in legs if leg not in LEGS]
    if unknown:
        raise ValueError(f"unknown legs: {unknown}; known: {list(LEGS)}")
    unbuilt = [leg for leg in legs if leg not in available]
    if unbuilt:
        raise RuntimeError(f"selected legs are not built: {unbuilt}")
    return tuple(leg for leg in LEGS if leg in legs)


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


# --- case patterns -----------------------------------------------------------
# ``identity`` is the shape every earlier version of the suites ran: physical =
# raw = arange(gather_rows) and lengths = TOPK everywhere. A kernel that ignored
# ``physical`` and ``raw`` entirely and indexed rows by slot position passed every
# test under that grid, and a kernel that ignored ``topk_lengths`` did too, since
# the values happened to coincide. The patterns below break that coincidence.
IDENTITY = "identity"

INDEX_PATTERNS = frozenset({"permuted", "duplicated", "ragged", "interior_slots"})
MASK_PATTERNS = frozenset({"no_tail", "half_pair", "extreme_coord"})

ADVERSARIAL_PATTERNS: tuple[str, ...] = (
    "permuted",
    "duplicated",
    "ragged",
    "interior_slots",
    "no_tail",
    "half_pair",
    "extreme_coord",
)


def adversarial_workload(workloads):
    """The workload the adversarial patterns run on: the smallest with batch > 1.

    ``ragged`` needs several batch rows before varying a length across them means
    anything, and the mask patterns want enough rows that a re-selected coordinate
    is not a one-row fluke. ``long`` would multiply the grid for no extra
    bug-catching, so the first multi-row workload is the one.
    """
    for workload in workloads:
        if workload.batch > 1:
            return workload
    return workloads[0]


def case_grid(workloads) -> list[tuple[Workload, str]]:
    """``(workload, pattern)`` pairs: every workload at ``identity``, plus the
    adversarial patterns once each on :func:`adversarial_workload`.

    The empty guard is load-bearing: ``test_backend_selection`` patches
    ``WORKLOADS`` to ``()`` to drive the entry/exit path with no allocation, so an
    unguarded ``workloads[0]`` would ``IndexError`` there.
    """
    grid = [(workload, IDENTITY) for workload in workloads]
    if workloads:
        adversarial = adversarial_workload(workloads)
        grid += [(adversarial, pattern) for pattern in ADVERSARIAL_PATTERNS]
    return grid


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
    pattern: str  # which adversarial pattern shaped the indices and the mask
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


def _pattern_mask(pattern: str, latent: torch.Tensor) -> torch.Tensor:
    """The keep-mask for one pattern. Always exactly ``PACKED_KEPT_VALUES`` per row.

    The mask patterns construct the kept *set* directly instead of steering the
    magnitude ranking and hoping it lands where intended -- with 512 real-valued
    coordinates, "the 256 largest excluding the tail" does not reliably exclude
    exactly the tail. Constructing it also has the side benefit of feeding the
    kernels masks that TopMag50 would never emit on random data, which is the
    point: the ABI takes the mask as input and must accept any 256-wide set.

    ``|kept| == 256`` is fixed-width -- ``reference.pack_rows_ref`` raises if any
    row keeps a different count -- so these patterns re-select *which* 256
    coordinates survive and never change the count.
    """
    if pattern not in MASK_PATTERNS:
        if pattern != IDENTITY and pattern not in INDEX_PATTERNS:
            raise ValueError(f"unknown case pattern: {pattern}")
        return reference.topmag_keep_mask(latent, 0.5)

    magnitude = latent.float().abs()
    keep = torch.zeros(latent.shape, dtype=torch.bool, device=latent.device)
    kept_nope = config.PACKED_KEPT_VALUES  # 256; trimmed below for the tail patterns

    if pattern == "no_tail":
        # Nothing survives in the 64 tail coordinates, so bitmap word 7 is
        # all-zero and the tile has no kept value to scale. That is the case a
        # decoder dividing by word 7's UE8M0 scale, or assuming a non-empty tile,
        # gets wrong -- and the packed values buffer is then 256 codes deep for a
        # record whose last tile contributes none of them.
        picked = magnitude[:, : config.NOPE_DIM].topk(kept_nope, dim=-1).indices
    elif pattern == "half_pair":
        # Exactly the even lane of every RoPE real/imag pair, backfilled with NoPE
        # coordinates to hold the count at 256. No pair is ever both-kept or
        # both-pruned, which is the assumption an in-kernel rotate that reads its
        # partner unconditionally would rely on.
        kept_nope -= config.ROPE_DIM // 2
        even = torch.arange(
            config.NOPE_DIM, config.HEAD_DIM, 2, device=latent.device
        )
        keep[:, even] = True
        picked = magnitude[:, : config.NOPE_DIM].topk(kept_nope, dim=-1).indices
    elif pattern == "extreme_coord":
        # The final coordinate, held in, so rank 255 is exercised rather than
        # assumed -- and the code that lands in the last rank slot is a real one.
        kept_nope = config.PACKED_KEPT_VALUES - 1
        keep[:, config.HEAD_DIM - 1] = True
        picked = magnitude[:, : config.HEAD_DIM - 1].topk(kept_nope, dim=-1).indices
    else:
        raise ValueError(f"unknown mask pattern: {pattern}")

    keep.scatter_(dim=-1, index=picked, value=True)
    return keep


def _pattern_indices(pattern: str, physical: torch.Tensor, lengths: torch.Tensor):
    """Perturb ``physical`` / ``lengths`` for one index pattern.

    ``raw`` is left to the caller to clone from ``physical``, because the two must
    stay equal for the legs to be phase-aligned -- see the note in
    :func:`build_case`.
    """
    if pattern not in INDEX_PATTERNS:
        return physical, lengths

    batch, topk = physical.shape
    device = physical.device

    if pattern == "permuted":
        # A kernel that indexes the gather output by slot position rather than by
        # ``physical`` now reads the wrong record everywhere.
        shuffled = torch.randperm(batch * topk, device=device)
        return shuffled.reshape(batch, topk).to(torch.int32).contiguous(), lengths

    if pattern == "duplicated":
        # Records gathered twice, each at its own raw. A gather that assumes record
        # ids are distinct, or memoises on ``physical``, disagrees between the two
        # slots that must hold identical rows.
        half = (batch * topk) // 2
        repeated = torch.cat([torch.arange(half, device=device)] * 2)
        return repeated.reshape(batch, topk).to(torch.int32).contiguous(), lengths

    if pattern == "ragged":
        # A ragged length per batch row, including an empty one. Beyond
        # ``lengths[b]`` every slot is -1, which both the Triton gather (masking on
        # ``topk_lengths``) and ``flash_mla_sparse_fwd`` (masking on -1) must treat
        # as absent; an off-by-one in either shows up as a disagreement.
        step = torch.tensor([topk, topk // 2, 0, topk // 3], device=device)
        lengths = step[torch.arange(batch, device=device) % 4].to(torch.int32).contiguous()
        slot = torch.arange(topk, device=device).view(1, topk)
        physical = torch.where(slot < lengths.view(batch, 1), physical, -1)
        return physical.to(torch.int32).contiguous(), lengths

    if pattern == "interior_slots":
        # -1 scattered inside the valid range. The suffix-only probe the suite used
        # to run lets a validity predicate written as "k < topk_length" pass, while
        # a per-slot "physical >= 0" check -- which is what the kernels do -- is
        # what this actually needs.
        physical = physical.clone()
        physical[:, 3::7] = -1
        return physical, lengths

    raise ValueError(f"unknown index pattern: {pattern}")


def build_case(
    workload: Workload, device, seed: int = 20260910, pattern: str = IDENTITY
) -> Case:
    """Build the untouched latent and every index tensor for one workload.

    ``pattern`` selects how the index tensors and the TopMag mask are shaped;
    ``identity`` reproduces the original grid exactly.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    rows = workload.context_rows

    latent = torch.randn(rows, config.HEAD_DIM, dtype=torch.bfloat16, device=device)
    mask = _pattern_mask(pattern, latent)
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
    # record i, which is what makes the rows-stage tail comparison meaningful.
    plan_rows[:, 0] = COMPRESS_RATIO * (
        1 + torch.arange(rows, dtype=torch.int32, device=device)
    )

    gather = torch.arange(workload.gather_rows, dtype=torch.int32, device=device)
    physical = gather.reshape(workload.batch, TOPK).contiguous()
    lengths = torch.full((workload.batch,), TOPK, dtype=torch.int32, device=device)
    physical, lengths = _pattern_indices(pattern, physical, lengths)

    return Case(
        workload=workload,
        device=device,
        pattern=pattern,
        latent=latent,
        masked_latent=masked_latent,
        mask=mask,
        weight=weight,
        freqs=freqs,
        locations=locations,
        plan=DecodePlan(plan_rows),
        physical=physical,
        # raw == physical is a hard requirement, not a convenience: the native store
        # bakes the rotation position in at store time from the plan while the
        # packed gather derives it from raw, so only equality phase-aligns the two
        # legs. No pattern may decorrelate them.
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


def native_gather(case: Case, native_rows: torch.Tensor) -> torch.Tensor:
    """``(gather_rows, HEAD_DIM)`` native rows in gather order, invalid slots zero.

    The reference for slot ``j`` is native record ``case.physical[j]``, **not**
    ``native_rows[j]``. Those coincide only when ``physical`` is the identity --
    which is the only shape the suites ran before the pattern grid -- so the
    general form is spelled out once here instead of sliced at each call site.

    A slot is invalid when ``physical`` is negative or when it sits past its row's
    ``topk_length``; both read as zeros, matching what the Triton gather and
    ``flash_mla_sparse_fwd`` do with such a slot.
    """
    flat = case.physical.reshape(-1).to(torch.int64)
    lengths = case.lengths.reshape(-1).repeat_interleave(TOPK)
    slot = torch.arange(TOPK, device=flat.device).repeat(case.batch)
    valid = (flat >= 0) & (slot < lengths)
    rows = native_rows[flat.clamp_min(0)]
    return torch.where(valid.view(-1, 1), rows, torch.zeros_like(rows))


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

    This is what lets the ``rows`` stage hold the fused/Triton reconstruct to the
    *native* bar: both the 584-byte reference store and the packed reconstruct are
    read back through the same ``dequantize_k_cache_paged`` the patch replaces, so
    a layout error cannot hide behind a bespoke reader.
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


def probe_query(
    case: Case, dims: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot queries and the head dim each one reads.

    Head ``h`` is one-hot at dim ``dims[h]``, so ``scores[q, h, j]`` reduces to a
    single KV coordinate, ``kv[q * TOPK + j, dims[h]]``. With no ``dims`` the
    heads span ``0, 8, ... 504`` -- 56 NoPE dims (0..440) and 8 tail dims
    (448..504) -- which is what makes an in-kernel RoPE error separable from an
    FP8 decode error, at the cost of sampling only 64 of the 512 coordinates.

    Pass an explicit ``dims`` to read a different chunk, e.g.
    ``arange(HEAD_COUNT) + base``. Sweeping ``base`` over
    ``range(0, HEAD_DIM, HEAD_COUNT)`` covers all 512 dims exactly once, and the
    final chunk (``base=448``) is precisely the 64 RoPE tail dims -- so NoPE and
    tail fall on opposite sides of a chunk boundary rather than being interleaved.
    """
    if dims is None:
        dims = torch.arange(HEAD_COUNT, device=case.device) * 8
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

    Slots the case marks absent -- ``physical < 0``, or past ``lengths[b]`` -- come
    out as ``-1``, which is how FlashMLA is told to skip a slot. Zeroing the row in
    the dense buffer is not enough on its own: a valid index into a zeroed row
    still enters the softmax as a logit of 0, whereas ``-1`` drops it. The two
    kernels therefore have to agree on what "absent" means, and this is where the
    agreement is expressed. For an identity case nothing is absent and the output
    is the plain row-major enumeration.
    """
    base = torch.arange(case.batch, dtype=torch.int32, device=case.device) * TOPK
    offsets = torch.arange(TOPK, dtype=torch.int32, device=case.device)
    flat = (base.view(-1, 1) + offsets.view(1, -1)).view(case.batch, 1, TOPK)
    absent = (case.physical < 0) | (offsets.view(1, -1) >= case.lengths.view(-1, 1))
    return torch.where(absent.unsqueeze(1), torch.full_like(flat, -1), flat).contiguous()


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
