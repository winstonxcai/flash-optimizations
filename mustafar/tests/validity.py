"""Validity suite: every kernel leg held to native, stage by stage.

Four **legs**, four **stages**, over a case grid built so that it can fail.

Legs (:data:`LEGS`) -- one column each, matching the two the report already names
(Native, stock 584-B; Packed, ``mustafar packed-328``) plus the two CUDA
extensions under test:

  ``native``       stock SGLang with every ``SGLANG_OPT_TOPMAG*`` flag off:
                   ``compress_norm_rope_store``, ``dequantize_k_cache_paged``,
                   ``flash_mla_sparse_fwd``. No mustafar code, no Triton. This is
                   the bar every candidate is held to.
  ``packed.bf16``  the 328-byte ABI through Triton only -- ``pack_rows`` then
                   ``unpack_gather_bf16`` into dense BF16. This is the
                   multi-token-extend call site (``patches/attention.py``
                   ``compressed_slice``), where it is the *only* implementation:
                   no fused or sparse variant exists for it.
  ``packed.native`` the same store, ``unpack_gather_native`` into the 584-byte
                   layout plus remapped indices -- the decode/small-extend call
                   site. A strict superset of ``packed.bf16``: with ``FUSED=0`` it
                   calls ``unpack_gather_bf16`` internally and then repacks, so a
                   failure here that ``packed.bf16`` does not show is in the
                   repack or in the remapped indices, not in the decompression.
                   Their *tails* are not bit-comparable (this one is quantised
                   twice), so the two are compared only through their shared bar.
  ``fused``        ``mustafar._fused``, replacing ``packed.native``. Needs
                   ``mustafar._fused``.
  ``sparse``       ``mustafar._sparse``, reading 328-byte records directly with
                   no reassembly. Needs ``mustafar._sparse``. Single-token decode
                   only: the gate is ``q.shape[1] == 1 and not _is_sm120``
                   (``patches/attention.py:190``) and there is no multi-token
                   variant to test.

Stages (:data:`STAGES`) -- each asks one question and names the bar it uses:

  ``store``      native store vs the packed store, both fed the *same* kept set.
                 NoPE FP8 codes and UE8M0 scales bit-exact, and the packed bitmap
                 must equal the input mask exactly.
  ``rows``       each leg's row readout vs :func:`harness.native_gather`. The 448
                 NoPE dims bit-exact, the 64 RoPE dims bounded by
                 ``TAIL_ATOL``/``TAIL_RTOL``. The sparse leg has no dense row
                 output, so instead of a row it reports one-hot probe scores
                 covering all 512 coordinates.
  ``attention``  each leg's c4 ``(o, lse)`` vs native rows through
                 ``flash_mla_sparse_fwd``. ``o`` and ``lse`` are asserted
                 *separately* under ``ATTN_ATOL``/``ATTN_RTOL`` -- a
                 correct-lse/wrong-o split is the likeliest real failure and the
                 merged output of :func:`mustafar.sparse.merge_lse` would hide it.
  ``pruning``    how far each leg drifts from an *uncompressed* native answer --
                 the cost of TopMag50 itself, not of any kernel. Bounded by
                 ``QUALITY_ATOL``/``QUALITY_RTOL``, which are deliberately loose
                 sanity ceilings, and pinned tightly on the first GPU run by
                 ``--write-baseline``. A failure here is a finding about TopMag50.

The first three stages feed native the *pruned* latent (``case.masked_latent``),
so every kernel difference is a defect and the asserts stay exact. ``pruning``
feeds it the untouched latent, which is the true all-flags-off baseline. Holding
both to one tolerance, as the single-constant layout did, silently makes "defect"
and "pruning cost" the same number.

The RoPE tail is BF16 on the native path and FP8 on the packed path: that is the
design, not a defect, so every tail comparison carries a tolerance rather than an
equality. Everything else -- codes, scales, the 448 NoPE dims -- is bit-exact.

Direct run prints per-case JSON::

    python3 -m mustafar.tests.validity
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from .. import config, reference
from ..packed import PackedBuffers
from . import harness

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
N = 256

BASELINE = Path(__file__).resolve().parent / "fixtures" / "validity-baseline.json"

STAGES = ("store", "rows", "attention", "pruning")

# The candidate legs, in report order. ``native`` is deliberately absent: it is the
# bar every one of these is compared against, never a candidate. The vocabulary
# itself lives in harness.py, so both suites name the same legs in the same way.
LEGS = harness.LEGS


# --- storage reporting (moved from unit.py) ----------------------------------
def packed_storage_report(buffers: PackedBuffers, occupied_rows: int) -> dict:
    storage_bytes = sum(t.untyped_storage().nbytes() for t in buffers)
    return {
        "logical_bytes_per_row": config.PACKED_RECORD_BYTES,
        "occupied_rows": int(occupied_rows),
        "occupied_bytes": int(occupied_rows) * config.PACKED_RECORD_BYTES,
        "storage_bytes": int(storage_bytes),
        "logical_compression": config.NATIVE_RECORD_BYTES / config.PACKED_RECORD_BYTES,
    }


def project_request_storage(
    seq_len: int = 128 * 1024,
    layers: int = harness.CSA_LAYERS,
    page_size: int = harness.PAGE_SIZE,
) -> dict[str, int | float]:
    """Logical and page-rounded bytes for one request."""
    rows_per_layer = (seq_len + 3) // 4
    pages_per_layer = (rows_per_layer + page_size - 1) // page_size
    native_page_bytes = harness.native_page_stride(page_size)
    packed_page_bytes = config.PACKED_RECORD_BYTES * page_size
    logical_native = rows_per_layer * layers * config.NATIVE_RECORD_BYTES
    logical_packed = rows_per_layer * layers * config.PACKED_RECORD_BYTES
    allocated_native = pages_per_layer * layers * native_page_bytes
    allocated_packed = pages_per_layer * layers * packed_page_bytes
    return {
        "seq_len": seq_len,
        "layers": layers,
        "rows_per_layer": rows_per_layer,
        "pages_per_layer": pages_per_layer,
        "logical_native_bytes": logical_native,
        "logical_packed_bytes": logical_packed,
        "allocated_native_bytes": allocated_native,
        "allocated_packed_bytes": allocated_packed,
        "native_page_padding_bytes": allocated_native - logical_native,
        "packed_page_padding_bytes": allocated_packed - logical_packed,
        "logical_compression": logical_native / logical_packed,
        "allocated_compression": allocated_native / allocated_packed,
    }


# --- CPU tier ----------------------------------------------------------------
def run_reference() -> None:
    """Validate exact dense-zero TopMag pruning semantics (CPU or GPU)."""
    torch.manual_seed(0)
    n = N
    prune_k = config.HEAD_DIM - int(round(config.HEAD_DIM * 0.5))
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(n, config.HEAD_DIM, device=DEV, dtype=dtype)
        orig = x.clone()
        keep_mask = reference.topmag_keep_mask(x, 0.5)
        reference.topmag_zero_from_mask(x, keep_mask)

        zeros = (x == 0).sum(-1)
        assert bool((zeros == prune_k).all()), (
            f"[{dtype}] expected {prune_k} zero coords/row, "
            f"got {zeros.unique().tolist()}"
        )

        retained = x != 0
        assert torch.equal(x[retained], orig[retained]), (
            f"[{dtype}] non-zeroed coords changed (should be bit-identical)"
        )

        magnitude = orig.abs().float()
        prune_idx = magnitude.topk(prune_k, dim=-1, largest=False).indices
        expected = torch.zeros_like(x)
        expected.scatter_(-1, prune_idx, 1.0)
        assert torch.equal(expected.bool(), x == 0), (
            f"[{dtype}] zeroed set != smallest-{prune_k}-per-row"
        )

        unchanged = orig.clone()
        keep_all = reference.topmag_keep_mask(unchanged, 1.0)
        reference.topmag_zero_from_mask(unchanged, keep_all)
        assert torch.equal(unchanged, orig), f"[{dtype}] keep=1.0 not a no-op"

    print(
        f"[validity-cpu] OK: explicit TopMag mask keep=0.5 -> exactly {prune_k}/row, "
        f"zeros=smallest-{prune_k}, nonzero coords bit-identical, keep=1.0 no-op"
    )


def run_packed_reference() -> None:
    """Validate the fixed 328-byte FP8/bitmap/scale ABI on CPU or GPU."""
    from ..bitmap import bitmap_to_mask, mask_to_bitmap
    from ..packed import NativeWorkspace, _as_buffers, unpack_gather_native
    from ..reference import pack_rows_ref, unpack_rows_ref

    torch.manual_seed(7)
    rows = 9
    x = torch.randn(rows, config.HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    # Natural zeros and cutoff ties must not be inferred from value != 0.
    x[0, :300] = 0
    x[1].fill_(1)
    mask = reference.topmag_keep_mask(x, 0.5)
    weight = torch.linspace(0.8, 1.2, config.HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    values, bitmaps, scales = pack_rows_ref(x, mask, weight, 1.0e-6)

    assert values.shape == (rows, 256) and values.dtype == torch.uint8
    assert bitmaps.shape == (rows, 8) and bitmaps.dtype == torch.uint64
    assert scales.shape == (rows, 8) and scales.dtype == torch.uint8
    decoded_mask = bitmap_to_mask(bitmaps)
    assert torch.equal(decoded_mask, mask)
    assert bool((decoded_mask.sum(-1) == 256).all())

    # Ascending-coordinate order: the packed bytes are gathered from sorted
    # mask coordinates, including coordinates whose source value is naturally 0.
    columns = torch.nonzero(mask, as_tuple=False)[:, 1].reshape(rows, 256)
    assert bool((columns[:, 1:] > columns[:, :-1]).all())
    reconstructed = unpack_rows_ref(values, bitmaps, scales)
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()
    assert bool((reconstructed[~mask] == 0).all())

    single = torch.zeros(1, config.HEAD_DIM, dtype=torch.bool, device=DEV)
    single[0, 0] = True
    assert mask_to_bitmap(single)[0, 0].item() == -(2**63)
    single.zero_()
    single[0, 63] = True
    assert mask_to_bitmap(single)[0, 0].item() == 1
    single.zero_()
    single[0, 64] = True
    assert mask_to_bitmap(single)[0, 1].item() == -(2**63)

    buffers = PackedBuffers(values, bitmaps, scales)
    graph_accessor_buffers = _as_buffers((values, bitmaps, scales))
    assert isinstance(graph_accessor_buffers, PackedBuffers)
    assert all(
        actual is expected for actual, expected in zip(graph_accessor_buffers, buffers)
    )
    report = packed_storage_report(buffers, occupied_rows=rows)
    assert report["logical_bytes_per_row"] == 328
    assert report["occupied_bytes"] == rows * 328
    assert abs(report["logical_compression"] - 584 / 328) < 1.0e-12
    assert report["storage_bytes"] == rows * 328
    projection = project_request_storage()
    assert projection["logical_native_bytes"] == 383.25 * 1024**2
    assert projection["logical_packed_bytes"] == 215.25 * 1024**2
    assert projection["packed_page_padding_bytes"] == 0
    assert projection["native_page_padding_bytes"] == 21 * 512 * 64

    # The native workspace is intentionally decode/small-extend sized. A large
    # extend must be rejected before launching Triton so the backend can route
    # it through the existing sparse-prefill workspace instead.
    native_workspace = NativeWorkspace.allocate(2, 4, 64, "cpu")
    assert native_workspace.dense_bf16 is not None
    assert native_workspace.max_queries == 2
    assert native_workspace.selected_k == 4
    too_many = torch.zeros((3, 4), dtype=torch.int32)
    try:
        unpack_gather_native(
            PackedBuffers(
                torch.zeros((1, 256), dtype=torch.uint8),
                torch.zeros((1, 8), dtype=torch.uint64),
                torch.zeros((1, 8), dtype=torch.uint8),
            ),
            too_many,
            too_many,
            torch.full((3,), 4, dtype=torch.int32),
            torch.empty(0, dtype=torch.complex64),
            native_workspace,
        )
    except ValueError as exc:
        assert "route this extend through sparse prefill" in str(exc)
    else:
        raise AssertionError("oversized native gather did not fail early")

    with patch.dict(
        os.environ,
        {
            "SGLANG_OPT_TOPMAG": "1",
            "KEEP": "0.5",
            "SGLANG_OPT_TOPMAG_PACKED": "1",
            "SGLANG_OPT_TOPMAG_FUSED": "1",
        },
    ):
        config.validate_packed_static_config()
        assert NativeWorkspace.allocate(2, 4, 64, "cpu").dense_bf16 is None
    with patch.dict(
        os.environ, {"SGLANG_OPT_TOPMAG_PACKED": "0", "SGLANG_OPT_TOPMAG_FUSED": "1"}
    ):
        try:
            config.validate_packed_static_config()
        except RuntimeError as exc:
            assert "requires SGLANG_OPT_TOPMAG_PACKED=1" in str(exc)
        else:
            raise AssertionError("Fused accepted a disabled packed pool")

    # Position identity required by unpack RoPE: native rotates at seq_len - 4,
    # the packed gather at raw * 4, so seq_len = 4 * (raw + 1) aligns them.
    raw = torch.arange(1, 33, dtype=torch.int32, device=DEV)
    assert torch.equal(4 * raw, (4 * raw + 4) - 4)

    empty = torch.empty(0, config.HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    empty_mask = torch.empty_like(empty, dtype=torch.bool)
    ev, eb, es = pack_rows_ref(empty, empty_mask, weight, 1.0e-6)
    assert ev.shape == (0, 256) and eb.shape == (0, 8) and es.shape == (0, 8)

    print(
        "[validity-cpu] OK: exact-256 mask, MSB-first bitmap, ascending "
        "FP8 codes, 8 UE8M0 scales, natural-zero/tie safety, 328 B/row"
    )


# --- per-stage comparison helpers --------------------------------------------
def _native_codes_scales(
    kvcache: torch.Tensor, locations: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the 448 FP8 codes and 7 real UE8M0 scales of each native record."""
    flat = kvcache.reshape(-1)
    stride = kvcache.shape[-1]
    page = (locations // page_size).to(torch.int64)
    within = (locations % page_size).to(torch.int64)
    device = locations.device
    value_base = page * stride + within * config.NATIVE_RECORD_BYTES
    value_offs = torch.arange(config.NOPE_DIM, device=device)
    scale_base = page * stride + page_size * config.NATIVE_RECORD_BYTES + within * 8
    scale_offs = torch.arange(config.NOPE_DIM // config.TILE_SIZE, device=device)
    return (
        flat[value_base[:, None] + value_offs[None, :]],
        flat[scale_base[:, None] + scale_offs[None, :]],
    )


def _store(case: harness.Case, buffers, native_cache, locations) -> dict:
    """NoPE codes and scales bit-exact, and the bitmap equal to the input mask.

    Only the seven NoPE tiles are comparable: native keeps the 64 RoPE dims in
    BF16, so it has no FP8 tail scale for the eighth tile to match.
    """
    from ..bitmap import bitmap_to_mask

    rows = case.rows
    native_codes, native_scales = _native_codes_scales(
        native_cache, locations, harness.PAGE_SIZE
    )
    dense_codes = torch.zeros(rows, config.HEAD_DIM, dtype=torch.uint8, device=case.device)
    columns = torch.nonzero(case.mask, as_tuple=False)[:, 1].reshape(
        rows, config.PACKED_KEPT_VALUES
    )
    dense_codes.scatter_(1, columns, buffers.values)

    assert torch.equal(native_codes, dense_codes[:, : config.NOPE_DIM]), (
        f"[{_name(case)}/store] packed NoPE FP8 codes != native store"
    )
    assert torch.equal(native_scales, buffers.scales[:, :7]), (
        f"[{_name(case)}/store] packed NoPE scales != native store"
    )
    decoded = bitmap_to_mask(buffers.bitmaps)
    assert torch.equal(decoded, case.mask), (
        f"[{_name(case)}/store] packed bitmap is not the input mask "
        f"({int((decoded != case.mask).sum().item())} coordinates differ)"
    )
    return {
        "codes_bit_exact": True,
        "scales_bit_exact": True,
        "bitmap_bit_exact": True,
    }


def _read_rows(case: harness.Case, buffers, candidates) -> dict[str, torch.Tensor]:
    """Dense BF16 ``(gather_rows, HEAD_DIM)`` rows for every leg that has them.

    ``packed.native`` gets the dense BF16 workspace because its Triton path
    reconstructs through it; ``fused`` writes the native layout directly and does
    not. Allocating it for the fused leg anyway would also mask a real allocation
    regression, so this matches production dispatch exactly.
    """
    rows: dict[str, torch.Tensor] = {}
    for leg in candidates:
        if leg == "packed.bf16":
            rows[leg] = harness.packed_dense(case, buffers)
            continue
        if leg in ("packed.native", "fused"):
            workspace = harness.native_workspace(
                case, with_dense=leg == "packed.native"
            )
            with harness.leg_env(leg):
                harness.packed_native(case, buffers, workspace)
                rows[leg] = harness.workspace_dense(workspace)
    return rows


def _sparse_probe(case: harness.Case, buffers) -> torch.Tensor:
    """``(batch, HEAD_DIM, TOPK)`` one-hot probe scores for the sparse leg.

    The sparse kernel has no dense row output, so there is no row to compare. A
    one-hot query makes ``scores[b, h, j]`` the single KV coordinate head ``h``
    reads, which degrades it to a row readout. The heads of one launch span a
    contiguous 64-dim chunk and eight launches tile all 512 coordinates exactly
    once, with the last chunk (448..511) being precisely the RoPE tail -- so NoPE
    and tail split on a chunk boundary instead of being interleaved the way a
    fixed ``stride`` sample leaves them.
    """
    from .. import sparse

    out = torch.empty(
        case.batch, config.HEAD_DIM, harness.TOPK,
        dtype=torch.float32, device=case.device,
    )
    with harness.leg_env("sparse"):
        for base in range(0, config.HEAD_DIM, harness.HEAD_COUNT):
            dims = torch.arange(harness.HEAD_COUNT, device=case.device) + base
            probe_q, _ = harness.probe_query(case, dims)
            chunk = sparse.scores(
                probe_q, buffers.values, buffers.bitmaps, buffers.scales,
                case.physical, case.raw, case.freqs, 1.0,
                topk_lengths=case.lengths,
            )
            out[:, dims, :] = chunk.float()
    return out


def _split_error(
    case: harness.Case, stage: str, leg: str, got: torch.Tensor, bar: torch.Tensor
) -> dict[str, float]:
    """Assert NoPE bit-exactness and a bounded tail, and report both maxima."""
    nope, tail = slice(0, config.NOPE_DIM), slice(config.NOPE_DIM, config.HEAD_DIM)
    nope_abs = (got[..., nope] - bar[..., nope]).abs()
    assert bool((nope_abs == 0).all()), (
        f"[{_name(case)}/{stage}/{leg}] NoPE is not bit-exact; "
        f"max_abs={float(nope_abs.max().item())} "
        f"violations={int((nope_abs != 0).sum().item())}"
    )
    tail_abs = (got[..., tail] - bar[..., tail]).abs()
    tail_tol = harness.TAIL_ATOL + harness.TAIL_RTOL * bar[..., tail].abs()
    assert bool((tail_abs <= tail_tol).all()), (
        f"[{_name(case)}/{stage}/{leg}] RoPE tail exceeded "
        f"atol={harness.TAIL_ATOL} rtol={harness.TAIL_RTOL}; "
        f"max_abs={float(tail_abs.max().item())} "
        f"violations={int((tail_abs > tail_tol).sum().item())}"
    )
    return {
        "nope_max_abs": float(nope_abs.max().item()),
        "tail_max_abs": float(tail_abs.max().item()),
    }


def _leg_attention(case: harness.Case, leg: str, packed_rows, buffers, q, indices):
    """One leg's c4 ``(o, lse)``, read the way production reads it."""
    if leg == "sparse":
        from .. import sparse

        with harness.leg_env("sparse"):
            return sparse.c4_leg(
                q, buffers.values, buffers.bitmaps, buffers.scales,
                case.physical, case.raw, case.freqs, harness.SM_SCALE,
                topk_lengths=case.lengths,
            )
    kv = packed_rows.view(case.workload.gather_rows, 1, config.HEAD_DIM)
    out, _, lse = harness.c4_bar(q, kv, indices, harness.SM_SCALE)
    return out, lse


def _live_rows(case: harness.Case) -> torch.Tensor:
    """Batch rows that have at least one live slot.

    An empty row has no attention to compare -- every index is ``-1`` for both the
    bar and the candidate -- so its ``o``/``lse`` are excluded rather than compared
    against a similarly undefined number. The ``ragged`` pattern produces rows of
    length 0 on purpose; their row-level readout is still checked, in the ``rows``
    stage, where "all zeros" is well defined.
    """
    return (case.physical >= 0).any(dim=-1) & (case.lengths > 0)


def _attention(
    case: harness.Case,
    candidates,
    leg_rows: dict[str, torch.Tensor],
    buffers,
    q,
    indices,
    bar_name: str,
    bar_out,
    bar_lse,
    atol: float,
    rtol: float,
) -> dict:
    """Each candidate's ``(o, lse)`` against one bar, asserted separately."""
    live = _live_rows(case)
    bar_o = bar_out.float()[live]
    bar_l = bar_lse.float()[live]
    legs: dict[str, dict[str, float]] = {}
    for leg in candidates:
        out, lse = _leg_attention(case, leg, leg_rows.get(leg), buffers, q, indices)
        out = out.float()[live]
        lse = lse.float()[live]
        assert torch.isfinite(lse).all() and torch.isfinite(out).all(), (
            f"[{_name(case)}/attention/{leg}] non-finite output; "
            f"o infinite={int((~torch.isfinite(out)).sum().item())} "
            f"lse infinite={int((~torch.isfinite(lse)).sum().item())}"
        )
        out_abs = (out - bar_o).abs()
        lse_abs = (lse - bar_l).abs()
        for label, error, reference in (("o", out_abs, bar_o), ("lse", lse_abs, bar_l)):
            limit = atol + rtol * reference.abs()
            assert bool((error <= limit).all()), (
                f"[{_name(case)}/{bar_name}/{leg}] {label} diverged from "
                f"the bar; max_abs={float(error.max().item())} "
                f"violations={int((error > limit).sum().item())} "
                f"(atol={atol} rtol={rtol})"
            )
        legs[leg] = {
            "o_max_abs": float(out_abs.max().item()),
            "lse_max_abs": float(lse_abs.max().item()),
        }
    return {"bar": bar_name, "legs": legs}


def _name(case: harness.Case) -> str:
    return f"{case.workload.name}/{case.pattern}"


def _run_case(case: harness.Case, candidates) -> dict:
    """Every stage for one case, each candidate held to its bar."""
    gathered = case.workload.gather_rows

    # Both stores are fed the *same* kept-coordinate set: ``pack_rows`` applies
    # ``case.mask`` itself and native gets the masked latent.
    with harness.leg_env("packed.bf16"):
        buffers = harness.packed_buffers(case)
    native_cache, locations = harness.native_store(case, case.masked_latent)
    native_rows = harness.native_gather(
        case, harness.native_dense(native_cache, locations).float()
    )

    store = {"bar": "native", "legs": {}}
    if "packed.bf16" in candidates:
        store["legs"]["packed.bf16"] = _store(case, buffers, native_cache, locations)

    leg_rows = _read_rows(case, buffers, candidates)
    rows = {"bar": "native", "legs": {}}
    for leg, packed_rows in leg_rows.items():
        rows["legs"][leg] = _split_error(
            case, "rows", leg, packed_rows.float(), native_rows
        )
    if "sparse" in candidates:
        # Transposed into the same ``(..., HEAD_DIM)`` convention the dense row
        # readouts use, so NoPE and tail split on the same axis for every leg.
        probe = _sparse_probe(case, buffers).permute(0, 2, 1)
        bar_probe = native_rows.view(case.batch, harness.TOPK, config.HEAD_DIM)
        rows["legs"]["sparse"] = _split_error(case, "rows", "sparse", probe, bar_probe)

    q = harness.c4_query(case)
    indices = harness.flat_indices(case)
    bar_out, _, bar_lse = harness.c4_bar(
        q,
        native_rows.view(gathered, 1, config.HEAD_DIM),
        indices,
        harness.SM_SCALE,
    )
    attention = _attention(
        case, candidates, leg_rows, buffers, q, indices,
        "native", bar_out, bar_lse, harness.ATTN_ATOL, harness.ATTN_RTOL,
    )

    # The pruning stage's bar is native over the *untouched* latent -- the real
    # all-flags-off answer -- so what it measures is TopMag50's cost, not a
    # kernel's fidelity.
    full_cache, full_locations = harness.native_store(case, case.latent)
    full_rows = harness.native_gather(
        case, harness.native_dense(full_cache, full_locations).float()
    )
    full_out, _, full_lse = harness.c4_bar(
        q,
        full_rows.view(gathered, 1, config.HEAD_DIM),
        indices,
        harness.SM_SCALE,
    )
    pruning = _attention(
        case, candidates, leg_rows, buffers, q, indices,
        "native-untouched", full_out, full_lse,
        harness.QUALITY_ATOL, harness.QUALITY_RTOL,
    )

    return {
        "workload": case.workload.name,
        "pattern": case.pattern,
        "batch": case.batch,
        "gather_rows": gathered,
        "context_rows": case.rows,
        "store": store,
        "rows": rows,
        "attention": attention,
        "pruning": pruning,
    }


# --- GPU entrypoint ----------------------------------------------------------
@patch.dict(
    os.environ,
    SGLANG_OPT_TOPMAG="0",
    SGLANG_OPT_TOPMAG_PACKED="0",
    SGLANG_OPT_TOPMAG_FUSED="0",
    SGLANG_OPT_TOPMAG_SPARSE="0",
)
def run_validity(
    *,
    sanitizer_case: bool = False,
    legs: tuple[str, ...] | None = None,
    write_baseline: bool = False,
) -> dict[str, object]:
    """Run every stage over every case in the grid, for every available leg.

    ``legs`` selects candidate legs (``None`` means all available); ``native`` is
    always evaluated because it is the bar. ``sanitizer_case`` narrows to the
    smallest workload at the default pattern with every available leg, so the
    Modal app can wrap this in ``compute-sanitizer`` without waiting on the
    adversarial grid. ``write_baseline`` records the observed maxima as the
    regression gate instead of checking against an existing one.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("validity requires CUDA")
    device = torch.device("cuda")

    # ``available`` is what was built; ``selected`` is what this run asked for,
    # validated against it. Both are kept: the summary reports the first, the
    # sweep runs the second.
    available = harness.available_legs()
    selected = harness.select_legs(legs)

    workloads = harness.WORKLOADS[:1] if sanitizer_case else harness.WORKLOADS
    cases = (
        harness.case_grid(workloads)
        if workloads
        else []
    )
    if sanitizer_case and cases:
        # The adversarial patterns would make compute-sanitizer's runtime
        # unreasonable, so the sanitizer run takes the smallest workload at the
        # default pattern and every available leg.
        cases = [cases[0]]

    print(
        f"[validity] legs available={list(available)} selected={list(selected)} "
        f"cases={len(cases)}",
        flush=True,
    )
    baseline = None
    if write_baseline:
        print(f"[validity] --write-baseline: not checking {BASELINE}", flush=True)
    elif BASELINE.exists():
        baseline = json.loads(BASELINE.read_text())

    reported: dict[str, object] = {}
    for workload, pattern in cases:
        case = harness.build_case(workload, device, pattern=pattern)
        entry = _run_case(case, selected)
        reported[_name(case)] = entry
        print(json.dumps({_name(case): entry}, sort_keys=True), flush=True)
        if baseline is not None and _name(case) in baseline:
            _assert_no_regression(_name(case), entry, baseline[_name(case)])

    if write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(reported, indent=2, sort_keys=True))
        print(f"[validity] wrote {BASELINE}", flush=True)
    elif baseline is None:
        print(
            "[validity] no fixtures/validity-baseline.json; regression gate "
            "skipped. Re-run with --write-baseline to pin the printed maxima.",
            flush=True,
        )
    summary = {
        "gpu": torch.cuda.get_device_name(),
        "legs": {leg: leg in available for leg in LEGS},
        "cases": reported,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _assert_no_regression(name: str, entry: dict, pinned: dict) -> None:
    """Fail when an observed maximum regresses past what was calibrated.

    Pinned exactly, not with headroom: the same GPU on the same inputs is expected
    to reproduce these maxima bit-for-bit, and a gate that flaps because a number
    was inflated at write time would not be a gate. If one does flap, the cause is
    non-determinism in the leg and is worth knowing, not worth widening.
    """
    for stage in STAGES:
        pinned_legs = pinned.get(stage, {}).get("legs", {})
        for leg, metrics in entry[stage]["legs"].items():
            for metric, value in metrics.items():
                if metric not in pinned_legs.get(leg, {}):
                    continue
                limit = float(pinned_legs[leg][metric])
                if value > limit:
                    raise AssertionError(
                        f"[{name}] regression on {stage}.{leg}.{metric}: "
                        f"{value} > pinned {limit}"
                    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sanitizer-case",
        action="store_true",
        help="smallest workload, default pattern, every available leg, "
        "for compute-sanitizer",
    )
    parser.add_argument(
        "--legs",
        default=None,
        help="comma-separated subset of "
        f"{','.join(LEGS)} (default: all available)",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help=f"record the observed maxima to {BASELINE} instead of checking them",
    )
    args = parser.parse_args()
    if args.sanitizer_case or args.legs or args.write_baseline:
        run_validity(
            sanitizer_case=args.sanitizer_case,
            legs=tuple(args.legs.split(",")) if args.legs else None,
            write_baseline=args.write_baseline,
        )
    else:
        run_reference()
        run_packed_reference()
        run_validity()
