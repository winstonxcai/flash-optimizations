"""Validity suite: packed must match native, asserted.

Three tiers, each a hard assert, over :data:`mustafar.tests.harness.WORKLOADS`:

  * :func:`run_reference` -- CPU tier. TopMag pruning semantics and the fixed
    328-byte ABI. No CUDA required.
  * T1 store fidelity -- the *same* kept-coordinate set through both stores
    (native ``compress_norm_rope_store`` and Triton ``pack_rows``). NoPE FP8
    codes and UE8M0 scales must be bit-identical.
  * T2 reconstruct fidelity -- the 584-byte store decoded with the production
    ``dequantize_k_cache_paged`` versus the 328-byte store reconstructed by each
    packed backend. NoPE is bit-exact; the RoPE tail is bounded by
    ``harness.TAIL_ATOL``/``TAIL_RTOL``.
  * T3 end-to-end quality -- both legs start from the *untouched* latent, so
    this is the real "does TopMag50 preserve attention" comparison. qk logits and
    softmax attention output are hard-asserted, for every packed backend.
  * T4 direct read -- :func:`run_sparse_t4`, the sparse MLA kernel reading the
    328-byte records with no reassembly, against the Triton gather +
    ``flash_mla_sparse_fwd``. A one-hot probe separates an in-kernel RoPE error
    from an FP8 decode error; ``(o, lse)`` are asserted separately.

The packed store is inherently lossy in the tail: native keeps the 64 RoPE dims
in BF16, the 328-byte ABI quantises them to FP8. That is the design, not a
defect, so T2/T3 carry a tolerance rather than an equality. Everything else --
codes, scales, the 448 NoPE dims -- is held to bit-exactness.

Direct run prints per-workload JSON::

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

# Packed reconstruct backends held to the same native bar. ``triton-dense`` is
# the compressed_slice call site; the two ``native`` legs are the
# decode/small-extend call site, which is where the fused kernel is dispatched.
BACKENDS = ("triton-dense", "triton-native", "fused-native")


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


# --- packed backends ---------------------------------------------------------
def _packed_dense(case: harness.Case, buffers, backend: str) -> torch.Tensor:
    """Reconstruct the gathered dense rows through one packed backend."""
    if backend == "triton-dense":
        return harness.packed_dense(case, buffers)
    if backend == "fused-native" and not _fused_available():
        raise RuntimeError("fused backend requested without a built _fused")
    # The Triton path reconstructs through the dense BF16 workspace; the fused
    # path writes the native layout directly and does not need one. Allocating
    # the dense buffer for the fused leg would also mask a real allocation
    # regression, so match production dispatch exactly.
    workspace = harness.native_workspace(case, with_dense=backend == "triton-native")
    with patch.dict(
        os.environ,
        {"SGLANG_OPT_TOPMAG_FUSED": "1" if backend == "fused-native" else "0"},
    ):
        harness.packed_native(case, buffers, workspace)
        return harness.workspace_dense(workspace)


def _fused_available() -> bool:
    from ..fused import fused_available

    return bool(fused_available())


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


# --- GPU tiers ---------------------------------------------------------------
@patch.dict(
    os.environ,
    SGLANG_OPT_TOPMAG="1",
    KEEP="0.5",
    SGLANG_OPT_TOPMAG_PACKED="1",
    SGLANG_OPT_TOPMAG_FUSED="0",
)
def run_validity(*, sanitizer_case: bool = False) -> dict[str, object]:
    """Run T1/T2/T3 over every workload and every packed backend.

    ``sanitizer_case`` narrows to the smallest workload and the fused backend
    only, so the Modal app can wrap the same entrypoint in ``compute-sanitizer``
    without waiting on the full grid or the noisiest kernels.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("validity requires CUDA")
    device = torch.device("cuda")
    backends = [b for b in BACKENDS if b != "fused-native" or _fused_available()]
    workloads = harness.WORKLOADS[:1] if sanitizer_case else harness.WORKLOADS
    if sanitizer_case:
        backends = [b for b in backends if b == "fused-native"]
        if not backends:
            raise RuntimeError("sanitizer case requires the fused extension")
    elif len(backends) != len(BACKENDS):
        print("[validity] fused extension absent; fused-native leg skipped")

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    reported: dict[str, object] = {}

    for workload in workloads:
        case = harness.build_case(workload, device)
        rows = case.rows
        gathered = workload.gather_rows

        # T1 + T2 reference: both stores fed the *same* kept-coordinate set.
        # pack_rows applies case.mask itself; native gets the masked latent.
        buffers = harness.packed_buffers(case)
        native_cache, locations = harness.native_store(case, case.masked_latent)
        native_codes, native_scales = _native_codes_scales(native_cache, locations, harness.PAGE_SIZE)

        dense_codes = torch.zeros(rows, config.HEAD_DIM, dtype=torch.uint8, device=device)
        columns = torch.nonzero(case.mask, as_tuple=False)[:, 1].reshape(
            rows, config.PACKED_KEPT_VALUES
        )
        dense_codes.scatter_(1, columns, buffers.values)
        assert torch.equal(native_codes, dense_codes[:, : config.NOPE_DIM]), (
            f"[{workload.name}] T1: packed NoPE FP8 codes != native store"
        )
        assert torch.equal(native_scales, buffers.scales[:, :7]), (
            f"[{workload.name}] T1: packed NoPE scales != native store"
        )

        native_dense_rows = harness.native_dense(native_cache, locations).float()
        t2: dict[str, dict[str, float]] = {}
        for backend in backends:
            # The packed legs are already exactly `gathered` rows; the native
            # reference is a full `context_rows` cache, so trim it to match.
            packed_rows = _packed_dense(case, buffers, backend).float()
            reference_rows = native_dense_rows[:gathered]
            assert torch.equal(
                packed_rows[:, : config.NOPE_DIM],
                reference_rows[:, : config.NOPE_DIM],
            ), f"[{workload.name}/{backend}] T2: reconstructed NoPE is not bit-exact"
            tail_abs = (packed_rows[:, config.NOPE_DIM :] - reference_rows[:, config.NOPE_DIM :]).abs()
            tail_tol = harness.TAIL_ATOL + harness.TAIL_RTOL * reference_rows[:, config.NOPE_DIM :].abs()
            assert bool((tail_abs <= tail_tol).all()), (
                f"[{workload.name}/{backend}] T2: RoPE tail exceeded "
                f"atol={harness.TAIL_ATOL} rtol={harness.TAIL_RTOL}; "
                f"max_abs={tail_abs.max().item()} "
                f"violations={int((tail_abs > tail_tol).sum().item())}"
            )
            t2[backend] = {"tail_max_abs": float(tail_abs.max().item())}

        if "fused-native" in backends:
            # The fused adapter must not assume the default stream.
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                off_stream = _packed_dense(case, buffers, "fused-native")
            side.synchronize()
            assert torch.equal(
                off_stream[:, : config.NOPE_DIM],
                native_dense_rows[:gathered, : config.NOPE_DIM],
            ), f"[{workload.name}] fused backend differs on a non-default stream"

        # Invalid top-k slots must be zeroed and duplicate gathers must agree.
        slot = torch.tensor([[0, 1, -1, 1]], dtype=torch.int32, device=device)
        slot_out = torch.empty(4, config.HEAD_DIM, dtype=torch.bfloat16, device=device)
        from ..packed import unpack_gather_bf16

        unpack_gather_bf16(
            buffers,
            slot,
            slot,
            torch.full((1,), 4, dtype=torch.int32, device=device),
            case.freqs,
            slot_out,
        )
        assert bool((slot_out[2] == 0).all()), (
            f"[{workload.name}] T2: invalid top-k slot was not zeroed"
        )
        assert torch.equal(slot_out[1], slot_out[3]), (
            f"[{workload.name}] T2: duplicate gather rows disagree"
        )

        # T3: the real quality comparison. Both legs start from the untouched
        # latent, so this measures TopMag50's pruning, not just ABI fidelity.
        full_cache, full_locations = harness.native_store(case, case.latent)
        full_native = harness.native_dense(full_cache, full_locations).float()[:gathered]
        query = torch.randn(64, config.HEAD_DIM, dtype=torch.bfloat16, device=device).float()
        scale = config.HEAD_DIM**0.5
        value = torch.randn(64, 64, dtype=torch.bfloat16, device=device).float()
        native_logits = query @ full_native.T / scale
        native_attention = torch.softmax(native_logits, dim=-1) @ value

        t3: dict[str, dict[str, float]] = {}
        for backend in backends:
            packed = _packed_dense(case, buffers, backend).float()
            packed_logits = query @ packed.T / scale
            packed_attention = torch.softmax(packed_logits, dim=-1) @ value
            assert torch.isfinite(packed_logits).all(), f"[{workload.name}/{backend}] T3: non-finite logits"
            assert torch.allclose(
                packed_logits, native_logits, atol=harness.TAIL_ATOL, rtol=harness.TAIL_RTOL
            ), (
                f"[{workload.name}/{backend}] T3: qk logits diverged; "
                f"max_abs={(packed_logits - native_logits).abs().max().item()}"
            )
            assert torch.allclose(
                packed_attention,
                native_attention,
                atol=harness.TAIL_ATOL,
                rtol=harness.TAIL_RTOL,
            ), (
                f"[{workload.name}/{backend}] T3: attention output diverged; "
                f"max_abs={(packed_attention - native_attention).abs().max().item()}"
            )
            t3[backend] = {
                "logits_max_abs": float((packed_logits - native_logits).abs().max().item()),
                "attention_max_abs": float(
                    (packed_attention - native_attention).abs().max().item()
                ),
            }

        entry = {
            "gather_rows": gathered,
            "context_rows": rows,
            "batches": case.batch,
            "t1_bit_exact": True,
            "t2": t2,
            "t3": t3,
        }
        reported[workload.name] = entry
        print(json.dumps({workload.name: entry}, sort_keys=True), flush=True)

        if baseline is not None and workload.name in baseline:
            _assert_no_regression(workload.name, entry, baseline[workload.name])

    if baseline is None:
        print(
            "[validity] no fixtures/validity-baseline.json; regression gate skipped. "
            "Pin the printed maxima to enable it.",
            flush=True,
        )
    print(json.dumps({"workloads": reported}, sort_keys=True), flush=True)
    return reported


# --- T4: direct 328-byte sparse MLA ------------------------------------------
@patch.dict(
    os.environ,
    SGLANG_OPT_TOPMAG="1",
    KEEP="0.5",
    SGLANG_OPT_TOPMAG_PACKED="1",
    SGLANG_OPT_TOPMAG_FUSED="0",
    SGLANG_OPT_TOPMAG_SPARSE="1",
)
def run_sparse_t4() -> dict[str, object]:
    """T4: the direct 328-byte kernel held to the gather + FlashMLA bar.

    The bar is what production does today -- ``unpack_gather_bf16`` into dense
    BF16 KV, then ``flash_mla_sparse_fwd``. Two probes per workload:

      * T4a, one-hot: head ``h`` reads exactly one KV coordinate, so a kernel-side
        RoPE error and an FP8 decode error fall in disjoint dim ranges (the 56
        NoPE dims vs the 8 tail dims ``stride=8`` lands on) and are reported as
        separate maxima instead of being averaged into one number.
      * T4b, end-to-end: ``(o, lse)`` asserted separately, because a
        correct-lse/wrong-o split is the likeliest real failure and the merged
        output of :func:`mustafar.sparse.merge_lse` would hide it.

    ``attn_sink`` is deliberately absent from both sides: it belongs to the native
    SWA leg, so T4 exercises the c4 leg alone. Fused is pinned off because the
    static gate rejects sparse and fused together.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("sparse MLA T4 requires CUDA")
    from .. import sparse

    if not sparse.sparse_available():
        raise RuntimeError("sparse MLA T4 requires a built mustafar._sparse")

    device = torch.device("cuda")
    reported: dict[str, object] = {}

    for workload in harness.WORKLOADS:
        case = harness.build_case(workload, device)
        gathered = workload.gather_rows
        buffers = harness.packed_buffers(case)

        # The bar's KV rows: exactly the dense BF16 rows production feeds
        # flash_mla today, so any difference is the kernel's, not the gather's.
        bar_rows = harness.packed_dense(case, buffers)
        bar_kv = bar_rows.view(gathered, 1, config.HEAD_DIM)

        # T4a: one-hot probe, NoPE and tail reported separately.
        probe_q, dims = harness.probe_query(case)
        probe = sparse.scores(
            probe_q, buffers.values, buffers.bitmaps, buffers.scales,
            case.physical, case.raw, case.freqs, 1.0, topk_lengths=case.lengths,
        )
        # scores[b, h, j] must equal the single coordinate head h reads.
        expect = (
            bar_rows.view(case.batch, harness.TOPK, config.HEAD_DIM)[:, :, dims]
            .permute(0, 2, 1)
            .float()
        )
        probe_abs = (probe - expect).abs()
        is_nope = dims.cpu() < config.NOPE_DIM
        probe_tol = harness.TAIL_ATOL + harness.TAIL_RTOL * expect.abs()
        for label, columns in (("NoPE", is_nope), ("tail", ~is_nope)):
            err = probe_abs[:, columns, :]
            tol = probe_tol[:, columns, :]
            assert bool((err <= tol).all()), (
                f"[{workload.name}] T4a {label}: direct read != gathered rows; "
                f"max_abs={err.max().item()} "
                f"violations={int((err > tol).sum().item())}"
            )

        # T4b: end-to-end, output and lse asserted separately.
        q = harness.c4_query(case)
        ref_out, _, ref_lse = harness.c4_bar(
            q, bar_kv, harness.flat_indices(case), harness.SM_SCALE
        )
        our_out, our_lse = sparse.c4_leg(
            q, buffers.values, buffers.bitmaps, buffers.scales,
            case.physical, case.raw, case.freqs, harness.SM_SCALE,
            topk_lengths=case.lengths,
        )
        assert torch.isfinite(our_lse).all(), (
            f"[{workload.name}] T4b: non-finite lse from the direct read"
        )
        out_abs = (our_out.float() - ref_out.float()).abs()
        out_tol = harness.TAIL_ATOL + harness.TAIL_RTOL * ref_out.float().abs()
        assert bool((out_abs <= out_tol).all()), (
            f"[{workload.name}] T4b output: max_abs={out_abs.max().item()} "
            f"violations={int((out_abs > out_tol).sum().item())} "
            f"(atol={harness.TAIL_ATOL} rtol={harness.TAIL_RTOL})"
        )
        lse_abs = (our_lse.float() - ref_lse.float()).abs()
        lse_tol = harness.TAIL_ATOL + harness.TAIL_RTOL * ref_lse.float().abs()
        assert bool((lse_abs <= lse_tol).all()), (
            f"[{workload.name}] T4b lse: max_abs={lse_abs.max().item()} "
            f"violations={int((lse_abs > lse_tol).sum().item())}"
        )

        entry = {
            "gather_rows": gathered,
            "context_rows": case.rows,
            "batches": case.batch,
            "probe_nope_max_abs": float(probe_abs[:, is_nope, :].max().item()),
            "probe_tail_max_abs": float(probe_abs[:, ~is_nope, :].max().item()),
            "output_max_abs": float(out_abs.max().item()),
            "lse_max_abs": float(lse_abs.max().item()),
        }
        reported[workload.name] = entry
        print(json.dumps({workload.name: entry}, sort_keys=True), flush=True)

    print(json.dumps({"sparse_t4": reported}, sort_keys=True), flush=True)
    return reported


def _assert_no_regression(name: str, entry: dict, pinned: dict) -> None:
    """Fail when an observed maximum regresses past what was calibrated."""
    for tier in ("t2", "t3"):
        for backend, observed in entry[tier].items():
            for metric, value in observed.items():
                key = f"{tier}.{backend}.{metric}"
                if key not in pinned:
                    continue
                limit = float(pinned[key])
                if value > limit:
                    raise AssertionError(
                        f"[{name}] regression on {key}: {value} > pinned {limit}"
                    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sanitizer-case",
        action="store_true",
        help="smallest workload, fused backend only, for compute-sanitizer",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="T4 only: the direct 328-byte sparse MLA leg",
    )
    parser.add_argument(
        "--with-sparse",
        action="store_true",
        help="run T1/T2/T3 and then T4",
    )
    args = parser.parse_args()
    if args.sanitizer_case:
        run_validity(sanitizer_case=True)
    elif args.sparse:
        run_sparse_t4()
    else:
        run_reference()
        run_packed_reference()
        run_validity()
        if args.with_sparse:
            run_sparse_t4()
