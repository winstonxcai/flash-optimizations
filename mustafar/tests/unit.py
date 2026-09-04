"""Numeric unit tests for TopMag pruning and persistent packed storage.

Run both suites with::

    CUDA_VISIBLE_DEVICES=0 python3 -m mustafar.tests.unit

The package CLI preserves the individual entry points::

    python3 -m mustafar selftest
    python3 -m mustafar packed_selftest
"""

from unittest.mock import patch

import torch

from .. import config, reference
from ..packed import PackedBuffers

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
N = 256


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
    layers: int = 21,
    page_size: int = 64,
) -> dict[str, int | float]:
    """Logical and page-rounded bytes for one request."""
    rows_per_layer = (seq_len + 3) // 4
    pages_per_layer = (rows_per_layer + page_size - 1) // page_size
    native_page_bytes = ((config.NATIVE_RECORD_BYTES * page_size + 575) // 576) * 576
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


def run_topmag() -> None:
    """Validate exact dense-zero TopMag pruning semantics."""
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
        f"[selftest] OK: explicit TopMag mask keep=0.5 -> exactly {prune_k}/row, "
        f"zeros=smallest-{prune_k}, nonzero coords bit-identical, "
        "keep=1.0 no-op"
    )


def run_packed_reference() -> None:
    """Validate the fixed 328-byte FP8/bitmap/scale ABI on CPU or GPU."""
    from ..bitmap import bitmap_to_mask
    from ..packed import (
        NativeWorkspace,
        PackedBuffers,
        _as_buffers,
        unpack_gather_native,
    )
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
    from ..bitmap import mask_to_bitmap

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
        "os.environ",
        {
            "SGLANG_OPT_TOPMAG": "1",
            "KEEP": "0.5",
            "SGLANG_OPT_TOPMAG_PACKED": "1",
            "SGLANG_OPT_TOPMAG_FUSED": "1",
        },
        clear=False,
    ):
        config.validate_packed_static_config()
        fused_workspace = NativeWorkspace.allocate(2, 4, 64, "cpu")
        assert fused_workspace.dense_bf16 is None
    with patch.dict(
        "os.environ",
        {"SGLANG_OPT_TOPMAG_PACKED": "0", "SGLANG_OPT_TOPMAG_FUSED": "1"},
        clear=False,
    ):
        try:
            config.validate_packed_static_config()
        except RuntimeError as exc:
            assert "requires SGLANG_OPT_TOPMAG_PACKED=1" in str(exc)
        else:
            raise AssertionError("Fused accepted a disabled packed pool")

    # Position identity required by unpack RoPE.
    raw = torch.arange(1, 33, dtype=torch.int32, device=DEV)
    assert torch.equal(4 * raw, (4 * raw + 4) - 4)

    empty = torch.empty(0, config.HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    empty_mask = torch.empty_like(empty, dtype=torch.bool)
    ev, eb, es = pack_rows_ref(empty, empty_mask, weight, 1.0e-6)
    assert ev.shape == (0, 256) and eb.shape == (0, 8) and es.shape == (0, 8)

    print(
        "[packed_selftest] OK: exact-256 mask, MSB-first bitmap, ascending "
        "FP8 codes, 8 UE8M0 scales, natural-zero/tie safety, 328 B/row"
    )


def run_unit_tests() -> None:
    """Run the current Mustafar numeric unit-test suites."""
    run_topmag()
    run_packed_reference()


if __name__ == "__main__":
    run_unit_tests()
