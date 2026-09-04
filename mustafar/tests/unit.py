"""Numeric unit tests for TopMag pruning and persistent packed C4 storage.

Run both suites with::

    CUDA_VISIBLE_DEVICES=0 python3 -m mustafar.tests.unit

The package CLI preserves the individual entry points::

    python3 -m mustafar selftest
    python3 -m mustafar packedselftest
"""

from unittest.mock import patch

import torch

from .. import config, reference


DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
N = 256


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
    """Validate the fixed 328-byte FP8/bitmap/scale C4 ABI on CPU or GPU."""
    from ..bitmap import bitmap_to_bits
    from ..packed_c4 import (
        NativeC4Workspace,
        PackedC4Buffers,
        _as_buffers,
        pack_c4_rows_ref,
        packed_storage_report,
        project_request_storage,
        unpack_c4_rows_ref,
        unpack_gather_c4_native,
    )

    torch.manual_seed(7)
    rows = 9
    x = torch.randn(rows, config.HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    # Natural zeros and cutoff ties must not be inferred from value != 0.
    x[0, :300] = 0
    x[1].fill_(1)
    mask = reference.topmag_keep_mask(x, 0.5)
    weight = torch.linspace(
        0.8, 1.2, config.HEAD_DIM, dtype=torch.bfloat16, device=DEV
    )
    values, bitmaps, scales = pack_c4_rows_ref(x, mask, weight, 1.0e-6)

    assert values.shape == (rows, 256) and values.dtype == torch.uint8
    assert bitmaps.shape == (rows, 8) and bitmaps.dtype == torch.uint64
    assert scales.shape == (rows, 8) and scales.dtype == torch.uint8
    decoded_mask = bitmap_to_bits(bitmaps)
    assert torch.equal(decoded_mask, mask)
    assert bool((decoded_mask.sum(-1) == 256).all())

    # Ascending-coordinate order: the packed bytes are gathered from sorted
    # mask coordinates, including coordinates whose source value is naturally 0.
    columns = torch.nonzero(mask, as_tuple=False)[:, 1].reshape(rows, 256)
    assert bool((columns[:, 1:] > columns[:, :-1]).all())
    reconstructed = unpack_c4_rows_ref(values, bitmaps, scales)
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()
    assert bool((reconstructed[~mask] == 0).all())

    single = torch.zeros(1, config.HEAD_DIM, dtype=torch.bool, device=DEV)
    single[0, 0] = True
    from ..bitmap import mask_to_bitmap
    assert mask_to_bitmap(single)[0, 0].item() == -(2**63)
    single.zero_(); single[0, 63] = True
    assert mask_to_bitmap(single)[0, 0].item() == 1
    single.zero_(); single[0, 64] = True
    assert mask_to_bitmap(single)[0, 1].item() == -(2**63)

    buffers = PackedC4Buffers(values, bitmaps, scales)
    graph_accessor_buffers = _as_buffers((values, bitmaps, scales))
    assert isinstance(graph_accessor_buffers, PackedC4Buffers)
    assert all(
        actual is expected
        for actual, expected in zip(graph_accessor_buffers, buffers)
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
    native_workspace = NativeC4Workspace.allocate(2, 4, 64, "cpu")
    assert native_workspace.dense is not None
    assert native_workspace.max_queries == 2
    assert native_workspace.selected_k == 4
    too_many = torch.zeros((3, 4), dtype=torch.int32)
    try:
        unpack_gather_c4_native(
            PackedC4Buffers(
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
            "XKV_TOPMAG_KEEP": "0.5",
            "SGLANG_OPT_TOPMAG_PACKED_C4": "1",
            "SGLANG_OPT_TOPMAG_STAGE2A": "1",
        },
        clear=False,
    ):
        config.validate_packed_static_config()
        stage2a_workspace = NativeC4Workspace.allocate(2, 4, 64, "cpu")
        assert stage2a_workspace.dense is None
    with patch.dict(
        "os.environ",
        {"SGLANG_OPT_TOPMAG_PACKED_C4": "0", "SGLANG_OPT_TOPMAG_STAGE2A": "1"},
        clear=False,
    ):
        try:
            config.validate_packed_static_config()
        except RuntimeError as exc:
            assert "requires SGLANG_OPT_TOPMAG_PACKED_C4=1" in str(exc)
        else:
            raise AssertionError("Stage 2A accepted a disabled packed-C4 pool")

    # Position identity required by unpack RoPE.
    raw = torch.arange(1, 33, dtype=torch.int32, device=DEV)
    assert torch.equal(4 * raw, (4 * raw + 4) - 4)

    empty = torch.empty(0, config.HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    empty_mask = torch.empty_like(empty, dtype=torch.bool)
    ev, eb, es = pack_c4_rows_ref(empty, empty_mask, weight, 1.0e-6)
    assert ev.shape == (0, 256) and eb.shape == (0, 8) and es.shape == (0, 8)

    print(
        "[packedselftest] OK: exact-256 mask, MSB-first bitmap, ascending "
        "FP8 codes, 8 UE8M0 scales, natural-zero/tie safety, 328 B/row"
    )


def run() -> None:
    """Run the current Mustafar numeric unit-test suites."""
    run_topmag()
    run_packed_reference()


if __name__ == "__main__":
    run()
