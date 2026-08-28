"""Numeric unit tests for TopMag pruning and Stage-0 sparse pack/unpack.

Run both suites with::

    CUDA_VISIBLE_DEVICES=0 python3 -m mustafar.tests.unit

The package CLI preserves the individual entry points::

    python3 -m mustafar selftest
    python3 -m mustafar sparseselftest
"""

import torch

from .. import config, reference


DEV = "cuda:0"
N = 256


def run_topmag() -> None:
    """Validate exact dense-zero TopMag pruning semantics."""
    torch.manual_seed(0)
    n = N
    prune_k = config.HEAD_DIM - int(round(config.HEAD_DIM * 0.5))
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(n, config.HEAD_DIM, device=DEV, dtype=dtype)
        orig = x.clone()
        reference.topmag_zero(x, 0.5)

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
        reference.topmag_zero(unchanged, 1.0)
        assert torch.equal(unchanged, orig), f"[{dtype}] keep=1.0 not a no-op"

    print(
        f"[selftest] OK: topmag_zero keep=0.5 -> exactly {prune_k}/row, "
        f"zeros=smallest-{prune_k}, nonzero coords bit-identical, "
        "keep=1.0 no-op"
    )


def _roundtrip_ok(x: torch.Tensor, keep: float) -> tuple:
    """Assert round-trip equals dense-zero on the exact same mask."""
    from .. import sparse

    keep_k = sparse._keep_count(keep)
    mask = reference.topmag_keep_mask(x, keep)
    assert mask.dtype == torch.bool and mask.shape == x.shape
    counts = mask.sum(-1)
    assert bool((counts == keep_k).all()), (
        f"[{x.dtype}] keep={keep}: mask popcount "
        f"{counts.unique().tolist()} != {keep_k}"
    )
    packed, bitmap = sparse.pack_ccomp(x, mask, keep_k)
    reconstructed = sparse.unpack_ccomp(packed, bitmap)
    expected = x.masked_fill(~mask, 0.0)
    assert torch.equal(reconstructed, expected), (
        f"[{x.dtype}] keep={keep}: round-trip != dense-zero TopMag"
    )
    return mask, packed, bitmap, reconstructed, expected


def _check_storage(
    x: torch.Tensor,
    packed: torch.Tensor,
    bitmap: torch.Tensor,
    keep: float,
) -> None:
    element_bytes = x.element_size()
    packed_bytes = packed.numel() * element_bytes
    bitmap_bytes = bitmap.numel() * 8
    dense_bytes = x.numel() * element_bytes
    assert packed_bytes + bitmap_bytes < dense_bytes, (
        f"[{x.dtype}] keep={keep}: {packed_bytes + bitmap_bytes} B "
        f"!< dense {dense_bytes} B"
    )
    packed_per_row = (packed_bytes + bitmap_bytes) / x.shape[0]
    dense_per_row = dense_bytes / x.shape[0]
    print(
        f"    keep={keep}: packed+bitmap {packed_per_row:.0f} B/row "
        f"< dense {dense_per_row:.0f} B/row"
    )


def run_sparse() -> None:
    """Validate Stage-0 Triton sparse packing and reconstruction."""
    from .. import sparse

    torch.manual_seed(0)
    n = N

    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(n, config.HEAD_DIM, device=DEV, dtype=dtype)

        mask, packed, bitmap, reconstructed, _ = _roundtrip_ok(x, 0.5)
        assert packed.dtype == dtype and packed.shape == (n, 256), packed.shape
        assert bitmap.dtype == torch.int64 and bitmap.shape == (n, 8), bitmap.shape
        assert torch.equal(reconstructed[mask], x[mask]), (
            "kept coords not bit-identical"
        )
        assert bool((reconstructed[~mask] == 0).all()), (
            "pruned coords not exactly 0"
        )

        keep_k = sparse._keep_count(0.5)
        packed_ref, bitmap_ref = sparse.pack_ccomp_ref(x, mask, keep_k)
        assert torch.equal(packed, packed_ref), "packed != torch ref"
        assert torch.equal(bitmap, bitmap_ref), "bitmap != torch ref"
        assert torch.equal(
            reconstructed, sparse.unpack_ccomp_ref(packed, bitmap)
        ), "unpack != torch ref"

        _roundtrip_ok(
            torch.full(
                (n, config.HEAD_DIM), 1.0, device=DEV, dtype=dtype
            ),
            0.5,
        )
        tied = (
            (torch.arange(config.HEAD_DIM, device=DEV) % 5)[None, :]
            .expand(n, config.HEAD_DIM)
            .to(dtype)
        )
        _roundtrip_ok(tied, 0.5)
        natural_zeros = torch.randn(
            n, config.HEAD_DIM, device=DEV, dtype=dtype
        )
        natural_zeros[:, :300] = 0.0
        zero_mask, _, zero_bitmap, _, _ = _roundtrip_ok(natural_zeros, 0.5)
        assert torch.equal(sparse._bitmap_to_bits(zero_bitmap), zero_mask), (
            "bitmap does not decode to the exact keep-mask"
        )

        for keep in (1.0, 0.5, 0.375, 0.3125):
            keep_k = sparse._keep_count(keep)
            sweep_mask = reference.topmag_keep_mask(x, keep)
            sweep_packed, sweep_bitmap = sparse.pack_ccomp(
                x, sweep_mask, keep_k
            )
            assert sweep_packed.shape == (n, keep_k), (
                keep,
                sweep_packed.shape,
            )
            sweep_reconstructed = sparse.unpack_ccomp(
                sweep_packed, sweep_bitmap
            )
            assert torch.equal(
                sweep_reconstructed, x.masked_fill(~sweep_mask, 0.0)
            ), f"keep={keep}: round-trip mismatch"
            if keep == 1.0:
                assert bool((sweep_bitmap == -1).all()), (
                    "keep=1.0 bitmap not all-ones"
                )
                assert torch.equal(sweep_reconstructed, x), (
                    "keep=1.0 not identity"
                )

        _check_storage(x, packed, bitmap, 0.5)

    bit_zero = torch.zeros(
        1, config.HEAD_DIM, dtype=torch.bool, device=DEV
    )
    bit_zero[0, 0] = True
    assert sparse._mask_to_bitmap(bit_zero)[0, 0].item() == -(2**63), (
        "col 0 -> MSB"
    )
    bit_63 = torch.zeros_like(bit_zero)
    bit_63[0, 63] = True
    assert sparse._mask_to_bitmap(bit_63)[0, 0].item() == 1, (
        "col 63 -> LSB"
    )
    bit_64 = torch.zeros_like(bit_zero)
    bit_64[0, 64] = True
    assert sparse._mask_to_bitmap(bit_64)[0, 1].item() == -(2**63), (
        "col 64 -> word 1 MSB"
    )
    random_mask = torch.rand(n, config.HEAD_DIM, device=DEV) < 0.4
    assert torch.equal(
        sparse._bitmap_to_bits(sparse._mask_to_bitmap(random_mask)),
        random_mask,
    ), "bitmap round-trip does not recover the mask"

    empty = torch.empty(0, config.HEAD_DIM, device=DEV)
    packed_empty, bitmap_empty = sparse.pack_ccomp(
        empty, empty.to(torch.bool), 0
    )
    assert packed_empty.shape == (0, 0)
    assert bitmap_empty.shape == (0, 8)
    assert sparse.unpack_ccomp(packed_empty, bitmap_empty).shape == (
        0,
        config.HEAD_DIM,
    )

    print(
        "[sparseselftest] OK: bit-exact round-trip == dense-zero TopMag "
        "(bf16+fp32), ties+natural-zeros, keep sweep, storage savings, "
        "bitmap pins, zero-row"
    )


def run() -> None:
    """Run both Mustafar numeric unit-test suites."""
    run_topmag()
    run_sparse()


if __name__ == "__main__":
    run()
