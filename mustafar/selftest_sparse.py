"""Numeric self-test for the Stage-0 Triton sparse pack/unpack (compression-only).

Runs on one GPU. Verifies the Stage-0 invariant
    unpack(pack(latent, keep_mask, keep_k)) == dense-zero TopMag(latent)
bit-exactly (original dtype, no FP8), the pure-torch cross-check, the bitmap
bit-convention pins, tie / natural-zero cases, the KEEP_K sparsity sweep, the
physical storage savings, and the zero-row edge.

Run:  cd flash-optimizations && CUDA_VISIBLE_DEVICES=0 python3 -m mustafar sparseselftest
"""
import torch

from . import config, reference, sparse

DEV = "cuda:0"
N = 256


def _roundtrip_ok(x: torch.Tensor, keep: float) -> tuple:
    """Assert round-trip == dense-zero on the SAME mask; return intermediates."""
    keep_k = sparse._keep_count(keep)
    mask = reference.topmag_keep_mask(x, keep)
    assert mask.dtype == torch.bool and mask.shape == x.shape
    counts = mask.sum(-1)
    assert bool((counts == keep_k).all()), \
        f"[{x.dtype}] keep={keep}: mask popcount {counts.unique().tolist()} != {keep_k}"
    packed, bitmap = sparse.pack_ccomp(x, mask, keep_k)
    recon = sparse.unpack_ccomp(packed, bitmap)
    expect = x.masked_fill(~mask, 0.0)                 # dense-zero baseline, same mask
    assert torch.equal(recon, expect), \
        f"[{x.dtype}] keep={keep}: round-trip != dense-zero TopMag"
    return mask, packed, bitmap, recon, expect


def _check_storage(x: torch.Tensor, packed: torch.Tensor,
                   bitmap: torch.Tensor, keep: float) -> None:
    elem = x.element_size()
    packed_bytes = packed.numel() * elem
    bitmap_bytes = bitmap.numel() * 8                  # int64
    dense_bytes = x.numel() * elem
    assert packed_bytes + bitmap_bytes < dense_bytes, \
        (f"[{x.dtype}] keep={keep}: {packed_bytes + bitmap_bytes} B "
         f"!< dense {dense_bytes} B")
    per_row = packed_bytes / x.shape[0] + bitmap_bytes / x.shape[0]
    print(f"    keep={keep}: packed+bitmap {per_row:.0f} B/row < dense "
          f"{dense_bytes / x.shape[0]:.0f} B/row")


def run() -> None:
    torch.manual_seed(0)
    n = N

    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(n, config.HEAD_DIM, device=DEV, dtype=dtype)

        # 1. bit-exact round-trip, shapes/dtypes, kept/pruned coords
        mask, packed, bitmap, recon, _ = _roundtrip_ok(x, 0.5)
        assert packed.dtype == dtype and packed.shape == (n, 256), packed.shape
        assert bitmap.dtype == torch.int64 and bitmap.shape == (n, 8), bitmap.shape
        assert torch.equal(recon[mask], x[mask]), "kept coords not bit-identical"
        assert bool((recon[~mask] == 0).all()), "pruned coords not exactly 0"

        # 2. pure-torch cross-check (valid: every packed slot is written)
        keep_k = sparse._keep_count(0.5)
        packed_ref, bitmap_ref = sparse.pack_ccomp_ref(x, mask, keep_k)
        assert torch.equal(packed, packed_ref), "packed != torch ref"
        assert torch.equal(bitmap, bitmap_ref), "bitmap != torch ref"
        assert torch.equal(recon, sparse.unpack_ccomp_ref(packed, bitmap)), \
            "unpack != torch ref"

        # 4. tie cases (same mask consumed by both paths, bit-exact round-trip)
        _roundtrip_ok(torch.full((n, config.HEAD_DIM), 1.0,
                                 device=DEV, dtype=dtype), 0.5)     # all-equal |.|
        tie = (torch.arange(config.HEAD_DIM, device=DEV) % 5)[None, :].expand(
            n, config.HEAD_DIM).to(dtype)
        _roundtrip_ok(tie, 0.5)                                      # many-way ties
        z = torch.randn(n, config.HEAD_DIM, device=DEV, dtype=dtype)
        z[:, :300] = 0.0                                            # >256 natural zeros
        zmask, zpacked, zbitmap, zrecon, _ = _roundtrip_ok(z, 0.5)
        assert torch.equal(sparse._bitmap_to_bits(zbitmap), zmask), \
            "bitmap does not decode to the exact keep-mask"

        # 5. sparsity sweep via KEEP_K variants
        for keep in (1.0, 0.5, 0.375, 0.3125):
            keep_k = sparse._keep_count(keep)
            m = reference.topmag_keep_mask(x, keep)
            p, b = sparse.pack_ccomp(x, m, keep_k)
            assert p.shape == (n, keep_k), (keep, p.shape)
            assert torch.equal(sparse.unpack_ccomp(p, b), x.masked_fill(~m, 0.0)), \
                f"keep={keep}: round-trip mismatch"
            if keep == 1.0:
                assert bool((b == -1).all()), "keep=1.0 bitmap not all-ones"
                assert torch.equal(sparse.unpack_ccomp(p, b), x), \
                    "keep=1.0 not identity"

        # 6. storage savings are physical
        _check_storage(x, packed, bitmap, 0.5)

    # 3. bit-convention pins (MSB = lane 0)
    b0 = torch.zeros(1, config.HEAD_DIM, dtype=torch.bool, device=DEV)
    b0[0, 0] = True
    assert sparse._mask_to_bitmap(b0)[0, 0].item() == -2**63, "col 0 -> MSB"
    b63 = torch.zeros_like(b0); b63[0, 63] = True
    assert sparse._mask_to_bitmap(b63)[0, 0].item() == 1, "col 63 -> LSB"
    b64 = torch.zeros_like(b0); b64[0, 64] = True
    assert sparse._mask_to_bitmap(b64)[0, 1].item() == -2**63, "col 64 -> word 1 MSB"
    rand = torch.rand(n, config.HEAD_DIM, device=DEV) < 0.4
    assert torch.equal(sparse._bitmap_to_bits(sparse._mask_to_bitmap(rand)), rand), \
        "bitmap round-trip does not recover the mask"

    # 7. zero-row edge (no kernel launch)
    empty = torch.empty(0, config.HEAD_DIM, device=DEV)
    p0, bm0 = sparse.pack_ccomp(empty, empty.to(torch.bool), 0)
    assert p0.shape == (0, 0) and bm0.shape == (0, 8), (p0.shape, bm0.shape)
    assert sparse.unpack_ccomp(p0, bm0).shape == (0, config.HEAD_DIM)

    print(f"[sparseselftest] OK: bit-exact round-trip == dense-zero TopMag "
          f"(bf16+fp32), ties+natural-zeros, keep sweep, storage savings, "
          f"bitmap pins, zero-row")


if __name__ == "__main__":
    run()
