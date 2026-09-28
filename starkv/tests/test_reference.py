"""Reference maths and record-layout checks for STAR-CSA.

Runs on CPU: `python -m starkv selftest`. Each test is also a pytest function.

The three identities (see reference.py) are checked against independently
computed quantities, not against the same helper twice.
"""

import torch

from .. import config, reference


def _random_latents(m=512, seed=0, scale=1.0):
    """Rows of random 448-dim latents.

    The default row count is above NOPE_DIM so that `fit_basis` can produce a
    basis at any rank in the sweep: an SVD cannot return more right singular
    vectors than it has rows.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, config.NOPE_DIM, generator=g) * scale


def _random_head_latents(m=8, seed=0):
    """Full 512-dim latents, for the norm/rope helpers that touch the tail."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, config.HEAD_DIM, generator=g)


def _subspace_latents(m=512, r_true=64, seed=0, noise=0.0):
    """Latents that actually live near a `r_true`-dim subspace.

    Isotropic random rows are the wrong fixture for a retention claim: every row
    uses all 448 directions, so a rank-r basis keeps only r/448 of the energy no
    matter how well it is fit. Real latents concentrate, which is the whole
    premise, so the fixture has to concentrate too.
    """
    g = torch.Generator().manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(config.NOPE_DIM, r_true, generator=g))[0]
    coeff = torch.randn(m, r_true, generator=g)
    x = coeff @ basis.T
    if noise:
        x = x + noise * torch.randn(m, config.NOPE_DIM, generator=g)
    return x


def test_roundtrip_at_full_rank_is_exact():
    """r = 448 with the identity basis must reproduce x bit-exactly."""
    x = _random_latents()
    d = torch.eye(config.NOPE_DIM)
    z = reference.encode(x, d)
    assert z.shape == (x.shape[0], config.NOPE_DIM)
    assert torch.equal(reference.reconstruct(z, d).float(), x.float())


def test_low_rank_roundtrip_is_rank_limited():
    """r < 448 is lossy, and the residual lies outside the basis span."""
    x = _random_latents(m=256)
    d = reference.fit_basis(x, 128)
    assert d.shape == (config.NOPE_DIM, 128)
    assert torch.allclose(d.T @ d, torch.eye(128), atol=1e-5)
    x_hat = reference.reconstruct(reference.encode(x, d), d)
    resid = x - x_hat
    # Every residual column must be orthogonal to the basis.
    assert (d.T @ resid.T).abs().max() < 1e-4


def test_selffit_retention_is_high_and_frozen_basis_is_not():
    """The headline distinction: a basis fit on the latents being compressed
    spans them, a basis fit on unrelated latents does not."""
    x = _subspace_latents(m=512, r_true=64, seed=1)
    d_self = reference.fit_basis(x, 128)
    assert reference.retention(x, d_self).min() > 0.99

    other = _subspace_latents(m=512, r_true=64, seed=2)
    d_frozen = reference.fit_basis(other, 128)
    assert reference.retention(x, d_frozen).mean() < 0.9


def test_score_identity_latent_matches_reconstruct():
    """q_N^T (D z) == (D^T q_N)^T z, to fp accumulation order."""
    x = _random_latents(seed=3)
    d = reference.fit_basis(x, config.RANK)
    z = reference.encode(x, d)
    tail = torch.randn(x.shape[0], config.ROPE_DIM)
    q_nope = torch.randn(16, config.NOPE_DIM)
    q_rope = torch.randn(16, config.ROPE_DIM)

    latent = reference.latent_scores(q_nope, z, q_rope, tail, d, scale=0.5)
    recon = reference.reconstruct_scores(q_nope, z, q_rope, tail, d, scale=0.5)
    assert latent.shape == recon.shape == (16, x.shape[0])
    assert torch.allclose(latent, recon, atol=1e-3, rtol=1e-4)


def test_value_identity_and_output_absorption():
    """sum_i p_i [D z_i | tail_i] equals the latent aggregation, and W_o,a
    absorbs the reconstruction through its NoPE block."""
    x = _random_latents(seed=4)
    d = reference.fit_basis(x, config.RANK)
    z = reference.encode(x, d)
    tail = torch.randn(x.shape[0], config.ROPE_DIM)
    probs = torch.softmax(torch.randn(8, x.shape[0]), dim=-1)

    o = reference.latent_values(probs, z, tail, d)
    assert o.shape == (8, config.HEAD_DIM)

    # Reconstruct-then-aggregate must give the same thing.
    k = reference.assemble(z, tail, d)
    assert torch.allclose(o, probs @ k, atol=1e-3, rtol=1e-4)

    # Absorption: the group projection consumes [D z_bar | tail_bar], and the
    # folded weights reproduce it from the latent alone.
    w = torch.randn(4, config.HEAD_DIM)
    w_folded = reference.absorb_output_projection(w, d)
    assert w_folded.shape == (4, config.RANK)
    z_bar = probs @ z
    tail_bar = probs @ tail
    assert torch.allclose(
        z_bar @ w_folded.T + tail_bar @ w[:, config.NOPE_DIM :].T,
        (probs @ k) @ w.T,
        atol=1e-3,
        rtol=1e-4,
    )


def test_output_absorption_only_touches_the_nope_block():
    """A projection that ignores the tail block must be unaffected by it."""
    x = _random_latents(m=512, seed=6)
    d = reference.fit_basis(x, config.RANK)
    z = reference.encode(x[:8], d)
    w = torch.randn(4, config.HEAD_DIM)
    w[:, config.NOPE_DIM :] = 0.0
    k = reference.assemble(z, torch.randn(8, config.ROPE_DIM), d)
    assert torch.allclose(
        z @ reference.absorb_output_projection(w, d).T,
        k @ w.T,
        atol=1e-3,
        rtol=1e-4,
    )


def test_record_layout_matches_formula():
    """The record is [z][scales][bf16 tail][pad], sized by config."""
    assert config.NOPE_DIM == 448
    assert config.ROPE_DIM == 64
    assert config.ROPE_BYTES == 128
    assert config.SCALE_OFFSET == config.RANK
    assert config.ROPE_OFFSET == config.RANK + config.SCALE_TILES
    assert config.BYTES_PER_TOKEN % 4 == 0
    # Equality only at the no-compression cell (RANK == NOPE_DIM, i.e. 584 B);
    # every rank below it must come in under the native record.
    assert config.BYTES_PER_TOKEN <= config.NATIVE_RECORD_BYTES


def test_rank_sweep_sizes_are_monotone_and_smaller_than_native():
    sizes = {r: config.bytes_for_rank(r) for r in (128, 192, 256, 320, 384, 448)}
    assert list(sizes.values()) == sorted(sizes.values())
    assert all(v <= config.NATIVE_RECORD_BYTES for v in sizes.values())
    # The headline cells the study reports.
    assert sizes[320] == 456
    assert sizes[256] == 388


def test_pack_unpack_roundtrip():
    """fp8 quantization is lossy; the bf16 tail must come back exactly."""
    x = _random_latents(seed=5)
    d = reference.fit_basis(x, config.RANK)
    z = reference.encode(x, d)
    tail = torch.randn(x.shape[0], config.ROPE_DIM)

    rec = reference.pack_record(z, tail)
    assert rec.shape == (x.shape[0], config.BYTES_PER_TOKEN)
    assert rec.dtype == torch.uint8

    z_hat, tail_hat = reference.unpack_record(rec)
    assert torch.equal(tail_hat, tail.to(torch.bfloat16).float())
    rel = (z_hat - z).norm() / z.norm()
    assert rel < 0.05, f"fp8 payload error too large: {rel}"


def test_pool_store_and_gather_roundtrip():
    """The page-offset arithmetic survives scattered writes and page crossings.

    A wrong stride here is silent: records land in the wrong slot and the model
    reads a neighbouring token's latent. So gather back both by the written ids
    and by an independently computed page walk.
    """
    page_size, pages = 4, 8
    buf = torch.zeros(pages, page_size * config.BYTES_PER_TOKEN, dtype=torch.uint8)
    loc = torch.tensor([0, 3, 4, 7, 17, 31], dtype=torch.long)

    z = torch.randn(loc.numel(), config.RANK)
    tail = torch.randn(loc.numel(), config.ROPE_DIM)
    records = reference.pack_record(z, tail)

    reference.store_records(buf, loc, records, page_size)
    got = reference.gather_records(buf, loc, page_size)
    assert torch.equal(got, records)

    # An independent walk of the same layout, token by token, over the flat
    # byte space (slicing the 2-D buffer would walk pages, not bytes).
    flat = buf.reshape(-1)
    for i, tok in enumerate(loc.tolist()):
        page, slot = divmod(tok, page_size)
        start = page * buf.shape[-1] + slot * config.BYTES_PER_TOKEN
        assert torch.equal(flat[start : start + config.BYTES_PER_TOKEN], records[i])

    # Distinct slots must not alias.
    assert not torch.equal(got[0], got[1])


def test_rms_norm_matches_its_definition():
    x = _random_head_latents(m=8, seed=7)
    weight = torch.rand(config.HEAD_DIM) + 0.5
    got = reference.rms_norm(x, weight, 1e-6)
    want = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * weight
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_rope_matches_interleaved_complex_rotation():
    """Pair j is (2j, 2j+1); the rotation is a complex multiply per pair."""
    n, max_pos = 5, 16
    x = _random_head_latents(m=n, seed=8)
    freqs = torch.polar(
        torch.ones(max_pos, config.ROPE_DIM // 2),
        torch.rand(max_pos, config.ROPE_DIM // 2),
    )
    pos = torch.tensor([0, 3, 7, 15, 2])

    got = reference.apply_rope(x, freqs, pos)
    assert got.shape == (n, config.ROPE_DIM)

    pairs = x[:, config.NOPE_DIM:].view(n, config.ROPE_DIM // 2, 2)
    complex_pairs = torch.complex(pairs[..., 0], pairs[..., 1])
    want = (complex_pairs * freqs[pos]).view(n * (config.ROPE_DIM // 2))
    got_complex = torch.complex(got[:, 0::2].reshape(-1), got[:, 1::2].reshape(-1))
    assert torch.allclose(got_complex, want, atol=1e-4, rtol=1e-4)


def test_store_zeroes_non_boundary_rows_but_keeps_the_slot():
    """Decode writes every row; non-boundary rows are zeroed onto loc 0.

    This mirrors the vendor's own store, which zero-masks non-boundary decode
    rows rather than compacting them, so the op stays fixed-shape.
    """
    page_size = 4
    buf = torch.zeros(4, page_size * config.BYTES_PER_TOKEN, dtype=torch.uint8)
    z = torch.randn(3, config.RANK)
    tail = torch.randn(3, config.ROPE_DIM)
    records = reference.pack_record(z, tail)
    boundary = torch.tensor([True, False, True])
    records = torch.where(boundary[:, None], records, records.new_zeros(()))

    reference.store_records(buf, torch.tensor([5, 0, 6]), records, page_size)
    got = reference.gather_records(buf, torch.tensor([5, 0, 6]), page_size)
    assert torch.equal(got[0], records[0])
    assert torch.equal(got[1], torch.zeros_like(got[1]))
    assert torch.equal(got[2], records[2])


TESTS = (
    test_roundtrip_at_full_rank_is_exact,
    test_low_rank_roundtrip_is_rank_limited,
    test_selffit_retention_is_high_and_frozen_basis_is_not,
    test_score_identity_latent_matches_reconstruct,
    test_value_identity_and_output_absorption,
    test_output_absorption_only_touches_the_nope_block,
    test_record_layout_matches_formula,
    test_rank_sweep_sizes_are_monotone_and_smaller_than_native,
    test_pack_unpack_roundtrip,
    test_pool_store_and_gather_roundtrip,
    test_rms_norm_matches_its_definition,
    test_rope_matches_interleaved_complex_rotation,
    test_store_zeroes_non_boundary_rows_but_keeps_the_slot,
)


def run() -> int:
    failures = 0
    for t in TESTS:
        try:
            t()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {t.__name__}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
