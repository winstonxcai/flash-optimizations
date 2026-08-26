import torch

from xkv import config, reference


def test_rank192_layout():
    assert config.COEFF_DIM == 192
    assert config.SCALE_TILES == 3
    assert config.PAD_BYTES == 1
    assert config.COEFF_SCALE_BYTES == 196
    assert config.BYTES_PER_TOKEN == 200


def test_paged_store_round_trip_metadata():
    buf = torch.zeros(2, 4 * config.BYTES_PER_TOKEN, dtype=torch.uint8)
    loc = torch.tensor([0, 4, 5], dtype=torch.long)
    coeff = torch.zeros(3, config.COEFF_DIM, dtype=reference.fp8_dtype)
    scales = torch.tensor([[127, 127, 127]] * 3, dtype=torch.uint8)
    pos = torch.tensor([3, 7, 11], dtype=torch.int32)
    reference.store_torch(buf, loc, coeff, scales, pos, 4)
    raw = buf.view(torch.int32).reshape(-1)
    base = (loc // 4) * buf.shape[1] + (loc % 4) * config.BYTES_PER_TOKEN
    assert torch.equal(raw[(base + config.COEFF_SCALE_BYTES) // 4], pos)


def test_quantize_dequantize_is_finite_and_bounded():
    x = torch.randn(8, config.COEFF_DIM) * 3
    coeff, scales = reference.quantize(x)
    restored = reference.dequantize(coeff, scales)
    assert torch.isfinite(restored).all()
    assert (restored - x).abs().mean() < 0.2


def test_rope_tail_matches_complex_rotation():
    freqs = torch.polar(torch.ones(16, config.ROPE_DIM // 2), torch.full((16, config.ROPE_DIM // 2), 0.5))
    pos = torch.tensor([2], dtype=torch.int32)
    x = torch.zeros(1, config.HEAD_DIM)
    x[:, config.NOPE_DIM:] = 1
    reference.apply_rope_tail(x, freqs, pos)
    assert torch.allclose(x[:, config.NOPE_DIM::2], torch.cos(torch.tensor(0.5)), atol=1e-6)
    assert torch.allclose(x[:, config.NOPE_DIM + 1::2], torch.sin(torch.tensor(0.5)), atol=1e-6)

