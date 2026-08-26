import pytest
import torch

from xkv import config, reference


def test_projection_reconstruction_rank192():
    torch.manual_seed(0)
    basis, _ = torch.linalg.qr(torch.randn(config.HEAD_DIM, config.COEFF_DIM))
    x = torch.randn(4, config.HEAD_DIM)
    coeff, scales = reference.quantize(x @ basis)
    restored = reference.dequantize(coeff, scales) @ basis.T
    assert restored.shape == x.shape
    assert torch.isfinite(restored).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton unavailable")
def test_triton_reconstruction_matches_reference():
    from xkv.triton import fused_indexer

    device = torch.device("cuda")
    layer = 0
    basis, _ = torch.linalg.qr(torch.randn(config.HEAD_DIM, config.COEFF_DIM, device=device))
    reference._Vr[layer] = basis.cpu()
    reference._VrT[layer] = basis.T.cpu()
    reference._Vr_dev.clear()
    reference._VrT_dev.clear()
    reference._VrT_bf16_dev.clear()
    freqs = torch.polar(torch.ones(64, config.ROPE_DIM // 2, device=device), torch.zeros(64, config.ROPE_DIM // 2, device=device))
    reference.set_freqs(freqs)
    buf = torch.zeros(2, 32 * config.BYTES_PER_TOKEN, dtype=torch.uint8, device=device)
    loc = torch.arange(32, device=device, dtype=torch.long)
    coeff, scales = reference.quantize(torch.randn(32, config.COEFF_DIM, device=device))
    reference.store_torch(buf, loc, coeff, scales, torch.arange(32, device=device, dtype=torch.int32), 32)
    out_ref = torch.zeros(32, 1, config.HEAD_DIM, dtype=torch.bfloat16, device=device)
    out_tri = torch.zeros_like(out_ref)
    reference.reconstruct_torch(buf, loc, page_size=32, layer_id=layer, out=out_ref)
    fused_indexer.reconstruct(buf, loc, page_size=32, layer_id=layer, out=out_tri, freqs_cis=freqs)
    torch.cuda.synchronize()
    assert torch.allclose(out_ref, out_tri, atol=0.02, rtol=0.02)

