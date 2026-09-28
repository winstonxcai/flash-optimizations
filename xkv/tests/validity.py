"""Validity checks for the W3 low-rank store: `python -m xkv selftest`.

Four properties, in the order the runtime depends on them:

  quant      ue8m0 per-64-tile fp8 quantize/dequantize stays within the
             mantissa bound the coefficient record was sized for.
  store      the masked triton store writes exactly the rows the eager
             filter-then-scatter path writes, and leaves every other byte of the
             pool alone. That equivalence is the cuda-graph fix: decode keeps a
             row per plan entry and masks, rather than compacting in torch.
  recon      the fused triton reconstruction matches the torch reference on
             identical inputs, to fp8 reconstruction error.
  padding    a -1 padding slot does not turn into a negative offset. The op is
             reached with clamped indices, but the kernel clamps too.

Needs a CUDA device (triton). Run inside the xkv container:
`python -m xkv selftest`.
"""

import os
import shutil
import tempfile

import torch

from .. import config, reference
from ..triton import fused_indexer, score_cache

PAGE_SIZE = 64
PAGES = 8
LAYER = 7
TOL_QUANT = 2.0 ** -3          # e4m3 keeps 3 mantissa bits
TOL_RECON = 0.25               # accumulated over 64 tiles x a rank-192 basis


class _Norm:
    """Stand-in for the compressor's RMSNorm module."""

    def __init__(self, dim, device):
        self.weight = torch.randn(dim, device=device).abs() + 0.5
        self.variance_epsilon = 1e-6


def _device():
    if not torch.cuda.is_available():
        raise SystemExit("[xkv] selftest needs a CUDA device; run it in the xkv container")
    return torch.device("cuda")


def _basis_dir(device):
    """A random SPD second moment per layer, saved where `reference` looks."""
    path = tempfile.mkdtemp(prefix="xkv-selftest-")
    a = torch.randn(config.HEAD_DIM, config.HEAD_DIM)
    a = a @ a.T + config.HEAD_DIM * torch.eye(config.HEAD_DIM)
    torch.save(a, os.path.join(path, f"A_{LAYER:03d}.pt"))
    reference.set_basis_dir(path)
    return path


def _freqs(max_pos, device):
    half = config.ROPE_DIM // 2
    theta = 10000.0 ** (-torch.arange(half, device=device, dtype=torch.float32) / half)
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)[:, None]
    ang = pos * theta[None, :]
    return torch.polar(torch.ones_like(ang), ang)


def _normed(x, norm):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    return x * norm.weight.float()


def _check(name, ok, detail):
    print(f"[xkv] {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def _test_quant(device):
    x = torch.randn(37, config.COEFF_DIM, device=device)
    coeff, scale = reference.quantize(x)
    back = reference.dequantize(coeff, scale)
    err = (back - x).abs().max().item()
    rel = err / x.abs().max().item()
    return _check("quant", rel < TOL_QUANT,
                  f"max_rel_err={rel:.2e} (tol {TOL_QUANT:.2e})")


def _test_store(device):
    """Masked store == filter-then-scatter, and nothing else is touched."""
    n, page_bytes = 32, PAGE_SIZE * config.BYTES_PER_TOKEN
    loc = torch.randint(0, PAGE_SIZE * PAGES, (n,), device=device)
    valid = torch.rand(n, device=device) < 0.5
    x = _normed(torch.randn(n, config.HEAD_DIM, device=device), _Norm(config.HEAD_DIM, device))
    vr = reference.vr_for(LAYER, device)
    coeff, scale = reference.quantize(x @ vr)
    pos = torch.randint(0, 4096, (n,), device=device).to(torch.int32)

    # Path A: the runtime's decode form -- every row kept, masked in-kernel.
    masked = torch.zeros(PAGES, page_bytes, dtype=torch.uint8, device=device)
    score_cache.store(masked, loc, coeff, scale, pos, PAGE_SIZE, valid=valid)
    # Path B: the eager form -- drop the rows, then scatter what remains.
    filtered = torch.zeros(PAGES, page_bytes, dtype=torch.uint8, device=device)
    reference.store_torch(filtered, loc, coeff, scale, pos, PAGE_SIZE, valid=valid)

    same = torch.equal(masked, filtered)
    # Every row the mask excluded must still be a hole in both buffers. A slot
    # written by a *kept* row is exempt: duplicate locs are a real possibility.
    flat = masked.view(-1)
    touched = set(loc[valid].tolist())
    untouched = all(
        not flat[l * config.BYTES_PER_TOKEN:(l + 1) * config.BYTES_PER_TOKEN].any()
        for l in loc[~valid].tolist() if l not in touched
    )
    return _check("store", same and untouched,
                  f"identical_buffers={same} rejected_rows_untouched={untouched} "
                  f"kept={int(valid.sum())}/{n}")


def _test_recon(device):
    """Fused triton reconstruction vs the torch reference."""
    n = 64
    page_bytes = PAGE_SIZE * config.BYTES_PER_TOKEN
    buf = torch.zeros(PAGES, page_bytes, dtype=torch.uint8, device=device)
    loc = torch.randint(0, PAGE_SIZE * PAGES, (n,), device=device)
    x = _normed(torch.randn(n, config.HEAD_DIM, device=device), _Norm(config.HEAD_DIM, device))
    vr = reference.vr_for(LAYER, device)
    coeff, scale = reference.quantize(x @ vr)
    pos = torch.randint(0, 4096, (n,), device=device).to(torch.int32)
    score_cache.store(buf, loc, coeff, scale, pos, PAGE_SIZE)

    reference.set_freqs(_freqs(4096, device))
    a = torch.empty(n, 1, config.HEAD_DIM, dtype=torch.bfloat16, device=device)
    b = torch.empty(n, 1, config.HEAD_DIM, dtype=torch.bfloat16, device=device)
    fused_indexer.reconstruct(buf, loc, page_size=PAGE_SIZE, layer_id=LAYER, out=a,
                              freqs_cis=reference._freqs_cis)
    reference.reconstruct_torch(buf, loc, page_size=PAGE_SIZE, layer_id=LAYER, out=b)

    diff = (a.float() - b.float()).abs().max().item()
    scale_ref = b.float().abs().max().item()
    finite = bool(torch.isfinite(a.float()).all())
    return _check("recon", finite and diff / max(scale_ref, 1e-6) < TOL_RECON,
                  f"max_abs_diff={diff:.3e} rel={diff / max(scale_ref, 1e-6):.2e} "
                  f"finite={finite}")


def _test_padding(device):
    """A -1 slot is clamped, not wrapped to a negative offset."""
    n = 8
    page_bytes = PAGE_SIZE * config.BYTES_PER_TOKEN
    buf = torch.zeros(PAGES, page_bytes, dtype=torch.uint8, device=device)
    loc = torch.tensor([-1] * 4 + [3, 5, 7, 9], device=device)
    coeff, scale = reference.quantize(
        _normed(torch.randn(n, config.HEAD_DIM, device=device), _Norm(config.HEAD_DIM, device))
        @ reference.vr_for(LAYER, device))
    score_cache.store(buf, loc.clamp_min(0), coeff, scale,
                      torch.zeros(n, dtype=torch.int32, device=device), PAGE_SIZE)

    reference.set_freqs(_freqs(64, device))
    out = torch.empty(n, 1, config.HEAD_DIM, dtype=torch.bfloat16, device=device)
    fused_indexer.reconstruct(buf, loc, page_size=PAGE_SIZE, layer_id=LAYER, out=out,
                              freqs_cis=reference._freqs_cis)
    ok = bool(torch.isfinite(out.float()).all())
    return _check("padding", ok, f"no_NaN_or_Inf_with_{int((loc < 0).sum())}_negative_slots={ok}")


def run_reference():
    device = _device()
    basis = _basis_dir(device)
    try:
        if reference.prewarm([LAYER], device) != 1:
            raise SystemExit(f"[xkv] selftest could not load the test basis from {basis}")
        results = [
            _test_quant(device),
            _test_store(device),
            _test_recon(device),
            _test_padding(device),
        ]
    finally:
        shutil.rmtree(basis, ignore_errors=True)
    passed = sum(results)
    print(f"[xkv] selftest: {passed}/{len(results)} passed")
    if passed != len(results):
        raise SystemExit(1)
