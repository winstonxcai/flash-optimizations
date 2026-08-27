"""Numeric self-test for TopMag pruning on the native c4 latent.

Runs on one GPU. Verifies:
  1. keep=0.5 zeros exactly 256 of 512 coords per row.
  2. Non-zeroed coords are bit-identical to the input (only zeros introduced).
  3. The zeroed set is exactly the smallest-|.|-k per row.
  4. keep=1.0 is a no-op.

Run:  cd flash-optimizations && CUDA_VISIBLE_DEVICES=0 python3 -m mustafar selftest
"""
import torch

from . import config, reference


def run():
    torch.manual_seed(0)
    dev = "cuda:0"
    n, k = 256, config.HEAD_DIM - int(round(config.HEAD_DIM * 0.5))  # 256
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(n, config.HEAD_DIM, device=dev, dtype=dtype)
        orig = x.clone()
        reference.topmag_zero(x, 0.5)

        nz = (x == 0).sum(-1)
        assert (nz == k).all(), \
            f"[{dtype}] expected {k} zero coords/row, got {nz.unique().tolist()}"

        mask = x != 0
        assert torch.equal(x[mask], orig[mask]), \
            f"[{dtype}] non-zeroed coords changed (should be bit-identical)"

        # Zeroed set == the topk smallest-|.| indices (same call the prune made;
        # compare index sets directly, since bf16 ties at the k-th cutoff make a
        # magnitude-threshold comparison select >k tied values).
        mag = orig.abs().float()
        kidx = mag.topk(k, dim=-1, largest=False).indices
        expected = torch.zeros_like(x)
        expected.scatter_(-1, kidx, 1.0)
        assert torch.equal(expected.bool(), x == 0), \
            f"[{dtype}] zeroed set != smallest-{k}-per-row"

        y = orig.clone()
        reference.topmag_zero(y, 1.0)
        assert torch.equal(y, orig), f"[{dtype}] keep=1.0 not a no-op"

    print(f"[selftest] OK: topmag_zero keep=0.5 -> exactly {k}/row, "
          f"zeros=smallest-{k}, nonzero coords bit-identical, keep=1.0 no-op")


if __name__ == "__main__":
    run()
