"""Fit the W3 cross-layer rank-192 basis from live serving traffic.

The store consumes a 512-dim pre-RoPE RMS-normed latent per CSA boundary token
and needs a `vr = [512, 192]` projection per layer. W3 shares ONE basis across
each consecutive triple of CSA layers, so the object to estimate per triple is
the top-192 eigenspace of the summed second moment

    A = sum_{l in triple} sum_t x_{l,t} x_{l,t}^T          (512 x 512)

A previous version of this study fitted `A` offline and shipped the resulting
`A_<layer>.pt` tensors out of band; those artifacts did not survive, and no
surviving script wrote them (the capture scripts under the old `transferibility/`
tree were analysis-only -- SVD energy / CKA / cosine -- and never dumped a basis).
So the basis is refit here, from the model's own prefill traffic, through the
same patch anchor the store already uses.

Two phases, both driven by xkv/scripts/local/calibrate.sh:

  capture   boot the server with XKV_CALIB=1; every prefill chunk appends
            x^T x into a per-layer running sum and periodically writes the sums
            to $XKV_CALIB_OUT. Output is discarded -- the store is bypassed.
  finalize  `python -m xkv calib_finalize` sums each triple and writes
            `A_<layer>.pt` for all three members, into $SG_LOWRANK_BASIS.

The accumulated sums are rank-scaling invariant (every TP rank sees the identical
latent, and eigenvectors are unaffected by a positive scalar), so no cross-rank
reduction is needed.
"""

import os
import time
from typing import Optional

import torch

from . import config

_S = {}            # layer_id -> [512, 512] fp32 second moment, on device
_n = {}            # layer_id -> token count folded in so far
_out_dir = ""
_last_flush = 0.0


def enabled() -> bool:
    return os.environ.get("XKV_CALIB") == "1"


def out_dir() -> str:
    return os.environ.get("XKV_CALIB_OUT", os.path.join(config.ctrl_dir(), "calib"))


def _flush_interval() -> float:
    return float(os.environ.get("XKV_CALIB_FLUSH_S", "30"))


def maybe_capture(layer_id, kv_compressed, plan, norm, compress_ratio, is_indexer):
    """Fold this prefill chunk's normed latent into the running second moment.

    Mirrors the store's own gate (CSA layers only, boundary tokens only) and its
    RMSNorm arithmetic, so the fitted basis sees exactly the vectors the store
    will later project. Prefill only -- decode contributes one row per request
    per step and would just re-weight the running mean.
    """
    global _out_dir, _last_flush
    if layer_id is None or is_indexer or compress_ratio != 4:
        return
    if kv_compressed is None or kv_compressed.shape[0] == 0:
        return

    x = kv_compressed.detach()
    if x.shape[0] == plan[1].view(torch.int32).shape[0]:
        seq_len = plan[1].view(torch.int32)[:, 0].long()
        x = x[seq_len != -1]           # prefill plan: -1 marks a non-boundary row
        if x.shape[0] == 0:
            return

    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    x = x * norm.weight.float()

    if not _out_dir:
        _out_dir = out_dir()
        os.makedirs(_out_dir, exist_ok=True)
    if not _last_flush:
        # Start the clock at the first captured chunk. `_last_flush` defaults to
        # 0.0, so without this the very first chunk trips the interval check and
        # writes a one-layer sum -- which, if the run died early, would leave a
        # partial S_all.pt that `finalize` rejects.
        _last_flush = time.time()
    acc = _S.get(layer_id)
    if acc is None:
        acc = _S[layer_id] = torch.zeros(config.HEAD_DIM, config.HEAD_DIM,
                                         device=x.device, dtype=torch.float32)
    acc += x.T @ x
    _n[layer_id] = _n.get(layer_id, 0) + int(x.shape[0])

    now = time.time()
    if now - _last_flush >= _flush_interval():
        _last_flush = now
        _write_sums()


def _write_sums():
    """Dump every layer's running sum. Called on a timer, not per chunk: a 64k
    prefill at chunk 8192 is 8 rounds x 21 layers = 168 calls, and re-writing a
    21 MB dict on each would dwarf the capture itself. The driver waits out one
    flush interval after the last request before tearing the server down.

    Every TP rank accumulates the same sums (all ranks see the identical
    latent), so the destination is written by whichever rank flushes last and
    the temp file is per-pid -- a shared temp name is a race, and the loser's
    `os.replace` raises into the scheduler and takes the server down with it.
    """
    if not _out_dir or not _S:
        return
    payload = {int(lid): {"S": s.detach().cpu(), "n": _n.get(lid, 0)}
               for lid, s in _S.items()}
    tmp = os.path.join(_out_dir, f".S_all.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, os.path.join(_out_dir, "S_all.pt"))
    except OSError as exc:
        # A dropped flush costs one interval, not the run: the next one rewrites
        # the same sums, and an empty capture surfaces at `finalize`.
        print(f"[xkv] calib flush failed ({exc}); will retry next interval", flush=True)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def finalize(src: Optional[str] = None, dst: Optional[str] = None) -> int:
    """Sum each consecutive triple of CSA layers and emit `A_<layer>.pt`.

    Deployed W3 shares one basis per triple, so A_<layer> is the *triple's* summed
    second moment, written identically to all three members -- which is what makes
    `vr_for(layer_id)` return the same basis for every layer of a triple.
    """
    src = src or out_dir()
    dst = dst or config.basis_dir()
    path = os.path.join(src, "S_all.pt")
    if not os.path.exists(path):
        raise SystemExit(f"[xkv] no calibration sums at {path}")
    sums = torch.load(path, map_location="cpu")
    missing = [l for l in config.CSA_LAYERS if l not in sums]
    if missing:
        raise SystemExit(f"[xkv] layers never captured: {missing}")

    os.makedirs(dst, exist_ok=True)
    n_triples = 0
    for start in range(0, len(config.CSA_LAYERS), config.WINDOW):
        triple = config.CSA_LAYERS[start:start + config.WINDOW]
        a = sum(sums[l]["S"] for l in triple).float()
        a = (a + a.T) / 2
        toks = sum(sums[l]["n"] for l in triple)
        for l in triple:
            torch.save(a, os.path.join(dst, f"A_{l:03d}.pt"))
        n_triples += 1
        ev = torch.linalg.eigvalsh(a)
        top = ev[-config.COEFF_DIM:].clamp_min(0).sum() / ev.clamp_min(0).sum()
        print(f"[xkv] triple {triple} tokens={toks:>9} "
              f"top-{config.COEFF_DIM} energy={float(top):.4f} -> {dst}/A_{triple[0]:03d}.pt")
    print(f"[xkv] wrote {n_triples} bases covering {len(config.CSA_LAYERS)} CSA layers")
    return n_triples


def run_finalize():
    """CLI entry: `python -m xkv calib_finalize`."""
    finalize()
