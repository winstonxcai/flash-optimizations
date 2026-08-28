"""Store-time TopMag pruning hook + patch CLI (native c4-latent scope).

This module is imported by the sglang hook as `import mustafar as _sg_lr`.
The injected call `_sg_lr.maybe_prune(kv_compressed)` zeroes the smallest-|.|
(1 - XKV_TOPMAG_KEEP) fraction of each c4 latent vector in place, right before
the native fused store. Everything else (memory pool, layout, decode) is the
stock DeepSeek-V4 build — no lowrank KV anywhere.
"""
import json
import os
import shutil
import subprocess

import torch

from . import config
from . import reference


def _dbg(msg: str, **fields) -> None:
    if os.environ.get("XKV_DEBUG") == "1":
        line = json.dumps({"mustafar": msg, **fields}, default=str)
        os.makedirs(config.ctrl_dir(), exist_ok=True)
        with open(os.path.join(config.ctrl_dir(), "debug.log"), "a") as f:
            f.write(line + "\n")


_dbg("import", src=__file__, head_dim=config.HEAD_DIM,
     keep=config.topmag_keep())


def maybe_prune(kv_compressed) -> None:
    """Hook injected into compressor_v2 just before the native c4 fused store.

    Computes the exact-global keep-mask ONCE from the unmodified latent, then
    passes that same mask to (a) the dense-zero baseline and (b), only when
    XKV_SPARSE_SHADOW=1, the Stage-0 Triton pack->unpack shadow check. The
    native store then quantizes+writes the pruned latent exactly as usual. With
    the shadow flag off the server performs no sparse work and is numerically
    identical to the previous scatter-based topmag_zero.
    """
    if not config.topmag_enabled():
        return
    if kv_compressed is None or kv_compressed.numel() == 0:
        return
    if kv_compressed.shape[-1] < config.HEAD_DIM:
        # Non-c4 latent (e.g. the 128-dim indexer) — leave untouched.
        _dbg("prune_skip", dim=int(kv_compressed.shape[-1]))
        return
    try:
        keep = config.topmag_keep()
        shadow = config.sparse_shadow()
        orig = kv_compressed.clone() if shadow else None
        keep_mask = reference.topmag_keep_mask(kv_compressed, keep)
        reference.topmag_zero_from_mask(kv_compressed, keep_mask)
        if shadow:
            from . import sparse as _sparse
            keep_k = _sparse._keep_count(keep)
            packed, bitmap = _sparse.pack_ccomp(orig, keep_mask, keep_k)
            recon = _sparse.unpack_ccomp(packed, bitmap)
            ok = bool(torch.equal(recon, kv_compressed))
            _dbg("shadow", ok=ok, rows=int(kv_compressed.shape[0]),
                 packed_bytes=int(packed.numel() * packed.element_size()),
                 mismatch=int((recon != kv_compressed).sum().item()) if not ok else 0)
        if os.environ.get("XKV_DEBUG") == "1":
            _dbg("prune", rows=int(kv_compressed.shape[0]),
                 zeroed=config.topmag_zero_count())
    except Exception as e:
        _dbg("prune_error", err=repr(e),
             dim=int(kv_compressed.shape[-1]),
             ndim=kv_compressed.dim())


# --- patch / unpatch / verify --------------------------------------------------
def _apply(path: str, edits) -> None:
    with open(path) as f:
        s = f.read()
    if config.MARKER in s:
        print(f"[mustafar] {path}: already patched")
        return
    for anchor, new in edits:
        if s.count(anchor) != 1:
            raise AssertionError(
                f"[mustafar] anchor count != 1 in {path}: {anchor[:70]!r}")
        s = s.replace(anchor, new, 1)
    with open(path, "w") as f:
        f.write(s)
    print(f"[mustafar] {path}: ok")


def _import_block() -> str:
    return ("\n" + config.MARKER + " (import)\n"
            "import sys as _sg_lr_sys\n"
            f"if {config.PACKAGE_ROOT!r} not in _sg_lr_sys.path:\n"
            f"    _sg_lr_sys.path.insert(0, {config.PACKAGE_ROOT!r})\n"
            "import mustafar as _sg_lr\n")


def patch() -> None:
    os.makedirs(config.ctrl_dir(), exist_ok=True)
    stale = [p for p in config.OLD_PATCH_FILES
             if "XKV_LOWRANK" in open(p).read()]
    if stale:
        raise SystemExit(
            f"[mustafar] old XKV_LOWRANK patch still present in {stale}; "
            "run 'python -m mustafar unpatch' first")

    _apply(config.COMPRESSOR_V2, [
        ("from __future__ import annotations\n",
         "from __future__ import annotations\n" + _import_block()),
        ("        # Step 2: norm + rope + store\n"
         "        compress_norm_rope_store(\n",
         "        " + config.MARKER + " (topmag)\n"
         "        _sg_lr.maybe_prune(kv_compressed)\n"
         "        # Step 2: norm + rope + store\n"
         "        compress_norm_rope_store(\n"),
    ])


def unpatch() -> None:
    """Restore all 4 previously-patched files to pristine, clearing the old
    lowrank injection (transferibility's `## XKV_LOWRANK`)."""
    for p in config.OLD_PATCH_FILES:
        bak = p + ".lr.bak"
        if os.path.exists(bak):
            shutil.copy(bak, p)
            print(f"[mustafar] restored {p} from .lr.bak")
        else:
            subprocess.run(["git", "checkout", "--", p], cwd=config.SRC_ROOT,
                           check=False)
            print(f"[mustafar] git checkout {p}")


def verify() -> None:
    for p in config.OLD_PATCH_FILES:
        with open(p) as f:
            s = f.read()
        print(f"{p}: mustafar_markers={s.count(config.MARKER)} "
              f"old_lowrank={'PRESENT' if 'XKV_LOWRANK' in s else 'absent'}")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"patch": patch, "unpatch": unpatch, "verify": verify}[cmd]()
