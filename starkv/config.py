"""Geometry, record layout, runtime gates and patch targets for STAR-CSA.

STAR-CSA compresses the 448 NoPE dims of the 512-dim C4 compressed latent to a
rank-r code and leaves the 64 RoPE dims exact, so a cached entry is
``[z in R^r, c_R in R^64]`` instead of ``c in R^512``. The compressor, the
Lightning Indexer and every non-c4 pool are untouched.

The native DSV4 record (``deepseek_v4_memory_pool.py``) asserts exactly

    448 + 64*2 + 8 == 584 bytes/token
    nope FP8 (448) + nope FP8 scales (7) + scale pad (1) + rope BF16 (64*2)

so the native store already keeps the RoPE tail at bf16. We mirror that: the
tail bytes keep their precision and only the NoPE payload shrinks.

Runtime switches are read on each access; source paths are fixed at import time.
"""

import os
from pathlib import Path

# --- compressed-latent geometry (fixed by DeepSeek-V4) ----------------------
HEAD_DIM = 512  # full compressed-latent dimension
ROPE_DIM = 64  # rotary tail dims
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448 -- the only part STAR-CSA compresses

# --- rank and record layout -------------------------------------------------
# STARKV_RANK is the rank of the per-layer basis D_l over the 448 NoPE dims.
RANK = int(os.environ.get("STARKV_RANK", "320"))

TILE_SIZE = 64  # native store's fp8 scale tile
SCALE_TILES = (RANK + TILE_SIZE - 1) // TILE_SIZE
ROPE_BYTES = ROPE_DIM * 2  # bf16 tail, same precision the native store uses

# [z: RANK fp8][scales: SCALE_TILES u8][tail: ROPE_BYTES bf16][pad to 4]
Z_OFFSET = 0
SCALE_OFFSET = Z_OFFSET + RANK
ROPE_OFFSET = SCALE_OFFSET + SCALE_TILES
_RAW_BYTES = ROPE_OFFSET + ROPE_BYTES
PAD_BYTES = (-_RAW_BYTES) % 4
BYTES_PER_TOKEN = _RAW_BYTES + PAD_BYTES

NATIVE_RECORD_BYTES = 584  # for the compression ratio only; never allocated here

# The 21 c4 layers (compress_ratio == 4), in model order. One basis file per
# layer: <basis_dir>/D_<layer:03d>_r<rank>.pt
STARKV_LAYERS = tuple(range(2, 43, 2))

# --- sglang patch targets ---------------------------------------------------
# The fork tree is a runtime clone of the image's pristine v0.5.18 tree; the
# anchors below are validated against it by `python -m starkv drift` before any
# patch is written.
SRC_ROOT = os.environ.get("STARKV_SRC", "/sgl-workspace/sglang-starkv/python")
COMPRESSOR_V2 = f"{SRC_ROOT}/sglang/srt/layers/attention/dsv4/compressor_v2.py"
MEM_POOL = f"{SRC_ROOT}/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
POOL_CFG = f"{SRC_ROOT}/sglang/srt/model_executor/pool_configurator.py"
DSV4_BACKEND = f"{SRC_ROOT}/sglang/srt/layers/attention/deepseek_v4_backend.py"
PATCH_FILES = (COMPRESSOR_V2, MEM_POOL, POOL_CFG, DSV4_BACKEND)
MARKER = "## STARKV"

# Package import root: inside the eval container this resolves to the mounted
# flash-optimizations checkout, so the sglang hook's `import starkv` finds the
# live host copy with no docker cp.
PACKAGE_ROOT = os.environ.get(
    "STARKV_PACKAGE_DIR", str(Path(__file__).resolve().parent.parent)
)


def starkv_enabled() -> bool:
    return os.environ.get("SGLANG_OPT_STARKV") == "1"


def recon_mode() -> bool:
    """True for the reconstruct MVE: store low-rank, rebuild 512-D on read.

    The alternative (latent) read path never materialises the NoPE dims; it is
    kernel work and is not wired up yet.
    """
    return os.environ.get("STARKV_RECON", "1") == "1"


def basis_mode() -> str:
    """'global' (frozen offline basis) or 'selffit' (fit on the latents)."""
    return os.environ.get("STARKV_BASIS_MODE", "global")


def debug_enabled() -> bool:
    return os.environ.get("STARKV_DEBUG") == "1"


def ctrl_dir() -> str:
    return os.environ.get("STARKV_CTRL_DIR", str(Path(__file__).resolve().parent / "ctrl"))


def basis_dir() -> str:
    return os.environ.get("STARKV_BASIS", os.path.join(ctrl_dir(), "basis"))


def captures_dir() -> str:
    return os.environ.get(
        "STARKV_CAPTURES", str(Path(__file__).resolve().parent / "captures")
    )


def results_dir() -> str:
    return os.environ.get(
        "STARKV_RESULTS", str(Path(__file__).resolve().parent / "results")
    )


# --- capture caps -----------------------------------------------------------
# The capture hooks run inside the decode loop, so every buffer is bounded and
# written once it fills. A layer is deliberately sampled to one buffer: the
# measurement wants a few thousand rows per layer, not a session-long firehose.


def capture_tag() -> str:
    """Capture session tag; empty string means capture is off."""
    return os.environ.get("STARKV_CAPTURE", "")


def capture_rows() -> int:
    """Runaway guard on stored rows per layer (hook A).

    Not the usual window limit: the window is `capture_steps()` decode calls,
    and a decode call carries only one row per batch element, so this is far
    above what the step cap produces. It exists so a session that somehow feeds
    huge decode batches cannot grow the buffer without bound.
    """
    return int(os.environ.get("STARKV_CAPTURE_ROWS", "16384"))


def capture_steps() -> int:
    """Decode calls to sample per layer -- the length of the capture window.

    Both hooks close their windows here, which is what makes them overlap. Wide
    enough that the drift quartile (`capture_steps()/4` calls) still holds
    more rows than the rank being fitted; at a decode batch of ~15 that is
    ~960 boundary rows at rank 320.
    """
    return int(os.environ.get("STARKV_CAPTURE_STEPS", "256"))


def capture_heads() -> int:
    """Attention heads to sample per decode step (hook B).

    Logit error is a per-head statistic, so a handful of heads estimates it as
    well as all 64 -- at a sixty-fourth of the disk and a sixty-fourth of the
    host-to-device copy per step.
    """
    return int(os.environ.get("STARKV_CAPTURE_HEADS", "8"))


def bytes_for_rank(rank: int) -> int:
    """Record size for an arbitrary rank, for the rank sweep."""
    scales = (rank + TILE_SIZE - 1) // TILE_SIZE
    raw = rank + scales + ROPE_BYTES
    return raw + (-raw) % 4
