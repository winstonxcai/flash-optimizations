"""TopMag pruning and packed storage for the native CSA c4 latent.

Scope: NO lowrank KV — nothing related to the windowed self-fit / xkv build.
Dense mode retains the original store-time zeroing hook. Packed mode replaces
only the C4 cache with a fixed-stride bitmap/FP8 record; the C4 indexer and CSA
attention consumer keep their native semantics.

Env switches are read at import time (the serving process sets them first).
"""
import os
from pathlib import Path

# --- c4 latent geometry (fixed by DeepSeek-V4) --------------------------------
HEAD_DIM = 512          # full c4 compressor latent dim
ROPE_DIM = 64           # rotary tail dims
NOPE_DIM = HEAD_DIM - ROPE_DIM          # 448
TILE_SIZE = 64          # native store's fp8 scale tile
SCALE_TILES = HEAD_DIM // TILE_SIZE
KEEP_DIM = HEAD_DIM // 2
BITMAP_WORDS = HEAD_DIM // 64
VALUES_BYTES = KEEP_DIM
BITMAP_BYTES = BITMAP_WORDS * 8
SCALES_BYTES = SCALE_TILES
PACKED_BYTES_PER_ROW = VALUES_BYTES + BITMAP_BYTES + SCALES_BYTES  # 328
NATIVE_BYTES_PER_ROW = 584
C4_RATIO = 4
C4_TOPK = 512

# --- env switches ---------------------------------------------------------------
# Gate: SGLANG_OPT_TOPMAG=1 enables the prune hook. Keep fraction:
# XKV_TOPMAG_KEEP=0.5 keeps the largest 50% of coords per vector (zeros the rest).
TOPMAG_KEEP = float(os.environ.get("XKV_TOPMAG_KEEP", "1.0"))

# --- sglang patch targets -------------------------------------------------------
# We patch ONLY the compressor store site. The 4-file list is kept for unpatch
# (clearing the previous lowrank injection) and verify.
SRC_ROOT = os.environ.get("SG_LOWRANK_SRC", "/sgl-workspace/sglang-lowrank/python")
COMPRESSOR_V2 = f"{SRC_ROOT}/sglang/srt/layers/attention/dsv4/compressor_v2.py"
MEM_POOL = f"{SRC_ROOT}/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
POOL_CFG = f"{SRC_ROOT}/sglang/srt/model_executor/pool_configurator.py"
DSV4_BACKEND = f"{SRC_ROOT}/sglang/srt/layers/attention/deepseek_v4_backend.py"
PATCH_FILES = (COMPRESSOR_V2, MEM_POOL, POOL_CFG, DSV4_BACKEND)
OLD_PATCH_FILES = (COMPRESSOR_V2, MEM_POOL, POOL_CFG, DSV4_BACKEND)

# Package import root: inside the eval container this resolves to the mounted
# /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations, so the sglang
# hook's `import mustafar` finds the live host copy with no docker cp.
PACKAGE_ROOT = os.environ.get("XKV_PACKAGE_DIR",
                              str(Path(__file__).resolve().parent.parent))
MARKER = "## MUSTAFAR"


def ctrl_dir() -> str:
    """Directory for runtime debug logs (XKV_DEBUG=1)."""
    return os.environ.get("SG_CTRL_DIR", str(Path(__file__).resolve().parent / "ctrl"))


def topmag_enabled() -> bool:
    return os.environ.get("SGLANG_OPT_TOPMAG") == "1"


def packed_enabled() -> bool:
    return os.environ.get("SGLANG_OPT_TOPMAG_PACKED_C4") == "1"


def validate_packed_config() -> None:
    """Reject layouts that would violate the fixed 256-value ABI."""
    if packed_enabled() and topmag_keep() != 0.5:
        raise ValueError(
            "SGLANG_OPT_TOPMAG_PACKED_C4 requires XKV_TOPMAG_KEEP=0.5"
        )


def topmag_keep() -> float:
    return float(os.environ.get("XKV_TOPMAG_KEEP", "1.0"))


def topmag_zero_count() -> int:
    """Coords zeroed per latent row (0 when keep>=1)."""
    k = HEAD_DIM - int(round(HEAD_DIM * topmag_keep()))
    return max(0, min(k, HEAD_DIM))
