"""TopMag and persistent packed C4 configuration (Mustafar).

Dense mode preserves the native 584-byte DeepSeek-V4 store. Packed mode uses
the 328-byte FP8/bitmap/scale ABI and reconstructs the native attention input.

Env switches are read at import time (the serving process sets them first).
"""
import os
from pathlib import Path

# --- c4 latent geometry (fixed by DeepSeek-V4) --------------------------------
HEAD_DIM = 512          # full c4 compressor latent dim
ROPE_DIM = 64           # rotary tail dims
NOPE_DIM = HEAD_DIM - ROPE_DIM          # 448
TILE_SIZE = 64          # native store's fp8 scale tile
BITMAP_WORDS = HEAD_DIM // 64           # 8  (uint64 words per packed row)
PACKED_KEEP = HEAD_DIM // 2              # exact TopMag50 survivors
PACKED_VALUE_BYTES = PACKED_KEEP         # raw FP8 E4M3 codes
PACKED_BITMAP_BYTES = BITMAP_WORDS * 8
PACKED_SCALE_BYTES = HEAD_DIM // TILE_SIZE
PACKED_C4_BYTES = (
    PACKED_VALUE_BYTES + PACKED_BITMAP_BYTES + PACKED_SCALE_BYTES
)                                           # 328 B
NATIVE_C4_BYTES = 584
NATIVE_VALUE_BYTES = 576
FP8_E4M3_MAX = 448.0

# --- env switches ---------------------------------------------------------------
# Gate: SGLANG_OPT_TOPMAG=1 enables the prune hook. Keep fraction:
# XKV_TOPMAG_KEEP=0.5 keeps the largest 50% of coords per vector (zeros the rest).
TOPMAG_KEEP = float(os.environ.get("XKV_TOPMAG_KEEP", "1.0"))

# --- sglang patch targets -------------------------------------------------------
# Stage 1 patches the store, persistent pool, capacity model, raw-index output,
# and the two unchanged-attention reconstruction call sites.
SRC_ROOT = os.environ.get("SG_LOWRANK_SRC", "/sgl-workspace/sglang-lowrank/python")
COMPRESSOR_V2 = f"{SRC_ROOT}/sglang/srt/layers/attention/dsv4/compressor_v2.py"
MEM_POOL = f"{SRC_ROOT}/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
POOL_CFG = f"{SRC_ROOT}/sglang/srt/model_executor/pool_configurator.py"
DSV4_BACKEND = f"{SRC_ROOT}/sglang/srt/layers/attention/deepseek_v4_backend.py"
INDEXER = f"{SRC_ROOT}/sglang/srt/layers/attention/dsv4/indexer.py"
PATCH_FILES = (COMPRESSOR_V2, MEM_POOL, POOL_CFG, DSV4_BACKEND, INDEXER)
OLD_PATCH_FILES = PATCH_FILES

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


def topmag_keep() -> float:
    return float(os.environ.get("XKV_TOPMAG_KEEP", "1.0"))


def topmag_zero_count() -> int:
    """Coords zeroed per latent row (0 when keep>=1)."""
    k = HEAD_DIM - int(round(HEAD_DIM * topmag_keep()))
    return max(0, min(k, HEAD_DIM))


def sparse_shadow() -> bool:
    """Stage-0 shadow check: XKV_SPARSE_SHADOW=1 runs pack->unpack on real latent
    rows at the store site and compares against the dense-pruned tensor (default
    off -> the server performs no sparse pack/unpack work)."""
    return os.environ.get("XKV_SPARSE_SHADOW") == "1"


def packed_c4_enabled() -> bool:
    """Persistent 328-byte C4 pool gate (off by default)."""
    return os.environ.get("SGLANG_OPT_TOPMAG_PACKED_C4") == "1"


def validate_packed_static_config() -> None:
    """Fail early for settings that would change the Stage-1 ABI."""
    if not packed_c4_enabled():
        return
    if not topmag_enabled():
        raise RuntimeError(
            "SGLANG_OPT_TOPMAG_PACKED_C4=1 requires SGLANG_OPT_TOPMAG=1"
        )
    if topmag_keep() != 0.5:
        raise RuntimeError(
            "packed C4 requires XKV_TOPMAG_KEEP=0.5 (exactly 256/512 dims)"
        )
