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
FP8_E4M3_MAX = 448.0

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

# Package import root: inside the eval container this resolves to the mounted
# /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations, so the sglang
# hook's `import mustafar` finds the live host copy with no docker cp.
PACKAGE_ROOT = os.environ.get("XKV_PACKAGE_DIR",
                              str(Path(__file__).resolve().parent.parent))
MARKER = "## MUSTAFAR"


def topmag_enabled() -> bool:
    return os.environ.get("SGLANG_OPT_TOPMAG") == "1"


def topmag_keep() -> float:
    return float(os.environ.get("XKV_TOPMAG_KEEP", "1.0"))


def packed_c4_enabled() -> bool:
    """Persistent 328-byte C4 pool gate (off by default)."""
    return os.environ.get("SGLANG_OPT_TOPMAG_PACKED_C4") == "1"


def stage2a_enabled() -> bool:
    """Single-kernel packed-to-native reconstruction gate (off by default)."""
    return os.environ.get("SGLANG_OPT_TOPMAG_STAGE2A") == "1"


def validate_packed_static_config() -> None:
    """Fail early for settings that would change the Stage-1 ABI."""
    if stage2a_enabled() and not packed_c4_enabled():
        raise RuntimeError(
            "SGLANG_OPT_TOPMAG_STAGE2A=1 requires "
            "SGLANG_OPT_TOPMAG_PACKED_C4=1"
        )
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
