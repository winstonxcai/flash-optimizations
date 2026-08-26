"""Configuration and record-layout constants for the W3 CSA low-rank store."""
import os
from pathlib import Path

COEFF_DIM = int(os.environ.get("XKV_COEFF_DIM", "192"))
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
TILE_SIZE = 64
SCALE_TILES = COEFF_DIM // TILE_SIZE
POS_BYTES = 4
PAD_BYTES = (4 - (COEFF_DIM + SCALE_TILES) % 4) % 4
COEFF_SCALE_BYTES = COEFF_DIM + SCALE_TILES + PAD_BYTES
BYTES_PER_TOKEN = COEFF_SCALE_BYTES + POS_BYTES

BLOCK_M = 32
NUM_WARPS = 8
NUM_STAGES = 1
BLOCK_NOPE = 512
BLOCK_NOPE_H = 256
NUM_HALVES = BLOCK_NOPE // BLOCK_NOPE_H

SRC_ROOT = os.environ.get("SG_LOWRANK_SRC", "/sgl-workspace/sglang/python")
PACKAGE_ROOT = os.environ.get("XKV_PACKAGE_DIR", str(Path(__file__).resolve().parent.parent))
COMPRESSOR_V2 = f"{SRC_ROOT}/sglang/srt/layers/attention/dsv4/compressor_v2.py"
MEM_POOL = f"{SRC_ROOT}/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
POOL_CFG = f"{SRC_ROOT}/sglang/srt/model_executor/pool_configurator.py"
DSV4_BACKEND = f"{SRC_ROOT}/sglang/srt/layers/attention/deepseek_v4_backend.py"
MARKER = "## XKV_LOWRANK"


def ctrl_dir() -> str:
    return os.environ.get("SG_CTRL_DIR", str(Path(__file__).resolve().parent / "ctrl"))


def lowrank_enabled() -> bool:
    return os.environ.get("SGLANG_OPT_LOWRANK_KV_STORE") == "1"


def basis_dir() -> str:
    return os.environ.get("SG_LOWRANK_BASIS", os.path.join(ctrl_dir(), "basis"))
