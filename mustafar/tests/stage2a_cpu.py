"""CPU-only ABI and E4M3 decode gate for the Stage-2A image build."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .. import config


def decode_e4m3fn(code: int) -> float:
    sign = code >> 7
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        value = math.ldexp(float(mantissa), -9)
    elif exponent == 15 and mantissa == 7:
        return math.nan
    else:
        value = math.ldexp(float(8 + mantissa), exponent - 10)
    return -value if sign else value


def run() -> None:
    assert config.PACKED_C4_BYTES == 328
    assert config.NATIVE_C4_BYTES == 584
    assert config.NOPE_DIM == 448 and config.ROPE_DIM == 64
    assert config.PACKED_KEEP == 256 and config.BITMAP_WORDS == 8

    codes = torch.arange(256, dtype=torch.uint8)
    torch_values = codes.view(torch.float8_e4m3fn).float()
    for code, expected in enumerate(torch_values.tolist()):
        actual = decode_e4m3fn(code)
        if math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == expected, (code, actual, expected)

    extension_candidates = list(
        Path(__file__).resolve().parents[1].glob("_stage2a_cuda*.so")
    )
    if not extension_candidates:
        raise AssertionError("Stage-2A CUDA extension was not built in-place")
    print("[stage2a-cpu] OK: ABI constants, all E4M3FN codes, extension artifact")


if __name__ == "__main__":
    run()
