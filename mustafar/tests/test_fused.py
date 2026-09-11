"""CPU-only ABI and E4M3 decode gate for the Fused image build.

Nothing here needs a device: the ABI constants are pure config, and
``decode_e4m3fn`` is the Python mirror of the kernel's decode, cross-checked
against ``torch``'s own ``float8_e4m3fn``.

``test_extension_built_in_place`` is deliberately strict rather than skipped.
``docker/modal.Dockerfile`` builds the extension on the line above running this
module, so a missing ``_fused*.so`` must fail the image build, not pass it
by skipping. Running this module off-image without a build will fail for the
same reason -- that is the check doing its job.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from .. import config

_EXTENSION_GLOB = "_fused*.so"


def decode_e4m3fn(code: int) -> float:
    """Python mirror of the kernel's E4M3 decode, for all 256 codes."""
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


class FusedCpuTests(unittest.TestCase):
    def test_abi_constants(self):
        self.assertEqual(config.PACKED_RECORD_BYTES, 328)
        self.assertEqual(config.NATIVE_RECORD_BYTES, 584)
        self.assertEqual(config.NOPE_DIM, 448)
        self.assertEqual(config.ROPE_DIM, 64)
        self.assertEqual(config.PACKED_KEPT_VALUES, 256)
        self.assertEqual(config.BITMAP_WORDS, 8)

    @unittest.skipIf(torch is None, "requires PyTorch")
    def test_e4m3_decode_matches_torch(self):
        codes = torch.arange(256, dtype=torch.uint8)
        torch_values = codes.view(torch.float8_e4m3fn).float()
        for code, expected in enumerate(torch_values.tolist()):
            actual = decode_e4m3fn(code)
            if math.isnan(expected):
                self.assertTrue(math.isnan(actual), f"code {code}: {actual}")
            else:
                self.assertEqual(actual, expected, f"code {code}")

    def test_extension_built_in_place(self):
        candidates = list(
            Path(__file__).resolve().parents[1].glob(_EXTENSION_GLOB)
        )
        self.assertTrue(
            candidates, f"Fused CUDA extension was not built in-place ({_EXTENSION_GLOB})"
        )


if __name__ == "__main__":
    unittest.main()
