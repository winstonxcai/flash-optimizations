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
from unittest.mock import Mock, patch

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

    @unittest.skipIf(torch is None, "requires PyTorch")
    def test_optimized_dispatch_promotes_geometry_for_active_shape(self):
        from .. import fused

        extension = type(
            "Extension",
            (),
            {
                "packed_to_native": staticmethod(Mock()),
                "packed_to_native_optimized": staticmethod(Mock()),
                "packed_to_native_geometry": staticmethod(Mock()),
            },
        )()
        values = torch.zeros(256, dtype=torch.uint8)
        bitmaps = torch.zeros(8, dtype=torch.uint64)
        scales = torch.zeros(8, dtype=torch.uint8)
        indices = torch.zeros((1, 512), dtype=torch.int64)
        lengths = torch.zeros(1, dtype=torch.int64)
        frequencies = torch.zeros((1, 32, 2), dtype=torch.float32)
        native = torch.zeros(32 * 16 * 584, dtype=torch.uint8)

        fused._validated_devices.clear()
        fused._markers_emitted.clear()
        with (
            patch.object(fused.torch.cuda, "is_available", return_value=True),
            patch.object(fused.torch.cuda, "current_device", return_value=0),
            patch.object(fused.torch.cuda, "get_device_capability", return_value=(9, 0)),
            patch.object(fused, "_load", return_value=extension),
            patch("builtins.print"),
        ):
            fused.packed_to_native(
                values,
                bitmaps,
                scales,
                indices,
                indices.clone(),
                lengths,
                frequencies,
                native,
                16,
                16 * 584,
                optimized=True,
            )

        extension.packed_to_native_geometry.assert_called_once()
        extension.packed_to_native_optimized.assert_not_called()


if __name__ == "__main__":
    unittest.main()
