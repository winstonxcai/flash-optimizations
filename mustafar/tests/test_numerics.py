"""Discoverable numerical suites; GPU checks require their runtime dependencies."""

import importlib.util
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

HAS_CUDA = torch is not None and torch.cuda.is_available()
HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_SGLANG = importlib.util.find_spec("sglang") is not None
HAS_FUSED = importlib.util.find_spec("mustafar._fused_cuda") is not None


@unittest.skipIf(torch is None, "requires PyTorch")
class ReferenceTests(unittest.TestCase):
    def test_topmag(self):
        from .unit import run_topmag

        run_topmag()

    @patch.dict("os.environ", SGLANG_OPT_TOPMAG_FUSED="0")
    def test_packed_reference(self):
        from .unit import run_packed_reference

        run_packed_reference()

    def test_public_names_and_workspace_layout(self):
        import mustafar
        from mustafar import config, packed, patching

        self.assertEqual(mustafar.PACKED_RECORD_BYTES, 328)
        self.assertEqual(config.PACKED_KEPT_VALUES, 256)
        self.assertEqual(config.NATIVE_RECORD_BYTES, 584)
        self.assertEqual(len(mustafar.__all__), len(set(mustafar.__all__)))
        for name, module in mustafar._LAZY_EXPORTS.items():
            self.assertIs(
                getattr(mustafar, name),
                getattr(importlib.import_module(f"mustafar.{module}"), name),
            )
        self.assertIs(mustafar.patch, patching.patch)
        workspace = packed.NativeWorkspace.allocate(2, 4, 64, "cpu", with_dense=True)
        self.assertEqual(workspace.native_bytes.dtype, torch.uint8)
        self.assertEqual(workspace.dense_bf16.dtype, torch.bfloat16)
        self.assertEqual(tuple(workspace.dense_bf16.shape), (8, 512))

    def test_extension_loader_does_not_swallow_interrupts(self):
        from mustafar import fused

        with (
            patch.object(fused, "_extension", None),
            patch.object(fused, "_load_error", None),
        ):
            with (
                patch.object(
                    fused.importlib, "import_module", side_effect=KeyboardInterrupt
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                fused.fused_available()
            self.assertIsNone(fused._load_error)
            with patch.object(
                fused.importlib, "import_module", side_effect=ImportError("absent")
            ):
                self.assertFalse(fused.fused_available())
            self.assertIsInstance(fused._load_error, ImportError)


class KernelTests(unittest.TestCase):
    @unittest.skipUnless(
        HAS_CUDA and HAS_TRITON and HAS_SGLANG, "requires CUDA, Triton, and SGLang"
    )
    def test_packed(self):
        from .gpu_packed import run_packed_validation

        run_packed_validation()

    @unittest.skipUnless(
        HAS_CUDA and HAS_TRITON and HAS_FUSED,
        "requires CUDA, Triton, and the fused extension",
    )
    def test_fused(self):
        from .gpu_fused import run_fused_validation

        run_fused_validation()


if __name__ == "__main__":
    unittest.main()
