"""CPU checks that paired tests cannot inherit the fused serving dispatcher."""

import importlib
import importlib.util
import os
import unittest
from unittest.mock import patch

from mustafar import config


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU PyTorch")
class BackendSelectionTests(unittest.TestCase):
    def test_comparison_entrypoints_pin_flags_and_restore_on_failure(self):
        for module_name, entrypoint in (
            ("bench_fused", "run_fused_benchmark"),
            ("gpu_fused", "run_fused_validation"),
            ("bench_packed", "run_packed_benchmark"),
            ("gpu_packed", "run_packed_validation"),
        ):
            module = importlib.import_module(f"mustafar.tests.{module_name}")

            def check_flags():
                self.assertFalse(config.fused_enabled())
                self.assertTrue(config.packed_enabled())
                self.assertTrue(config.topmag_enabled())
                self.assertEqual(config.topmag_keep(), 0.5)
                return False  # Stop before any GPU allocation.

            with (
                self.subTest(module=module_name),
                patch.dict(
                    os.environ,
                    SGLANG_OPT_TOPMAG_FUSED="1",
                    SGLANG_OPT_TOPMAG_PACKED="0",
                    SGLANG_OPT_TOPMAG="0",
                    KEEP="1.0",
                ),
            ):
                before = dict(os.environ)
                with (
                    patch.object(
                        module.torch.cuda, "is_available", side_effect=check_flags
                    ),
                    self.assertRaisesRegex(RuntimeError, "requires CUDA"),
                ):
                    getattr(module, entrypoint)()
                self.assertEqual(dict(os.environ), before)

    def test_validation_restores_flags_after_success(self):
        from mustafar.tests import gpu_fused

        def check_shape(*_):
            self.assertFalse(config.fused_enabled())
            return {"mocked_cpu_check": True}

        with patch.dict(os.environ, SGLANG_OPT_TOPMAG_FUSED="1"):
            before = dict(os.environ)
            with (
                patch.object(gpu_fused.torch.cuda, "is_available", return_value=True),
                patch.object(
                    gpu_fused.torch.cuda, "get_device_name", return_value="mock"
                ),
                patch.object(
                    gpu_fused.torch.cuda, "get_device_capability", return_value=(0, 0)
                ),
                patch.object(gpu_fused, "_run_shape", side_effect=check_shape) as shape,
                patch("builtins.print"),
            ):
                gpu_fused.run_fused_validation()
                self.assertEqual(shape.call_count, 5)
            self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
