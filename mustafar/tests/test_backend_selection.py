"""CPU checks that paired tests cannot inherit the fused serving dispatcher.

Both kernel suites must pin the packed flags for the whole call and restore
``os.environ`` byte-exactly on success and on failure. They are also required to
refuse a CUDA-less host -- checked before any allocation, by calling
``torch.cuda.is_available()`` and raising ``RuntimeError`` matching
``"requires CUDA"``.
"""

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
            ("validity", "run_validity"),
            ("validity", "run_sparse_t4"),
            ("speed", "run_speed"),
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

    def test_validity_restores_flags_after_success(self):
        from mustafar.tests import validity

        # An empty workload list exercises the full entry/exit path -- flag
        # pinning, the CUDA check, and the summary -- with no kernels launched.
        with patch.dict(os.environ, SGLANG_OPT_TOPMAG_FUSED="1"):
            before = dict(os.environ)
            with (
                patch.object(validity.torch.cuda, "is_available", return_value=True),
                patch.object(validity.torch.cuda, "get_device_name", return_value="mock"),
                patch.object(validity.harness, "WORKLOADS", ()),
                patch.object(validity, "_fused_available", return_value=False),
                patch("builtins.print"),
            ):
                validity.run_validity()
            self.assertEqual(dict(os.environ), before)

    def test_sparse_t4_restores_flags_after_success(self):
        from mustafar.tests import validity

        # As above, an empty workload list drives the whole entry/exit path --
        # both guards, the flag pin, the summary -- with no kernel launched. The
        # outer SPARSE=0 is what the pin has to overwrite and restore.
        with patch.dict(os.environ, SGLANG_OPT_TOPMAG_SPARSE="0"):
            before = dict(os.environ)
            with (
                patch.object(validity.torch.cuda, "is_available", return_value=True),
                patch(
                    "mustafar.sparse.sparse_available", return_value=True
                ),
                patch.object(validity.harness, "WORKLOADS", ()),
                patch("builtins.print"),
            ):
                validity.run_sparse_t4()
            self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
