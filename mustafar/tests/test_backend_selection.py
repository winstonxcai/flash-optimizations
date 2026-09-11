"""CPU checks that the comparison entrypoints cannot inherit ambient flags.

Every ``run_*()`` must pin the flags for the whole call and restore
``os.environ`` byte-exactly on success and on failure. ``validity`` pins the
all-flags-off native base -- ``native`` is *defined* as stock SGLang with every
``SGLANG_OPT_TOPMAG*`` switch off -- and turns each leg's flags back on inside its
own leg block; ``speed`` pins the packed base directly.

Both are also required to refuse a CUDA-less host, checked before any allocation,
by calling ``torch.cuda.is_available()`` and raising ``RuntimeError`` matching
``"requires CUDA"``.
"""

import importlib
import importlib.util
import os
import unittest
from unittest.mock import patch

from mustafar import config

# entrypoint -> the flags it must have pinned at the CUDA guard, overriding the
# hostile ambient environment the test installs. Keys absent from a mapping are
# not asserted: ``speed`` does not pin the sparse gate, and asserting it would be
# asserting the ambient value rather than the entrypoint's own pin.
ENTRYPOINTS = (
    (
        "validity",
        "run_validity",
        {"topmag": False, "packed": False, "fused": False, "sparse": False},
    ),
    ("speed", "run_speed", {"topmag": True, "packed": True, "fused": False}),
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU PyTorch")
class BackendSelectionTests(unittest.TestCase):
    def test_comparison_entrypoints_pin_flags_and_restore_on_failure(self):
        for module_name, entrypoint, expected in ENTRYPOINTS:
            module = importlib.import_module(f"mustafar.tests.{module_name}")

            def check_flags(expected=expected):
                self.assertEqual(config.topmag_enabled(), expected["topmag"])
                self.assertEqual(config.packed_enabled(), expected["packed"])
                self.assertEqual(config.fused_enabled(), expected["fused"])
                if "sparse" in expected:
                    self.assertEqual(config.sparse_enabled(), expected["sparse"])
                return False  # Stop before any GPU allocation.

            with (
                self.subTest(module=module_name),
                patch.dict(
                    os.environ,
                    SGLANG_OPT_TOPMAG_FUSED="1",
                    SGLANG_OPT_TOPMAG_PACKED="0",
                    SGLANG_OPT_TOPMAG="0",
                    SGLANG_OPT_TOPMAG_SPARSE="0",
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

    def test_leg_env_pins_are_legal(self):
        from mustafar.tests import validity

        for leg in validity.LEGS:
            with self.subTest(leg=leg), validity._leg_env(leg):
                # _leg_env runs validate_packed_static_config on entry, so reaching
                # here already means the pinned set is a legal configuration.
                self.assertTrue(config.topmag_enabled())
                self.assertTrue(config.packed_enabled())
                self.assertEqual(config.topmag_keep(), 0.5)

    def test_no_leg_pins_both_c4_gates(self):
        """The fused and sparse switches rewrite the same decode call site."""
        from mustafar.tests import validity

        for leg in ("fused", "sparse"):
            with self.subTest(leg=leg), validity._leg_env(leg):
                self.assertNotEqual(
                    config.fused_enabled(), config.sparse_enabled()
                )

    def test_validity_restores_flags_after_success(self):
        from mustafar.tests import validity

        # An empty workload list exercises the full entry/exit path -- flag
        # pinning, the CUDA check, leg discovery, and the summary -- with no
        # kernels launched.
        with patch.dict(os.environ, SGLANG_OPT_TOPMAG_FUSED="1"):
            before = dict(os.environ)
            with (
                patch.object(validity.torch.cuda, "is_available", return_value=True),
                patch.object(
                    validity.torch.cuda, "get_device_name", return_value="mock"
                ),
                patch.object(validity.harness, "WORKLOADS", ()),
                patch.object(validity, "_fused_available", return_value=False),
                patch.object(validity, "_sparse_available", return_value=False),
                patch("builtins.print"),
            ):
                validity.run_validity()
            self.assertEqual(dict(os.environ), before)

    def test_validity_sparse_leg_restores_flags_after_success(self):
        from mustafar.tests import validity

        # Selecting the sparse leg replaces the old separate entrypoint. The
        # outer SPARSE=0 is what the leg's own pin has to overwrite and restore.
        with patch.dict(os.environ, SGLANG_OPT_TOPMAG_SPARSE="0"):
            before = dict(os.environ)
            with (
                patch.object(validity.torch.cuda, "is_available", return_value=True),
                patch.object(
                    validity.torch.cuda, "get_device_name", return_value="mock"
                ),
                patch("mustafar.sparse.sparse_available", return_value=True),
                patch.object(validity.harness, "WORKLOADS", ()),
                patch("builtins.print"),
            ):
                validity.run_validity(legs=("sparse",))
            self.assertEqual(dict(os.environ), before)

    def test_packed_legs_do_not_depend_on_the_extensions(self):
        """The extension-free fallback has to stay reachable with both absent.

        ``packed`` is the leg that can never regress and the leg that localises a
        failure; if it quietly became conditional on a CUDA extension being built,
        neither property would hold.
        """
        from mustafar.tests import validity

        with (
            patch.object(validity, "_fused_available", return_value=False),
            patch.object(validity, "_sparse_available", return_value=False),
        ):
            self.assertTrue(validity._leg_available("packed.bf16"))
            self.assertTrue(validity._leg_available("packed.native"))
            self.assertFalse(validity._leg_available("fused"))
            self.assertFalse(validity._leg_available("sparse"))

    def test_validity_refuses_an_unbuilt_leg(self):
        from mustafar.tests import validity

        with (
            patch.object(validity.torch.cuda, "is_available", return_value=True),
            patch.object(validity, "_fused_available", return_value=False),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not built"):
                validity.run_validity(legs=("fused",))

    def test_validity_rejects_an_unknown_leg(self):
        from mustafar.tests import validity

        with (
            patch.object(validity.torch.cuda, "is_available", return_value=True),
            patch.object(validity, "_fused_available", return_value=False),
            patch("builtins.print"),
        ):
            with self.assertRaisesRegex(ValueError, "unknown legs"):
                validity.run_validity(legs=("triton-dense",))


if __name__ == "__main__":
    unittest.main()
