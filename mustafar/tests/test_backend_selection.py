"""CPU checks that the comparison entrypoints cannot inherit ambient flags.

Both ``run_*()`` pin the flags for the whole call and restore ``os.environ``
byte-exactly on success and on failure. Both pin the same all-flags-off native
base -- ``native`` is *defined* as stock SGLang with every ``SGLANG_OPT_TOPMAG*``
switch off -- and turn each leg's flags back on inside its own leg block, so a
leg is never whatever the ambient environment happened to have.

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

from . import harness

# The base both entrypoints pin: every gate off. Spelled once, because the two
# entrypoints pinning different bases is exactly the drift this file exists to
# catch -- a leg timed under an inherited flag would be a different leg.
ALL_OFF = {"topmag": False, "packed": False, "fused": False, "sparse": False}

# entrypoint -> the flags it must have pinned at the CUDA guard, overriding the
# hostile ambient environment the test installs.
ENTRYPOINTS = (("validity", "run_validity"), ("speed", "run_speed"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires CPU PyTorch")
class BackendSelectionTests(unittest.TestCase):
    def test_comparison_entrypoints_pin_flags_and_restore_on_failure(self):
        for module_name, entrypoint in ENTRYPOINTS:
            module = importlib.import_module(f"mustafar.tests.{module_name}")

            def check_flags():
                self.assertEqual(config.topmag_enabled(), ALL_OFF["topmag"])
                self.assertEqual(config.packed_enabled(), ALL_OFF["packed"])
                self.assertEqual(config.fused_enabled(), ALL_OFF["fused"])
                self.assertEqual(config.sparse_enabled(), ALL_OFF["sparse"])
                return False  # Stop before any GPU allocation.

            with (
                self.subTest(module=module_name),
                patch.dict(
                    os.environ,
                    SGLANG_OPT_TOPMAG_FUSED="1",
                    SGLANG_OPT_TOPMAG_PACKED="0",
                    SGLANG_OPT_TOPMAG="0",
                    SGLANG_OPT_TOPMAG_SPARSE="1",
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

    def test_the_native_pin_turns_every_gate_off(self):
        """``native`` is not "whatever was set": it is every switch off, pinned."""
        with patch.dict(os.environ, SGLANG_OPT_TOPMAG_FUSED="1", KEEP="1.0"):
            with harness.leg_env(harness.NATIVE):
                self.assertFalse(config.topmag_enabled())
                self.assertFalse(config.packed_enabled())
                self.assertFalse(config.fused_enabled())
                self.assertFalse(config.sparse_enabled())

    def test_candidate_leg_pins_are_legal(self):
        for leg in harness.LEGS:
            with self.subTest(leg=leg), harness.leg_env(leg):
                # leg_env runs validate_packed_static_config on entry, so reaching
                # here already means the pinned set is a legal configuration.
                self.assertTrue(config.topmag_enabled())
                self.assertTrue(config.packed_enabled())
                self.assertEqual(config.topmag_keep(), 0.5)

    def test_no_leg_pins_both_c4_gates(self):
        """The fused and sparse switches rewrite the same decode call site."""
        for leg in ("fused", "sparse"):
            with self.subTest(leg=leg), harness.leg_env(leg):
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
                patch.object(harness, "_fused_available", return_value=False),
                patch.object(harness, "_sparse_available", return_value=False),
                patch("builtins.print"),
            ):
                validity.run_validity()
            self.assertEqual(dict(os.environ), before)

    def test_speed_restores_flags_after_success(self):
        from mustafar.tests import speed

        # Same contract as validity's, over an empty grid: the workload loop is
        # what applies a leg's own pin, so with no workloads this covers the
        # base pin's exit path and the summary. Operators are stubbed because
        # resolving them imports SGLang, which is the one thing absent here.
        with patch.dict(os.environ, SGLANG_OPT_TOPMAG_FUSED="1"):
            before = dict(os.environ)
            with (
                patch.object(speed.torch.cuda, "is_available", return_value=True),
                patch.object(
                    speed.torch.cuda, "get_device_name", return_value="mock"
                ),
                patch.object(speed.harness, "WORKLOADS", ()),
                patch.object(speed, "_resolve_ops"),
                patch("builtins.print"),
            ):
                speed.run_speed()
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
        with (
            patch.object(harness, "_fused_available", return_value=False),
            patch.object(harness, "_sparse_available", return_value=False),
        ):
            self.assertTrue(harness.leg_available("packed.bf16"))
            self.assertTrue(harness.leg_available("packed.native"))
            self.assertFalse(harness.leg_available("fused"))
            self.assertFalse(harness.leg_available("sparse"))

    def test_leg_selection_rejects_an_unknown_leg(self):
        with patch.object(harness, "_fused_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "unknown legs"):
                harness.select_legs(("triton-dense",))

    def test_leg_selection_rejects_an_unbuilt_leg(self):
        with patch.object(harness, "_fused_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not built"):
                harness.select_legs(("fused",))

    def test_leg_selection_defaults_to_every_available_candidate(self):
        with (
            patch.object(harness, "_fused_available", return_value=False),
            patch.object(harness, "_sparse_available", return_value=False),
        ):
            self.assertEqual(
                harness.select_legs(None), ("packed.bf16", "packed.native")
            )

    def test_leg_selection_returns_candidates_in_report_order(self):
        """Request order is not report order, and the bar is never in the result."""
        with (
            patch.object(harness, "_fused_available", return_value=True),
            patch.object(harness, "_sparse_available", return_value=True),
        ):
            self.assertEqual(
                harness.select_legs(("sparse", "packed.bf16")),
                ("packed.bf16", "sparse"),
            )
            self.assertEqual(harness.select_legs(harness.LEGS), harness.LEGS)

    def test_the_bar_is_not_selectable(self):
        """It is always run, so ``--legs native`` is a usage error, not an unknown
        name -- and the message has to say which of the two it is."""
        with self.assertRaisesRegex(ValueError, "cannot be selected"):
            harness.select_legs((harness.NATIVE,))


if __name__ == "__main__":
    unittest.main()
