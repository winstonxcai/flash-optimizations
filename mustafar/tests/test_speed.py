"""CPU checks that the speed suite's table is the table it runs.

:mod:`mustafar.tests.speed` declares its stages and legs as data (``MATRIX``,
``CONTRASTS``) and derives the timed callables from that declaration, so the
declaration is what a reader trusts and therefore what these checks hold: a stage
with no bar, a leg listed both present and absent, or a contrast between legs its
own stage does not time would each make the printed table mean something other
than it says. None of that needs a device -- whether the matrix is right is
decided before any kernel runs.
"""

import unittest

try:
    import torch
except ImportError:
    torch = None

from mustafar import config

from . import harness, speed

# The stages both suites name. Spelled out here rather than compared against
# speed.STAGES, which is derived from MATRIX and would make the check a tautology.
EXPECTED_STAGES = ("store", "rows.dense_bf16", "rows.native_layout", "attention")


class MatrixTests(unittest.TestCase):
    def test_the_stages_are_the_shared_vocabulary(self):
        self.assertEqual(speed.STAGES, EXPECTED_STAGES)

    def test_pruning_is_not_a_timed_stage(self):
        """It is a correctness stage: no operator of its own to time."""
        self.assertNotIn("pruning", speed.STAGES)

    def test_every_named_leg_is_a_column(self):
        for row in speed.MATRIX:
            for leg, _ in row.present + row.absent:
                self.assertIn(leg, harness.COLUMNS, row.stage)

    def test_every_stage_accounts_for_every_column_exactly_once(self):
        """No column may go unmentioned: a leg that is neither present nor absent
        is one a reader has to guess about."""
        for row in speed.MATRIX:
            with self.subTest(stage=row.stage):
                present = {leg for leg, _ in row.present}
                absent = {leg for leg, _ in row.absent}
                self.assertEqual(present & absent, set())
                self.assertEqual(present | absent, set(harness.COLUMNS))

    def test_every_cell_carries_its_note_or_its_reason(self):
        for row in speed.MATRIX:
            with self.subTest(stage=row.stage):
                self.assertTrue(row.note.strip())
                for leg, note in row.present:
                    self.assertTrue(note.strip(), f"{row.stage}/{leg}")
                for leg, reason in row.absent:
                    self.assertTrue(reason.strip(), f"{row.stage}/{leg}")

    def test_every_bar_is_a_leg_its_own_stage_times(self):
        for row in speed.MATRIX:
            with self.subTest(stage=row.stage):
                self.assertIn(row.bar, {leg for leg, _ in row.present})

    def test_the_bar_is_native_wherever_native_has_an_operator(self):
        for row in speed.MATRIX:
            present = {leg for leg, _ in row.present}
            if harness.NATIVE in present:
                self.assertEqual(row.bar, harness.NATIVE, row.stage)

    def test_the_layout_row_holds_no_native_bar(self):
        """What that row rests on: native hands its 584-byte buffer to attention
        unchanged, so there is no operator to time and no native figure to invent.

        The bar is the Triton reconstruct the fused kernel has to beat instead.
        """
        row = next(row for row in speed.MATRIX if row.stage == "rows.native_layout")
        self.assertNotIn(harness.NATIVE, {leg for leg, _ in row.present})
        self.assertIn(harness.NATIVE, dict(row.absent))
        self.assertEqual(row.bar, "packed.native")

    def test_the_attention_row_times_every_leg(self):
        row = next(row for row in speed.MATRIX if row.stage == "attention")
        self.assertEqual({leg for leg, _ in row.present}, set(harness.COLUMNS))
        self.assertEqual(row.absent, ())


class ContrastTests(unittest.TestCase):
    def test_every_contrast_resolves_to_legs_its_stage_times(self):
        for contrast in speed.CONTRASTS:
            with self.subTest(contrast=contrast.label):
                rows = [row for row in speed.MATRIX if row.stage == contrast.stage]
                self.assertEqual(len(rows), 1, f"no single stage {contrast.stage}")
                present = {leg for leg, _ in rows[0].present}
                self.assertIn(contrast.numerator, present)
                self.assertIn(contrast.denominator, present)
                self.assertNotEqual(contrast.numerator, contrast.denominator)

    def test_every_contrast_says_what_it_is_for(self):
        for contrast in speed.CONTRASTS:
            self.assertTrue(contrast.reason.strip(), contrast.label)

    def test_only_the_fused_kernel_is_gated(self):
        """``sparse_over_packed.bf16`` is reported, never asserted.

        V1 runs its softmax in Python between two kernel launches, so it loses to
        the reassembling path; the measured ratio is a finding about v1, and
        asserting the other direction would assert something false.
        """
        gated = {contrast.label for contrast in speed.CONTRASTS if contrast.gate}
        self.assertEqual(gated, {"fused_over_packed.native"})


class LegEnvTests(unittest.TestCase):
    """The vocabulary a leg name resolves through, shared with validity."""

    def test_every_column_has_a_pin(self):
        self.assertEqual(set(harness.LEG_ENV), set(harness.COLUMNS))

    def test_the_bar_is_the_all_off_base(self):
        self.assertIs(harness.LEG_ENV[harness.NATIVE], harness.OFF_ENV)
        self.assertTrue(all(value == "0" for value in harness.OFF_ENV.values()))

    def test_off_env_and_the_packed_base_toggle_the_same_gates(self):
        """A flag spelled in one and not the other would leave ``native`` at the
        ambient value of a gate the packed legs turn on."""
        self.assertEqual(set(harness._PACKED_ENV) - {"KEEP"}, set(harness.OFF_ENV))

    def test_the_bar_is_never_a_candidate(self):
        self.assertNotIn(harness.NATIVE, harness.available_legs())
        self.assertTrue(harness.leg_available(harness.NATIVE))


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class CellTests(unittest.TestCase):
    """The builders cover exactly the legs ``MATRIX`` declares, with no device."""

    def _ctx(self) -> speed._Ctx:
        """A ctx something like the one ``_prepare`` builds, without a device.

        Building a callable reads the case and the workspace slots, and nothing
        else: every tensor is read when the callable *runs*, which this tier never
        does. The three row buffers are still real CPU tensors so the fake has the
        shape ``_prepare`` allocates; the 328-byte records and the native cache are
        the parts that need a GPU, so they stay ``None``, and a builder that
        started reading one at build time would be reading a device tensor outside
        the preallocation contract.
        """
        case = harness.build_case(harness.WORKLOADS[0], "cpu")
        gathered = case.workload.gather_rows

        def rows() -> torch.Tensor:
            return torch.empty(gathered, 1, config.HEAD_DIM, dtype=torch.bfloat16)

        return speed._Ctx(
            case=case,
            buffers=None,
            native_cache=None,
            native_locations=None,
            native_rows=rows(),
            bf16_rows=rows(),
            dense_rows=rows(),
            workspaces={"packed.native": None, "fused": None},
            workspace_locations={"packed.native": None, "fused": None},
            q=None,
            indices=None,
        )

    def _ops(self) -> speed._Ops:
        """Operators that fail if called, which is the build-time contract."""

        def unused(*args, **kwargs):
            raise AssertionError("a builder invoked an operator; it must only close over it")

        return speed._Ops(
            store=unused,
            dequant=unused,
            pack_rows=unused,
            gather_bf16=unused,
            unpack_native=unused,
            c4_leg=unused,
        )

    def _declared(self) -> set[tuple[str, str]]:
        return {(row.stage, leg) for row in speed.MATRIX for leg, _ in row.present}

    def test_cells_cover_exactly_the_declared_legs(self):
        cells = speed._cells(self._ctx(), self._ops(), harness.COLUMNS)
        speed._check_cells(cells, harness.COLUMNS)
        built = {
            (stage, leg) for stage, stage_cells in cells.items() for leg in stage_cells
        }
        self.assertEqual(built, self._declared())

    def test_a_leg_subset_narrows_the_matrix_without_adding_cells(self):
        selected = (harness.NATIVE, "packed.bf16")
        cells = speed._cells(self._ctx(), self._ops(), selected)
        speed._check_cells(cells, selected)
        self.assertEqual(set(cells), set(speed.STAGES))
        for stage, stage_cells in cells.items():
            self.assertTrue(set(stage_cells) <= set(selected), stage)

    def test_the_layout_stage_times_no_native_cell(self):
        cells = speed._cells(self._ctx(), self._ops(), harness.COLUMNS)
        self.assertNotIn(harness.NATIVE, cells["rows.native_layout"])

    def test_the_cross_check_fires_on_a_cell_the_matrix_does_not_declare(self):
        cells = speed._cells(self._ctx(), self._ops(), harness.COLUMNS)
        cells["store"]["sparse"] = lambda: None  # declared absent there
        with self.assertRaises(AssertionError):
            speed._check_cells(cells, harness.COLUMNS)

    def test_the_cross_check_fires_on_a_declared_cell_that_is_missing(self):
        cells = speed._cells(self._ctx(), self._ops(), harness.COLUMNS)
        del cells["attention"]["fused"]
        with self.assertRaises(AssertionError):
            speed._check_cells(cells, harness.COLUMNS)


if __name__ == "__main__":
    unittest.main()
