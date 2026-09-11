"""CPU checks that the case grid perturbs what it claims to perturb.

The grid's whole value is that it stops ``physical``/``raw``/``lengths`` and the
keep-mask from coinciding with the shapes a kernel can trivially get right. A
typo in a pattern silently reverts it to ``identity`` and the coverage becomes
imaginary, with nothing failing to say so. These checks are the ones that fail
when that happens, and they need no device.
"""

import unittest

try:
    import torch
except ImportError:
    torch = None

from mustafar import config

from . import harness

# The mask patterns re-select *which* 256 coordinates survive; the index patterns
# reshape the gather. Both sets must leave the two invariants alone.
INDEX_PATTERNS = sorted(harness.INDEX_PATTERNS)
MASK_PATTERNS = sorted(harness.MASK_PATTERNS)


def _slots(batch: int) -> torch.Tensor:
    """``(batch, TOPK)`` slot index within each row."""
    return torch.arange(harness.TOPK).view(1, -1).expand(batch, -1)


def _live(case: harness.Case) -> torch.Tensor:
    """``(batch, TOPK)`` slots the ABI calls present."""
    return (case.physical >= 0) & (_slots(case.batch) < case.lengths.view(-1, 1))


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class GridTests(unittest.TestCase):
    def test_every_workload_appears_once_at_identity(self):
        grid = harness.case_grid(harness.WORKLOADS)
        for workload in harness.WORKLOADS:
            self.assertEqual(grid.count((workload, harness.IDENTITY)), 1)

    def test_every_adversarial_pattern_runs_exactly_once(self):
        grid = harness.case_grid(harness.WORKLOADS)
        patterns = [pattern for _, pattern in grid]
        for pattern in harness.ADVERSARIAL_PATTERNS:
            self.assertEqual(patterns.count(pattern), 1, pattern)
        self.assertEqual(
            len(grid), len(harness.WORKLOADS) + len(harness.ADVERSARIAL_PATTERNS)
        )

    def test_empty_workloads_give_an_empty_grid(self):
        """``test_backend_selection`` drives the entry/exit path this way."""
        self.assertEqual(harness.case_grid(()), [])

    def test_adversarial_workload_has_more_than_one_batch_row(self):
        workload = harness.adversarial_workload(harness.WORKLOADS)
        self.assertGreater(workload.batch, 1)
        self.assertIn(workload, harness.WORKLOADS)

    def test_identity_reproduces_the_pre_pattern_construction(self):
        """``speed.py`` drives this path, so ``identity`` must be byte-identical.

        Its grid has no pattern dimension; if the default drifted, the speed
        table would silently start timing a different workload.
        """
        for workload in harness.WORKLOADS[:1]:
            case = harness.build_case(workload, "cpu")
            self.assertEqual(case.pattern, harness.IDENTITY)
            gather = torch.arange(workload.gather_rows, dtype=torch.int32)
            self.assertTrue(
                torch.equal(
                    case.physical, gather.reshape(workload.batch, harness.TOPK)
                )
            )
            self.assertTrue(torch.equal(case.raw, case.physical))
            self.assertTrue(
                torch.equal(
                    case.lengths,
                    torch.full((workload.batch,), harness.TOPK, dtype=torch.int32),
                )
            )

    def test_adversarial_workload_always_resolves(self):
        single = (harness.Workload("only", 1, 512),)
        self.assertIs(harness.adversarial_workload(single), single[0])


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class PatternInvariantTests(unittest.TestCase):
    """Both invariants every pattern must preserve, checked on all of them."""

    def _cases(self):
        workload = harness.adversarial_workload(harness.WORKLOADS)
        yield harness.build_case(workload, "cpu")
        for pattern in harness.ADVERSARIAL_PATTERNS:
            yield harness.build_case(workload, "cpu", pattern=pattern)

    def test_mask_keeps_exactly_the_fixed_width(self):
        # ``reference.pack_rows_ref`` raises on any other count, so a pattern that
        # changed it would fail as a confusing packer error instead of here.
        for case in self._cases():
            with self.subTest(pattern=case.pattern):
                kept = case.mask.sum(-1)
                self.assertTrue(bool((kept == config.PACKED_KEPT_VALUES).all()))

    def test_raw_always_equals_physical(self):
        # Native bakes the rotation position in at store time; the packed gather
        # derives it from raw. Only equality phase-aligns the two legs.
        for case in self._cases():
            with self.subTest(pattern=case.pattern):
                self.assertTrue(torch.equal(case.raw, case.physical))

    def test_physical_tiles_the_batch_and_stays_in_range(self):
        for case in self._cases():
            with self.subTest(pattern=case.pattern):
                self.assertEqual(
                    tuple(case.physical.shape), (case.batch, harness.TOPK)
                )
                self.assertEqual(case.physical.dtype, torch.int32)
                present = case.physical[case.physical >= 0]
                self.assertLess(int(present.max()), case.rows)

    def test_lengths_stay_within_the_select(self):
        for case in self._cases():
            with self.subTest(pattern=case.pattern):
                self.assertTrue(bool((case.lengths >= 0).all()))
                self.assertTrue(bool((case.lengths <= harness.TOPK).all()))

    def test_masked_latent_is_derived_from_the_mask(self):
        for case in self._cases():
            with self.subTest(pattern=case.pattern):
                self.assertTrue(
                    torch.equal(
                        case.masked_latent, case.latent.masked_fill(~case.mask, 0)
                    )
                )


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class IndexPatternTests(unittest.TestCase):
    def setUp(self):
        self.workload = harness.adversarial_workload(harness.WORKLOADS)
        self.identity = harness.build_case(self.workload, "cpu")

    def _build(self, pattern):
        return harness.build_case(self.workload, "cpu", pattern=pattern)

    def test_each_index_pattern_moves_the_gather(self):
        for pattern in ("permuted", "duplicated", "interior_slots"):
            with self.subTest(pattern=pattern):
                case = self._build(pattern)
                self.assertFalse(
                    torch.equal(case.physical, self.identity.physical),
                    f"{pattern} left physical unchanged",
                )
                self.assertTrue(torch.equal(case.lengths, self.identity.lengths))

    def test_permuted_is_a_bisection_of_the_record_range(self):
        case = self._build("permuted")
        flat = case.physical.reshape(-1)
        self.assertEqual(torch.sort(flat).values.tolist(), torch.arange(flat.numel()).tolist())

    def test_duplicated_gathers_each_record_twice(self):
        case = self._build("duplicated")
        flat = case.physical.reshape(-1)
        self.assertEqual(flat.numel() % 2, 0)
        first, second = flat[: flat.numel() // 2], flat[flat.numel() // 2 :]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(set(first.tolist())), len(first.tolist()))

    def test_ragged_varies_the_length_including_to_empty(self):
        case = self._build("ragged")
        self.assertFalse(torch.equal(case.lengths, self.identity.lengths))
        self.assertEqual(int((case.lengths == 0).sum()), 4)
        # Beyond a row's length the ABI says absent, and says it as -1.
        absent = case.physical < 0
        self.assertTrue(bool(absent.any()))
        self.assertTrue(bool((absent == ~_live(case)).all()))

    def test_interior_slots_hides_present_slots_inside_the_valid_range(self):
        """The suffix-only probe this replaces could not fail a prefix predicate."""
        case = self._build("interior_slots")
        self.assertTrue(torch.equal(case.lengths, self.identity.lengths))
        absent = case.physical < 0
        self.assertTrue(bool(absent.any()))
        inside = absent & (_slots(case.batch) < case.lengths.view(-1, 1))
        self.assertTrue(
            bool(inside.any()),
            "every absent slot is a suffix, so masking on topk_lengths still passes",
        )
        # And the suffix is *not* all absent, so it is not a ragged case in disguise.
        self.assertFalse(bool((absent & ~inside).any()))


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class MaskPatternTests(unittest.TestCase):
    def setUp(self):
        self.workload = harness.adversarial_workload(harness.WORKLOADS)
        self.identity = harness.build_case(self.workload, "cpu")

    def _build(self, pattern):
        return harness.build_case(self.workload, "cpu", pattern=pattern)

    def test_each_mask_pattern_reselects_the_kept_set(self):
        for pattern in MASK_PATTERNS:
            with self.subTest(pattern=pattern):
                case = self._build(pattern)
                self.assertFalse(
                    torch.equal(case.mask, self.identity.mask),
                    f"{pattern} left the mask unchanged",
                )
                # Re-selection only: the gather is untouched.
                self.assertTrue(torch.equal(case.physical, self.identity.physical))

    def test_no_tail_prunes_the_whole_rope_tail(self):
        case = self._build("no_tail")
        self.assertFalse(bool(case.mask[:, config.NOPE_DIM :].any()))
        self.assertEqual(int(case.mask.sum()), case.rows * config.PACKED_KEPT_VALUES)

    def test_half_pair_keeps_exactly_one_lane_per_pair(self):
        case = self._build("half_pair")
        pairs = case.mask[:, config.NOPE_DIM :].reshape(
            case.rows, config.ROPE_DIM // 2, 2
        )
        self.assertTrue(bool(pairs[:, :, 0].all()), "real lane must be kept")
        self.assertFalse(bool(pairs[:, :, 1].any()), "imag lane must be pruned")

    def test_extreme_coord_holds_in_the_last_coordinate(self):
        case = self._build("extreme_coord")
        self.assertTrue(bool(case.mask[:, config.HEAD_DIM - 1].all()))


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class NativeGatherTests(unittest.TestCase):
    """The general per-slot reference: record ``physical[j]``, not row ``j``."""

    def setUp(self):
        self.workload = harness.adversarial_workload(harness.WORKLOADS)

    def _table(self, case):
        """A table whose every coordinate identifies its own record."""
        return torch.arange(case.rows, dtype=torch.float32).view(-1, 1).expand(
            -1, config.HEAD_DIM
        ).contiguous()

    def test_identity_slice_is_the_special_case_it_replaced(self):
        case = harness.build_case(self.workload, "cpu")
        table = self._table(case)
        gathered = harness.native_gather(case, table)
        self.assertTrue(
            torch.equal(gathered, table[: self.workload.gather_rows])
        )

    def test_permuted_gather_does_not_reduce_to_a_slice(self):
        case = harness.build_case(self.workload, "cpu", pattern="permuted")
        table = self._table(case)
        gathered = harness.native_gather(case, table)
        self.assertFalse(
            torch.equal(gathered, table[: self.workload.gather_rows])
        )
        self.assertTrue(torch.equal(gathered[:, 0], case.physical.reshape(-1).float()))

    def test_duplicated_slots_agree(self):
        case = harness.build_case(self.workload, "cpu", pattern="duplicated")
        gathered = harness.native_gather(case, self._table(case))
        half = gathered.shape[0] // 2
        self.assertTrue(torch.equal(gathered[:half], gathered[half:]))

    def test_absent_slots_read_as_zero(self):
        for pattern in ("ragged", "interior_slots"):
            with self.subTest(pattern=pattern):
                case = harness.build_case(self.workload, "cpu", pattern=pattern)
                gathered = harness.native_gather(
                    case, torch.ones(case.rows, config.HEAD_DIM)
                ).view(case.batch, harness.TOPK, config.HEAD_DIM)
                live = _live(case)
                self.assertTrue(bool((gathered[live] == 1).all()))
                self.assertTrue(bool((gathered[~live] == 0).all()))


@unittest.skipIf(torch is None, "requires CPU PyTorch")
class ProbeAndIndexTests(unittest.TestCase):
    def setUp(self):
        self.workload = harness.adversarial_workload(harness.WORKLOADS)
        self.case = harness.build_case(self.workload, "cpu")

    def test_probe_query_default_is_the_stride_sample(self):
        q, dims = harness.probe_query(self.case)
        self.assertTrue(
            torch.equal(dims, torch.arange(harness.HEAD_COUNT) * 8)
        )
        self.assertTrue(bool((q.sum(-1) == 1).all()))

    def test_probe_query_chunks_tile_every_coordinate_once(self):
        seen = []
        for base in range(0, config.HEAD_DIM, harness.HEAD_COUNT):
            dims = torch.arange(harness.HEAD_COUNT) + base
            q, returned = harness.probe_query(self.case, dims)
            self.assertTrue(torch.equal(returned, dims))
            self.assertEqual(
                tuple(q.shape), (self.case.batch, harness.HEAD_COUNT, config.HEAD_DIM)
            )
            self.assertTrue(bool((q.sum(-1) == 1).all()))
            seen.append(dims)
        every = torch.cat(seen)
        self.assertEqual(sorted(every.tolist()), list(range(config.HEAD_DIM)))
        # The last chunk is exactly the RoPE tail, so NoPE and tail split on a
        # chunk boundary instead of being interleaved.
        self.assertTrue(bool((seen[-1] >= config.NOPE_DIM).all()))

    def test_flat_indices_enumerates_row_major_when_nothing_is_absent(self):
        indices = harness.flat_indices(self.case).view(self.case.batch, harness.TOPK)
        expected = (
            torch.arange(self.case.batch).view(-1, 1) * harness.TOPK
            + torch.arange(harness.TOPK).view(1, -1)
        )
        self.assertTrue(torch.equal(indices, expected))

    def test_flat_indices_marks_absent_slots_negative(self):
        case = harness.build_case(self.workload, "cpu", pattern="ragged")
        indices = harness.flat_indices(case).view(case.batch, harness.TOPK)
        live = _live(case)
        expected = (
            torch.arange(case.batch).view(-1, 1) * harness.TOPK
            + torch.arange(harness.TOPK).view(1, -1)
        )
        self.assertTrue(bool((indices[~live] == -1).all()))
        self.assertTrue(torch.equal(indices[live], expected[live]))


if __name__ == "__main__":
    unittest.main()
