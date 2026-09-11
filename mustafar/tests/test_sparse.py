"""CPU checks for the direct 328-byte sparse MLA path.

The GPU half -- ``sparse_scores``/``sparse_output`` against
``unpack_gather_bf16`` + ``flash_mla_sparse_fwd`` -- belongs in ``validity.py``
and needs a device. What is checkable without one:

* the gate's static validation, which is pure Python;
* ``merge_lse`` / ``merge_lse_lse``, which are pure torch;
* the bitmap-scan decompression ported from dhjoo98/mustafar, modelled here in
  Python and cross-checked against ``reference.unpack_rows_ref``.

That last one is the point of this module. The ``__clzll`` MSB scan, the
popcount prefix rank and the UE8M0 scale application are the parts of the
vendored kernel most likely to be transcribed wrong, and they are entirely
CPU-checkable.
"""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from mustafar import config, reference, sparse

_SIGN_BIT = 0x8000000000000000
_MASK64 = 0xFFFFFFFFFFFFFFFF


def _popcount(value: int) -> int:
    return bin(value).count("1")


def _clz64(value: int) -> int:
    """``__clzll`` for a non-zero 64-bit value."""
    assert value != 0, "__clzll is undefined on zero (the scan never sees one)"
    return 64 - value.bit_length()


def _decode_e4m3fn(code: int) -> float:
    """Mirror of ``decode_e4m3fn`` in sparse_kernel.cu, exactly.

    Notably the kernel maps E4M3's NaN encoding to 0.0f rather than NaN; the
    reference packer never emits it, so the two agree either way, but the model
    has to follow the kernel to be a real check of the kernel's logic.
    """
    sign = code >> 7
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7
    if exponent == 0:
        value = float(mantissa) * 2.0**-9
    elif exponent == 15 and mantissa == 7:
        value = 0.0
    else:
        value = float(8 + mantissa) * 2.0 ** (exponent - 10)
    return -value if sign else value


def scan_decompress(codes, bitmaps, scales):
    """Python model of ``decompress_row`` in sparse_kernel.cu.

    Structured like the kernel rather than like idiomatic torch: it recomputes
    the popcount prefix per row, walks each bitmap with the same ``__clzll`` +
    clear-MSB step, and scatters only the kept lanes.

    The accumulator is float32 and the cast to bfloat16 happens once at the end,
    matching ``unpack_rows_ref`` so the comparison can be bit-exact. The kernel
    instead rounds each value to fp16 for the MMA (see the header comment in
    sparse_kernel.cu); that difference is a precision question for the GPU
    tier, not a logic question for this one.
    """
    n = codes.shape[0]
    accumulator = torch.zeros(n, config.HEAD_DIM, dtype=torch.float32)

    bm = bitmaps.cpu()
    if bm.dtype != torch.int64:
        bm = bm.view(torch.int64)
    cd = codes.cpu()
    sc = scales.cpu()

    for row in range(n):
        prefix = []
        rank = 0
        for w in range(config.BITMAP_WORDS):
            prefix.append(rank)
            rank += _popcount(int(bm[row, w]) & _MASK64)

        for g in range(config.BITMAP_WORDS):
            word = int(bm[row, g]) & _MASK64
            scale = torch.tensor(
                2.0 ** (int(sc[row, g]) - 127), dtype=torch.float32
            )
            src = prefix[g]
            nnz = _popcount(word)
            for j in range(nnz):
                pos = _clz64(word)
                word &= ~(_SIGN_BIT >> pos)
                code = torch.tensor(_decode_e4m3fn(int(cd[row, src + j])),
                                    dtype=torch.float32)
                accumulator[row, g * 64 + pos] = code * scale
    return accumulator.to(torch.bfloat16)


def _random_packed_rows(n: int, seed: int = 0):
    """Build packed rows through the reference packer, so the ABI is the real one."""
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(n, config.HEAD_DIM, generator=generator)
    mask = reference.topmag_keep_mask(latent, 0.5)
    return reference.pack_rows_ref(latent, mask, torch.ones(config.HEAD_DIM), 1e-6)


@unittest.skipIf(torch is None, "requires PyTorch")
class ScanDecompressTests(unittest.TestCase):
    def test_scan_matches_reference_unpack(self):
        codes, bitmaps, scales = _random_packed_rows(8, seed=1)
        expected = reference.unpack_rows_ref(codes, bitmaps, scales)
        self.assertTrue(torch.equal(expected, scan_decompress(codes, bitmaps, scales)))

    def test_popcount_prefix_ranks_exactly_256_values(self):
        """The rank invariant the kernel relies on in place of dhjoo98's idx table."""
        codes, bitmaps, scales = _random_packed_rows(4, seed=2)
        del codes, scales
        bm = bitmaps.cpu()
        if bm.dtype != torch.int64:
            bm = bm.view(torch.int64)
        for row in range(bm.shape[0]):
            per_word = [_popcount(int(v) & _MASK64) for v in bm[row]]
            self.assertEqual(sum(per_word), config.PACKED_KEPT_VALUES)
            # The last word's values must end exactly at the last stored code.
            self.assertEqual(sum(per_word[: config.BITMAP_WORDS - 1]) + per_word[-1],
                             config.PACKED_KEPT_VALUES)

    def test_pruned_lanes_are_zero(self):
        codes, bitmaps, scales = _random_packed_rows(4, seed=3)
        out = scan_decompress(codes, bitmaps, scales)
        keep = reference.bitmap_to_mask(bitmaps)
        self.assertTrue(torch.all(out[~keep] == 0), "pruned coords decompress to zero")
        self.assertTrue(torch.all(keep.sum(-1) == config.PACKED_KEPT_VALUES))

    def test_scale_byte_selects_its_own_group(self):
        """A scale applied to the wrong 64-channel group would pass every test above."""
        codes, bitmaps, scales = _random_packed_rows(2, seed=4)
        shifted = scales.clone()
        shifted[:, 0] = (shifted[:, 0].to(torch.int32) + 1).to(torch.uint8)
        baseline = scan_decompress(codes, bitmaps, scales)
        altered = scan_decompress(codes, bitmaps, shifted)

        keep = reference.bitmap_to_mask(bitmaps)[:, : config.TILE_SIZE]
        touched = keep & (baseline[:, : config.TILE_SIZE] != 0)
        self.assertTrue(touched.any(), "seed produced no kept lane in group 0")
        # Doubling the scale byte exactly doubles the dequantized value.
        self.assertTrue(
            torch.equal(altered[:, : config.TILE_SIZE][touched],
                        2.0 * baseline[:, : config.TILE_SIZE][touched])
        )
        self.assertTrue(torch.equal(altered[:, config.TILE_SIZE:],
                                    baseline[:, config.TILE_SIZE:]))


@unittest.skipIf(torch is None, "requires PyTorch")
class MergeLseTests(unittest.TestCase):
    def _random(self, n=2, h=3, seed=0):
        generator = torch.Generator().manual_seed(seed)
        return (
            torch.randn(n, h, config.HEAD_DIM, generator=generator),
            torch.randn(n, h, generator=generator) * 4,
        )

    def test_equal_lse_is_a_plain_average(self):
        out_a, _ = self._random(seed=1)
        out_b, lse = self._random(seed=2)
        merged = sparse.merge_lse(out_a, lse, out_b, lse)
        self.assertEqual(merged.shape, out_a.shape)
        self.assertTrue(torch.allclose(merged.float(), ((out_a + out_b) / 2).float(),
                                       atol=1e-2))

    def test_dominant_leg_wins(self):
        out_a = torch.zeros(1, 1, config.HEAD_DIM)
        out_b = torch.ones(1, 1, config.HEAD_DIM)
        merged = sparse.merge_lse(out_a, torch.tensor([[0.0]]), out_b,
                                      torch.tensor([[40.0]]))
        self.assertTrue(torch.allclose(merged.float(), out_b.float(), atol=1e-3))

    def test_both_legs_empty_does_not_produce_nan(self):
        """-inf - -inf is NaN in the naive split; this is the regression guard."""
        neg_inf = torch.full((1, 2), float("-inf"))
        out = torch.randn(1, 2, config.HEAD_DIM)
        merged = sparse.merge_lse(out, neg_inf, out, neg_inf)
        self.assertFalse(torch.isnan(merged).any())
        self.assertTrue(torch.all(merged == 0))

    def test_one_leg_empty_returns_the_other(self):
        out_a = torch.randn(1, 2, config.HEAD_DIM)
        out_b = torch.randn(1, 2, config.HEAD_DIM)
        merged = sparse.merge_lse(out_a, torch.full((1, 2), float("-inf")),
                                      out_b, torch.zeros(1, 2))
        self.assertTrue(torch.allclose(merged.float(), out_b.float(), atol=1e-3))
        merged = sparse.merge_lse(out_a, torch.zeros(1, 2), out_b,
                                      torch.full((1, 2), float("-inf")))
        self.assertTrue(torch.allclose(merged.float(), out_a.float(), atol=1e-3))

    def test_combined_lse_matches_direct_logsumexp(self):
        lse_a = torch.tensor([[1.0, 2.0]])
        lse_b = torch.tensor([[3.0, -5.0]])
        combined = sparse.merge_lse_lse(lse_a, lse_b)
        expected = torch.log2(torch.exp2(lse_a) + torch.exp2(lse_b))
        self.assertTrue(torch.allclose(combined, expected, atol=1e-6))

    def test_combined_lse_of_two_empty_legs_is_finite(self):
        neg_inf = torch.full((1, 2), float("-inf"))
        self.assertFalse(torch.isnan(sparse.merge_lse_lse(neg_inf, neg_inf)).any())

    def test_shape_mismatch_is_rejected(self):
        out = torch.randn(1, 2, config.HEAD_DIM)
        with self.assertRaises(ValueError):
            sparse.merge_lse(out, torch.zeros(1, 2), out, torch.zeros(1, 3))


@unittest.skipIf(torch is None, "requires PyTorch")
class GateTests(unittest.TestCase):
    _BASE = {
        "SGLANG_OPT_TOPMAG": "1",
        "KEEP": "0.5",
        "SGLANG_OPT_TOPMAG_PACKED": "1",
    }

    def _validate(self, **extra):
        env = dict(self._BASE)
        env.update(extra)
        with patch.dict("os.environ", env, clear=True):
            config.validate_packed_static_config()

    def test_valid_combination_passes(self):
        self._validate(SGLANG_OPT_TOPMAG_SPARSE="1")

    def test_enabled_only_by_exact_value(self):
        for value in ("", "0", "true", "2"):
            with patch.dict("os.environ", {"SGLANG_OPT_TOPMAG_SPARSE": value},
                            clear=True):
                self.assertFalse(config.sparse_enabled())

    def test_requires_packed(self):
        with patch.dict("os.environ", {"SGLANG_OPT_TOPMAG": "1",
                                       "SGLANG_OPT_TOPMAG_SPARSE": "1"},
                        clear=True):
            with self.assertRaisesRegex(RuntimeError,
                                        "requires SGLANG_OPT_TOPMAG_PACKED"):
                config.validate_packed_static_config()

    def test_mutually_exclusive_with_fused(self):
        with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
            self._validate(SGLANG_OPT_TOPMAG_SPARSE="1",
                           SGLANG_OPT_TOPMAG_FUSED="1")

    def test_fused_still_works_without_sparse(self):
        """The _FUSED fallback must not be collateral damage."""
        self._validate(SGLANG_OPT_TOPMAG_FUSED="1")


@unittest.skipIf(torch is None, "requires PyTorch")
class CudaGuardTests(unittest.TestCase):
    def test_c4_leg_refuses_a_cuda_less_host_before_allocating(self):
        """The guard must precede the dlopen, on any host.

        Patched rather than skipped so this runs identically on the CPU tier and
        the GPU tier -- a skipUnless(is_available()) here would mean the check it
        exists to perform only ever ran where it could not fail.
        """
        empty = torch.zeros(1, 1, dtype=torch.int32)
        with patch.object(torch.cuda, "is_available", return_value=False):
            with patch.object(
                sparse, "_load",
                side_effect=AssertionError("dlopen ran before the CUDA guard"),
            ):
                with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                    sparse.c4_leg(
                        torch.zeros(1, 64, config.HEAD_DIM, dtype=torch.bfloat16),
                        torch.zeros(1, config.PACKED_KEPT_VALUES, dtype=torch.uint8),
                        torch.zeros(1, config.BITMAP_WORDS),
                        torch.zeros(1, config.BITMAP_WORDS, dtype=torch.uint8),
                        empty, empty,
                        torch.zeros(4, config.ROPE_DIM // 2, dtype=torch.complex64),
                        0.1,
                    )


_BACKEND = Path(config.DSV4_BACKEND)
_BACKEND_ORIGINAL = Path(config.DSV4_BACKEND + ".mustafar.orig")


@unittest.skipUnless(
    _BACKEND_ORIGINAL.exists(), "pinned SGLang tree not present (set SG_LOWRANK_SRC)"
)
class BackendBranchTests(unittest.TestCase):
    """Structural checks on the injected decode branch.

    The branch cannot be exercised without a GPU, but its *shape* is a pure
    string property: it must be gated, it must sit after the sparse-prefill
    dispatch (which handles large extends itself), and it must sit before the
    packed reconstruct block so no reassembly happens when it fires.
    """

    @classmethod
    def setUpClass(cls):
        from mustafar import patching
        from mustafar.patches.attention import _backend_edits

        original = _BACKEND_ORIGINAL.read_text()
        # _render never touches the disk; it only anchors, replaces and compiles.
        cls.rendered = patching._render(
            _BACKEND, original, _backend_edits(source=original)
        )

    def _offsets(self):
        out = self.rendered
        return {
            "prefill": out.index("return self._forward_prefill_sparse("),
            "guard": out.index("_sg_lr.sparse_enabled()"),
            "packed": out.index("## MUSTAFAR (decode/small-extend native"),
            "dispatch": out.index("            if _is_sm120:\n"),
        }

    def test_branch_is_ordered_around_the_paths_it_must_not_disturb(self):
        at = self._offsets()
        self.assertLess(at["prefill"], at["guard"])
        self.assertLess(at["guard"], at["packed"])
        self.assertLess(at["packed"], at["dispatch"])

    def test_branch_returns_before_the_reconstruct_block(self):
        out = self.rendered
        at = self._offsets()
        segment = out[at["guard"] : at["packed"]]
        self.assertIn("return _sg_lr.sparse_forward(", segment)
        self.assertNotIn("unpack_gather_native", segment)

    def test_branch_is_narrowed_to_single_token_decode(self):
        at = self._offsets()
        condition = self.rendered[at["guard"] - 120 : at["guard"] + 200]
        self.assertIn("not _is_sm120", condition)
        self.assertIn("q.shape[1] == 1", condition)
        self.assertIn("compress_ratio == 4", condition)

    def test_sink_is_counted_once_and_only_on_the_native_leg(self):
        at = self._offsets()
        segment = self.rendered[at["guard"] : at["packed"]]
        self.assertEqual(segment.count("attn_sink=attn_sink"), 1)
        call = segment[segment.index("return _sg_lr.sparse_forward(") :]
        self.assertNotIn("attn_sink", call)

    def test_rendered_source_compiles(self):
        compile(self.rendered, str(_BACKEND), "exec")


@unittest.skipIf(torch is None, "requires PyTorch")
class LoaderTests(unittest.TestCase):
    def test_loader_does_not_swallow_interrupts(self):
        """Ctrl-C during the dlopen must propagate, not be cached as a load error."""
        with patch.object(sparse, "_extension", None), patch.object(
            sparse, "_load_error", None
        ):
            with patch.object(sparse.importlib, "import_module",
                              side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    sparse.sparse_available()
            self.assertIsNone(sparse._load_error)

    def test_load_failure_becomes_a_runtime_error(self):
        with patch.object(sparse, "_extension", None), patch.object(
            sparse, "_load_error", None
        ):
            with patch.object(sparse.importlib, "import_module",
                              side_effect=ImportError("no such extension")):
                self.assertFalse(sparse.sparse_available())
            self.assertIsInstance(sparse._load_error, ImportError)

    def test_available_reflects_an_installed_extension(self):
        self.assertEqual(
            sparse.sparse_available(),
            importlib.util.find_spec("mustafar._sparse") is not None,
        )


if __name__ == "__main__":
    unittest.main()
