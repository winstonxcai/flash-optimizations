"""Move-only regression checks captured before the refactor at ecd3047.

Offline: python -m unittest mustafar.tests.test_refactor
Pinned-source integration (CPU only, downloads five source files):
    MUSTAFAR_TEST_FETCH_SGLANG=1 python -m unittest mustafar.tests.test_refactor

The snapshot is retained unchanged. Approved identifier renames and the explicit
c4_topk correction are normalized only when comparing historical snapshot hashes.
The corrected upstream guard and current public API are tested separately.
Do not regenerate the snapshot to make a code change pass.
"""

import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mustafar import config, patching

ROOT = Path(__file__).resolve().parents[2]
BASELINE = json.loads(
    (Path(__file__).parent / "fixtures/refactor-baseline.json").read_text()
)
TARGETS = dict(
    zip(
        BASELINE["sources"],
        ("COMPRESSOR_V2", "MEM_POOL", "POOL_CFG", "INDEXER", "DSV4_BACKEND"),
    )
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def historical_guard(text):
    # This one known-bad guard exists in the historical snapshot, not upstream.
    # Keep all other snapshot checks intact while testing the corrected guard below.
    return (
        text.replace("if self.c4_topk != 512:", "if self._topk != 512:")
        .replace("PACKED_RECORD_BYTES", "PACKED_BYTES")
        .replace(
            "        # It must never be treated as a native hybrid allocation.",
            "# It must never be treated as a native hybrid allocation.",
        )
    )


def historical_names(source):
    """Allow only the explicit identifier renames; keep the stored AST unchanged."""
    renames = {
        "PACKED_RECORD_BYTES": "PACKED_BYTES",
        "PACKED_KEPT_VALUES": "PACKED_KEEP",
        "bitmap_to_mask": "bitmap_to_bits",
        "native_bytes": "raw",
        "dense_bf16": "dense",
    }
    source = re.sub(
        r"\b(" + "|".join(renames) + r")\b",
        lambda match: renames[match.group()],
        source,
    )
    return source.replace(
        "Fused accepted a disabled packed pool",
        "Fused accepted a disabled packed- pool",
    )


class MoveTests(unittest.TestCase):
    def test_generated_edits_are_byte_identical(self):
        for variant, anchor in BASELINE["anchors"].items():
            with (
                patch.object(config, "PACKAGE_ROOT", "/mustafar-regression-root"),
                patch.object(Path, "read_text", return_value=anchor),
            ):
                for name, expected in BASELINE["edits"][variant].items():
                    with self.subTest(variant=variant, name=name):
                        edits = json.dumps(getattr(patching, name)())
                        self.assertEqual(digest(historical_guard(edits)), expected)

    def test_runtime_and_reference_bodies_unchanged(self):
        # Search both old and new homes, so this check runs before and after moving.
        definitions = {}
        for relative in ("packed.py", "reference.py", "tests/unit.py"):
            source = (ROOT / "mustafar" / relative).read_text()
            tree = ast.parse(historical_names(source))
            # Python 3.12 adds empty type_params fields; the baseline is 3.11.
            for node in ast.walk(tree):
                if getattr(node, "type_params", None) == []:
                    node._fields = tuple(f for f in node._fields if f != "type_params")
            definitions.update(
                {
                    n.name: digest(ast.dump(n))
                    for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                }
            )
        for original in BASELINE["ast"].values():
            for name, expected in original.items():
                with self.subTest(name=name):
                    self.assertEqual(definitions[name], expected)

    def test_patch_api_imports_without_torch_or_sglang(self):
        subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                (
                    "import sys, mustafar; from mustafar import patching; "
                    "assert mustafar.patch is patching.patch; "
                    "assert mustafar.unpatch is patching.unpatch; "
                    "assert mustafar.verify is patching.verify; "
                    "assert 'torch' not in sys.modules and 'sglang' not in sys.modules"
                ),
            ],
            cwd=ROOT,
            check=True,
        )

    def test_missing_or_duplicate_anchor_does_not_write_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            for content in ("untouched\n", "anchor\nanchor\n"):
                path.write_text(content)
                with self.assertRaises(AssertionError):
                    patching._render(path, content, [("anchor\n", "replacement\n")])
                self.assertEqual(path.read_text(), content)
                self.assertFalse(Path(str(path) + ".mustafar.orig").exists())

    def test_backend_guard_uses_upstream_attribute_in_both_variants(self):
        import textwrap

        for source in BASELINE["anchors"].values():
            replacements = "".join(
                new for _, new in patching._backend_edits(source=source)
            )
            self.assertNotIn("self._topk", replacements)
            start = replacements.index("            if self.c4_topk != 512:")
            end = replacements.index("            if self.token_to_kv_pool", start)
            guard = compile(textwrap.dedent(replacements[start:end]), "guard", "exec")
            exec(guard, {"self": SimpleNamespace(c4_topk=512)})  # noqa: S102 - local generated guard
            with self.assertRaisesRegex(RuntimeError, "index_topk=512"):
                exec(guard, {"self": SimpleNamespace(c4_topk=128)})  # noqa: S102


@unittest.skipUnless(
    os.environ.get("MUSTAFAR_TEST_FETCH_SGLANG") == "1",
    "set MUSTAFAR_TEST_FETCH_SGLANG=1 for pinned-source checks",
)
class PinnedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def fetch(item):
            name, spec = item
            url = (
                "https://raw.githubusercontent.com/sgl-project/sglang/"
                + BASELINE["sglang_commit"]
                + "/python/"
                + spec["path"]
            )
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode()
            if digest(text) != spec["before"]:
                raise AssertionError(f"pinned source mismatch: {name}")
            return name, text

        with ThreadPoolExecutor(max_workers=5) as executor:
            cls.originals = dict(executor.map(fetch, BASELINE["sources"].items()))

    def setUp(self):
        stack = self.enterContext(ExitStack())
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.paths = {}
        for name, content in self.originals.items():
            path = root / BASELINE["sources"][name]["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            self.paths[name] = path
            stack.enter_context(patch.object(config, TARGETS[name], str(path)))
        stack.enter_context(
            patch.object(config, "PATCH_FILES", tuple(map(str, self.paths.values())))
        )
        stack.enter_context(
            patch.object(config, "PACKAGE_ROOT", "/mustafar-regression-root")
        )
        self.output = stack.enter_context(redirect_stdout(io.StringIO()))

    def test_patch_output_compile_verify_and_restore(self):
        patching.patch()
        for name, path in self.paths.items():
            patched = path.read_text()
            self.assertEqual(
                digest(historical_guard(patched)), BASELINE["sources"][name]["after"]
            )
            compile(patched, str(path), "exec")
            self.assertEqual(
                Path(str(path) + ".mustafar.orig").read_text(), self.originals[name]
            )
        patching.verify()
        self.assertEqual(self.output.getvalue().count("mustafar_markers="), 5)
        patching.unpatch()
        for name, path in self.paths.items():
            self.assertEqual(path.read_text(), self.originals[name])

    def test_repeat_patch_is_idempotent(self):
        patching.patch()
        before = {name: path.read_text() for name, path in self.paths.items()}
        backups = {
            name: Path(str(path) + ".mustafar.orig").read_bytes()
            for name, path in self.paths.items()
        }
        patching.patch()
        self.assertEqual(
            {name: path.read_text() for name, path in self.paths.items()}, before
        )
        self.assertEqual(
            {
                name: Path(str(path) + ".mustafar.orig").read_bytes()
                for name, path in self.paths.items()
            },
            backups,
        )

    def test_patch_failure_is_transactional(self):
        self.paths["_backend_edits"].write_text("unsupported source\n")
        before = {name: path.read_text() for name, path in self.paths.items()}
        with self.assertRaises(AssertionError):
            patching.patch()
        self.assertEqual(
            {name: path.read_text() for name, path in self.paths.items()}, before
        )
        for path in self.paths.values():
            self.assertFalse(Path(str(path) + ".mustafar.orig").exists())

    def test_verify_rejects_missing_patch(self):
        with self.assertRaises((AssertionError, RuntimeError)):
            patching.verify()

    def test_upstream_defines_c4_topk(self):
        tree = ast.parse(self.originals["_backend_edits"])
        assigned = {
            target.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        }
        self.assertIn("c4_topk", assigned)
        self.assertNotIn("_topk", assigned)

    def test_write_error_rolls_back_sources_and_new_backups(self):
        real_replace = os.replace
        for action in (patching.patch, patching.unpatch):
            with self.subTest(action=action.__name__):
                if action is patching.unpatch:
                    patching.patch()
                calls = 0

                def fail_once(source, destination):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise OSError("injected write failure")
                    real_replace(source, destination)

                before = {name: path.read_bytes() for name, path in self.paths.items()}
                modes = {name: path.stat().st_mode for name, path in self.paths.items()}
                with (
                    patch.object(patching.os, "replace", side_effect=fail_once),
                    self.assertRaisesRegex(OSError, "injected write failure"),
                ):
                    action()
                self.assertEqual(
                    {name: path.read_bytes() for name, path in self.paths.items()},
                    before,
                )
                self.assertEqual(
                    {name: path.stat().st_mode for name, path in self.paths.items()},
                    modes,
                )
                for name, path in self.paths.items():
                    backup = Path(str(path) + ".mustafar.orig")
                    if action is patching.patch:
                        self.assertFalse(backup.exists())
                    else:
                        self.assertEqual(backup.read_text(), self.originals[name])
                    self.assertFalse(list(path.parent.glob(f".{path.name}.*")))

    def test_preexisting_user_edits_survive_patch_and_restore(self):
        path = self.paths["_backend_edits"]
        path.write_text(path.read_text() + "\n# pre-existing user edit\n")
        before = {name: p.read_bytes() for name, p in self.paths.items()}
        patching.patch()
        patching.verify()
        patching.unpatch()
        patching.unpatch()
        self.assertEqual(
            {name: p.read_bytes() for name, p in self.paths.items()}, before
        )

    def test_verify_rejects_a_pristine_file_in_an_otherwise_patched_tree(self):
        patching.patch()
        name = "_backend_edits"
        self.paths[name].write_text(self.originals[name])
        with self.assertRaisesRegex(RuntimeError, "missing or incomplete patch"):
            patching.verify()

    def test_partial_patch_and_user_edits_are_rejected_without_writes(self):
        patching.patch()
        path = self.paths["_backend_edits"]
        good = path.read_text()
        for changed in (
            good.replace("self.c4_topk != 512", "self._topk != 512"),
            good + "\n# user edit\n",
        ):
            path.write_text(changed)
            before = {name: p.read_bytes() for name, p in self.paths.items()}
            for action in (patching.verify, patching.patch, patching.unpatch):
                with self.assertRaisesRegex(RuntimeError, "unexpected edits"):
                    action()
                self.assertEqual(
                    {name: p.read_bytes() for name, p in self.paths.items()}, before
                )

    def test_missing_backup_never_uses_git_or_restores_other_files(self):
        patching.patch()
        path = self.paths["_backend_edits"]
        Path(str(path) + ".mustafar.orig").unlink()
        before = {name: p.read_bytes() for name, p in self.paths.items()}
        with patch("subprocess.run") as run:
            for action in (patching.patch, patching.unpatch, patching.verify):
                with self.assertRaisesRegex(RuntimeError, "missing original backup"):
                    action()
            run.assert_not_called()
        self.assertEqual(
            {name: p.read_bytes() for name, p in self.paths.items()}, before
        )

    def test_unpatch_without_backups_preserves_user_files(self):
        path = self.paths["_backend_edits"]
        path.write_text("# unrelated user content\n")
        before = {name: p.read_bytes() for name, p in self.paths.items()}
        with patch("subprocess.run") as run:
            patching.unpatch()
            run.assert_not_called()
        self.assertEqual(
            {name: p.read_bytes() for name, p in self.paths.items()}, before
        )


if __name__ == "__main__":
    unittest.main()
