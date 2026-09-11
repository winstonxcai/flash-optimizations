"""Live behaviour of :mod:`mustafar.patching` -- offline, no network, no SGLang.

The patch machinery is what the whole package depends on, so it is tested
directly. Two classes of question, kept apart:

* :class:`PatchMachineryTests` -- does ``patch``/``unpatch``/``verify`` do the
  right thing? Driven by small self-authored fixtures with self-authored
  anchors, so it is hermetic and never goes stale.
* :class:`RealAnchorTests` -- do the *real* anchors still apply to the pinned
  SGLang tree? Needs SRC_ROOT on disk, so it skips cleanly on a bare host. This
  is the same tree ``patching.drift()`` (run by ``container.sh``) censuses.

Synthetic fixtures cannot be built from the real anchor list: those anchors are
fragments of one large call expression (bare argument lists, partial ``if``
bodies), and ``_render`` compiles what it produces. A fragment fixture has no
valid Python form, which is exactly why the real-anchor question is answered
against the real files instead of a reconstruction of them.

Run offline::

    python3 -m unittest mustafar.tests.test_patching -v
"""

import ast
import inspect
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mustafar import config, patching

ROOT = Path(__file__).resolve().parents[2]

# ``patching._TARGETS`` holds resolved path strings, so the config attribute
# names are listed separately and verified against it. A reorder or a dropped
# target shows up as a clear failure here rather than a confusing patch error.
TARGET_NAMES = (
    "COMPRESSOR_V2",
    "MEM_POOL",
    "POOL_CFG",
    "INDEXER",
    "DSV4_BACKEND",
    "HICACHE_ASSEMBLER",
)
if len(TARGET_NAMES) != len(patching._TARGETS) or any(
    getattr(config, name) != path
    for name, (path, _) in zip(TARGET_NAMES, patching._TARGETS)
):
    raise AssertionError("TARGET_NAMES no longer lines up with patching._TARGETS")


# --- self-authored machinery fixtures ----------------------------------------
# Replacements carry config.MARKER, as the real factories do via _import_block.
# The marker is load-bearing in _plan: it is how an already-patched file with a
# missing backup is told apart from a pristine one.
MARK = f"{config.MARKER}\n"


def _value_edits():
    return [("VALUE = 1\n", "VALUE = 2\n" + MARK)]


def _import_edits():
    return [
        (
            "from __future__ import annotations\n",
            "from __future__ import annotations\nimport os\n" + MARK,
        )
    ]


def _function_edits():
    return [("    return 1\n", "    return 2\n    " + MARK)]


def _guard_edits():
    return [
        (
            "    if TOPK != 512:\n        raise RuntimeError('bad')\n",
            "    "
            + MARK
            + "    if TOPK != 512:\n"
            "        raise RuntimeError('index_topk=512 required')\n",
        )
    ]


# (name, fixture text, factory). Each anchor appears exactly once in its text.
MACHINERY_FIXTURES = (
    ("ALPHA", "VALUE = 1\n", _value_edits),
    ("BETA", "from __future__ import annotations\n\nVALUE = 1\n", _import_edits),
    ("GAMMA", "def f():\n    return 1\n", _function_edits),
    ("DELTA", "def g(TOPK):\n    if TOPK != 512:\n        raise RuntimeError('bad')\n", _guard_edits),
)


def _backend_anchors() -> dict[str, str]:
    """Recover the literal prefill anchors from ``_backend_edits``' own source.

    Read from the AST rather than duplicated here, so the guard test cannot
    drift from the patch it is checking. The second ``prefill_anchor``
    assignment is a ``Name``, not a constant, and is filtered out.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(patching._backend_edits)))
    anchors = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("prefill_anchor", "v17_prefill_anchor")
            and isinstance(node.value, ast.Constant)
        ):
            anchors[node.targets[0].id] = node.value.value
    if set(anchors) != {"prefill_anchor", "v17_prefill_anchor"}:
        raise AssertionError(f"could not recover both backend anchors: {list(anchors)}")
    return anchors


class _PatchedTree:
    """Point ``patching._TARGETS`` at a tempdir and run patch operations there."""

    def __init__(self, stack: ExitStack, fixtures) -> None:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.contents: dict[str, str] = {}
        self.paths: dict[str, Path] = {}
        targets = []
        for name, text, factory in fixtures:
            self.contents[name] = text
            path = root / f"{name.lower()}.py"
            path.write_text(text)
            self.paths[name] = path
            targets.append((str(path), factory))
        # _TARGETS captures resolved path strings at import time, so patching
        # config attributes would not reach _plan: the tuple itself is the seam.
        stack.enter_context(patch.object(patching, "_TARGETS", tuple(targets)))
        stack.enter_context(
            patch.object(config, "PACKAGE_ROOT", "/mustafar-regression-root")
        )
        self.output = stack.enter_context(redirect_stdout(io.StringIO()))

    def backup(self, name: str) -> Path:
        return Path(str(self.paths[name]) + ".mustafar.orig")

    def read_all(self) -> dict[str, bytes]:
        return {n: p.read_bytes() for n, p in self.paths.items()}


class PatchMachineryTests(unittest.TestCase):
    """patch/unpatch/verify round-trips over hermetic fixtures."""

    def setUp(self):
        self.tree = _PatchedTree(self.enterContext(ExitStack()), MACHINERY_FIXTURES)

    def test_patch_compile_verify_and_restore(self):
        patching.patch()
        for name, path in self.tree.paths.items():
            patched = path.read_text()
            self.assertIn(config.MARKER, patched)
            compile(patched, str(path), "exec")
            self.assertEqual(self.tree.backup(name).read_text(), self.tree.contents[name])
        patching.verify()
        self.assertEqual(
            self.tree.output.getvalue().count("mustafar_markers="),
            len(MACHINERY_FIXTURES),
        )
        patching.unpatch()
        for name, path in self.tree.paths.items():
            self.assertEqual(path.read_text(), self.tree.contents[name])

    def test_repeat_patch_is_idempotent(self):
        patching.patch()
        before = {n: p.read_text() for n, p in self.tree.paths.items()}
        backups = {n: self.tree.backup(n).read_bytes() for n in self.tree.paths}
        patching.patch()
        self.assertEqual({n: p.read_text() for n, p in self.tree.paths.items()}, before)
        self.assertEqual(
            {n: self.tree.backup(n).read_bytes() for n in self.tree.paths}, backups
        )

    def test_missing_or_duplicate_anchor_does_not_write_target(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = root / "source.py"
        for content in ("untouched\n", "anchor\nanchor\n"):
            path.write_text(content)
            with self.assertRaises(AssertionError):
                patching._render(path, content, [("anchor\n", "replacement\n")])
            self.assertEqual(path.read_text(), content)
            self.assertFalse(Path(str(path) + ".mustafar.orig").exists())

    def test_failure_is_transactional(self):
        # A target whose text cannot satisfy its own anchors: patch() must
        # prevalidate, so nothing is written anywhere.
        self.tree.paths["DELTA"].write_text("unsupported source\n")
        before = self.tree.read_all()
        with self.assertRaises(AssertionError):
            patching.patch()
        self.assertEqual(self.tree.read_all(), before)
        for name in self.tree.paths:
            self.assertFalse(self.tree.backup(name).exists())

    def test_verify_rejects_missing_patch(self):
        with self.assertRaises((AssertionError, RuntimeError)):
            patching.verify()

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

                before = self.tree.read_all()
                modes = {n: p.stat().st_mode for n, p in self.tree.paths.items()}
                with (
                    patch.object(patching.os, "replace", side_effect=fail_once),
                    self.assertRaisesRegex(OSError, "injected write failure"),
                ):
                    action()
                self.assertEqual(self.tree.read_all(), before)
                self.assertEqual(
                    {n: p.stat().st_mode for n, p in self.tree.paths.items()}, modes
                )
                for name, path in self.tree.paths.items():
                    if action is patching.patch:
                        self.assertFalse(self.tree.backup(name).exists())
                    else:
                        self.assertEqual(
                            self.tree.backup(name).read_text(), self.tree.contents[name]
                        )
                    self.assertFalse(list(path.parent.glob(f".{path.name}.*")))

    def test_preexisting_user_edits_outside_anchors_survive(self):
        path = self.tree.paths["GAMMA"]
        path.write_text(path.read_text() + "\n# pre-existing user edit\n")
        before = self.tree.read_all()
        patching.patch()
        patching.verify()
        patching.unpatch()
        patching.unpatch()
        self.assertEqual(self.tree.read_all(), before)

    def test_user_edit_inside_patched_region_is_refused_without_writes(self):
        patching.patch()
        path = self.tree.paths["DELTA"]
        good = path.read_text()
        for changed in (
            good.replace("index_topk=512 required", "tampered"),
            good + "\n# user edit\n",
        ):
            path.write_text(changed)
            before = self.tree.read_all()
            for action in (patching.verify, patching.patch, patching.unpatch):
                with self.assertRaisesRegex(RuntimeError, "unexpected edits"):
                    action()
                self.assertEqual(self.tree.read_all(), before)

    def test_missing_backup_never_uses_git_or_touches_other_files(self):
        patching.patch()
        self.tree.backup("DELTA").unlink()
        before = self.tree.read_all()
        with patch("subprocess.run") as run:
            for action in (patching.patch, patching.unpatch, patching.verify):
                with self.assertRaisesRegex(RuntimeError, "missing original backup"):
                    action()
            run.assert_not_called()
        self.assertEqual(self.tree.read_all(), before)

    def test_unpatch_without_backups_preserves_user_files(self):
        path = self.tree.paths["DELTA"]
        path.write_text("# unrelated user content\n")
        before = self.tree.read_all()
        with patch("subprocess.run") as run:
            patching.unpatch()
            run.assert_not_called()
        self.assertEqual(self.tree.read_all(), before)


class RealAnchorTests(unittest.TestCase):
    """The real anchors, applied to the real pinned tree (needs SRC_ROOT)."""

    @classmethod
    def setUpClass(cls):
        cls.src_root = Path(config.SRC_ROOT)
        if not cls.src_root.is_dir():
            raise unittest.SkipTest(f"{cls.src_root} not present")
        # SRC_ROOT is the runtime clone, so it may already carry the patch. The
        # pristine base is the .mustafar.orig backup when present, else the file
        # on disk -- the same rule patching.drift() censuses with. Assuming the
        # clone is pristine makes patch() fail on a marker with no backup.
        cls.originals = {}
        for name in TARGET_NAMES:
            source = Path(getattr(config, name))
            backup = Path(str(source) + ".mustafar.orig")
            if backup.exists():
                original = backup.read_text()
            else:
                original = source.read_text()
                if config.MARKER in original:
                    raise unittest.SkipTest(
                        f"{source} is patched with no .mustafar.orig backup; "
                        "the pristine base cannot be recovered"
                    )
            cls.originals[name] = original

    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.tree = _PatchedTree(
            stack,
            [(name, self.originals[name], _factory_for(name)) for name in TARGET_NAMES],
        )

    def test_real_anchors_patch_verify_and_restore(self):
        patching.patch()
        for name, path in self.tree.paths.items():
            patched = path.read_text()
            self.assertIn(config.MARKER, patched)
            compile(patched, str(path), "exec")
            self.assertNotEqual(patched, self.originals[name])
        patching.verify()
        patching.unpatch()
        for name, path in self.tree.paths.items():
            self.assertEqual(path.read_text(), self.originals[name])

    def test_real_anchors_roll_back_on_write_error(self):
        real_replace = os.replace
        calls = 0

        def fail_once(source, destination):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("injected write failure")
            real_replace(source, destination)

        before = self.tree.read_all()
        with (
            patch.object(patching.os, "replace", side_effect=fail_once),
            self.assertRaisesRegex(OSError, "injected write failure"),
        ):
            patching.patch()
        self.assertEqual(self.tree.read_all(), before)
        for name in self.tree.paths:
            self.assertFalse(self.tree.backup(name).exists())


def _factory_for(name: str):
    for attribute, factory in patching._TARGETS:
        if attribute == getattr(config, name):
            return factory
    raise AssertionError(f"no factory for {name}")


class BackendGuardTests(unittest.TestCase):
    """The generated c4_topk guard must use the upstream attribute name."""

    def test_guard_uses_upstream_attribute_in_both_variants(self):
        for variant, source in _backend_anchors().items():
            with self.subTest(variant=variant):
                replacements = "".join(
                    new for _, new in patching._backend_edits(source=source)
                )
                self.assertNotIn("self._topk", replacements)
                start = replacements.index("            if self.c4_topk != 512:")
                end = replacements.index("            if self.token_to_kv_pool", start)
                guard = compile(textwrap.dedent(replacements[start:end]), "guard", "exec")
                # 512 is the only legal packed configuration, so it must pass.
                exec(guard, {"self": SimpleNamespace(c4_topk=512)})  # noqa: S102
                with self.assertRaisesRegex(RuntimeError, "index_topk=512"):
                    exec(guard, {"self": SimpleNamespace(c4_topk=128)})  # noqa: S102


class ImportContractTests(unittest.TestCase):
    """``mustafar.patch`` and friends resolve without torch or sglang present."""

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


if __name__ == "__main__":
    unittest.main()
