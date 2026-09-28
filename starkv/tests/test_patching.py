"""Patch-machinery tests.

These exercise the apply/verify/unpatch/drift lifecycle on a synthetic tree, so
they run anywhere (no torch, no container, no real SGLang checkout). Whether the
*real* anchors still match the fork tree is a different question, answered by
`python -m starkv drift`, which `container.sh prep` runs before the first patch.

What is checked here that the real anchors cannot check by themselves: that the
machinery refuses an ambiguous anchor, never clobbers a hand-edit, restores
byte-exactly, and stays idempotent.
"""

import shutil
import tempfile
from pathlib import Path

from .. import config, patching
from ..patches import _import_block
from ..patches.attention import _backend_edits
from ..patches.compressor import _compressor_edits
from ..patches.pool import _memory_pool_edits, _pool_config_edits

_ORIGINAL = '''"""Synthetic module for the machinery test."""
from __future__ import annotations


def forward(x, plan):
    if x is None:
        return None
    return x + 1
'''

_EDITS = [
    ("from __future__ import annotations\n",
     "from __future__ import annotations\n" + config.MARKER + " (import)\n"),
    ("    if x is None:\n        return None\n",
     "    if x is None:\n        return None\n    x = x * 2\n"),
]


class _Sandbox:
    """A one-file tree with patching._TARGETS pointed at it."""

    def __enter__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="starkv-patch-test-"))
        self.path = self.dir / "synthetic.py"
        self.path.write_text(_ORIGINAL)
        self._saved = patching._TARGETS
        patching._TARGETS = ((str(self.path), lambda: _EDITS),)
        return self

    def __exit__(self, *exc):
        patching._TARGETS = self._saved
        shutil.rmtree(self.dir, ignore_errors=True)
        return False

    @property
    def backup(self) -> Path:
        return Path(str(self.path) + ".starkv.orig")

    @property
    def text(self) -> str:
        return self.path.read_text()


def test_render_refuses_a_missing_or_ambiguous_anchor():
    """count != 1 is an error in both directions, before anything is written."""
    try:
        patching._render(Path("x.py"), _ORIGINAL, [("not in the source\n", "y\n")])
    except AssertionError:
        pass
    else:
        raise AssertionError("a missing anchor was accepted")

    # `from __future__ import annotations` appears once, but an anchor that is
    # an empty string matches everywhere -- that must be caught too.
    try:
        patching._render(Path("x.py"), _ORIGINAL, [("", "y\n")])
    except AssertionError:
        pass
    else:
        raise AssertionError("an ambiguous anchor was accepted")


def test_render_output_compiles_and_carries_the_marker():
    rendered = patching._render(Path("x.py"), _ORIGINAL, _EDITS)
    compile(rendered, "x.py", "exec")
    assert config.MARKER in rendered
    assert "    x = x * 2\n" in rendered


def test_patch_verify_unpatch_round_trip():
    with _Sandbox() as box:
        patching.patch()
        assert box.text != _ORIGINAL
        assert box.backup.exists()
        assert box.backup.read_text() == _ORIGINAL
        patching.verify()

        # Idempotent: a second patch is a no-op, not a second insertion.
        once = box.text
        patching.patch()
        assert box.text == once

        patching.unpatch()
        assert box.text == _ORIGINAL


def test_unpatch_restores_over_a_hand_edit():
    """The backup is the authority, so a hand-edited tree is still restorable."""
    with _Sandbox() as box:
        patching.patch()
        box.path.write_text(box.text + "\n# hand edit\n")
        patching.unpatch()
        assert box.text == _ORIGINAL


def test_patch_refuses_to_overwrite_an_unexpected_edit():
    """Once patched, an edit that is neither the original nor ours must stop the
    patch rather than be folded in or silently dropped.

    Before the first patch there is nothing to compare against -- the file on
    disk is by definition the baseline -- so the guard is defined on the
    patched state, which is the one where a stray edit means real confusion.
    """
    with _Sandbox() as box:
        patching.patch()
        patched = box.text
        box.path.write_text(patched + "\n# concurrent edit\n")
        try:
            patching.patch()
        except RuntimeError as exc:
            assert "unexpected edits" in str(exc)
        else:
            raise AssertionError("an unrelated edit was overwritten")
        assert box.text == patched + "\n# concurrent edit\n"


def test_marker_without_a_backup_is_an_error():
    """A marked file with no backup means the original was lost."""
    with _Sandbox() as box:
        marked = config.MARKER + "\n" + _ORIGINAL
        box.path.write_text(marked)
        try:
            patching.patch()
        except RuntimeError as exc:
            assert "original" in str(exc)
        else:
            raise AssertionError("a lost original was silently patched")
        assert box.text == marked


def test_drift_counts_files_and_writes_nothing():
    with _Sandbox() as box:
        assert patching.drift() == 0
        assert box.text == _ORIGINAL
        assert not box.backup.exists()

        # A tree whose anchor context moved.
        box.path.write_text(_ORIGINAL.replace("    if x is None:", "    if x is None :"))
        assert patching.drift() == 1
        assert box.text == _ORIGINAL.replace("    if x is None:", "    if x is None :")


def test_drift_reports_a_missing_target():
    with _Sandbox() as box:
        box.path.unlink()
        assert patching.drift() == 1


def test_every_factory_has_unique_nonempty_anchors():
    """Two identical anchors in one file could never both apply.

    `_render` would raise on the first (count == 2), so this is the same bug
    caught earlier and without a tree.
    """
    for name, factory in (
        ("compressor", _compressor_edits),
        ("pool", _memory_pool_edits),
        ("pool_config", _pool_config_edits),
        ("attention", _backend_edits),
    ):
        edits = factory()
        anchors = [a for a, _ in edits]
        assert len(edits) >= 2, f"{name}: suspiciously small edit list"
        assert all(a.strip() for a in anchors), f"{name}: empty anchor"
        assert len(set(anchors)) == len(anchors), f"{name}: duplicate anchor"
        for anchor, new in edits:
            assert anchor != new, f"{name}: no-op edit"


def test_import_block_targets_the_live_package():
    block = _import_block()
    assert config.PACKAGE_ROOT in block
    assert "import starkv as _sg_lr" in block
    assert "except Exception" in block  # a broken package must not break the server


TESTS = (
    test_render_refuses_a_missing_or_ambiguous_anchor,
    test_render_output_compiles_and_carries_the_marker,
    test_patch_verify_unpatch_round_trip,
    test_unpatch_restores_over_a_hand_edit,
    test_patch_refuses_to_overwrite_an_unexpected_edit,
    test_marker_without_a_backup_is_an_error,
    test_drift_counts_files_and_writes_nothing,
    test_drift_reports_a_missing_target,
    test_every_factory_has_unique_nonempty_anchors,
    test_import_block_targets_the_live_package,
)


def run() -> int:
    failures = 0
    for t in TESTS:
        try:
            t()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {t.__name__}")
    print(f"{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
