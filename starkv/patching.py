"""Apply, restore, verify, and census STAR-CSA patches in the SGLang tree.

Edits are declared as ``(anchor, replacement)`` lists so the same declarations
can be applied, reverted, or censused read-only against a different SGLang
revision (``drift``). Anchors are validated against the file's pristine base --
its ``.starkv.orig`` backup when the tree is already patched, else the file as
it is on disk -- so a census reads exactly the text ``patch()`` would target.
"""

import os
import shutil
import tempfile
from pathlib import Path

from . import config
from .patches.attention import _backend_edits
from .patches.compressor import _compressor_edits
from .patches.pool import _memory_pool_edits, _pool_config_edits

# Ordered targets for patch/unpatch/verify/drift. The Indexer, SWA and c128
# pools are deliberately absent: they keep their native records.
_TARGETS = (
    (config.COMPRESSOR_V2, _compressor_edits),
    (config.MEM_POOL, _memory_pool_edits),
    (config.POOL_CFG, _pool_config_edits),
    (config.DSV4_BACKEND, _backend_edits),
)


def _render(path: Path, source: str, edits) -> str:
    """Validate every anchor without touching the source or its backup."""
    for anchor, new in edits:
        if source.count(anchor) != 1:
            raise AssertionError(f"[starkv] anchor count != 1 in {path}: {anchor[:70]!r}")
        source = source.replace(anchor, new, 1)
    compile(source, str(path), "exec")
    return source


def _plan(*, restoring: bool = False):
    """Rebuild expected patches from originals, never from already-patched text.

    Restoring is driven by the backup alone, not by re-rendering today's
    declarations: after a re-base the text on disk is a *previous* patch that
    today's factory can no longer produce, and refusing to restore it would
    strand the tree. The backup is the authority for what the original was; the
    only thing checked is that the tree is one we patched (marker + backup), so a
    hand-edit is still never silently discarded.
    """
    plan = []
    for filename, factory in _TARGETS:
        path = Path(filename)
        current = path.read_text()
        backup = Path(filename + ".starkv.orig")
        if not backup.exists():
            if restoring:
                if config.MARKER in current:
                    raise RuntimeError(f"[starkv] missing original backup: {path}")
                continue  # Nothing owned by STAR-CSA to restore; never git checkout.
        original = backup.read_text() if backup.exists() else current
        if config.MARKER in original:
            raise RuntimeError(f"[starkv] backup is not an unpatched original: {backup}")
        if restoring:
            plan.append((path, current, original, original))
            continue
        expected = _render(path, original, factory())
        if current not in (original, expected):
            raise RuntimeError(
                f"[starkv] unexpected edits or incompatible patch in {path}; "
                "preserve/reconcile them before patching or restoring"
            )
        plan.append((path, current, original, expected))
    return plan


def _atomic_write(path: Path, content: str) -> None:
    """Replace a source file without exposing a partially written Python module."""
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _commit(plan, *, restoring: bool = False) -> None:
    changed = []
    created_backups = []
    try:
        for path, current, original, expected in plan:
            if path.read_text() != current:
                raise RuntimeError(f"[starkv] source changed during patch operation: {path}")
            target = original if restoring else expected
            if current == target:
                continue
            backup = Path(str(path) + ".starkv.orig")
            if not restoring and not backup.exists():
                created_backups.append(backup)
                shutil.copy2(path, backup)
            _atomic_write(path, target)
            changed.append((path, current))
    except BaseException:
        # If rollback itself fails, keep backups for manual recovery.
        for path, current in reversed(changed):
            _atomic_write(path, current)
        for backup in created_backups:
            backup.unlink(missing_ok=True)
        raise
    for path, *_ in plan:
        written = any(written_path == path for written_path, _ in changed)
        if written:
            state = "restored" if restoring else "patched"
        else:
            state = "already clean" if restoring else "already patched"
        print(f"[starkv] {path}: {state}")


def patch() -> None:
    """Prevalidate all targets, apply idempotently, and roll back on write errors."""
    os.makedirs(config.ctrl_dir(), exist_ok=True)
    _commit(_plan())


def unpatch() -> None:
    """Restore verified backups only; refuse to overwrite unexpected user edits."""
    _commit(_plan(restoring=True), restoring=True)


def verify() -> None:
    """Require the complete expected patch and its original backup in every file."""
    plan = _plan()
    for path, current, _, expected in plan:
        if current != expected:
            raise RuntimeError(f"[starkv] missing or incomplete patch: {path}")
    for path, current, *_ in plan:
        print(f"{path}: starkv_markers={current.count(config.MARKER)}")


def drift() -> int:
    """Read-only anchor census against the tree at SRC_ROOT.

    Checks EVERY target file instead of raising on the first mismatch. For each
    file it reports every anchor whose count in the pristine source is != 1
    (missing=0, duplicated>1, or context changed) with the anchor count, printing
    a file -> anchor -> count table. Never writes. Returns the number of files
    that drifted (missing target or any broken anchor); 0 means every anchor of
    every edit still matches this tree one-to-one. Run before the first patch.
    """
    drifted = 0
    for filename, factory in _TARGETS:
        path = Path(filename)
        status = "patched" if Path(str(path) + ".starkv.orig").exists() else "pristine"
        if not path.exists():
            print(f"[drift] MISSING  {path}  (target dropped by this tree?)")
            drifted += 1
            continue
        source = path.read_text()
        backup = Path(str(path) + ".starkv.orig")
        base = backup.read_text() if backup.exists() else source
        edits = factory()
        broken = [(a, base.count(a)) for a, _ in edits if base.count(a) != 1]
        if broken:
            drifted += 1
            print(f"[drift] DRIFT {len(broken)}/{len(edits)}  {path}  ({status})")
            for anchor, count in broken:
                line = next((ln.strip() for ln in anchor.splitlines() if ln.strip()), "")
                print(f"        count={count:<2}  {line[:78]!r}")
        else:
            print(f"[drift] OK    {len(edits)}/{len(edits)}  {path}  ({status})")
    if drifted:
        print(f"[drift] {drifted} file(s) drifted -- anchors need re-basing")
    else:
        print("[drift] all anchors intact against this tree")
    return drifted
