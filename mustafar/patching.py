"""Apply, restore, and verify Mustafar patches in the SGLang source tree."""

import os
import shutil
import tempfile
from pathlib import Path

from . import config
from .patches.attention import _backend_edits, _indexer_edits
from .patches.compressor import _compressor_edits
from .patches.pool import _memory_pool_edits, _pool_config_edits


def _render(path: Path, source: str, edits) -> str:
    """Validate every anchor without touching the source or its backup."""
    for anchor, new in edits:
        if source.count(anchor) != 1:
            raise AssertionError(
                f"[mustafar] anchor count != 1 in {path}: {anchor[:70]!r}"
            )
        source = source.replace(anchor, new, 1)
    compile(source, str(path), "exec")
    return source


def _plan(*, restoring: bool = False):
    """Rebuild expected patches from originals, never from already-patched text."""
    targets = (
        (config.COMPRESSOR_V2, _compressor_edits),
        (config.MEM_POOL, _memory_pool_edits),
        (config.POOL_CFG, _pool_config_edits),
        (config.INDEXER, _indexer_edits),
        (config.DSV4_BACKEND, _backend_edits),
    )
    plan = []
    for filename, factory in targets:
        path = Path(filename)
        current = path.read_text()
        backup = Path(filename + ".mustafar.orig")
        if not backup.exists():
            if config.MARKER in current:
                raise RuntimeError(f"[mustafar] missing original backup: {path}")
            if restoring:
                continue  # Nothing owned by Mustafar to restore; never use git checkout.
        original = backup.read_text() if backup.exists() else current
        if config.MARKER in original:
            raise RuntimeError(
                f"[mustafar] backup is not an unpatched original: {backup}"
            )
        edits = factory(source=original) if factory is _backend_edits else factory()
        expected = _render(path, original, edits)
        if current not in (original, expected):
            raise RuntimeError(
                f"[mustafar] unexpected edits or incompatible patch in {path}; "
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
                raise RuntimeError(
                    f"[mustafar] source changed during patch operation: {path}"
                )
            target = original if restoring else expected
            if current == target:
                continue
            backup = Path(str(path) + ".mustafar.orig")
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
        print(f"[mustafar] {path}: {'restored' if restoring else 'patched/verified'}")


def patch() -> None:
    """Prevalidate all targets, apply idempotently, and roll back on write errors."""
    _commit(_plan())


def unpatch() -> None:
    """Restore verified backups only; refuse to overwrite unexpected user edits."""
    _commit(_plan(restoring=True), restoring=True)


def verify() -> None:
    """Require the complete expected patch and its original backup in every file."""
    plan = _plan()
    for path, current, _, expected in plan:
        if current != expected:
            raise RuntimeError(f"[mustafar] missing or incomplete patch: {path}")
    for path, current, *_ in plan:
        print(f"{path}: mustafar_markers={current.count(config.MARKER)}")
