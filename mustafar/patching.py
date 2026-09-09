"""Apply, restore, and verify Mustafar patches in the SGLang source tree."""

import os
import shutil
import tempfile
from pathlib import Path

from . import config
from .patches.attention import _backend_edits, _indexer_edits
from .patches.compressor import _compressor_edits
from .patches.hicache import _assembler_edits
from .patches.pool import _memory_pool_edits, _pool_config_edits

# Ordered targets for patch/unpatch/verify/drift. Each factory returns the
# (anchor, replacement) edit list for its file; _backend_edits inspects the
# source, the rest are source-independent.
_TARGETS = (
    (config.COMPRESSOR_V2, _compressor_edits),
    (config.MEM_POOL, _memory_pool_edits),
    (config.POOL_CFG, _pool_config_edits),
    (config.INDEXER, _indexer_edits),
    (config.DSV4_BACKEND, _backend_edits),
    (config.HICACHE_ASSEMBLER, _assembler_edits),
)


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
    plan = []
    for filename, factory in _TARGETS:
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


def drift() -> int:
    """Read-only anchor census against the tree at SRC_ROOT (e.g. a v0.5.18 clone).

    Checks EVERY target file instead of raising on the first mismatch. For each
    file it reports every anchor whose count in the pristine source is != 1
    (missing=0, duplicated>1, or context changed) and the anchor count, printing
    a file -> anchor -> count table. Never writes. Returns the number of files
    that drifted (missing target or any broken anchor); 0 means every anchor of
    every edit still matches this tree one-to-one.

    Anchors are validated against the file's pristine base -- its
    .mustafar.orig backup when the tree is already patched, else the file as it
    is on disk -- so the census reads the same "original" text patch() targets.
    Used by container.sh to scope the v0.5.15 -> v0.5.18 re-base.
    """
    drifted = 0
    for filename, factory in _TARGETS:
        path = Path(filename)
        status = "patched" if Path(str(path) + ".mustafar.orig").exists() else "pristine"
        if not path.exists():
            print(f"[drift] MISSING  {path}  (target dropped by this tree?)")
            drifted += 1
            continue
        source = path.read_text()
        base = source
        backup = Path(str(path) + ".mustafar.orig")
        if backup.exists():
            base = backup.read_text()
        edits = factory(source=base) if factory is _backend_edits else factory()
        broken = [(anchor, base.count(anchor)) for anchor, _ in edits if base.count(anchor) != 1]
        if broken:
            drifted += 1
            print(f"[drift] DRIFT {len(broken)}/{len(edits)}  {path}  ({status})")
            for anchor, count in broken:
                line = next((l.strip() for l in anchor.splitlines() if l.strip()), "")
                print(f"        count={count:<2}  {line[:78]!r}")
        else:
            print(f"[drift] OK    {len(edits)}/{len(edits)}  {path}  ({status})")
    if drifted:
        print(f"[drift] {drifted} file(s) drifted -- anchors need re-basing "
              f"(see container.sh drift-v0.5.18.md)")
    else:
        print("[drift] all anchors intact against this tree")
    return drifted
