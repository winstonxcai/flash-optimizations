"""Advisory lock held while a Modal CLI process reads local image inputs. Author: Winston Cai.

Modal rejects a build when a file copied into the image changes during the build.
The launcher writes `.modal-build.lock` (JSON with the owner's pid) at the
repository root for the lifetime of a `modal run`/`modal deploy` process; a
Claude Code hook (`.claude/hooks/deny-edit-during-build.py`) denies edits to
image inputs while an owner is alive. A dead owner's lock is stale and ignored.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path

LOCK_NAME = ".modal-build.lock"
MODAL_CLI_NAMES = frozenset({"modal"})


def lock_owner(repo: Path) -> int | None:
    """Return the live owner's pid, or None when no lock is held or it is stale."""
    path = repo / LOCK_NAME
    try:
        pid = int(json.loads(path.read_text())["pid"])
        os.kill(pid, 0)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return pid


def acquire(repo: Path) -> Path:
    """Write the lock for this process; refuse when another live process holds it."""
    owner = lock_owner(repo)
    if owner is not None and owner != os.getpid():
        raise RuntimeError(
            f"Modal build already running (pid {owner}, {repo / LOCK_NAME}); wait for it "
            "to finish or cancel it before launching another."
        )
    path = repo / LOCK_NAME
    path.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}))
    atexit.register(release, repo)
    return path


def release(repo: Path) -> None:
    """Remove the lock only if this process owns it."""
    path = repo / LOCK_NAME
    try:
        if int(json.loads(path.read_text())["pid"]) == os.getpid():
            path.unlink()
    except (OSError, ValueError, KeyError, TypeError):
        pass


def acquire_for_modal_cli(repo: Path, argv0: str | None = None) -> Path | None:
    """Acquire the lock only inside the Modal CLI, not when tests import a launcher."""
    if Path(argv0 if argv0 is not None else sys.argv[0]).name not in MODAL_CLI_NAMES:
        return None
    return acquire(repo)
