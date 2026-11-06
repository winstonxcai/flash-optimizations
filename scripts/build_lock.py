"""Advisory lock held while a Modal CLI process may read local image inputs. Author: Winston Cai.

Modal rejects a build when a file copied into the image changes during the build.
The launcher writes `.modal-build.lock` (JSON with the owner's pid and process
start time) at the repository root for the lifetime of a `modal run`/`modal
deploy` process, not just the image-build step: the image is built lazily after
the launcher is imported, so the whole CLI process is the safe bound. While the
lock is held a Claude Code hook (`.claude/hooks/deny-edit-during-build.py`)
denies edits to image inputs, and a second Modal CLI launch from this checkout
is refused. Delete the file by hand only after confirming the owner is gone.

The lock is created atomically (hard link of a fully written temp file), so a
reader never observes a partial file. A lock is live only while its pid exists
and its recorded start time still matches, so a reused pid is treated as stale.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LOCK_NAME = ".modal-build.lock"
MODAL_CLI_NAMES = frozenset({"modal"})


def _start_time(pid: int) -> str | None:
    """Return the kernel-reported start time of `pid`, or None if unavailable."""
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:  # os.kill(0, ...) would signal the whole process group
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:  # exists, owned by another user
        return True
    except OSError:
        return False
    return True


def _read(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text())
        int(record["pid"])
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def lock_owner(repo: Path) -> int | None:
    """Return the live owner's pid, or None when no lock is held or it is stale."""
    record = _read(repo / LOCK_NAME)
    if record is None:
        return None
    pid = int(record["pid"])
    if not _pid_alive(pid):
        return None
    recorded, actual = record.get("start"), _start_time(pid)
    if recorded and actual and recorded != actual:
        return None  # the pid was reused by a different process
    return pid


def acquire(repo: Path) -> Path:
    """Create the lock for this process; refuse when another live process holds it."""
    path = repo / LOCK_NAME
    record = json.dumps({"pid": os.getpid(), "start": _start_time(os.getpid()), "created": time.time()})
    temporary = repo / f"{LOCK_NAME}.{os.getpid()}.tmp"
    temporary.write_text(record)
    try:
        for _ in range(2):
            try:
                os.link(temporary, path)  # atomic, fails if the lock exists
            except FileExistsError:
                owner = lock_owner(repo)
                if owner == os.getpid():
                    break
                if owner is not None:
                    raise RuntimeError(
                        f"Modal build already running (pid {owner}, {path}); wait for it "
                        "to finish or cancel it before launching another."
                    ) from None
                path.unlink(missing_ok=True)  # stale or unreadable: replace it
            else:
                break
        else:
            raise RuntimeError(f"Could not acquire {path}; another launcher is racing for it.")
    finally:
        temporary.unlink(missing_ok=True)
    atexit.register(release, repo)
    return path


def release(repo: Path) -> None:
    """Remove the lock only if this process owns it."""
    path = repo / LOCK_NAME
    record = _read(path)
    if record is not None and int(record["pid"]) == os.getpid():
        path.unlink(missing_ok=True)


def _is_modal_cli(argv0: str) -> bool:
    path = Path(argv0)
    return path.name in MODAL_CLI_NAMES or (path.name == "__main__.py" and path.parent.name == "modal")


def acquire_for_modal_cli(repo: Path, argv0: str | None = None) -> Path | None:
    """Acquire the lock only inside the Modal CLI, not when tests import a launcher."""
    if not _is_modal_cli(argv0 if argv0 is not None else sys.argv[0]):
        return None
    return acquire(repo)
