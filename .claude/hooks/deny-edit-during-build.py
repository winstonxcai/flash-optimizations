#!/usr/bin/env python3
"""PreToolUse hook: deny edits to Modal image inputs while a build lock is held.

Author: Winston Cai.

Reads the tool call as JSON on stdin. Exit code 2 blocks the edit and shows
stderr to the model. Shell commands that write files are not covered; the lock
file itself (scripts/build_lock.py) is the source of truth.
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(REPO))

from scripts.build_lock import LOCK_NAME, lock_owner  # noqa: E402

# Files and trees copied into the Modal images by scripts/modal/kernels.py and
# serve.py (add_local_*, from_dockerfile with the repository as build context).
IMAGE_INPUTS = (
    "scripts/modal/",
    "scripts/local/",
    "scripts/sources.py",
    "scripts/build_lock.py",
    "third_party/",
    "Dockerfile",
    ".dockerignore",
)


def is_image_input(path: str) -> bool:
    try:
        relative = Path(path).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return False
    return relative.startswith(IMAGE_INPUTS)


def main() -> int:
    call = json.load(sys.stdin)
    tool_input = call.get("tool_input", {})
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    owner = lock_owner(REPO)
    if owner is not None and path and is_image_input(path):
        print(
            f"Blocked: {path} is a Modal image input and a build is running (pid {owner}, "
            f"{LOCK_NAME}). Editing it now makes Modal reject the build. Wait for the build "
            "to finish or cancel it.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
