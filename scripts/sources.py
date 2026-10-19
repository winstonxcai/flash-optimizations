"""Read-only source gate shared by local and Modal launchers. Author: Winston Cai.

Run before builds or host-side benchmarks: python3 scripts/sources.py .
Explicit Modal source overrides must be Git checkouts; their dirty contents and
nested sources are fingerprinted, not silently treated as a pinned release.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.PIPE, timeout=30
        ).rstrip("\n")
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Source preflight cannot inspect {root}: {error}") from error


def check_sources(repo: Path, overrides: dict[str, Path] | None = None) -> dict:
    """Require exact clean fork pins, or record explicitly overridden Git trees.

    Parent edits are allowed. Fork staged, unstaged and untracked changes are not,
    unless the caller explicitly selects that fork through an experiment override.
    This checks submitted source, not the identity of an already-built image.
    """
    repo = repo.resolve()
    overrides = overrides or {}
    unknown = overrides.keys() - {"sglang", "flashmla"}
    if unknown:
        raise ValueError(f"Unknown source overrides: {sorted(unknown)}")
    records = {"parent": {"path": str(repo), "commit": _git(repo, "rev-parse", "HEAD")}}
    recovery = (
        "Inspect and preserve changes first. For a clean checkout, run "
        "git submodule update --init --recursive --checkout from the parent; "
        "this preflight never repairs, stashes or discards work."
    )

    def inspect(root: Path, name: str, expected: str, override: bool) -> None:
        root = root.resolve()
        # Git otherwise walks up to the parent when a submodule is uninitialized.
        if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
            raise RuntimeError(f"Source {name} is not an initialized Git checkout: {root}. {recovery}")
        actual = _git(root, "rev-parse", "HEAD")
        if actual != expected and not override:
            raise RuntimeError(
                f"Source {name} pin mismatch: expected {expected}, actual {actual} at {root}. {recovery}"
            )
        for entry in _git(root, "ls-tree", "-rz", "HEAD").split("\0"):
            if entry.startswith("160000 "):
                metadata, child = entry.split("\t", 1)
                inspect(root / child, f"{name}/{child}", metadata.split()[2], override)
        dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
        if dirty and not override:
            raise RuntimeError(
                f"Source {name} is dirty at {root} (expected/actual {expected}):\n{dirty}\n{recovery}"
            )
        record = {
            "path": str(root), "expected_commit": expected, "commit": actual,
            "override": override, "dirty": bool(dirty),
        }
        if override:
            digest = hashlib.sha256()
            files = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
            for relative in sorted(set(files) - {""}):
                path = root / relative
                digest.update(relative.encode() + b"\0")
                if path.is_symlink():
                    digest.update(b"link\0" + hashlib.sha256(str(path.readlink()).encode()).digest())
                elif path.is_file():
                    digest.update(b"file\0" + hashlib.sha256(path.read_bytes()).digest())
                elif not path.exists():
                    digest.update(b"deleted\0")
            record["sha256"] = digest.hexdigest()
        records[name] = record

    for name in ("sglang", "flashmla"):
        relative = f"third_party/{name}"
        entry = _git(repo, "ls-tree", "HEAD", "--", relative)
        if not entry.startswith("160000 "):
            raise RuntimeError(f"Parent HEAD has no submodule pin for {relative}")
        inspect(overrides.get(name, repo / relative), name, entry.split()[2], name in overrides)
    return records


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("usage: python3 scripts/sources.py <repository>")
        print(json.dumps(check_sources(Path(sys.argv[1])), indent=2))
    except (RuntimeError, ValueError) as error:
        sys.exit(str(error))
