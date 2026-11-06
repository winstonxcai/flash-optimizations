"""Map a diff to the validation it needs. Author: Winston Cai.

    python3 scripts/validation_tier.py [--base origin/main] [--profile parent|fork]

Prints JSON with the tier (low, medium or high), the tier of each changed path
and the checks that tier requires. The tier is guidance for authors and
reviewers, not a gate: it exists so a docs or launcher change does not default
to the multi-hour kernel qualification, and a kernel change is not waved through
with CPU tests.

`low` is an allowlist of pure documentation and run-record data. Anything else
that is not recognized is `medium`, never `low`. The diff covers commits since
the merge base with --base, plus staged, unstaged and untracked files; renames
are reported as a delete plus an add so a moved executable keeps its old path.

`parent` classifies this repository's diff, where a fork shows up only as a
changed gitlink (a pin bump). `fork` classifies a fork repository's own diff.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Literal

Tier = Literal["low", "medium", "high"]
TIER_ORDER: tuple[Tier, ...] = ("low", "medium", "high")

REQUIRED_CHECKS: dict[Tier, list[str]] = {
    "low": ["Document review only; no tests or GPU runs."],
    "medium": [
        "python3 scripts/sources.py .",
        "python3 -m unittest scripts.modal.tests",
        "No GPU run unless the change alters what validate_kernels, benchmark_kernels "
        "or serve.py execute; then one budgeted Modal run with user authorization.",
    ],
    "high": [
        "Focused CPU tests for the changed contract (docs/agents/numerical-and-validation.md).",
        "python3 -m unittest scripts.modal.tests",
        "validate_kernels --sanitizer on H100 before promotion. Tools may run as separate "
        "jobs via --sanitizer-tools; a subset run records sanitizer_qualification=partial, "
        "and qualification needs all three tools passing on identical source identities.",
        "Fresh-context review against REVIEW.md.",
    ],
}

# Text and run-record data. Nothing here is executed or consumed as configuration.
_DOC_SUFFIXES = (".md",)
_RECORD_SUFFIXES = (".json", ".csv", ".txt", ".log", ".xml", ".png", ".svg", ".pdf", ".md")
_RECORD_DIRS = ("transfer/", "raw/", "docs/")
_FORK_GITLINK = re.compile(r"third_party/(sglang|flashmla)[^/]*")
# Image recipes and submodule sources decide which code a validation run builds and tests.
_PARENT_HIGH = (".gitmodules", "Dockerfile", ".dockerignore")
_FORK_MEDIUM_PREFIXES = (
    "test/", "tests/", "benchmark/", "python/sglang/test/", "python/sglang/benchmark/", "examples/",
)


def _is_low(posix: str) -> bool:
    name = PurePosixPath(posix).name
    if posix.startswith(".claude/") or posix.startswith(".github/workflows/"):
        return False  # agent configuration and executable CI
    lowered = posix.lower()
    if name == "SHA256SUMS" or lowered.endswith(_DOC_SUFFIXES):
        return True
    in_records = posix.startswith(_RECORD_DIRS) or "/runs/" in f"/{posix}"
    return in_records and lowered.endswith(_RECORD_SUFFIXES)


def classify_path(path: str, profile: Literal["parent", "fork"] = "parent") -> Tier:
    """Tier of one repository-relative path."""
    posix = PurePosixPath(path).as_posix()
    if profile == "parent":
        if _FORK_GITLINK.fullmatch(posix) or posix.startswith(_PARENT_HIGH):
            return "high"  # the pin or the build recipe changed: the code under test moved
        return "low" if _is_low(posix) else "medium"
    if _is_low(posix):
        return "low"
    return "medium" if posix.startswith(_FORK_MEDIUM_PREFIXES) else "high"


def classify(paths: list[str], profile: Literal["parent", "fork"] = "parent") -> dict:
    """Overall tier (the highest of any path) plus per-path tiers and required checks."""
    by_path = {path: classify_path(path, profile) for path in paths}
    tier: Tier = max(by_path.values(), key=TIER_ORDER.index, default="low")
    result = {"tier": tier, "paths": by_path, "required_checks": REQUIRED_CHECKS[tier]}
    if not paths:
        result["note"] = "No changes found against the base; check --base before trusting 'low'."
    return result


def _git(*arguments: str, cwd: str | None = None) -> str:
    try:
        return subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True, timeout=60, cwd=cwd
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {error}") from error


def changed_paths(base: str) -> list[str]:
    """Repository-root-relative paths changed since the merge base with `base`.

    Includes staged, unstaged and untracked files. Every command runs from the
    repository root: `ls-files --others` and a `diff.relative` config would
    otherwise report paths relative to the caller's subdirectory, and changes
    outside it (for example `.gitmodules`) would be missed.
    """
    root = _git("rev-parse", "--show-toplevel").strip()
    commands = (
        ("diff", "--name-only", "--no-renames", "--no-relative", f"{base}...HEAD"),
        ("diff", "--name-only", "--no-renames", "--no-relative", "HEAD"),
        ("ls-files", "--others", "--exclude-standard"),
    )
    paths: set[str] = set()
    for command in commands:
        paths.update(path for path in _git(*command, "-z", cwd=root).split("\0") if path)
    return sorted(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--profile", choices=("parent", "fork"), default="parent")
    arguments = parser.parse_args()
    try:
        paths = changed_paths(arguments.base)
    except RuntimeError as error:
        sys.exit(f"validation_tier: {error}")
    print(json.dumps(classify(paths, arguments.profile), indent=2))
