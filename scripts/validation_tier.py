"""Map a diff to the validation it needs. Author: Winston Cai.

    python3 scripts/validation_tier.py [--base origin/main] [--profile parent|fork]

Prints JSON with the tier (low, medium or high), the tier of each changed path
and the checks that tier requires. The tier is guidance for authors and
reviewers, not a gate: it exists so a docs or launcher change does not default
to the multi-hour kernel qualification, and a kernel change is not waved through
with CPU tests. Paths that match no rule are `medium`, never `low`.

`parent` classifies this repository's diff, where a fork shows up only as a
changed gitlink (a pin bump). `fork` classifies a fork repository's own diff.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
        "jobs via --sanitizer-tools; each run is partial evidence (sanitizer_complete is "
        "false) and qualification needs all three tools passing on identical source identities.",
        "Fresh-context review against REVIEW.md.",
    ],
}

# Documentation, run records and CI metadata: no executable behavior.
_LOW_PREFIXES = ("docs/", "transfer/", "raw/", ".github/")
_LOW_SUFFIXES = (".md", ".txt", "SHA256SUMS")
_FORK_GITLINK = re.compile(r"third_party/(sglang|flashmla)[^/]*")
# Launchers, preflights, experiment harnesses, training and agent configuration.
_PARENT_MEDIUM_PREFIXES = (
    "scripts/", "benchmarks/", "training/", ".claude/", "Dockerfile", ".dockerignore", ".gitignore",
)
_FORK_MEDIUM_PREFIXES = (
    "test/", "tests/", "benchmark/", "python/sglang/test/", "python/sglang/benchmark/", "examples/",
)


def classify_path(path: str, profile: Literal["parent", "fork"] = "parent") -> Tier:
    """Tier of one repository-relative path."""
    posix = PurePosixPath(path).as_posix()
    if profile == "parent" and _FORK_GITLINK.fullmatch(posix):
        return "high"  # a fork gitlink: the pin moved, so production source changed
    if posix.startswith(_LOW_PREFIXES) or posix.endswith(_LOW_SUFFIXES) or "/runs/" in f"/{posix}":
        return "low"
    if profile == "parent":
        return "medium"  # launchers, harnesses, training and anything unrecognized
    return "medium" if posix.startswith(_FORK_MEDIUM_PREFIXES) else "high"


def classify(paths: list[str], profile: Literal["parent", "fork"] = "parent") -> dict:
    """Overall tier (the highest of any path) plus per-path tiers and required checks."""
    by_path = {path: classify_path(path, profile) for path in paths}
    tier: Tier = max(by_path.values(), key=TIER_ORDER.index, default="low")
    return {"tier": tier, "paths": by_path, "required_checks": REQUIRED_CHECKS[tier]}


def changed_paths(base: str) -> list[str]:
    """Paths changed on HEAD since its merge base with `base` (committed work only)."""
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot list changes against {base}: {error}") from error
    return [line for line in out.splitlines() if line]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--profile", choices=("parent", "fork"), default="parent")
    arguments = parser.parse_args()
    print(json.dumps(classify(changed_paths(arguments.base), arguments.profile), indent=2))
