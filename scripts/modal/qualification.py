"""Pure qualification parsing and FP4 receipt policy for CPU/GPU consumers.

Author: Winston Cai. No Modal, Torch or CUDA imports; existing launcher tests own
regressions. A failed tool is never equivalent to an interrupted/not-run tool.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOOLS = frozenset({"memcheck", "racecheck", "synccheck"})
FP4_CASE_COUNT = 34
IDENTITY_FIELDS = (
    "validation_suite",
    "integration",
    "base_image",
    "sources",
    "source_identities",
    "coverage",
    "model_config",
    "torch",
    "cuda",
    "cuda_python",
    "cuda_bindings",
    "cuda_binding_requirements",
    "sanitizer",
    "arch",
    "binary_architectures",
    "fp4_precompile_artifacts",
    "native_dtype",
    "binaries",
    "nvcc",
)


@dataclass(frozen=True)
class QualificationIdentity:
    """Canonical fingerprint of every input that may affect qualification."""

    serialized: str

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> QualificationIdentity:
        missing = [key for key in IDENTITY_FIELDS if key not in manifest]
        if missing:
            raise RuntimeError(f"Incomplete qualification identity: {missing}")
        return cls(
            json.dumps({key: manifest[key] for key in IDENTITY_FIELDS}, sort_keys=True)
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.serialized.encode()).hexdigest()


@dataclass(frozen=True)
class QualificationReceipt:
    run_id: str
    result_sha256: str
    tools: frozenset[str]
    case_ids: frozenset[tuple[str, str]]


def qualification_run_ids(value: str) -> tuple[str, ...]:
    ids = tuple(value.split(","))
    if not value or len(set(ids)) != len(ids):
        raise ValueError("FP4 timing requires unique qualification run IDs")
    for item in ids:
        validate_run_id(item)
    return ids


def read_receipt(
    root: Path, run_id: str, identity: QualificationIdentity
) -> QualificationReceipt:
    """Accept clean complete tools; reject errors even if another run passed."""
    base = root / "microbench" / run_id / "validation"
    manifest = json.loads((base / "build-manifest.json").read_text())
    if QualificationIdentity.from_manifest(manifest) != identity:
        raise RuntimeError(f"Qualification identity changed: {run_id}")
    raw = (base / "result.json").read_bytes()
    result = json.loads(raw)
    if result.get("allocation", {}).get("status") != "passed":
        raise RuntimeError(f"Missing allocation qualification: {run_id}")
    states = result.get("sanitizer_results", {})
    if set(states) != TOOLS:
        raise RuntimeError(f"Incomplete sanitizer states: {run_id}")
    passed = set()
    expected_ids = None
    for tool, state in states.items():
        status = state.get("status")
        if status in {"not_run", "interrupted"}:
            continue
        if status != "passed":
            raise RuntimeError(
                f"Unresolved sanitizer failure: {run_id}/{tool}: {state}"
            )
        verify_sanitizer_artifacts(base, tool, state)
        cases, _ = read_pytest_report(base / f"{tool}.xml")
        ids = frozenset(
            (case.get("classname", ""), case.get("name", "")) for case in cases
        )
        if (
            len(cases) != FP4_CASE_COUNT
            or len(ids) != len(cases)
            or any(list(case) for case in cases)
            or any(not a or not b for a, b in ids)
        ):
            raise RuntimeError(f"Incomplete qualification XML: {run_id}/{tool}")
        if expected_ids is not None and ids != expected_ids:
            raise RuntimeError(f"Sanitizer case identities differ: {run_id}/{tool}")
        expected_ids = ids
        passed.add(tool)
    return QualificationReceipt(
        run_id,
        hashlib.sha256(raw).hexdigest(),
        frozenset(passed),
        expected_ids or frozenset(),
    )


def qualify_fp4_runs(
    root: Path, run_ids: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    identity = QualificationIdentity.from_manifest(manifest)
    receipts = [
        read_receipt(root, name, identity) for name in qualification_run_ids(run_ids)
    ]
    passed = set().union(*(receipt.tools for receipt in receipts))
    case_sets = {receipt.case_ids for receipt in receipts if receipt.tools}
    if passed != TOOLS or len(case_sets) != 1:
        raise RuntimeError(
            f"FP4 timing blocked by incomplete/different sanitizer coverage: {sorted(passed)}"
        )
    return {
        "receipts": {receipt.run_id: receipt.result_sha256 for receipt in receipts},
        "tools": sorted(passed),
        "case_count_per_tool": len(next(iter(case_sets))),
        "identity_sha256": identity.sha256,
    }



def validate_run_id(value: str, *, max_length: int = 128) -> None:
    """Validate one existing path-safe experiment identifier."""
    if not isinstance(value, str) or not 1 <= len(value) <= max_length or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", value
    ):
        raise ValueError(f"Invalid experiment run ID: {value!r}")


def read_pytest_report(path: Path) -> tuple[list[ET.Element], dict[str, int]]:
    """Read counts and testcase metadata; callers own their coverage policy."""
    root = ET.parse(path).getroot()
    counts = {
        key: sum(int(suite.get(key, 0)) for suite in root.iter("testsuite"))
        for key in ("tests", "errors", "failures", "skipped")
    }
    return list(root.iter("testcase")), counts


def verify_sanitizer_artifacts(root: Path, tool: str, state: dict[str, Any]) -> None:
    """New receipts bind raw logs/reports; historical receipt policy is unchanged."""
    if state.get("report_schema") is None:
        return
    if state["report_schema"] != "remnant-sanitizer-raw-v1":
        raise RuntimeError(f"Unknown sanitizer report schema: {tool}")
    for suffix in ("log", "xml"):
        path = root / f"{tool}.{suffix}"
        if hashlib.sha256(path.read_bytes()).hexdigest() != state.get(f"{suffix}_sha256"):
            raise RuntimeError(f"Sanitizer raw report changed: {tool}/{suffix}")
    check_sanitizer_report((root / f"{tool}.log").read_text(), tool)


def check_sanitizer_report(output: str, tool: str) -> None:
    if tool not in TOOLS:
        raise ValueError(f"Unknown sanitizer tool: {tool}")
    expected = (
        "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
        if tool == "racecheck"
        else "ERROR SUMMARY: 0 errors"
    )
    records = [r.strip() for r in re.findall(r"========= ([^\r\n]*)", output)]
    summaries = [
        r for r in records if r.startswith(("ERROR SUMMARY:", "RACECHECK SUMMARY:"))
    ]
    if (
        summaries != [expected]
        or any(r.startswith(("Warning:", "Error:")) for r in records)
        or "terminated before first instrumented API call" in output
    ):
        raise RuntimeError(f"Incomplete sanitizer evidence {tool}")
