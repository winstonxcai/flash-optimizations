"""GLM evidence qualification policy, independent of Modal and build setup.

Author: Winston Cai.

Consumes canonical JSON/XML/log records. Historical records lacking runtime
hashes are bound to immutable source hashes; a shallow checkout needs no old
Git objects. Missing or changed evidence fails closed.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

GLM_BASELINE = "9e42a0d8886dd93944dc6b47ef3bffef4e033b5a"
# SHA256 of each exact baseline blob under python/sglang; never current HEAD.
GLM_BASELINE_RUNTIME_HASHES = {
    "kernels/ops/attention/dsa/tilelang_kernel.py": "54dc7149990b1afbeebd076e9d1055f7d711e1d952a18c4fc3af7cef06fb7dcf",
    "srt/remnant/glm.py": "c1fbaba6d417c202f2643d66f978b07d6e1a165ea288b672bec0f5463732c19e",
    "srt/remnant/glm_pool.py": "4654b6f13b44a330e43af7bb845667f7881639bc80e472e9b3b2b59de0531612",
    "srt/remnant/reference.py": "6231ec9622623643eba8ab50d714f9433cd13672be927030c4070ae05452021e",
    "srt/layers/layernorm.py": "98072d836e0c77312058521ee1015c80177b2c370da7408fd066eec61015ff21",
}
GLM_DECODE_REQUIRED = {
    "test_packed_loader_bytes",
    "test_decode_partial_graph[8-1]",
    "test_decode_partial_graph[8-2]",
    "test_decode_partial_graph[16-1]",
    "test_decode_partial_graph[16-2]",
}


@dataclass(frozen=True)
class QualificationContext:
    """Explicit evidence inputs; no source checkout, image or GPU side effects."""

    results_root: Path
    manifest: dict
    runtime_hashes: dict[str, str]
    sanitizer_order: tuple[str, ...]


def qualification_run_ids(run_ids: str) -> list[str]:
    ids = run_ids.split(",")
    if not run_ids or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", x) for x in ids
    ):
        raise ValueError("GLM timing requires explicit --qualification-runs IDs")
    return ids


def qualify_glm_timing(
    run_ids: str,
    *,
    context: QualificationContext,
    allow_incomplete: bool = False,
    phase: str = "all",
) -> dict:
    """Accept per-tool evidence only with unchanged production code/toolchains.

    A prior overall failed run may supply a completed synccheck group; its
    failed or not-run groups cannot qualify. New partial-tool runs explicitly
    remain incomplete as full qualifications until merged here.
    """
    ids = qualification_run_ids(run_ids)
    if phase not in ("all", "decode"):
        raise ValueError("GLM qualification phase must be all or decode")
    manifest = context.manifest
    current = context.runtime_hashes
    groups = {}
    for run_id in ids:
        root = context.results_root / "microbench" / run_id / "validation"
        result = json.loads((root / "result.json").read_text())
        old = json.loads((root / "build-manifest.json").read_text())
        if result.get("model_family") != "glm53":
            raise RuntimeError(f"Not GLM qualification: {run_id}")
        if result.get("runtime_hashes") is not None:
            if result["runtime_hashes"] != current:
                raise RuntimeError(f"GLM runtime hashes changed: {run_id}")
        elif (
            old["source_identities"]["sglang"]["commit"] != GLM_BASELINE
            or current != GLM_BASELINE_RUNTIME_HASHES
        ):
            raise RuntimeError(
                f"Cannot establish unchanged historical GLM runtime: {run_id}"
            )
        if phase == "decode" and (
            old.get("sources") != manifest.get("sources")
            or old["glm_compile"].get("cache_files")
            != manifest["glm_compile"].get("cache_files")
            or old.get("nvcc") != manifest.get("nvcc")
        ):
            raise RuntimeError(
                f"GLM focused fixture/binary/source identity changed: {run_id}"
            )
        for key in (
            "torch",
            "cuda",
            "cuda_python",
            "cuda_bindings",
            "triton",
            "tilelang",
            "binaries",
        ):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(
                    f"GLM qualification dependency changed: {run_id}/{key}"
                )
        for key in ("triton", "tilelang", "attention_scale"):
            if old["glm_compile"][key] != manifest["glm_compile"][key]:
                raise RuntimeError(
                    f"GLM compiled variant identity changed: {run_id}/{key}"
                )
        if old["sanitizer"]["binary_sha256"] != manifest["sanitizer"]["binary_sha256"]:
            raise RuntimeError(f"GLM sanitizer identity changed: {run_id}")
        for tool, status in result.get("sanitizer_results", {}).items():
            if status.get("status") == "not_run":
                continue
            log = (root / f"{tool}.log").read_text()
            if status.get("status") != "passed":
                # An intentional graceful checkpoint is not a passed run. Reuse
                # only its completed cases, independently bound to raw reports.
                note = root / f"{tool}-coverage-checkpoint.json"
                if not note.is_file():
                    continue
                check_glm_interrupted_checkpoint(
                    json.loads(note.read_text()), root, run_id, tool, log
                )
            check_sanitizer_report(log, tool)
            if phase == "decode" and result.get("sanitizer_scope") == "glm-decode":
                audit = result.get("scope_audit", {})
                audit_log = (root / "scope-audit.log").read_text()
                if (
                    audit.get("status") != "passed"
                    or audit.get("filter") != "regex=glm_packed_(partial|bytes)"
                    or audit.get("log_sha256")
                    != hashlib.sha256(audit_log.encode()).hexdigest()
                    or audit.get("entrypoints")
                    != manifest["glm_compile"].get("scope_symbol_audit")
                ):
                    raise RuntimeError(
                        f"Focused instrumentation audit changed: {run_id}"
                    )
                check_sanitizer_report(audit_log, "racecheck")
                check_glm_scope_audit(audit_log, run_id)
            if phase == "decode":
                tree = ET.parse(root / f"{tool}.xml")
                tests = list(tree.iter("testcase"))
                names = {x.attrib["name"] for x in tests}
                if (
                    not names
                    or not names <= GLM_DECODE_REQUIRED
                    or len(names) != len(tests)
                    or any(
                        not x.get("classname", "").endswith(
                            "attention.unittests.test_glm5_remnant"
                        )
                        for x in tests
                    )
                    or any(
                        x.find(k) is not None
                        for x in tests
                        for k in ("failure", "error", "skipped")
                    )
                ):
                    raise RuntimeError(f"Incomplete focused cases: {run_id}/{tool}")
                for node in tests:
                    name = node.attrib["name"]
                    if name.startswith("test_decode_partial_graph["):
                        heads, inner = map(
                            int, name.split("[")[1].rstrip("]").split("-")
                        )
                        props = {
                            x.get("name"): x.get("value")
                            for x in node.findall("properties/property")
                        }
                        expected_geometry = dict(
                            heads=heads,
                            inner=inner,
                            banks=2,
                            queries=1,
                            tails=[0, 1, 2, 3],
                            nondefault_stream=True,
                            launched_partitions=1,
                            binary_partitions=66 // inner,
                        )
                        cubins = {
                            str(v["packed"]): v["cubin_sha256"]
                            for v in manifest["glm_compile"]["tilelang"]
                            if v["heads"] == heads and v["inner"] == inner
                        }
                        if json.loads(props.get("partial_cubins", "{}")) != cubins:
                            raise RuntimeError(
                                f"Focused production cubin changed: {run_id}/{name}"
                            )
                        if (
                            json.loads(props.get("decode_geometry", "{}"))
                            != expected_geometry
                        ):
                            raise RuntimeError(
                                f"Focused decode geometry changed: {run_id}/{name}"
                            )
            else:
                names = check_glm_sanitizer_cases(
                    ET.parse(root / f"{tool}.xml"),
                    run_id,
                    historical=result.get("runtime_hashes") is None,
                    allow_partial=True,
                )
            group = groups.setdefault(tool, dict(cases=set(), runs=[]))
            group["cases"].update(names)
            group["runs"].append(
                dict(
                    run_id=run_id,
                    cases=sorted(names),
                    overall_run_status=result.get("status"),
                    tool_status=status.get("status"),
                    result_sha256=hashlib.sha256(
                        (root / "result.json").read_bytes()
                    ).hexdigest(),
                )
            )
    required = (
        GLM_DECODE_REQUIRED
        if phase == "decode"
        else required_glm_sanitizer_cases(historical=False)
    )
    missing = {
        tool: sorted(required - groups.get(tool, {}).get("cases", set()))
        for tool in context.sanitizer_order
    }
    incomplete = any(missing.values())
    if incomplete and not allow_incomplete:
        raise RuntimeError(f"GLM timing blocked by missing sanitizer gates: {missing}")
    return dict(
        status="incomplete" if incomplete else "passed",
        missing=missing,
        tools={
            tool: dict(cases=sorted(group["cases"]), runs=group["runs"])
            for tool, group in groups.items()
        },
        runtime_hashes=current,
    )


def check_glm_scope_audit(log: str, identity: str) -> None:
    """Pinned tool trace proves helper warp and handoff CTA synchronization."""
    if (
        "glm_packed_bytes" not in log
        or not re.search(r"(?i)(warp|warpsync)", log)
        or not re.search(r"(?i)(block|cta|barrier)", log)
    ):
        raise RuntimeError(
            f"Missing positive TileLang synchronization trace: {identity}"
        )


def check_glm_interrupted_checkpoint(
    note: dict, root: Path, run_id: str, tool: str, log: str
) -> None:
    if (
        note.get("schema") != "remnant-glm-sanitizer-checkpoint-v1"
        or note.get("run_id") != run_id
        or note.get("tool") != tool
        or note.get("action") != "intentional SIGINT to owned pytest"
        or "KeyboardInterrupt" not in log
    ):
        raise RuntimeError(f"Invalid intentional sanitizer checkpoint: {run_id}/{tool}")
    for suffix in ("log", "xml"):
        actual = hashlib.sha256((root / f"{tool}.{suffix}").read_bytes()).hexdigest()
        if note.get(f"{suffix}_sha256") != actual:
            raise RuntimeError(
                f"Sanitizer checkpoint report changed: {run_id}/{tool}.{suffix}"
            )


def required_glm_sanitizer_cases(*, historical: bool) -> set[str]:
    required = {
        "test_producer_reconstruction",
        "test_native_fp8_conversion",
        "test_allocator_accounting",
        "test_invalid_attention",
        "test_partial_outputs",
        "test_attention[3-8-8]",
        "test_changing_input_graph[adapter]",
        "test_changing_input_graph[direct]",
        "test_production_backend_constructor",
        "test_attention[0-1-64]",
        "test_pool_lifecycle",
    }
    if not historical:
        required |= {
            "test_benchmark_sanitizer_bank_graph[8-2-2-33]",
            "test_benchmark_sanitizer_bank_graph[16-4-2-66]",
        }
    return required


def check_glm_sanitizer_cases(
    tree: ET.ElementTree, run_id: str, *, historical: bool, allow_partial: bool = False
) -> set[str]:
    required = required_glm_sanitizer_cases(historical=historical)
    nodes = list(tree.getroot().iter("testcase"))
    names = {
        node.get("name")
        for node in nodes
        if node.get("classname", "").endswith("attention.unittests.test_glm5_remnant")
    }
    suites = list(tree.getroot().iter("testsuite"))
    if (
        not names
        or not names <= required
        or (not allow_partial and not required <= names)
        or any(
            node.find(k) is not None
            for node in nodes
            for k in ("failure", "error", "skipped")
        )
        or any(
            int(x.get(k, 0)) for x in suites for k in ("errors", "failures", "skipped")
        )
        or sum(int(x.get("tests", 0)) for x in suites) != len(names)
    ):
        raise RuntimeError(
            f"Incomplete GLM sanitizer fixture: {run_id}; missing={sorted(required - names)}"
        )
    for node in nodes:
        if node.get("name", "").startswith("test_benchmark_sanitizer_bank_graph["):
            dimensions = tuple(
                map(int, node.get("name").split("[")[1].rstrip("]").split("-"))
            )
            properties = {
                x.get("name"): x.get("value")
                for x in node.findall("properties/property")
            }
            geometry = json.loads(properties.get("probe_geometry", "{}"))
            if (
                tuple(
                    geometry.get(k)
                    for k in ("heads", "banks", "queries", "forced_inner")
                )
                != dimensions
                or geometry.get("warmup_calls") != 1
                or geometry.get("measured_warmup_calls") != 10
                or geometry.get("actual_warmup_calls_per_mode") != dimensions[1]
            ):
                raise RuntimeError(
                    f"Missing focused sanitizer geometry: {run_id}/{node.get('name')}"
                )
    return names


def check_sanitizer_report(output: str, tool: str) -> None:
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
