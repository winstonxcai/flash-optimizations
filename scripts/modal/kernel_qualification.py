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

from scripts.modal.qualification import (check_sanitizer_report,
                                         read_pytest_report, validate_run_id,
                                         verify_sanitizer_artifacts)

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


def glm_decode_combine_cases(manifest: dict) -> dict[str, dict]:
    """Map the exact prepared candidate shapes to their sanitizer case IDs."""
    variants = manifest["glm_compile"].get("decode_combine", {})
    if not variants:
        return {}
    stock_splits = {"66x128", "33x512"}
    tuned_splits = stock_splits | {"22x512", "11x512", "6x512"}
    if variants.keys() not in (stock_splits, tuned_splits):
        raise RuntimeError("Incomplete GLM decode combine manifest")
    return {
        f"test_packed_decode_combine_graph[{shape.replace('x', '-')}]": variant
        for shape, variant in variants.items()
    }


def glm_tuned_inner_splits(compiled: dict) -> set[int]:
    variants = glm_decode_combine_cases({"glm_compile": compiled})
    return {3, 6, 11} if len(variants) == 5 else set()


def glm_wgmma_cases(manifest: dict) -> dict[str, dict]:
    """Identify exact experimental partial and combine launches from the image."""
    variants = (manifest.get("upstream_compile") or {}).get("decode_wgmma", {})
    if not variants:
        return {}
    if set(variants) != {"8", "16", "24", "32", "64"}:
        raise RuntimeError("Incomplete GLM WGMMA decode variants")
    cases = {}
    for batch, variant in variants.items():
        geometry = variant["geometry"]
        partial = variant["partial"]
        combine = variant["combine"]
        if partial.get("symbol") != "glm_packed_partial_wgmma_kernel":
            raise RuntimeError(f"Unbound GLM WGMMA partial launch: {batch=}")
        cases[f"test_wgmma_decode_partial_graph[{batch}]"] = dict(
            partial_cubin_sha256=partial["cubin_sha256"],
            decode_geometry=geometry,
            short_split_ctas=str(
                geometry["partial_groups"] if 33 % geometry["inner_iter"] else 0
            ),
        )
        cases[f"test_wgmma_decode_combine_graph[{batch}]"] = dict(
            combine_cubin_sha256=combine["cubin_sha256"]
        )
    return cases


def verify_glm_tile32_variants(compiled: dict) -> None:
    expected = {
        (heads, inner, packed)
        for heads in (8, 16, 32, 64)
        for inner in (1, 2, 33, 66)
        for packed in (False, True)
    } | {(16, inner, packed) for inner in glm_tuned_inner_splits(compiled) for packed in (False, True)}
    actual = [
        (v["heads"], v["inner"], v["packed"])
        for v in compiled.get("tilelang", []) if v.get("block_i", 32) == 32
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError("Incomplete CPU-compiled GLM TileLang variants")


def glm_decode_scope_filter(*, combine: bool = False) -> str:
    return r"regex=glm_packed_(partial|bytes|decode_combine_kernel\b)" if combine else "regex=glm_packed_(partial|bytes)"


@dataclass(frozen=True)
class QualificationContext:
    """Explicit evidence inputs; no source checkout, image or GPU side effects."""

    results_root: Path
    manifest: dict
    runtime_hashes: dict[str, str]
    sanitizer_order: tuple[str, ...]


def qualification_run_ids(run_ids: str) -> list[str]:
    ids = run_ids.split(",")
    if not run_ids:
        raise ValueError("GLM timing requires explicit --qualification-runs IDs")
    for item in ids:
        validate_run_id(item)
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
    if phase not in ("all", "decode", "upstream-decode"):
        raise ValueError("GLM qualification phase must be all, decode or upstream-decode")
    manifest = context.manifest
    current = context.runtime_hashes
    combine_cases = glm_decode_combine_cases(manifest) if phase == "upstream-decode" else {}
    wgmma_cases = glm_wgmma_cases(manifest) if phase == "upstream-decode" else {}
    decode_cases = GLM_DECODE_REQUIRED | combine_cases.keys() | wgmma_cases.keys()
    if phase == "upstream-decode":
        decode_cases |= {
            f"test_decode_partial_graph[16-{inner}]"
            for inner in glm_tuned_inner_splits(manifest["glm_compile"])
        }
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
        # The upstream scope changes only benchmark/baseline sources. Keep
        # the existing Direct compiler and artifact equality gates intact.
        if phase in ("decode", "upstream-decode") and (
            (phase == "decode" and old.get("sources") != manifest.get("sources"))
            or old["glm_compile"].get("cache_files")
            != manifest["glm_compile"].get("cache_files")
            or old.get("nvcc") != manifest.get("nvcc")
        ):
            raise RuntimeError(
                f"GLM focused fixture/binary/source identity changed: {run_id}"
            )
        if wgmma_cases and (
            (old.get("upstream_compile") or {}).get("decode_wgmma")
            != manifest["upstream_compile"]["decode_wgmma"]
        ):
            raise RuntimeError(f"GLM WGMMA artifact identity changed: {run_id}")
        if wgmma_cases and old.get("sources") != manifest.get("sources"):
            raise RuntimeError(f"GLM WGMMA probe source identity changed: {run_id}")
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
        if combine_cases and result.get("sanitizer_scope") != "glm-decode":
            raise RuntimeError(f"Candidate combine requires the focused decode scope: {run_id}")
        for tool, status in result.get("sanitizer_results", {}).items():
            if status.get("status") in {"not_run", "not_selected"}:
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
            if combine_cases and status.get("report_schema") != "remnant-sanitizer-raw-v1":
                raise RuntimeError(f"Missing candidate sanitizer report hashes: {run_id}/{tool}")
            verify_sanitizer_artifacts(root, tool, status)
            check_sanitizer_report(log, tool)
            if phase in ("decode", "upstream-decode") and result.get("sanitizer_scope") in ("glm-decode", "glm-merge"):
                audit = result.get("scope_audit", {})
                audit_log = (root / "scope-audit.log").read_text()
                if (
                    audit.get("status") != "passed"
                    or audit.get("filter") != glm_decode_scope_filter(combine=bool(combine_cases))
                    or audit.get("log_sha256")
                    != hashlib.sha256(audit_log.encode()).hexdigest()
                    or audit.get("entrypoints")
                    != manifest["glm_compile"].get("scope_symbol_audit")
                ):
                    raise RuntimeError(
                        f"Focused instrumentation audit changed: {run_id}"
                    )
                check_sanitizer_report(audit_log, "racecheck")
                check_glm_scope_audit(
                    audit_log, run_id, combine=bool(combine_cases), wgmma=bool(wgmma_cases)
                )
            if phase in ("decode", "upstream-decode"):
                tests, _ = read_pytest_report(root / f"{tool}.xml")
                names = {x.attrib["name"] for x in tests}
                if (
                    not names
                    or not names <= (decode_cases | ({"test_prefill_partial_graph[64]", "test_prefill_partial_graph_h8"} if phase == "upstream-decode" else set()))
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
                    if name in wgmma_cases:
                        props = {
                            x.get("name"): x.get("value")
                            for x in node.findall("properties/property")
                        }
                        expected = wgmma_cases[name]
                        if "decode_geometry" in expected:
                            props["decode_geometry"] = json.loads(
                                props.get("decode_geometry", "{}")
                            )
                        if any(props.get(key) != value for key, value in expected.items()):
                            raise RuntimeError(
                                f"GLM WGMMA launch evidence changed: {run_id}/{name}"
                            )
                    if name in combine_cases:
                        props = {
                            x.get("name"): x.get("value")
                            for x in node.findall("properties/property")
                        }
                        if props.get("combine_cubin_sha256") != combine_cases[name]["cubin_sha256"]:
                            raise RuntimeError(f"Focused GLM combine cubin changed: {run_id}/{name}")
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
                            if v["heads"] == heads and v["inner"] == inner and v.get("block_i", 32) == 32
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
        decode_cases
        if phase in ("decode", "upstream-decode")
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


def check_glm_scope_audit(
    log: str, identity: str, *, combine: bool = False, wgmma: bool = False
) -> None:
    """Pinned tool trace proves helper warp and handoff CTA synchronization."""
    wgmma_barrier = wgmma and any(
        "glm_packed_partial_wgmma_kernel" in entry[0]
        for entry in re.finditer(
            r"Barrier called[^\n]*(?:\n={9}[ \t]+Device Frame:[^\n]*)*", log
        )
    )
    if (
        "glm_packed_bytes" not in log
        or (combine and not re.search(r"glm_packed_decode_combine_kernel\b", log))
        or (wgmma and not wgmma_barrier)
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
