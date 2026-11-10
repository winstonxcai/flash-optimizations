"""Model-free H100 validation, timing and Nsight Compute profiles.

Author: Winston Cai.

    modal run scripts/modal/kernels.py::prepare_kernels --model-family glm53
    modal run scripts/modal/kernels.py::validate_kernels --run-id <unique-id>
    modal run scripts/modal/kernels.py::benchmark_kernels --run-id <unique-id>
    modal run scripts/modal/kernels.py::profile_kernels --run-id <unique-id> --cases <cases>

Defaults follow the selected fork: v0.5.18 has eight decode cases (K512/K317),
v0.5.22 has ten (K512); both have 25 store cases and 30 timing rounds.
Default profile captures: 30 for v0.5.18, 36 for v0.5.22; three modes per case.
--keep-count accepts 64..448 in steps of 64; non-256 defaults to Native/Triton.
--modes selects supported implementations explicitly; Kernel remains 256-only.
--cases chooses comma-separated fork cases. Profiles are not acceptance timing.
--suite prefill-read compares Native/Packed192/Packed256 on identical fixtures,
with separate reconstruction/attention/complete stages and warm/rotating caches.
--prefill-reader fused selects the opt-in unpack/RoPE candidate for that suite.
GLM: REMNANT_INTEGRATION=v0.5.22 REMNANT_MODEL_FAMILY=glm53 selects the
320-byte codec, config-only preparation and CPU SM90 compilation. validate_kernels
requires --model-family glm53 and --stage numerical|sanitizer|corrective. Stages
share the caller's authorized aggregate ledger; a stage grants no new budget.
Set REMNANT_VALIDATION_SUITE=v41-fp4 for scoped model-free SM90 FP4 validation.
validate_kernels --sanitizer runs memcheck, racecheck and synccheck (v0.5.22 order: synccheck,
racecheck, memcheck). --sanitizer-tools a,b selects a subset, run in the order given (one tool
per budgeted job). result.json then has sanitizer_qualification="partial" even when status is
"passed"; only a passing run of all three tools records "complete".
python3 scripts/validation_tier.py [--base origin/main] says which validation a diff needs.
CPU build/preflight precedes H100; Modal caches source/binaries. Authorization required.
Limits are caps, not ETAs; no weights/retries. Results: remnant-stage2a-results/microbench/<run-id>/<action>.
Download: modal volume get remnant-stage2a-results microbench/<run-id> <local-dir>
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import modal

from scripts.modal.kernel_qualification import QualificationContext
from scripts.modal.kernel_qualification import \
    check_glm_scope_audit as _check_glm_scope_audit
from scripts.modal.kernel_qualification import (glm_decode_combine_cases,
                                                glm_decode_scope_filter,
                                                glm_wgmma_cases,
                                                qualification_run_ids,
                                                qualify_glm_timing,
                                                verify_glm_tile32_variants)
from scripts.modal.qualification import check_fp4_scope_audit
from scripts.modal.qualification import \
    check_sanitizer_report as _check_sanitizer_report
from scripts.modal.qualification import (qualify_fp4_runs, read_pytest_report,
                                         validate_run_id)
from scripts.model_profiles import model_profile
from scripts.sources import check_sources, integration_profile


def _resolve_image_sources(
    model_family: str, environment: Mapping[str, str]
) -> tuple[str | None, str | None]:
    """Resolve mutually exclusive reuse/rebuild settings before image creation."""
    if model_family != "glm53" and environment.get("REMNANT_VALIDATION_SUITE") != "v41-fp4":
        return None, None
    compiled = environment.get("REMNANT_COMPILED_IMAGE") or None
    rebuild = environment.get("REMNANT_GLM_CPU_REBUILD_FROM") or None
    if model_family != "glm53" and rebuild:
        raise ValueError("REMNANT_GLM_CPU_REBUILD_FROM requires glm53")
    if compiled and rebuild:
        raise ValueError(
            "REMNANT_COMPILED_IMAGE and REMNANT_GLM_CPU_REBUILD_FROM are mutually "
            "exclusive; unset the inherited setting before selecting an image"
        )
    return compiled, rebuild


INTEGRATION = integration_profile()
MODEL_FAMILY = os.environ.get("REMNANT_MODEL_FAMILY", "dsv4")
_UPSTREAM_DECODE = os.environ.get("REMNANT_GLM_UPSTREAM_DECODE") == "1"
if _UPSTREAM_DECODE and MODEL_FAMILY != "glm53":
    raise ValueError("Upstream decode requires REMNANT_MODEL_FAMILY=glm53")
if MODEL_FAMILY not in ("dsv4", "glm53") or (MODEL_FAMILY == "glm53" and INTEGRATION["version"] != "v0.5.22"):
    raise ValueError("GLM validation requires REMNANT_INTEGRATION=v0.5.22")
compiled_image_id, rebuild_image_id = _resolve_image_sources(MODEL_FAMILY, os.environ)
MODEL_PROFILE = model_profile(MODEL_FAMILY, INTEGRATION["version"])
GLM_TEST = "test/registered/attention/unittests/test_glm5_remnant.py"
GLM_CPU_TEST = "test/registered/unit/test_glm5_mla_cache_format.py"
GLM_BENCHMARK_CPU = "benchmark/remnant/test_microbench.py"
GLM_BENCHMARK_GRAPH = GLM_TEST + "::test_benchmark_bank_graph_ownership"
GLM_DECODE_GRAPH = GLM_TEST + "::test_decode_bank_graph"
GLM_DECODE_PARTIAL = GLM_TEST + "::test_decode_partial_graph"
GLM_SANITIZER_GRAPH = GLM_TEST + "::test_benchmark_sanitizer_bank_graph"
GLM_SMOKE = (GLM_TEST + "::test_producer_reconstruction", GLM_TEST + "::test_native_fp8_conversion", GLM_TEST + "::test_allocator_accounting", GLM_TEST + "::test_invalid_attention", GLM_TEST + "::test_partial_outputs",
             GLM_TEST + "::test_attention[3-8-8]",
             GLM_TEST + "::test_changing_input_graph", GLM_TEST + "::test_production_backend_constructor")
GLM_SANITIZER = (*GLM_SMOKE, GLM_TEST + "::test_attention[0-1-64]", GLM_TEST + "::test_pool_lifecycle")
GLM_MERGE_SANITIZER = (
    GLM_TEST + "::test_packed_loader_bytes", GLM_DECODE_PARTIAL,
    GLM_TEST + "::test_prefill_partial_graph[64]",
    GLM_TEST + "::test_prefill_partial_graph_h8",
)

VALIDATION_SUITE = os.environ.get("REMNANT_VALIDATION_SUITE", "v4")
if VALIDATION_SUITE not in ("v4", "v41-fp4", "v41-decode") or (
    VALIDATION_SUITE in ("v41-fp4", "v41-decode") and INTEGRATION["version"] != "v0.5.22"
):
    raise ValueError("REMNANT_VALIDATION_SUITE must be v4, or v41-fp4 with v0.5.22")
if MODEL_FAMILY == "glm53" and VALIDATION_SUITE != "v4":
    raise ValueError("GLM cannot use a DeepSeek FP4 validation suite")

# Match only production FP4 store/read and FlashMLA attention, including combine
# and FP4 prefill used by graph tests. Independent reference kernels stay outside
# this focused instrumentation; the default full qualification is unchanged.
FP4_KERNEL_PATTERN = (
    "(_store|_read|flash_fwd_splitkv_mla_fp8_sparse_kernel|"
    "flash_fwd_mla_combine_kernel|sparse_attn_fwd_kernel)"
)


REMOTE_REPO = Path("/opt/flash-optimizations")
REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPO
SGLANG_ROOT = Path("/sgl-workspace/sglang-remnant")
FLASHMLA_ROOT = Path("/opt/flashmla-remnant")
SGLANG_SOURCE = Path(
    os.getenv("REMNANT_SGLANG_SOURCE_OVERRIDE", str(REPO_ROOT / INTEGRATION["sglang"]))
)
FLASHMLA_SOURCE = Path(
    os.getenv(
        "REMNANT_FLASHMLA_SOURCE_OVERRIDE", str(REPO_ROOT / INTEGRATION["flashmla"])
    )
)
if not modal.is_local():
    SGLANG_SOURCE, FLASHMLA_SOURCE = SGLANG_ROOT, FLASHMLA_ROOT
if modal.is_local():
    from scripts.build_lock import acquire_for_modal_cli
    from scripts.sources import check_sources

    # Hold the lock until this `modal run` exits so image inputs are not edited mid-build.
    acquire_for_modal_cli(REPO_ROOT)
    if INTEGRATION["version"] == "v0.5.22" and any(
        actual.resolve() != (REPO_ROOT / INTEGRATION[name]).resolve()
        for name, actual in (("sglang", SGLANG_SOURCE), ("flashmla", FLASHMLA_SOURCE))
    ):
        raise RuntimeError(
            "The v0.5.22 serving recipe requires canonical pinned sources"
        )
    SOURCE_IDENTITIES = check_sources(
        REPO_ROOT,
        {
            name: path
            for name, path in (("sglang", SGLANG_SOURCE), ("flashmla", FLASHMLA_SOURCE))
            if f"REMNANT_{name.upper()}_SOURCE_OVERRIDE" in os.environ
        },
    )
else:
    SOURCE_IDENTITIES = json.loads(os.environ["REMNANT_SOURCE_IDENTITIES"])
RESULTS_ROOT = Path("/results")
NATIVE_LIBRARY = Path("/opt/remnant-native-store.so")
MANIFEST = Path("/opt/remnant-build.json")
BUILD_ROOT = Path("/opt/remnant-build")
SANITIZER_ROOT = Path("/opt/remnant-sanitizer")
UPSTREAM_FLASHMLA_ROOT = Path("/opt/flashmla-upstream-native")
NATIVE_CONTROL_MANIFEST = Path("/opt/remnant-native-control.json")
UPSTREAM_FLASHMLA_SHA = "6fbbf3a61285d4078b8ac0f2874e99b7b0b11866"
if modal.is_local() and VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
    value = os.environ.get("REMNANT_UPSTREAM_FLASHMLA_SOURCE")
    if not value:
        raise RuntimeError(
            "V4.1 qualification requires REMNANT_UPSTREAM_FLASHMLA_SOURCE: "
            f"a clean isolated checkout of {UPSTREAM_FLASHMLA_SHA} with pinned CUTLASS"
        )
    UPSTREAM_FLASHMLA_SOURCE = Path(value).resolve()
    for path, expected in (
        (UPSTREAM_FLASHMLA_SOURCE, UPSTREAM_FLASHMLA_SHA),
        (
            UPSTREAM_FLASHMLA_SOURCE / "csrc/cutlass",
            "147f5673d0c1c3dcf66f78d677fd647e4a020219",
        ),
    ):
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=30
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            timeout=30,
        ).strip()
        if actual != expected or dirty:
            raise RuntimeError(
                f"Native control source mismatch at {path}: expected {expected}, actual {actual}, dirty={bool(dirty)}"
            )
    SOURCE_IDENTITIES["upstream_native"] = {
        "commit": UPSTREAM_FLASHMLA_SHA,
        "path": str(UPSTREAM_FLASHMLA_SOURCE),
    }
SANITIZER = (
    str(SANITIZER_ROOT / "usr/local/cuda-13.3/compute-sanitizer/compute-sanitizer")
    if INTEGRATION["version"] == "v0.5.22"
    else "compute-sanitizer"
)
# Include the upstream control in CPU build time; reserve time for cleanup.
GPU_TIMEOUT_S = 3600
CPU_BUILD_TIMEOUT_S = 900 if _UPSTREAM_DECODE else 1200 if VALIDATION_SUITE in ("v41-fp4", "v41-decode") else 3600
VALIDATION_TIMEOUT_S = (
    int(
        os.environ.get(
            "REMNANT_VALIDATION_TIMEOUT_S",
            "5400" if VALIDATION_SUITE in ("v41-fp4", "v41-decode") else "2700",
        )
    )
    if INTEGRATION["version"] == "v0.5.22"
    else GPU_TIMEOUT_S
)
VALIDATION_TIMEOUT_CAP_S = 5400 if VALIDATION_SUITE in ("v41-fp4", "v41-decode") else 2700
if (
    INTEGRATION["version"] == "v0.5.22"
    and not 120 <= VALIDATION_TIMEOUT_S <= VALIDATION_TIMEOUT_CAP_S
):
    raise ValueError(
        f"REMNANT_VALIDATION_TIMEOUT_S must be between 120 and {VALIDATION_TIMEOUT_CAP_S}"
    )
CPU_BUILD_BUDGET_S = CPU_BUILD_TIMEOUT_S - 200
RUN_BUDGETS_S = {"timing": 850, "validation": 1100, "profile": 1800}
PROFILE_CAPTURE_TIMEOUT_S, PROFILE_EXPORT_TIMEOUT_S = 110, 45
PROFILE_MAX_BUDGET_S = GPU_TIMEOUT_S - 100
TEST_ROOT = "test/registered/attention/unittests/dsv4/"
TEST_MODULES = {
    "cache_format": "test/registered/unit/test_dsv4_c4_cache_format.py",
    **{
        name: TEST_ROOT + f"test_remnant_{name}.py"
        for name in [
            "pool",
            "pack",
            "backend",
            "compressor_store",
            "fused_store",
            "cuda_graph",
            "flashmla_direct",
        ]
    },
    "pack_kernel": "python/sglang/test/kernels/deepseek_v4/test_remnant_pack_kernel.py",
    "unpack_kernel": "python/sglang/test/kernels/deepseek_v4/test_remnant_unpack_kernel.py",
}
VALIDATION_TESTS = tuple(TEST_MODULES.values())
CPU_TESTS = tuple(
    TEST_MODULES[name]
    for name in ("cache_format", "pack", "backend", "compressor_store")
)
BENCHMARK_TESTS = ("benchmark/remnant/test_microbench.py", TEST_MODULES["fused_store"])
if INTEGRATION["version"] == "v0.5.22":
    TEST_MODULES["hicache"] = TEST_ROOT + "test_remnant_hicache.py"
    VALIDATION_TESTS += (TEST_MODULES["hicache"], BENCHMARK_TESTS[0])
    CPU_TESTS += (
        TEST_MODULES["pool"] + "::TestRemnantPool",
        BENCHMARK_TESTS[0] + "::test_packed_hicache_routes_all_planes_without_cuda",
        BENCHMARK_TESTS[0] + "::test_prefill_fixture_matches_combine_keyword_contract",
        "test/registered/unit/utils/test_nvtx_utils.py",
        "test/registered/unit/observability/test_req_time_stats.py",
    )
SANITIZER_TESTS = (
    TEST_MODULES["unpack_kernel"]
    + "::test_triton_native_pages_match_independent_reference",
    TEST_MODULES["cuda_graph"]
    + "::TestRemnantCudaGraph::test_store_and_native_reconstruction_replays_changed_inputs",
    TEST_MODULES["pack_kernel"] + "::test_remnant_triton_pack_matches_reference",
    TEST_MODULES["cuda_graph"]
    + "::TestRemnantCudaGraph::test_consecutive_mixed_layers_replay_with_stable_buffers",
    TEST_MODULES["cuda_graph"]
    + "::TestRemnantCudaGraph::test_fused_read_replays_changed_planes_indices_lengths_and_frequencies",
)
if INTEGRATION["version"] == "v0.5.22":
    SANITIZER_TESTS += (
        TEST_MODULES["flashmla_direct"]
        + "::TestRemnantFlashMLADirect::test_direct_matches_fused_adapter[kernel-512-8-64-topmag]",
        TEST_MODULES["flashmla_direct"]
        + "::TestRemnantFlashMLADirect::test_direct_matches_fused_adapter[kernel-512-8-128-bitmap-edges]",
        TEST_MODULES["flashmla_direct"]
        + "::TestRemnantFlashMLADirect::test_direct_decode_cuda_graph_replay[8-64]",
        TEST_MODULES["fused_store"]
        + "::test_graph_replay_with_changed_inputs_and_plans",
        TEST_MODULES["hicache"]
        + "::TestRemnantHiCache::test_backup_restore_preserves_all_planes_across_layers_and_pages",
    )
if MODEL_FAMILY == "glm53":
    VALIDATION_TESTS = (GLM_CPU_TEST, GLM_TEST)
    CPU_TESTS = (GLM_CPU_TEST, GLM_BENCHMARK_CPU + "::test_glm_matrix_order_and_baseline_identity", GLM_BENCHMARK_CPU + "::test_glm_grouped_causal_prefill_and_shared_prefix", GLM_BENCHMARK_CPU + "::test_glm_separate_decode_requests_and_page_mapping", GLM_BENCHMARK_CPU + "::test_glm_rotation_and_call_normalization", GLM_BENCHMARK_CPU + "::test_glm_shared_round_schedule_and_preconditioning", GLM_BENCHMARK_CPU + "::test_glm_cli_rejects_adapter_and_deepseek_options", GLM_BENCHMARK_CPU + "::test_glm_entrypoint_dispatches_before_deepseek_imports", GLM_BENCHMARK_CPU + "::test_glm_rejects_incomplete_timing", GLM_BENCHMARK_CPU + "::test_glm_cpu_oracle_preserves_fp8_rounding_and_empty_rows")
    CPU_TESTS += (GLM_BENCHMARK_CPU + "::test_glm_full_prefill_scope_and_resume",
                  GLM_BENCHMARK_CPU + "::test_glm_graph_probe_metadata")
    SANITIZER_TESTS = (*GLM_SANITIZER, GLM_SANITIZER_GRAPH)
    BENCHMARK_TESTS = ()

# Finish missing synchronization coverage before retrying the slower memcheck.
# Tool order changes neither the selected cases nor the all-three passing gate.
SANITIZER_ORDER = (
    ("synccheck", "racecheck", "memcheck")
    if INTEGRATION["version"] == "v0.5.22"
    else ("memcheck", "racecheck", "synccheck")
)
# Scoped FP4 qualification keeps the legacy suite intact. Collection and XML
# gates reject skips; these explicit lists are copied into each result manifest.
if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
    fp4_tests = TEST_ROOT + "test_remnant_fp4.py"
    VALIDATION_TESTS = (
        fp4_tests,
        TEST_MODULES["flashmla_direct"]
        + "::TestRemnantFlashMLADirect::test_direct_matches_fused_adapter[kernel-512-8-64-topmag]",
        TEST_MODULES["flashmla_direct"]
        + "::TestRemnantFlashMLADirect::test_direct_matches_fused_adapter[kernel-512-8-128-bitmap-edges]",
    )
    CPU_TESTS = (
        fp4_tests + "::test_fp4_startup_contract",
        fp4_tests + "::test_fp4_backend_dispatch",
        TEST_MODULES["cache_format"],
        "benchmark/remnant/test_microbench.py::test_v41_fp4_matrix_and_timing_gate",
        "benchmark/remnant/test_microbench.py::test_v41_native_reference_preserves_tail_and_rounds_fp8_ties",
        "benchmark/remnant/test_microbench.py::test_v41_three_path_ratios",
        "benchmark/remnant/test_microbench.py::test_v41_native_call_does_not_reuse_warmup_metadata",
    )
    SANITIZER_TESTS = tuple(
        fp4_tests + "::" + name
        for name in (
            "test_fp4_store_and_reader",
            "test_fp4_direct",
            "test_fp4_graph_and_stream",
            "test_fp4_compressor_boundary",
            "test_fp4_c2_decode_store",
            "test_fp4_fragment_masks",
        )
    )
    native_tests = (
        "benchmark/remnant/test_microbench.py::test_gpu_v41_upstream_native_comparison"
    )
    VALIDATION_TESTS += (native_tests,)
    SANITIZER_TESTS += (native_tests,)

# Decode-only qualification excludes producer, reconstruction and prefill execution.
# Each existing control case checks Native/dense/Packed output, graph replay and
# a nondefault stream against independent cache references on a production page.
if VALIDATION_SUITE == "v41-decode":
    VALIDATION_TESTS = (native_tests, fp4_tests + "::test_fp4_pool_allocations")
    SANITIZER_TESTS = (native_tests,)

V41_MERGE_SANITIZER = tuple(
    TEST_ROOT + "test_remnant_fp4.py::" + name
    for name in (
        'test_fp4_store_and_reader[random-256-False]',
        'test_fp4_store_and_reader[random-256-True]',
        'test_fp4_store_and_reader[boundaries-128-False]',
        'test_fp4_store_and_reader[boundaries-128-True]',
        'test_fp4_direct[128-64-False]',
        'test_fp4_direct[128-64-True]',
        'test_fp4_direct[256-64-False]',
        'test_fp4_direct[256-64-True]',
        'test_fp4_graph_and_stream[64-False]',
        'test_fp4_graph_and_stream[64-True]',
        'test_fp4_compressor_boundary[False-1]',
        'test_fp4_compressor_boundary[True-1]',
        'test_fp4_compressor_boundary[False-2]',
        'test_fp4_compressor_boundary[True-2]',
        'test_fp4_fragment_masks[64]',
    )
)


def _select_sanitizer_tools(spec: str, available: Sequence[str]) -> tuple[str, ...]:
    """Resolve a comma-separated `--sanitizer-tools` value; empty selects every tool in order.

    Each tool is a separate compute-sanitizer pass over the same case set, so a run
    may cover a subset (for example one tool per budgeted job). The caller records
    whether the selection was complete; a subset is never a full qualification.
    """
    if not spec.strip():
        return tuple(available)
    chosen = [name.strip() for name in spec.split(",")]
    unknown = sorted(set(chosen) - set(available))
    if unknown or "" in chosen or len(set(chosen)) != len(chosen):
        raise ValueError(
            f"sanitizer_tools must be distinct names from {', '.join(available)}; got {spec!r}"
        )
    return tuple(chosen)


NCU_SECTIONS = [
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "LaunchStats",
    "Occupancy",
    "WarpStateStats",
    "SourceCounters",
    "InstructionStats",
]
PREFILL_METRICS = (
    "gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,"
    "lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct"
)


# CPU image preparation; the recipe must be defined before Modal uses it.

_SGLANG_TEST_FILES = {
    test.partition("::")[0]
    for test in (*VALIDATION_TESTS, *CPU_TESTS, *BENCHMARK_TESTS, *SANITIZER_TESTS)
}
if VALIDATION_SUITE == "v41-decode":
    # The allocation and CPU fixtures import helpers from these modules.
    _SGLANG_TEST_FILES.update((TEST_MODULES["flashmla_direct"], TEST_MODULES["pool"]))


def _sglang_ignored(path: Path) -> bool:
    allowed = (
        path.parts[0] == "python"
        or path.parts[:2] == ("benchmark", "remnant")
        or path.as_posix() in _SGLANG_TEST_FILES
        or path.as_posix() == "test/registered/attention/unittests/conftest.py"
        or path.name.startswith("LICENSE")
    )
    return not allowed or _ignore(path)


def _flashmla_ignored(path: Path) -> bool:
    return not (path.parts[0] == "csrc" or path.name.startswith("LICENSE")) or _ignore(
        path
    )


def _source_hashes() -> dict[str, str]:
    hashes = {}
    roots = [("sglang", SGLANG_ROOT), ("flashmla", FLASHMLA_ROOT)]
    if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
        roots.append(("upstream_flashmla", UPSTREAM_FLASHMLA_ROOT))
    for label, root in roots:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            # Docker omits both .git directories and nested submodule pointer
            # files. Neither is a build input or part of source content identity.
            if (
                path.is_file()
                and ".git" not in relative.parts
                and not _ignore(relative)
            ):
                digest.update(str(relative).encode() + b"\0")
                digest.update(path.read_bytes())
        hashes[label] = digest.hexdigest()
    # Policy and shared metadata are build inputs too, even outside fork trees.
    for name in ("scripts/modal/kernel_qualification.py", "scripts/modal/qualification.py", "scripts/model_profiles.py"):
        hashes[name] = hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
    return hashes


def _check_cuda_binding_requirements() -> list[dict[str, str]]:
    """Check changed bindings and consumers' base requirements, not unrelated extras.

    The vendor image already fails global pip check on vision/video/Dynamo
    packages. Do not repair those here or ignore a conflicting CUDA consumer.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    targets = {"cuda-python", "cuda-bindings"}
    for name in targets:
        if importlib.metadata.version(name) != "13.0.3":
            raise RuntimeError(f"Serving image requires {name}==13.0.3")
    checks = []
    for distribution in importlib.metadata.distributions():
        owner = canonicalize_name(distribution.metadata["Name"])
        for specification in distribution.requires or ():
            requirement = Requirement(specification)
            dependency = canonicalize_name(requirement.name)
            if owner not in targets and dependency not in targets:
                continue
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = importlib.metadata.version(dependency)
            except importlib.metadata.PackageNotFoundError as error:
                raise RuntimeError(f"{owner}: missing {requirement}") from error
            if not requirement.specifier.contains(installed, prereleases=True):
                raise RuntimeError(
                    f"{owner}: requires {requirement}; installed {installed}"
                )
            checks.append(
                {
                    "consumer": owner,
                    "requirement": str(requirement),
                    "installed": installed,
                }
            )
    return sorted(checks, key=lambda item: (item["consumer"], item["requirement"]))


def _glm_launch_metadata(build: Path, deadline: float, environment: dict) -> None:
    # TileLang imports use the same CPU driver stubs as compilation/preflight.
    code = (
        "import json; from pathlib import Path; from sglang.benchmark.remnant.glm_artifacts import bind_partial_launch_metadata; "
        "p=Path('/opt/remnant-glm-compile.json'); d=json.loads(p.read_text()); "
        "bind_partial_launch_metadata(d); p.write_text(json.dumps(d,indent=2)+'\\n')"
    )
    _run_command(
        [sys.executable, "-c", code],
        build / "glm-launch-metadata.log",
        deadline,
        environment=environment,
    )


def _reuse_glm_cpu_artifacts(build: Path, deadline: float, binding_requirements: dict) -> bool:
    """Fast CPU preparation for harness-only changes, with strict binary binding."""
    from scripts.sources import glm_compile_owner_digest

    if os.environ.get("REMNANT_GLM_CPU_REBUILD_FROM"):
        return False
    snapshot = Path("/opt/remnant-glm-reuse-base.json")
    if not os.environ.get("REMNANT_COMPILED_IMAGE") or not snapshot.is_file():
        return False
    previous = json.loads(snapshot.read_text())
    if previous["compile_owner_digest"] != glm_compile_owner_digest(SGLANG_ROOT):
        return False
    for path, expected in {**previous["binaries"], **previous["glm_compile"]["cache_files"]}.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Harness-only CPU reuse found changed artifact: {path}")
    for package,key in (("torch","torch"),("triton","triton"),("tilelang","tilelang"),
                        ("cuda-python","cuda_python"),("cuda-bindings","cuda_bindings")):
        if importlib.metadata.version(package) != previous[key]:
            raise RuntimeError(f"Harness-only CPU reuse toolchain changed: {package}")
    if (hashlib.sha256(Path(SANITIZER).read_bytes()).hexdigest() != previous["sanitizer"]["binary_sha256"]
            or hashlib.sha256(Path("/opt/remnant-glm-config.json").read_bytes()).hexdigest() != previous["model_config"]["sha256"]):
        raise RuntimeError("Harness-only CPU reuse sanitizer/config artifact changed")
    if _run_command(["nvcc","--version"],build / "nvcc.log",deadline) != previous["nvcc"]:
        raise RuntimeError("Harness-only CPU reuse NVCC changed")
    compiled = previous["glm_compile"]
    verify_glm_tile32_variants(compiled)
    Path("/opt/remnant-glm-compile.json").write_text(json.dumps(compiled,indent=2)+"\n")
    environment = _cpu_preflight_environment(build)
    _glm_launch_metadata(build,deadline,environment)
    compiled = json.loads(Path("/opt/remnant-glm-compile.json").read_text())
    _run_command(_benchmark_command("--model-family","glm53","--suite","glm-attention","--check-imports"),
                 build / "glm-benchmark-imports.log",deadline,environment=environment)
    _run_pytest(CPU_TESTS,build,"cpu-tests",deadline,environment=environment)
    _run_pytest((GLM_TEST,),build,"collection",deadline,collect=True,environment=environment)
    previous.update(glm_compile=compiled, sources=_source_hashes(),source_identities=SOURCE_IDENTITIES,
        launcher_sha256=hashlib.sha256(Path("/opt/remnant-launcher.py").read_bytes()).hexdigest(),
        compiled_base_image=os.environ["REMNANT_COMPILED_IMAGE"],cpu_rebuild_from=None,
        cuda_binding_requirements=binding_requirements,
        cpu_import_driver_stub_sha256=hashlib.sha256((build / "cpu-driver-stubs/libcuda.so.1").read_bytes()).hexdigest(),
        artifact_reuse="Verified unchanged compiled sources and compile owners; harness-only overlay")
    MANIFEST.write_text(json.dumps(previous,indent=2)+"\n")
    return True


def _prepare_upstream_decode(build: Path, deadline: float) -> None:
    """CPU-only delta compilation after verified reuse of production binaries."""
    # The reuse preflight already owns build/cpu-driver-stubs. Give the
    # additional baseline compilation a separate scratch directory.
    build = build / "upstream"
    build.mkdir()
    environment = _cpu_preflight_environment(build)
    _run_command(
        [sys.executable, "-m", "sglang.benchmark.remnant.glm_upstream"],
        build / "upstream-sm90-compile.log", deadline, environment=environment,
    )
    _run_pytest(tuple(
        GLM_BENCHMARK_CPU + "::" + name for name in (
            "test_glm_upstream_decode_matrix_and_labels",
            "test_glm_upstream_source_and_helper_provenance",
            "test_glm_upstream_compile_artifacts_are_bound",
            "test_glm_upstream_result_identity_and_csv",
            "test_glm_upstream_wgmma_timing_requires_candidate_geometry_and_score",
            "test_glm_upstream_profile_requires_complete_matched_combine_sweep",
        )
    ), build, "upstream-cpu-tests", deadline, environment=environment)
    manifest = json.loads(MANIFEST.read_text())
    manifest["upstream_compile"] = json.loads(
        Path("/opt/remnant-glm-upstream-compile.json").read_text()
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def _fp4_sanitizer_prefix(tool: str, *, trace: bool = False) -> tuple[str, ...]:
    """Audit and tool runs cover the same production kernels in a mixed test."""
    if tool not in ("memcheck", "racecheck", "synccheck") or (trace and tool != "racecheck"):
        raise ValueError("Invalid FP4 sanitizer tool/trace selection")
    prefix = (
        SANITIZER, "--tool", tool, "--error-exitcode", "1",
        "--kernel-name", "regex=" + FP4_KERNEL_PATTERN, "--print-level", "info",
    )
    return prefix + (("--racecheck-trace-sync", "yes") if trace else ())


def _validation_sanitizer_order(value: str) -> tuple[str, ...]:
    """Prioritize pending tools without permitting missing/duplicate tools."""
    if not value:
        return SANITIZER_ORDER
    tools = tuple(value.split(","))
    if len(tools) != 3 or set(tools) != set(SANITIZER_ORDER):
        raise ValueError(
            "sanitizer_order must list memcheck,racecheck,synccheck exactly once"
        )
    return tools


def _build_upstream_native_control(
    build: Path, deadline: float, cpu_environment: dict[str, str]
) -> None:
    """Build the original selected upstream package on CPU. Author: Winston Cai.

    The public Python wrapper, scheduling and CUDA arithmetic come from upstream.
    The isolated control carries the recorded SM90 synchronization corrections. The upstream build flags select SM90a only; no SM100 target
    or model is loaded. Keep this package separate from the fork's sgl_kernel.
    """
    environment = dict(cpu_environment)
    environment.update(
        FLASH_MLA_DISABLE_SM100="1",
        FLASH_MLA_DISABLE_SM90="0",
        MAX_JOBS="4",
        NVCC_THREADS="1",
    )
    _run_command(
        [
            sys.executable,
            "-c",
            "import os, runpy, sys; "
            f"os.chdir({str(UPSTREAM_FLASHMLA_ROOT)!r}); "
            "sys.argv=['setup.py', 'build_ext', '--inplace']; "
            "runpy.run_path('setup.py', run_name='__main__')",
        ],
        build / "upstream-native-build.log",
        deadline,
        environment=environment,
    )
    matches = list((UPSTREAM_FLASHMLA_ROOT / "flash_mla").glob("cuda*.so"))
    if len(matches) != 1:
        raise RuntimeError(f"Upstream Native expected one cuda binary: {matches}")
    binary = matches[0].resolve()
    listing = _run_command(
        ["cuobjdump", "--list-elf", str(binary)],
        build / "upstream-native-architectures.log",
        deadline,
    )
    architectures = sorted(set(re.findall(r"sm_[0-9]+[a-z]?", listing)))
    if architectures != ["sm_90a"]:
        raise RuntimeError(f"Upstream Native has unexpected targets: {architectures}")
    python = UPSTREAM_FLASHMLA_ROOT / "flash_mla/flash_mla_interface.py"
    NATIVE_CONTROL_MANIFEST.write_text(
        json.dumps(
            {
                "source_commit": UPSTREAM_FLASHMLA_SHA,
                "source_sha256": _source_hashes()["upstream_flashmla"],
                "binary": str(binary),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "python": str(python),
                "python_sha256": hashlib.sha256(python.read_bytes()).hexdigest(),
                "architectures": architectures,
                "build_flags": {
                    key: environment[key]
                    for key in (
                        "FLASH_MLA_DISABLE_SM100",
                        "FLASH_MLA_DISABLE_SM90",
                        "MAX_JOBS",
                        "NVCC_THREADS",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )


def _build_image() -> None:
    """CPU image layer: compile existing SM90 targets, install, then preflight."""
    import torch
    from sglang.kernels.jit.utils import override_jit_cuda_arch

    # CPU-only orchestration boundary checks before any GPU allocation.
    expected_order = ("racecheck", "memcheck", "synccheck")
    if _validation_sanitizer_order(",".join(expected_order)) != expected_order:
        raise RuntimeError("Sanitizer order preflight changed the requested order")
    for invalid in (
        "memcheck",
        "racecheck,memcheck",
        "memcheck,memcheck,synccheck",
        "racecheck,memcheck,unknown",
    ):
        try:
            _validation_sanitizer_order(invalid)
        except ValueError:
            continue
        raise RuntimeError(
            f"Sanitizer order preflight accepted invalid input: {invalid}"
        )
    deadline = time.monotonic() + CPU_BUILD_BUDGET_S
    build = BUILD_ROOT
    # This directory belongs solely to CPU preparation. A verified reused image
    # may contain prior reports; compiler caches live outside it and are retained.
    if build.exists() and (os.environ.get("REMNANT_COMPILED_IMAGE") or os.environ.get("REMNANT_GLM_CPU_REBUILD_FROM")):
        shutil.rmtree(build)
    build.mkdir()
    if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
        # Run pure receipt and mocked dispatch tests in the existing CPU builder.
        # Child vendor Python lacks Modal's injected SDK/dependencies/context.
        # Keep these tests before core precompilation and GPU allocation.
        import unittest

        from scripts.modal import tests as launcher_tests

        suite = unittest.TestSuite(
            unittest.defaultTestLoader.loadTestsFromTestCase(case)
            for case in (
                launcher_tests.FP4QualificationTests,
                launcher_tests.FP4PreflightTests,
            )
        )
        for name in (
            "test_pytest_rejects_skipped_failed_and_empty_coverage",
            "test_sanitizer_reordering_preserves_all_tools_and_checkpoints_receipts",
            "test_sanitizer_failures_preserve_later_results_and_fail_the_run",
            "test_sanitizer_deadline_keeps_unstarted_tools_explicitly_incomplete",
            "test_sanitizer_prelaunch_timeout_keeps_tool_not_run",
        ):
            suite.addTest(launcher_tests.LauncherTests(name))
        report = build / "qualification-tests.log"
        with report.open("w") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise RuntimeError(
                f"CPU launcher contract tests failed: {report.read_text()}"
            )
    binding_requirements = (
        _check_cuda_binding_requirements()
        if INTEGRATION["version"] == "v0.5.22"
        else None
    )
    if MODEL_FAMILY == "glm53" and _reuse_glm_cpu_artifacts(build, deadline, binding_requirements):
        if _UPSTREAM_DECODE:
            _prepare_upstream_decode(build, deadline)
        return
    sanitizer_identity = None
    if INTEGRATION["version"] == "v0.5.22":
        # 2025.3.1 predates Hopper warpgroup/indirect-cluster-barrier fixes.
        # Extract only this standalone tool: do not replace CUDA, drivers or
        # serving binaries. Strict unsuppressed checks still decide acceptance.
        url = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-sanitizer-13-3_13.3.75-1_amd64.deb"
        expected = "ab5467d9473adfb528481d1b6166b9bc718ab668563a3466d0f2c5d8dd27aa4f"
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RuntimeError("Pinned Compute Sanitizer package content changed")
        archive = build / "sanitizer.deb"
        archive.write_bytes(payload)
        _run_command(
            ["dpkg-deb", "-x", str(archive), str(SANITIZER_ROOT)],
            build / "sanitizer-extract.log",
            deadline,
        )
        sanitizer_identity = {
            "path": SANITIZER,
            "url": url,
            "package_sha256": expected,
            "binary_sha256": hashlib.sha256(Path(SANITIZER).read_bytes()).hexdigest(),
            "version": _run_command(
                [SANITIZER, "--version"],
                build / "sanitizer-version.log",
                deadline,
            ),
        }
    if MODEL_FAMILY == "glm53" or VALIDATION_SUITE == "v41-fp4":
        help_text = _run_command([SANITIZER, "--help"], build / "sanitizer-help.log", deadline)
        if any(option not in help_text for option in ("kernel-name", "racecheck-trace-sync")):
            raise RuntimeError("Qualified sanitizer lacks focused filter/synchronization audit")
    cpu_environment = _cpu_preflight_environment(build)
    # Fail on a broken base-image driver linkage before paying for compilation.
    _run_command(
        [
            sys.executable,
            "-c",
            "import sglang, sgl_kernel; print(sgl_kernel.common_ops.__file__)",
        ],
        build / "base-imports.log",
        deadline,
        environment=cpu_environment,
    )
    aot = SGLANG_ROOT / "python/sglang/kernels/aot"
    # Content identity includes dirty submitted sources, not just Git commits.
    sources = _source_hashes()
    if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
        _build_upstream_native_control(build, deadline, cpu_environment)
    # The v0.5.22 serving Dockerfile already compiled these exact extensions.
    if INTEGRATION["version"] == "v0.5.22":
        package = Path(sysconfig.get_paths()["purelib"]) / "sgl_kernel"
        binaries = []
        for target in ("remnant_ops", "flashmla_ops"):
            matches = list(package.glob(f"{target}*.so"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Serving image expected one {target}, found {matches}"
                )
            binaries.append(matches[0])
    else:
        # 1. Compile only the two existing SM90 extension targets.
        options = {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_CUDA_FLAGS": "-lineinfo",
            "ENABLE_BELOW_SM90": "OFF",
            "SGL_KERNEL_ENABLE_FA3": "OFF",
            "SGL_KERNEL_COMPILE_THREADS": "1",
            "SGL_KERNEL_ENABLE_FLASHMLA_SM100": "OFF",
            "REMNANT_FLASHMLA_SOURCE_DIR": str(FLASHMLA_ROOT),
            "CMAKE_PREFIX_PATH": torch.utils.cmake_prefix_path,
            "CUDA_VERSION": "13.0",
        }
        configure = ["cmake", "-S", str(aot), "-B", str(build / "cmake")]
        configure += [f"-D{key}={value}" for key, value in options.items()]
        _run_command(configure, build / "configure.log", deadline)
        _run_command(
            ["cmake", "--build", str(build / "cmake")]
            + ["--target", "remnant_ops", "flashmla_ops", "--parallel", "2"],
            build / "compile.log",
            deadline,
        )
        # 2. Install matching binaries and the fork Python binding.
        package = Path(sysconfig.get_paths()["purelib"]) / "sgl_kernel"
        binaries = []
        for target in ("remnant_ops", "flashmla_ops"):
            matches = list((build / "cmake").rglob(f"{target}*.so"))
            if len(matches) != 1:
                raise RuntimeError(f"CPU build expected one {target}, found {matches}")
            destination = package / matches[0].name
            shutil.copy2(matches[0], destination)
            binaries.append(destination)
        shutil.copy2(aot / "python/sgl_kernel/flash_mla.py", package / "flash_mla.py")
    if MODEL_FAMILY != "glm53":
        # 3. Precompile the Native store using the fork's dependency-valid cache.
        from sglang.kernels.jit.utils.compile.cache import find_prebuilt
        from sglang.kernels.ops.attention.dsv4.compress import \
            _jit_compress_norm_rope_module

        with override_jit_cuda_arch(9, 0, "a"):
            native_options = {}
            if INTEGRATION["version"] == "v0.5.22":
                from sglang.kernels.ops.attention.dsv4.kv_layout import \
                    KVLayout

                native_options["layout"] = KVLayout.V4
            _jit_compress_norm_rope_module(
                torch.float32, 512, 64, 64, False, **native_options
            )
        cache = Path(os.environ["SGLANG_JIT_CACHE_DIR"])
        scopes = {p.parent.parent for p in cache.rglob("*.so")}
        libraries = {
            p
            for scope in scopes
            if (p := find_prebuilt(scope=scope, module_name=scope.parent.name)) is not None
        }
        if len(libraries) != 1:
            raise RuntimeError(
                f"Expected one dependency-valid Native library, found {libraries}"
            )
        shutil.copy2(libraries.pop(), NATIVE_LIBRARY)
        binaries.append(NATIVE_LIBRARY)
    if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
        # Compile these after resolving the compressor library: the JIT cache
        # now contains three valid modules rather than a single Native store.
        from sglang.kernels.ops.attention.dsv4.attn import _jit_fused_store_module
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        with override_jit_cuda_arch(9, 0, "a"):
            for page_size in (128, 256):
                _jit_fused_store_module(
                    "flashmla", torch.bfloat16, torch.int64, page_size, KVLayout.V4
                )
        for scope in {p.parent.parent for p in cache.rglob("*.so")}:
            library = find_prebuilt(scope=scope, module_name=scope.parent.name)
            if library is not None and library not in binaries:
                binaries.append(library)
        control = json.loads(NATIVE_CONTROL_MANIFEST.read_text())
        binaries.append(Path(control["binary"]))
    binary_architectures = {}
    if INTEGRATION["version"] == "v0.5.22":
        for binary in binaries:
            listing = _run_command(
                ["cuobjdump", "--list-elf", str(binary)],
                build / f"{binary.name}.architectures.log",
                deadline,
            )
            architectures = sorted(set(re.findall(r"sm_[0-9]+[a-z]?", listing)))
            if not architectures or set(architectures) - {"sm_90", "sm_90a"}:
                raise RuntimeError(
                    f"Unexpected binary architectures: {binary}: {architectures}"
                )
            binary_architectures[str(binary)] = architectures
    model_config = None
    if MODEL_FAMILY == "glm53":
        revision = MODEL_PROFILE.revision
        url = MODEL_PROFILE.config_url
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != MODEL_PROFILE.config_sha256:
            raise RuntimeError("Pinned GLM configuration content changed")
        Path("/opt/remnant-glm-config.json").write_bytes(payload)
        model_config = {"revision": revision, "sha256": digest, "url": url}
        _run_command([sys.executable, "-c", f"import runpy; tests=runpy.run_path({str(SGLANG_ROOT / GLM_TEST)!r}); tests['test_allocator_accounting'](); tests['test_compiled_launch_contract']()"],
                     build / "glm-allocator-cpu.log", deadline, environment=cpu_environment)
        _run_command([sys.executable, "-m", "sglang.benchmark.remnant.glm_artifacts"],
                     build / "glm-sm90-compile.log", deadline, environment=cpu_environment)
        compile_path = Path("/opt/remnant-glm-compile.json")
        compiled = json.loads(compile_path.read_text())
        entrypoints = {}
        probe_sources = {}
        current = {v["source_sha256"]: v for v in compiled["tilelang"]}
        for path, digest in compiled["cache_files"].items():
            if not path.endswith("/device_kernel.cu"):
                continue
            text = Path(path).read_text()
            names = sorted(set(re.findall(r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?(\w+)\s*\(", text)))
            if digest in current:
                expected = "glm_packed_partial" if current[digest]["packed"] else "main"
                if len(names) != 1 or not names[0].startswith(expected):
                    raise RuntimeError(f"Unexpected current TileLang entrypoint: {path}: {names}")
                entrypoints[path] = dict(sha256=digest, symbol=names[0])
            if any(name.startswith("glm_packed_bytes") for name in names):
                probe_sources[path] = dict(sha256=digest, symbol=names[0])
        missing = set(current) - {entry["sha256"] for entry in entrypoints.values()}
        if missing or not probe_sources:
            raise RuntimeError(
                f"Incomplete verified TileLang entrypoint sources: {sorted(missing)}, "
                f"bytes probe present={bool(probe_sources)}"
            )
        combines = compiled.get("decode_combine", {})
        compiled["scope_symbol_audit"] = dict(
            filter=glm_decode_scope_filter(combine=bool(combines)),
            partial_sources=entrypoints,
            probe_sources=probe_sources,
            excluded="Unchanged Native and framework kernels" if combines
            else "Unchanged Native, combine and framework kernels",
        )
        if combines:
            compiled["scope_symbol_audit"]["decode_combine"] = combines
        compile_path.write_text(json.dumps(compiled, indent=2) + "\n")
    else:
        model_config = None
        if INTEGRATION["version"] == "v0.5.22":
            revision = MODEL_PROFILE.revision
            url = MODEL_PROFILE.config_url
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != MODEL_PROFILE.config_sha256:
                raise RuntimeError("Pinned 0731 configuration content changed")
            Path("/opt/remnant-0731-config.json").write_bytes(payload)
            model_config = {"revision": revision, "sha256": digest, "url": url}
    if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
        # Compile every new core format/geometry on CPU before an H100 exists.
        code = """
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from sglang.kernels.ops.attention.dsv4.fp4_cache import _store, _read
from sglang.kernels.ops.attention.dsv4.c2_decode_pool import _c2_decode_pool_kernel
from sgl_kernel import flashmla_ops
import torch
assert hasattr(torch.ops.sgl_kernel, 'fp4_sparse_decode_fwd')
assert hasattr(torch.ops.sgl_kernel, 'fp4_sparse_prefill_fwd')
for p in (128,256):
    for packed in (False,True):
        stride=p*(224 if packed else 288)
        for rotate in (False,True):
            for n in ((2,4) if p == 128 and rotate else (2,)):
                src=ASTSource(_store, signature={'X':'*bf16','F':'*fp32' if rotate else '*bf16','LOC':'*i64','C':'*u8'}, constexprs={'XS':512,'FS':64 if rotate else 512,'P':p,'STRIDE':stride,'N':n,'PACKED':packed,'DATA_BYTES':192 if packed else 256,'VALUE_BYTES':128 if packed else 256,'ROTATE':rotate})
                triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':4,'enable_fp_fusion':False})
        src=ASTSource(_read,signature={'C':'*u8','IDS':'*i32','OUT':'*bf16'},constexprs={'OS':512,'P':p,'STRIDE':stride,'N':2,'PACKED':packed,'DATA_BYTES':192 if packed else 256,'VALUE_BYTES':128 if packed else 256})
        triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':4})
# The actual C2 decode producer keeps FP32 key/score halves in an interleaved
# request ring. Compile its exercised ABI before model-free H100 integration.
src=ASTSource(_c2_decode_pool_kernel, signature={'kv_ptr':'*fp32','score_ptr':'*fp32','pos_ptr':'*i64','raw_out_loc_ptr':'*i64','out_loc_ptr':'*i64','req_ptr':'*i64','state_kv_ptr':'*fp32','state_score_ptr':'*fp32','pooled_ptr':'*fp32','group_pos_ptr':'*i64','slots_ptr':'*i64','pad_row':'i32'}, constexprs={'RING_SIZE':2,'STATE_KV_STRIDE':1024,'STATE_SCORE_STRIDE':1024,'D':512,'BLOCK_D':512})
triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':4})
print('FP4 CPU SM90 precompilation passed')
"""
        _run_command(
            [sys.executable, "-c", code],
            build / "fp4-precompile.log",
            deadline,
            environment=cpu_environment,
        )
    # 4. Preflight imports, coverage and profiler availability on CPU.
    if MODEL_FAMILY == "dsv4":
        _run_command(
            _benchmark_command("--check-imports"),
            build / "imports.log",
            deadline,
            environment=cpu_environment,
        )
        _run_command(
            _benchmark_command("--suite", "prefill-read", "--check-imports"),
            build / "prefill-imports.log",
            deadline,
            environment=cpu_environment,
        )
    if MODEL_FAMILY == "glm53":
        _run_command(_benchmark_command("--model-family", "glm53", "--suite", "glm-attention", "--check-imports"),
                     build / "glm-benchmark-imports.log", deadline, environment=cpu_environment)
    if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
        _run_command(
            _benchmark_command("--suite", "v41-fp4-decode", "--check-imports"),
            build / "fp4-benchmark-imports.log",
            deadline,
            environment=cpu_environment,
        )
    _run_pytest(
        VALIDATION_TESTS,
        build,
        "collection",
        deadline,
        collect=True,
        environment=cpu_environment,
    )
    _run_pytest(CPU_TESTS, build, "cpu-tests", deadline, environment=cpu_environment)
    if BENCHMARK_TESTS:
        _run_pytest(
            BENCHMARK_TESTS,
            build,
            "benchmark-collection",
            deadline,
            collect=True,
            environment=cpu_environment,
        )
    if INTEGRATION["version"] == "v0.5.22":
        _run_pytest(
            SANITIZER_TESTS,
            build,
            "sanitizer-collection",
            deadline,
            collect=True,
            environment=cpu_environment,
        )
    if INTEGRATION["version"] != "v0.5.22":
        _check_profiler_tools(build, deadline)
    fp4_scope_symbols = None
    if VALIDATION_SUITE == "v41-fp4":
        cached = {
            path.stem: str(path)
            for path in Path(os.environ["TRITON_CACHE_DIR"]).rglob("*.cubin")
            if path.stem in ("_store", "_read")
        }
        if set(cached) != {"_store", "_read"}:
            raise RuntimeError("FP4 focused filter lacks CPU-compiled store/read symbols")
        flashmla = next(p for p in binaries if p.name.startswith("flashmla_ops"))
        symbols = _run_command(
            ["cuobjdump", "--dump-elf-symbols", str(flashmla)],
            build / "fp4-symbols.log", deadline,
        )
        required = ("flash_fwd_splitkv_mla_fp8_sparse_kernel",
                    "flash_fwd_mla_combine_kernel", "sparse_attn_fwd_kernel")
        if any(name not in symbols for name in required):
            raise RuntimeError("FP4 focused filter lacks compiled FlashMLA symbols")
        fp4_scope_symbols = {
            "filter": "regex=" + FP4_KERNEL_PATTERN, "triton_symbols": cached,
            "flashmla_symbols": required,
            "flashmla_symbol_log_sha256": hashlib.sha256(symbols.encode()).hexdigest(),
        }
    # Profiler tools are not a correctness gate for the serving-image qualification.
    manifest = {
        "model_family": MODEL_FAMILY,
        "fp4_scope_symbols": fp4_scope_symbols,
        "glm_compile": (json.loads(Path("/opt/remnant-glm-compile.json").read_text()) if MODEL_FAMILY == "glm53" else None),
        "validation_suite": VALIDATION_SUITE,
        "coverage": {
            "validation": VALIDATION_TESTS,
            "cpu": CPU_TESTS,
            "sanitizer": SANITIZER_TESTS,
        },
        "integration": INTEGRATION["version"],
        "base_image": INTEGRATION["base_image"],
        "compiled_base_image": os.environ.get("REMNANT_COMPILED_IMAGE"),
        "cpu_rebuild_from": os.environ.get("REMNANT_GLM_CPU_REBUILD_FROM"),
        "formatting_transition": (json.loads(Path("/opt/remnant-cpu-rebuild-base.json").read_text()).get("formatting_transition")
                                  if os.environ.get("REMNANT_GLM_CPU_REBUILD_FROM") else None),
        "launcher_sha256": hashlib.sha256(Path("/opt/remnant-launcher.py").read_bytes()).hexdigest(),
        "sources": sources,
        "source_identities": SOURCE_IDENTITIES,
        "model_config": model_config,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_python": (
            importlib.metadata.version("cuda-python")
            if INTEGRATION["version"] == "v0.5.22"
            else None
        ),
        "cuda_bindings": (
            importlib.metadata.version("cuda-bindings")
            if INTEGRATION["version"] == "v0.5.22"
            else None
        ),
        "cuda_binding_requirements": binding_requirements,
        "sanitizer": sanitizer_identity,
        "arch": "sm_90a",
        "binary_architectures": binary_architectures,
        "native_dtype": "bfloat16" if MODEL_FAMILY == "glm53" else "float32",
        "triton": importlib.metadata.version("triton") if MODEL_FAMILY == "glm53" else None,
        "tilelang": importlib.metadata.version("tilelang") if MODEL_FAMILY == "glm53" else None,
        "fp4_precompile_artifacts": (
            {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(os.environ["TRITON_CACHE_DIR"]).rglob("*")
                if p.is_file()
            }
            if VALIDATION_SUITE in ("v41-fp4", "v41-decode")
            else {}
        ),
        "upstream_native_control": (
            json.loads(NATIVE_CONTROL_MANIFEST.read_text())
            if VALIDATION_SUITE in ("v41-fp4", "v41-decode")
            else None
        ),
        "cpu_import_driver_stub_sha256": hashlib.sha256(
            (build / "cpu-driver-stubs/libcuda.so.1").read_bytes()
        ).hexdigest(),
        "binaries": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries
        },
        "nvcc": _run_command(["nvcc", "--version"], build / "nvcc.log", deadline),
        "ncu": (
            None
            if INTEGRATION["version"] == "v0.5.22"
            else _run_command(["ncu", "--version"], build / "ncu.log", deadline)
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    if _UPSTREAM_DECODE:
        _prepare_upstream_decode(build, deadline)


def _check_profiler_tools(build: Path, deadline: float | None = None) -> None:
    deadline = deadline if deadline is not None else time.monotonic() + 120
    sections = _run_command(
        ["ncu", "--list-sections"], build / "sections.log", deadline
    )
    if any(section not in sections for section in NCU_SECTIONS):
        raise RuntimeError(
            "CPU preflight: Nsight Compute lacks required sections; see sections.log"
        )
    metrics = _run_command(
        ["ncu", "--query-metrics-mode", "all", "--chips", "gh100"],
        build / "prefill-metrics.log",
        deadline,
    )
    missing = [name for name in PREFILL_METRICS.split(",")[1:] if name not in metrics]
    if missing:
        raise RuntimeError(f"CPU preflight: missing H100 traffic metrics {missing}")


app = modal.App("remnant-kernel-dev")
results_volume = modal.Volume.from_name(
    "remnant-stage2a-results", create_if_missing=True
)
triton_volume = modal.Volume.from_name("remnant-triton-cache", create_if_missing=True)
_ignore = modal.FilePatternMatcher(
    "**/.git/**",
    ".git",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/*.pyc",
    "**/*.so",
    "**/*.[oa]",
    "**/build/**",
    "**/dist/**",
)


transition = None
if compiled_image_id:
    from scripts.sources import compiled_source_digest

    expected = compiled_source_digest([SGLANG_SOURCE / "python/sglang/kernels", FLASHMLA_SOURCE / "csrc"])
    code = ("from scripts.sources import verify_compiled_image_sources, snapshot_glm_artifacts; "
            f"verify_compiled_image_sources({expected!r}, {[str(SGLANG_ROOT / 'python/sglang/kernels'), str(FLASHMLA_ROOT / 'csrc')]!r}, {INTEGRATION['version']!r}); snapshot_glm_artifacts({str(SGLANG_ROOT)!r})")
    kernel_base = (
        modal.Image.from_id(compiled_image_id)
        .env({"PYTHONPATH": f"{REMOTE_REPO}:{SGLANG_ROOT / 'python'}"})
        .add_local_file(REPO_ROOT / "scripts/sources.py", str(REMOTE_REPO / "scripts/sources.py"), copy=True)
        .run_commands("python3 -c " + shlex.quote(code))
        .add_local_dir(SGLANG_SOURCE, str(SGLANG_ROOT), copy=True, ignore=_ignore)
        .dockerfile_commands(f"LABEL remnant.integration={INTEGRATION['version']} remnant.sglang={SOURCE_IDENTITIES['sglang']['commit']} remnant.flashmla={SOURCE_IDENTITIES['flashmla']['commit']}")
    )
elif rebuild_image_id:
    # Explicit CPU rebuild, never a relaxed ordinary source-reuse check.
    from scripts.sources import glm_rebuild_source_digest

    transition_path = os.environ.get("REMNANT_GLM_FORMAT_TRANSITION")
    transition_json = os.environ.get("REMNANT_GLM_FORMAT_TRANSITION_JSON")
    transition = (
        json.loads(transition_json)
        if transition_json
        else json.loads(Path(transition_path).read_text())
        if transition_path
        else None
    )
    if (
        transition
        and transition["base_image"] != rebuild_image_id
    ):
        raise RuntimeError("GLM formatting transition names a different base image")
    roots = [SGLANG_SOURCE / "python/sglang/kernels", FLASHMLA_SOURCE / "csrc"]
    candidate = glm_rebuild_source_digest(
        roots, tuple(transition["files"]) if transition else ()
    )
    rebuild_base_sglang = os.environ.get(
        "REMNANT_GLM_CPU_REBUILD_BASE_SGLANG",
        "889edf56b33226c80b17ddf3472c1206df62c3f0",
    )
    if not re.fullmatch(r"[0-9a-f]{40}", rebuild_base_sglang):
        raise ValueError("CPU rebuild base SGLang identity must be a full commit hash")
    base = (
        transition["base_commits"]
        if transition
        else dict(
            sglang=rebuild_base_sglang,
            flashmla=SOURCE_IDENTITIES["flashmla"]["commit"],
        )
    )
    remote_roots = [
        str(SGLANG_ROOT / "python/sglang/kernels"),
        str(FLASHMLA_ROOT / "csrc"),
    ]
    code = (
        "from scripts.sources import verify_glm_cpu_rebuild; "
        f"verify_glm_cpu_rebuild({base!r}, {candidate!r}, {remote_roots!r}, {transition!r})"
    )
    kernel_base = (
        modal.Image.from_id(rebuild_image_id)
        .env({"PYTHONPATH": f"{REMOTE_REPO}:{SGLANG_ROOT / 'python'}"})
        .add_local_file(
            REPO_ROOT / "scripts/sources.py",
            str(REMOTE_REPO / "scripts/sources.py"),
            copy=True,
        )
    )
    if transition:
        kernel_base = kernel_base.pip_install("clang-format==20.1.7")
    kernel_base = kernel_base.run_commands(
        "python3 -c " + shlex.quote(code)
    ).add_local_dir(SGLANG_SOURCE, str(SGLANG_ROOT), copy=True, ignore=_ignore)
    if transition:
        proof = (
            "from scripts.sources import verify_glm_formatting; "
            f"verify_glm_formatting({transition!r}, {remote_roots!r}, candidate=True)"
        )
        kernel_base = kernel_base.run_commands("python3 -c " + shlex.quote(proof))
    kernel_base = kernel_base.dockerfile_commands(
        f"LABEL remnant.integration={INTEGRATION['version']} remnant.sglang={SOURCE_IDENTITIES['sglang']['commit']} remnant.flashmla={SOURCE_IDENTITIES['flashmla']['commit']}"
    )
elif INTEGRATION["version"] == "v0.5.22":
    kernel_base = modal.Image.from_dockerfile(
        str(REPO_ROOT / INTEGRATION["dockerfile"]),
        context_dir=str(REPO_ROOT),
        build_args={
            "REMNANT_BASE_IMAGE": INTEGRATION["base_image"],
            "REMNANT_INTEGRATION": INTEGRATION["version"],
            "REMNANT_SGLANG_SOURCE": INTEGRATION["sglang"],
            "REMNANT_FLASHMLA_SOURCE": INTEGRATION["flashmla"],
            "REMNANT_SGLANG_SHA": SOURCE_IDENTITIES["sglang"]["commit"],
            "REMNANT_FLASHMLA_SHA": SOURCE_IDENTITIES["flashmla"]["commit"],
        },
        ignore=modal.FilePatternMatcher.from_file(REPO_ROOT / ".dockerignore"),
    )
else:
    kernel_base = (
        modal.Image.from_registry(INTEGRATION["base_image"])
        .pip_install("typing_extensions==4.15.0")
        .add_local_dir(
            SGLANG_SOURCE, str(SGLANG_ROOT), copy=True, ignore=_sglang_ignored
        )
        .add_local_dir(
            FLASHMLA_SOURCE, str(FLASHMLA_ROOT), copy=True, ignore=_flashmla_ignored
        )
    )

if _UPSTREAM_DECODE:
    if modal.is_local():
        source_path = SGLANG_SOURCE / "python/sglang/benchmark/remnant/glm_upstream.py"
        spec = importlib.util.spec_from_file_location("remnant_upstream_prepare", source_path)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        export = REPO_ROOT / "data/producer-artifacts/glm-upstream-v0522"
        baseline.prepare_source(SGLANG_SOURCE, export)
        kernel_base = kernel_base.add_local_dir(export, baseline.IMAGE_SOURCE, copy=True)
    kernel_base = kernel_base.env({
        "REMNANT_GLM_UPSTREAM_DECODE": "1",
        "REMNANT_GLM_UPSTREAM_SOURCE": "/opt/remnant-glm-upstream-v0522",
    })

if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
    kernel_base = kernel_base.add_local_dir(
        UPSTREAM_FLASHMLA_SOURCE if modal.is_local() else UPSTREAM_FLASHMLA_ROOT,
        str(UPSTREAM_FLASHMLA_ROOT),
        copy=True,
        ignore=_ignore,
    )

kernel_image = (
    # Legacy image builders inject the SDK without its Python dependencies.
    # Install the submitting version before CPU launcher tests import Modal.
    kernel_base.pip_install(f"modal=={modal.__version__}").env(
        {
            "REMNANT_VALIDATION_SUITE": VALIDATION_SUITE,
            "REMNANT_INTEGRATION": INTEGRATION["version"],
            "REMNANT_BUILT_INTEGRATION": INTEGRATION["version"],
            "REMNANT_SOURCE_IDENTITIES": json.dumps(SOURCE_IDENTITIES),
            "PYTHONPATH": (
                "/opt/sglang-runtime-fixes:"
                if INTEGRATION["version"] == "v0.5.22"
                else ""
            )
            + f"{SGLANG_ROOT / 'python'}:{REMOTE_REPO}",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "REMNANT_NATIVE_CONTROL_MANIFEST": str(NATIVE_CONTROL_MANIFEST),
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "SGLANG_JIT_CACHE_DIR": "/opt/remnant-jit",
            "REMNANT_FLASHMLA_SOURCE_DIR": str(FLASHMLA_ROOT),
            "REMNANT_NATIVE_STORE_LIBRARY": str(NATIVE_LIBRARY),
            "REMNANT_BUILD_MANIFEST_PATH": str(MANIFEST),
            "REMNANT_VALIDATION_TIMEOUT_S": str(VALIDATION_TIMEOUT_S),
            "REMNANT_MODEL_CONFIG": "/opt/remnant-0731-config.json",
            **(
                {"REMNANT_VALIDATION_TIMEOUT_S": str(VALIDATION_TIMEOUT_S)}
                if INTEGRATION["version"] == "v0.5.22"
                else {}
            ),
            "TRITON_CACHE_DIR": "/opt/remnant-triton-glm" if MODEL_FAMILY == "glm53" else "/triton-cache",
            "REMNANT_MODEL_FAMILY": MODEL_FAMILY,
            "REMNANT_COMPILED_IMAGE": compiled_image_id or "",
            "REMNANT_GLM_CPU_REBUILD_FROM": rebuild_image_id or "",
            "REMNANT_GLM_MODEL_CONFIG": "/opt/remnant-glm-config.json",
            "REMNANT_GLM_COMPILE_MANIFEST": "/opt/remnant-glm-compile.json",
            "REMNANT_GLM_FORMAT_TRANSITION_JSON": json.dumps(transition) if transition else "",
            **({"TILELANG_DEFAULT_TARGET": json.dumps({"kind": "cuda", "arch": "sm_90a"}), "TILELANG_CACHE_DIR": "/opt/remnant-tilelang-glm"} if MODEL_FAMILY == "glm53" else {}),
        }
    )
    .add_local_file(
        REPO_ROOT / "scripts/model_profiles.py",
        str(REMOTE_REPO / "scripts/model_profiles.py"), copy=True,
    )
    # Include launcher helpers in the image cache key, not just _build_image.
    .add_local_file(
        REPO_ROOT / "scripts/modal/kernels.py", "/opt/remnant-launcher.py", copy=True
    )
    .add_local_file(
        REPO_ROOT / "scripts/sources.py",
        str(REMOTE_REPO / "scripts/sources.py"),
        copy=True,
    )
    .add_local_file(
        REPO_ROOT / "scripts/modal/kernel_qualification.py",
        str(REMOTE_REPO / "scripts/modal/kernel_qualification.py"), copy=True,
    )
    .add_local_file(
        REPO_ROOT / ".dockerignore", str(REMOTE_REPO / ".dockerignore"), copy=True
    )
    .add_local_file(
        REPO_ROOT / "scripts/modal/kernels.py",
        str(REMOTE_REPO / "scripts/modal/kernels.py"), copy=True
    )
    .add_local_file(
        REPO_ROOT / "scripts/modal/qualification.py",
        str(REMOTE_REPO / "scripts/modal/qualification.py"), copy=True
    )
    .add_local_file(
        REPO_ROOT / "scripts/modal/tests.py",
        str(REMOTE_REPO / "scripts/modal/tests.py"), copy=True
    )
    .run_function(_build_image, cpu=16, memory=65536, timeout=CPU_BUILD_TIMEOUT_S)
)
_GPU = {
    "image": kernel_image,
    "gpu": "H100!",
    "cpu": 4 if MODEL_FAMILY == "glm53" else 8,
    "memory": 32768 if MODEL_FAMILY == "glm53" else 65536,
    "timeout": 1620 if MODEL_FAMILY == "glm53" else GPU_TIMEOUT_S,
    "retries": 0,
    "max_containers": 1,
    "scaledown_window": 2,
    # FP4 precompilation populates this image path on CPU. Mounting a volume
    # over a nonempty path fails Modal startup; retain its verified image cache.
    "volumes": {
        str(RESULTS_ROOT): results_volume,
        **({"/triton-cache": triton_volume} if VALIDATION_SUITE not in ("v41-fp4", "v41-decode") else {}),
    },
}


@app.local_entrypoint()
def prepare_kernels(model_family: str = "dsv4") -> None:
    """Prepare the selected image on CPU; never call a GPU Function."""
    if model_family != MODEL_FAMILY:
        raise ValueError("--model-family must match REMNANT_MODEL_FAMILY")
    # Registered Functions hydrate/build this image before the local entrypoint.
    # Building again here needlessly resolves the entire image graph twice.
    image = kernel_image
    print(json.dumps({"model_family": MODEL_FAMILY, "image_id": image.object_id,
                      "source_identities": SOURCE_IDENTITIES}, sort_keys=True))


# Public workflows.


_GLM_PREFILL_H200 = os.environ.get("REMNANT_GLM_PREFILL_H200") == "1"
_GLM_PREFILL_TIMEOUT_S = 2400
if _UPSTREAM_DECODE and _GLM_PREFILL_H200:
    raise ValueError("Upstream decode and prefill image selection are exclusive")
if _GLM_PREFILL_H200 and MODEL_FAMILY != "glm53":
    raise ValueError("REMNANT_GLM_PREFILL_H200 requires GLM")

@app.function(image=kernel_image, cpu=1, memory=2048, timeout=60, retries=0)
def preflight_kernels() -> dict:
    """Return the completed CPU build identity without GPU allocation. Author: Winston Cai."""
    manifest = json.loads(MANIFEST.read_text())
    if manifest["sources"] != _source_hashes():
        raise RuntimeError("CPU preflight source identity changed")
    for path, expected in manifest["binaries"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"CPU preflight binary changed: {path}")
    return {"status": "passed", "gpu": False, "manifest": manifest}


def _fp4_timing_qualification(run_ids: str) -> dict:
    """Author: Winston Cai. Aggregate only complete tools with identical inputs."""
    current = json.loads(MANIFEST.read_text())
    if (
        current["sources"] != _source_hashes()
        or current["source_identities"] != SOURCE_IDENTITIES
    ):
        raise RuntimeError("Timing source identity mismatch")
    for path, digest in current["binaries"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Timing binary changed: {path}")
    return qualify_fp4_runs(RESULTS_ROOT, run_ids, current)


@app.function(
    image=kernel_image,
    cpu=2,
    memory=4096,
    timeout=180,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={str(RESULTS_ROOT): results_volume},
)
def _preflight_fp4_timing(run_id: str, qualification_runs: str) -> dict:
    """Check receipts and the matrix contract on CPU before reserving an H100."""
    results_volume.reload()
    with _run_context(run_id, "preflight", 120, resource="cpu") as (
        root,
        record,
        deadline,
    ):
        record["qualification"] = _fp4_timing_qualification(qualification_runs)
        _run_pytest(
            (
                "benchmark/remnant/test_microbench.py::test_v41_fp4_matrix_and_timing_gate",
            ),
            root,
            "contract",
            deadline,
        )
    return record["qualification"]


@app.function(
    image=kernel_image, cpu=2, memory=4096, timeout=180, retries=0,
    max_containers=1, scaledown_window=2,
    volumes={str(RESULTS_ROOT): results_volume},
)
def _preflight_glm_timing(
    run_id: str, qualification_runs: str, phase: str,
    allow_incomplete: bool, resume_run: str, suite: str = "glm-attention",
    profile_decode: bool = False,
) -> dict | None:
    """Check GLM image and evidence before scheduling its GPU worker."""
    if profile_decode and (suite != "glm-upstream-decode" or phase != "decode"
                           or qualification_runs or allow_incomplete or resume_run):
        raise ValueError("Decode profiling requires a fresh diagnostic run")
    results_volume.reload()
    if resume_run:
        _glm_timing_checkpoint(resume_run)
    if phase in ("prefill", "prefill-small"):
        if qualification_runs:
            raise ValueError("Fresh prefill qualifies its selected cubins in the worker")
        required_heads = (16, 8) if phase == "prefill" else (16,)
        for heads in required_heads:
            if phase == "prefill":
                _glm_prefill_variant(heads, 64)
        return None
    if suite == "glm-upstream-decode":
        from sglang.benchmark.remnant.glm_upstream import verify_compilation

        manifest = _verify_glm_compiled_artifacts()
        verify_compilation(
            manifest.get("upstream_compile", {}),
            require_wgmma=True,
        )
    with _run_context(run_id, "preflight", 120, resource="cpu") as (_, record, _):
        if profile_decode:
            record["qualification"] = {"status": "diagnostic", "artifact_verification": "passed"}
            return record["qualification"]
        record["qualification"] = _glm_timing_qualification(
            qualification_runs, allow_incomplete=allow_incomplete,
            phase="upstream-decode" if suite == "glm-upstream-decode" else phase
        )
    return record["qualification"]


def _dispatch_benchmark(
    run_id: str, keep_count: int, modes: str, suite: str,
    prefill_reader: str, qualification_runs: str,
    *, model_family: str = "dsv4", max_seconds: int = RUN_BUDGETS_S["timing"],
    resume_run: str = "", allow_incomplete_qualification: bool = False,
    phase: str = "all", packed_prefill_tile: int = 32,
    profile_decode: bool = False,
) -> str:
    if model_family != MODEL_FAMILY:
        raise ValueError("--model-family must match REMNANT_MODEL_FAMILY")
    preflight = None
    upstream = suite == "glm-upstream-decode"
    if profile_decode and (not upstream or resume_run or qualification_runs):
        raise ValueError("Decode profiling requires a fresh upstream decode run")
    if upstream and (not _UPSTREAM_DECODE or model_family != "glm53" or phase != "decode"
                     or packed_prefill_tile != 32 or allow_incomplete_qualification):
        raise ValueError("Upstream suite requires its H200 image, GLM decode, production tile32 and qualification")
    if _UPSTREAM_DECODE and not upstream:
        raise ValueError("Upstream image requires suite glm-upstream-decode")
    if model_family == "glm53":
        if (
            suite not in ("glm-attention", "glm-upstream-decode") or modes not in ("", "upstream,direct" if upstream else "native,direct")
            or keep_count != 256 or prefill_reader != "split"
            or phase not in ("all", "decode", "prefill", "prefill-small")
        ):
            raise ValueError("GLM requires its attention suite and Native/Direct paths")
        if not upstream and (phase in ("prefill", "prefill-small")) != _GLM_PREFILL_H200:
            raise ValueError("GLM prefill scopes require the H200 image selection")
        if allow_incomplete_qualification and phase != "all":
            raise ValueError("This GLM scope requires qualification")
        if phase == "prefill" and packed_prefill_tile != 64:
            raise ValueError("Full GLM prefill requires tile64")
        limit = _GLM_PREFILL_TIMEOUT_S - 60 if phase == "prefill" else 1140
        if not 60 <= max_seconds <= limit:
            raise ValueError(f"GLM timing max_seconds must be between 60 and {limit}")
        preflight = _preflight_glm_timing.remote(
            run_id, qualification_runs, phase, allow_incomplete_qualification, resume_run,
            **({"suite": suite} if upstream else {}),
            **({"profile_decode": True} if profile_decode else {}),
        )
    elif suite == "v41-fp4-decode":
        if (
            VALIDATION_SUITE not in ("v41-fp4", "v41-decode")
            or keep_count != 256
            or modes not in ("", "native,dense_fp4,kernel")
            or prefill_reader != "split"
        ):
            raise ValueError(
                "FP4 timing requires v41-fp4 qualification and upstream Native/custom dense FP4/Packed keep256"
            )
        preflight = _preflight_fp4_timing.remote(run_id, qualification_runs)
    elif qualification_runs:
        raise ValueError("qualification_runs requires GLM or v41-fp4-decode")
    return _benchmark_worker.remote(
        run_id, keep_count, modes, suite, prefill_reader, qualification_runs, preflight,
        model_family, max_seconds, resume_run, allow_incomplete_qualification,
        phase, packed_prefill_tile, profile_decode,
    )


@app.local_entrypoint()
def benchmark_kernels(
    run_id: str, keep_count: int = 256, modes: str = "", suite: str = "all",
    prefill_reader: str = "split", qualification_runs: str = "",
    model_family: str = "dsv4", max_seconds: int = RUN_BUDGETS_S["timing"],
    resume_run: str = "", allow_incomplete_qualification: bool = False,
    phase: str = "all", packed_prefill_tile: int = 32,
    profile_decode: bool = False,
) -> None:
    """Public CLI: validate the requested suite before scheduling its worker."""
    print(_dispatch_benchmark(
        run_id, keep_count, modes, suite, prefill_reader, qualification_runs,
        model_family=model_family, max_seconds=max_seconds,
        resume_run=resume_run, allow_incomplete_qualification=allow_incomplete_qualification,
        phase=phase, packed_prefill_tile=packed_prefill_tile, profile_decode=profile_decode,
    ))


@app.function(**{
    **_GPU,
    "gpu": "H200" if _UPSTREAM_DECODE or _GLM_PREFILL_H200 else _GPU["gpu"],
    "image": kernel_image.env({"REMNANT_GLM_PREFILL_H200": "1"})
        if _GLM_PREFILL_H200 else kernel_image,
    "timeout": 1320 if _UPSTREAM_DECODE else _GLM_PREFILL_TIMEOUT_S if _GLM_PREFILL_H200
        else (1200 if MODEL_FAMILY == "glm53" else 900),
    **({"cpu": 2, "memory": 16384} if VALIDATION_SUITE in ("v41-fp4", "v41-decode") else {}),
})
def _benchmark_worker(
    run_id: str,
    keep_count: int = 256,
    modes: str = "",
    suite: str = "all",
    prefill_reader: str = "split",
    qualification_runs: str = "",
    preflight: dict | None = None,
    model_family: str = "dsv4", max_seconds: int = RUN_BUDGETS_S["timing"],
    resume_run: str = "", allow_incomplete_qualification: bool = False,
    phase: str = "all", packed_prefill_tile: int = 32,
    profile_decode: bool = False,
) -> str:
    """Validate and time the complete matrix for the selected survivor budget."""
    if model_family != MODEL_FAMILY:
        raise ValueError("--model-family must match REMNANT_MODEL_FAMILY")
    upstream = suite == "glm-upstream-decode"
    if profile_decode and (not upstream or resume_run or qualification_runs):
        raise ValueError("Decode profiling requires a fresh upstream decode run")
    if upstream and (not _UPSTREAM_DECODE or model_family != "glm53" or phase != "decode"
                     or packed_prefill_tile != 32 or allow_incomplete_qualification):
        raise ValueError("Upstream suite requires its H200 image, GLM decode, production tile32 and qualification")
    prefill_h200 = phase in ("prefill", "prefill-small")
    if phase == "prefill":
        if model_family != "glm53" or packed_prefill_tile != 64:
            raise ValueError("Full GLM prefill requires --packed-prefill-tile 64")
    elif packed_prefill_tile != 32 and (
        packed_prefill_tile not in (64, 128) or model_family != "glm53" or phase != "prefill-small" or resume_run
    ):
        raise ValueError("Wide Packed tile requires prefill or a fresh GLM prefill-small run")
    if model_family == "glm53":
        if phase not in ("all", "decode", "prefill", "prefill-small"):
            raise ValueError("GLM phase must be all, decode, prefill, or prefill-small")
        if not upstream and prefill_h200 != _GLM_PREFILL_H200:
            raise ValueError("GLM prefill scopes require REMNANT_GLM_PREFILL_H200=1")
        if (phase == "decode" or prefill_h200) and allow_incomplete_qualification:
            raise ValueError(f"{phase} timing requires qualification")
        if suite not in ("glm-attention", "glm-upstream-decode") or keep_count != 256 or modes not in ("", "upstream,direct" if upstream else "native,direct") or prefill_reader != "split":
            raise ValueError("GLM timing requires --suite glm-attention and exactly native,direct")
        limit = _GLM_PREFILL_TIMEOUT_S - 60 if phase == "prefill" else 1140
        if not 60 <= max_seconds <= limit:
            raise ValueError(f"GLM timing max_seconds must be between 60 and {limit}")
        stage = "decode-profile" if profile_decode else "timing"
        with _run_context(run_id, stage, max_seconds) as (root, record, deadline):
            _verify_glm_image()
            if profile_decode:
                if preflight != {"status": "diagnostic", "artifact_verification": "passed"}:
                    raise RuntimeError("Missing CPU artifact verification for decode profiling")
                record["qualification"] = preflight
            elif phase == "prefill-small":
                if qualification_runs or resume_run:
                    raise ValueError("Small prefill qualifies and confirms within a fresh run")
                _qualify_glm_prefill(root, record, deadline, packed_prefill_tile)
            elif phase == "prefill":
                if qualification_runs:
                    raise ValueError("Full prefill qualifies both heads in its first run")
                _qualify_full_glm_prefill(root, record, deadline, resume_run)
            else:
                record["qualification"] = _glm_timing_qualification(
                    qualification_runs, allow_incomplete=allow_incomplete_qualification, phase="upstream-decode" if upstream else phase
                )
            record["experiment_status"] = (
                "diagnostic" if profile_decode
                else "qualified" if record["qualification"]["status"] == "passed"
                else "unqualified"
            )
            if not prefill_h200 and not upstream:
                _run_pytest((GLM_DECODE_GRAPH if phase == "decode" else GLM_BENCHMARK_GRAPH,), root, "benchmark-graphs", deadline)
            result_name = "profile" if profile_decode else "timing"
            argv = _benchmark_command("--model-family", "glm53", "--suite", suite,
                                      "--phase", phase, "--packed-prefill-tile", str(packed_prefill_tile),
                                      "--output", str(root / result_name),
                                      "--max-seconds", str(max(60, int(deadline-time.monotonic())-10)))
            if profile_decode:
                argv += ["--profile-decode"]
            if resume_run:
                checkpoint = _glm_timing_checkpoint(resume_run)
                argv += ["--resume", str(checkpoint)]
                record["resume_run"] = resume_run
            record.update(command=argv, modes=("upstream", "direct") if upstream else ("native", "direct"), suite=suite)
            _run_command(argv, root / f"{result_name}.log", deadline)
            contract = _glm_benchmark_contract()
            first = json.loads((root / f"{result_name}.json").read_text())
            if profile_decode:
                contract.validate_decode_profile(first)
                record["profile"] = first
                return str(root)
            contract.validate_timing(first)
            if phase == "prefill-small":
                second_argv = argv.copy()
                second_argv[second_argv.index("--output") + 1] = str(root / "confirmation")
                second_argv[second_argv.index("--max-seconds") + 1] = str(max(60, int(deadline-time.monotonic())-10))
                record["confirmation_command"] = second_argv
                _run_command(second_argv, root / "confirmation.log", deadline)
                second = json.loads((root / "confirmation.json").read_text())
                contract.validate_timing(second)
                if first["source"] != second["source"]:
                    raise RuntimeError("Prefill confirmation changed source or artifact identity")
                ratios = {
                    a["name"]: [
                        row["paired_ratios"]["direct_over_native"]["median"]
                        for row in (a, b)
                    ]
                    for a, b in zip(first["cases"], second["cases"], strict=True)
                }
                record["prefill_parity"] = dict(
                    paired_medians=ratios,
                    status="passed" if all(max(pair) <= 1.0 for pair in ratios.values()) else "missed",
                )
        return str(root)
    if resume_run or allow_incomplete_qualification or phase != "all":
        raise ValueError("resume_run/allow_incomplete_qualification/phase are GLM-only")
    if qualification_runs and suite != "v41-fp4-decode":
        raise ValueError("qualification_runs requires v41-fp4-decode")
    contract = _benchmark_cases()
    is_prefill = suite == "prefill-read"
    is_fp4 = suite == "v41-fp4-decode"
    if is_fp4:
        if (
            VALIDATION_SUITE not in ("v41-fp4", "v41-decode")
            or keep_count != 256
            or modes not in ("", "native,dense_fp4,kernel")
        ):
            raise ValueError(
                "FP4 timing requires upstream Native, custom dense FP4 and Packed keep256"
            )
        if prefill_reader != "split":
            raise ValueError("FP4 timing uses the Direct decode paths")
        results_volume.reload()
        with _run_context(run_id, "timing", RUN_BUDGETS_S["timing"]) as (
            root,
            record,
            deadline,
        ):
            # Recheck after allocation to detect receipts changed since CPU preflight.
            # Missing/failed receipts have already failed without an H100 allocation.
            if preflight is None:
                raise RuntimeError("FP4 timing requires the CPU preflight entry point")
            qualification = _fp4_timing_qualification(qualification_runs)
            if qualification != preflight:
                raise RuntimeError("Qualification changed after CPU preflight")
            record["qualification"] = qualification
            argv = _benchmark_command(
                "--suite", suite, "--output", str(root / "timing")
            )
            record.update(command=argv, suite=suite)
            _run_command(argv, root / "timing.log", deadline)
            contract.validate_timing(
                json.loads((root / "timing.json").read_text()), suite=suite
            )
        return str(root)
    if qualification_runs:
        raise ValueError("qualification_runs is only supported by v41-fp4-decode")
    if prefill_reader not in ("split", "fused") or (
        not is_prefill and prefill_reader != "split"
    ):
        raise ValueError(
            "prefill_reader must be split, or fused with suite prefill-read"
        )
    if suite not in ("all", "prefill-read") or (
        is_prefill and (modes or keep_count != 256)
    ):
        raise ValueError(
            "suite must be all or prefill-read; prefill compares all three fixed variants"
        )
    selected_modes = (
        contract.PREFILL_VARIANTS
        if is_prefill
        else contract.benchmark_modes(keep_count, modes or None)
    )
    with _run_context(run_id, "timing", RUN_BUDGETS_S["timing"]) as (
        root,
        record,
        deadline,
    ):
        tests = BENCHMARK_TESTS
        if prefill_reader == "fused":
            tests += (TEST_MODULES["unpack_kernel"], TEST_MODULES["cuda_graph"])
        _run_pytest(tests, root, "validation", deadline)
        if prefill_reader == "fused":
            for tool in ("memcheck", "racecheck", "synccheck"):
                prefix = ("compute-sanitizer", "--tool", tool, "--error-exitcode", "1")
                _run_pytest(SANITIZER_TESTS, root, tool, deadline, prefix=prefix)
            record["sanitizers"] = ["memcheck", "racecheck", "synccheck"]
        options = (
            []
            if is_prefill
            else ["--keep-count", str(keep_count), "--modes", ",".join(selected_modes)]
        )
        if prefill_reader == "fused":
            options += ["--prefill-reader", prefill_reader]
        argv = _benchmark_command(
            "--suite", suite, *options, "--output", str(root / "timing")
        )
        record.update(
            command=argv, keep_count=keep_count, modes=selected_modes, suite=suite
        )
        if is_prefill:
            record["prefill_reader"] = prefill_reader
        _run_command(argv, root / "timing.log", deadline)
        timing = json.loads((root / "timing.json").read_text())
        if is_prefill:
            contract.validate_timing(
                timing,
                suite=suite,
                **(
                    {"prefill_reader": prefill_reader}
                    if prefill_reader == "fused"
                    else {}
                ),
            )
        else:
            contract.validate_timing(
                timing, keep_count=keep_count, modes=",".join(selected_modes)
            )
    return str(root)


@app.function(
    **{
        **_GPU,
        "timeout": VALIDATION_TIMEOUT_S,
        "gpu": "H200" if _UPSTREAM_DECODE else _GPU["gpu"],
        **({"cpu": 2, "memory": 16384} if VALIDATION_SUITE in ("v41-fp4", "v41-decode") else {}),
    }
)
def validate_kernels(
    run_id: str, sanitizer: bool = False, max_seconds: int = RUN_BUDGETS_S["validation"],
    model_family: str = "dsv4", stage: str = "numerical", cases: str = "", sanitizer_tools: str = "",
    sanitizer_scope: str = "all", sanitizer_order: str = "",
) -> str:
    """Run existing GPU tests; skipped tests fail coverage.

    A passing sanitizer subset records ``sanitizer_qualification=partial``.
    The scope-specific qualification fields additionally require the full
    selected test contract; focused coverage never qualifies the full suite.
    """
    if model_family != MODEL_FAMILY:
        raise ValueError("--model-family must match REMNANT_MODEL_FAMILY used to prepare the image")
    if stage not in ("numerical", "sanitizer", "corrective"):
        raise ValueError("stage must be numerical, sanitizer or corrective")
    if stage == "corrective":
        selected_tests = tuple(cases.split(","))
        owner = GLM_TEST if MODEL_FAMILY == "glm53" else TEST_ROOT + "test_remnant_fp4.py"
        if (
            (MODEL_FAMILY != "glm53" and VALIDATION_SUITE != "v41-fp4")
            or not cases
            or len(set(selected_tests)) != len(selected_tests)
            or any(not case.startswith(owner + "::test_") for case in selected_tests)
        ):
            raise ValueError("Corrective cases must belong to the selected model's test module")
        if MODEL_FAMILY != "glm53":
            allowed = {"test_fp4_store_and_reader", "test_fp4_direct", "test_fp4_graph_and_stream", "test_fp4_compressor_boundary", "test_fp4_fragment_masks", "test_fp4_startup_contract", "test_fp4_backend_dispatch", "test_fp4_pool_allocations"}
            if any(case.split("::", 1)[1].split("[", 1)[0] not in allowed for case in selected_tests):
                raise ValueError("FP4 corrective cases must use an allowlisted test ID")
    else:
        if cases:
            raise ValueError("--cases is only supported for the GLM corrective stage")
        selected_tests = VALIDATION_TESTS
    if sanitizer_scope not in ("all", "glm-decode", "glm-merge", "v41-merge"):
        raise ValueError("sanitizer_scope must be all, glm-decode, glm-merge or v41-merge")
    if sanitizer_scope == "glm-decode" and (MODEL_FAMILY != "glm53" or stage != "corrective"
            or not sanitizer or set(selected_tests) != {GLM_TEST + "::test_packed_loader_bytes", GLM_DECODE_GRAPH}):
        raise ValueError("glm-decode sanitizer scope requires the exact focused decode tests")
    if sanitizer_scope == "glm-merge" and (
        MODEL_FAMILY != "glm53" or stage != "corrective" or not sanitizer
        or set(selected_tests) != set(GLM_MERGE_SANITIZER)
    ):
        raise ValueError("glm-merge sanitizer scope requires the exact loader/decode/prefill tests")
    if sanitizer_scope == "v41-merge" and (
        MODEL_FAMILY != "dsv4" or VALIDATION_SUITE != "v41-fp4"
        or stage != "corrective" or not sanitizer
        or set(selected_tests) != set(V41_MERGE_SANITIZER)
    ):
        raise ValueError("v41-merge requires the exact FP4 store/attention/graph/producer cases")
    sanitizer = sanitizer or stage == "sanitizer"
    if sanitizer_tools and sanitizer_order:
        raise ValueError("Use sanitizer_tools or sanitizer_order, not both")
    if sanitizer_order and not sanitizer:
        raise ValueError("sanitizer_order requires --sanitizer")
    selected_tools = (
        _validation_sanitizer_order(sanitizer_order) if sanitizer_order
        else _selected_sanitizer_tools(sanitizer_tools, sanitizer)
    )
    tool_order = selected_tools
    maximum = min(VALIDATION_TIMEOUT_CAP_S - 60, VALIDATION_TIMEOUT_S - 60)
    if not 60 <= max_seconds <= maximum:
        raise ValueError(f"max_seconds must be between 60 and {maximum}")
    with _run_context(run_id, "validation", max_seconds) as (
        root,
        record,
        deadline,
    ):
        environment = dict(os.environ)
        record["validation_suite"] = VALIDATION_SUITE
        record["coverage"] = {
            "validation": VALIDATION_TESTS,
            "sanitizer": SANITIZER_TESTS,
        }
        record["sanitizer"] = sanitizer
        record["sanitizer_scope"] = sanitizer_scope
        record["selected_sanitizer_tools"] = selected_tools
        record["sanitizer_qualification_complete"] = False
        if MODEL_FAMILY == "glm53":
            record["runtime_hashes"] = _glm_benchmark_contract().runtime_hashes(SGLANG_ROOT / "python/sglang")
        record["sanitizer_order"] = tool_order
        if sanitizer:
            record["sanitizer_tools"] = list(selected_tools)
            record["sanitizer_qualification"] = "incomplete"
            record["sanitizer_results"] = {
                tool: {"status": "not_run" if tool in selected_tools else "not_selected"}
                for tool in SANITIZER_ORDER
            }
        if INTEGRATION["version"] == "v0.5.22":
            import torch

            gpu_family = "H200" if _UPSTREAM_DECODE else "H100"
            if (
                torch.cuda.get_device_capability() != (9, 0)
                or gpu_family not in torch.cuda.get_device_name()
            ):
                raise RuntimeError(f"Qualification requires one {gpu_family}/SM90")
            if "cpu-driver-stubs" in environment.get("LD_LIBRARY_PATH", ""):
                raise RuntimeError("CPU-only driver stubs leaked into the GPU loader")
            manifest = json.loads(MANIFEST.read_text())
            if manifest.get("validation_suite", "v4") != VALIDATION_SUITE:
                raise RuntimeError("Validation suite mismatch")
            if manifest["integration"] != "v0.5.22":
                raise RuntimeError("Serving image integration mismatch")
            if (
                manifest["source_identities"] != SOURCE_IDENTITIES
                or manifest["sources"] != _source_hashes()
            ):
                raise RuntimeError("Serving image source identity mismatch")
            from sgl_kernel import flashmla_ops, remnant_ops

            loaded = {
                str(Path(module.__file__).resolve())
                for module in (flashmla_ops, remnant_ops)
            }
            if not loaded <= set(manifest["binaries"]):
                raise RuntimeError(f"Loaded unverified extensions: {loaded}")
            for path, expected in manifest["binaries"].items():
                if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"Loaded binary input changed: {path}")
            if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
                artifacts = manifest.get("fp4_precompile_artifacts", {})
                if not artifacts:
                    raise RuntimeError("Missing CPU-precompiled FP4 artifacts")
                for path, expected in artifacts.items():
                    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                        raise RuntimeError(f"FP4 precompiled artifact changed: {path}")
            tool = manifest["sanitizer"]
            if (
                tool["path"] != SANITIZER
                or hashlib.sha256(Path(SANITIZER).read_bytes()).hexdigest()
                != tool["binary_sha256"]
            ):
                raise RuntimeError("Compute Sanitizer identity mismatch")
            record["sanitizer_tool"] = tool
            if MODEL_FAMILY == "glm53":
                compiled = manifest.get("glm_compile")
                if not compiled or compiled.get("attention_scale") != 0.0625:
                    raise RuntimeError("GLM compile manifest has an unexpected attention scale")
                verify_glm_tile32_variants(compiled)
                for path, expected in compiled["cache_files"].items():
                    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                        raise RuntimeError(f"GLM compiled cache artifact changed: {path}")
                from sglang.srt.remnant.glm import compile_kernels
                if [k.hash for k in compile_kernels()] != compiled["triton"]:
                    raise RuntimeError("GLM Triton artifact identity mismatch")
                record["glm_compile"] = compiled
            environment["REMNANT_ALLOCATION_REPORT"] = str(root / "allocation.json")
            shutil.copy2("/opt/remnant-glm-config.json" if MODEL_FAMILY == "glm53" else "/opt/remnant-0731-config.json", root / "model-config.json")
            record["gpu"] = torch.cuda.get_device_name()
            record["loaded_extensions"] = sorted(loaded)
            record["platform_timeout_seconds"] = VALIDATION_TIMEOUT_S
            cpu_reports = root / "cpu-preflight"
            cpu_reports.mkdir()
            for path in BUILD_ROOT.iterdir():
                if path.is_file() and path.suffix in {".log", ".xml"}:
                    shutil.copy2(path, cpu_reports / path.name)
            if sanitizer:
                # Separate dependency initialization from our kernel checks.
                # Keep API-error reporting unchanged; a failed control is a
                # failed qualification, not an allowlisted sanitizer report.
                _run_command(
                    [SANITIZER, "--version"],
                    root / "sanitizer-version.log",
                    deadline,
                )
                _run_command(
                    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
                    root / "driver-version.log",
                    deadline,
                )
                controls = (
                    ("empty", "print('empty Python control', flush=True)"),
                    (
                        "bindings-import",
                        "print('before binding import', flush=True); "
                        "from cuda.bindings import driver; "
                        "print('after binding import', driver.__file__, flush=True)",
                    ),
                    (
                        "bindings-query",
                        "import importlib.metadata as m; "
                        "print('cuda-bindings', m.version('cuda-bindings'), flush=True); "
                        "from cuda.bindings import driver; "
                        "result, version = driver.cuDriverGetVersion(); "
                        "print('driver', result, version, flush=True); "
                        "assert result == driver.CUresult.CUDA_SUCCESS",
                    ),
                )
                for name, code in controls:
                    _run_command(
                        [
                            SANITIZER,
                            "--tool",
                            "memcheck",
                            "--error-exitcode",
                            "1",
                            "--require-cuda-init",
                            "no",
                            sys.executable,
                            "-c",
                            code,
                        ],
                        root / f"sanitizer-control-{name}.log",
                        deadline,
                    )
        record["model_family"] = MODEL_FAMILY
        record["stage"] = stage
        if MODEL_FAMILY == "glm53":
            # First-ten-minute smoke gate before scheduling the broader matrix.
            if stage == "numerical":
                _run_pytest(GLM_SMOKE, root, "smoke", min(deadline, time.monotonic() + 600), environment=environment)
                _run_pytest(VALIDATION_TESTS, root, "validation", deadline, environment=environment)
            elif stage == "corrective":
                if sanitizer and any(node.startswith(GLM_SANITIZER_GRAPH) for node in selected_tests):
                    _run_pytest((GLM_BENCHMARK_GRAPH,), root, "oracle-bridge", deadline, environment=environment)
                record["corrective_tests"] = selected_tests
                _run_pytest(selected_tests, root, "corrective", deadline, environment=environment)
            else:
                _run_pytest((GLM_CPU_TEST,), root, "startup-regressions", deadline, environment=environment)
                _run_pytest((GLM_BENCHMARK_GRAPH,), root, "benchmark-graph-smoke", min(deadline, time.monotonic()+300), environment=environment)
        else:
            _run_pytest(
                selected_tests if stage == "corrective" else VALIDATION_TESTS,
                root, "validation", deadline, environment=environment,
            )
        if INTEGRATION["version"] == "v0.5.22" and (stage == "numerical" or MODEL_FAMILY != "glm53" and stage != "corrective" or (stage == "corrective" and GLM_TEST + "::test_real_gpu_allocations" in selected_tests)):
            allocation = json.loads((root / "allocation.json").read_text())
            if allocation["status"] != "passed" or {
                r["mode"] for r in allocation["modes"]
            } != (
                {"native", "adapter", "direct"} if MODEL_FAMILY == "glm53"
                else {"Native", "Packed"} if VALIDATION_SUITE in ("v41-fp4", "v41-decode")
                else {"native", "triton", "kernel"}
            ):
                raise RuntimeError("Incomplete real-allocation coverage")
            record["allocation"] = allocation
        combine_tests = ()
        wgmma_tests = ()
        decode_tests = ()
        glm_scope_filter = glm_decode_scope_filter()
        if sanitizer and sanitizer_scope in ("glm-decode", "glm-merge"):
            if _UPSTREAM_DECODE and sanitizer_scope == "glm-decode":
                if not glm_decode_combine_cases(manifest) or not glm_wgmma_cases(manifest):
                    raise RuntimeError("Upstream decode qualification requires all candidate artifacts")
                combine_tests = (GLM_TEST + "::test_packed_decode_combine_graph",)
                glm_scope_filter = glm_decode_scope_filter(combine=True)
                wgmma_tests = (
                    GLM_TEST + "::test_wgmma_decode_partial_graph",
                    GLM_TEST + "::test_wgmma_decode_combine_graph",
                )
            audit_prefix = (SANITIZER, "--tool", "racecheck", "--error-exitcode", "1",
                            "--kernel-name", glm_scope_filter, "--racecheck-trace-sync", "yes")
            wgmma_audit = (wgmma_tests[0] + "[8]",) if wgmma_tests else ()
            if sanitizer_scope == "glm-decode":
                decode_tests = (
                    GLM_TEST + "::test_packed_loader_bytes", GLM_DECODE_PARTIAL,
                    *combine_tests, *wgmma_tests,
                )
                record["coverage"]["sanitizer"] = decode_tests
            # The trace establishes instrumentation for each entry point.
            # Every combine specialization still runs under each tool below.
            combine_audit = (combine_tests[0] + "[6-512]",) if combine_tests else ()
            _run_pytest((GLM_TEST + "::test_packed_loader_bytes", *combine_audit, *wgmma_audit), root, "scope-audit",
                        min(deadline, time.monotonic() + 180), prefix=audit_prefix)
            audit_log = (root / "scope-audit.log").read_text()
            _check_sanitizer_report(audit_log, "racecheck")
            _check_glm_scope_audit(
                audit_log, run_id, combine=bool(combine_tests), wgmma=bool(wgmma_tests)
            )
            record["scope_audit"] = dict(status="passed", filter=glm_scope_filter,
                log_sha256=hashlib.sha256(audit_log.encode()).hexdigest(),
                entrypoints=manifest["glm_compile"]["scope_symbol_audit"],
                basis="Racecheck synchronization trace; other tools use completed tests, identical entrypoints and common filter contract")
        if sanitizer and sanitizer_scope == "v41-merge":
            symbols = manifest.get("fp4_scope_symbols")
            if not symbols or symbols["filter"] != "regex=" + FP4_KERNEL_PATTERN:
                raise RuntimeError("Missing CPU-verified FP4 instrumentation symbols")
            audit_prefix = _fp4_sanitizer_prefix("racecheck", trace=True)
            _run_pytest((TEST_ROOT + "test_remnant_fp4.py::test_fp4_direct[128-64-True]",),
                        root, "scope-audit", min(deadline, time.monotonic() + 180),
                        prefix=audit_prefix)
            audit_log = (root / "scope-audit.log").read_text()
            check_fp4_scope_audit(audit_log, run_id)
            record["scope_audit"] = {
                "status": "passed", "symbols": symbols,
                "log_sha256": hashlib.sha256(audit_log.encode()).hexdigest(),
                "basis": "Positive Packed decode synchronization trace; store/read/prefill/combine names verified in CPU artifacts and common filter",
            }
        if sanitizer:
            for tool in selected_tools:
                # Commands use whole-second timeouts. Preserve the incomplete
                # tool matrix instead of attempting a launch with zero seconds.
                if int(deadline - time.monotonic()) <= 0:
                    break
                prefix = (SANITIZER, "--tool", tool, "--error-exitcode", "1")
                if sanitizer_scope in ("glm-decode", "glm-merge"):
                    # Distinct Packed entrypoints include their inline helper.
                    # One exact production CTA per bank; stock Native partial
                    # is the control. Full attention has its own numerical gate.
                    prefix += ("--kernel-name", glm_scope_filter, "--print-level", "info")
                if sanitizer_scope == "v41-merge":
                    prefix = _fp4_sanitizer_prefix(tool)

                # A hard kill must leave an incomplete running checkpoint, never
                # a misleading not_run state that hides partial failure output.
                record["sanitizer_results"][tool] = {"status": "running"}
                (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
                results_volume.commit()
                try:
                    tool_environment = dict(environment)
                    if sanitizer_scope in ("glm-decode", "glm-merge"):
                        tool_environment["PYTEST_ADDOPTS"] = "--full-trace -o faulthandler_timeout=120"
                    _run_pytest(decode_tests if sanitizer_scope == "glm-decode"
                                else (selected_tests if stage == "corrective" else SANITIZER_TESTS),
                                root, tool, deadline, prefix=prefix,
                                environment=tool_environment if sanitizer_scope in ("glm-decode", "glm-merge") else None)
                except TimeoutError:
                    # The boundary may expire between this loop and the command
                    # check. No process started; return this tool to not_run.
                    record["sanitizer_results"][tool] = {"status": "not_run"}
                    break
                except RuntimeError as error:
                    record["sanitizer_results"][tool] = {
                        "status": "failed",
                        "error": str(error),
                    }
                    # Each tool owns a fresh pytest process. Retain later tools'
                    # evidence without converting this failure into a pass.
                except BaseException as error:
                    # Reuse of completed siblings requires an independent audit
                    # of this tool's partial XML/log; cancellation is never a pass.
                    record["sanitizer_results"][tool] = {
                        "status": "interrupted",
                        "error": repr(error),
                    }
                    raise
                else:
                    record["sanitizer_results"][tool] = {"status": "passed"}
                finally:
                    state = record["sanitizer_results"][tool]
                    if state["status"] != "not_run":
                        state["report_schema"] = "remnant-sanitizer-raw-v1"
                        for suffix in ("log", "xml"):
                            path = root / f"{tool}.{suffix}"
                            if path.is_file():
                                state[f"{suffix}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    # Preserve completed evidence even if a later tool is cancelled.
                    (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
                    results_volume.commit()
            incomplete = [
                tool
                for tool in selected_tools
                if record["sanitizer_results"][tool]["status"] != "passed"
            ]
            all_tools_passed = all(result["status"] == "passed" for result in record["sanitizer_results"].values())
            focused = stage == "corrective" and set(selected_tests) == {
                GLM_TEST + "::test_packed_loader_bytes", GLM_DECODE_GRAPH}
            record["qualification_scope"] = "focused_fp4_merge" if sanitizer_scope == "v41-merge" else "focused_merge" if sanitizer_scope == "glm-merge" else "focused_decode" if focused else stage
            record["focused_fp4_merge_qualification_complete"] = sanitizer_scope == "v41-merge" and all_tools_passed
            record["focused_merge_qualification_complete"] = sanitizer_scope == "glm-merge" and all_tools_passed
            record["focused_decode_qualification_complete"] = focused and all_tools_passed
            record["sanitizer_qualification_complete"] = all_tools_passed and stage != "corrective"
            if incomplete:
                raise RuntimeError(
                    f"Sanitizer qualification incomplete: {incomplete}; see tool logs"
                )
            record["sanitizer_qualification"] = (
                "complete" if set(selected_tools) == set(SANITIZER_ORDER) else "partial"
            )
    return str(root)


@app.function(**_GPU)
def profile_kernels(
    run_id: str,
    cases: str = "",
    max_seconds: int = RUN_BUDGETS_S["profile"],
    keep_count: int = 256,
    modes: str = "",
    suite: str = "all",
    prefill_reader: str = "split",
) -> str:
    """Profile selected cases and supported modes, separately from timing."""
    if not 300 <= max_seconds <= PROFILE_MAX_BUDGET_S:
        raise ValueError(f"max_seconds must be between 300 and {PROFILE_MAX_BUDGET_S}")
    contract = _benchmark_cases()
    is_prefill = suite == "prefill-read"
    if prefill_reader not in ("split", "fused") or (
        not is_prefill and prefill_reader != "split"
    ):
        raise ValueError(
            "prefill_reader must be split, or fused with suite prefill-read"
        )
    if suite not in ("all", "prefill-read") or (
        is_prefill and (modes or keep_count != 256)
    ):
        raise ValueError(
            "suite must be all or prefill-read; prefill compares all three fixed variants"
        )
    selected_modes = (
        contract.PREFILL_VARIANTS
        if is_prefill
        else contract.benchmark_modes(keep_count, modes or None)
    )
    matrix = contract.selected_profile_matrix(
        cases,
        keep_count=keep_count,
        modes=None if is_prefill else ",".join(selected_modes),
        suite=suite,
    )
    with _run_context(run_id, "profile", max_seconds) as (root, record, deadline):
        if INTEGRATION["version"] == "v0.5.22":
            _check_profiler_tools(root, deadline)
        record.update(
            matrix=matrix, runs=[], keep_count=keep_count, modes=selected_modes
        )
        # Batch cases per variant so NCU application replay pays Python/CUDA
        # startup three times per report, rather than three times per case.
        # NVTX-filtered exports retain separate per-case kernel attribution.
        jobs = (
            [
                ("prefill-read", ",".join(c for s, c, m in matrix if m == mode), mode)
                for mode in selected_modes
            ]
            if is_prefill
            else matrix
        )
        for suite, case, mode in jobs:
            name = f"{suite}-{mode}" if is_prefill else f"{suite}-{case}-{mode}"
            output = root / name
            options = (
                ["--profile-path", mode, "--profile-replay-id", f"{run_id}:{name}"]
                if is_prefill
                else [
                    "--profile-mode",
                    mode,
                    "--keep-count",
                    str(keep_count),
                    "--modes",
                    ",".join(selected_modes),
                ]
            )
            if prefill_reader == "fused":
                options += ["--prefill-reader", prefill_reader]
            target = _benchmark_command(
                "--suite", suite, "--case", case, *options, "--output", str(output)
            )
            options = {
                "--profile-from-start": "off",
                "--replay-mode": "application" if is_prefill else "kernel",
                "--cache-control": "none",
                "--clock-control": "none",
                "--import-source": "yes",
            }
            argv = ["ncu", *[arg for pair in options.items() for arg in pair]]
            if is_prefill:
                argv += ["--nvtx"]
            for section in [] if is_prefill else NCU_SECTIONS:
                argv += ["--section", section]
            argv += [
                "--metrics",
                (
                    PREFILL_METRICS
                    if is_prefill
                    else "gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"
                ),
                "--export",
                str(output),
                "--",
                *target,
            ]
            capture_seconds = 900 if is_prefill else PROFILE_CAPTURE_TIMEOUT_S
            _run_command(
                argv,
                root / f"{name}.log",
                min(deadline, time.monotonic() + capture_seconds),
            )
            report = output.with_suffix(".ncu-rep")
            if not report.is_file() or not report.stat().st_size:
                raise RuntimeError(f"Missing profile: {report}")
            capture = json.loads(output.with_suffix(".json").read_text())
            expected_cases = case.split(",")
            if (
                capture["status"] != "passed"
                or len(capture["cases"]) != len(expected_cases)
                or {item.get("name") for item in capture["cases"]}
                != set(expected_cases)
                or capture.get("config", {}).get("keep_count") != keep_count
            ):
                raise RuntimeError(f"Invalid capture: {name}")
            for item in capture["cases"]:
                case_name = item["name"]
                if (
                    not item.get("valid")
                    or (item.get("path") if is_prefill else item.get("mode")) != mode
                ):
                    raise RuntimeError(f"Invalid profile operation: {name}/{case_name}")
                if is_prefill:
                    expected = contract.prefill_path_layouts()[mode]
                    if (
                        capture.get("protocol") != contract.PREFILL_PROTOCOL
                        or item.get("mode") != expected["implementation"]
                        or item.get("keep_count")
                        != (expected["packed_keep_count"] or 256)
                        or item.get("path_layouts") != contract.prefill_path_layouts()
                        or item.get("capture_range") != f"remnant_{case_name}_{mode}"
                        or capture.get("config", {}).get("prefill_reader", "split")
                        != prefill_reader
                        or item.get("prefill_reader", "split") != prefill_reader
                    ):
                        raise RuntimeError(
                            f"Prefill capture identity mismatch: {name}/{case_name}"
                        )
                export = ["ncu", "--import", str(report)]
                if is_prefill:
                    # NVTX is already recorded; import uses the saved context.
                    export += ["--nvtx-include", item["capture_range"] + "/"]
                export += [
                    "--page",
                    "raw",
                    "--csv",
                    "--print-units",
                    "base",
                    "--print-kernel-base",
                    "demangled",
                    "--rename-kernels",
                    "off",
                ]
                metrics_path = root / f"{suite}-{case_name}-{mode}.metrics.csv"
                raw = _run_command(
                    export,
                    metrics_path,
                    min(deadline, time.monotonic() + PROFILE_EXPORT_TIMEOUT_S),
                )
                summary_options = (
                    {"traffic_only": True, "minimum_launches": item["minimum_launches"]}
                    if is_prefill
                    else {}
                )
                entry = {
                    "suite": suite,
                    "case": case_name,
                    "mode": mode,
                    "command": argv,
                    "export_command": export,
                    "report": str(report),
                    "capture": {**capture, "cases": [item]},
                    "summary": contract.profile_summary(
                        raw, item["expected_kernels"], **summary_options
                    ),
                }
                if prefill_reader == "fused" and any(
                    "rope_tail_complex" in name for name in entry["summary"]["kernels"]
                ):
                    raise RuntimeError(
                        f"Fused reader still launches separate RoPE: {case_name}/{mode}"
                    )
                record["runs"].append(entry)
                (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
                results_volume.commit()
        for suite, case in {(s, c) for s, c, _ in matrix}:
            group = [
                r for r in record["runs"] if (r["suite"], r["case"]) == (suite, case)
            ]
            if (
                {r["mode"] for r in group} != set(selected_modes)
                or len(group) != len(selected_modes)
                or len(
                    {r["capture"]["cases"][0]["common_inputs_sha256"] for r in group}
                )
                != 1
            ):
                raise RuntimeError(f"Profile inputs differ: {suite}/{case}")
    return str(root)


# Shared command execution, validation and result handling.


def _cpu_preflight_environment(build: Path) -> dict[str, str]:
    """Resolve driver symbols for CPU imports without changing GPU loader paths.

    The prebuilt sgl_kernel package links libcuda even for import/collection.
    CUDA supplies a linker stub on CPU builders, but its filename lacks the
    libcuda.so.1 SONAME needed by the dynamic loader. Only CPU child processes
    receive this private alias; the image and GPU environment remain untouched.
    """
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    candidates = (
        cuda_home / "lib64/stubs/libcuda.so",
        cuda_home / "targets/x86_64-linux/lib/stubs/libcuda.so",
    )
    library = next((p for p in candidates if p.is_file()), None)
    if library is None:
        raise RuntimeError(
            f"CPU import preflight requires CUDA's libcuda stub: {candidates}"
        )
    directory = build / "cpu-driver-stubs"
    directory.mkdir()
    (directory / "libcuda.so.1").symlink_to(library.resolve())
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(directory) + (
        ":" + environment["LD_LIBRARY_PATH"]
        if environment.get("LD_LIBRARY_PATH")
        else ""
    )
    return environment


def _stop_command_group(process: subprocess.Popen) -> None:
    """Stop our session, including compiler workers after their leader exits.

    The outer allocation reserves cleanup time. Waiting for the leader alone
    does not establish that its workers stopped, so always signal the group
    again after waiting for the leader. Orphan zombies need no additional waiting.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _run_command(
    argv: Sequence[str],
    output: Path,
    deadline: float,
    *,
    environment: dict[str, str] | None = None,
) -> str:
    """Bound an owned process group; retain logs and clean up on interruption."""
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError(f"Deadline reached before {list(argv)}")
    child_environment = dict(os.environ if environment is None else environment)
    # Modal adds its serialized /root/kernels.py to PYTHONPATH. It shadows
    # Transformers' optional Hugging Face `kernels` dependency in fresh children.
    # Keep the two declared source roots; installed site packages remain visible.
    child_environment["PYTHONPATH"] = (
        "/opt/sglang-runtime-fixes:" if INTEGRATION["version"] == "v0.5.22" else ""
    ) + f"{SGLANG_ROOT / 'python'}:{REMOTE_REPO}"
    if VALIDATION_SUITE in ("v41-fp4", "v41-decode"):
        child_environment["PYTHONPATH"] += f":{UPSTREAM_FLASHMLA_ROOT}"
    try:
        with output.open("w") as log:
            process = subprocess.Popen(
                list(argv),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=SGLANG_ROOT,
                env=child_environment,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=remaining)
                if returncode:
                    raise subprocess.CalledProcessError(returncode, list(argv))
            except BaseException as error:
                try:
                    _stop_command_group(process)
                except Exception as cleanup_error:
                    error.add_note(f"Process group cleanup failed: {cleanup_error!r}")
                    raise error from cleanup_error
                raise
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        tail = output.read_text(errors="replace")[-4000:] if output.is_file() else ""
        raise RuntimeError(
            f"Command failed: {list(argv)}\nLog: {output}\n{tail}"
        ) from error
    return output.read_text()


def _benchmark_command(*options: str) -> list[str]:
    script = SGLANG_ROOT / "benchmark/remnant/microbench.py"
    return [sys.executable, str(script),
            *([] if MODEL_FAMILY == "glm53" else ["--native-library", str(NATIVE_LIBRARY)]), *options]


def _run_pytest(
    tests: Sequence[str],
    root: Path,
    name: str,
    deadline: float,
    *,
    collect: bool = False,
    prefix: Sequence[str] = (),
    environment: dict[str, str] | None = None,
) -> None:
    report = root / f"{name}.xml"
    argv = [
        *prefix,
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--disable-warnings",
        "--tb=short",
    ]
    argv += [str(SGLANG_ROOT / path) for path in tests]
    argv += ["--collect-only"] if collect else [f"--junitxml={report}"]
    output = _run_command(argv, root / f"{name}.log", deadline, environment=environment)
    if not collect:
        try:
            _, counts = read_pytest_report(report)
        except (OSError, ET.ParseError, ValueError) as error:
            raise RuntimeError(
                f"Incomplete validation {name}: unreadable {report}"
            ) from error
        if not counts["tests"] or any(
            counts[k] for k in ("errors", "failures", "skipped")
        ):
            raise RuntimeError(f"Incomplete validation {name}: {counts}")
        if prefix:
            tool = prefix[prefix.index("--tool") + 1]
            try:
                _check_sanitizer_report(output, tool)
            except RuntimeError as error:
                raise RuntimeError(f"Incomplete sanitizer evidence {tool}; see {root / f'{name}.log'}") from error



def _selected_sanitizer_tools(value: str, enabled: bool) -> tuple[str, ...]:
    if value and not enabled:
        raise ValueError("sanitizer_tools requires sanitizer=True")
    return _select_sanitizer_tools(value, SANITIZER_ORDER)


def _glm_benchmark_contract():
    path = (SGLANG_SOURCE if modal.is_local() else SGLANG_ROOT) / "python/sglang/benchmark/remnant/glm_attention.py"
    spec = importlib.util.spec_from_file_location("remnant_glm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_glm_compiled_artifacts() -> dict:
    """Check image sources and CPU-produced binaries without initializing CUDA."""
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("model_family") != "glm53" or manifest["source_identities"] != SOURCE_IDENTITIES or manifest["sources"] != _source_hashes():
        raise RuntimeError("GLM timing image source identity mismatch")
    verify_glm_tile32_variants(manifest["glm_compile"])
    for artifacts in (manifest["binaries"], manifest["glm_compile"]["cache_files"]):
        for path, expected in artifacts.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"GLM timing artifact changed: {path}")
    return manifest


def _verify_glm_image() -> None:
    from sglang.srt.remnant.glm import compile_kernels

    manifest = _verify_glm_compiled_artifacts()
    if [k.hash for k in compile_kernels()] != manifest["glm_compile"]["triton"]:
        raise RuntimeError("GLM timing Triton variant mismatch")


def _qualify_full_glm_prefill(
    root: Path, record: dict, deadline: float, resume_run: str
) -> None:
    if resume_run:
        previous_root = _glm_timing_checkpoint(resume_run).parent
        previous = json.loads((previous_root / "result.json").read_text())
        origin = previous.get("qualification_run", resume_run)
        origin_root = _glm_timing_checkpoint(origin).parent
        qualification = json.loads((origin_root / "result.json").read_text())["qualification"]
        if (
            (origin_root / "build-manifest.json").read_bytes() != MANIFEST.read_bytes()
            or qualification["status"] != "passed"
            or set(qualification["by_heads"]) != {"16", "8"}
        ):
            raise RuntimeError("Full prefill checkpoint lacks matching two-head qualification")
        for heads, result in qualification["by_heads"].items():
            variant = _glm_prefill_variant(int(heads), 64)
            if (
                result["status"] != "passed"
                or result["tile_rows"] != 64
                or result["inner_iter"] != variant["inner"]
                or any(result[key] != variant[key] for key in ("cubin_sha256", "source_sha256", "dynamic_shared"))
            ):
                raise RuntimeError(f"Full prefill checkpoint has incomplete H{heads} qualification")
            for tool in SANITIZER_ORDER:
                name = f"prefill-{tool}" if heads == "16" else f"prefill-h8-{tool}"
                output = (origin_root / f"{name}.log").read_bytes()
                report = (origin_root / f"{name}.xml").read_bytes()
                if (
                    hashlib.sha256(output).hexdigest() != result["tools"][tool]["log_sha256"]
                    or hashlib.sha256(report).hexdigest() != result["tools"][tool]["report_sha256"]
                ):
                    raise RuntimeError(f"Full prefill checkpoint sanitizer evidence changed: {name}")
                _check_sanitizer_report(output.decode(), tool)
        record.update(qualification=qualification, qualification_run=origin)
        return

    by_heads = {}
    record["qualification"] = qualification = dict(status="incomplete", by_heads=by_heads)
    for heads in (16, 8):
        head_record = {}
        try:
            _qualify_glm_prefill(root, head_record, deadline, 64, heads=heads)
        finally:
            by_heads[str(heads)] = head_record.get("qualification", {"status": "incomplete"})
    qualification["status"] = "passed"


def _glm_prefill_variant(heads: int, tile_rows: int) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    variants = manifest["glm_compile"]["tilelang"]
    inner = {32: 66, 64: 17, 128: 9}[tile_rows]
    matches = [
        variant for variant in variants
        if (variant["heads"], variant["inner"], variant["packed"], variant.get("block_i", 32))
        == (heads, inner, True, tile_rows)
    ]
    if len(matches) != 1 or not matches[0].get("cubin_sha256"):
        raise RuntimeError(f"Missing verified H{heads} Packed prefill artifact for tile {tile_rows}")
    return matches[0]


def _qualify_glm_prefill(
    root: Path, record: dict, deadline: float, tile_rows: int, *, heads: int = 16
) -> None:
    """Instrument the selected production partial on the same image and device as timing."""
    import torch

    device = torch.cuda.get_device_name()
    if "H200" not in device or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError(f"GLM prefill requires H200 SM90, got {device}")
    variant = _glm_prefill_variant(heads, tile_rows)
    qualification = dict(
        status="incomplete",
        device=device,
        tile_rows=tile_rows,
        inner_iter=variant["inner"],
        cubin_sha256=variant["cubin_sha256"],
        source_sha256=variant["source_sha256"],
        dynamic_shared=variant["dynamic_shared"],
        tools={},
    )
    record["qualification"] = qualification
    node = (f"{GLM_TEST}::test_prefill_partial_graph[{tile_rows}]" if heads == 16
            else f"{GLM_TEST}::test_prefill_partial_graph_h8")
    for tool in SANITIZER_ORDER:
        name = f"prefill-{tool}" if heads == 16 else f"prefill-h8-{tool}"
        prefix = (SANITIZER, "--tool", tool, "--error-exitcode", "1", "--kernel-name",
                  "regex=glm_packed_partial_kernel")
        _run_pytest((node,), root, name, min(deadline, time.monotonic()+240), prefix=prefix)
        output = (root / f"{name}.log").read_bytes()
        _check_sanitizer_report(output.decode(), tool)
        report = (root / f"{name}.xml").read_bytes()
        properties = {prop.get("name"): prop.get("value") for prop in ET.fromstring(report).iter("property")}
        if json.loads(properties.get("partial_cubins", "{}")).get("True") != variant["cubin_sha256"]:
            raise RuntimeError(f"Prefill sanitizer did not test the selected cubin: {name}")
        qualification["tools"][tool] = dict(
            status="passed", prefix=prefix, test=node,
            log_sha256=hashlib.sha256(output).hexdigest(),
            report_sha256=hashlib.sha256(report).hexdigest(),
        )
    qualification["status"] = "passed"


def _glm_timing_checkpoint(run_id: str) -> Path:
    validate_run_id(run_id)
    checkpoint = RESULTS_ROOT / "microbench" / run_id / "timing" / "timing.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GLM checkpoint not found: {checkpoint}")
    return checkpoint


def _glm_timing_qualification(
    run_ids: str, *, allow_incomplete: bool = False, phase: str = "all"
) -> dict:
    qualification_run_ids(run_ids)
    context = QualificationContext(
        results_root=RESULTS_ROOT,
        manifest=json.loads(MANIFEST.read_text()),
        runtime_hashes=_glm_benchmark_contract().runtime_hashes(
            SGLANG_ROOT / "python/sglang"
        ),
        sanitizer_order=SANITIZER_ORDER,
    )
    return qualify_glm_timing(
        run_ids, context=context, allow_incomplete=allow_incomplete, phase=phase
    )


def _benchmark_cases():
    path = (
        SGLANG_SOURCE if modal.is_local() else SGLANG_ROOT
    ) / "python/sglang/benchmark/remnant/cases.py"
    spec = importlib.util.spec_from_file_location("remnant_modal_cases", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _run_context(
    run_id: str, action: str, seconds: int, *, resource: str = "gpu"
) -> Iterator[tuple[Path, dict, float]]:
    validate_run_id(run_id)
    root = RESULTS_ROOT / "microbench" / run_id / action
    root.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "action": action, "model_load": False}
    started = time.monotonic()
    try:
        shutil.copy2(MANIFEST, root / "build-manifest.json")
        yield root, record, started + seconds
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record[f"{resource}_function_seconds"] = time.monotonic() - started
        (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        try:
            results_volume.commit()
        finally:
            if resource == "gpu" and VALIDATION_SUITE not in ("v41-fp4", "v41-decode"):
                triton_volume.commit()
