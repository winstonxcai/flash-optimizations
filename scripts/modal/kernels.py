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
share the caller's aggregate $15 ledger; a stage does not grant another budget.
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
import shutil
import shlex
import subprocess
import sys
import sysconfig
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import modal
from scripts.sources import check_sources, integration_profile

INTEGRATION = integration_profile()
MODEL_FAMILY = os.environ.get("REMNANT_MODEL_FAMILY", "dsv4")
if MODEL_FAMILY not in ("dsv4", "glm53") or (MODEL_FAMILY == "glm53" and INTEGRATION["version"] != "v0.5.22"):
    raise ValueError("GLM validation requires REMNANT_INTEGRATION=v0.5.22")
GLM_TEST = "test/registered/attention/unittests/test_glm5_remnant.py"
GLM_CPU_TEST = "test/registered/unit/test_glm5_mla_cache_format.py"
GLM_BENCHMARK_CPU = "benchmark/remnant/test_microbench.py"
GLM_BENCHMARK_GRAPH = GLM_TEST + "::test_benchmark_bank_graph_ownership"
GLM_DECODE_GRAPH = GLM_TEST + "::test_decode_bank_graph"
GLM_DECODE_PARTIAL = GLM_TEST + "::test_decode_partial_graph"
GLM_DECODE_REQUIRED = {"test_packed_loader_bytes", "test_decode_partial_graph[8-1]", "test_decode_partial_graph[8-2]",
                       "test_decode_partial_graph[16-1]", "test_decode_partial_graph[16-2]"}
GLM_SANITIZER_GRAPH = GLM_TEST + "::test_benchmark_sanitizer_bank_graph"
GLM_BASELINE = "9e42a0d8886dd93944dc6b47ef3bffef4e033b5a"
GLM_SMOKE = (GLM_TEST + "::test_producer_reconstruction", GLM_TEST + "::test_native_fp8_conversion", GLM_TEST + "::test_allocator_accounting", GLM_TEST + "::test_invalid_attention", GLM_TEST + "::test_partial_outputs",
             GLM_TEST + "::test_attention[3-8-8]",
             GLM_TEST + "::test_changing_input_graph", GLM_TEST + "::test_production_backend_constructor")
GLM_SANITIZER = (*GLM_SMOKE, GLM_TEST + "::test_attention[0-1-64]", GLM_TEST + "::test_pool_lifecycle")

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
    from scripts.sources import check_sources

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
if MODEL_FAMILY == "glm53" and modal.is_local():
    GLM_BASELINE_RUNTIME_HASHES = {
        path: hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(SGLANG_SOURCE), "show", f"{GLM_BASELINE}:python/sglang/{path}"], timeout=30
        )).hexdigest()
        for path in ("kernels/ops/attention/dsa/tilelang_kernel.py", "srt/remnant/glm.py",
                     "srt/remnant/glm_pool.py", "srt/remnant/reference.py", "srt/layers/layernorm.py")
    }
else:
    GLM_BASELINE_RUNTIME_HASHES = json.loads(os.environ.get("REMNANT_GLM_BASELINE_RUNTIME_HASHES", "{}"))
RESULTS_ROOT = Path("/results")
NATIVE_LIBRARY = Path("/opt/remnant-native-store.so")
MANIFEST = Path("/opt/remnant-build.json")
BUILD_ROOT = Path("/opt/remnant-build")
SANITIZER_ROOT = Path("/opt/remnant-sanitizer")
SANITIZER = (
    str(SANITIZER_ROOT / "usr/local/cuda-13.3/compute-sanitizer/compute-sanitizer")
    if INTEGRATION["version"] == "v0.5.22"
    else "compute-sanitizer"
)
# Preserve existing work caps; reserve time inside the outer timeout for cleanup.
CPU_BUILD_TIMEOUT_S = GPU_TIMEOUT_S = 3600
VALIDATION_TIMEOUT_S = (
    int(os.environ.get("REMNANT_VALIDATION_TIMEOUT_S", "2700"))
    if INTEGRATION["version"] == "v0.5.22"
    else GPU_TIMEOUT_S
)
if INTEGRATION["version"] == "v0.5.22" and not 120 <= VALIDATION_TIMEOUT_S <= 2700:
    raise ValueError("REMNANT_VALIDATION_TIMEOUT_S must be between 120 and 2700")
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
    SANITIZER_TESTS = (*GLM_SANITIZER, GLM_SANITIZER_GRAPH)
    BENCHMARK_TESTS = ()

# Finish missing synchronization coverage before retrying the slower memcheck.
# Tool order changes neither the selected cases nor the all-three passing gate.
SANITIZER_ORDER = (
    ("synccheck", "racecheck", "memcheck")
    if INTEGRATION["version"] == "v0.5.22"
    else ("memcheck", "racecheck", "synccheck")
)
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


def _sglang_ignored(path: Path) -> bool:
    allowed = (
        path.parts[0] == "python"
        or path.parts[:2] == ("benchmark", "remnant")
        or path.as_posix() in VALIDATION_TESTS
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
    for label, root in (("sglang", SGLANG_ROOT), ("flashmla", FLASHMLA_ROOT)):
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


def _reuse_glm_cpu_artifacts(build: Path, deadline: float, binding_requirements: dict) -> bool:
    """Fast CPU preparation for harness-only changes, with strict binary binding."""
    from scripts.sources import glm_compile_owner_digest, glm_partial_launch_metadata

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
    expected = {(h,i,p) for h in (8,16,32,64) for i in (1,2,33,66) for p in (False,True)}
    if {(v["heads"],v["inner"],v["packed"]) for v in compiled["tilelang"]} != expected:
        raise RuntimeError("Harness-only CPU reuse lacks maintained SM90 variants")
    glm_partial_launch_metadata(compiled)
    Path("/opt/remnant-glm-compile.json").write_text(json.dumps(compiled,indent=2)+"\n")
    environment = _cpu_preflight_environment(build)
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


def _build_image() -> None:
    """CPU image layer: compile existing SM90 targets, install, then preflight."""
    import torch
    from sglang.kernels.jit.utils import override_jit_cuda_arch

    deadline = time.monotonic() + CPU_BUILD_BUDGET_S
    build = BUILD_ROOT
    # This directory belongs solely to CPU preparation. A verified reused image
    # may contain prior reports; compiler caches live outside it and are retained.
    if build.exists() and MODEL_FAMILY == "glm53" and (os.environ.get("REMNANT_COMPILED_IMAGE") or os.environ.get("REMNANT_GLM_CPU_REBUILD_FROM")):
        shutil.rmtree(build)
    build.mkdir()
    binding_requirements = (
        _check_cuda_binding_requirements()
        if INTEGRATION["version"] == "v0.5.22"
        else None
    )
    if MODEL_FAMILY == "glm53" and _reuse_glm_cpu_artifacts(build, deadline, binding_requirements):
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
    if MODEL_FAMILY == "glm53":
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
    if MODEL_FAMILY == "dsv4":
        # 3. Precompile the Native store using the fork's dependency-valid cache.
        from sglang.kernels.jit.utils.compile.cache import find_prebuilt
        from sglang.kernels.ops.attention.dsv4.compress import (
            _jit_compress_norm_rope_module,
        )

        with override_jit_cuda_arch(9, 0, "a"):
            native_options = {}
            if INTEGRATION["version"] == "v0.5.22":
                from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

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
        revision = "46a050bec13002ad29ee795c16e61cf4fa3c4a9c"
        url = f"https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/{revision}/config.json"
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f":
            raise RuntimeError("Pinned GLM configuration content changed")
        Path("/opt/remnant-glm-config.json").write_bytes(payload)
        model_config = {"revision": revision, "sha256": digest, "url": url}
        _run_command([sys.executable, "-c", f"import runpy; tests=runpy.run_path({str(SGLANG_ROOT / GLM_TEST)!r}); tests['test_allocator_accounting'](); tests['test_compiled_launch_contract']()"],
                     build / "glm-allocator-cpu.log", deadline, environment=cpu_environment)
        _run_command([sys.executable, str(SGLANG_ROOT / GLM_TEST), "--compile-only"],
                     build / "glm-sm90-compile.log", deadline, environment=cpu_environment)
        compile_path = Path("/opt/remnant-glm-compile.json")
        compiled = json.loads(compile_path.read_text())
        from scripts.sources import glm_partial_launch_metadata
        glm_partial_launch_metadata(compiled)
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
        if len(entrypoints) != 32 or not probe_sources:
            raise RuntimeError("Incomplete verified TileLang entrypoint sources")
        compiled["scope_symbol_audit"] = dict(filter="regex=glm_packed_(partial|bytes)",
            partial_sources=entrypoints, probe_sources=probe_sources,
            excluded="Unchanged Native, combine and framework kernels")
        compile_path.write_text(json.dumps(compiled, indent=2) + "\n")
    else:
        model_config = None
        if INTEGRATION["version"] == "v0.5.22":
            revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
            url = f"https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/{revision}/config.json"
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023":
                raise RuntimeError("Pinned 0731 configuration content changed")
            Path("/opt/remnant-0731-config.json").write_bytes(payload)
            model_config = {"revision": revision, "sha256": digest, "url": url}
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
    # Profiler tools are not a correctness gate for the serving-image qualification.
    manifest = {
        "model_family": MODEL_FAMILY,
        "glm_compile": (json.loads(Path("/opt/remnant-glm-compile.json").read_text()) if MODEL_FAMILY == "glm53" else None),
        "integration": INTEGRATION["version"],
        "base_image": INTEGRATION["base_image"],
        "compiled_base_image": os.environ.get("REMNANT_COMPILED_IMAGE"),
        "cpu_rebuild_from": os.environ.get("REMNANT_GLM_CPU_REBUILD_FROM"),
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


compiled_image_id = os.environ.get("REMNANT_COMPILED_IMAGE") if MODEL_FAMILY == "glm53" else None
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
elif MODEL_FAMILY == "glm53" and os.environ.get("REMNANT_GLM_CPU_REBUILD_FROM"):
    # Explicit CPU rebuild, never a relaxed ordinary source-reuse check.
    from scripts.sources import glm_rebuild_source_digest
    candidate = glm_rebuild_source_digest([SGLANG_SOURCE / "python/sglang/kernels", FLASHMLA_SOURCE / "csrc"])
    base = dict(sglang="889edf56b33226c80b17ddf3472c1206df62c3f0", flashmla="3eb85944fd7dfa66a39fbb6fbb9a3560339c57f8")
    code = ("from scripts.sources import verify_glm_cpu_rebuild; "
            f"verify_glm_cpu_rebuild({base!r}, {candidate!r}, {[str(SGLANG_ROOT / 'python/sglang/kernels'), str(FLASHMLA_ROOT / 'csrc')]!r})")
    kernel_base = (modal.Image.from_id(os.environ["REMNANT_GLM_CPU_REBUILD_FROM"])
        .env({"PYTHONPATH": f"{REMOTE_REPO}:{SGLANG_ROOT / 'python'}"})
        .add_local_file(REPO_ROOT / "scripts/sources.py", str(REMOTE_REPO / "scripts/sources.py"), copy=True)
        .run_commands("python3 -c " + shlex.quote(code))
        .add_local_dir(SGLANG_SOURCE, str(SGLANG_ROOT), copy=True, ignore=_ignore))
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

kernel_image = (
    kernel_base.env(
        {
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
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "SGLANG_JIT_CACHE_DIR": "/opt/remnant-jit",
            "REMNANT_FLASHMLA_SOURCE_DIR": str(FLASHMLA_ROOT),
            "REMNANT_NATIVE_STORE_LIBRARY": str(NATIVE_LIBRARY),
            "REMNANT_BUILD_MANIFEST_PATH": str(MANIFEST),
            "REMNANT_GLM_BASELINE_RUNTIME_HASHES": json.dumps(GLM_BASELINE_RUNTIME_HASHES),
            "REMNANT_VALIDATION_TIMEOUT_S": str(VALIDATION_TIMEOUT_S),
            "REMNANT_MODEL_CONFIG": "/opt/remnant-0731-config.json",
            **(
                {"REMNANT_VALIDATION_TIMEOUT_S": str(VALIDATION_TIMEOUT_S)}
                if INTEGRATION["version"] == "v0.5.22"
                else {}
            ),
            "TRITON_CACHE_DIR": "/opt/remnant-triton-glm" if MODEL_FAMILY == "glm53" else "/triton-cache",
            "REMNANT_MODEL_FAMILY": MODEL_FAMILY,
            **({"REMNANT_COMPILED_IMAGE": compiled_image_id} if compiled_image_id else {}),
            **({"REMNANT_GLM_CPU_REBUILD_FROM": os.environ["REMNANT_GLM_CPU_REBUILD_FROM"]} if os.environ.get("REMNANT_GLM_CPU_REBUILD_FROM") else {}),
            "REMNANT_GLM_MODEL_CONFIG": "/opt/remnant-glm-config.json",
            "REMNANT_GLM_COMPILE_MANIFEST": "/opt/remnant-glm-compile.json",
            **({"TILELANG_DEFAULT_TARGET": json.dumps({"kind": "cuda", "arch": "sm_90a"}), "TILELANG_CACHE_DIR": "/opt/remnant-tilelang-glm"} if MODEL_FAMILY == "glm53" else {}),
        }
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
    "volumes": {str(RESULTS_ROOT): results_volume, "/triton-cache": triton_volume},
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


@app.function(**{**_GPU, "timeout": 1200 if MODEL_FAMILY == "glm53" else 900})
def benchmark_kernels(
    run_id: str,
    keep_count: int = 256,
    modes: str = "",
    suite: str = "all",
    prefill_reader: str = "split",
    model_family: str = "dsv4", max_seconds: int = RUN_BUDGETS_S["timing"],
    qualification_runs: str = "", resume_run: str = "",
    allow_incomplete_qualification: bool = False, phase: str = "all",
) -> str:
    """Validate and time the complete matrix for the selected survivor budget."""
    if model_family != MODEL_FAMILY:
        raise ValueError("--model-family must match REMNANT_MODEL_FAMILY")
    if model_family == "glm53":
        if phase not in ("all", "decode"):
            raise ValueError("GLM phase must be all or decode")
        if phase == "decode" and allow_incomplete_qualification:
            raise ValueError("Decode timing requires all focused gates")
        if suite != "glm-attention" or keep_count != 256 or modes not in ("", "native,direct") or prefill_reader != "split":
            raise ValueError("GLM timing requires --suite glm-attention and exactly native,direct")
        if not 60 <= max_seconds <= 1140:
            raise ValueError("GLM timing max_seconds must be between 60 and 1140")
        with _run_context(run_id, "timing", max_seconds) as (root, record, deadline):
            _verify_glm_image()
            record["qualification"] = _glm_timing_qualification(
                qualification_runs, allow_incomplete=allow_incomplete_qualification, phase=phase
            )
            record["experiment_status"] = (
                "unqualified" if record["qualification"]["status"] != "passed"
                else "qualified"
            )
            _run_pytest((GLM_DECODE_GRAPH if phase == "decode" else GLM_BENCHMARK_GRAPH,), root, "benchmark-graphs", deadline)
            argv = _benchmark_command("--model-family", "glm53", "--suite", suite,
                                      "--phase", phase, "--output", str(root / "timing"),
                                      "--max-seconds", str(max(60, int(deadline-time.monotonic())-10)))
            if resume_run:
                checkpoint = _glm_timing_checkpoint(resume_run)
                argv += ["--resume", str(checkpoint)]
                record["resume_run"] = resume_run
            record.update(command=argv, modes=("native", "direct"), suite=suite)
            _run_command(argv, root / "timing.log", deadline)
            _glm_benchmark_contract().validate_timing(json.loads((root / "timing.json").read_text()))
        return str(root)
    if qualification_runs or resume_run or allow_incomplete_qualification or phase != "all":
        raise ValueError("qualification_runs/resume_run/allow_incomplete_qualification are GLM-only")
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
    }
)
def validate_kernels(
    run_id: str, sanitizer: bool = False, max_seconds: int = RUN_BUDGETS_S["validation"],
    model_family: str = "dsv4", stage: str = "numerical", cases: str = "", sanitizer_tools: str = "",
    sanitizer_scope: str = "all",
) -> str:
    """Run existing GPU tests; skipped tests fail coverage. Optional three sanitizers."""
    if model_family != MODEL_FAMILY:
        raise ValueError("--model-family must match REMNANT_MODEL_FAMILY used to prepare the image")
    if stage not in ("numerical", "sanitizer", "corrective"):
        raise ValueError("stage must be numerical, sanitizer or corrective")
    if stage == "corrective":
        selected_tests = tuple(cases.split(","))
        if MODEL_FAMILY != "glm53" or not cases or any(not case.startswith(GLM_TEST + "::test_") for case in selected_tests):
            raise ValueError("GLM corrective stage requires explicit --cases pytest node IDs")
    else:
        if cases:
            raise ValueError("--cases is only supported for the GLM corrective stage")
        selected_tests = VALIDATION_TESTS
    if sanitizer_scope not in ("all", "glm-decode"):
        raise ValueError("sanitizer_scope must be all or glm-decode")
    if sanitizer_scope == "glm-decode" and (MODEL_FAMILY != "glm53" or stage != "corrective"
            or not sanitizer or set(selected_tests) != {GLM_TEST + "::test_packed_loader_bytes", GLM_DECODE_GRAPH}):
        raise ValueError("glm-decode sanitizer scope requires the exact focused decode tests")
    sanitizer = sanitizer or stage == "sanitizer"
    selected_tools = _selected_sanitizer_tools(sanitizer_tools, sanitizer, model_family)
    maximum = min(2640, VALIDATION_TIMEOUT_S - 60)
    if not 60 <= max_seconds <= maximum:
        raise ValueError(f"max_seconds must be between 60 and {maximum}")
    with _run_context(run_id, "validation", max_seconds) as (
        root,
        record,
        deadline,
    ):
        environment = dict(os.environ)
        record["sanitizer"] = sanitizer
        record["sanitizer_scope"] = sanitizer_scope
        record["selected_sanitizer_tools"] = selected_tools
        record["sanitizer_qualification_complete"] = False
        if MODEL_FAMILY == "glm53":
            record["runtime_hashes"] = _glm_benchmark_contract().runtime_hashes(SGLANG_ROOT / "python/sglang")
        if sanitizer:
            record["sanitizer_results"] = {
                tool: {"status": "not_run"} for tool in SANITIZER_ORDER
            }
        if INTEGRATION["version"] == "v0.5.22":
            import torch

            if (
                torch.cuda.get_device_capability() != (9, 0)
                or "H100" not in torch.cuda.get_device_name()
            ):
                raise RuntimeError("Qualification requires one H100/SM90")
            if "cpu-driver-stubs" in environment.get("LD_LIBRARY_PATH", ""):
                raise RuntimeError("CPU-only driver stubs leaked into the GPU loader")
            manifest = json.loads(MANIFEST.read_text())
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
                expected_variants = {(h, i, p) for h in (8, 16, 32, 64) for i in (1, 2, 33, 66) for p in (False, True)}
                if not compiled or compiled.get("attention_scale") != 0.0625 or {(v["heads"], v["inner"], v["packed"]) for v in compiled["tilelang"]} != expected_variants:
                    raise RuntimeError("Incomplete CPU-compiled GLM TileLang variants")
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
            _run_pytest(VALIDATION_TESTS, root, "validation", deadline, environment=environment)
        if INTEGRATION["version"] == "v0.5.22" and (stage == "numerical" or (stage == "corrective" and GLM_TEST + "::test_real_gpu_allocations" in selected_tests)):
            allocation = json.loads((root / "allocation.json").read_text())
            if allocation["status"] != "passed" or {
                r["mode"] for r in allocation["modes"]
            } != ({"native", "adapter", "direct"} if MODEL_FAMILY == "glm53" else {"native", "triton", "kernel"}):
                raise RuntimeError("Incomplete real-allocation coverage")
            record["allocation"] = allocation
        if sanitizer and sanitizer_scope == "glm-decode":
            audit_prefix = (SANITIZER, "--tool", "racecheck", "--error-exitcode", "1",
                            "--kernel-name", "regex=glm_packed_(partial|bytes)", "--racecheck-trace-sync", "yes")
            _run_pytest((GLM_TEST + "::test_packed_loader_bytes",), root, "scope-audit",
                        min(deadline, time.monotonic() + 180), prefix=audit_prefix)
            audit_log = (root / "scope-audit.log").read_text()
            _check_sanitizer_report(audit_log, "racecheck")
            _check_glm_scope_audit(audit_log, run_id)
            record["scope_audit"] = dict(status="passed", filter="regex=glm_packed_(partial|bytes)",
                log_sha256=hashlib.sha256(audit_log.encode()).hexdigest(),
                entrypoints=manifest["glm_compile"]["scope_symbol_audit"],
                basis="Racecheck synchronization trace; other tools use completed tests, identical entrypoints and common filter contract")
        if sanitizer:
            for tool in selected_tools:
                # Commands use whole-second timeouts. Preserve the incomplete
                # tool matrix instead of attempting a launch with zero seconds.
                if int(deadline - time.monotonic()) <= 0:
                    break
                prefix = (SANITIZER, "--tool", tool, "--error-exitcode", "1")
                if sanitizer_scope == "glm-decode":
                    # Distinct Packed entrypoints include their inline helper.
                    # One exact production CTA per bank; stock Native partial
                    # is the control. Full attention has its own numerical gate.
                    prefix += ("--kernel-name", "regex=glm_packed_(partial|bytes)", "--print-level", "info")
                try:
                    tool_environment = dict(environment)
                    if sanitizer_scope == "glm-decode":
                        tool_environment["PYTEST_ADDOPTS"] = "--full-trace -o faulthandler_timeout=120"
                    _run_pytest((GLM_TEST + "::test_packed_loader_bytes", GLM_DECODE_PARTIAL) if sanitizer_scope == "glm-decode"
                                else (selected_tests if stage == "corrective" else SANITIZER_TESTS),
                                root, tool, deadline, prefix=prefix,
                                environment=tool_environment if sanitizer_scope == "glm-decode" else None)
                except TimeoutError:
                    # The boundary may expire between this loop and the command
                    # check. No process started; keep this tool marked not_run.
                    break
                except RuntimeError as error:
                    record["sanitizer_results"][tool] = {
                        "status": "failed",
                        "error": str(error),
                    }
                    # Each tool owns a fresh pytest process. Retain later tools'
                    # evidence without converting this failure into a pass.
                else:
                    record["sanitizer_results"][tool] = {"status": "passed"}
                if MODEL_FAMILY == "glm53":
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
            record["qualification_scope"] = "focused_decode" if focused else stage
            record["focused_decode_qualification_complete"] = focused and all_tools_passed
            record["sanitizer_qualification_complete"] = all_tools_passed and stage != "corrective"
            if incomplete:
                raise RuntimeError(
                    f"Sanitizer qualification incomplete: {incomplete}; see tool logs"
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
            for section in ([] if is_prefill else NCU_SECTIONS):
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


def _run_command(
    argv: Sequence[str],
    output: Path,
    deadline: float,
    *,
    environment: dict[str, str] | None = None,
) -> str:
    """Bound the child by the allocation budget; retain its output on failure."""
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
    try:
        with output.open("w") as log:
            subprocess.run(
                list(argv),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=remaining,
                cwd=SGLANG_ROOT,
                env=child_environment,
            )
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
    argv = [*prefix, sys.executable, *["-m", "pytest", "-q"]]
    argv += [str(SGLANG_ROOT / path) for path in tests]
    argv += ["--collect-only"] if collect else [f"--junitxml={report}"]
    output = _run_command(argv, root / f"{name}.log", deadline, environment=environment)
    if not collect:
        try:
            counts = {
                key: sum(
                    int(s.get(key, 0))
                    for s in ET.parse(report).getroot().iter("testsuite")
                )
                for key in ("tests", "errors", "failures", "skipped")
            }
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



def _selected_sanitizer_tools(value: str, enabled: bool, family: str) -> tuple[str, ...]:
    if value and (not enabled or family != "glm53"):
        raise ValueError("--sanitizer-tools requires GLM sanitizer validation")
    tools = tuple(value.split(",")) if value else SANITIZER_ORDER
    if not tools or len(set(tools)) != len(tools) or set(tools) - set(SANITIZER_ORDER):
        raise ValueError("--sanitizer-tools must select unique memcheck,racecheck,synccheck")
    return tools


def _glm_benchmark_contract():
    path = (SGLANG_SOURCE if modal.is_local() else SGLANG_ROOT) / "python/sglang/benchmark/remnant/glm_attention.py"
    spec = importlib.util.spec_from_file_location("remnant_glm_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verify_glm_image() -> None:
    """Verify image inputs and CPU-produced binaries before benchmark work."""
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("model_family") != "glm53" or manifest["source_identities"] != SOURCE_IDENTITIES or manifest["sources"] != _source_hashes():
        raise RuntimeError("GLM timing image source identity mismatch")
    for artifacts in (manifest["binaries"], manifest["glm_compile"]["cache_files"]):
        for path, expected in artifacts.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"GLM timing artifact changed: {path}")
    from sglang.srt.remnant.glm import compile_kernels
    if [k.hash for k in compile_kernels()] != manifest["glm_compile"]["triton"]:
        raise RuntimeError("GLM timing Triton variant mismatch")


def _glm_timing_checkpoint(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("Invalid GLM checkpoint run ID")
    checkpoint = RESULTS_ROOT / "microbench" / run_id / "timing" / "timing.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GLM checkpoint not found: {checkpoint}")
    return checkpoint


def _glm_timing_qualification(run_ids: str, *, allow_incomplete: bool = False, phase: str = "all") -> dict:
    """Accept per-tool evidence only with unchanged production code/toolchains.

    A prior overall failed run may supply a completed synccheck group; its
    failed or not-run groups cannot qualify. New partial-tool runs explicitly
    remain incomplete as full qualifications until merged here.
    """
    ids = run_ids.split(",")
    if not run_ids or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", x) for x in ids):
        raise ValueError("GLM timing requires explicit --qualification-runs IDs")
    manifest = json.loads(MANIFEST.read_text())
    current = _glm_benchmark_contract().runtime_hashes(SGLANG_ROOT / "python/sglang")
    groups = {}
    for run_id in ids:
        root = RESULTS_ROOT / "microbench" / run_id / "validation"
        result = json.loads((root / "result.json").read_text())
        old = json.loads((root / "build-manifest.json").read_text())
        if result.get("model_family") != "glm53":
            raise RuntimeError(f"Not GLM qualification: {run_id}")
        if result.get("runtime_hashes") is not None:
            if result["runtime_hashes"] != current:
                raise RuntimeError(f"GLM runtime hashes changed: {run_id}")
        elif old["source_identities"]["sglang"]["commit"] != GLM_BASELINE or current != GLM_BASELINE_RUNTIME_HASHES:
            raise RuntimeError(f"Cannot establish unchanged historical GLM runtime: {run_id}")
        if phase == "decode" and (old.get("sources") != manifest.get("sources")
                or old['glm_compile'].get('cache_files') != manifest['glm_compile'].get('cache_files')
                or old.get('nvcc') != manifest.get('nvcc')):
            raise RuntimeError(f"GLM focused fixture/binary/source identity changed: {run_id}")
        for key in ("torch", "cuda", "cuda_python", "cuda_bindings", "triton", "tilelang", "binaries"):
            if old.get(key) != manifest.get(key):
                raise RuntimeError(f"GLM qualification dependency changed: {run_id}/{key}")
        for key in ("triton", "tilelang", "attention_scale"):
            if old['glm_compile'][key] != manifest['glm_compile'][key]:
                raise RuntimeError(f"GLM compiled variant identity changed: {run_id}/{key}")
        if old['sanitizer']['binary_sha256'] != manifest['sanitizer']['binary_sha256']:
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
                _check_glm_interrupted_checkpoint(json.loads(note.read_text()), root, run_id, tool, log)
            _check_sanitizer_report(log, tool)
            if phase == "decode" and result.get("sanitizer_scope") == "glm-decode":
                audit = result.get("scope_audit", {})
                audit_log = (root / "scope-audit.log").read_text()
                if (audit.get("status") != "passed" or audit.get("filter") != "regex=glm_packed_(partial|bytes)"
                        or audit.get("log_sha256") != hashlib.sha256(audit_log.encode()).hexdigest()
                        or audit.get("entrypoints") != manifest["glm_compile"].get("scope_symbol_audit")):
                    raise RuntimeError(f"Focused instrumentation audit changed: {run_id}")
                _check_sanitizer_report(audit_log, "racecheck")
                _check_glm_scope_audit(audit_log, run_id)
            if phase == "decode":
                tree = ET.parse(root / f"{tool}.xml")
                tests = list(tree.iter("testcase"))
                names = {x.attrib["name"] for x in tests}
                if (not names or not names <= GLM_DECODE_REQUIRED or len(names) != len(tests)
                        or any(not x.get("classname", "").endswith("attention.unittests.test_glm5_remnant") for x in tests)
                        or any(x.find(k) is not None for x in tests for k in ("failure", "error", "skipped"))):
                    raise RuntimeError(f"Incomplete focused cases: {run_id}/{tool}")
                for node in tests:
                    name = node.attrib["name"]
                    if name.startswith("test_decode_partial_graph["):
                        heads, inner = map(int, name.split("[")[1].rstrip("]").split("-"))
                        props = {x.get("name"): x.get("value") for x in node.findall("properties/property")}
                        expected_geometry = dict(heads=heads, inner=inner, banks=2, queries=1,
                                                 tails=[0,1,2,3], nondefault_stream=True,
                                                 launched_partitions=1, binary_partitions=66//inner)
                        cubins = {str(v["packed"]):v["cubin_sha256"] for v in manifest["glm_compile"]["tilelang"]
                                  if v["heads"] == heads and v["inner"] == inner}
                        if json.loads(props.get("partial_cubins", "{}")) != cubins:
                            raise RuntimeError(f"Focused production cubin changed: {run_id}/{name}")
                        if json.loads(props.get("decode_geometry", "{}")) != expected_geometry:
                            raise RuntimeError(f"Focused decode geometry changed: {run_id}/{name}")
            else:
                names = _check_glm_sanitizer_cases(ET.parse(root / f"{tool}.xml"), run_id,
                                                 historical=result.get('runtime_hashes') is None,
                                                 allow_partial=True)
            group = groups.setdefault(tool, dict(cases=set(), runs=[]))
            group['cases'].update(names)
            group['runs'].append(dict(run_id=run_id, cases=sorted(names),
                                      overall_run_status=result.get('status'),
                                      tool_status=status.get('status'),
                                      result_sha256=hashlib.sha256((root / 'result.json').read_bytes()).hexdigest()))
    required = GLM_DECODE_REQUIRED if phase == "decode" else _glm_required_sanitizer_cases(historical=False)
    missing = {tool: sorted(required - groups.get(tool, {}).get('cases', set())) for tool in SANITIZER_ORDER}
    incomplete = any(missing.values())
    if incomplete and not allow_incomplete:
        raise RuntimeError(f"GLM timing blocked by missing sanitizer gates: {missing}")
    return dict(status="incomplete" if incomplete else "passed", missing=missing,
                tools={tool: dict(cases=sorted(group['cases']), runs=group['runs'])
                                        for tool, group in groups.items()}, runtime_hashes=current)


def _check_glm_scope_audit(log: str, identity: str) -> None:
    """Pinned tool trace proves helper warp and handoff CTA synchronization."""
    if ("glm_packed_bytes" not in log or not re.search(r"(?i)(warp|warpsync)", log)
            or not re.search(r"(?i)(block|cta|barrier)", log)):
        raise RuntimeError(f"Missing positive TileLang synchronization trace: {identity}")


def _check_glm_interrupted_checkpoint(note: dict, root: Path, run_id: str, tool: str, log: str) -> None:
    if (note.get('schema') != 'remnant-glm-sanitizer-checkpoint-v1'
            or note.get('run_id') != run_id or note.get('tool') != tool
            or note.get('action') != 'intentional SIGINT to owned pytest'
            or 'KeyboardInterrupt' not in log):
        raise RuntimeError(f"Invalid intentional sanitizer checkpoint: {run_id}/{tool}")
    for suffix in ('log', 'xml'):
        actual = hashlib.sha256((root / f'{tool}.{suffix}').read_bytes()).hexdigest()
        if note.get(f'{suffix}_sha256') != actual:
            raise RuntimeError(f"Sanitizer checkpoint report changed: {run_id}/{tool}.{suffix}")


def _glm_required_sanitizer_cases(*, historical: bool) -> set[str]:
    required = {
        "test_producer_reconstruction", "test_native_fp8_conversion", "test_allocator_accounting",
        "test_invalid_attention", "test_partial_outputs", "test_attention[3-8-8]",
        "test_changing_input_graph[adapter]", "test_changing_input_graph[direct]",
        "test_production_backend_constructor", "test_attention[0-1-64]", "test_pool_lifecycle",
    }
    if not historical:
        required |= {"test_benchmark_sanitizer_bank_graph[8-2-2-33]", "test_benchmark_sanitizer_bank_graph[16-4-2-66]"}
    return required


def _check_glm_sanitizer_cases(tree: ET.ElementTree, run_id: str, *, historical: bool,
                              allow_partial: bool = False) -> set[str]:
    required = _glm_required_sanitizer_cases(historical=historical)
    nodes = list(tree.getroot().iter("testcase"))
    names = {node.get("name") for node in nodes
             if node.get("classname", "").endswith("attention.unittests.test_glm5_remnant")}
    suites = list(tree.getroot().iter("testsuite"))
    if (not names or not names <= required or (not allow_partial and not required <= names)
            or any(node.find(k) is not None for node in nodes for k in ("failure", "error", "skipped"))
            or any(int(x.get(k, 0)) for x in suites for k in ("errors", "failures", "skipped"))
            or sum(int(x.get('tests', 0)) for x in suites) != len(names)):
        raise RuntimeError(f"Incomplete GLM sanitizer fixture: {run_id}; missing={sorted(required-names)}")
    for node in nodes:
        if node.get('name', '').startswith('test_benchmark_sanitizer_bank_graph['):
            dimensions = tuple(map(int, node.get('name').split('[')[1].rstrip(']').split('-')))
            properties = {x.get('name'): x.get('value') for x in node.findall('properties/property')}
            geometry = json.loads(properties.get('probe_geometry', '{}'))
            if tuple(geometry.get(k) for k in ('heads', 'banks', 'queries', 'forced_inner')) != dimensions or geometry.get('warmup_calls') != 1 or geometry.get('measured_warmup_calls') != 10 or geometry.get('actual_warmup_calls_per_mode') != dimensions[1]:
                raise RuntimeError(f"Missing focused sanitizer geometry: {run_id}/{node.get('name')}")
    return names


def _check_sanitizer_report(output: str, tool: str) -> None:
    expected = ("RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                if tool == "racecheck" else "ERROR SUMMARY: 0 errors")
    records = [r.strip() for r in re.findall(r"========= ([^\r\n]*)", output)]
    summaries = [r for r in records if r.startswith(("ERROR SUMMARY:", "RACECHECK SUMMARY:"))]
    if summaries != [expected] or any(r.startswith(("Warning:", "Error:")) for r in records) or "terminated before first instrumented API call" in output:
        raise RuntimeError(f"Incomplete sanitizer evidence {tool}")


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
    run_id: str, action: str, seconds: int
) -> Iterator[tuple[Path, dict, float]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("run_id must be a unique simple identifier")
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
        record["gpu_function_seconds"] = time.monotonic() - started
        (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        try:
            results_volume.commit()
        finally:
            triton_volume.commit()
