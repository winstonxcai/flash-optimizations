"""Model-free H100 validation, timing and Nsight Compute profiles.

Author: Winston Cai.

    modal run scripts/modal/kernels.py::validate_kernels --run-id <unique-id>
    modal run scripts/modal/kernels.py::benchmark_kernels --run-id <unique-id>
    modal run scripts/modal/kernels.py::profile_kernels --run-id <unique-id> --cases <cases>

Defaults: ten validation modules (skips fail); Native/Triton/Kernel timing on eight
decode + 25 store cases, 30 rounds; ten profile cases x three modes = 30 captures.
--cases chooses comma-separated fork cases. Profiles are not acceptance timing.
CPU build/preflight precedes H100; Modal caches source/binaries. Authorization required.
Limits are caps, not ETAs; no weights/retries. Results: remnant-stage2a-results/microbench/<run-id>/<action>.
Download: modal volume get remnant-stage2a-results microbench/<run-id> <local-dir>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import modal

REMOTE_REPO = Path("/opt/flash-optimizations")
REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPO
SGLANG_ROOT = Path("/sgl-workspace/sglang-remnant")
FLASHMLA_ROOT = Path("/opt/flashmla-remnant")
SGLANG_SOURCE = Path(os.getenv("REMNANT_SGLANG_SOURCE_OVERRIDE", str(REPO_ROOT / "third_party/sglang")))
FLASHMLA_SOURCE = Path(os.getenv("REMNANT_FLASHMLA_SOURCE_OVERRIDE", str(REPO_ROOT / "third_party/flashmla")))
if not modal.is_local():
    SGLANG_SOURCE, FLASHMLA_SOURCE = SGLANG_ROOT, FLASHMLA_ROOT
RESULTS_ROOT = Path("/results")
NATIVE_LIBRARY = Path("/opt/remnant-native-store.so")
MANIFEST = Path("/opt/remnant-build.json")
BUILD_ROOT = Path("/opt/remnant-build")
# Preserve existing work caps; reserve time inside the outer timeout for cleanup.
CPU_BUILD_TIMEOUT_S = GPU_TIMEOUT_S = 3600
CPU_BUILD_BUDGET_S = CPU_BUILD_TIMEOUT_S - 200
RUN_BUDGETS_S = {"timing": 850, "validation": 1100, "profile": 1800}
PROFILE_CAPTURE_TIMEOUT_S, PROFILE_EXPORT_TIMEOUT_S = 110, 45
PROFILE_MAX_BUDGET_S = GPU_TIMEOUT_S - 100
TEST_ROOT = "test/registered/attention/unittests/dsv4/"
TEST_MODULES = {
    "cache_format": "test/registered/unit/test_dsv4_c4_cache_format.py",
    **{
        name: TEST_ROOT + f"test_remnant_{name}.py"
        for name in ["pool", "pack", "backend", "compressor_store", "fused_store", "cuda_graph", "flashmla_direct"]
    },
    "pack_kernel": "python/sglang/test/kernels/deepseek_v4/test_remnant_pack_kernel.py",
    "unpack_kernel": "python/sglang/test/kernels/deepseek_v4/test_remnant_unpack_kernel.py",
}
VALIDATION_TESTS = tuple(TEST_MODULES.values())
CPU_TESTS = tuple(TEST_MODULES[name] for name in ("cache_format", "pack", "backend", "compressor_store"))
BENCHMARK_TESTS = ("benchmark/remnant/test_microbench.py", TEST_MODULES["fused_store"])
SANITIZER_TESTS = (
    TEST_MODULES["unpack_kernel"] + "::test_triton_native_pages_match_independent_reference",
    TEST_MODULES["cuda_graph"] + "::TestRemnantCudaGraph::test_store_and_native_reconstruction_replays_changed_inputs",
    TEST_MODULES["pack_kernel"] + "::test_remnant_triton_pack_matches_reference",
    TEST_MODULES["cuda_graph"] + "::TestRemnantCudaGraph::test_consecutive_mixed_layers_replay_with_stable_buffers",
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
    return not (path.parts[0] == "csrc" or path.name.startswith("LICENSE")) or _ignore(path)


def _build_image() -> None:
    """CPU image layer: compile existing SM90 targets, install, then preflight."""
    import torch
    from sglang.kernels.jit.utils import override_jit_cuda_arch

    deadline = time.monotonic() + CPU_BUILD_BUDGET_S
    build = BUILD_ROOT
    build.mkdir()
    aot = SGLANG_ROOT / "python/sglang/kernels/aot"
    # Content identity includes dirty submitted sources, not just Git commits.
    sources = {}
    for label, root in (("sglang", SGLANG_ROOT), ("flashmla", FLASHMLA_ROOT)):
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode() + b"\0")
                digest.update(path.read_bytes())
        sources[label] = digest.hexdigest()
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
        ["cmake", "--build", str(build / "cmake")] + ["--target", "remnant_ops", "flashmla_ops", "--parallel", "2"],
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
    # 3. Precompile the Native store using the fork's dependency-valid cache.
    from sglang.kernels.jit.utils.compile.cache import find_prebuilt
    from sglang.kernels.ops.attention.dsv4.compress import (
        _jit_compress_norm_rope_module,
    )

    with override_jit_cuda_arch(9, 0, "a"):
        _jit_compress_norm_rope_module(torch.float32, 512, 64, 64, False)
    cache = Path(os.environ["SGLANG_JIT_CACHE_DIR"])
    scopes = {p.parent.parent for p in cache.rglob("*.so")}
    libraries = {p for scope in scopes if (p := find_prebuilt(scope=scope, module_name=scope.parent.name)) is not None}
    if len(libraries) != 1:
        raise RuntimeError(f"Expected one dependency-valid Native library, found {libraries}")
    shutil.copy2(libraries.pop(), NATIVE_LIBRARY)
    binaries.append(NATIVE_LIBRARY)
    # 4. Preflight imports, coverage and profiler availability on CPU.
    _run_command(_benchmark_command("--check-imports"), build / "imports.log", deadline)
    _run_pytest(VALIDATION_TESTS, build, "collection", deadline, collect=True)
    _run_pytest(CPU_TESTS, build, "cpu-tests", deadline)
    _run_pytest(BENCHMARK_TESTS, build, "benchmark-collection", deadline, collect=True)
    sections = _run_command(["ncu", "--list-sections"], build / "sections.log", deadline)
    if any(section not in sections for section in NCU_SECTIONS):
        raise RuntimeError("CPU preflight: Nsight Compute lacks required sections; see sections.log")
    # 5. Save exact source/binary provenance for every GPU run.
    manifest = {
        "sources": sources,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "arch": "sm_90a",
        "native_dtype": "float32",
        "binaries": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries},
        "nvcc": _run_command(["nvcc", "--version"], build / "nvcc.log", deadline),
        "ncu": _run_command(["ncu", "--version"], build / "ncu.log", deadline),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


app = modal.App("remnant-kernel-dev")
results_volume = modal.Volume.from_name("remnant-stage2a-results", create_if_missing=True)
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


kernel_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.18-cu130")
    .pip_install("typing_extensions==4.15.0")
    .env(
        {
            "PYTHONPATH": f"{SGLANG_ROOT / 'python'}:{REMOTE_REPO}",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "SGLANG_JIT_CACHE_DIR": "/opt/remnant-jit",
            "REMNANT_FLASHMLA_SOURCE_DIR": str(FLASHMLA_ROOT),
            "REMNANT_NATIVE_STORE_LIBRARY": str(NATIVE_LIBRARY),
            "TRITON_CACHE_DIR": "/triton-cache",
        }
    )
    .add_local_dir(SGLANG_SOURCE, str(SGLANG_ROOT), copy=True, ignore=_sglang_ignored)
    .add_local_dir(FLASHMLA_SOURCE, str(FLASHMLA_ROOT), copy=True, ignore=_flashmla_ignored)
    # Include launcher helpers in the image cache key, not just _build_image.
    .add_local_file(REPO_ROOT / "scripts/modal/kernels.py", "/opt/remnant-launcher.py", copy=True)
    .run_function(_build_image, cpu=16, memory=65536, timeout=CPU_BUILD_TIMEOUT_S)
)
_GPU = {
    "image": kernel_image,
    "gpu": "H100!",
    "cpu": 8,
    "memory": 65536,
    "timeout": GPU_TIMEOUT_S,
    "retries": 0,
    "max_containers": 1,
    "scaledown_window": 2,
    "volumes": {str(RESULTS_ROOT): results_volume, "/triton-cache": triton_volume},
}


# Public workflows.


@app.function(**_GPU)
def benchmark_kernels(run_id: str) -> str:
    """Validate and time the existing complete three-mode matrix; no profiling."""
    with _run_context(run_id, "timing", RUN_BUDGETS_S["timing"]) as (root, record, deadline):
        _run_pytest(BENCHMARK_TESTS, root, "validation", deadline)
        argv = _benchmark_command("--suite", "all", "--output", str(root / "timing"))
        record["command"] = argv
        _run_command(argv, root / "timing.log", deadline)
        timing = json.loads((root / "timing.json").read_text())
        _benchmark_cases().validate_timing(timing)
    return str(root)


@app.function(**_GPU)
def validate_kernels(run_id: str, sanitizer: bool = False) -> str:
    """Run existing GPU tests; skipped tests fail coverage. Optional three sanitizers."""
    with _run_context(run_id, "validation", RUN_BUDGETS_S["validation"]) as (root, record, deadline):
        _run_pytest(VALIDATION_TESTS, root, "validation", deadline)
        if sanitizer:
            for tool in ("memcheck", "racecheck", "synccheck"):
                prefix = ("compute-sanitizer", "--tool", tool, "--error-exitcode", "1")
                _run_pytest(SANITIZER_TESTS, root, tool, deadline, prefix=prefix)
        record["sanitizer"] = sanitizer
    return str(root)


@app.function(**_GPU)
def profile_kernels(run_id: str, cases: str = "", max_seconds: int = RUN_BUDGETS_S["profile"]) -> str:
    """Profile selected existing cases in all three modes, separately from timing."""
    if not 300 <= max_seconds <= PROFILE_MAX_BUDGET_S:
        raise ValueError(f"max_seconds must be between 300 and {PROFILE_MAX_BUDGET_S}")
    contract = _benchmark_cases()
    matrix = contract.selected_profile_matrix(cases)
    with _run_context(run_id, "profile", max_seconds) as (root, record, deadline):
        record.update(matrix=matrix, runs=[])
        for suite, case, mode in matrix:
            name = f"{suite}-{case}-{mode}"
            output = root / name
            target = _benchmark_command(
                "--suite", suite, "--case", case, "--profile-mode", mode, "--output", str(output)
            )
            options = {
                "--profile-from-start": "off",
                "--replay-mode": "kernel",
                "--cache-control": "none",
                "--clock-control": "none",
                "--import-source": "yes",
            }
            argv = ["ncu", *[arg for pair in options.items() for arg in pair]]
            for section in NCU_SECTIONS:
                argv += ["--section", section]
            argv += [
                "--metrics",
                "gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
                "--export",
                str(output),
                "--",
                *target,
            ]
            _run_command(argv, root / f"{name}.log", min(deadline, time.monotonic() + PROFILE_CAPTURE_TIMEOUT_S))
            report = output.with_suffix(".ncu-rep")
            if not report.is_file() or not report.stat().st_size:
                raise RuntimeError(f"Missing profile: {report}")
            capture = json.loads(output.with_suffix(".json").read_text())
            if capture["status"] != "passed" or len(capture["cases"]) != 1 or not capture["cases"][0]["valid"]:
                raise RuntimeError(f"Invalid capture: {name}")
            raw = _run_command(
                ["ncu", "--import", str(report), "--page", "raw", "--csv"],
                output.with_suffix(".metrics.csv"),
                min(deadline, time.monotonic() + PROFILE_EXPORT_TIMEOUT_S),
            )
            entry = {"suite": suite, "case": case, "mode": mode, "command": argv, "capture": capture}
            entry["summary"] = contract.profile_summary(raw, capture["cases"][0]["expected_kernels"])
            record["runs"].append(entry)
            (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            results_volume.commit()
        for suite, case in {(s, c) for s, c, _ in matrix}:
            group = [r for r in record["runs"] if (r["suite"], r["case"]) == (suite, case)]
            if len(group) != 3 or len({r["capture"]["cases"][0]["common_inputs_sha256"] for r in group}) != 1:
                raise RuntimeError(f"Profile inputs differ: {suite}/{case}")
    return str(root)


# Shared command execution, validation and result handling.


def _run_command(argv: Sequence[str], output: Path, deadline: float) -> str:
    """Bound the child by the allocation budget; retain its output on failure."""
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError(f"Deadline reached before {list(argv)}")
    try:
        with output.open("w") as log:
            subprocess.run(
                list(argv), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=remaining, cwd=SGLANG_ROOT
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        tail = output.read_text(errors="replace")[-4000:] if output.is_file() else ""
        raise RuntimeError(f"Command failed: {list(argv)}\nLog: {output}\n{tail}") from error
    return output.read_text()


def _benchmark_command(*options: str) -> list[str]:
    script = SGLANG_ROOT / "benchmark/remnant/microbench.py"
    return [sys.executable, str(script), "--native-library", str(NATIVE_LIBRARY), *options]


def _run_pytest(
    tests: Sequence[str],
    root: Path,
    name: str,
    deadline: float,
    *,
    collect: bool = False,
    prefix: Sequence[str] = (),
) -> None:
    report = root / f"{name}.xml"
    argv = [*prefix, sys.executable, *["-m", "pytest", "-q"]]
    argv += [str(SGLANG_ROOT / path) for path in tests]
    argv += ["--collect-only"] if collect else [f"--junitxml={report}"]
    _run_command(argv, root / f"{name}.log", deadline)
    if not collect:
        counts = {
            key: sum(int(s.get(key, 0)) for s in ET.parse(report).getroot().iter("testsuite"))
            for key in ("tests", "errors", "failures", "skipped")
        }
        if not counts["tests"] or any(counts[k] for k in ("errors", "failures", "skipped")):
            raise RuntimeError(f"Incomplete validation {name}: {counts}")


def _benchmark_cases():
    path = (SGLANG_SOURCE if modal.is_local() else SGLANG_ROOT) / "python/sglang/benchmark/remnant/cases.py"
    spec = importlib.util.spec_from_file_location("remnant_modal_cases", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _run_context(run_id: str, action: str, seconds: int) -> Iterator[tuple[Path, dict, float]]:
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
