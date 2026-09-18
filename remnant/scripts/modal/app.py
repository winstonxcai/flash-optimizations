"""Modal resources only; serving runs the same Bash script as local H100s."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
MODEL_ROOT = Path("/models")
MODEL_DIR = MODEL_ROOT / "DeepSeek-V4-Flash-0731"
RESULTS_ROOT = Path("/results")
SGLANG_ROOT = Path("/sgl-workspace/sglang-lowrank")
REMOTE_REPO = Path("/opt/remnant/flash-optimizations")


def _repo_root() -> Path:
    """Resolve the repository locally and from Modal's mounted /root/app.py."""
    candidates = (
        Path.cwd(),
        Path(__file__).resolve().parent,
        REMOTE_REPO,
    )
    for candidate in candidates:
        for root in (candidate, *candidate.parents):
            if (root / "remnant").is_dir():
                return root
    return REMOTE_REPO


REPO_ROOT = _repo_root()

app = modal.App("remnant")
model_volume = modal.Volume.from_name("deepseek-v4-flash-0731", create_if_missing=True)
# Keep the existing volume name: renaming the public modes must not orphan results.
results_volume = modal.Volume.from_name(
    "remnant-stage2a-results", create_if_missing=True
)
download_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface-hub[hf-xet]==0.34.4",
)
server_image = (
    modal.Image.from_dockerfile(
        str(REPO_ROOT / "remnant" / "docker" / "modal.Dockerfile"),
        context_dir=str(REPO_ROOT),
        # Keep test_fused.py in the build context: the Dockerfile runs it.
        ignore=(
            "remnant/scripts/**",
            "remnant/tests/bench_*.py",
            "remnant/tests/test_bench_serving.py",
            "remnant/tests/fixtures/**",
        ),
    )
    .apt_install("curl", "jq", "util-linux", "coreutils")
    .add_local_dir(
        REPO_ROOT / "remnant" / "scripts",
        REMOTE_REPO / "remnant" / "scripts",
    )
    .add_local_dir(
        REPO_ROOT / "remnant" / "tests",
        REMOTE_REPO / "remnant" / "tests",
    )
)


@app.function(
    image=download_image,
    cpu=8,
    memory=32768,
    timeout=6 * 3600,
    volumes={str(MODEL_ROOT): model_volume},
)
def download_model() -> str:
    """CPU-only pinned checkpoint download; existing shards are reused."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        max_workers=4,
    )
    model_volume.commit()
    return path


@app.function(
    image=server_image,
    gpu="H100!:4",
    cpu=32,
    memory=262144,
    timeout=4 * 3600 + 60,
    retries=0,
    volumes={str(MODEL_ROOT): model_volume, str(RESULTS_ROOT): results_volume},
)
def bench_serving(
    mode: str = "native",
    input_tokens: int = 32768,
    output_tokens: int = 2048,
    concurrency: int = 8,
    timeout_minutes: int = 60,
) -> str:
    """One configuration per call, identical to the local shell command."""
    if not 1 <= timeout_minutes <= 240:
        raise ValueError("timeout_minutes must be 1–240")
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "MODEL_PATH": str(MODEL_DIR),
        "SGLANG_ROOT": str(SGLANG_ROOT),
        "RESULTS_DIR": str(RESULTS_ROOT),
    }
    try:
        subprocess.run(
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=30s",
                f"{timeout_minutes}m",
                "bash",
                str(REMOTE_REPO / "remnant/scripts/local/bench-serving.sh"),
                mode,
                str(input_tokens),
                str(output_tokens),
                str(concurrency),
            ],
            env=env,
            cwd=REMOTE_REPO,
            check=True,
        )
    finally:
        results_volume.commit()
    return str(RESULTS_ROOT)


def _kernel_run(
    modules: list[str],
    *,
    kind: str,
    timeout: int,
    sanitizer: bool = False,
    arguments: list[str] | None = None,
) -> str:
    """Keep model-free kernel checks separate from serving."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = RESULTS_ROOT / f"{stamp}-{kind}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SGLANG_OPT_TOPMAG", "KEEP"))
    }
    env.update(
        SGLANG_OPT_TOPMAG="1",
        KEEP="0.5",
        SGLANG_OPT_TOPMAG_PACKED="1",
        SGLANG_OPT_TOPMAG_FUSED="0",
        REMNANT_RESULTS_DIR=str(directory),
        REMNANT_FUSED_RESULTS_DIR=str(directory),
    )
    module_arguments = arguments or []
    commands = [
        [sys.executable, "-m", module, *module_arguments] for module in modules
    ]
    if sanitizer:
        commands.append(
            [
                "compute-sanitizer",
                "--tool",
                "memcheck",
                "--error-exitcode",
                "99",
                sys.executable,
                "-m",
                modules[-1],
                *module_arguments,
                "--sanitizer-case",
            ]
        )
    try:
        for i, command in enumerate(commands):
            with (directory / f"{i}.log").open("w") as log:
                subprocess.run(
                    command,
                    env=env,
                    cwd=REMOTE_REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=timeout,
                )
    finally:
        results_volume.commit()
    return str(directory)


@app.function(
    image=server_image,
    gpu="H100!",
    timeout=3600,
    retries=0,
    volumes={str(RESULTS_ROOT): results_volume},
)
def validate_packed() -> str:
    """H100: every validity stage, every available leg, over the case grid."""
    return _kernel_run(
        ["remnant.tests.validity"], kind="validate-packed", timeout=3500
    )


@app.function(
    image=server_image,
    gpu="L4",
    timeout=1800,
    retries=0,
    volumes={str(RESULTS_ROOT): results_volume},
)
def validate_fused() -> str:
    """L4: fused adapter correctness, graph/stream checks, and memcheck.

    The sanitizer pass runs the same entrypoint narrowed to the smallest workload
    at the default pattern and the two fused implementations.
    """
    return _kernel_run(
        ["remnant.tests.validity"],
        kind="validate-packed-fused",
        timeout=1700,
        sanitizer=True,
        arguments=["--legs", "fused,fused.optimized"],
    )


@app.function(
    image=server_image,
    gpu="H100!",
    timeout=1800,
    retries=0,
    volumes={str(RESULTS_ROOT): results_volume},
)
def bench_kernels(
    suite: str = "speed",
    focused_128k: bool = False,
    decode_samples: int = 500,
    decode_rounds: int = 3,
    decode_selection_sets: int = 32,
    decode_modes: str = "",
    decode_batches: str = "",
) -> str:
    """H100: kernel and store-side benchmark suites.

    The candidate legs live in one module now, so the legacy ``packed``/``fused``
    suite names all select it.
    """
    modules = {
        "speed": "remnant.tests.speed",
        "packed": "remnant.tests.speed",
        "fused": "remnant.tests.speed",
        "decode": "remnant.tests.speed",
        "store": "remnant.tests.speed",
    }
    if suite not in modules:
        raise ValueError(f"suite must be one of {tuple(modules)}")
    arguments = []
    if suite == "fused":
        arguments.extend(["--legs", "fused,fused.optimized"])
    if focused_128k:
        arguments.append("--focused-128k")
    if suite == "decode":
        arguments.extend(
            [
                "--production-decode",
                "--decode-samples",
                str(decode_samples),
                "--decode-rounds",
                str(decode_rounds),
                "--decode-selection-sets",
                str(decode_selection_sets),
            ]
        )
        if decode_modes:
            arguments.extend(["--decode-modes", decode_modes])
        if decode_batches:
            arguments.extend(["--decode-batches", decode_batches])
    if suite == "store":
        arguments.append("--store-breakdown")
    return _kernel_run(
        [modules[suite]],
        kind=("bench-production-decode" if suite == "decode" else
              "bench-store-breakdown" if suite == "store" else
              "bench-speed-128k" if focused_128k else "bench-speed"),
        timeout=1700,
        arguments=arguments,
    )


@app.function(
    image=server_image,
    gpu="H100!",
    timeout=1800,
    retries=0,
    volumes={str(RESULTS_ROOT): results_volume},
)
def profile_decode() -> str:
    """H100: short graph trace and targeted optimized reconstruction profile."""
    for tool in ("nsys", "ncu"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is not installed in the server image")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = RESULTS_ROOT / f"{stamp}-profile-decode-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("SGLANG_OPT_TOPMAG", "KEEP"))
    }
    env.update(
        SGLANG_OPT_TOPMAG="1",
        KEEP="0.5",
        SGLANG_OPT_TOPMAG_PACKED="1",
        SGLANG_OPT_TOPMAG_FUSED="1",
        SGLANG_OPT_TOPMAG_FUSED_OPTIMIZED="1",
        REMNANT_RESULTS_DIR=str(directory),
        REMNANT_FUSED_RESULTS_DIR=str(directory),
    )
    target = [
        sys.executable,
        "-m",
        "remnant.tests.speed",
        "--production-decode",
        "--decode-samples",
        "1",
        "--decode-rounds",
        "1",
        "--decode-selection-sets",
        "1",
        "--decode-modes",
        "optimized",
        "--decode-batches",
        "21",
    ]
    commands = {
        "nsys": [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
            "--output",
            str(directory / "decode-nsys"),
            *target,
        ],
        "ncu": [
            "ncu",
            "--set",
            "full",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "function",
            "--kernel-name",
            "packed_to_native_kernel_optimized",
            "--clock-control",
            "none",
            "--launch-count",
            "1",
            "--export",
            str(directory / "decode-ncu"),
            "--force-overwrite",
            *target,
        ],
    }
    try:
        for name, command in commands.items():
            with (directory / f"{name}.log").open("w") as log:
                subprocess.run(
                    command,
                    env=env,
                    cwd=REMOTE_REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=700,
                )
        # Keep the raw report for offline inspection.  Report names differ
        # across Nsight Systems versions; invoking a guessed report here can
        # produce a successful-looking artifact containing only error text.
        # The trace is parsed locally after download instead.
        ncu_report = directory / "decode-ncu.ncu-rep"
        with (directory / "ncu-stats.csv").open("w") as stats:
            subprocess.run(
                ["ncu", "--import", str(ncu_report), "--page", "raw", "--csv"],
                stdout=stats,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
    finally:
        results_volume.commit()
    return str(directory)


@app.function(
    image=server_image, gpu="H100!", cpu=4, memory=16384,
    timeout=900, retries=0, scaledown_window=2,
    volumes={str(RESULTS_ROOT): results_volume},
)
def profile_reconstruction(
    phase: str = "initial", modes: str = "fused,generic,optimized,combined",
    profile_mode: str = "optimized",
) -> str:
    """Bounded model-free reconstruction pass; at most 900s per invocation.

    Reserve each invocation against the experiment's aggregate allocation budget
    before launching. Never auto-retry a failed GPU allocation.
    """
    if phase not in ("initial", "confirm", "timing", "profile", "sanitizer"):
        raise ValueError("unknown reconstruction phase")
    if profile_mode not in ("fused", "generic", "optimized", "combined"):
        raise ValueError("unknown reconstruction profile mode")
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = RESULTS_ROOT / f"{stamp}-reconstruction-{phase}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SGLANG_OPT_TOPMAG", "KEEP"))}
    env.update(SGLANG_OPT_TOPMAG="1", KEEP="0.5", SGLANG_OPT_TOPMAG_PACKED="1")
    steps = []

    def execute(name, command, limit=240):
        remaining = 840 - (time.monotonic() - started)
        if remaining < 30:
            raise TimeoutError("reconstruction allocation budget exhausted; reserving export time")
        target_dir = directory / name
        target_dir.mkdir(exist_ok=True)
        child_env = {**env, "REMNANT_RESULTS_DIR": str(target_dir)}
        print(f"[reconstruction] {name}: {remaining:.0f}s allocation work budget left", flush=True)
        begin = time.monotonic()
        with (directory / f"{name}.log").open("w") as log:
            process = subprocess.run(command, cwd=REMOTE_REPO, env=child_env,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     timeout=min(limit, remaining), check=False)
        steps.append({"name": name, "command": command, "seconds": time.monotonic() - begin,
                      "exit_code": process.returncode})
        print(f"[reconstruction] {name}: exit={process.returncode}", flush=True)
        if process.returncode:
            print((directory / f"{name}.log").read_text()[-5000:], flush=True)
            raise RuntimeError(f"{name} failed")

    base = [sys.executable, "-m", "remnant.tests.speed"]
    try:
        if phase in ("initial", "confirm", "timing"):
            execute("timing", base + ["--reconstruction", "timing", "--decode-modes", modes], 300)
        if phase in ("initial", "sanitizer"):
            for tool in ("memcheck", "racecheck", "synccheck"):
                execute(tool, ["compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                               *base, "--reconstruction", "sanitizer", "--decode-modes", modes], 120)
        if phase in ("initial", "confirm", "profile"):
            execute("sections", ["ncu", "--list-sections"])
            section_text = (directory / "sections.log").read_text()
            sections = ["LaunchStats", "Occupancy", "SchedulerStats", "WarpStateStats",
                        "MemoryWorkloadAnalysis", "SourceCounters"]
            if any(section not in section_text for section in sections):
                raise RuntimeError("installed NCU lacks required sections; inspect sections.log")
            target = base + ["--reconstruction", "profile", "--decode-batches", "21",
                             "--decode-modes", profile_mode, "--decode-samples", "8"]
            execute("nsys", ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
                             "--cuda-graph-trace=node", "--force-overwrite=true", "-o",
                             str(directory / "reconstruction"), *target], 120)
            section_args = [arg for section in sections for arg in ("--section", section)]
            execute("ncu", ["ncu", *section_args, "--nvtx", "--nvtx-include", "reconstruction-profile/",
                            "--clock-control", "none", "--cache-control", "none",
                            "--launch-count", "3", "--import-source", "yes", "-o",
                            str(directory / "reconstruction"), "--force-overwrite", *target], 240)
            report = str(directory / "reconstruction.ncu-rep")
            execute("ncu-raw", ["ncu", "--import", report, "--page", "raw", "--csv"], 60)
            execute("ncu-source", ["ncu", "--import", report, "--page", "source", "--print-source", "cuda,sass"], 60)
            execute("nsys-export", ["nsys", "export", "--type", "sqlite", "--force-overwrite=true",
                                    "-o", str(directory / "reconstruction.sqlite"),
                                    str(directory / "reconstruction.nsys-rep")], 60)
            libraries = list((REMOTE_REPO / "remnant").glob("_fused*.so"))
            if len(libraries) != 1:
                raise RuntimeError("expected one compiled fused extension")
            execute("sass", ["cuobjdump", "--dump-sass", "--dump-resource-usage", str(libraries[0])], 60)
    finally:
        ledger = {"account": os.environ.get("MODAL_WORKSPACE", "fxcai21"), "phase": phase,
                  "function_elapsed_seconds": time.monotonic() - started,
                  "allocation_reservation_seconds": 900, "steps": steps}
        (directory / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
        results_volume.commit()
        print(f"[reconstruction] artifacts: {directory}", flush=True)
    return str(directory)
