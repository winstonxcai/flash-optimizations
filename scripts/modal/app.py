"""Modal resources only; serving runs the same Bash script as local H100s."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
MODEL_ROOT = Path("/models")
MODEL_DIR = MODEL_ROOT / "DeepSeek-V4-Flash-0731"
RESULTS_ROOT = Path("/results")
SGLANG_ROOT = Path("/sgl-workspace/sglang-remnant")
REMOTE_REPO = Path("/opt/flash-optimizations")

def _repo_root() -> Path:
    """Resolve the repository locally and from Modal's mounted /root/app.py."""
    candidates = (
        Path.cwd(),
        Path(__file__).resolve().parent,
        REMOTE_REPO,
    )
    for candidate in candidates:
        for root in (candidate, *candidate.parents):
            if (root / "third_party" / "sglang").is_dir():
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
fork_image = modal.Image.from_dockerfile(
    str(REPO_ROOT / "Dockerfile"),
    context_dir=str(REPO_ROOT),
    ignore=(
        "scripts/**",
        "**/.git/**",
        "third_party/sglang/.git",
        "third_party/flashmla/.git",
    ),
)
server_image = (
    fork_image
    .apt_install("curl", "jq", "util-linux", "coreutils")
    .add_local_dir(
        REPO_ROOT / "scripts",
        REMOTE_REPO / "scripts",
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
                str(REMOTE_REPO / "scripts/local/bench-serving.sh"),
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


def _build_sglang_kernel(*, enable_sm100: bool = False) -> None:
    """Build and overlay the local FlashMLA extension for the target GPU."""
    aot_root = SGLANG_ROOT / "python" / "sglang" / "kernels" / "aot"
    build_root = Path("/tmp/remnant-flashmla-build")
    install_root = Path("/tmp/remnant-flashmla-install")
    shutil.rmtree(build_root, ignore_errors=True)
    shutil.rmtree(install_root, ignore_errors=True)
    torch_prefix = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.utils.cmake_prefix_path)",
        ],
        text=True,
    ).strip()
    configure = [
        "cmake",
        "-S",
        str(aot_root),
        "-B",
        str(build_root),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DENABLE_BELOW_SM90=OFF",
        "-DSGL_KERNEL_ENABLE_FA3=OFF",
        "-DSGL_KERNEL_COMPILE_THREADS=1",
        f"-DSGL_KERNEL_ENABLE_FLASHMLA_SM100={'ON' if enable_sm100 else 'OFF'}",
        "-DREMNANT_FLASHMLA_SOURCE_DIR=/opt/flashmla-remnant",
        f"-DCMAKE_PREFIX_PATH={torch_prefix}",
        "-DCUDA_VERSION=13.0",
    ]
    print(f"[flashmla-build] {' '.join(configure)}", flush=True)
    subprocess.run(
        configure,
        cwd=aot_root,
        check=True,
        timeout=600,
    )
    for target in ("remnant_ops", "flashmla_ops"):
        build = ["cmake", "--build", str(build_root), "--target", target, "--parallel", "2"]
        print(f"[flashmla-build] {' '.join(build)}", flush=True)
        subprocess.run(build, cwd=aot_root, check=True, timeout=1800)
    package_dir = Path(
        subprocess.check_output(
            [sys.executable, "-c", "import pathlib, sgl_kernel; print(pathlib.Path(sgl_kernel.__file__).parent)"],
            text=True,
        ).strip()
    )
    installed = list(build_root.glob("flashmla_ops*.so"))
    if not installed:
        raise RuntimeError(f"FlashMLA build produced no extension under {install_root}")
    for source in installed:
        shutil.copy2(source, package_dir / source.name)
    remnant = list(build_root.glob("remnant_ops*.so"))
    if not remnant:
        raise RuntimeError(f"Remnant adapter build produced no extension under {build_root}")
    for source in remnant:
        shutil.copy2(source, package_dir / source.name)
    shutil.copy2(
        aot_root / "python" / "sgl_kernel" / "flash_mla.py",
        package_dir / "flash_mla.py",
    )
    print(
        f"[flashmla-build] installed {installed[0].name} and {remnant[0].name} into {package_dir}",
        flush=True,
    )


@app.function(
    image=fork_image,
    gpu="H100!",
    cpu=8,
    timeout=3600,
    retries=0,
)
def validate_remnant_non_model(
    benchmark_rows: int = 1024,
    benchmark_repeats: int = 100,
) -> str:
    """Run fork tests and synthetic packed timing without loading weights."""
    if benchmark_rows <= 0 or benchmark_repeats <= 0:
        raise ValueError("benchmark_rows and benchmark_repeats must be positive")

    env = {
        **os.environ,
        "PYTHONPATH": f"{SGLANG_ROOT / 'python'}:{os.environ.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
    }
    test_paths = [
        "test/registered/unit/test_dsv4_c4_cache_format.py",
        "test/registered/attention/unittests/dsv4/test_remnant_pool.py",
        "test/registered/attention/unittests/dsv4/test_remnant_pack.py",
        "test/registered/attention/unittests/dsv4/test_remnant_backend.py",
        "test/registered/attention/unittests/dsv4/test_remnant_cuda_graph.py",
        "python/sglang/test/kernels/deepseek_v4/test_remnant_pack_kernel.py",
        "python/sglang/test/kernels/deepseek_v4/test_remnant_unpack_kernel.py",
    ]
    commands = [
        [sys.executable, "-m", "pytest", "-q", *test_paths],
        [
            sys.executable,
            "benchmark/remnant/bench_packed.py",
            "--rows",
            str(benchmark_rows),
            "--repeats",
            str(benchmark_repeats),
        ],
    ]
    for command in commands:
        print(f"[remnant-non-model] {' '.join(command)}", flush=True)
        subprocess.run(command, env=env, cwd=SGLANG_ROOT, check=True)
    return "remnant non-model tests and benchmark passed"


@app.function(
    image=server_image,
    gpu="H100!",
    memory=262144,
    timeout=2400,
    retries=0,
    volumes={str(RESULTS_ROOT): results_volume},
)
def validate_flashmla_direct_decode(
    batches: str = "8,16",
    repeats: int = 100,
    warmup: int = 10,
    rounds: int = 9,
) -> str:
    """Build and validate direct FlashMLA decode without loading model weights."""
    if repeats <= 0 or rounds <= 0 or warmup < 0:
        raise ValueError("repeats/rounds must be positive and warmup nonnegative")
    _build_sglang_kernel()
    env = {
        **os.environ,
        "PYTHONPATH": f"{SGLANG_ROOT / 'python'}:{os.environ.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
    }
    test = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-x",
        "test/registered/attention/unittests/dsv4/test_remnant_flashmla_direct.py",
    ]
    benchmark = [
        sys.executable,
        "benchmark/remnant/bench_flashmla_decode.py",
        "--batches",
        batches,
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--rounds",
        str(rounds),
    ]
    for command in (test, benchmark):
        print(f"[flashmla-direct] {' '.join(command)}", flush=True)
        subprocess.run(command, env=env, cwd=SGLANG_ROOT, check=True)
    return "direct FlashMLA parity and benchmark passed"


@app.function(
    image=server_image,
    gpu="H100!",
    memory=262144,
    timeout=2400,
    retries=0,
)
def sanitize_flashmla_direct_decode() -> str:
    """Run model-free memory, race, and synchronization checks on direct decode."""
    _build_sglang_kernel(enable_sm100=False)
    if shutil.which("compute-sanitizer") is None:
        raise RuntimeError("compute-sanitizer is not installed in the server image")
    env = {
        **os.environ,
        "PYTHONPATH": f"{SGLANG_ROOT / 'python'}:{os.environ.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
    }
    test = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test/registered/attention/unittests/dsv4/test_remnant_flashmla_direct.py",
        "-k",
        "direct_matches_native_adapter",
    ]
    for tool in ("memcheck", "racecheck", "synccheck"):
        command = [
            "compute-sanitizer",
            "--tool",
            tool,
            "--error-exitcode",
            "99",
            *test,
        ]
        print(f"[flashmla-sanitizer:{tool}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=SGLANG_ROOT, env=env, check=True, timeout=750)
    return "direct FlashMLA sanitizer checks passed"


@app.function(
    image=server_image,
    gpu="H100!",
    memory=262144,
    timeout=2400,
    retries=0,
)
def validate_flashmla_fork() -> str:
    """Build and test the checked-out FlashMLA source independently of SGLang."""
    env = {
        **os.environ,
        "FLASH_MLA_DISABLE_SM100": "1",
        "MAX_JOBS": "2",
        "NVCC_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
    }
    source = Path("/opt/flashmla-remnant")
    if not (source / "csrc" / "python_api.cpp").exists():
        raise RuntimeError("local FlashMLA source is missing from the Modal image")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", "."],
        cwd=source,
        env=env,
        check=True,
        timeout=1500,
    )
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_flash_mla_remnant_decoding.py"],
        cwd=source,
        env=env,
        check=True,
        timeout=600,
    )
    return "standalone FlashMLA tests passed"


@app.function(
    image=server_image,
    gpu="H100!",
    memory=262144,
    timeout=2400,
    retries=0,
    volumes={str(RESULTS_ROOT): results_volume},
)
def profile_flashmla_direct() -> str:
    """Capture direct FlashMLA decode timing and H100 resource counters."""
    _build_sglang_kernel(enable_sm100=False)
    for tool in ("nsys", "ncu", "cuobjdump"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is not installed in the server image")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = RESULTS_ROOT / f"{stamp}-profile-flashmla-direct-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    env = {
        **os.environ,
        "PYTHONPATH": f"{SGLANG_ROOT / 'python'}:{os.environ.get('PYTHONPATH', '')}",
        "PYTHONUNBUFFERED": "1",
    }
    def target(path: str, heads: int | str, batches: str) -> list[str]:
        return [
            sys.executable,
            "benchmark/remnant/bench_flashmla_decode.py",
            "--heads",
            str(heads),
            "--batches",
            batches,
            "--repeats",
            "1",
            "--rounds",
            "1",
            "--warmup",
            "3",
            "--path",
            path,
        ]
    try:
        for path in ("native", "direct"):
            subprocess.run(
                [
                    "nsys",
                    "profile",
                    "--trace=cuda,nvtx",
                    "--sample=none",
                    "--force-overwrite=true",
                    "--output",
                    str(directory / f"{path}-decode-nsys"),
                    *target(path, "64,128", "8,16"),
                ],
                cwd=SGLANG_ROOT,
                env=env,
                check=True,
                timeout=900,
            )
        for path in ("native", "direct"):
            for heads in (64, 128):
                for batch in (8, 16):
                    stem = f"{path}-h{heads}-b{batch}-ncu"
                    report = directory / stem
                    subprocess.run(
                        [
                            "ncu",
                            "--set",
                            "full",
                            "--target-processes",
                            "all",
                            "--kernel-name-base",
                            "function",
                            "--kernel-name",
                            "regex:flash_fwd_splitkv_mla_fp8_sparse_kernel",
                            "--launch-count",
                            "1",
                            "--clock-control",
                            "none",
                            "--export",
                            str(report),
                            "--force-overwrite",
                            *target(path, heads, str(batch)),
                        ],
                        cwd=SGLANG_ROOT,
                        env=env,
                        check=True,
                        timeout=1200,
                    )
                    with (directory / f"{stem}.csv").open("w") as output:
                        subprocess.run(
                            [
                                "ncu",
                                "--import",
                                str(report.with_suffix(".ncu-rep")),
                                "--page",
                                "raw",
                                "--csv",
                            ],
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=180,
                        )
        package_dir = Path(
            subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import pathlib, sgl_kernel; print(pathlib.Path(sgl_kernel.__file__).parent)",
                ],
                env=env,
                text=True,
            ).strip()
        )
        libraries = list(package_dir.glob("flashmla_ops*.so"))
        if libraries:
            with (directory / "sass.txt").open("w") as output:
                subprocess.run(
                    ["cuobjdump", "--dump-sass", "--dump-resource-usage", str(libraries[0])],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    finally:
        results_volume.commit()
    return str(directory)
