"""Modal resources only; serving runs the same Bash script as local H100s."""
from __future__ import annotations

import os
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
SGLANG_ROOT = Path("/sgl-workspace/sglang-lowrank")
REMOTE_REPO = Path("/opt/mustafar/flash-optimizations")
REPO_ROOT = Path(__file__).resolve().parents[3]

app = modal.App("mustafar")
model_volume = modal.Volume.from_name("deepseek-v4-flash-0731", create_if_missing=True)
# Keep the existing volume name: renaming the public modes must not orphan results.
results_volume = modal.Volume.from_name(
    "mustafar-stage2a-results", create_if_missing=True
)
download_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface-hub[hf-xet]==0.34.4",
)
server_image = (
    modal.Image.from_dockerfile(
        str(REPO_ROOT / "mustafar" / "Dockerfile"),
        context_dir=str(REPO_ROOT),
        # Keep stage2a_cpu.py in the build context: the Dockerfile runs it.
        ignore=(
            "mustafar/scripts/**",
            "mustafar/tests/bench_*.py",
            "mustafar/tests/test_bench_serving.py",
            "mustafar/tests/fixtures/**",
        ),
    )
    .apt_install("curl", "jq", "util-linux", "coreutils")
    .add_local_dir(
        REPO_ROOT / "mustafar" / "scripts",
        REMOTE_REPO / "mustafar" / "scripts",
    )
    .add_local_dir(
        REPO_ROOT / "mustafar" / "tests",
        REMOTE_REPO / "mustafar" / "tests",
    )
)



@app.function(image=download_image, cpu=8, memory=32768, timeout=6 * 3600,
              volumes={str(MODEL_ROOT): model_volume})
def download_model() -> str:
    """CPU-only pinned checkpoint download; existing shards are reused."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION,
                             local_dir=str(MODEL_DIR), max_workers=4)
    model_volume.commit()
    return path


@app.function(image=server_image, gpu="H100!:4", cpu=32, memory=262144,
              timeout=4 * 3600 + 60, retries=0,
              volumes={str(MODEL_ROOT): model_volume, str(RESULTS_ROOT): results_volume})
def bench_serving(mode: str = "native", input_tokens: int = 32768,
                  output_tokens: int = 2048, concurrency: int = 8,
                  timeout_minutes: int = 60) -> str:
    """One configuration per call, identical to the local shell command."""
    if not 1 <= timeout_minutes <= 240:
        raise ValueError("timeout_minutes must be 1–240")
    env = {**os.environ, "PYTHON": sys.executable, "MODEL_PATH": str(MODEL_DIR),
           "SGLANG_ROOT": str(SGLANG_ROOT), "RESULTS_DIR": str(RESULTS_ROOT)}
    try:
        subprocess.run(
            ["timeout", "--signal=TERM", "--kill-after=30s", f"{timeout_minutes}m",
             "bash", str(REMOTE_REPO / "mustafar/scripts/local/bench_serving.sh"),
             mode, str(input_tokens), str(output_tokens), str(concurrency)],
            env=env, cwd=REMOTE_REPO, check=True,
        )
    finally:
        results_volume.commit()
    return str(RESULTS_ROOT)


def _kernel_run(modules: list[str], *, kind: str, timeout: int,
                sanitizer: bool = False) -> str:
    """Keep model-free kernel checks separate from serving."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = RESULTS_ROOT / f"{stamp}-{kind}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SGLANG_OPT_TOPMAG", "XKV_TOPMAG"))}
    env.update(SGLANG_OPT_TOPMAG="1", XKV_TOPMAG_KEEP="0.5",
               SGLANG_OPT_TOPMAG_PACKED_C4="1", SGLANG_OPT_TOPMAG_STAGE2A="0",
               MUSTAFAR_RESULTS_DIR=str(directory), MUSTAFAR_STAGE2A_RESULTS_DIR=str(directory))
    commands = [[sys.executable, "-m", module] for module in modules]
    if sanitizer:
        commands.append(["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99",
                         sys.executable, "-m", modules[-1], "--sanitizer-case"])
    try:
        for i, command in enumerate(commands):
            with (directory / f"{i}.log").open("w") as log:
                subprocess.run(command, env=env, cwd=REMOTE_REPO, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=timeout)
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
    """H100: existing packed-format correctness suite."""
    return _kernel_run(
        ["mustafar.tests.gpu_packed"], kind="validate-packed", timeout=3500
    )


@app.function(
    image=server_image,
    gpu="L4",
    timeout=1800,
    retries=0,
    volumes={str(RESULTS_ROOT): results_volume},
)
def validate_packed_fused() -> str:
    """L4: fused adapter correctness, graph/stream checks, and memcheck."""
    return _kernel_run(
        ["mustafar.tests.gpu_stage2a"],
        kind="validate-packed-fused",
        timeout=1700,
        sanitizer=True,
    )


@app.function(
    image=server_image,
    gpu="H100!",
    timeout=1800,
    retries=0,
    volumes={str(RESULTS_ROOT): results_volume},
)
def bench_kernels(suite: str = "packed_fused") -> str:
    """H100: packed component timings or the packed/fused comparison gate."""
    modules = {
        "packed": "mustafar.tests.bench_packed",
        "packed_fused": "mustafar.tests.bench_stage2a",
    }
    if suite not in modules:
        raise ValueError(f"suite must be one of {tuple(modules)}")
    return _kernel_run([modules[suite]], kind=f"bench-{suite}", timeout=1700)
