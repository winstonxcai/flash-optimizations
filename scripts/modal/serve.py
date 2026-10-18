"""Full-model serving benchmarks and pinned checkpoint management.

Author: Winston Cai.

Commands (each remote run requires budget authorization):
    modal run scripts/modal/serve.py::inspect_model_cache
    modal run scripts/modal/serve.py::download_model
    modal run scripts/modal/serve.py::benchmark_serving --mode native

Default benchmark: TP4/4xH100, Native, 32k input, 2048 output, concurrency 8;
60-minute work limit (not an ETA). Uses the same Bash launcher as local H100s.
The default image builds from Dockerfile. REMNANT_COMPILED_IMAGE reuses a known
image only after CPU source verification, before overlaying Python updates.
Model-free work: kernels.py. Capture: training/capture.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import modal

MODEL_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
MODEL_ROOT = Path("/models")
MODEL_DIR = MODEL_ROOT / "DeepSeek-V4-Flash-0731"
RESULTS_ROOT = Path("/results")
SGLANG_ROOT = Path("/sgl-workspace/sglang-remnant")
REMOTE_REPO = Path("/opt/flash-optimizations")


REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPO


def _source_revision(path: Path, identity: str) -> str:
    """Images receive source identities explicitly; no Git checkout is needed remotely."""
    if not modal.is_local():
        revision = os.environ.get(f"REMNANT_{identity.upper()}_SHA", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"missing submitted source identity: {identity}")
        return revision
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=10).strip()


def _compiled_source_digest(roots: list[Path]) -> str:
    digest = hashlib.sha256()
    for root in map(Path, roots):
        if not root.is_dir():
            raise RuntimeError(f"compiled source directory absent: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {
                ".cu",
                ".cuh",
                ".cc",
                ".cpp",
                ".h",
                ".hpp",
                ".cmake",
                ".txt",
                ".py",
            }:
                digest.update(str(path.relative_to(root)).encode() + b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _verify_compiled_sources(expected: str) -> None:
    """CPU-only gate: never overlay Python onto mismatched compiled kernel sources."""
    actual = _compiled_source_digest([SGLANG_ROOT / "python/sglang/kernels", Path("/opt/flashmla-remnant/csrc")])
    if actual != expected:
        raise RuntimeError("Compiled source changed; rebuild Dockerfile instead of reusing this image")


SOURCE_REVISIONS = {
    "parent": _source_revision(REPO_ROOT, "parent"),
    "sglang": _source_revision(REPO_ROOT / "third_party" / "sglang", "sglang"),
    "flashmla": _source_revision(REPO_ROOT / "third_party" / "flashmla", "flashmla"),
}
SOURCE_ENV = {f"REMNANT_{name.upper()}_SHA": revision for name, revision in SOURCE_REVISIONS.items()}
app = modal.App("remnant")
model_volume = modal.Volume.from_name("deepseek-v4-flash-0731", create_if_missing=True)
# Keep the existing volume name: renaming the public modes must not orphan results.
results_volume = modal.Volume.from_name("remnant-stage2a-results", create_if_missing=True)
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub[hf-xet]==0.34.4",
    )
    .env(SOURCE_ENV)
)


compiled_image_id = os.environ.get("REMNANT_COMPILED_IMAGE")
if compiled_image_id:
    # Explicit CPU-checked reuse for Python-only iteration. Never refresh CUDA
    # source over old binaries: reject mismatches before mounting Python files.
    expected = _compiled_source_digest(
        [
            REPO_ROOT / "third_party/sglang/python/sglang/kernels",
            REPO_ROOT / "third_party/flashmla/csrc",
        ]
    )
    fork_image = (
        modal.Image.from_id(compiled_image_id)
        .pip_install(f"modal=={modal.__version__}")
        .env(SOURCE_ENV)
        # Bake the verifier's helpers into the cache key; execute ordinary Python.
        .add_local_file(REPO_ROOT / "scripts/modal/serve.py", "/opt/remnant-serve.py", copy=True)
        .run_function(_verify_compiled_sources, kwargs={"expected": expected}, timeout=180)
        .add_local_dir(
            REPO_ROOT / "third_party/sglang",
            str(SGLANG_ROOT),
            copy=True,
            ignore=(
                "python/sglang/kernels/**",
                ".git",
                "**/__pycache__/**",
                "**/*.pyc",
            ),
        )
    )
else:
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
    fork_image.env(SOURCE_ENV)
    .apt_install("curl", "jq", "util-linux", "coreutils")
    .add_local_dir(
        REPO_ROOT / "scripts/local",
        REMOTE_REPO / "scripts/local",
        ignore=("**/__pycache__/**", "**/*.pyc"),
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
    image=download_image,
    cpu=1,
    memory=1024,
    timeout=180,
    retries=0,
    volumes={str(MODEL_ROOT): model_volume},
)
def inspect_model_cache() -> dict:
    """CPU-only revision/size gate; no downloads or GPU allocation."""
    config = json.loads((MODEL_DIR / "config.json").read_text())
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    if len(names) != 48:
        raise RuntimeError("expected 48 pinned checkpoint shards")
    for name in names + ["config.json", "model.safetensors.index.json"]:
        path = MODEL_DIR / name
        receipt = MODEL_DIR / ".cache/huggingface/download" / (name + ".metadata")
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or not receipt.is_file()
            or receipt.read_text().splitlines()[:1] != [MODEL_REVISION]
        ):
            raise RuntimeError(f"missing or unpinned checkpoint file: {name}")
    result = {
        "revision": MODEL_REVISION,
        "shards": len(names),
        "bytes": sum((MODEL_DIR / n).stat().st_size for n in names),
        "heads": config["num_attention_heads"],
        "layers": len(config["compress_ratios"]),
    }
    print(json.dumps(result), flush=True)
    return result


@app.function(
    image=server_image,
    gpu="H100!:4",
    cpu=32,
    memory=262144,
    timeout=4 * 3600 + 60,
    retries=0,
    volumes={str(MODEL_ROOT): model_volume, str(RESULTS_ROOT): results_volume},
)
def benchmark_serving(
    mode: str = "native",
    input_tokens: int = 32768,
    output_tokens: int = 2048,
    concurrency: int = 8,
    timeout_minutes: int = 60,
) -> str:
    """One configuration per call, identical to the local shell command."""
    if mode not in {"native", "triton", "kernel", "packed"}:
        raise ValueError("mode must be native, triton, or kernel (packed aliases kernel)")
    if mode == "packed":
        mode = "kernel"
    if not 1 <= timeout_minutes <= 240:
        raise ValueError("timeout_minutes must be 1–240")
    if not _kernel_extensions_ready():
        raise RuntimeError(
            "Required Remnant extensions are missing. Build the serving image "
            "on CPU before launching; GPU compilation is disabled."
        )
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "MODEL_PATH": str(MODEL_DIR),
        "SGLANG_ROOT": str(SGLANG_ROOT),
        "RESULTS_DIR": str(RESULTS_ROOT),
        **SOURCE_ENV,
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


# Installed extension checks (never compile inside a GPU function).


def _kernel_package_dir() -> Path:
    spec = importlib.util.find_spec("sgl_kernel")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("CPU-built sgl_kernel package is missing from the image")
    return Path(next(iter(spec.submodule_search_locations)))


def _kernel_extensions_ready() -> bool:
    try:
        package_dir = _kernel_package_dir()
        flash_mla = package_dir / "flash_mla.py"
        flash_mla_source = flash_mla.read_text()
        return (
            any(package_dir.glob("flashmla_ops*.so"))
            and any(package_dir.glob("remnant_ops*.so"))
            and "remnant_buffers" in flash_mla_source
            and "remnant_sparse_decode_fwd" in flash_mla_source
        )
    except (OSError, ImportError):
        return False
