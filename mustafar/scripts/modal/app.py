"""Modal entry points for Mustafar Stage-1 validation.

The model is intentionally stored in a Modal Volume rather than baked into
the server image.  Run the one-time population step with::

    modal run mustafar/scripts/modal/app.py::download_model
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import time

import modal


APP_NAME = "mustafar-stage1"
MODEL_REPO = "sgl-project/DeepSeek-V4-Flash-FP8"
MODEL_REVISION = "ae01d80c06cdfe30581edfd0e1c5449dc7ed7f17"
MODEL_VOLUME_NAME = "deepseek-v4-flash-fp8"
MODEL_VOLUME_ROOT = Path("/models")
MODEL_DIR = MODEL_VOLUME_ROOT / "DeepSeek-V4-Flash-FP8"
RESULTS_VOLUME_NAME = "mustafar-stage1-results"
RESULTS_ROOT = Path("/results")
_LOCAL_FILE = Path(__file__).resolve()
REPO_ROOT = (
    _LOCAL_FILE.parents[3]
    if len(_LOCAL_FILE.parents) > 3
    else Path("/opt/mustafar/flash-optimizations")
)


app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

download_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface-hub[hf-xet]==0.34.4",
)

server_image = modal.Image.from_dockerfile(
    str(REPO_ROOT / "mustafar" / "Dockerfile"),
    context_dir=str(REPO_ROOT),
)


@app.function(
    image=download_image,
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={str(MODEL_VOLUME_ROOT): model_volume},
)
def download_model() -> dict[str, object]:
    """Populate the persistent model Volume at the pinned HF revision."""
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        max_workers=16,
    )
    model_volume.commit()

    files = [path for path in MODEL_DIR.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    result = {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "path": resolved,
        "files": len(files),
        "bytes": total_bytes,
    }
    print(result, flush=True)
    return result


@app.function(
    image=server_image,
    gpu="H100!",
    timeout=60 * 60,
    volumes={str(RESULTS_ROOT): results_volume},
)
def kernel_validation() -> dict[str, object]:
    """Compile and validate Stage-1 Triton kernels on one H100 SXM."""
    env = os.environ.copy()
    env.update(
        SGLANG_OPT_TOPMAG="1",
        XKV_TOPMAG_KEEP="0.5",
        SGLANG_OPT_TOPMAG_PACKED_C4="1",
        PYTHONPATH="/opt/mustafar/flash-optimizations",
    )
    proc = subprocess.run(
        ["python3", "-m", "mustafar.tests.gpu_packed"],
        cwd="/opt/mustafar/flash-optimizations",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    if proc.returncode:
        raise RuntimeError(f"kernel validation failed with {proc.returncode}")
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    result_path = RESULTS_ROOT / "kernel-validation.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    results_volume.commit()
    return result


@app.function(
    image=server_image,
    gpu="H100!",
    timeout=2 * 60 * 60,
    volumes={str(RESULTS_ROOT): results_volume},
)
def microbenchmark() -> dict[str, object]:
    """Run the required batch/top-k matrix on one H100 SXM."""
    env = os.environ.copy()
    env.update(
        SGLANG_OPT_TOPMAG="1",
        XKV_TOPMAG_KEEP="0.5",
        SGLANG_OPT_TOPMAG_PACKED_C4="1",
        PYTHONPATH="/opt/mustafar/flash-optimizations",
        MUSTAFAR_RESULTS_DIR=str(RESULTS_ROOT),
    )
    proc = subprocess.run(
        ["python3", "-m", "mustafar.tests.bench_packed"],
        cwd="/opt/mustafar/flash-optimizations",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    if proc.returncode:
        raise RuntimeError(f"microbenchmark failed with {proc.returncode}")
    results_volume.commit()
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _wait_for_server(port: int, process: subprocess.Popen, timeout: int = 1800) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server did not become healthy on port {port}")


@app.function(
    image=server_image,
    gpu="H100!:4",
    timeout=24 * 60 * 60,
    volumes={
        str(MODEL_VOLUME_ROOT): model_volume,
        str(RESULTS_ROOT): results_volume,
    },
)
def paired_tp4_smoke() -> dict[str, object]:
    """Run dense-zero and packed servers sequentially on the same TP4 host.

    This entry point is deliberately a serving smoke. The full load/accuracy
    matrix consumes the same artifact schema and is run only after kernel gates.
    """
    import urllib.request

    base_env = os.environ.copy()
    base_env.update(
        MODEL_PATH=str(MODEL_DIR),
        TP="4",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        SGLANG_OPT_TOPMAG="1",
        XKV_TOPMAG_KEEP="0.5",
        PYTHONPATH="/sgl-workspace/sglang-lowrank/python:/opt/mustafar/flash-optimizations",
    )
    command = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", str(MODEL_DIR),
        "--served-model-name", "deepseek-v4-flash",
        "--tp", "4", "--trust-remote-code",
        "--mem-fraction-static", "0.93",
        "--context-length", "135168",
        "--max-running-requests", "64",
        "--chunked-prefill-size", "4096",
        "--fp8-gemm-backend", "triton",
        "--host", "0.0.0.0", "--port", "30211",
        "--disable-cuda-graph",
        "--reasoning-parser", "deepseek-v4",
        "--tool-call-parser", "deepseekv4",
    ]
    artifacts: dict[str, object] = {"model_revision": MODEL_REVISION, "legs": {}}
    for name, packed in (("dense_zero", False), ("packed", True)):
        env = dict(base_env)
        env["SGLANG_OPT_TOPMAG_PACKED_C4"] = "1" if packed else "0"
        log_path = RESULTS_ROOT / f"{name}-server.log"
        started = time.time()
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                cwd="/sgl-workspace/sglang-lowrank/python",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                _wait_for_server(30211, process)
                payload = json.dumps(
                    {
                        "model": "deepseek-v4-flash",
                        "messages": [{"role": "user", "content": "Reply OK."}],
                        "temperature": 0,
                        "max_tokens": 8,
                    }
                ).encode()
                req = urllib.request.Request(
                    "http://127.0.0.1:30211/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=300) as response:
                    body = json.loads(response.read())
                artifacts["legs"][name] = {
                    "packed": packed,
                    "startup_seconds": time.time() - started,
                    "response": body,
                }
            finally:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
        results_volume.commit()
    output = RESULTS_ROOT / "paired-tp4-smoke.json"
    output.write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n")
    results_volume.commit()
    return artifacts


@app.local_entrypoint()
def main() -> None:
    """Default to the safe, idempotent model population operation."""
    print(download_model.remote())
