"""Full-model serving benchmarks and pinned checkpoint management.

Author: Winston Cai.

Commands (each remote run requires budget authorization):
    modal run scripts/modal/serve.py::inspect_model_cache
    modal run scripts/modal/serve.py::download_model
    modal run scripts/modal/serve.py::benchmark_serving --mode native
    modal run scripts/modal/serve.py::prepare_glm_serving
    modal run scripts/modal/serve.py::smoke_glm_serving --run-id <unique-id>

Default benchmark: TP4/4xH100, Native, 32k input, 2048 output, concurrency 8;
60-minute work limit (not an ETA). Uses the same Bash launcher as local H100s.
Images use Dockerfile.v0.5.18 or Dockerfile.v0.5.22, selected by
REMNANT_INTEGRATION. REMNANT_COMPILED_IMAGE reuses a known
image only after CPU source verification, before overlaying Python updates.
Model-free work: kernels.py. Capture: scripts/local/capture-latent.sh.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import modal
from scripts.sources import check_sources, integration_profile, verify_compiled_image_sources
from scripts.model_profiles import model_profile

INTEGRATION = integration_profile()

MODEL_FAMILY = os.environ.get("REMNANT_MODEL_FAMILY", "dsv4")
if MODEL_FAMILY not in ("dsv4", "glm53") or (MODEL_FAMILY == "glm53" and INTEGRATION["version"] != "v0.5.22"):
    raise ValueError("GLM launchers require REMNANT_INTEGRATION=v0.5.22")
MODEL_PROFILE = model_profile(MODEL_FAMILY, INTEGRATION["version"])
MODEL_REPO = MODEL_PROFILE.repository
MODEL_REVISION = MODEL_PROFILE.revision
MODEL_ROOT = Path("/models")
MODEL_DIR = MODEL_ROOT / MODEL_PROFILE.directory
RESULTS_ROOT = Path("/results")
SGLANG_ROOT = Path("/sgl-workspace/sglang-remnant")
REMOTE_REPO = Path("/opt/flash-optimizations")


REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPO
if modal.is_local():
    from scripts.sources import check_sources

    source_override = os.environ.get("REMNANT_SGLANG_SOURCE_OVERRIDE")
    SOURCE_IDENTITIES = check_sources(
        REPO_ROOT, {"sglang": Path(source_override)} if source_override else None
    )
else:
    SOURCE_IDENTITIES = json.loads(os.environ["REMNANT_SOURCE_IDENTITIES"])


def _source_revision(identity: str) -> str:
    """Images receive source identities explicitly; no Git checkout is needed remotely."""
    if not modal.is_local():
        revision = os.environ.get(f"REMNANT_{identity.upper()}_SHA", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"missing submitted source identity: {identity}")
        return revision
    return SOURCE_IDENTITIES[identity]["commit"]


from scripts.sources import compiled_source_digest as _compiled_source_digest


def _verify_compiled_sources(expected: str) -> None:
    """CPU-only gate: never overlay Python onto mismatched compiled kernel sources."""
    verify_compiled_image_sources(
        expected,
        [str(SGLANG_ROOT / "python/sglang/kernels"), "/opt/flashmla-remnant/csrc"],
        INTEGRATION["version"],
    )


SOURCE_REVISIONS = {
    identity: _source_revision(identity)
    for identity in ("parent", "sglang", "flashmla")
}
SOURCE_ENV = {
    f"REMNANT_{name.upper()}_SHA": revision
    for name, revision in SOURCE_REVISIONS.items()
}
SOURCE_ENV["REMNANT_SOURCE_IDENTITIES"] = json.dumps(SOURCE_IDENTITIES)
SOURCE_ENV["REMNANT_INTEGRATION"] = INTEGRATION["version"]
SOURCE_ENV["REMNANT_MODEL_FAMILY"] = MODEL_FAMILY
SOURCE_ENV["REMNANT_COMPILED_IMAGE"] = os.environ.get("REMNANT_COMPILED_IMAGE", "")
app = modal.App("remnant")
checkpoint_app = modal.App("remnant-checkpoint")
SGLANG_SOURCE = Path(SOURCE_IDENTITIES["sglang"]["path"])
model_volume = modal.Volume.from_name(MODEL_PROFILE.volume, create_if_missing=True)
# Keep the existing volume name: renaming the public modes must not orphan results.
results_volume = modal.Volume.from_name(
    "remnant-stage2a-results", create_if_missing=True
)
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub[hf-xet]==0.34.4",
    )
    .env({**SOURCE_ENV, "PYTHONPATH": str(REMOTE_REPO)})
    .add_local_file(
        REPO_ROOT / "scripts/model_profiles.py",
        str(REMOTE_REPO / "scripts/model_profiles.py"), copy=True,
    )
    .add_local_file(
        REPO_ROOT / "scripts/sources.py",
        str(REMOTE_REPO / "scripts/sources.py"),
        copy=True,
    )
)


compiled_image_id = os.environ.get("REMNANT_COMPILED_IMAGE")
if modal.is_local() and SOURCE_IDENTITIES["sglang"]["override"] and not compiled_image_id:
    raise ValueError("Serving from a source override requires a verified compiled image")
if modal.is_local():
    if compiled_image_id:
        # Explicit CPU-checked reuse for Python-only iteration. Never refresh CUDA
        # source over old binaries: reject mismatches before mounting Python files.
        expected = _compiled_source_digest(
            [
                SGLANG_SOURCE / "python/sglang/kernels",
                REPO_ROOT / INTEGRATION["flashmla"] / "csrc",
            ]
        )
        fork_image = (
            modal.Image.from_id(compiled_image_id)
            .pip_install(f"modal=={modal.__version__}")
            .env(SOURCE_ENV)
            # Bake the verifier's helpers into the cache key; execute ordinary Python.
            .add_local_file(
                REPO_ROOT / "scripts/modal/serve.py", "/opt/remnant-serve.py", copy=True
            )
            .add_local_file(
                REPO_ROOT / "scripts/sources.py",
                str(REMOTE_REPO / "scripts/sources.py"),
                copy=True,
            )
            .add_local_file(
                REPO_ROOT / "scripts/model_profiles.py",
                str(REMOTE_REPO / "scripts/model_profiles.py"), copy=True,
            )
            .run_function(
                _verify_compiled_sources, kwargs={"expected": expected}, timeout=180
            )
            .add_local_dir(
                SGLANG_SOURCE,
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
            str(REPO_ROOT / INTEGRATION["dockerfile"]),
            context_dir=str(REPO_ROOT),
            build_args={
                "REMNANT_BASE_IMAGE": INTEGRATION["base_image"],
                "REMNANT_INTEGRATION": INTEGRATION["version"],
                "REMNANT_SGLANG_SOURCE": INTEGRATION["sglang"],
                "REMNANT_FLASHMLA_SOURCE": INTEGRATION["flashmla"],
                "REMNANT_SGLANG_SHA": SOURCE_REVISIONS["sglang"],
                "REMNANT_FLASHMLA_SHA": SOURCE_REVISIONS["flashmla"],
            },
            ignore=modal.FilePatternMatcher.from_file(REPO_ROOT / ".dockerignore"),
        )
    server_image = (
        fork_image.env(SOURCE_ENV)
        .add_local_file(
            REPO_ROOT / "scripts/model_profiles.py",
            str(REMOTE_REPO / "scripts/model_profiles.py"), copy=True,
        )
        .add_local_file(
            REPO_ROOT / "scripts/sources.py",
            str(REMOTE_REPO / "scripts/sources.py"),
            copy=True,
        )
        .add_local_file(
            REPO_ROOT / "scripts/modal/qualification.py",
            str(REMOTE_REPO / "scripts/modal/qualification.py"), copy=True,
        )
        .apt_install("curl", "jq", "util-linux", "coreutils")
        .add_local_dir(
            REPO_ROOT / "scripts/local",
            REMOTE_REPO / "scripts/local",
            ignore=("**/__pycache__/**", "**/*.pyc"),
        )
    )
else:
    server_image = download_image


@checkpoint_app.function(
    image=download_image,
    cpu=8,
    memory=32768,
    timeout=3600 if MODEL_FAMILY == "glm53" else 6 * 3600,
    retries=0,
    volumes={str(MODEL_ROOT): model_volume},
)
def download_model() -> str:
    """CPU-only pinned checkpoint download; existing shards are reused."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        max_workers=8 if MODEL_FAMILY == "glm53" else 4,
    )
    model_volume.commit()
    _check_model_cache()
    return path


@checkpoint_app.function(
    image=download_image,
    cpu=1,
    memory=1024,
    timeout=180,
    retries=0,
    volumes={str(MODEL_ROOT): model_volume},
)
def inspect_model_cache() -> dict:
    """CPU-only revision/size gate; no downloads or GPU allocation."""
    return _check_model_cache()


def _check_model_cache() -> dict:
    config = json.loads((MODEL_DIR / "config.json").read_text())
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()))
    expected_shards = MODEL_PROFILE.shards
    if len(names) != expected_shards:
        raise RuntimeError(f"Expected {expected_shards} pinned checkpoint shards, got {len(names)}")
    metadata_files = list(MODEL_PROFILE.metadata_files)
    for name in names + metadata_files:
        path = MODEL_DIR / name
        receipt = MODEL_DIR / ".cache/huggingface/download" / (name + ".metadata")
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or not receipt.is_file()
            or receipt.read_text().splitlines()[:1] != [MODEL_REVISION]
        ):
            raise RuntimeError(f"missing or unpinned checkpoint file: {name}")
    size = sum((MODEL_DIR / name).stat().st_size for name in names)
    if MODEL_PROFILE.checkpoint_bytes is not None and size != MODEL_PROFILE.checkpoint_bytes:
        raise RuntimeError(f"Expected {MODEL_PROFILE.checkpoint_bytes} checkpoint bytes, got {size}")
    result = {
        "revision": MODEL_REVISION,
        "shards": len(names),
        "bytes": size,
        "heads": config.get("text_config", config)["num_attention_heads"],
        "layers": config["text_config"]["num_hidden_layers"] if MODEL_FAMILY == "glm53" else len(config["compress_ratios"]),
    }
    print(json.dumps(result), flush=True)
    return result


def _stop_serving_server(process: subprocess.Popen) -> None:
    """Stop only our session, including workers left after their leader exits."""
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait(timeout=1)
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            rows = subprocess.check_output(
                ["ps", "-eo", "pgid=,stat="],
                text=True,
                timeout=3,
            )
            # Container PID 1 may leave zombies; they cannot retain GPUs.
            live = any(
                int(group) == process.pid and state[0] not in "ZX"
                for group, state in (row.split() for row in rows.splitlines())
            )
            if not live:
                process.wait(timeout=1)
                return
            time.sleep(0.2)
    raise RuntimeError(f"serving process group {process.pid} survived SIGKILL")


@contextmanager
def _running_serving_server(
    env: dict[str, str], log: Path, deadline: float
) -> Iterator[None]:
    """The Modal caller owns server lifetime; the benchmark is only a client."""
    # Refuse an existing listener rather than accepting its health response.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", int(env["PORT"])))
        except OSError as exc:
            raise RuntimeError(f"serving port {env['PORT']} is already in use") from exc
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                "bash",
                str(REMOTE_REPO / "scripts/local/run-server.sh"),
                env["SERVING_MODE"],
                str(MODEL_DIR),
                "127.0.0.1",
                env["PORT"],
            ],
            env=env,
            cwd=REMOTE_REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"serving process exited during startup; see {log}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"server startup exhausted job budget; see {log}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{env['PORT']}/health",
                        timeout=min(3, remaining),
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass  # Readiness failures are retried only within the job budget.
                time.sleep(min(1, remaining))
            yield
        finally:
            _stop_serving_server(process)


@app.function(
    image=server_image,
    gpu="H100!:8" if MODEL_FAMILY == "glm53" else "H100!:4",
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
    """Own one server boot and call the shared client against it."""
    if mode not in {"native", "direct"}:
        raise ValueError("mode must be native or direct")
    if not 1 <= timeout_minutes <= 240:
        raise ValueError("timeout_minutes must be 1–240")
    if min(input_tokens, output_tokens, concurrency) < 1:
        raise ValueError("input_tokens, output_tokens and concurrency must be positive")
    if not _kernel_extensions_ready():
        raise RuntimeError(
            "Required Remnant extensions are missing. Build the serving image "
            "on CPU before launching; GPU compilation is disabled."
        )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{mode}-", dir=RESULTS_ROOT))
    deadline = time.monotonic() + timeout_minutes * 60
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "MODEL_PATH": str(MODEL_DIR),
        "MODEL_FAMILY": MODEL_FAMILY,
        **({"TP": "8", "MODEL_NAME": "glm53-flash"} if MODEL_FAMILY == "glm53" else {}),
        "SGLANG_ROOT": str(SGLANG_ROOT),
        "RESULTS_HOST": str(run),
        "BENCH_EXECUTION": "direct",
        "SERVING_MODE": mode,
        "PORT": os.environ.get("PORT", "30211"),
        "DECODE_CFG": os.environ.get("DECODE_CFG")
        or (
            REMOTE_REPO / "scripts/local/config/decode-graphs-extended.json"
        ).read_text(),
        **SOURCE_ENV,
    }
    (run / "source-revisions.json").write_text(
        json.dumps(SOURCE_REVISIONS, indent=2) + "\n"
    )
    (run / "source-identities.json").write_text(
        json.dumps(SOURCE_IDENTITIES, indent=2) + "\n"
    )
    try:
        with _running_serving_server(env, run / "server.log", deadline):
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("server startup exhausted the benchmark budget")
            env["BENCH_TIMEOUT_S"] = str(remaining)
            subprocess.run(
                [
                    "bash",
                    str(REMOTE_REPO / "scripts/local/bench-serving.sh"),
                    mode,
                    env["PORT"],
                    str(input_tokens),
                    str(output_tokens),
                    str(concurrency),
                ],
                env=env,
                cwd=REMOTE_REPO,
                check=True,
                timeout=remaining,
            )
    finally:
        results_volume.commit()
    return str(run)


GLM_SMOKE_MODE_SECONDS = 540
GLM_SMOKE_TIMEOUT_SECONDS = 1140
GLM_SMOKE_PYTHON = "/opt/sglang/bin/python"


def _check_glm_serving_image() -> dict:
    if MODEL_FAMILY != "glm53" or not compiled_image_id:
        raise ValueError("GLM serving smoke requires glm53 and a verified REMNANT_COMPILED_IMAGE")
    build = Path("/opt/remnant-build.json")
    if not build.is_file() or not Path(GLM_SMOKE_PYTHON).is_file():
        raise RuntimeError("The verified GLM runtime or its build manifest is missing")
    config = json.loads(build.read_text())["model_config"]
    if config["revision"] != MODEL_REVISION:
        raise RuntimeError("The compiled image and serving checkpoint revisions differ")
    return config


def _check_glm_serving_checkpoint(compiled_config: dict) -> dict:
    checkpoint = _check_model_cache()
    digest = hashlib.sha256((MODEL_DIR / "config.json").read_bytes()).hexdigest()
    if digest != compiled_config["sha256"]:
        raise RuntimeError("Mounted GLM config does not match the verified image")
    return {**checkpoint, "config_sha256": digest}


@app.function(
    image=server_image,
    cpu=2,
    memory=4096,
    timeout=180,
    retries=0,
    volumes={str(MODEL_ROOT): model_volume.read_only()},
)
def prepare_glm_serving(check_checkpoint: bool = True) -> dict:
    """Check the serving imports and optionally verify the pinned weights on CPU."""
    compiled_config = _check_glm_serving_image()
    checkpoint = _check_glm_serving_checkpoint(compiled_config) if check_checkpoint else None
    subprocess.run(
        [GLM_SMOKE_PYTHON, str(Path(GLM_SMOKE_PYTHON).with_name("sglang")),
         "serve", "--model-type", "llm", "--help"],
        check=True,
        timeout=90,
        stdout=subprocess.DEVNULL,
    )
    return {"checkpoint": checkpoint, "compiled_image": compiled_image_id}


def _glm_chat_request(port: str, deadline: float) -> dict:
    payload = {
        "model": "glm53-flash",
        "messages": [{"role": "user", "content": "Write the numbers one to five in order."}],
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "temperature": 0,
        "max_tokens": 256,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("No time remaining for GLM chat completion")
    with urllib.request.urlopen(request, timeout=min(120, remaining)) as response:
        status = response.status
        body = json.load(response)
    choices = body.get("choices") or []
    choice = choices[0] if choices else {}
    content = choice.get("message", {}).get("content")
    if (
        status != 200
        or body.get("model") != payload["model"]
        or not isinstance(content, str)
        or content.strip() != "1, 2, 3, 4, 5"
        or choice.get("finish_reason") != "stop"
        or body.get("usage", {}).get("completion_tokens", 0) < 2
    ):
        raise RuntimeError(f"Invalid GLM chat completion: {body!r}")
    return {"http_status": status, "request": payload, "response": body}


@app.function(
    image=server_image,
    gpu="H200:4",
    cpu=24,
    memory=196608,
    timeout=GLM_SMOKE_TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={str(MODEL_ROOT): model_volume.read_only(), str(RESULTS_ROOT): results_volume},
)
def smoke_glm_serving(run_id: str, modes: str = "native,direct") -> dict:
    """Boot the same pinned checkpoint in each mode and make a chat request."""
    selected = modes.split(",")
    if modes not in ("native,direct", "native", "direct"):
        raise ValueError("modes must be native,direct, native, or direct")
    from scripts.modal.qualification import validate_run_id

    validate_run_id(run_id, max_length=101)
    checkpoint = _check_glm_serving_checkpoint(_check_glm_serving_image())
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).splitlines()
    if len(devices) != 4 or any("H200" not in device for device in devices):
        raise RuntimeError(f"Expected four H200 devices, got {devices!r}")
    root = RESULTS_ROOT / "glm53-serving" / run_id
    root.mkdir(parents=True, exist_ok=False)
    record = {
        "checkpoint": checkpoint,
        "sources": SOURCE_IDENTITIES,
        "launcher": str(REMOTE_REPO / "scripts/local/run-server.sh"),
        "model_path": str(MODEL_DIR),
        "compiled_image": compiled_image_id,
        "gpus": devices,
        "mode_results": {},
    }
    settings = {
        "MODEL_FAMILY": "glm53",
        "MODEL_NAME": "glm53-flash",
        "TP": "4",
        "EP": "4",
        "PORT": "30211",
        "CTX_LEN": "8192",
        "MAX_RUN": "4",
        "CHUNK": "2048",
        "MAX_TOTAL_TOKENS": "16384",
        # Build only the DeepGEMM shapes used by this small smoke workload.
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "MEM_FRAC": "0.86",
        # Readiness probes must not queue work during first-use JIT.
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
        # The verified kernel image does not bundle the optional Rust tree.
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "GLM_REASONING_PARSER": "auto",
        "DECODE_CFG": json.dumps(
            {"decode": {"backend": "full", "max_bs": 4, "bs": [1, 2, 4]},
             "prefill": {"backend": "disabled"}}
        ),
    }
    record["settings"] = settings
    allocation_started = time.monotonic()
    worker_deadline = allocation_started + GLM_SMOKE_TIMEOUT_SECONDS - 30
    record["started_at"] = time.time()
    for mode in selected:
        if worker_deadline - time.monotonic() < GLM_SMOKE_MODE_SECONDS + 30:
            record["mode_results"][mode] = {"status": "pending", "error": "Insufficient time for full mode allowance and cleanup"}
            (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            results_volume.commit()
            break
        start = time.monotonic()
        deadline = start + GLM_SMOKE_MODE_SECONDS
        log = root / f"{mode}.log"
        result = record["mode_results"][mode] = {"status": "failed", "started_at": time.time()}
        env = {
            **os.environ,
            **settings,
            "PATH": f"{Path(GLM_SMOKE_PYTHON).parent}:{os.environ['PATH']}",
            "SERVING_MODE": mode,
        }
        try:
            print(f"Starting GLM {mode} on four H200s", flush=True)
            with _running_serving_server(env, log, deadline):
                result["health_http_status"] = 200
                result["ready_seconds"] = time.monotonic() - start
                result["chat_started_at"] = time.time()
                result["chat"] = _glm_chat_request(env["PORT"], deadline)
                result["chat_completed_at"] = time.time()
            result["status"] = "passed"
        except (OSError, RuntimeError, ValueError) as exc:
            result["error"] = str(exc)
            if log.exists():
                print("\n".join(log.read_text(errors="replace").splitlines()[-40:]), flush=True)
        finally:
            result["terminated_at"] = time.time()
            result["worker_seconds"] = time.monotonic() - start
            (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
            results_volume.commit()
    print(json.dumps(record), flush=True)
    if any(item["status"] != "passed" for item in record["mode_results"].values()):
        raise RuntimeError(f"GLM serving smoke failed. See {root}/result.json")
    return record


# Installed extension checks (never compile inside a GPU function).


def _kernel_package_dir() -> Path:
    spec = importlib.util.find_spec("sgl_kernel")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(
            "CPU-built sgl_kernel package is missing from the image"
        )
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
