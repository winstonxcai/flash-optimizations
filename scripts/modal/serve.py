"""Full-model serving benchmarks and pinned checkpoint management.

Author: Winston Cai.

Commands (each remote run requires budget authorization):
    modal run scripts/modal/serve.py::inspect_model_cache
    modal run scripts/modal/serve.py::download_model
    modal run scripts/modal/serve.py::benchmark_serving --mode native
    modal run scripts/modal/serve.py::prepare_lswb
    modal run scripts/modal/serve.py::benchmark_lswb --mode v41-fp4

V4.1-Flash-1017 LSWB: REMNANT_MODEL=v41-1017 REMNANT_INTEGRATION=v0.5.22 selects
the checkpoint volume; upload transfer/longswebench.jsonl.gz to the lswb-replay
volume first (modal volume put lswb-replay transfer/longswebench.jsonl.gz
/longswebench.jsonl.gz).

Default benchmark: TP4/4xH100, Native, 32k input, 2048 output, concurrency 8;
60-minute work limit (not an ETA). Uses the same Bash launcher as local H100s.
Images use Dockerfile.v0.5.18 or Dockerfile.v0.5.22, selected by
REMNANT_INTEGRATION. REMNANT_COMPILED_IMAGE reuses a known
image only after CPU source verification, before overlaying Python updates.
Model-free work: kernels.py. Capture: scripts/local/capture-latent.sh.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
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
from scripts.sources import check_sources, integration_profile

INTEGRATION = integration_profile()

# REMNANT_MODEL selects the pinned checkpoint and its per-workspace volume.
MODELS = {
    "0731": (
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        "deepseek-v4-flash-0731",
    ),
    "v41-1017": (
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "deepseek-v4-1-flash-1017",
    ),
}
MODEL_KEY = os.environ.get("REMNANT_MODEL", "0731")
if MODEL_KEY not in MODELS:
    raise ValueError(f"REMNANT_MODEL must be one of {sorted(MODELS)}")
MODEL_REPO, MODEL_REVISION, MODEL_VOLUME = MODELS[MODEL_KEY]
MODEL_ROOT = Path("/models")
MODEL_DIR = MODEL_ROOT / MODEL_REPO.split("/")[1]
RESULTS_ROOT = Path("/results")
LSWB_ROOT = Path("/lswb")
SGLANG_ROOT = Path("/sgl-workspace/sglang-remnant")
REMOTE_REPO = Path("/opt/flash-optimizations")


REPO_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPO
if modal.is_local():
    from scripts.sources import check_sources

    SOURCE_IDENTITIES = check_sources(REPO_ROOT)
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
    if os.environ.get("REMNANT_BUILT_INTEGRATION", "v0.5.18") != INTEGRATION["version"]:
        raise RuntimeError("Compiled image belongs to a different REMNANT_INTEGRATION")
    actual = _compiled_source_digest(
        [SGLANG_ROOT / "python/sglang/kernels", Path("/opt/flashmla-remnant/csrc")]
    )
    if actual != expected:
        raise RuntimeError(
            "Compiled source changed; rebuild the selected Dockerfile "
            "instead of reusing this image"
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
SOURCE_ENV["REMNANT_MODEL"] = MODEL_KEY
app = modal.App("remnant")
model_volume = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
lswb_volume = modal.Volume.from_name("lswb-replay", create_if_missing=True)
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
    # The module imports scripts.sources at load time, also in this CPU container.
    .add_local_file(
        REPO_ROOT / "scripts/sources.py",
        str(REMOTE_REPO / "scripts/sources.py"),
        copy=True,
    )
)


compiled_image_id = os.environ.get("REMNANT_COMPILED_IMAGE")
if compiled_image_id:
    # Explicit CPU-checked reuse for Python-only iteration. Never refresh CUDA
    # source over old binaries: reject mismatches before mounting Python files.
    expected = _compiled_source_digest(
        [
            REPO_ROOT / INTEGRATION["sglang"] / "python/sglang/kernels",
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
        .run_function(
            _verify_compiled_sources, kwargs={"expected": expected}, timeout=180
        )
        .add_local_dir(
            REPO_ROOT / INTEGRATION["sglang"],
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
    # The sglang base image carries no Modal client dependencies (grpclib), so
    # the runtime fails at startup without this, as in the compiled-image path.
    fork_image.pip_install(f"modal=={modal.__version__}")
    .env(
        {
            **SOURCE_ENV,
            # Keep the Dockerfile's import path and add the mounted launcher scripts.
            "PYTHONPATH": f"/opt/sglang-runtime-fixes:{SGLANG_ROOT / 'python'}:{REMOTE_REPO}",
        }
    )
    .add_local_file(
        REPO_ROOT / "scripts/sources.py",
        str(REMOTE_REPO / "scripts/sources.py"),
        copy=True,
    )
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
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def download_model() -> str:
    """CPU-only pinned checkpoint download; existing shards are reused."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        max_workers=16,
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
    config = config.get("text_config", config)  # V4.1 nests the language model.
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
    env: dict[str, str],
    log: Path,
    deadline: float,
    fatal_markers: tuple[str, ...] = (),
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
        scanned = 0
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"serving process exited during startup; see {log}"
                    )
                if fatal_markers:
                    # A wedged worker can log its failure and keep the port closed.
                    with log.open("rb") as written:
                        written.seek(scanned)
                        fresh = written.read().decode(errors="replace")
                        scanned = written.tell()
                    for marker in fatal_markers:
                        if marker in fresh:
                            raise RuntimeError(f"startup log reports {marker!r}; see {log}")
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
    """Own one server boot and call the shared client against it."""
    if mode not in {"native", "triton", "kernel", "packed"}:
        raise ValueError(
            "mode must be native, triton, or kernel (packed aliases kernel)"
        )
    if mode == "packed":
        mode = "kernel"
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


# ---------------------------------------------------------------------------
# Paired LSWB replay (V4.1): one boot per arm, matched C_low then C_high.

LSWB_SERVER_MODES = {"v41-fp4": "fp4", "v41-remnant": "remnant"}
LSWB_REQUESTS, LSWB_USERS = 4916, 92  # The frozen business-user-replay-v2 corpus.
LSWB_SEED = 20260919
LSWB_FATAL_MARKERS = ("CUDA out of memory", "NCCL error", "CUDA error", "Segmentation fault")
LSWB_GPU = os.environ.get("REMNANT_LSWB_GPU", "H200:4")
LSWB_TP = 4
LSWB_MEM_FRAC = "0.92"
LSWB_MAX_RUN = 48
LSWB_MODEL_NAME = "deepseek-v4-1-flash"


@app.function(
    image=download_image.add_local_dir(
        REPO_ROOT / "scripts/local",
        REMOTE_REPO / "scripts/local",
        ignore=("**/__pycache__/**", "**/*.pyc"),
    ),
    cpu=2,
    memory=8192,
    timeout=1800,
    retries=0,
    volumes={str(LSWB_ROOT): lswb_volume},
)
def prepare_lswb() -> dict:
    """CPU-only: convert the uploaded raw LSWB JSONL once and record its identity."""
    packed = LSWB_ROOT / "longswebench.jsonl.gz"
    prepared = LSWB_ROOT / "replay"
    if not packed.is_file():
        raise FileNotFoundError(
            f"{packed} is missing; upload it with: modal volume put lswb-replay "
            "transfer/longswebench.jsonl.gz /longswebench.jsonl.gz"
        )
    if not prepared.exists():
        # The converter reads a plain file; keep the expansion off the volume.
        with tempfile.TemporaryDirectory() as scratch:
            raw = Path(scratch) / "longswebench.jsonl"
            with gzip.open(packed, "rb") as source, raw.open("wb") as target:
                shutil.copyfileobj(source, target, 1 << 20)
            subprocess.run(
                [
                    sys.executable,
                    str(REMOTE_REPO / "scripts/local/convert_lswb.py"),
                    "--format",
                    "replay",
                    "--input",
                    str(raw),
                    "--output-dir",
                    str(prepared),
                ],
                check=True,
                timeout=1500,
            )
    manifest = json.loads((prepared / "manifest.json").read_text())
    if (manifest["selected_requests"], manifest["selected_users"]) != (
        LSWB_REQUESTS,
        LSWB_USERS,
    ):
        raise RuntimeError(
            f"expected {LSWB_REQUESTS} requests from {LSWB_USERS} users, got "
            f"{manifest['selected_requests']} from {manifest['selected_users']}"
        )
    receipt = {
        key: manifest[key]
        for key in (
            "source_dataset_sha256",
            "dataset_sha256",
            "selected_requests",
            "selected_users",
        )
    }
    (LSWB_ROOT / "prepare-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    lswb_volume.commit()
    print(json.dumps(receipt), flush=True)
    return receipt


def _http_json(url: str, payload: dict | None = None, timeout: float = 60) -> dict:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _check_server_info(info: dict, fmt: str) -> None:
    """Fail before the replay if the booted server is not the requested arm."""
    expected = {
        "tp_size": LSWB_TP,
        "dsv41_main_kv_cache_format": fmt,
        "random_seed": LSWB_SEED,
        "max_running_requests": LSWB_MAX_RUN,
        "mem_fraction_static": float(LSWB_MEM_FRAC),
    }
    wrong = {k: info.get(k) for k, v in expected.items() if info.get(k) != v}
    if wrong:
        raise RuntimeError(f"server_info differs from the request: {wrong}; expected {expected}")
    sizes = info.get("cuda_graph_config", {}).get("decode", {}).get("bs") or []
    if max(sizes, default=0) < LSWB_MAX_RUN:
        raise RuntimeError(f"decode graphs {sizes} do not cover max_running_requests={LSWB_MAX_RUN}")


def _check_devices() -> list[str]:
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=30
    ).splitlines()
    if len(names) != LSWB_TP or not all("H200" in name for name in names):
        raise RuntimeError(f"expected {LSWB_TP} H200 devices, found {names}")
    return names


def _sanity_check(port: str) -> list[str]:
    """Greedy short prompts: non-degenerate text, so a bad layout fails before the replay."""
    replies = []
    for prompt in (
        "Name the capital of France in one sentence.",
        "Write a Python function that reverses a string.",
        "List three prime numbers greater than 100.",
    ):
        message = _http_json(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            {
                "model": LSWB_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 128,
            },
        )["choices"][0]["message"]
        text = (message.get("content") or "") + (message.get("reasoning_content") or "")
        if len(set(text)) < 8:
            raise RuntimeError(f"degenerate sanity reply for {prompt!r}: {text!r}")
        replies.append(text)
    return replies


def _replay_stalled(responses: Path, elapsed: float) -> str | None:
    """Reason to stop a replay early: nothing completes, or too many requests fail."""
    rows = [json.loads(line) for line in responses.read_text().splitlines()] if responses.is_file() else []
    if elapsed > 300 and not rows:
        return "no request completed in the first 300 s"
    failed = sum(not row.get("valid", False) for row in rows)
    if len(rows) >= 50 and failed / len(rows) > 0.05:
        return f"{failed}/{len(rows)} requests failed"
    return None


def _run_replay_point(env: dict[str, str], run: Path, tag: str, concurrency: int, duration: int) -> Path:
    root = run / "lswb-replay" / tag
    process = subprocess.Popen(
        [
            "bash",
            str(REMOTE_REPO / "scripts/local/bench-lswb.sh"),
            tag,
            env["PORT"],
            str(concurrency),
            str(duration),
        ],
        env={**env, "LSWB_RUN_ROOT": str(root)},
        cwd=REMOTE_REPO,
        start_new_session=True,
    )
    started = time.monotonic()
    limit = started + duration + 900
    try:
        while process.poll() is None:
            reason = None
            if time.monotonic() > limit:
                reason = "replay exceeded its duration plus drain allowance"
            else:
                reason = _replay_stalled(root / "client/responses.jsonl", time.monotonic() - started)
            if reason:
                raise RuntimeError(f"{tag}: {reason}")
            time.sleep(10)
        if process.returncode != 0:
            raise RuntimeError(f"{tag}: bench-lswb.sh exited {process.returncode}")
    except BaseException:
        _stop_serving_server(process)
        raise
    return root


@app.function(
    image=server_image,
    gpu=LSWB_GPU,
    cpu=8.0,
    memory=65536,
    timeout=3900,
    retries=0,
    volumes={
        str(MODEL_ROOT): model_volume,
        str(RESULTS_ROOT): results_volume,
        str(LSWB_ROOT): lswb_volume,
    },
)
def benchmark_lswb(
    mode: str,
    c_low: int = 0,
    c_high: int = 0,
    duration_s: int = 900,
    startup_minutes: int = 25,
    timeout_minutes: int = 62,
) -> str:
    """One server boot, then matched C_low and C_high replays on that server.

    C is derived from the booted pool unless both values are given (the second
    arm passes the first arm's c-plan.json values, so the pair stays matched).
    """
    from scripts.local.business_replay import capacity_plan

    if mode not in LSWB_SERVER_MODES:
        raise ValueError(f"mode must be one of {sorted(LSWB_SERVER_MODES)}")
    if MODEL_KEY != "v41-1017":
        raise ValueError("benchmark_lswb serves V4.1; set REMNANT_MODEL=v41-1017")
    if (c_low or c_high) and not 0 < c_low < c_high <= LSWB_MAX_RUN:
        raise ValueError(f"give both 0 < c_low < c_high <= {LSWB_MAX_RUN}, or neither")
    if not _kernel_extensions_ready():
        raise RuntimeError("Required Remnant extensions are missing; build the image on CPU first.")
    prepared = LSWB_ROOT / "replay"
    receipt = json.loads((LSWB_ROOT / "prepare-receipt.json").read_text())
    sessions = capacity_plan.session_sizes(json.loads((prepared / "manifest.json").read_text()))
    devices = _check_devices()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"{mode}-", dir=RESULTS_ROOT))
    started = time.monotonic()
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "HOST_REPO": str(REMOTE_REPO),
        "MODEL_PATH": str(MODEL_DIR),
        "SGLANG_ROOT": str(SGLANG_ROOT),
        "RESULTS_HOST": str(run),
        "LOG_HOST": str(run),
        "LSWB_EXECUTION": "direct",
        "REPLAY_DATASET": str(prepared / "records"),
        "REPLAY_MANIFEST": str(prepared / "manifest.json"),
        "SERVING_MODE": mode,
        "PORT": os.environ.get("PORT", "30211"),
        "TP": str(LSWB_TP),
        "MEM_FRAC": LSWB_MEM_FRAC,
        "MAX_RUN": str(LSWB_MAX_RUN),
        "MODEL_NAME": LSWB_MODEL_NAME,
        "SERVER_SEED": str(LSWB_SEED),
        "DECODE_CFG": (REMOTE_REPO / "scripts/local/config/decode-graphs-extended.json").read_text(),
        **SOURCE_ENV,
    }
    (run / "source-revisions.json").write_text(json.dumps(SOURCE_REVISIONS, indent=2) + "\n")
    (run / "source-identities.json").write_text(json.dumps(SOURCE_IDENTITIES, indent=2) + "\n")
    (run / "prepare-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (run / "devices.json").write_text(json.dumps(devices) + "\n")
    try:
        with _running_serving_server(
            env, run / f"serve_{mode}.log", started + startup_minutes * 60, LSWB_FATAL_MARKERS
        ):
            info = _http_json(f"http://127.0.0.1:{env['PORT']}/server_info", timeout=30)
            (run / "server-info.json").write_text(json.dumps(info, indent=2) + "\n")
            _check_server_info(info, LSWB_SERVER_MODES[mode])
            pool = int(info["max_total_num_tokens"])
            plan = capacity_plan.plan_concurrency(sessions, pool, LSWB_MAX_RUN)
            if c_high:
                # Explicit C from the first arm: only require that it fits this pool.
                peak = capacity_plan.peak_working_set(sessions, c_high)
                if peak > pool:
                    raise RuntimeError(f"C={c_high} peaks at {peak} tokens, above the {pool}-token pool")
                plan = {**plan, "c_low": c_low, "c_high": c_high, "given": True}
            (run / "c-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            (run / "sanity.json").write_text(json.dumps(_sanity_check(env["PORT"]), indent=2) + "\n")
            for concurrency in (plan["c_low"], plan["c_high"]):
                if time.monotonic() - started + duration_s + 900 > timeout_minutes * 60:
                    raise TimeoutError("remaining budget cannot cover another replay point")
                _run_replay_point(env, run, f"{mode}-c{concurrency}", concurrency, duration_s)
                results_volume.commit()
    finally:
        results_volume.commit()
    return str(run)


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
