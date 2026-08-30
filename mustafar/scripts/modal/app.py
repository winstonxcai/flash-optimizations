"""Modal entry points for Mustafar Stage-1 validation.

The model is intentionally stored in a Modal Volume rather than baked into
the server image.  Run the one-time population step with::

    modal run mustafar/scripts/modal/app.py::download_model
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import os
import re
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
    # Tests and Modal orchestration are mounted below. Keeping them out of the
    # Docker context means an assertion or benchmark edit does not invalidate
    # the large, CPU-built SGLang image.
    ignore=("mustafar/tests/**", "mustafar/scripts/modal/**"),
).add_local_dir(
    REPO_ROOT / "mustafar" / "tests",
    "/opt/mustafar/flash-optimizations/mustafar/tests",
).add_local_dir(
    REPO_ROOT / "mustafar" / "scripts" / "modal",
    "/opt/mustafar/flash-optimizations/mustafar/scripts/modal",
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
        PYTHONPATH=(
            "/opt/sglang-runtime-fixes:"
            "/sgl-workspace/sglang-lowrank/python:"
            "/opt/mustafar/flash-optimizations"
        ),
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
        PYTHONPATH=(
            "/opt/sglang-runtime-fixes:"
            "/sgl-workspace/sglang-lowrank/python:"
            "/opt/mustafar/flash-optimizations"
        ),
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


def _wait_for_pool_allocation(
    log_path: Path,
    process: subprocess.Popen,
    timeout: int = 1200,
) -> str:
    """Return the log once SGLang has physically allocated its KV pools."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            if "max_total_num_tokens=" in text and "Memory pool end." in text:
                return text
        if process.poll() is not None:
            text = log_path.read_text(errors="replace") if log_path.exists() else ""
            raise RuntimeError(
                f"server exited before pool allocation ({process.returncode})\n"
                + "\n".join(text.splitlines()[-80:])
            )
        time.sleep(2)
    raise TimeoutError("server did not allocate its KV pools before timeout")


def _parse_pool_ceiling(log_text: str) -> dict[str, object]:
    memory = re.search(
        r"bytes_per_full_token=([0-9.]+), available_bytes=([0-9.]+) GB, "
        r"c128_state_fixed=([0-9.]+) GB, full_token=([0-9]+)",
        log_text,
    )
    pools = re.search(
        r"DSV4 pool sizes: full=([0-9]+), swa=([0-9]+), c4=([0-9]+), "
        r"c128=([0-9]+), c4_state=([0-9]+), c128_state=([0-9]+)",
        log_text,
    )
    maximum = re.search(r"max_total_num_tokens=([0-9]+)", log_text)
    if memory is None or pools is None or maximum is None:
        raise RuntimeError("could not parse DSV4 pool allocation from server log")

    full_tokens = int(maximum.group(1))
    page_size = 256
    output_tokens = 16
    contexts: dict[str, object] = {}
    for label, context_tokens in (
        ("32k", 32 * 1024),
        ("64k", 64 * 1024),
        ("128k", 128 * 1024),
    ):
        occupied_per_request = math.ceil(
            (context_tokens + output_tokens) / page_size
        ) * page_size
        contexts[label] = {
            "context_tokens": context_tokens,
            "output_tokens": output_tokens,
            "page_rounded_tokens_per_request": occupied_per_request,
            "theoretical_context_only": full_tokens // context_tokens,
            "resident_requests_with_output": full_tokens // occupied_per_request,
            "remaining_tokens": full_tokens % occupied_per_request,
        }

    packed_bytes = re.search(r"logical_row_bytes=328 allocated_bytes=([0-9]+)", log_text)
    return {
        "bytes_per_full_token": float(memory.group(1)),
        "available_gib_reported": float(memory.group(2)),
        "c128_state_fixed_gib_reported": float(memory.group(3)),
        "full_tokens": full_tokens,
        "pools": {
            "full": int(pools.group(1)),
            "swa": int(pools.group(2)),
            "c4": int(pools.group(3)),
            "c128": int(pools.group(4)),
            "c4_state": int(pools.group(5)),
            "c128_state": int(pools.group(6)),
        },
        "packed_c4_allocated_bytes": (
            int(packed_bytes.group(1)) if packed_bytes is not None else None
        ),
        "contexts": contexts,
    }


@app.function(
    image=server_image,
    gpu="H100!:4",
    cpu=32,
    memory=262144,
    timeout=2 * 60 * 60,
    volumes={
        str(MODEL_VOLUME_ROOT): model_volume,
        str(RESULTS_ROOT): results_volume,
    },
)
def tp4_capacity_ceiling() -> dict[str, object]:
    """Measure dense and packed TP4 ceilings with official bench_serving.

    Runs one page-aligned ceiling point at each context rather than the
    0.8/1.0/1.15 sweep. Server logs distinguish resident from queued requests.
    """
    base_env = os.environ.copy()
    base_env.update(
        MODEL_PATH=str(MODEL_DIR),
        TP="4",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        SGLANG_OPT_TOPMAG="1",
        XKV_TOPMAG_KEEP="0.5",
        PYTHONPATH=(
            "/opt/sglang-runtime-fixes:"
            "/sgl-workspace/sglang-lowrank/python:"
            "/opt/mustafar/flash-optimizations"
        ),
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
        "--skip-server-warmup",
        "--reasoning-parser", "deepseek-v4",
        "--tool-call-parser", "deepseekv4",
        "--watchdog-timeout", "1800",
    ]
    started_all = time.time()
    result: dict[str, object] = {
        "experiment": "tp4-memory-bound-ceiling",
        "model_revision": MODEL_REVISION,
        "tp": 4,
        "mem_fraction_static": 0.93,
        "max_running_requests": 64,
        "cuda_graph": False,
        "method": "official python -m sglang.bench_serving at allocator ceiling",
        "legs": {},
    }
    run_dir = RESULTS_ROOT / "official-capacity"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Run the experimental leg first so an external interruption still leaves
    # the primary capacity result committed to the Volume.
    for name, packed in (("topmag50_packed", True), ("native", False)):
        env = dict(base_env)
        env["SGLANG_OPT_TOPMAG_PACKED_C4"] = "1" if packed else "0"
        log_path = RESULTS_ROOT / f"capacity-{name}-server.log"
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
                _wait_for_server(30211, process, timeout=1200)
                log_text = _wait_for_pool_allocation(log_path, process)
                leg_result: dict[str, object] = {
                    "packed": packed,
                    "seconds_to_server_ready": time.time() - started,
                    **_parse_pool_ceiling(log_text),
                    "benchmarks": {},
                }
                result["legs"][name] = leg_result
                for label, context in leg_result["contexts"].items():
                    concurrency = context["resident_requests_with_output"]
                    output_file = run_dir / f"{name}-{label}.jsonl"
                    if output_file.exists():
                        output_file.unlink()
                    before = len(log_path.read_text(errors="replace"))
                    bench_started = time.time()
                    bench = subprocess.run(
                        [
                            "python3", "-m", "sglang.bench_serving",
                            "--backend", "sglang",
                            "--host", "127.0.0.1", "--port", "30211",
                            "--model", str(MODEL_DIR),
                            "--tokenizer", str(MODEL_DIR),
                            "--dataset-name", "random",
                            "--random-input-len", str(context["context_tokens"]),
                            "--random-output-len", "16",
                            "--random-range-ratio", "1.0",
                            "--num-prompts", str(concurrency),
                            "--max-concurrency", str(concurrency),
                            "--request-rate", "inf",
                            "--warmup-requests", "0",
                            "--flush-cache",
                            "--tokenize-prompt",
                            "--output-file", str(output_file),
                            "--output-details",
                            "--seed", "42",
                        ],
                        cwd="/sgl-workspace/sglang-lowrank/python",
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=30 * 60,
                        check=False,
                    )
                    server_delta = log_path.read_text(errors="replace")[before:]
                    running = [
                        int(x) for x in re.findall(r"#running-req: ([0-9]+)", server_delta)
                    ]
                    queued = [
                        int(x) for x in re.findall(r"#queue-req: ([0-9]+)", server_delta)
                    ]
                    full_usage = [
                        float(x)
                        for x in re.findall(r"full token usage: ([0-9.]+)", server_delta)
                    ]
                    records = []
                    if output_file.exists():
                        records = [
                            json.loads(line)
                            for line in output_file.read_text().splitlines()
                            if line.strip()
                        ]
                    bench_result = {
                        "official_module": "sglang.bench_serving",
                        "returncode": bench.returncode,
                        "seconds": time.time() - bench_started,
                        "concurrency": concurrency,
                        "max_running_observed": max(running, default=0),
                        "max_queued_observed": max(queued, default=0),
                        "max_full_token_usage": max(full_usage, default=0.0),
                        "record": records[-1] if records else None,
                        "stdout_tail": bench.stdout.splitlines()[-80:],
                        "stderr_tail": bench.stderr.splitlines()[-40:],
                    }
                    leg_result["benchmarks"][label] = bench_result
                    print(
                        f"[{name} {label}] rc={bench.returncode} "
                        f"C={concurrency} running={bench_result['max_running_observed']} "
                        f"queued={bench_result['max_queued_observed']}",
                        flush=True,
                    )
                    if bench.returncode:
                        raise RuntimeError(
                            f"official bench_serving failed for {name}/{label}: "
                            + "\n".join(bench.stderr.splitlines()[-40:])
                        )
                    partial = RESULTS_ROOT / "tp4-capacity-ceiling.partial.json"
                    partial.write_text(
                        json.dumps(result, indent=2, sort_keys=True) + "\n"
                    )
                    results_volume.commit()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                log.flush()
                results_volume.commit()

    native = result["legs"]["native"]
    packed = result["legs"]["topmag50_packed"]
    result["full_token_gain"] = packed["full_tokens"] / native["full_tokens"]
    result["total_gpu_wall_seconds"] = time.time() - started_all
    output = RESULTS_ROOT / "tp4-capacity-ceiling.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    results_volume.commit()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _validate_decode_graph_record(
    record: dict[str, object] | None,
    *,
    expected_requests: int,
    expected_output_tokens: int,
) -> dict[str, object]:
    """Fail closed if bench_serving did not exercise the requested graph path."""
    if record is None:
        raise RuntimeError("bench_serving did not write a result record")
    if record.get("completed") != expected_requests:
        raise RuntimeError(
            "bench_serving completed an unexpected number of requests: "
            f"{record.get('completed')} != {expected_requests}"
        )
    errors = [error for error in record.get("errors", []) if error]
    if errors:
        raise RuntimeError(f"bench_serving returned request errors: {errors}")
    output_lens = record.get("output_lens", [])
    if len(output_lens) != expected_requests or any(
        length != expected_output_tokens for length in output_lens
    ):
        raise RuntimeError(
            "bench_serving did not generate the requested decode length: "
            f"{output_lens}"
        )

    server_info = record.get("server_info") or {}
    graph_config = server_info.get("cuda_graph_config") or {}
    decode = graph_config.get("decode") or {}
    backend = decode.get("backend")
    batches = decode.get("bs") or []
    if backend == "disabled" or not {1, 2}.issubset(set(batches)):
        raise RuntimeError(
            "decode CUDA graph was not configured for batch 1 and 2: "
            f"{decode}"
        )
    return {
        "decode_graph_backend": backend,
        "decode_graph_batches": batches,
    }


@app.function(
    image=server_image,
    gpu="H100!:4",
    cpu=32,
    memory=262144,
    timeout=2 * 60 * 60,
    volumes={
        str(MODEL_VOLUME_ROOT): model_volume,
        str(RESULTS_ROOT): results_volume,
    },
)
def tp4_graph_decode_ab() -> dict[str, object]:
    """Short native/packed TP4 decode A/B with BS1/BS2 CUDA graphs."""
    context_tokens = 64 * 1024
    output_tokens = 128
    graph_config = {
        "decode": {"backend": "full", "max_bs": 2, "bs": [1, 2]},
        "prefill": {"backend": "disabled"},
    }
    base_env = os.environ.copy()
    base_env.update(
        MODEL_PATH=str(MODEL_DIR),
        TP="4",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        SGLANG_OPT_TOPMAG="1",
        XKV_TOPMAG_KEEP="0.5",
        PYTHONPATH=(
            "/opt/sglang-runtime-fixes:"
            "/sgl-workspace/sglang-lowrank/python:"
            "/opt/mustafar/flash-optimizations"
        ),
    )
    command = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", str(MODEL_DIR),
        "--served-model-name", "deepseek-v4-flash",
        "--tp", "4", "--trust-remote-code",
        "--mem-fraction-static", "0.93",
        "--context-length", "135168",
        "--max-running-requests", "2",
        "--chunked-prefill-size", "4096",
        "--fp8-gemm-backend", "triton",
        "--host", "0.0.0.0", "--port", "30211",
        "--cuda-graph-config", json.dumps(graph_config, separators=(",", ":")),
        "--skip-server-warmup",
        "--reasoning-parser", "deepseek-v4",
        "--tool-call-parser", "deepseekv4",
        "--watchdog-timeout", "1800",
    ]
    started_all = time.time()
    result: dict[str, object] = {
        "experiment": "tp4-graph-decode-ab",
        "model_revision": MODEL_REVISION,
        "tp": 4,
        "context_tokens": context_tokens,
        "output_tokens": output_tokens,
        "mem_fraction_static": 0.93,
        "max_running_requests": 2,
        "cuda_graph_config": graph_config,
        "method": "official python -m sglang.bench_serving",
        "legs": {},
    }
    run_dir = RESULTS_ROOT / "official-graph-decode"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Run packed first: it is the experimental path and its artifact survives
    # even if an account limit interrupts the later native reference leg.
    for name, packed in (("topmag50_packed", True), ("native", False)):
        env = dict(base_env)
        env["SGLANG_OPT_TOPMAG_PACKED_C4"] = "1" if packed else "0"
        log_path = RESULTS_ROOT / f"graph-decode-{name}-server.log"
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
                _wait_for_server(30211, process, timeout=1800)
                log_text = _wait_for_pool_allocation(log_path, process, timeout=1800)
                leg_result: dict[str, object] = {
                    "packed": packed,
                    "seconds_to_server_ready": time.time() - started,
                    **_parse_pool_ceiling(log_text),
                    "benchmarks": {},
                }
                result["legs"][name] = leg_result

                for concurrency, num_prompts in ((1, 2), (2, 4)):
                    label = f"c{concurrency}"
                    output_file = run_dir / f"{name}-{label}.jsonl"
                    if output_file.exists():
                        output_file.unlink()
                    before = len(log_path.read_text(errors="replace"))
                    bench_started = time.time()
                    bench = subprocess.run(
                        [
                            "python3", "-m", "sglang.bench_serving",
                            "--backend", "sglang",
                            "--host", "127.0.0.1", "--port", "30211",
                            "--model", str(MODEL_DIR),
                            "--tokenizer", str(MODEL_DIR),
                            "--dataset-name", "random",
                            "--random-input-len", str(context_tokens),
                            "--random-output-len", str(output_tokens),
                            "--random-range-ratio", "1.0",
                            "--num-prompts", str(num_prompts),
                            "--max-concurrency", str(concurrency),
                            "--request-rate", "inf",
                            "--warmup-requests", "0",
                            "--flush-cache",
                            "--tokenize-prompt",
                            "--output-file", str(output_file),
                            "--output-details",
                            "--seed", str(4200 + concurrency),
                        ],
                        cwd="/sgl-workspace/sglang-lowrank/python",
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=30 * 60,
                        check=False,
                    )
                    records = []
                    if output_file.exists():
                        records = [
                            json.loads(line)
                            for line in output_file.read_text().splitlines()
                            if line.strip()
                        ]
                    record = records[-1] if records else None
                    if bench.returncode:
                        raise RuntimeError(
                            f"official bench_serving failed for {name}/{label}:\n"
                            + "\n".join(bench.stderr.splitlines()[-80:])
                        )
                    graph_evidence = _validate_decode_graph_record(
                        record,
                        expected_requests=num_prompts,
                        expected_output_tokens=output_tokens,
                    )
                    server_delta = log_path.read_text(errors="replace")[before:]
                    replay_values = re.findall(
                        r"cuda graph:\s*(True|False)", server_delta, re.IGNORECASE
                    )
                    bench_result = {
                        "official_module": "sglang.bench_serving",
                        "returncode": bench.returncode,
                        "seconds": time.time() - bench_started,
                        "concurrency": concurrency,
                        "num_prompts": num_prompts,
                        "record": record,
                        "graph_replay_log_values": replay_values,
                        **graph_evidence,
                        "stdout_tail": bench.stdout.splitlines()[-80:],
                        "stderr_tail": bench.stderr.splitlines()[-40:],
                    }
                    leg_result["benchmarks"][label] = bench_result
                    print(
                        f"[{name} {label}] rc=0 completed={num_prompts} "
                        f"graph={graph_evidence['decode_graph_backend']} "
                        f"ITL={record.get('median_itl_ms')} ms",
                        flush=True,
                    )
                    partial = RESULTS_ROOT / "tp4-graph-decode-ab.partial.json"
                    partial.write_text(
                        json.dumps(result, indent=2, sort_keys=True) + "\n"
                    )
                    results_volume.commit()

                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited after benchmarks with {process.returncode}"
                    )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                log.flush()
                results_volume.commit()

    result["total_gpu_wall_seconds"] = time.time() - started_all
    output = RESULTS_ROOT / "tp4-graph-decode-ab.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    results_volume.commit()
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    """Default to the safe, idempotent model population operation."""
    print(download_model.remote())
