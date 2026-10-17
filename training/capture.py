"""Native latent-KV capture in an already prepared runtime environment.

The shell launcher owns containers, images, mounts and GPU selection. This module
validates capture inputs, runs the existing server and calibration client, and
cleans up its server process group. It never launches Docker or builds kernels.
Author: Winston Cai.
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/local/run-server.sh"
RUNNER = Path(
    "/sgl-workspace/sglang-remnant/python/sglang/benchmark/remnant/calibrate.py"
)
MIN_SCRATCH_BYTES = 50 * 1024**3
SHUTDOWN_SECONDS = 15


@contextmanager
def _termination_signals():
    """Make signals unwind owned subprocess cleanup, then restore callers."""

    def terminate(number, _frame):
        raise SystemExit(128 + number)

    previous = {
        number: signal.signal(number, terminate)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def preflight(root: Path, published: Path, model: Path, port: int) -> None:
    """Validate mounted inputs and storage, without starting a server or model."""
    if not 1 <= port <= 65535:
        raise ValueError("port in 1..65535 required")
    if any(
        not path.is_absolute() or path.resolve() == Path("/")
        for path in (root, published, model)
    ):
        raise ValueError(
            "explicit absolute, non-root capture/model/publication directories required"
        )
    if root.resolve() == published.resolve():
        raise ValueError("scratch and persistent destinations must differ")
    if not model.is_dir() or not (root / "replay.jsonl").is_file():
        raise ValueError(
            "model directory and prepared capture root/replay.jsonl required"
        )
    config = json.loads((root / "capture-config.json").read_text())
    if config.get("storage_schema_version") != 2 or not isinstance(
        config.get("audit_shared"), bool
    ):
        raise ValueError("schema 2 and an explicit audit_shared policy are required")
    if config.get("concurrency") != 2:
        raise ValueError("capture pilot requires concurrency two")
    if not RUNNER.is_file():
        raise ValueError(f"calibration client is missing: {RUNNER}")
    if not config["audit_shared"]:
        spec = importlib.util.spec_from_file_location("capture_preflight", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.verify_shared_rollout(
            config, json.loads((root / "shared-audit-proof.json").read_text())
        )
    published.mkdir(parents=True, exist_ok=True)
    for destination in (root, published):
        # Detect read-only mounts even when access bits permit writes.
        with tempfile.TemporaryFile(dir=destination) as probe:
            probe.write(b"capture mount probe")
            probe.flush()
    if shutil.disk_usage(root).free < MIN_SCRATCH_BYTES:
        raise ValueError(f"local scratch {root} must leave 50 GiB free")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(
            "capture total budget exhausted (startup and replay share one deadline)"
        )
    return remaining


def _wait_healthy(server: subprocess.Popen, endpoint: str, deadline: float) -> None:
    while True:
        remaining = _remaining(deadline)
        if server.poll() is not None:
            raise RuntimeError(
                f"Native capture server exited: {server.returncode}; inspect /capture/server.log"
            )
        try:
            with urlopen(endpoint + "/health", timeout=min(3, remaining)):
                return
        except (URLError, TimeoutError, OSError):
            time.sleep(min(2, _remaining(deadline)))


def _stop_server(server: subprocess.Popen) -> None:
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=SHUTDOWN_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The session leader can exit before its worker processes do.
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait(timeout=5)


def run(
    port: int,
    *,
    deadline_epoch: float,
    root: Path = Path("/capture"),
    published: Path = Path("/published"),
    model: Path = Path("/model"),
) -> None:
    """Run capture under one total deadline; preserve partial artifacts on failure."""
    if not math.isfinite(deadline_epoch):
        raise ValueError("finite capture deadline required")
    # Convert the launcher's wall-clock deadline once; thereafter use monotonic time.
    deadline = time.monotonic() + deadline_epoch - time.time()
    _remaining(deadline)
    preflight(root, published, model, port)
    _remaining(deadline)
    environment = {
        **os.environ,
        "TP": "4",
        "MAX_RUN": "2",
        "MEM_FRAC": "0.88",
        "CHUNK": "8192",
        "CTX_LEN": "1048576",
        "SERVER_SEED": "42",
        "DECODE_CFG": '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}',
        "REMNANT_CAPTURE_DIR": str(root),
        "SERVER_PID_FILE": str(root / "server.pid"),
        "SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES": "1",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    }
    for name in ("NSYS_OUTPUT", "REQUEST_METRICS_DIR", "STORE_IMPLEMENTATION"):
        environment.pop(name, None)
    endpoint = f"http://127.0.0.1:{port}"
    with _termination_signals(), (root / "server.log").open("w") as log:
        server = subprocess.Popen(
            ["bash", str(LAUNCHER), "native", str(model), "127.0.0.1", str(port)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_healthy(server, endpoint, deadline)
            remaining = int(_remaining(deadline))
            if remaining < 1:
                raise TimeoutError("no replay budget remains after Native startup")
            subprocess.run(
                [
                    "python3",
                    str(RUNNER),
                    "capture-client",
                    "--root",
                    str(root),
                    "--endpoint",
                    endpoint,
                    "--seconds",
                    str(remaining),
                    "--publish-root",
                    str(published),
                ],
                check=True,
                text=True,
                env=environment,
                timeout=_remaining(deadline),
            )
        finally:
            _stop_server(server)


def main(arguments=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "run"):
        command = actions.add_parser(action)
        command.add_argument("--root", type=Path, default=Path("/capture"))
        command.add_argument("--publish-root", type=Path, default=Path("/published"))
        command.add_argument("--model", type=Path, default=Path("/model"))
        command.add_argument("--port", type=int, default=30312)
        if action == "run":
            command.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(arguments)
    if args.action == "preflight":
        preflight(args.root, args.publish_root, args.model, args.port)
    else:
        run(
            args.port,
            deadline_epoch=args.deadline_epoch,
            root=args.root,
            published=args.publish_root,
            model=args.model,
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Capture failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
