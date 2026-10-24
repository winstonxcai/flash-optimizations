"""Collect one terminal run and publish its packet. Author: Winston Cai."""
import hashlib, json, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

repo, batch = map(lambda p: Path(p).resolve(), sys.argv[1:])

def command(*args, timeout=60):
    return subprocess.check_output(args, cwd=repo, text=True, timeout=timeout)

ids = set(re.findall(
    r"\bjob-\d{8}-\d{6}-[0-9a-f]{8}\b", (batch/"job.txt").read_text()
))
if len(ids) != 1:
    raise SystemExit(f"Expected one submitted job: {ids}")
job_id = ids.pop()
deadline = time.monotonic() + 86400
previous = None

while True:
    snapshot = json.loads(command(
        "gpuq", "jobs", "--all", "--user", "winstonxcai", timeout=30
    ))
    jobs = snapshot["jobs"] if isinstance(snapshot, dict) else snapshot
    matches = [j for j in jobs if j.get("id") == job_id]
    if len(matches) != 1:
        raise SystemExit(f"Cannot identify {job_id}")
    job = matches[0]
    status = job.get("status", job.get("state"))
    if status != previous:
        print(job_id, status, flush=True)
        previous = status
    if status in {"succeeded", "failed", "cancelled", "canceled",
                  "timed_out", "timeout"}:
        break
    if time.monotonic() >= deadline:
        raise SystemExit("Transfer monitor timed out; GPUQ job was not cancelled")
    time.sleep(30)

dest = repo/"transfer"/batch.name
if dest.exists():
    raise SystemExit(f"Transfer already exists: {dest}")
dest.parent.mkdir(exist_ok=True)
missing = []

with tempfile.TemporaryDirectory(prefix=".c28-transfer-", dir=dest.parent) as tmp:
    packet = Path(tmp)/"packet"
    packet.mkdir()

    def copy(source, relative):
        if source.is_symlink() or not source.resolve().is_relative_to(repo):
            raise SystemExit(f"Unsafe artifact: {source}")
        if not source.is_file():
            missing.append(str(source.relative_to(repo)))
            return
        target = packet/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    for name in (
        "run.sh", "transfer.py", "job.txt", "parent-commit.txt", "image-id.txt",
        "source-manifest.json", "source-at-start.json", "launcher-sha256.txt",
        "assigned-gpus.txt", "result-root.txt", "job-exit-code.txt",
    ):
        copy(batch/name, name)
    for name in (
        "serve_kernel.log", "server-info.json",
        "gc-freeze.log", "replay-exit-code.txt",
    ):
        copy(batch/"boot"/name, f"boot/{name}")

    (packet/"jobs.json").write_text(json.dumps({"jobs": [job]}, indent=2)+"\n")
    (packet/"gpuq.log").write_text(command("gpuq", "logs", job_id, timeout=30))

    original = None
    pointer = batch/"result-root.txt"
    if pointer.is_file():
        tag = Path(pointer.read_text().strip()).resolve()
        if not tag.is_relative_to(repo/"results/lswb-replay"):
            raise SystemExit(f"Unexpected result root: {tag}")
        found = sorted(p for p in tag.iterdir() if p.is_dir()) if tag.is_dir() else []
        if len(found) > 1:
            raise SystemExit(f"Ambiguous replay results: {found}")
        if found:
            run = found[0]
            original = str(run.relative_to(repo))
            for name in (
                "client/summary.json", "client/responses.jsonl",
                "client/manifest.json", "client/dataset_manifest.json",
                "client/done.json", "client/active_requests.jsonl",
                "client.log", "cache-flush.log", "server.log",
                "server-info.json", "replay-window.json", "request-timings.jsonl",
                "request-timing-validation.json", "collection-config.json",
            ):
                copy(run/name, f"replay/{name}")

    if original is None:
        missing.append("No replay directory; boot and job failure diagnostics retained")

    (packet/"transfer-manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "description": "Packed Kernel256 C25 W1 TP4 MF0.88 HiCache-off original replay1200s n1",
        "original_run": original,
        "missing_artifacts": missing,
        "classification": "Screening evidence; response and timing qualification pending",
        "baseline": "frontend-preparation-workers-20261023_213929-RtkRUV/native-c24-w1",
    }, indent=2)+"\n")

    files = sorted(p for p in packet.rglob("*") if p.is_file())
    (packet/"SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(packet)}\n"
        for p in files
    ))
    packet.rename(dest)


import fcntl
with (repo/"logs/packed-w1-transfer.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if command("git", "branch", "--show-current").strip() != "main":
        raise SystemExit(f"Packet saved at {dest}; expected main")
    relative = str(dest.relative_to(repo))
    for args in (
        ("git", "add", "--", relative),
        ("git", "commit", "--only", "-m",
         f"Transfer {batch.name}", "--", relative),
        ("git", "push", "origin", "HEAD:main"),
    ):
        subprocess.run(args, cwd=repo, check=True, timeout=300)
print(f"Pushed {relative}; missing artifacts: {len(missing)}", flush=True)
