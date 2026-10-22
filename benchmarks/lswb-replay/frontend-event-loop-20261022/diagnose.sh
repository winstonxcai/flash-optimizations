#!/usr/bin/env bash
# Locate first-output residence time using existing serving/replay entry points.
# Author: Winston Cai. Experiment protocol and lifecycle: adjacent README.md.
set -euo pipefail

SCRIPT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/diagnose.sh
REPO=$(cd -- "$(dirname -- "$SCRIPT")/../../.." && pwd)
cd -- "$REPO"
export PYTHONDONTWRITEBYTECODE=1
export DSV4_ASYNC_PREPARATION=${DSV4_ASYNC_PREPARATION:-0}
case "$DSV4_ASYNC_PREPARATION" in 0|1) ;; *) echo "DSV4_ASYNC_PREPARATION must be 0 or 1" >&2; exit 2 ;; esac
# GPUQ executes as its runner account; trust only this experiment's four trees.
export GIT_CONFIG_COUNT=4
trees=("$REPO" "$REPO/third_party/sglang" "$REPO/third_party/flashmla" "$REPO/third_party/flashmla/csrc/cutlass")
for index in 0 1 2 3; do
  export "GIT_CONFIG_KEY_$index=safe.directory" "GIT_CONFIG_VALUE_$index=${trees[$index]}"
done
export SGLANG_ROOT_CT="/mnt/host_root$REPO/third_party/sglang"
export SGLANG_PY="$SGLANG_ROOT_CT/python"
if [[ "${1:-}" = preflight || "${1:-}" = submit ]] && [ "$#" -eq 3 ]; then
  export REPLAY_DATASET=$2 REPLAY_MANIFEST=$3
fi
. scripts/local/env.sh
. scripts/local/eval-env.sh

preflight() {
  check_sources
  require_replay_config
  [ -z "$(git status --porcelain --untracked-files=no)" ] || {
    echo "Commit tracked parent changes before this diagnostic" >&2; return 1;
  }
  timeout --foreground 120 docker exec -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT" \
    -e SGLANG_FRONTEND_DIAGNOSTICS=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -w "$SGLANG_ROOT_CT" "$CONTAINER" python3 \
    test/registered/unit/observability/test_req_time_stats.py
  if [ "$DSV4_ASYNC_PREPARATION" = 1 ]; then
    timeout --foreground 120 docker exec -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT" \
      -e SGLANG_DSV4_ASYNC_PREPARATION=1 -e PYTHONDONTWRITEBYTECODE=1 \
      -w "$SGLANG_ROOT_CT" "$CONTAINER" python3 \
      test/registered/unit/entrypoints/openai/test_serving_chat.py AsyncInputPreparationTests -v
  fi
  timeout --foreground 120 python3 -m unittest discover -s scripts/local/business_replay -p 'test_*.py' -v
  echo "CPU source, timing and replay preflight PASS; no GPU job allocated"
}

case "${1:-}" in
  preflight) preflight ;;
  submit)
    command -v gpuq >/dev/null
    ROOT=$(mktemp -d "$REPO/logs/frontend-first-output-asyncprep$DSV4_ASYNC_PREPARATION-$(date +%Y%m%d_%H%M%S)-XXXXXX")
    chgrp gpuq "$ROOT"
    chmod 2775 "$ROOT"
    preflight 2>&1 | tee "$ROOT/cpu-preflight.log"
    gpuq run --help > "$ROOT/gpuq-run-help.txt"
    printf '%s\n' "$DSV4_ASYNC_PREPARATION" > "$ROOT/async-preparation.txt"
    git rev-parse HEAD > "$ROOT/parent-commit.txt"
    docker inspect "$CONTAINER" --format '{{.Image}}' > "$ROOT/image-id.txt"
    sha256sum "$SCRIPT" scripts/local/{serve,run-server,env,eval-env,bench-lswb}.sh \
      scripts/local/business_replay/{runner,request_slots}.py \
      scripts/local/business_replay/adapters/openai_sse.py > "$ROOT/launcher-sha256.txt"
    check_sources > "$ROOT/source-manifest.json"
    for mode in native kernel; do
      if [ "$mode" = native ]; then port=30631; master=30031; else port=30632; master=30032; fi
      gpuq run --user winstonxcai --project flash-optimizations --gpus 4 \
        --cwd "$REPO" --env "REPLAY_DATASET=$REPLAY_DATASET" \
        --env "REPLAY_MANIFEST=$REPLAY_MANIFEST" --env "CONTAINER=$CONTAINER" \
        --env "DSV4_ASYNC_PREPARATION=$DSV4_ASYNC_PREPARATION" \
        --output "$ROOT/gpuq-$mode" --workload-key "frontend-first-output-$mode-$(basename "$ROOT")" \
        --expected-duration 30m --timeout 2400 --detach \
        -- bash "$SCRIPT" run "$mode" "$ROOT" "$port" "$master" \
        | tee "$ROOT/job-$mode.txt"
    done
    echo "Diagnostic batch: $ROOT"
    ;;
  collect)
    [ "$#" -eq 2 ] || { echo "usage: bash $SCRIPT collect <batch-directory>" >&2; exit 2; }
    python3 - "$REPO" "$2" <<'PY_COLLECT'
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

repo = Path(sys.argv[1]).resolve()
argument = Path(sys.argv[2])
if argument.is_absolute():
    batch = argument.resolve()
elif argument.parent == Path("."):
    batch = (repo / "logs" / argument).resolve()
else:
    batch = (repo / argument).resolve()
if batch.parent != (repo / "logs").resolve() or not batch.name.startswith("frontend-first-output-"):
    raise SystemExit(f"Unexpected diagnostic batch directory: {batch}")
if not batch.is_dir():
    raise SystemExit(f"Diagnostic batch does not exist: {batch}")

# Query only; collection never starts, cancels, or retries a GPU job.
snapshot = json.loads(subprocess.check_output(
    ["gpuq", "jobs", "--all", "--user", "winstonxcai"], text=True, timeout=30,
))
jobs = snapshot["jobs"] if isinstance(snapshot, dict) else snapshot
if not isinstance(jobs, list):
    raise SystemExit("GPUQ jobs response must contain a list")
terminal = {"succeeded", "failed", "cancelled", "canceled", "timed_out", "timeout"}
selected = {}
for mode in ("native", "kernel"):
    submission = batch / f"job-{mode}.txt"
    if not submission.is_file() or submission.is_symlink():
        raise SystemExit(f"Missing or unsafe saved submission: {submission}")
    ids = set(re.findall(r"\bjob-\d{8}-\d{6}-[0-9a-f]{8}\b", submission.read_text()))
    if len(ids) != 1:
        raise SystemExit(f"Expected one saved {mode} job ID in {submission}, got {sorted(ids)}")
    job_id = ids.pop()
    matches = [job for job in jobs if job.get("id") == job_id]
    if len(matches) != 1:
        raise SystemExit(f"Expected one GPUQ record for {mode} {job_id}, got {len(matches)}")
    job = matches[0]
    status = job.get("status", job.get("state"))
    if status not in terminal:
        raise SystemExit(f"Refusing collection: {mode} {job_id} is {status!r}, not terminal")
    expected = f"frontend-first-output-{mode}-{batch.name}"
    if job.get("workload_key") != expected:
        raise SystemExit(f"Job identity mismatch for {job_id}: expected workload_key {expected!r}")
    selected[mode] = job
if len({job["id"] for job in selected.values()}) != 2:
    raise SystemExit("Native and Kernel submissions must identify two distinct jobs")

transfer = repo / "transfer" / batch.name
if transfer.exists():
    raise SystemExit(f"Transfer already exists; refusing overwrite: {transfer}")
transfer.parent.mkdir(parents=True, exist_ok=True)
missing = []
collected = []
runs = []

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


with tempfile.TemporaryDirectory(prefix=f".{batch.name}-", dir=transfer.parent) as temporary:
    staging = Path(temporary) / "payload"
    staging.mkdir()

    def copy_artifact(source: Path, relative: str) -> None:
        if not source.is_file():
            missing.append(str(source.relative_to(repo)))
            return
        if source.is_symlink() or not source.resolve().is_relative_to(repo):
            raise RuntimeError(f"Refusing symlink or external artifact: {source}")
        before = source.stat()
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino
        ) or digest(source) != digest(destination):
            raise RuntimeError(f"Artifact changed while collecting: {source}")
        collected.append(relative)

    for name in (
        "parent-commit.txt", "async-preparation.txt", "image-id.txt", "launcher-sha256.txt", "source-manifest.json",
        "cpu-preflight.log", "gpuq-run-help.txt", "job-native.txt", "job-kernel.txt",
    ):
        copy_artifact(batch / name, name)
    for mode, job in selected.items():
        for name in (
            f"serve_{mode}.log", "server-info.json", "assigned-gpus.txt",
            "replay-exit-code.txt", "source-manifest.json", "gc-freeze.log",
            "host-memory-before.txt", "host-memory-after.txt",
        ):
            copy_artifact(batch / mode / name, f"{mode}/boot/{name}")
        for name in ("stdout.log", "stderr.log", "stdout", "stderr", "run.log", "job.json"):
            source = batch / f"gpuq-{mode}" / name
            if source.exists():
                copy_artifact(source, f"{mode}/gpuq/{name}")

        tag = f"{mode}-frontend-first-output-c40-hicache-mf088-{batch.name}"
        root = repo / "results" / "lswb-replay" / tag
        found = sorted(path for path in root.glob("20*") if path.is_dir())
        if len(found) > 1:
            raise RuntimeError(f"Ambiguous diagnostic results under {root}: {found}")
        run = found[0] if found else None
        record = {"mode": mode, "job_id": job["id"], "status": job.get("status", job.get("state")),
                  "exit_code": job.get("exit_code"), "original_run": None, "transfer_path": None}
        if run is None:
            missing.append(str(root.relative_to(repo)) + "/<run-directory>")
        else:
            relative = f"{mode}/{run.name}"
            record.update(original_run=str(run.relative_to(repo)), transfer_path=relative)
            # Explicit metadata whitelist: no requests.jsonl, raw prompts/tools,
            # model weights, compiled artifacts, or unfiltered request dumps.
            for name in (
                "client/REPORT.md", "client/summary.json", "client/manifest.json",
                "client/dataset_manifest.json", "client/done.json", "client/responses.jsonl",
                "client.log", "cache-flush.log", "gpu-samples.csv", "server-info.json",
                "server.log", "replay-window.json", "request-timings.jsonl",
                "request-timing-validation.json", "collection-config.json",
                "first-output-pipeline.json",
            ):
                copy_artifact(run / name, f"{relative}/{name}")
        runs.append(record)

    (staging / "jobs.json").write_text(json.dumps({"jobs": list(selected.values())}, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "batch": batch.name,
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "description": "Native/Packed Kernel-256 C40 HiCache first-output diagnostic, MF0.88, n=1 each",
        "async_preparation": int((batch / "async-preparation.txt").read_text().strip()) if (batch / "async-preparation.txt").is_file() else 0,
        "classification": "Transferred diagnostic evidence; failures preserved, qualification pending",
        "expected_jobs": 2, "terminal_jobs": len(selected),
        "collected_runs": sum(record["original_run"] is not None for record in runs),
        "runs": runs, "copied_artifacts": collected, "missing_artifacts": missing,
        "collection_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=30,
        ).strip(),
    }
    (staging / "transfer-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    (staging / "SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.relative_to(staging)}\n" for path in files
    ))
    if transfer.exists():
        raise RuntimeError(f"Transfer destination appeared during collection: {transfer}")
    staging.rename(transfer)
print(f"Collected {len(files)} checksummed files: {transfer}")
print(f"Missing artifacts: {len(missing)} (retained in transfer-manifest.json; not a success claim)")
print("Collection only: inspect the transfer before explicitly staging, committing and pushing it.")
PY_COLLECT
    ;;
  run)
    MODE=${2:?missing mode} ROOT=${3:?missing batch directory}
    export PORT=${4:?missing port} MASTER_PORT=${5:?missing master port}
    case "$MODE" in native|kernel) ;; *) exit 2 ;; esac
    case "$ROOT" in "$REPO"/logs/frontend-first-output-*) ;; *) echo "Unexpected batch directory" >&2; exit 2 ;; esac
    [ "$(git rev-parse HEAD)" = "$(cat "$ROOT/parent-commit.txt")" ] || {
      echo "Source changed after submission; refusing diagnostic" >&2; exit 1;
    }
    [ -z "$(git status --porcelain --untracked-files=no)" ] || {
      echo "Tracked parent source changed after submission; refusing diagnostic" >&2; exit 1;
    }
    sha256sum -c "$ROOT/launcher-sha256.txt"
    [ "$(docker inspect "$CONTAINER" --format '{{.Image}}')" = "$(cat "$ROOT/image-id.txt")" ] || {
      echo "Container image changed after submission" >&2; exit 1;
    }
    [ "$DSV4_ASYNC_PREPARATION" = "$(cat "$ROOT/async-preparation.txt" 2>/dev/null || echo 0)" ] || {
      echo "Preparation mode changed after submission" >&2; exit 1;
    }
    check_sources
    : "${CUDA_VISIBLE_DEVICES:?GPUQ must assign the devices}"
    IFS=, read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
    [ "${#assigned[@]}" -eq 4 ] || { echo "Diagnostic needs exactly four assigned GPUs" >&2; exit 1; }
    export GPUS="$CUDA_VISIBLE_DEVICES" TP=4 MEM_FRAC=0.88 SERVER_SEED=20260919
    export CTX_LEN=1048576 MAX_RUN=256 CHUNK=8192 HICACHE_RATIO=2
    export LOG_HOST="$ROOT/$MODE" REQUEST_METRICS_DIR="$ROOT/$MODE/request-metrics"
    export SERVER_PID_FILE_CT="/tmp/remnant-first-output-$MODE-$(basename "$ROOT").pid"
    export DECODE_CFG="$DECODE_CFG_EXT"
    export FRONTEND_DIAGNOSTICS=1 REPLAY_SCHEDULER=fixed-active-user-pool
    export REPLAY_RUNNER="$REPO/scripts/local/business_replay/runner.py"
    export REPLAY_ADAPTER="$REPO/scripts/local/business_replay/adapters/openai_sse.py"
    unset REPLAY_SESSION_POPULATION REPLAY_DRAIN_TIMEOUT REPLAY_TTFT_SLO_MS
    unset C4_LAYER_KEEP_COUNTS NSYS_OUTPUT REMNANT_CAPTURE_DIR
    export C4_KEEP_COUNT=256
    unset STORE_IMPLEMENTATION
    if [ "$MODE" = kernel ]; then export STORE_IMPLEMENTATION=kernel; fi
    mkdir -p "$LOG_HOST"
    printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$LOG_HOST/assigned-gpus.txt"
    if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "Diagnostic port $PORT is already serving; refusing to replace it" >&2; exit 1;
    fi
    cleanup() {
      local status=$?
      trap - EXIT INT TERM
      if ! bash scripts/local/serve.sh "$MODE" stop; then status=1; fi
      exit "$status"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    bash scripts/local/serve.sh "$MODE"
    if [ "$DSV4_ASYNC_PREPARATION" = 1 ]; then
      grep -q 'DSV4 asynchronous input preparation enabled' "$LOG_HOST/serve_$MODE.log" || {
        echo "Candidate input preparation activation is missing from the server log" >&2; exit 1;
      }
    fi
    curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" > "$LOG_HOST/server-info.json"
    python3 - "$LOG_HOST/server-info.json" "$MODE" <<'PY_CONFIG'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
mode = sys.argv[2]
expected = {
    "dsv4_c4_cache_format": "native" if mode == "native" else "remnant",
    "tp_size": 4, "mem_fraction_static": 0.88, "random_seed": 20260919,
    "enable_hierarchical_cache": True, "hicache_ratio": 2.0,
    "hicache_write_policy": "write_through", "hicache_io_backend": "kernel",
    "hicache_mem_layout": "layer_first", "hicache_storage_backend": None,
    "chunked_prefill_size": 8192, "dsv4_c4_cache_keep_count": 256,
    "dsv4_c4_cache_layer_keep_counts": None, "tokenizer_worker_num": 1,
    "detokenizer_worker_num": 1,
}
if mode == "kernel":
    expected.update(dsv4_c4_cache_implementation="kernel", dsv4_c4_cache_store_implementation="kernel")
for name, value in expected.items():
    if name not in info or info[name] != value:
        raise SystemExit(f"Diagnostic {mode}: expected {name}={value!r}, got {info.get(name)!r}")
print(f"Diagnostic server matrix PASS: {mode}")
PY_CONFIG
    curl --fail-with-body -sS --max-time 30 -X POST "http://127.0.0.1:$PORT/freeze_gc"
    TAG="$MODE-frontend-first-output-c40-hicache-mf088-$(basename "$ROOT")"
    # Preserve failure diagnostics even if the replay response-validity gate fails.
    set +e
    timeout --foreground --kill-after=60 1500 \
      bash scripts/local/bench-lswb.sh "$TAG" "$PORT" 40 900
    replay_status=$?
    set -e
    echo "$replay_status" > "$LOG_HOST/replay-exit-code.txt"
    echo "Replay results: $RESULTS_HOST/lswb-replay/$TAG"
    python3 - "$RESULTS_HOST/lswb-replay/$TAG" "$REPO" <<'PY_ANALYZE'
import importlib.util
import json
import sys
from pathlib import Path

tag, repo = map(Path, sys.argv[1:])
runs = list(tag.glob("*/replay-window.json"))
if len(runs) != 1:
    raise SystemExit(f"Expected one diagnostic replay under {tag}, got {len(runs)}")
run = runs[0].parent
path = repo / "benchmarks/lswb-replay/c24-packed256-store-decode-ablation-n1-20261014/analyze.py"
spec = importlib.util.spec_from_file_location("replay_analysis", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.first_output_pipeline_diagnostics(
    module.lines(run / "client/responses.jsonl"), module.lines(run / "request-timings.jsonl"),
    module.read(run / "replay-window.json")["started_at"],
)
result["post_180s_sample_gate"] = result["cohorts"]["after_180s"]["count"] >= 200
result["response_summary_valid"] = module.read(run / "client/summary.json")["valid"]
(run / "first-output-pipeline.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not result["valid"] or not result["post_180s_sample_gate"]:
    raise SystemExit("Diagnostic coverage failed; evidence retained, do not infer missing stages")
PY_ANALYZE
    exit "$replay_status"
    ;;
  *) echo "usage: bash $SCRIPT <preflight|submit> [dataset manifest], collect batch-directory, or run mode batch port master" >&2; exit 2 ;;
esac
