#!/usr/bin/env bash
# Random-text benchmark client for an already-running SGLang server.
# Author: Winston Cai.
set -euo pipefail

DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
usage () {
  cat <<'EOF'
usage: bench-serving.sh <tag> [port] [input_tokens] [output_tokens] [concurrency|auto]

Defaults: port from env.sh, 32768 input, 2048 output, concurrency 8.
Start the server separately with serve.sh. This client never starts/stops it.
Use the same concurrency for fair comparisons; auto uses the running pool's
capacity. BENCH_EXECUTION=docker (VDI, default) or direct (inside the image).
BENCH_TIMEOUT_S bounds each client wave (default 3600 seconds).
BENCH_DATASET_PATH selects a local ShareGPT JSON file for random-text sampling;
in Docker mode, use an absolute host path under HOST_REPO. Otherwise SGLang
uses its default Hugging Face download/cache behavior.
EOF
}
die () { echo "bench-serving.sh: $*" >&2; exit 1; }
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then usage; exit 0; fi
. "$DIR/env.sh"

TAG=${1:-} PORT=${2:-$PORT} INPUT=${3:-32768} OUTPUT=${4:-2048} C=${5:-8}
(( $# >= 1 && $# <= 5 )) || { usage >&2; exit 2; }
[[ "$TAG" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || die "tag must be a simple directory name"
for value in "$PORT" "$INPUT" "$OUTPUT" "${BENCH_TIMEOUT_S:=3600}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "port, lengths and timeout must be positive integers"
done
[[ "$C" = auto || "$C" =~ ^[1-9][0-9]*$ ]] || die "concurrency must be a positive integer or auto"
(( PORT <= 65535 )) || die "port must be at most 65535"
python=${PYTHON:-python3}
execution=${BENCH_EXECUTION:-docker}
case "$execution" in
  docker|direct) ;;
  *) die "BENCH_EXECUTION must be docker or direct" ;;
esac
for tool in "$python" curl timeout; do
  command -v "$tool" >/dev/null 2>&1 || die "missing required tool: $tool"
done
dataset_path=${BENCH_DATASET_PATH:-}
if [ -n "$dataset_path" ]; then
  [[ "$dataset_path" = /* ]] || die "BENCH_DATASET_PATH must be an absolute host path"
  [ -r "$dataset_path" ] && [ -f "$dataset_path" ] || die "cannot read dataset: $dataset_path"
  dataset_client_path=$dataset_path
  if [ "$execution" = docker ]; then
    case "$dataset_path" in
      "$HOST_REPO"/*) dataset_client_path=$(to_ct "$dataset_path") ;;
      *) die "Docker dataset must be under HOST_REPO: $dataset_path" ;;
    esac
  fi
fi
health || die "no server on port $PORT; start it with serve.sh first"

base="$RESULTS_HOST/serving/$TAG"
mkdir -p "$base"
RUN_ROOT=$(mktemp -d "$base/$(ts)-XXXXXX")
echo "Results: $RUN_ROOT"
curl -fsS --max-time 30 "http://127.0.0.1:$PORT/server_info" > "$RUN_ROOT/server-info.json"

# Resolve against the actual server, not a second copy of its launch settings.
"$python" - "$RUN_ROOT" "$C" "$INPUT" "$OUTPUT" "$SEED" "$0" "$dataset_path" <<'PY_PREFLIGHT'
import hashlib
import json
import sys
from pathlib import Path

root, requested, input_len, output_len, seed, script, dataset_path = sys.argv[1:]
root = Path(root)
info = json.loads((root / "server-info.json").read_text())
input_len, output_len, seed = int(input_len), int(output_len), int(seed)
concurrency = (int(info["max_total_num_tokens"]) // (input_len + output_len)
               if requested == "auto" else int(requested))
if concurrency < 1:
    raise SystemExit("allocator pool cannot hold one complete request")
if input_len + output_len > info["context_length"]:
    raise SystemExit("input + output exceeds the running server's context limit")
decode = info.get("cuda_graph_config", {}).get("decode", {})
sizes = decode.get("bs", [])
covered = (concurrency in sizes if info.get("disable_cuda_graph_padding")
           else any(size >= concurrency for size in sizes))
if decode.get("backend") != "full" or not covered:
    raise SystemExit(f"concurrency {concurrency} is not covered by full decode graphs: {decode}")
if concurrency > info["max_running_requests"]:
    raise SystemExit("concurrency exceeds the running server's request limit")
config = {
    "schema_version": 1, "input_tokens": input_len, "output_tokens": output_len,
    "concurrency": concurrency, "requested_concurrency": requested, "seed": seed,
    "warmup_requests": concurrency, "measured_requests": 3 * concurrency,
    "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
}
if dataset_path:
    path = Path(dataset_path)
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot load ShareGPT dataset {path}: {exc}") from exc
    if not isinstance(document, list) or not document:
        raise SystemExit(f"ShareGPT dataset must be a nonempty JSON list: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    config["dataset"] = {"path": str(path), "sha256": digest.hexdigest()}
(root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
(root / "client-args.txt").write_text("\n".join([
    str(concurrency), info["model_path"], info.get("tokenizer_path") or info["model_path"],
]) + "\n")
PY_PREFLIGHT
# Read separate lines to preserve paths containing spaces.
{ IFS= read -r C; IFS= read -r model; IFS= read -r tokenizer; } < "$RUN_ROOT/client-args.txt"
rm "$RUN_ROOT/client-args.txt"

runtime=(env)
client_python=$python
if [ "$execution" = docker ]; then
  case "$RUN_ROOT" in "$HOST_REPO"/*) ;; *) die "Docker results must be under HOST_REPO" ;; esac
  timeout --signal=TERM --kill-after=5s 30s docker inspect "$CONTAINER" --format '{{.Image}}' \
    > "$RUN_ROOT/image-at-collection.txt"
  runtime=(docker exec -i -e "PYTHONPATH=/opt/sglang-runtime-fixes:$SGLANG_PY:$REPO_CT"
    -w "$SGLANG_PY" "$CONTAINER")
  client_python=python3
fi
timeout --signal=TERM --kill-after=5s 60s "${runtime[@]}" "$client_python" - \
  > "$RUN_ROOT/client-source.json" <<'PY_SOURCE'
import hashlib
import importlib.machinery
import json
import os
import sys
from pathlib import Path

spec = importlib.machinery.PathFinder.find_spec("sglang", sys.path)
if spec is None or spec.origin is None:
    raise SystemExit("SGLang client package is missing from the selected environment")
root = Path(spec.origin).parent
digest = hashlib.sha256()
for path in sorted(root.rglob("*.py")):
    digest.update(str(path.relative_to(root)).encode() + b"\0")
    digest.update(path.read_bytes())
print(json.dumps({
    "python": sys.executable, "sglang_root": str(root),
    "sglang_python_tree_sha256": digest.hexdigest(),
    "submitted_source_revisions": {
        name: os.environ.get(f"REMNANT_{name.upper()}_SHA")
        for name in ("parent", "sglang", "flashmla")
    },
}, indent=2))
PY_SOURCE

bench_wave () {
  local name=$1 count=$2 destination="$RUN_ROOT/$1.jsonl"
  [ "$execution" != docker ] || destination=$(to_ct "$destination")
  # A remote timeout also bounds Docker's client if its host connection is lost.
  local command=(timeout --signal=TERM --kill-after=5s "${BENCH_TIMEOUT_S}s"
    "$client_python" -m sglang.bench_serving
    --backend sglang --host 127.0.0.1 --port "$PORT"
    --model "$model" --tokenizer "$tokenizer" --dataset-name random
    --random-input-len "$INPUT" --random-output-len "$OUTPUT"
    --random-range-ratio 1.0 --num-prompts "$count" --max-concurrency "$C"
    --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt
    --output-file "$destination" --output-details --seed "$SEED")
  if [ -n "$dataset_path" ]; then
    command+=(--dataset-path "$dataset_client_path")
  fi
  timeout --signal=TERM --kill-after=5s "$((BENCH_TIMEOUT_S + 10))s" \
    "${runtime[@]}" "${command[@]}" > "$RUN_ROOT/$name.log" 2>&1 \
    || die "$name failed; see $RUN_ROOT/$name.log"
  "$python" - "$RUN_ROOT/$name.jsonl" "$count" "$OUTPUT" <<'PY_VALIDATE'
import json
import sys

path, expected, output = sys.argv[1:]
with open(path) as stream:
    records = [json.loads(line) for line in stream if line.strip()]
if len(records) != 1:
    raise SystemExit(f"{path}: expected exactly one benchmark result")
record = records[0]
lengths = record.get("output_lens")
if (record.get("completed") != int(expected) or not isinstance(lengths, list)
        or len(lengths) != int(expected) or any(n != int(output) for n in lengths)
        or any(record.get("errors") or [])):
    raise SystemExit(f"{path}: incomplete requests, output lengths, or request errors")
print(f"{path}: {expected} requests validated")
PY_VALIDATE
}

bench_wave warmup "$C"
health || die "server unavailable after warm-up"
bench_wave measured "$((3 * C))"
echo "Completed: $RUN_ROOT/measured.jsonl (server left running)"
