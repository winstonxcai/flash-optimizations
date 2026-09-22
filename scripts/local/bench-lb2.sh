#!/usr/bin/env bash
# =====================================================================
# bench-lb2.sh <tag> [port] [out_dir]
#   Official LongBench v2 evaluation against an ALREADY-RUNNING SGLang server
#   (bring it up with serve.sh native|packed). The official lm-eval task is a
#   multiple-choice/log-likelihood task, so use SGLang's /v1/completions API
#   through lm-eval's local-completions backend.
#
#   tag     = results label (e.g. native-0731, packed-0731)
#   port    = server port, default $PORT (from env.sh)
#   out_dir = lm-eval output directory (default
#             <RESULTS_HOST>/lb2/<tag>/<ts>)
#
# Resumability is provided by lm-eval's SQLite request cache. Re-run with the
# same output directory and --use_cache prefix to reuse completed requests.
# =====================================================================
set -u
. "$(dirname "$0")/env.sh"

TAG=${1:-} PORT=${2:-$PORT} OUT=${3:-}
[ -n "$TAG" ] || { echo "usage: $0 <tag> [port] [out_dir]"; exit 1; }

LM_EVAL_BIN=${LM_EVAL_BIN:-lm-eval}
command -v "$LM_EVAL_BIN" >/dev/null 2>&1 \
  || { echo "FATAL: '$LM_EVAL_BIN' not found; install lm-eval[longbench]" >&2; exit 1; }

curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "FATAL: no server on 127.0.0.1:$PORT (run serve.sh first)"; exit 1; }

if [ -z "$OUT" ]; then
  OUT="$RESULTS_HOST/lb2/$TAG/$(ts)"
fi
mkdir -p "$OUT"

BASE_URL="http://127.0.0.1:$PORT/v1/completions"
probe_status=$(curl -sS -m 5 -o /dev/null -w '%{http_code}' \
  -X POST "$BASE_URL" \
  -H 'Content-Type: application/json' \
  --data "{\"model\":\"$MODEL_NAME\",\"prompt\":\"\",\"max_tokens\":0}" \
  2>/dev/null || true)
case "$probe_status" in
  200|400|422) ;;
  *)
    echo "FATAL: SGLang completions endpoint unavailable at $BASE_URL (HTTP ${probe_status:-no-response})" >&2
    exit 1
    ;;
esac

CACHE_PREFIX="$OUT/lm-eval-cache"

echo "== LongBench v2 official: tag=$TAG server=127.0.0.1:$PORT -> $OUT =="
"$LM_EVAL_BIN" run \
  --model local-completions \
  --model_args "model=$MODEL_NAME,base_url=$BASE_URL,tokenizer_backend=remote,num_concurrent=1,max_retries=3" \
  --tasks longbench2 \
  --batch_size 1 \
  --seed 0 \
  --output_path "$OUT" \
  --log_samples \
  --use_cache "$CACHE_PREFIX"
RC=$?
echo "[lb2] official lm-eval artifacts at $OUT"
exit "$RC"
