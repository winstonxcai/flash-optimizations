#!/usr/bin/env bash
# Host orchestrator for the fp4-native serving-capacity sweep: TopMag50 NATIVE
# 584-byte vs PACKED 328-byte C4, DeepSeek-V4-Flash-0731, TP4, on GPUs 4-7.
#
# Points = 32k/64k/128k/256k contexts, exactly 2048 output tokens, at (a) fair
# concurrency (both modes at the native-584 allocator ceiling) and (b) per-mode
# maximum (packed at its own ceiling) -- report.md serving methodology.
#
#   native:  C = 107, 55, 28, 14  (4 points)
#   packed:  fair C = 107,55,28,14 AND max C = 129,66,33,17  (8 points)
#
# Each point = one warm-up wave (NUM_PROMPTS=C) then three measured waves
# (NUM_PROMPTS=3C), official sglang.bench_serving, --warmup-requests 0.
#
# Usage:
#   run_serve_sweep_fp4.sh <packed|native>        # boot if not already healthy, then run points
#   run_serve_sweep_fp4.sh <packed|native> --no-boot   # assume server already up on the port
#
# Legs boot via launch_inner_serve_sweep_fp4_tp4.sh inside ruler-eval (extended
# decode CUDA graphs through max_bs 136). Results under
# mustafar/results/serve-sweep-fp4/<leg>/<ts>/ ; server logs mustafar/logs/serve_sweep_<leg>.log.
set -u
LEG=${1:-}
BOOT=${2:-boot}          # boot | --no-boot
[ -n "$LEG" ] || { echo "usage: $0 <packed|native> [--no-boot]" >&2; exit 2; }

HOST_REPO=/home/jovyan/winstonxcai/flash-optimizations
CONT_REPO=/mnt/host_root/home/jovyan/winstonxcai/flash-optimizations
CONTAINER=ruler-eval
MODEL=/mnt/host_root/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731
INNER_LAUNCH=$CONT_REPO/mustafar/scripts/local/launch_inner_serve_sweep_fp4_tp4.sh
RESULT_BASE=$HOST_REPO/mustafar/results/serve-sweep-fp4
LOG_BASE=$HOST_REPO/mustafar/logs
CT_LOG=$LOG_BASE/serve_sweep_${LEG}.log
TS=$(date +%Y%m%d_%H%M%S)
OUTLEN=2048
SEED=42

# leg -> port / master / packed
case "$LEG" in
  packed) PORT=30211; MPORT=29626; PACKED=1 ;;
  native) PORT=30212; MPORT=29628; PACKED=0 ;;
  *) echo "unknown leg '$LEG' (packed|native)" >&2; exit 2 ;;
esac

# allocator ceilings (resident_requests_with_output = floor(pool/(ctx+2048)))
declare -A NATC=( [32768]=107 [65536]=55 [131072]=28 [262144]=14 )
declare -A PCKC=( [32768]=129 [65536]=66 [131072]=33 [262144]=17 )
CTXS=(32768 65536 131072 262144)

RUN_ROOT="$RESULT_BASE/$LEG/$TS"
mkdir -p "$RUN_ROOT"

die () { echo "FATAL: $*" >&2; exit 1; }

wait_health () {   # $1 port, $2 poll cap in 5s steps
  local port=$1 n=${2:-150} i
  for i in $(seq 1 "$n"); do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "health OK after ~$((i*5))s"; return 0; }
    sleep 5
  done
  echo "health TIMEOUT after ~$((n*5))s on $port" >&2
  return 1
}

boot_leg () {
  [ "$BOOT" = "--no-boot" ] && return 0
  # only boot if not already healthy with graphs captured
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
     && grep -q "Capture target decode CUDA graph end" "$CT_LOG" 2>/dev/null; then
    echo "[$LEG] healthy server already up on $PORT -- reusing"
    return 0
  fi
  echo "== [$LEG] booting ($(date -u +%H:%M:%S)Z) on GPUs 4-7 port $PORT master $MPORT =="
  : > "$CT_LOG"
  docker exec "$CONTAINER" bash -c "LEG=$LEG bash $INNER_LAUNCH" || die "inner launcher failed ($LEG)"
  wait_health "$PORT" 180 || die "boot failed for $LEG (log: $CT_LOG)"
  # pool + graph-capture markers
  sleep 3
  if ! grep -q "max_total_num_tokens=" "$CT_LOG"; then
    echo "---- boot log tail ($LEG) ----"; tail -40 "$CT_LOG"; die "no pool allocation line"
  fi
}

bench_wave () {   # $1 ctx_tokens $2 concurrency $3 num_prompts $4 outfile(container path) $5 log(host path)
  local ctx=$1 C=$2 N=$3 OUT=$4 LOUT=$5
  docker exec "$CONTAINER" bash -c "
    export PYTHONPATH=/sgl-workspace/sglang-lowrank/python
    cd /sgl-workspace/sglang-lowrank/python
    python3 -m sglang.bench_serving \
      --backend sglang --host 127.0.0.1 --port $PORT \
      --model $MODEL --tokenizer $MODEL \
      --dataset-name random --random-input-len $ctx --random-output-len $OUTLEN \
      --random-range-ratio 1.0 --num-prompts $N --max-concurrency $C \
      --request-rate inf --warmup-requests 0 --flush-cache --tokenize-prompt \
      --output-file $OUT --output-details --seed $SEED" > "$LOUT" 2>&1
  return $?
}

run_point () {   # $1 ctx_tokens $2 concurrency $3 label(fair|max)
  local ctx=$1 C=$2 tag=$3
  local N=$((3 * C))
  local run_dir="$RUN_ROOT/ctx${ctx}-${tag}"
  mkdir -p "$run_dir"
  local OUT_CT="/mnt/host_root$run_dir/measured.jsonl"
  local OFFSET
  OFFSET=$(wc -c < "$CT_LOG")
  echo "==== [$LEG] ctx=${ctx} C=${C} ($tag) $(date -u +%H:%M:%S)Z ===="
  # warm-up wave, results discarded
  bench_wave "$ctx" "$C" "$C" "/tmp/sweep-warmup-${LEG}-${ctx}-${C}.jsonl" "$run_dir/warmup.log"
  echo "  warmup rc=$? (${C} prompts)"
  # measured: 3C prompts = three waves at concurrency C
  bench_wave "$ctx" "$C" "$N" "$OUT_CT" "$run_dir/measured.log"
  local RC=$?
  # capture server-log delta for this point (residency + decode-graph replay evidence)
  tail -c +$((OFFSET + 1)) "$CT_LOG" > "$run_dir/server.delta.log" 2>/dev/null || true
  echo "  measured rc=$RC (${N} prompts)"
  [ $RC -eq 0 ] || echo "  WARN: measured bench rc=$RC"
}

boot_leg
# boot-marker excerpt
{
  echo "=== serve-sweep $LEG boot markers ($(date -u)) ==="
  grep -aoE "logical_row_bytes=[0-9]+ layers=[0-9]+|max_total_num_tokens=[0-9]+|available_gpu_mem=[0-9.]+ GB|Capture target decode CUDA graph end[^,]*elapsed=[0-9.]+ s|is fired up and ready" "$CT_LOG" | head -20
} > "$RUN_ROOT/boot-markers.txt"
cat "$RUN_ROOT/boot-markers.txt"

# concurrency values are derived from the pool; verify the boot's actual pool
POOL=$(grep -aoE "max_total_num_tokens=[0-9]+" "$CT_LOG" | head -1 | grep -oE "[0-9]+")
if [ -z "$POOL" ]; then die "could not read max_total_num_tokens from $CT_LOG"; fi
if [ "$LEG" = packed ] && [ "$POOL" -ne 4519168 ]; then
  die "packed pool $POOL != expected 4519168 (C table invalid); check boot"
fi
if [ "$LEG" = native ] && [ "$POOL" -ne 3730944 ]; then
  die "native-584 pool $POOL != expected 3730944 (C table invalid); check boot"
fi
echo "[$LEG] actual pool max_total_num_tokens=$POOL (OK)"

if [ "$LEG" = native ]; then
  for ctx in "${CTXS[@]}"; do
    run_point "$ctx" "${NATC[$ctx]}" "max"      # native max == the fair ceiling
  done
else
  for ctx in "${CTXS[@]}"; do
    run_point "$ctx" "${NATC[$ctx]}" "fair"      # packed at native's ceiling
    run_point "$ctx" "${PCKC[$ctx]}" "max"       # packed at its own ceiling
  done
fi

echo "==== [$LEG] done $(date -u +%H:%M:%S)Z; artifacts at $RUN_ROOT ===="

# free the GPUs for the next leg (port-scoped: never kill other tenants' servers)
ps -eo pid,args | grep "[s]glang.launch_server" | grep -- "--port $PORT" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 4
echo "[$LEG] server on $PORT killed; gpus freed"
exit 0
