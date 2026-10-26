#!/usr/bin/env bash
# Resolve model geometry once and transport named settings into the container.
# Consumers: env.sh, run-server.sh, and serve.sh. Author: Winston Cai.

resolve_launch_config () {
  MODEL_FAMILY=${MODEL_FAMILY:-dsv4}
  case "$MODEL_FAMILY" in
    dsv4) TP=${TP:-4}; MODEL_NAME=${MODEL_NAME:-deepseek-v4-flash} ;;
    glm53) TP=${TP:-8}; MODEL_NAME=${MODEL_NAME:-glm53-flash}; EP=${EP:-$TP} ;;
    *) echo "MODEL_FAMILY must be dsv4 or glm53" >&2; return 2 ;;
  esac
  [[ "$TP" =~ ^[1-9][0-9]*$ ]] || { echo "TP must be a positive integer" >&2; return 2; }
  if [ "$MODEL_FAMILY" = glm53 ]; then
    case "$TP" in 1|2|4|8) ;; *) echo "GLM requires TP1/2/4/8" >&2; return 2 ;; esac
    [[ "$EP" =~ ^[1-9][0-9]*$ ]] && (( EP <= TP && TP % EP == 0 )) || {
      echo "GLM EP must be a positive divisor of TP (TP=$TP EP=$EP)" >&2; return 2;
    }
  fi
  # A caller's assignment (including scheduler GPU UUIDs) takes precedence.
  GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-}}
  if [ -z "$GPUS" ]; then
    GPUS=0
    local index
    for ((index=1; index<TP; index++)); do GPUS+=",$index"; done
  fi
  local -a devices
  IFS=, read -r -a devices <<< "$GPUS"
  if [ "$MODEL_FAMILY" = glm53 ] && {
    (( ${#devices[@]} != TP )) || [[ "$GPUS" == ,* || "$GPUS" == *, || "$GPUS" == *,,* ]];
  }; then
    echo "GLM needs TP=$TP visible GPUs; got GPUS=$GPUS" >&2; return 2
  fi
}

# Keep model and serving settings named across docker exec. Values are separate
# argv elements; spaces and JSON are never interpolated into executable text.
ct_launch_script () {
  local name
  local -a environment=()
  for name in MODEL_FAMILY MODEL_NAME TP EP GPUS MASTER_PORT MEM_FRAC CTX_LEN \
    MAX_RUN CHUNK DECODE_CFG SERVER_SEED REQUEST_METRICS_DIR STORE_IMPLEMENTATION \
    C4_KEEP_COUNT C4_LAYER_KEEP_COUNTS HICACHE_RATIO NSYS_GRAPH_TRACE \
    FRONTEND_DIAGNOSTICS DSV4_ASYNC_PREPARATION DSV4_PREPARATION_WORKERS; do
    environment+=(-e "$name=${!name-}")
  done
  docker exec -i "${environment[@]}" "$CONTAINER" bash -s -- "$@"
}
