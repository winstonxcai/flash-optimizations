#!/usr/bin/env bash
# =====================================================================
# container.sh -- bring up the production-fork container and verify its tree.
# Author: Winston Cai.
#
#   container.sh              ensure image + container exist, then prep
#   container.sh prep         prep only (container must already exist)
#   container.sh rebuild      rebuild the image and recreate the container
#   container.sh recreate     docker rm -f the container, recreate, prep
#
# The Dockerfile copies the checked-out production fork into
# /sgl-workspace/sglang-remnant. Native, Triton and Kernel serving use that tree;
# native is selected by the default cache-format argument.
#
# env.sh defaults to the production runtime trio (GPUS 0-3 / PORT 30212 /
# MASTER 29638). Choose a distinct port, master port, and GPU set when another
# server is already running on the node.
# =====================================================================
set -u
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$DIR/env.sh"

ACTION=${1:-up}
DOCKERFILE="$HOST_REPO/$REMNANT_DOCKERFILE"

die () { echo "FATAL: $*" >&2; exit 1; }

has_image () { docker image inspect "$IMAGE" >/dev/null 2>&1; }
has_container () { docker container inspect "$CONTAINER" >/dev/null 2>&1; }
container_up () { docker ps --filter "name=^/${CONTAINER}$" --format '{{.Names}}' | grep -qx "$CONTAINER"; }

wait_exec_ready () {  # poll until `docker exec` works inside the container
  local i
  for i in $(seq 1 40); do
    docker exec "$CONTAINER" true 2>/dev/null && return 0
    sleep 3
  done
  return 1
}

build_image () {
  echo "== build $IMAGE from $DOCKERFILE"
  # GPU nodes may have no docker0 bridge; build workers use the same host
  # network as the serving container rather than Docker's default bridge.
  docker build --network host -t "$IMAGE" -f "$DOCKERFILE" \
    --build-arg "REMNANT_INTEGRATION=$REMNANT_INTEGRATION" \
    --build-arg "REMNANT_BASE_IMAGE=$REMNANT_BASE_IMAGE" \
    --build-arg "REMNANT_SGLANG_SOURCE=$SGLANG_SOURCE_RELATIVE" \
    --build-arg "REMNANT_FLASHMLA_SOURCE=$FLASHMLA_SOURCE_RELATIVE" \
    --build-arg "REMNANT_SGLANG_SHA=$(git -C "$HOST_REPO/$SGLANG_SOURCE_RELATIVE" rev-parse HEAD)" \
    --build-arg "REMNANT_FLASHMLA_SHA=$(git -C "$HOST_REPO/$FLASHMLA_SOURCE_RELATIVE" rev-parse HEAD)" \
    "$HOST_REPO" || die "docker build failed"
}

create_container () {
  echo "== docker run --name $CONTAINER ($IMAGE)"
  docker run -d --name "$CONTAINER" --network host --gpus all --shm-size 120g \
    -v /:/mnt/host_root -v /home:/home \
    "$IMAGE" bash -c 'sleep infinity' || die "docker run failed"
  wait_exec_ready || die "container not exec-ready after launch"
}

prep () {
  has_container || die "container $CONTAINER does not exist; run: $0 (no args)"
  check_container_sources || die "container does not match the selected source pair"
  echo "== verify production SGLang fork at $SGLANG_ROOT_CT"
  ct_script "$SGLANG_ROOT_CT/python/sglang/srt" "$REMNANT_INTEGRATION" <<'BASH' \
    || die "production SGLang fork is not present or missing C4 cache selection flags"
set -e
srt=$1
integration=$2
server_args="$srt/server_args.py"
fields=$server_args
if [ "$integration" = v0.5.22 ]; then
  fields="$srt/arg_groups/fields/model.py"
fi
test -f "$server_args"
test -f "$fields"
grep -q _configure_dsv4_c4_cache_format "$server_args"
grep -q dsv4_c4_cache_implementation "$fields"
BASH
}

# ------------------------------ dispatch ------------------------------
check_sources || exit 1
case "$ACTION" in
  prep) prep ;;
  recreate)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    has_image || build_image
    create_container
    prep
    ;;
  rebuild)
    build_image
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    create_container
    prep
    ;;
  up)
    has_image || build_image
    if has_container; then
      echo "== container $CONTAINER exists (image $IMAGE); skipping create"
      container_up || { echo "   container stopped; starting"; docker start "$CONTAINER" >/dev/null || die "docker start failed"; }
    else
      create_container
    fi
    prep
    ;;
  *) echo "usage: $0 [prep|recreate|up|rebuild]" >&2; exit 2 ;;
esac
echo "== production image ready. serve with:  bash scripts/local/serve.sh <native|triton|kernel>"
