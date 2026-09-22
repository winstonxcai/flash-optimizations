#!/usr/bin/env bash
# =====================================================================
# container.sh -- bring up the production-fork container and verify its tree.
#
#   container.sh              ensure image + container exist, then prep
#   container.sh prep         prep only (container must already exist)
#   container.sh rebuild      rebuild the image and recreate the container
#   container.sh recreate     docker rm -f the container, recreate, prep
#
# The Dockerfile clones the reviewed production fork into
# /sgl-workspace/sglang-remnant. Both native and packed serving use that tree;
# native is selected by the default cache-format argument.
#
# env.sh defaults to the production runtime trio (GPUS 0-3 / PORT 30212 /
# MASTER 29638). Choose a distinct port, master port, and GPU set when another
# server is already running on the node.
# =====================================================================
set -u
export CONTAINER=${CONTAINER:-remnant}
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$DIR/env.sh"

ACTION=${1:-up}
IMAGE=${IMAGE:-remnant:v0.5.18}
DOCKERFILE="$HOST_REPO/Dockerfile"

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
  docker build -t "$IMAGE" -f "$DOCKERFILE" "$HOST_REPO" || die "docker build failed"
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
  echo "== verify production SGLang fork at $SGLANG_ROOT_CT"
  ct "test -f $SGLANG_ROOT_CT/python/sglang/srt/server_args.py" \
    || die "production SGLang fork is not present in the image"
  ct "grep -q dsv4-c4-cache-format $SGLANG_ROOT_CT/python/sglang/srt/server_args.py" \
    || die "SGLang fork is missing --dsv4-c4-cache-format"
}

# ------------------------------ dispatch ------------------------------
case "$ACTION" in
  prep) prep ;;
  recreate)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    has_image || build_image
    create_container
    prep
    ;;
  rebuild)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    build_image
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
echo "== production image ready. serve with:  bash scripts/local/serve.sh <native|packed>"
