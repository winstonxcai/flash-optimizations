#!/usr/bin/env bash
# =====================================================================
# container.sh -- bring up the xkv container and prep its two trees.
#
#   container.sh              ensure image + container exist, then prep
#   container.sh prep         prep only (container must already exist)
#   container.sh patch        prep, then apply the xkv patch to the
#                             lowrank tree (anchors are v0.5.18)
#   container.sh repatch      unpatch, then re-apply; use after the anchor
#                             declarations themselves change
#   container.sh recreate     docker rm -f the container, recreate, prep
#   container.sh drift        read-only anchor census only (no writes at all;
#                             works against remnant too, via CONTAINER=remnant)
#
# xkv is the two-tree v0.5.18 container: pristine /sgl-workspace/sglang (ships
# in the base image, byte-identical stock) plus a runtime clone
# /sgl-workspace/sglang-lowrank that carries the xkv patch. This script is
# idempotent and only *needs* to run once per container creation (or again after
# `container.sh recreate` / docker rm xkv).
#
# It is deliberately a different container from `remnant` (mustafar): different
# name, image, port and master port, so a running mustafar server can never be
# disturbed by, or serve, an xkv leg.
#
# prep clones the lowrank tree and writes the read-only anchor drift report that
# scopes any future re-base:
#   <RESULTS_HOST>/xkv/drift-v0.5.18.md
#   (0 drifted = anchors intact against this tree)
#
# patch is the runtime step that makes `xkv` servable after a fresh recreate: it
# runs `xkv patch && xkv verify` against the lowrank clone. Patching is never
# baked into the Dockerfile.
# =====================================================================
set -u
export CONTAINER=${CONTAINER:-xkv}
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$DIR/env.sh"

ACTION=${1:-up}
IMAGE=${IMAGE:-xkv:v0.5.18}
DOCKERFILE="$HOST_REPO/xkv/docker/local.Dockerfile"
DRIFT_DIR="$RESULTS_HOST/xkv"
DRIFT="$DRIFT_DIR/drift-v0.5.18.md"
LOWRANK=/sgl-workspace/sglang-lowrank

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

bootstrap_container () {
  # A bare `lmsysorg/sglang:v0.5.18-cu130` container is short-lived unless given
  # something to do; hold it open with sleep infinity.
  echo "== docker run --name $CONTAINER ($IMAGE)"
  docker run -d --name "$CONTAINER" --network host --gpus all --shm-size 120g \
    -v /:/mnt/host_root -v /home:/home \
    "$IMAGE" bash -c 'sleep infinity' || die "docker run failed"
  wait_exec_ready || die "container not exec-ready after launch"
}

# ------------------------- prep (two trees) --------------------------
clone_lowrank () {
  if ! ct "test -d $LOWRANK/.git"; then
    echo "== clone pristine -> $LOWRANK at v0.5.18"
    ct "git clone /sgl-workspace/sglang $LOWRANK \
        && git -C $LOWRANK checkout v0.5.18" \
      || die "git clone of pristine tree failed"
  else
    echo "== $LOWRANK already present"
  fi
  ct "git -C $LOWRANK describe --tags" | sed 's/^/   lowrank at /'
}

check_pristine () {
  echo "== confirm pristine /sgl-workspace/sglang is unpatched"
  local markers origs
  markers=$(ct "grep -rl '## XKV_LOWRANK' /sgl-workspace/sglang/python/sglang 2>/dev/null | wc -l")
  origs=$(ct "find /sgl-workspace/sglang -name '*.xkv.orig' 2>/dev/null | wc -l")
  echo "   pristine markers=$markers .orig files=$origs"
  [ "$markers" = 0 ] && [ "$origs" = 0 ] || die "pristine tree is NOT clean"
}

write_drift () {
  echo "== read-only anchor drift vs v0.5.18 -> $DRIFT"
  mkdir -p "$DRIFT_DIR"
  # Run from the live mounted repo (REPO_CT) so anchor edits in the host copy
  # are used, not a stale baked image copy. SG_LOWRANK_SRC pins the tree under
  # test inside the container.
  ct "cd $REPO_CT \
      && SG_LOWRANK_SRC=$LOWRANK/python python3 -m xkv drift" \
    > "$DRIFT" 2>&1
  local rc=$?
  echo "   drift rc=$rc (nonzero => anchors will need re-basing)"
  sed 's/^/   /' "$DRIFT" | head -40
  return 0   # drift finding is the deliverable, not a failure here
}

apply_patch () {
  echo "== apply xkv patch to $LOWRANK (v0.5.18 anchors)"
  ct "cd $REPO_CT \
      && SG_LOWRANK_SRC=$LOWRANK/python python3 -m xkv patch \
      && SG_LOWRANK_SRC=$LOWRANK/python python3 -m xkv verify" \
    || die "xkv patch/verify failed"
}

repatch () {
  # After the anchor declarations change, the tree on disk still holds the old
  # patch text, which no longer equals the new expected body -- so `patch` alone
  # refuses. Restore the originals from the .xkv.orig backups first, then apply.
  echo "== restore originals, then re-apply the current xkv patch"
  ct "cd $REPO_CT \
      && SG_LOWRANK_SRC=$LOWRANK/python python3 -m xkv unpatch" \
    || die "xkv unpatch failed"
  apply_patch
}

prep () {
  has_container || die "container $CONTAINER does not exist; run: $0 (no args)"
  clone_lowrank
  check_pristine
  write_drift
}

# ------------------------------ dispatch ------------------------------
case "$ACTION" in
  prep) prep ;;
  patch) prep; apply_patch ;;
  repatch) repatch ;;   # unpatch then patch; no prep (the tree is already there)
  drift)
    # Pure census against whatever container is named; writes nothing.
    has_container || die "container $CONTAINER does not exist"
    write_drift
    grep -q 'all anchors intact' "$DRIFT" \
      || die "anchors drifted; see $DRIFT"
    ;;
  recreate)
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    has_image || build_image
    bootstrap_container
    prep
    ;;
  up)
    has_image || build_image
    if has_container; then
      echo "== container $CONTAINER exists (image $IMAGE); skipping create"
      container_up || { echo "   container stopped; starting"; docker start "$CONTAINER" >/dev/null || die "docker start failed"; }
    else
      bootstrap_container
    fi
    prep
    ;;
  *) echo "usage: $0 [prep|patch|repatch|drift|recreate]" >&2; exit 2 ;;
esac
echo "== xkv ready. serve with:  bash scripts/local/serve.sh <native|xkv>"
echo "   (xkv needs the patch applied:  bash scripts/local/container.sh patch)"
