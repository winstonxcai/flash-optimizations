#!/usr/bin/env bash
# =====================================================================
# container.sh -- bring up the starkv control container and prep the fork tree.
#
#   container.sh              ensure image + container exist, then prep
#   container.sh prep         prep only (container must already exist)
#   container.sh patch        prep, then apply the starkv patch to the fork
#                             tree (anchors are v0.5.18)
#   container.sh repatch      unpatch, then re-apply; use after the anchor
#                             declarations themselves change
#   container.sh recreate     docker rm -f the container, recreate, prep
#   container.sh drift        read-only anchor census only (no writes at all)
#
# Two pieces of state, in two places:
#
#   1. The `starkv` container -- CPU-ONLY, created without `--gpus`. It exists
#      for `python -m starkv selftest` (which needs torch) and to read the
#      pristine tree that ships in the image. It cannot reach a GPU by
#      construction; every GPU leg is a throwaway pinned container instead, so
#      nothing on the GPU path depends on this box being up.
#   2. The fork tree -- on the HOST at $FORK_HOST, cloned once from the image's
#      pristine tree, then patched and censused here. It has to be on the host
#      (not in a container) because a GPU leg is created and destroyed per run
#      and can only see what is bind-mounted. `patch`/`verify`/`drift` are pure
#      text operations on it and run with host python3 -- no torch, no container.
#
# Idempotent; `prep` is safe to re-run. It is deliberately separate from the
# other studies on this host: different container name, image, port and master
# port, so a running server elsewhere can never be disturbed by a starkv leg.
#
# prep ensures the fork tree and writes the read-only anchor drift report that
# scopes any future re-base:
#   <RESULTS_HOST>/drift-v0.5.18.md
#   (0 drifted = anchors intact against this tree)
#
# Patching is never baked into the Dockerfile.
# =====================================================================
set -u
export CONTAINER=${CONTAINER:-starkv}
DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
. "$DIR/env.sh"

ACTION=${1:-up}
DOCKERFILE="$HOST_REPO/starkv/docker/local.Dockerfile"
DRIFT_DIR="$RESULTS_HOST"
DRIFT="$DRIFT_DIR/drift-v0.5.18.md"

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
  #
  # NO `--gpus`. This container is the CPU-only control box (prep, patch, drift,
  # selftest); without device nodes it is structurally incapable of starting a
  # CUDA process, so nothing here can land on a GPU that GPUQ did not grant.
  # Every GPU leg runs in its own throwaway container -- see serve.sh.
  echo "== docker run --name $CONTAINER ($IMAGE), no GPUs"
  docker run -d --name "$CONTAINER" --network host --shm-size 120g \
    -v /:/mnt/host_root -v /home:/home \
    "$IMAGE" bash -c 'sleep infinity' || die "docker run failed"
  wait_exec_ready || die "container not exec-ready after launch"
}

# ------------------------- prep (two trees) --------------------------
# The fork tree lives on the HOST, not in a container: a GPU leg is a throwaway
# container, so anything it must see has to be on a bind mount. /home is mounted
# identically everywhere, so one host path serves both roles -- and it can be
# patched and censused without a container at all, since `starkv patch`/`drift`
# are pure text operations that need no torch.
clone_fork () {
  if [ ! -d "$FORK_HOST/.git" ]; then
    echo "== clone pristine -> $FORK_HOST at v0.5.18"
    # The pristine tree ships inside the image, so it has to come out once. The
    # control container is the cheapest way to read it.
    ct "test -d /sgl-workspace/sglang" \
      || die "/sgl-workspace/sglang missing from the control container"
    docker exec "$CONTAINER" tar -c -C /sgl-workspace sglang \
      | tar -x -C "$(dirname "$FORK_HOST")" || die "copy of the pristine tree failed"
    mv "$(dirname "$FORK_HOST")/sglang" "$FORK_HOST" || die "could not place $FORK_HOST"
    git -C "$FORK_HOST" checkout v0.5.18 >/dev/null 2>&1 \
      || die "git checkout v0.5.18 failed in $FORK_HOST"
  else
    echo "== $FORK_HOST already present"
  fi
  git -C "$FORK_HOST" describe --tags 2>/dev/null | sed 's/^/   fork at /'
}

check_pristine () {
  echo "== confirm pristine /sgl-workspace/sglang is unpatched"
  local markers origs
  markers=$(ct "grep -rl '## STARKV' /sgl-workspace/sglang/python/sglang 2>/dev/null | wc -l")
  origs=$(ct "find /sgl-workspace/sglang -name '*.starkv.orig' 2>/dev/null | wc -l")
  echo "   pristine markers=$markers .orig files=$origs"
  [ "$markers" = 0 ] && [ "$origs" = 0 ] || die "pristine tree is NOT clean"
}

write_drift () {
  echo "== read-only anchor drift vs v0.5.18 -> $DRIFT"
  mkdir -p "$DRIFT_DIR"
  # Runs on the HOST: `starkv patch`/`verify`/`drift` are pure text operations
  # over the tree -- no torch, no container. Running them here means the census
  # reads the exact file the GPU containers will serve from.
  ( cd "$HOST_REPO" && STARKV_SRC="$FORK_HOST/python" python3 -m starkv drift ) \
    > "$DRIFT" 2>&1
  local rc=$?
  echo "   drift rc=$rc (nonzero => anchors will need re-basing)"
  sed 's/^/   /' "$DRIFT" | head -40
  return 0   # drift finding is the deliverable, not a failure here
}

apply_patch () {
  echo "== apply starkv patch to $FORK_HOST (v0.5.18 anchors)"
  ( cd "$HOST_REPO" \
      && STARKV_SRC="$FORK_HOST/python" python3 -m starkv patch \
      && STARKV_SRC="$FORK_HOST/python" python3 -m starkv verify ) \
    || die "starkv patch/verify failed"
}

repatch () {
  # After the anchor declarations change, the tree on disk still holds the old
  # patch text, which no longer equals the new expected body -- so `patch` alone
  # refuses. Restore the originals from the .starkv.orig backups first, then
  # apply.
  echo "== restore originals, then re-apply the current starkv patch"
  ( cd "$HOST_REPO" && STARKV_SRC="$FORK_HOST/python" python3 -m starkv unpatch ) \
    || die "starkv unpatch failed"
  apply_patch
}

prep () {
  has_container || die "container $CONTAINER does not exist; run: $0 (no args)"
  clone_fork
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
echo "== starkv ready. serve with:  bash scripts/local/serve.sh <native|starkv-recon>"
echo "   (the fork legs need the patch applied:  bash scripts/local/container.sh patch)"
