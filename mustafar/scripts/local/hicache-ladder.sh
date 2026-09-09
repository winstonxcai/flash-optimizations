#!/usr/bin/env bash
# hicache-ladder.sh <native|packed> <tag> <C> <dur_s> [master_port]
# One fresh-boot <dur_s>-SLO-concurrency leg under the LOCKED HiCache config:
#   stop current server, drain port, HICACHE=1 boot (small decode graphs if
#   C<=15 else extended), run bench-lswb, print the row.
#
# Runs against the env.sh defaults (remnant: GPUS 0-3 / PORT 30212 /
# MASTER 29638). Override via env (e.g. GPUS/PORT/MASTER_PORT) to aim at the
# frozen ruler-eval box instead.
set -u
cd /home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts/local || exit 1
. "$(dirname "$0")/env.sh"
[ $# -ge 5 ] && export MASTER_PORT=$5
MODE=${1:-}; TAG=${2:-}; C=${3:-}; DUR=${4:-1200}
[ -n "$MODE" ] && [ -n "$TAG" ] && [ -n "$C" ] || { echo "usage: $0 <native|packed> <tag> <C> [dur_s] [master_port]"; exit 1; }
log () { echo "[$(date -u +%H:%M:%S)Z] $*"; }

# decode graph: small (max_bs 15) if C<=15 else extended (max_bs 136)
if [ "$C" -gt 15 ]; then
  export DECODE_CFG=$DECODE_CFG_EXT
  log "$TAG C$C > 15 -> extended decode graphs"
else
  export DECODE_CFG=$DECODE_CFG_SMALL
  log "$TAG C$C <= 15 -> small decode graphs"
fi

# 1) stop whatever is up, drain port
if health; then
  log "stopping current server"
  ./serve.sh "$MODE" stop || true
  sleep 12
  for i in $(seq 1 10); do health || break; sleep 3; done
  health && { log "!! port still answering after stop -- aborting"; exit 1; }
fi

# 2) fresh-boot <mode> + HiCache (locked config)
log "fresh-booting HICACHE=1 $MODE server (empty radix)"
export HICACHE=1
./serve.sh "$MODE" || { log "!! $MODE+hicache boot failed"; exit 1; }
log "$MODE+hicache healthy on fresh boot"
unset HICACHE

# 3) the SLO leg, first traffic
log "== $TAG C$C @ ${DUR}s (fresh boot, first traffic) =="
./bench-lswb.sh "$TAG" "$PORT" "$C" "$DUR"
RC=$?
log "==== row ===="
./lswb-row.sh "$TAG" 2>&1 | tail -2
exit $RC
