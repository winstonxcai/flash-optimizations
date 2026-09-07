#!/usr/bin/env bash
# hicache-ladder.sh <native|packed> <tag> <C> <dur_s> [master_port]
# One fresh-boot 1200-s SLO-concurrency leg under the LOCKED HiCache config:
#   stop current server, drain port, HICACHE=1 boot (small graphs if C<=15 else
#   extended), run bench-lswb, print the row.
set -u
cd /home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts/local || exit 1
MODE=${1:-}; TAG=${2:-}; C=${3:-}; DUR=${4:-1200}; export GPUS=4,5,6,7 PORT=30212 MASTER_PORT=${5:-29648}
[ -n "$MODE" ] && [ -n "$TAG" ] && [ -n "$C" ] || { echo "usage: $0 <native|packed> <tag> <C> [dur_s] [master_port]"; exit 1; }
log(){ echo "[$(date -u +%H:%M:%S)Z] $*"; }
health () { curl -fsS -m 3 "http://127.0.0.1:30212/health" >/dev/null 2>&1; }

# decode graph: small (max_bs 15) if C<=15 else extended (max_bs 136)
if [ "$C" -le 15 ]; then
  export DECODE_CFG='{"decode":{"backend":"full","max_bs":15,"bs":[1,2,3,4,5,6,7,8,10,12,14,15]},"prefill":{"backend":"disabled"}}'
  log "$TAG C$C <= 15 -> small decode graphs"
else
  export DECODE_CFG='{"decode":{"backend":"full","max_bs":136,"bs":[1,2,3,4,5,6,7,8,10,12,14,15,16,18,20,24,28,32,34,40,48,56,64,68,80,96,112,120,136]},"prefill":{"backend":"disabled"}}'
  log "$TAG C$C > 15 -> extended decode graphs"
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
./bench-lswb.sh "$TAG" 30212 "$C" "$DUR"
RC=$?
log "==== row ===="
./lswb-row.sh "$TAG" 2>&1 | tail -2
exit $RC
