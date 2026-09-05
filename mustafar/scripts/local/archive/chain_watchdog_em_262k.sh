#!/usr/bin/env bash
# Chained watchdog for the overlap plan: wait for a hard leg's ALL_8_DONE, then
# launch its easy+medium leg on the same server, then wait for ALL_17_DONE.
#   arg $1 = leg: topmag (30211) | native (30212)
#   exit 0 = EM leg ALL_17_DONE
#   exit 2 = local server down
#   exit 3 = 48h cap on hard phase
#   exit 4 = EM launch not confirmed within 5 min
#   exit 5 = 6d cap on EM phase
set -u
LEG=$1
case "$LEG" in
  topmag)
    HARD_MASTER=/data/zyj/YJYBench/results/dsv4-topmag50-hard8-262k_master.log
    EM_MASTER=/data/zyj/YJYBench/results/dsv4-topmag50-em-262k_master.log
    SCRIPT=run_topmag50_em_262k.sh
    SERVER_PORT=30211 ;;
  native)
    HARD_MASTER=/data/zyj/YJYBench/results/dsv4-hardnative-262k_master.log
    EM_MASTER=/data/zyj/YJYBench/results/dsv4-native-em-262k_master.log
    SCRIPT=run_native_em_262k.sh
    SERVER_PORT=30212 ;;
  *) echo "usage: $0 topmag|native"; exit 64 ;;
esac
SSHPASS=/usr/bin/sshpass
SSH_ARGS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 root@10.57.3.76"
WATCH_LOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/chain_${LEG}_em_262k.log
mkdir -p "$(dirname "$WATCH_LOG")"
log() { echo "$(date '+%F %H:%M:%S') $*" >> "$WATCH_LOG"; }

log "armed phase1 wait-hard $HARD_MASTER"
T0=$(date +%s)
while true; do
  LAST=$($SSHPASS -p 'a' ssh $SSH_ARGS "tail -1 '$HARD_MASTER' 2>/dev/null" 2>/dev/null)
  case "$LAST" in
    *ALL_8_DONE*) log "HARD-DONE last='$LAST'"; break ;;
  esac
  if [ "$(ps -eo pid,args | grep "[s]glang.launch_server" | grep $SERVER_PORT | wc -l)" -eq 0 ]; then
    log "SERVER-DOWN($SERVER_PORT) last='$LAST'"; exit 2
  fi
  if [ $(( $(date +%s) - T0 )) -gt 172800 ]; then log "CAP-48h last='$LAST'"; exit 3; fi
  log "phase1 ok last='$LAST'"
  sleep 300
done

log "launching $SCRIPT on 10.57.3.76"
timeout 25 $SSHPASS -p 'a' ssh $SSH_ARGS \
  "cd /data/zyj/YJYBench && test -f ./$SCRIPT || exit 9; nohup bash ./$SCRIPT > results/${SCRIPT%.sh}_outer.log 2>&1 </dev/null & echo LAUNCHED" \
  >> "$WATCH_LOG" 2>&1
LAUNCH_RC=$?
if [ "$LAUNCH_RC" -ne 0 ]; then log "LAUNCH-SSH-RC=$LAUNCH_RC"; fi

log "confirming EM_START"
for i in $(seq 1 10); do
  sleep 30
  CNT=$($SSHPASS -p 'a' ssh $SSH_ARGS "grep -c 'EM_START' '$EM_MASTER' 2>/dev/null" 2>/dev/null)
  if [ "${CNT:-0}" != "0" ]; then log "EM_START confirmed"; break; fi
  if [ $i -eq 10 ]; then log "EM-START-NOT-SEEN (launch failed?)"; exit 4; fi
done

log "phase2 wait-EM $EM_MASTER"
T1=$(date +%s)
while true; do
  LAST=$($SSHPASS -p 'a' ssh $SSH_ARGS "tail -1 '$EM_MASTER' 2>/dev/null" 2>/dev/null)
  case "$LAST" in
    *ALL_17_DONE*) log "ALL-17-DONE last='$LAST'"; exit 0 ;;
  esac
  if [ "$(ps -eo pid,args | grep "[s]glang.launch_server" | grep $SERVER_PORT | wc -l)" -eq 0 ]; then
    log "EM SERVER-DOWN($SERVER_PORT) last='$LAST'"; exit 2
  fi
  if [ $(( $(date +%s) - T1 )) -gt 518400 ]; then log "CAP-6d last='$LAST'"; exit 5; fi
  log "phase2 ok last='$LAST'"
  sleep 600
done
