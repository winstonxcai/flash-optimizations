#!/usr/bin/env bash
# Watchdog for the NATIVE25 @262144 25-task run (two servers 30211+30212).
# Polls every 600s.
#   exit 0 = ALL_25_DONE seen
#   exit 2 = either local server (30211/30212) down
#   exit 3 = elapsed > 172800s (48h cap)
set -u
MASTER=/data/zyj/YJYBench/results/dsv4-native25-262k_master.log
WATCH_LOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/native25_262k_watchdog.log
SSHPASS=/usr/bin/sshpass
SSH_ARGS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 root@10.57.3.76"
mkdir -p "$(dirname "$WATCH_LOG")"
T0=$(date +%s)
while true; do
  TS=$(date '+%F %H:%M:%S')
  LAST=$($SSHPASS -p 'a' ssh $SSH_ARGS "tail -1 '$MASTER' 2>/dev/null" 2>/dev/null)
  DONE_CNT=$($SSHPASS -p 'a' ssh $SSH_ARGS "grep -c 'DONE ' '$MASTER' 2>/dev/null" 2>/dev/null)
  case "$LAST" in
    *ALL_25_DONE*) echo "$TS done last='$LAST'" >> "$WATCH_LOG"; exit 0 ;;
  esac
  A=$(ps -eo pid,args | grep "[s]glang.launch_server" | grep 30211 | wc -l)
  B=$(ps -eo pid,args | grep "[s]glang.launch_server" | grep 30212 | wc -l)
  if [ "$A" -eq 0 ] || [ "$B" -eq 0 ]; then
    echo "$TS SERVER-DOWN A=$A B=$B tasks_done=${DONE_CNT:-?} last='$LAST'" >> "$WATCH_LOG"; exit 2
  fi
  NOW=$(date +%s)
  if [ $((NOW - T0)) -gt 172800 ]; then
    echo "$TS CAP-48h tasks_done=${DONE_CNT:-?} last='$LAST'" >> "$WATCH_LOG"; exit 3
  fi
  echo "$TS ok tasks_done=${DONE_CNT:-?} last='$LAST'" >> "$WATCH_LOG"
  sleep 600
done
