#!/usr/bin/env bash
# Watchdog for the native @262144 8-hard rerun (30212/0-3). Polls every 600s.
#   exit 0 = ALL_8_DONE seen
#   exit 2 = local 30212 server down
#   exit 3 = elapsed > 172800s (48h cap)
set -u
MASTER=/data/zyj/YJYBench/results/dsv4-hardnative-262k_master.log
WATCH_LOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/native_hard8_262k_watchdog.log
SSHPASS=/usr/bin/sshpass
SSH_ARGS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 root@10.57.3.76"
mkdir -p "$(dirname "$WATCH_LOG")"
T0=$(date +%s)
while true; do
  TS=$(date '+%F %H:%M:%S')
  LAST=$($SSHPASS -p 'a' ssh $SSH_ARGS "tail -1 '$MASTER' 2>/dev/null" 2>/dev/null)
  case "$LAST" in
    *ALL_8_DONE*) echo "$TS done last='$LAST'" >> "$WATCH_LOG"; exit 0 ;;
  esac
  SERVER=$(ps -eo pid,args | grep "[s]glang.launch_server" | grep 30212 | wc -l)
  if [ "$SERVER" -eq 0 ]; then
    echo "$TS SERVER-DOWN last='$LAST'" >> "$WATCH_LOG"; exit 2
  fi
  NOW=$(date +%s)
  if [ $((NOW - T0)) -gt 172800 ]; then
    echo "$TS CAP-48h last='$LAST'" >> "$WATCH_LOG"; exit 3
  fi
  echo "$TS ok last='$LAST'" >> "$WATCH_LOG"
  sleep 600
done
