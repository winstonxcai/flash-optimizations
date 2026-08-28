#!/usr/bin/env bash
# Watchdog for the 5-distinct-task TopMag50-on-native-c4 eval
# (run_topmag50_5distinct.sh, launched 2026-08-28 ~09:59).
# Polls every 10 min: server 30211 alive, per-task progress, ALL_5_DONE marker.
#   exit 0 = all 5 done (writeup time)
#   exit 2 = server 30211 died (relaunch + resume needed)
#   exit 3 = 12h cap hit
set -u
WATCH_LOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/eval5d_watchdog.log
MASTER=/data/zyj/YJYBench/results/dsv4-topmag50-5d_master.log
SSHPASS=/usr/bin/sshpass
SSH_ARGS="-o StrictHostKeyChecking=no root@10.57.3.76"
START=$(date +%s)
echo "watchdog start $(date '+%F %H:%M:%S')" >> "$WATCH_LOG"

while true; do
  NOW=$(date '+%H:%M:%S')
  ELAPSED=$(( $(date +%s) - START ))
  SERVER=$(pgrep -fc 'sglang.launch_server.*30211' 2>/dev/null || echo 0)

  REMOTE=$($SSHPASS -p 'a' ssh $SSH_ARGS \
    "echo starts=\$(grep -c '] START ' '$MASTER' 2>/dev/null); echo dones=\$(grep -c '] DONE ' '$MASTER' 2>/dev/null); echo last=\$(tail -1 '$MASTER' 2>/dev/null)" 2>/dev/null)
  STARTS=$(echo "$REMOTE" | sed -n 's/^starts=//p')
  DONES=$(echo "$REMOTE" | sed -n 's/^dones=//p')
  LAST=$(echo "$REMOTE" | sed -n 's/^last=//p')
  [ -z "$STARTS" ] && STARTS="?"; [ -z "$DONES" ] && DONES="?"
  [ -z "$LAST" ] && LAST="(no remote log)"

  echo "$NOW elapsed=${ELAPSED}s server=${SERVER} starts=${STARTS} dones=${DONES} last='${LAST}'" >> "$WATCH_LOG"

  case "$LAST" in
    *ALL_5_DONE*)
      echo "COMPLETE $(date '+%F %H:%M:%S')" >> "$WATCH_LOG"
      exit 0 ;;
  esac
  if [ "$SERVER" -eq 0 ]; then
    echo "SERVER_DOWN $(date '+%F %H:%M:%S')" >> "$WATCH_LOG"
    exit 2
  fi
  if [ "$ELAPSED" -gt 43200 ]; then
    echo "TIMEOUT_CAP $(date '+%F %H:%M:%S')" >> "$WATCH_LOG"
    exit 3
  fi
  sleep 600
done
