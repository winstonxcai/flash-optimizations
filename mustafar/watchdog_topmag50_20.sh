#!/usr/bin/env bash
# Watchdog for the 20-sample TopMag50-on-native-c4 eval (run_topmag50_20.sh).
# Polls every 10 min: server 30211 alive, waves completed, ALL_20_DONE marker.
#   exit 0 = all 20 done (writeup time)
#   exit 2 = server 30211 died (relaunch + resume needed)
#   exit 3 = 12h cap hit
set -u
WATCH_LOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/eval20_watchdog.log
EVAL_LOG=/data/zyj/YJYBench/results/dsv4-topmag50-20_master.log
SSHPASS=/usr/bin/sshpass
SSH_ARGS="-o StrictHostKeyChecking=no root@10.57.3.76"
START=$(date +%s)
echo "watchdog start $(date '+%F %H:%M:%S')" >> "$WATCH_LOG"

while true; do
  NOW=$(date '+%H:%M:%S')
  ELAPSED=$(( $(date +%s) - START ))
  SERVER=$(pgrep -fc 'sglang.launch_server' 2>/dev/null || echo 0)

  REMOTE=$($SSHPASS -p 'a' ssh $SSH_ARGS \
    "echo waves=\$(grep -cE 'wave [0-9]+ done' '$EVAL_LOG' 2>/dev/null); echo last=\$(tail -1 '$EVAL_LOG' 2>/dev/null)" 2>/dev/null)
  WAVES=$(echo "$REMOTE" | sed -n 's/^waves=//p')
  LAST=$(echo "$REMOTE" | sed -n 's/^last=//p')
  [ -z "$WAVES" ] && WAVES="?"
  [ -z "$LAST" ] && LAST="(no remote log)"

  echo "$NOW elapsed=${ELAPSED}s server=${SERVER} waves=${WAVES} last='${LAST}'" >> "$WATCH_LOG"

  case "$LAST" in
    *ALL_20_DONE*)
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
