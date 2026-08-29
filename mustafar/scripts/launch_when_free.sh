#!/usr/bin/env bash
# Idempotent poll-and-launch gate for the 25-distinct TopMag50 eval.
# Safe to run every 5 min. Launches Phase B (server + remote eval + watchdog)
# exactly once, when 4 GPUs have >= 75 GB free.
#   exit 0 = eval already running, already done, or just launched (no-op now)
#   exit 1 = still waiting for GPUs
set -u
LOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/launch_when_free.log
SCRIPTS=/home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts
SSHPASS=/usr/bin/sshpass
SSH_ARGS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 root@10.57.3.76"
MASTER=/data/zyj/YJYBench/results/dsv4-topmag50-25d_master.log
TS=$(date '+%F %H:%M:%S')
mkdir -p "$(dirname "$LOG")"

# --- 1. already done / already running? (remote master log) ---
REMOTE=$($SSHPASS -p 'a' ssh $SSH_ARGS \
  "tail -1 '$MASTER' 2>/dev/null; echo __; grep -c 'MASTER_START' '$MASTER' 2>/dev/null" 2>/dev/null)
LAST=$(echo "$REMOTE" | sed -n '1p')
STARTED=$(echo "$REMOTE" | sed -n '3p')
case "$LAST" in
  *ALL_25_DONE*) echo "$TS noop already-complete last='$LAST'" >> "$LOG"; exit 0 ;;
esac
if [ "${STARTED:-0}" -ge 1 ] 2>/dev/null; then
  echo "$TS noop already-running starts=$STARTED last='$LAST'" >> "$LOG"; exit 0
fi

# --- 2. GPU gate: >= 4 GPUs with >= 75 GB free ---
FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' '{if ($2 >= 75000) n++} END {print n+0}')
if [ "$FREE" -lt 4 ]; then
  echo "$TS waiting gpus_free_ok=$FREE/4" >> "$LOG"
  exit 1
fi

# --- 3. launch (GPU gate passed, eval not started) ---
echo "$TS LAUNCH gpus_free_ok=$FREE/4" >> "$LOG"
bash "$SCRIPTS/launch.sh"                              # local: server in ruler-eval, waits for ready
$SSHPASS -p 'a' scp -q -o StrictHostKeyChecking=no \
  "$SCRIPTS/run_topmag50_25d.sh" root@10.57.3.76:/data/zyj/YJYBench/run_topmag50_25d.sh
$SSHPASS -p 'a' ssh $SSH_ARGS \
  "cd /data/zyj/YJYBench && nohup bash run_topmag50_25d.sh > results/dsv4-topmag50-25d_outer.log 2>&1 & echo eval-launched"
nohup bash "$SCRIPTS/watchdog_topmag50_25d.sh" \
  > /home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/eval25d_watchdog_outer.log 2>&1 &
echo "$TS LAUNCHED" >> "$LOG"
exit 0