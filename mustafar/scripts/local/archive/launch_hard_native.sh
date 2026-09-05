#!/usr/bin/env bash
# Idempotent poll-and-launch gate for the hard-task NATIVE-CSA control rerun.
# Safe to run every 5 min. Fires exactly once: after the TopMag25 eval is
# ALL_25_DONE AND GPUs 0-3 are free, kills the 30211 TopMag server, relaunches a
# NATIVE server on 0-3, starts the 8-hard-task eval on the remote, background watchdog.
#   exit 0 = done / running / just launched (no-op now)
#   exit 1 = still waiting (sangfor25 not done, or GPUs busy)
set -u
LOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/launch_hard_native.log
SCRIPTS=/home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts
SSHPASS=/usr/bin/sshpass
SSH_ARGS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 root@10.57.3.76"
MASTER25=/data/zyj/YJYBench/results/dsv4-topmag50-25d_master.log
MASTERHN=/data/zyj/YJYBench/results/dsv4-hardnative_master.log
TS=$(date '+%F %H:%M:%S')
mkdir -p "$(dirname "$LOG")"

# --- 1. hard-native already done or running? ---
REMOTE=$($SSHPASS -p 'a' ssh $SSH_ARGS \
  "tail -1 '$MASTERHN' 2>/dev/null; echo __; grep -c 'HARD_START' '$MASTERHN' 2>/dev/null" 2>/dev/null)
LAST=$(echo "$REMOTE" | sed -n '1p')
STARTED=$(echo "$REMOTE" | sed -n '3p')
case "$LAST" in
  *ALL_8_DONE*) echo "$TS noop already-done last='$LAST'" >> "$LOG"; exit 0 ;;
esac
if [ "${STARTED:-0}" -ge 1 ] 2>/dev/null; then
  echo "$TS noop already-running starts=$STARTED last='$LAST'" >> "$LOG"; exit 0
fi

# --- 2. TopMag25 eval done yet? ---
R25=$($SSHPASS -p 'a' ssh $SSH_ARGS "tail -1 '$MASTER25' 2>/dev/null" 2>/dev/null)
case "$R25" in
  *ALL_25_DONE*) : ;;
  *) echo "$TS wait sangfor25 last='$R25'" >> "$LOG"; exit 1 ;;
esac

# --- 3. GPU gate: GPUs 0-3 free ---
FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' '{if ($2 >= 75000) n++} END {print n+0}')
if [ "$FREE" -lt 4 ]; then
  echo "$TS wait gpus_free_ok=$FREE/4 (sangfor25 done)" >> "$LOG"; exit 1
fi

# --- 4. launch (sangfor25 done, GPUs free, native not started) ---
echo "$TS LAUNCH sangfor25-done gpus=$FREE/4" >> "$LOG"
ps -eo pid,args | grep "[s]glang.launch_server" | grep 30211 | awk '{print $1}' | xargs -r kill -9
sleep 5
docker exec ruler-eval bash /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts/local/launch_inner_native.sh
for i in $(seq 1 90); do
  R=$(docker exec ruler-eval bash -c "grep -c 'is fired up and ready' /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_native.log 2>/dev/null || echo 0")
  [ "$R" -ge 1 ] && break
  sleep 10
done
$SSHPASS -p 'a' scp -q -o StrictHostKeyChecking=no \
  "$SCRIPTS/run_hard_native.sh" root@10.57.3.76:/data/zyj/YJYBench/run_hard_native.sh
$SSHPASS -p 'a' ssh $SSH_ARGS \
  "cd /data/zyj/YJYBench && nohup bash run_hard_native.sh > results/dsv4-hardnative_outer.log 2>&1 & echo hardnative-launched"
nohup bash "$SCRIPTS/watchdog_hard_native.sh" \
  > /home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/hardnative_watchdog_outer.log 2>&1 &
echo "$TS LAUNCHED" >> "$LOG"
exit 0
