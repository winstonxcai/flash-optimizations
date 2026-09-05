#!/usr/bin/env bash
# Autopilot: monitor LB2 (4-7) + gate/launch/monitor hard-native rerun (0-3).
# Polls every 300s. Launches the hard-native eval exactly once, when the TopMag25
# eval is ALL_25_DONE AND GPUs 0-3 are free. Exits 0 when BOTH LB2 and hard-native
# are done. Logs everything to autopilot_lb2_hardnative.log.
set -u
SCRIPTS=/home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts
SSHPASS=/usr/bin/sshpass
SSH_ARGS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 root@10.57.3.76"
CT=/mnt/host_root/home/jovyan/winstonxcai/transferibility
AUTOLOG=/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/autopilot_lb2_hardnative.log
MASTER25=/data/zyj/YJYBench/results/dsv4-topmag50-25d_master.log
MASTERHN=/data/zyj/YJYBench/results/dsv4-hardnative_master.log
mkdir -p "$(dirname "$AUTOLOG")"

lb2_done() { docker exec ruler-eval bash -c "grep -cE 'deadline hit|wrote .*lb2_prune100.json' '$CT/logs/par_lb2_prune100.log' 2>/dev/null || echo 0" 2>/dev/null | tail -1; }
hn_started() { $SSHPASS -p 'a' ssh $SSH_ARGS "grep -c 'HARD_START' '$MASTERHN' 2>/dev/null" 2>/dev/null; }
hn_done() { $SSHPASS -p 'a' ssh $SSH_ARGS "grep -c 'ALL_8_DONE' '$MASTERHN' 2>/dev/null" 2>/dev/null; }
rf25_done() { $SSHPASS -p 'a' ssh $SSH_ARGS "grep -c 'ALL_25_DONE' '$MASTER25' 2>/dev/null" 2>/dev/null; }

launched_hn=0
T0=$(date +%s)
while true; do
  TS=$(date '+%F %H:%M:%S')
  NOW=$(date +%s)
  if [ $((NOW - T0)) -gt 200000 ]; then
    echo "$TS AUTOPILOT-CAP-55h (run exceeded 55h)" >> "$AUTOLOG"; exit 3
  fi
  RF=$(rf25_done); HN=$(hn_done); HNS=$(hn_started); LB=$(lb2_done)
  echo "$TS poll rf25=$RF hn_started=$HNS hn_done=$HN lb2_done=$LB" >> "$AUTOLOG"

  if [ "$launched_hn" -eq 0 ] && [ "${HN:-0}" -ge 1 ]; then
    echo "$TS hard-native already done (no launch needed)" >> "$AUTOLOG"; launched_hn=1
  fi
  if [ "$launched_hn" -eq 0 ] && [ "${HNS:-0}" -ge 1 ]; then
    echo "$TS hard-native already running (monitoring)" >> "$AUTOLOG"; launched_hn=1
  fi
  if [ "$launched_hn" -eq 0 ] && [ "${RF:-0}" -ge 1 ]; then
    FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', ' '{if ($2 >= 75000) n++} END {print n+0}')
    if [ "$FREE" -ge 4 ]; then
      echo "$TS LAUNCH hard-native (rf25 done, gpus=$FREE/4)" >> "$AUTOLOG"
      ps -eo pid,args | grep "[s]glang.launch_server" | grep 30211 | awk '{print $1}' | xargs -r kill -9
      sleep 5
      docker exec ruler-eval bash /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/scripts/local/launch_inner_native.sh
      READY=0
      for i in $(seq 1 90); do
        R=$(docker exec ruler-eval bash -c "grep -c 'is fired up and ready' /mnt/host_root/home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/serve_native.log 2>/dev/null || echo 0")
        [ "${R:-0}" -ge 1 ] && READY=1 && break
        sleep 10
      done
      echo "$TS native server ready=$READY" >> "$AUTOLOG"
      $SSHPASS -p 'a' scp -q -o StrictHostKeyChecking=no \
        "$SCRIPTS/run_hard_native.sh" root@10.57.3.76:/data/zyj/YJYBench/run_hard_native.sh
      $SSHPASS -p 'a' ssh $SSH_ARGS \
        "cd /data/zyj/YJYBench && nohup bash run_hard_native.sh > results/dsv4-hardnative_outer.log 2>&1 & echo hardnative-launched"
      nohup bash "$SCRIPTS/watchdog_hard_native.sh" \
        > /home/jovyan/winstonxcai/flash-optimizations/mustafar/logs/hardnative_watchdog_outer.log 2>&1 &
      launched_hn=1
      echo "$TS hard-native launched" >> "$AUTOLOG"
    fi
  fi

  HN=$(hn_done); LB=$(lb2_done)
  if [ "${HN:-0}" -ge 1 ] && [ "${LB:-0}" -ge 1 ]; then
    echo "$TS ALL_DONE (lb2 done, hard-native done)" >> "$AUTOLOG"
    exit 0
  fi
  sleep 300
done
