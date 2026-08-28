#!/usr/bin/env bash
# TopMag50-on-native-c4: 20 single-instance Sangfor-Bench samples, 2 concurrent.
#
# Each sample is a SEPARATE yjybench invocation with a UNIQUE run_id. This is
# required because the harness names eval containers `task_id__instance_id__MMDDHHMMSS`
# (yjybench/core/evaluation.py:_container_name); 20 identical instance_ids in one
# run collide on the second-granularity timestamp -> docker 409 Conflict.
#
# 10 waves x 2 concurrent samples; each sample ~44 min agent phase -> ~7.3 h total.
set -u
cd /data/zyj/YJYBench
BENCH_ARGS="--benchmark Sangfor-Bench --mode e2e --max_workers 1 --timeout 18000 --exp_name test --instance_ids gcjs_kube-log-check-recover_2cadb18b --docker_env_config /data/zyj/YJYBench/test_env/docker_env_config_dsv4-windowed.json --agent_type cc --agent_mode vibe --sangforbench_prompt_source claude_result-tasks.md"

for batch in $(seq 1 10); do
  for k in 1 2; do
    i=$(( (batch - 1) * 2 + k ))
    RID=$(printf "dsv4-topmag50-20-%02d_20260827" "$i")
    LOG="/data/zyj/YJYBench/results/${RID}_launch.log"
    echo "[$(date +%H:%M:%S)] wave $batch launching $RID"
    nohup .venv/bin/python -m yjybench.cli $BENCH_ARGS --run_id "$RID" > "$LOG" 2>&1 &
  done
  echo "[$(date +%H:%M:%S)] wave $batch started, waiting for pair"
  wait
  echo "[$(date +%H:%M:%S)] wave $batch done"
done
echo "ALL_20_DONE $(date +%H:%M:%S)"
