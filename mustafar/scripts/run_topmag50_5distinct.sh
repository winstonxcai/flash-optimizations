#!/usr/bin/env bash
# TopMag50-on-native-c4: 5 DISTINCT Sangfor-Bench tasks, run SEQUENTIALLY.
#
# Why sequential: decode on this build is ~2-4 tok/s shared across requests; n=7
# showed 2-concurrent waves took 2h10m-3h29m vs 42m for n=1 alone (sequential 2
# samples = ~84m). Sequential 5 tasks keeps every agent at full server throughput.
#
# Each sample is a SEPARATE yjybench invocation with a UNIQUE run_id (harness
# names eval containers task_id__instance_id__MMDDHHMMSS with no uniquifier ->
# identical instance_ids in one run collide on the timestamp -> docker 409).
#
# Language mix (per task_description): 3 EN + 2 CN. Difficulty: easy x2, medium x2,
# hard x1. Native outcomes: 3 resolved + 2 failed (paired comparison per task).
set -u
cd /data/zyj/YJYBench
STAMP=20260828
MASTER=/data/zyj/YJYBench/results/dsv4-topmag50-5d_master.log
COMMON="--benchmark Sangfor-Bench --mode e2e --max_workers 1 --timeout 18000 --exp_name test --docker_env_config /data/zyj/YJYBench/test_env/docker_env_config_dsv4-windowed.json --agent_type cc --agent_mode vibe --sangforbench_prompt_source claude_result-tasks.md"

# task  | instance_id                    | difficulty | lang | native
TASKS="
sri_esecgpt_ebc6bf7a|easy|EN|failed(50%)
apex_soar-app_b05c9039|easy|CN|resolved
sri_swe-bench_35a41525|medium|EN|resolved
sri_s1_f650e49b|medium|CN|failed(95.8%)
apex_chat-agent_9347a21|hard|CN|resolved
"

echo "MASTER_START $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
while IFS='|' read -r INST DIFF LANG NATIVE; do
  [ -z "$INST" ] && continue
  RID="dsv4-topmag50-5d-$(echo "$INST" | sed 's/[^a-z0-9-]/_/g' | cut -c1-24)_$STAMP"
  LOG="/data/zyj/YJYBench/results/${RID}_launch.log"
  echo "[$(date '+%H:%M:%S')] START $INST (diff=$DIFF lang=$LANG native=$NATIVE) rid=$RID" | tee -a "$MASTER"
  .venv/bin/python -m yjybench.cli $COMMON --instance_ids "$INST" --run_id "$RID" > "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%H:%M:%S')] DONE $INST rc=$RC" | tee -a "$MASTER"
done <<< "$TASKS"
echo "ALL_5_DONE $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
