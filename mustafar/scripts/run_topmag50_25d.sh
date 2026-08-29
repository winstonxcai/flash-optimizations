#!/usr/bin/env bash
# TopMag50-on-native-c4: 25 DISTINCT Sangfor-Bench tasks, run SEQUENTIALLY.
#
# Generalization eval: even native pass/fail split, difficulty varied.
# Ordered easy -> medium -> hard, pass/fail interleaved per block, so the
# signal is a steady mix and the long/hard tasks land last. 13 native-pass +
# 12 native-fail; 7 easy / 10 medium / 8 hard; 10 go / 15 python; 13 families.
#
# Why sequential: decode on this build is ~2-4 tok/s shared across requests; n=7
# showed 2-concurrent waves took 2h10m-3h29m vs 42m for n=1 alone (sequential 2
# samples = ~84m). Sequential keeps every agent at full server throughput.
#
# Each sample is a SEPARATE yjybench invocation with a UNIQUE run_id (harness
# names eval containers task_id__instance_id__MMDDHHMMSS with no uniquifier ->
# identical instance_ids in one run collide on the timestamp -> docker 409).
#
# Native baseline for the pass/fail labels: cloud run task_20260825_195126_744.
set -u
cd /data/zyj/YJYBench
STAMP=20260829
MASTER=/data/zyj/YJYBench/results/dsv4-topmag50-25d_master.log
COMMON="--benchmark Sangfor-Bench --mode e2e --max_workers 1 --timeout 18000 --exp_name test --docker_env_config /data/zyj/YJYBench/test_env/docker_env_config_dsv4-windowed.json --agent_type cc --agent_mode vibe --sangforbench_prompt_source claude_result-tasks.md"

# task  | instance_id                         | difficulty | lang   | native
TASKS="
gcjs_kube-log-check-recover_c6a12bfe|easy|go|122/122
gcjs_kube-log-check-recover_fc67bfda|easy|go|131/132
tw_esecgpt_4966005|easy|python|40/40
sri_chat-agent_035a16f0|easy|python|25/27
tw_esecgpt_6741243f|easy|python|40/40
gcjs_kube-log-check-recover_e04abbb7|easy|go|73/74
mss_drme-service_2a2095f8|easy|python|35/35
sri_esecgpt_cf8ba0fb|medium|go|268/268
gcjs_kube-log-check-recover_5b6a23ad|medium|go|101/254
sri_s1_00ce55e2|medium|python|126/126
aiyycp_sales-flow_d7329e44|medium|go|3/74
fy_gptanalystagent_fb3d6a3d|medium|python|111/111
sri_chat-agent_86ce36d3|medium|python|0/62
sri_chat-agent_b2f8ec64|medium|python|75/75
sri_s1_d060bef0|medium|python|118/131
gcjs_go-zero_22ab9e7d|medium|go|48/48
sri_ap-gpt_0dd68d23|medium|python|119/122
sri_swe-bench_5f5a7df7|hard|python|116/116
sri_esecgpt_48486b59|hard|go|75/227
sri_esecgpt_80fa3321|hard|go|267/267
sri_s1_cec32c82|hard|python|76/176
sri_swe-bench_fea293e6|hard|python|86/86
sri_ap-gpt_2bcf1160|hard|python|16/164
tw_esecgpt_f291630|hard|go|243/243
sri_ap-gpt_d7527749|hard|python|1/137
"

echo "MASTER_START $(date '+%F %H:%M:%S')" | tee -a "$MASTER"
while IFS='|' read -r INST DIFF LANG NATIVE; do
  [ -z "$INST" ] && continue
  # full instance id (max ~36 chars) so the 4 gcjs_kube instances keep their
  # distinct hex suffixes — a 24-char cut made all four collide.
  RID="dsv4-topmag50-25d-$(echo "$INST" | sed 's/[^a-z0-9-]/_/g')_$STAMP"
  LOG="/data/zyj/YJYBench/results/${RID}_launch.log"
  echo "[$(date '+%H:%M:%S')] START $INST (diff=$DIFF lang=$LANG native=$NATIVE) rid=$RID" | tee -a "$MASTER"
  .venv/bin/python -m yjybench.cli $COMMON --instance_ids "$INST" --run_id "$RID" > "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%H:%M:%S')] DONE $INST rc=$RC" | tee -a "$MASTER"
done <<< "$TASKS"
echo "ALL_25_DONE $(date '+%F %H:%M:%S')" | tee -a "$MASTER"