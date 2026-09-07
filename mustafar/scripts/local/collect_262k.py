#!/usr/bin/env python3
# Collect completed 262k results across the four legs and print a table.
import json, os
BASE = '/data/zyj/YJYBench/results/test'

HARD = ["sri_swe-bench_5f5a7df7","sri_esecgpt_48486b59","sri_esecgpt_80fa3321",
        "sri_s1_cec32c82","sri_swe-bench_fea293e6","sri_ap-gpt_2bcf1160",
        "tw_esecgpt_f291630","sri_ap-gpt_d7527749"]
EM_M400 = ["gcjs_kube-log-check-recover_5b6a23ad","sri_s1_00ce55e2",
           "aiyycp_sales-flow_d7329e44","sri_chat-agent_86ce36d3",
           "sri_chat-agent_b2f8ec64","sri_s1_d060bef0"]
EM_OTHER = ["sri_esecgpt_cf8ba0fb","fy_gptanalystagent_fb3d6a3d",
            "gcjs_go-zero_22ab9e7d","sri_ap-gpt_0dd68d23"]
EM_EASY = ["gcjs_kube-log-check-recover_c6a12bfe","gcjs_kube-log-check-recover_fc67bfda",
           "tw_esecgpt_4966005","sri_chat-agent_035a16f0","tw_esecgpt_6741243f",
           "gcjs_kube-log-check-recover_e04abbb7","mss_drme-service_2a2095f8"]

def res(prefix, task, stamp):
    p = os.path.join(BASE, f'Sangfor-Bench_cc_vibe_deepseek-v4-flash_{prefix}-{task}_{stamp}', task, 'test_result.json')
    if not os.path.exists(p):
        return 'NOT-FOUND'
    j = json.load(open(p))
    t = j.get('tests')
    if not t:
        return f"no-tests(err={j.get('error')})"
    return f"{t['passed']}/{t['total']} ({100*t['passed']/t['total']:.1f})"

def dump(label, prefix, tasks, stamp):
    print(f"\n=== {label} ===")
    for t in tasks:
        print(f"  {t:<42} {res(prefix, t, stamp)}")

dump("TOPMAG HARD (8/8)", "dsv4-topmag50-hard8-262k", HARD, "20260831")
dump("NATIVE HARD (8/8)", "dsv4-hardnative-262k", HARD, "20260831")
dump("TOPMAG EM medium-400 (6/6 done)", "dsv4-topmag50-em-262k", EM_M400, "20260901")
dump("TOPMAG EM other-medium (0/4 done)", "dsv4-topmag50-em-262k", EM_OTHER, "20260901")
dump("NATIVE EM medium-400 (2/6 done)", "dsv4-native-em-262k", EM_M400, "20260901")
