# TopMag 50% on the native c4 latent — Sangfor-Bench 5-distinct-task generalization

Date: 2026-08-28 · DeepSeek-V4-Flash (21 c4 latent layers, compress_ratio=4)
Build: `SGLANG_OPT_TOPMAG=1` + `XKV_TOPMAG_KEEP=0.5`, **`XKV_DEBUG=0`** (Mustafar package, `flash-optimizations/mustafar/`)
Change: **store-time only.** Smallest-|·| 256/512 coords zeroed in place before the stock fused `compress_norm_rope_store`. The memory pool (584 B/token native layout), decode, and every other path are the **stock DeepSeek-V4 build** — no lowrank KV, no basis/SVD, no pool change.

## Result — 4/5 tasks native-equivalent; the one task with headroom regressed

5 **distinct** Sangfor-Bench tasks (3 EN + 2 CN, easy×2 / medium×2 / hard×1, 3 native-resolved + 2 native-failed), run **sequentially** on one server (port 30211), each a separate single-instance eval with a unique run_id (`dsv4-topmag50-5d-<task>_20260828`). Native baselines = the cloud 0725 web run (`task_20260825_195126_744`, `newapi-ai.sangfor.com`).

| task | diff/lang | native (cloud) | **TopMag50** | verdict |
|---|---|---|---|---|
| sri_esecgpt_ebc6bf7a | easy/EN | 50% (5/10) | **100% (10/10)** | ✅ fixed |
| apex_soar-app_b05c9039 | easy/CN | 100% | 100% (10/10) | ✅ hold |
| sri_swe-bench_35a41525 | medium/EN | 100% | 100% (43/43) | ✅ hold |
| sri_s1_f650e49b | medium/CN | 95.8% (69/72) | **72.2% (52/72)** | ⚠️ **regression** |
| apex_chat-agent_9347a21 | hard/CN | 100% | 100% (56/56) | ✅ hold |

- The **hardest task held clean**: chat-agent (56-test multi-file agent loop) passed 56/56 — the place long-context KV loss should bite hardest.
- **sri_s1 is the first real regression signal**: 69/72 → 52/72, 17 extra test failures. It is also the only task in the set with headroom (native 95.8%, not 100%) — i.e. the only task that *could* reveal a TopMag sensitivity, and it did.

## Caveats

- **n=1 per task.** No variance bound.
- **Native baselines are the cloud (newapi-ai) web run, TopMag is local-stock** — a cloud-vs-local confound. This matters most for the one regression (sri_s1). A local native rerun of sri_s1 is the decisive check.
- **Ceiling effect on the held tasks:** native 100% → TopMag 100% can't distinguish "preserved fidelity" from "task too easy to expose KV loss". swe-bench (43) and chat-agent (56) carry meaningful test counts; the two 10-test tasks are weak evidence either way.

## Interpretation

The n=7 same-task run (`writeup/topmag-native-sangfor-n7.md`) established σ=0 on one instance. This run is the first cross-task claim: TopMag50-native held native's full pass on 4/5 distinct tasks including the hardest, but the single task with margin to lose degraded. The honest reading is that dense-zero TopMag at 50% is *close to* native-generalizable — and sri_s1 is the boundary case that needs one local rerun before it's called a real sensitivity.

## Artifacts

- 5 result dirs: `/data/zyj/YJYBench/results/test/Sangfor-Bench_cc_vibe_*dsv4-topmag50-5d-<task>_20260828/<task>/test_result.json`
- Master log: `/data/zyj/YJYBench/results/dsv4-topmag50-5d_master.log` (`ALL_5_DONE 2026-08-28 13:20:53`)
- Native reference: cloud 0725 web run `task_20260825_195126_744`
- Launcher: `flash-optimizations/mustafar/scripts/run_topmag50_5distinct.sh` · watchdog: `watchdog_topmag50_5distinct.sh`
- Stage 0 (bit-exact Triton sparse pack/unpack, storage 576<1024 B/row bf16) is committed — see plan `mustafar/triton/plan.md`; this eval used the dense-zero store, so bandwidth saving is still untested.

## Reproduce

```
SGLANG_OPT_TOPMAG=1 XKV_TOPMAG_KEEP=0.5 XKV_DEBUG=0 \
  python3 -m sglang.launch_server --model-path .../DeepSeek-V4-Flash-FP8 \
  --served-model-name deepseek-v4-flash --tp 4 --fp8-gemm-backend triton --disable-cuda-graph
# then, one invocation per instance:
python3 -m yjybench.cli --benchmark Sangfor-Bench --mode e2e --max_workers 1 --timeout 18000 \
  --exp_name test --docker_env_config docker_env_config_web_*.json --agent_type cc --agent_mode vibe \
  --sangforbench_prompt_source claude_result-tasks.md --instance_ids <task> \
  --run_id dsv4-topmag50-5d-<task>_20260828
```
