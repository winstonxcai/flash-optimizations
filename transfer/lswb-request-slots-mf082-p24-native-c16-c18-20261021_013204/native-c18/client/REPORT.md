# Business Replay Report

- Valid: `false`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 332 / 328`
- Success rate: `98.7952%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `0.258 req/s`
- Mean / P90 TTFT: `16161.148 / 30563.225 ms`
- Mean / P90 TPOT: `487.980 / 848.146 ms`
- Mean / P90 E2E: `66264.801 / 137233.270 ms`
- GPU / CPU / total KV cache hit rate: `58.7592% / 0.0000% / 58.7592%`
- KV cache detail coverage: `310 / 310`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'fail', 'ready_requests': 338, 'within_threshold': 39, 'violations': 299, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
