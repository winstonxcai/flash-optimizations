# Business Replay Report

- Valid: `true`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 2147 / 2134`
- Success rate: `99.3945%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `1.760 req/s`
- Mean / P90 TTFT: `6416.854 / 8165.407 ms`
- Mean / P90 TPOT: `49.078 / 56.208 ms`
- Mean / P90 E2E: `11831.062 / 18084.007 ms`
- GPU / CPU / total KV cache hit rate: `97.3730% / 0.0000% / 97.3730%`
- KV cache detail coverage: `2112 / 2112`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'pass', 'ready_requests': 2149, 'within_threshold': 1961, 'violations': 186, 'right_censored': 2, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
