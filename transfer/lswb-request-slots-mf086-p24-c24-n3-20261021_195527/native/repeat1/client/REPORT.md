# Business Replay Report

- Valid: `false`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 1311 / 1295`
- Success rate: `98.7796%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `1.059 req/s`
- Mean / P90 TTFT: `7169.441 / 9647.723 ms`
- Mean / P90 TPOT: `124.250 / 347.645 ms`
- Mean / P90 E2E: `21093.837 / 43873.841 ms`
- GPU / CPU / total KV cache hit rate: `93.1989% / 0.0000% / 93.1989%`
- KV cache detail coverage: `1271 / 1271`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'fail', 'ready_requests': 1311, 'within_threshold': 1177, 'violations': 134, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
