# Business Replay Report

- Valid: `true`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 48 / 48`
- Success rate: `100.0000%`
- Duration: `60.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `3`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `0.767 req/s`
- Mean / P90 TTFT: `1433.898 / 3016.924 ms`
- Mean / P90 TPOT: `15.619 / 12.344 ms`
- Mean / P90 E2E: `2507.082 / 8669.635 ms`
- GPU / CPU / total KV cache hit rate: `92.1789% / 0.0000% / 92.1789%`
- KV cache detail coverage: `46 / 46`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'pass', 'ready_requests': 49, 'within_threshold': 46, 'violations': 2, 'right_censored': 1, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
