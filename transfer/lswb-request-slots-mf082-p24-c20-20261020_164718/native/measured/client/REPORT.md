# Business Replay Report

- Valid: `false`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 331 / 325`
- Success rate: `98.1873%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `0.254 req/s`
- Mean / P90 TTFT: `18207.157 / 34243.988 ms`
- Mean / P90 TPOT: `517.722 / 909.594 ms`
- Mean / P90 E2E: `70932.859 / 153501.543 ms`
- GPU / CPU / total KV cache hit rate: `60.6392% / 0.0000% / 60.6392%`
- KV cache detail coverage: `305 / 305`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'fail', 'ready_requests': 335, 'within_threshold': 47, 'violations': 288, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
