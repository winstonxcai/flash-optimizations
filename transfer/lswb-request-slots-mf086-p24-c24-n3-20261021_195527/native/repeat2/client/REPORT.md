# Business Replay Report

- Valid: `false`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 1351 / 1337`
- Success rate: `98.9637%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `1.094 req/s`
- Mean / P90 TTFT: `7923.863 / 9528.376 ms`
- Mean / P90 TPOT: `113.550 / 319.557 ms`
- Mean / P90 E2E: `20983.017 / 41032.754 ms`
- GPU / CPU / total KV cache hit rate: `93.8361% / 0.0000% / 93.8361%`
- KV cache detail coverage: `1313 / 1313`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'pass', 'ready_requests': 1351, 'within_threshold': 1219, 'violations': 132, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
