# Business Replay Report

- Valid: `true`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 2169 / 2151`
- Success rate: `99.1701%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `1.772 req/s`
- Mean / P90 TTFT: `7371.778 / 8670.787 ms`
- Mean / P90 TPOT: `55.617 / 62.960 ms`
- Mean / P90 E2E: `13293.390 / 21290.753 ms`
- GPU / CPU / total KV cache hit rate: `97.3109% / 0.0000% / 97.3109%`
- KV cache detail coverage: `2127 / 2127`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'pass', 'ready_requests': 2169, 'within_threshold': 2012, 'violations': 157, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
