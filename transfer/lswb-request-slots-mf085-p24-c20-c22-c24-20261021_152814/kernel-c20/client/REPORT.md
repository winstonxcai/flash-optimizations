# Business Replay Report

- Valid: `true`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 2170 / 2155`
- Success rate: `99.3088%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `1.779 req/s`
- Mean / P90 TTFT: `5950.141 / 7705.452 ms`
- Mean / P90 TPOT: `44.387 / 52.221 ms`
- Mean / P90 E2E: `10968.301 / 18073.968 ms`
- GPU / CPU / total KV cache hit rate: `97.3991% / 0.0000% / 97.3991%`
- KV cache detail coverage: `2135 / 2135`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'fail', 'ready_requests': 2174, 'within_threshold': 1930, 'violations': 244, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
