# Business Replay Report

- Valid: `true`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 2179 / 2164`
- Success rate: `99.3116%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `1.787 req/s`
- Mean / P90 TTFT: `5802.515 / 7682.376 ms`
- Mean / P90 TPOT: `43.622 / 51.237 ms`
- Mean / P90 E2E: `10942.240 / 17022.270 ms`
- GPU / CPU / total KV cache hit rate: `97.3518% / 0.0000% / 97.3518%`
- KV cache detail coverage: `2144 / 2144`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'fail', 'ready_requests': 2183, 'within_threshold': 1939, 'violations': 240, 'right_censored': 4, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
