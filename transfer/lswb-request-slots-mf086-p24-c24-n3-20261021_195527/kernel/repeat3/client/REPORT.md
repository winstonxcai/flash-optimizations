# Business Replay Report

- Valid: `true`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 2230 / 2215`
- Success rate: `99.3274%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `1.826 req/s`
- Mean / P90 TTFT: `7133.811 / 8713.552 ms`
- Mean / P90 TPOT: `53.607 / 56.822 ms`
- Mean / P90 E2E: `12919.694 / 19716.984 ms`
- GPU / CPU / total KV cache hit rate: `97.3435% / 0.0000% / 97.3435%`
- KV cache detail coverage: `2191 / 2191`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'pass', 'ready_requests': 2230, 'within_threshold': 2108, 'violations': 122, 'right_censored': 0, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
