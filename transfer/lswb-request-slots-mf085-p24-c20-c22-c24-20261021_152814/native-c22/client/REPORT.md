# Business Replay Report

- Valid: `false`
- Protocol: `business-request-slot-replay-v1`
- Model: `deepseek-v4-flash`
- Adapter: `/home/jovyan/winstonxcai/flash-optimizations/scripts/local/business_replay/adapters/openai_sse.py`
- Selected / attempted / successful: `4916 / 964 / 952`
- Success rate: `98.7552%`
- Duration: `1200.000 s`
- Scheduler: `request-slot-fifo`
- Maximum active users: `24`
- Warmup: `none (0 requests, not in metrics)`
- Audit level: `candidate`
- Comparison eligibility: `candidate`
- Request throughput: `0.777 req/s`
- Mean / P90 TTFT: `8230.905 / 12932.067 ms`
- Mean / P90 TPOT: `136.780 / 370.732 ms`
- Mean / P90 E2E: `25928.861 / 61075.939 ms`
- GPU / CPU / total KV cache hit rate: `90.7030% / 0.0000% / 90.7030%`
- KV cache detail coverage: `933 / 933`

This report is a business replay episode and is not a baseline matrix result.

- Ready-to-first-output SLO: `{'metric': 'ready_to_first_output', 'threshold_ms': 10000.0, 'percentile': 90, 'status': 'fail', 'ready_requests': 967, 'within_threshold': 646, 'violations': 319, 'right_censored': 2, 'basis': 'All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively'}`
HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload.
