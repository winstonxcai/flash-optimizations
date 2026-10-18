# LongSWE C24: three repeats per implementation

Author: Winston Cai.

Question: compare Native, Packed-256 Triton and Packed-256 Kernel at matched concurrency, with three fresh-server runs per mode.

Protocol: TP4, C24, 1200-second dispatch windows, no profiling. Original run IDs and source paths are retained in manifest.json.

analysis.json is the canonical aggregate; summary.csv contains run-level measurements. SD and variance use the sample convention. TTFT percentile aggregates average run-level percentiles, rather than pooling requests.

Successful execution and saved timing validation do not establish numerical kernel correctness or coding-task quality. Timing coverage is count-based, not a verified client/server request-ID join. Full runtime source identity is not inferred from the container image ID.

Selected request metadata is transferred separately. Full logs and original artifacts remain in VDI results/. Raw prompts are not copied.
