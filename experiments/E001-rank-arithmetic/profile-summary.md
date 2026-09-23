# E001 profile summary

Baseline profile:

```text
remnant-stage2a-results:/20260922T081611Z-profile-flashmla-direct-19a1035c/
```

Candidate profile:

```text
remnant-stage2a-results:/20260922T163804Z-profile-flashmla-direct-e6cd43cd/
```

The candidate Direct kernel was faster in all four matched Nsight Compute
cases. Registers remained at 168 and register spills remained zero.

| Shape | Time change | Instruction change | Shared wavefront change | Bank-conflict change | Long-scoreboard change |
|---|---:|---:|---:|---:|---:|
| H64/B8 | −7.47% | +2.03% | −0.77% | −3.27% | −11.45% |
| H64/B16 | −10.37% | +2.24% | −0.89% | −3.55% | −12.79% |
| H128/B8 | −8.78% | +1.60% | −0.86% | −4.11% | −13.52% |
| H128/B16 | −12.97% | +1.56% | −0.61% | −2.78% | −13.15% |
