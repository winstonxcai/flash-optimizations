# Modal authorization and cost

- A Modal launch requires user authorization covering the experiment and
  budget, including CPU image builds and full-model runs. Existing approval
  for scoped runs/retries remains valid; do not repeatedly request it.
- Before requesting a new launch, finish local diagnosis/preflight and explain
  the fixed failures, remaining GPU checks, and expected resource use.
- Check the active profile, **compute spend**, cached artifacts/model volumes,
  and competing allocations with the Modal CLI. Follow the active experiment's
  account exclusions and caps; do not assume past account headroom persists.
- Build native extensions on CPU/image-build workers and reuse artifacts keyed
  by source/toolchain hashes. Verify binaries, Python dispatch, and manifests
  before GPU allocation. Stop on a failed preflight instead of rebuilding on
  H100 or blindly relaunching.
- Use the authorized timeout, retry, and concurrency limits. After a monitoring
  timeout, inspect the existing job before starting another. Stop allocations
  promptly after collection and record failures as part of the cost.
