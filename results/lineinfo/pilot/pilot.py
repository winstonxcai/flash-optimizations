#!/usr/bin/env python3
"""Bounded single-GPU pilot: H64 decode parity + 4 ncu captures.

Uses ONLY the installed AOT shim (`sgl_kernel.flash_mla`), keyword metadata form.
  * native  -> flash_mla_with_kvcache(..., extra_k_cache/native_indices, no remnant_*)
  * remnant -> flash_mla_with_kvcache(..., remnant_buffers/raw_indices/freqs)
Fixture: third_party/flashmla/tests/lib.make_remnant_case (seed 20260919),
h64 batch 32 topk 512.

Modes (selected by PILOT_MODE):
  * default (parity) : parity gate + CUDA-graph timing, NO profiling.
  * native / remnant : import, warm the SELECTED mode only, then profile exactly
    ONE eager call of that mode inside a profiler window (ncu --launch-count 1).
Each mode runs in its OWN child process so a profiled report can contain only
that path's launch -- the previously-captured reports were invalid comparisons
because one window held both a native and a remnant launch.
No kernel changes, no retries.
"""
import json
import os
import sys
import time

import torch

# The lib/ fixture dir is provided under the copied harness; this must be set
# before importing lib so it can import kernelkit/quant/flash_mla from there.
_FIX = os.environ.get("PILOT_FIXTURE_DIR")
if _FIX:
    sys.path.insert(0, _FIX)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "flashmla", "tests"))

# External-harness shim replaces the test fixture's `import flash_mla` with the
# installed AOT shim -- the fix that used to live as a hunk in the tracked
# third_party/flashmla/tests/lib.py (now reverted; see pilot/harness_shim.py).
import harness_shim  # noqa: E402
flash_mla = harness_shim.install_flash_mla_fallback()
import lib  # noqa: E402

H = 64
B = 32
TOPK = 512
WARMUP = 10
SEED = 20260919
OUT = os.environ["PILOT_OUT"]
SELFCHECK = os.environ.get("PILOT_SELFCHECK") == "1"
MODE = os.environ.get("PILOT_MODE", "parity")  # parity | native | remnant
# Fixture settings for the report (h64 mode is verified from these + launch
# resources + binary metadata, not from template text in the profiler name).
FIXTURE_META = {"num_heads": H, "topk_length": TOPK, "batch": B, "seed": SEED}


def run_native(case, meta):
    # native h64 = KernelTemplate<MODEL1,64,false>: plain extra block, no remnant_*.
    return flash_mla.flash_mla_with_kvcache(
        q=case.q, k_cache=case.swa_cache, block_table=None, cache_seqlens=None,
        head_dim_v=512, tile_scheduler_metadata=meta,
        softmax_scale=case.sm_scale, is_fp8_kvcache=True,
        indices=case.swa_indices, attn_sink=case.sink,
        extra_k_cache=case.native_cache,
        extra_indices_in_kvcache=case.native_indices,
        topk_length=case.swa_length, extra_topk_length=case.extra_length,
    )


def run_remnant(case, meta):
    # remnant h64 = KernelTemplate<MODEL1,64,true>: shim packs remnant_buffers
    # into extra_k_cache and dispatches remnant_sparse_decode_fwd.
    return flash_mla.flash_mla_with_kvcache(
        q=case.q, k_cache=case.swa_cache, block_table=None, cache_seqlens=None,
        head_dim_v=512, tile_scheduler_metadata=meta,
        softmax_scale=case.sm_scale, is_fp8_kvcache=True,
        indices=case.swa_indices, attn_sink=case.sink,
        extra_indices_in_kvcache=case.packed_indices,
        topk_length=case.swa_length, extra_topk_length=case.extra_length,
        remnant_buffers=case.packed_buffers,
        remnant_raw_indices=case.raw_indices,
        remnant_freqs=case.freqs,
    )


def new_meta():
    # Fresh FlashMLASchedMeta per path: the shim asserts the cached scheduler
    # config matches the call, and native vs remnant build the extra page
    # differently, so a shared instance fails the extra_page_block_size assert.
    return flash_mla.get_mla_metadata()[0]


def print_resources(tag):
    # Launch resources the report will show; recorded for cross-check.
    dev = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(dev)
    props = {"tag": tag, "device_index": dev, "device_name": p.name,
             "sm_count": p.multi_processor_count,
             "cc_major": p.major, "cc_minor": p.minor}
    print("RESOURCES " + json.dumps(props), flush=True)


def main():
    if SELFCHECK:
        import hashlib
        import sgl_kernel.flashmla_ops as ext
        import sgl_kernel.flash_mla as fm
        want = os.environ["PILOT_LINEINFO_SHA"]
        got = hashlib.sha256(open(ext.__file__, "rb").read()).hexdigest()
        assert got == want, f"resolved sha {got} != lineinfo sha {want}"
        print("SELFCHECK_IMPORT_OK", ext.__file__, got, flush=True)
        assert hasattr(fm, "flash_mla_with_kvcache")
        print("SELFCHECK_DONE", flush=True)
        return

    torch.manual_seed(SEED)
    case = lib.make_remnant_case(H, TOPK, batch=B, device="cuda",
                                 unique_selections=False)

    # ---- profiled single-mode child: warm ONLY the selected mode, 1 eager call.
    if MODE in ("native", "remnant"):
        meta = new_meta()
        fn = run_native if MODE == "native" else run_remnant
        for _ in range(WARMUP):
            fn(case, meta)
        torch.cuda.synchronize()
        print_resources(MODE)
        print(f"MODE_BEGIN {MODE}", flush=True)
        torch.cuda.profiler.start()
        fn(case, meta)                        # exactly one instrumented launch
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        print(f"MODE_DONE {MODE}", flush=True)
        return

    # ---- parity + timing parent (NO profiling; nb: ncu never runs this path).
    meta_nat = new_meta()
    meta_rem = new_meta()
    torch.cuda.synchronize()

    n_out, n_lse = run_native(case, meta_nat)
    r_out, r_lse = run_remnant(case, meta_rem)
    torch.cuda.synchronize()
    assert torch.isfinite(n_out).all() and torch.isfinite(n_lse).all()
    assert torch.isfinite(r_out).all() and torch.isfinite(r_lse).all()
    torch.testing.assert_close(r_out, n_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(r_lse, n_lse, atol=2.0e-2, rtol=2.0e-2)
    parity = {
        "h": H, "b": B, "topk": TOPK, "seed": SEED,
        "out_shape": list(n_out.shape), "lse_shape": list(n_lse.shape),
        "out_max_abs_diff": float((r_out.float() - n_out.float()).abs().max()),
        "lse_max_abs_diff": float((r_lse.float() - n_lse.float()).abs().max()),
        "atol": 2.0e-2, "rtol": 2.0e-2, "status": "PASS",
    }
    with open(os.path.join(OUT, "parity.json"), "w") as f:
        json.dump(parity, f, indent=2)
    print("PARITY_PASS", parity["out_max_abs_diff"],
          parity["lse_max_abs_diff"], flush=True)

    for _ in range(WARMUP):
        run_native(case, meta_nat)
        run_remnant(case, meta_rem)
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_native(case, meta_nat)
            run_remnant(case, meta_rem)
    torch.cuda.current_stream().wait_stream(s)

    g_nat = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_nat):
        run_native(case, meta_nat)
    g_rem = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_rem):
        run_remnant(case, meta_rem)

    def timed(graph, iters=50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6

    t = {"native_us": [], "remnant_us": []}
    for _ in range(5):
        t["native_us"].append(timed(g_nat, 50))
        t["remnant_us"].append(timed(g_rem, 50))
    summary = {
        "native_us_median": float(torch.tensor(t["native_us"]).median()),
        "remnant_us_median": float(torch.tensor(t["remnant_us"]).median()),
        "native_us_runs": t["native_us"], "remnant_us_runs": t["remnant_us"],
    }
    with open(os.path.join(OUT, "timing.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("TIMING", json.dumps(summary), flush=True)
    print("PARITY_MODE_DONE", flush=True)


if __name__ == "__main__":
    main()
