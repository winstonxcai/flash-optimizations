#!/usr/bin/env python3
# Fixed harness: CUDA-event timing of a captured graph containing 100 complete
# decode calls, elapsed GPU time / 100.  Both graphs warmed; 10 matched A/B/B/A
# rounds.  Prior wall-clock timing preserved as *_wallclock for comparison.
#
# Fixture (same as before): h64, batch 32, q1, swa64/c4512, seed 20260919.
# Each complete call = splitkv + combine (combine launched inside the C++ op).
import json
import os
import statistics
import sys
import time

import torch

_FIX = os.environ.get("PILOT_FIXTURE_DIR")
if _FIX:
    sys.path.insert(0, _FIX)

import harness_shim  # noqa: E402
flash_mla = harness_shim.install_flash_mla_fallback()
import lib  # noqa: E402

H = 64
B = 32
TOPK = 512
SEED = 20260919
CALLS = 100          # complete decode calls inside each graph
WARM = 10            # A/B/B/A rounds
OUT = os.environ.get("PILOT_OUT", "/tmp")


def run_native(case, meta):
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
    return flash_mla.get_mla_metadata()[0]


def main():
    torch.manual_seed(SEED)
    case = lib.make_remnant_case(H, TOPK, batch=B, device="cuda",
                                 unique_selections=False)
    torch.cuda.synchronize()          # materialize native workspace before timing

    meta_nat = new_meta()
    meta_rem = new_meta()

    # ---- (2) parity of the COMPLETE decode (native vs direct), finite+close ----
    n_out, n_lse = run_native(case, meta_nat)
    r_out, r_lse = run_remnant(case, meta_rem)
    torch.cuda.synchronize()
    assert torch.isfinite(n_out).all() and torch.isfinite(n_lse).all()
    assert torch.isfinite(r_out).all() and torch.isfinite(r_lse).all()
    out_mad = float((r_out.float() - n_out.float()).abs().max())
    lse_mad = float((r_lse.float() - n_lse.float()).abs().max())
    torch.testing.assert_close(r_out, n_out, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(r_lse, n_lse, atol=2.0e-2, rtol=2.0e-2)
    parity = {
        "h": H, "b": B, "topk": TOPK, "seed": SEED,
        "out_shape": list(n_out.shape), "lse_shape": list(n_lse.shape),
        "out_max_abs_diff": out_mad, "lse_max_abs_diff": lse_mad,
        "atol": 2.0e-2, "rtol": 2.0e-2, "status": "PASS",
    }
    with open(os.path.join(OUT, "mb_parity.json"), "w") as f:
        json.dump(parity, f, indent=2)
    print("PARITY_PASS", out_mad, lse_mad, flush=True)

    # ---- (3) two graphs, each 100 complete decode calls ----
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_native(case, meta_nat)
            run_remnant(case, meta_rem)
    torch.cuda.current_stream().wait_stream(s)

    g_nat = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_nat):
        for _ in range(CALLS):
            n_go, n_gl = run_native(case, meta_nat)
    g_dir = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_dir):
        for _ in range(CALLS):
            r_go, r_gl = run_remnant(case, meta_rem)

    # ---- warm both graphs + confirm graphs reproduce parity ----
    for _ in range(5):
        g_nat.replay()
        g_dir.replay()
    torch.cuda.synchronize()
    g_out_mad = float((r_go.float() - n_go.float()).abs().max())
    g_lse_mad = float((r_gl.float() - n_gl.float()).abs().max())
    assert torch.isfinite(n_go).all() and torch.isfinite(r_go).all()
    torch.testing.assert_close(r_go, n_go, atol=2.0e-2, rtol=2.0e-2)
    torch.testing.assert_close(r_gl, n_gl, atol=2.0e-2, rtol=2.0e-2)
    print("GRAPH_PARITY", g_out_mad, g_lse_mad, flush=True)

    # ---- (4) CUDA-event timing: elapsed GPU time of the graph / CALLS ----
    def ev_time(graph):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        e0.record()
        graph.replay()                # one replay = CALLS complete decodes
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) * 1e3 / CALLS    # ms -> us per call

    def wallclock_time(graph):
        # prior methodology, preserved for comparison: single replay, host clock
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e6

    nat_us = []       # cuda-event derived
    dir_us = []
    nat_wall = []     # preserved wall-clock
    dir_wall = []
    # 10 matched A/B/B/A rounds: nat, dir, dir, nat per round
    for _ in range(WARM):
        nat_us.append(ev_time(g_nat));  nat_wall.append(wallclock_time(g_nat))
        dir_us.append(ev_time(g_dir));  dir_wall.append(wallclock_time(g_dir))
        dir_us.append(ev_time(g_dir));  dir_wall.append(wallclock_time(g_dir))
        nat_us.append(ev_time(g_nat));  nat_wall.append(wallclock_time(g_nat))

    def stats(x):
        return {"median": statistics.median(x), "mean": statistics.mean(x),
                "min": min(x), "max": max(x), "raw": x}

    summary = {
        "fixture": {"H": H, "B": B, "TOPK": TOPK, "SEED": SEED,
                    "note": "h64/b32/q1/swa64-c4512/seed20260919"},
        "method": "cuda_event_graph_100calls_div100",
        "include_combine": True,
        "calls_per_graph": CALLS,
        "warm_rounds": WARM,
        "round_order": ["A/B/B/A native,direct,direct,native"],
        "profiler_attached": False,
        "graph_parity": {"out_max_abs_diff": g_out_mad,
                          "lse_max_abs_diff": g_lse_mad},
        "native_us": stats(nat_us),
        "direct_us": stats(dir_us),
        "native_us_wallclock": stats(nat_wall),
        "direct_us_wallclock": stats(dir_wall),
    }
    with open(os.path.join(OUT, "mb_timing.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("NATIVE_MEDIAN_US", statistics.median(nat_us), flush=True)
    print("DIRECT_MEDIAN_US", statistics.median(dir_us), flush=True)
    print("NATIVE_RAW", nat_us, flush=True)
    print("DIRECT_RAW", dir_us, flush=True)
    print("TIMING " + json.dumps(summary), flush=True)
    print("MB_DONE", flush=True)


if __name__ == "__main__":
    main()
