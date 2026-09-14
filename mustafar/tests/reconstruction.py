"""Reconstruction-only validation, changing-graph timing and profiler target.

Invoked by the unified speed entrypoint. Full kernels only; no phase ablations.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from .. import fused
from ..packed import NativeWorkspace, unpack_gather_native
from . import harness

MODES = ("fused", "generic", "optimized", "combined")


def workspace(fixture):
    return NativeWorkspace.allocate(fixture.batch, harness.TOPK, 16,
                                    fixture.freqs.device, with_dense=False)


def invocation(mode, fixture, out, current):
    physical, raw, lengths = current
    # Retain every tensor view in the callable for the entire graph lifetime.
    frequencies = torch.view_as_real(fixture.freqs)

    def call():
        fused.packed_to_native(
            *fixture.buffers, physical, raw, lengths, frequencies,
            out.native_bytes, out.page_size, out.bytes_per_page,
            optimized=mode != "fused",
            candidate=(
                "generic" if mode == "generic"
                else "combined" if mode == "combined"
                else None
            ),
        )
    return call


def fields(out, rows):
    values = out.native_bytes[:, :out.page_size * 576].reshape(-1, 576)[:rows]
    scales = out.native_bytes[:, out.page_size * 576:out.page_size * 584].reshape(-1, 8)[:rows]
    return values[:, :448], scales, values[:, 448:].contiguous().view(torch.bfloat16)


def assert_layout(actual, expected, current, fixture):
    codes, scales, tail = fields(actual, fixture.batch * harness.TOPK)
    ref_codes, ref_scales, ref_tail = fields(expected, fixture.batch * harness.TOPK)
    torch.testing.assert_close(codes, ref_codes, atol=0, rtol=0)
    torch.testing.assert_close(scales[:, :7], ref_scales[:, :7], atol=0, rtol=0)
    torch.testing.assert_close(tail, ref_tail, atol=harness.TAIL_ATOL, rtol=harness.TAIL_RTOL)
    physical, raw, lengths = current
    valid = ((physical >= 0) & (physical < fixture.batch * fixture.context_rows)
             & (raw >= 0) & (raw < fixture.context_rows)
             & (torch.arange(harness.TOPK, device=raw.device)[None] < lengths[:, None]))
    invalid = ~valid.flatten()
    assert not codes[invalid].count_nonzero().item()
    assert not scales[invalid].count_nonzero().item()
    assert not tail[invalid].count_nonzero().item()


def validation_sets(fixture):
    normal = fixture.selections[0]
    boundary = tuple(t.clone() for t in normal)
    boundary[1][:, :4] = torch.tensor([0, 15, 16, fixture.context_rows - 1], device=fixture.freqs.device)
    boundary[0][:, :4] = fixture.page_map.gather(1, boundary[1][:, :4].long() // 16) * 16 + boundary[1][:, :4] % 16
    duplicate = tuple(t.clone() for t in normal)
    duplicate[0][:, 1] = duplicate[0][:, 0]
    duplicate[1][:, 1] = duplicate[1][:, 0]
    invalid = tuple(t.clone() for t in boundary)
    invalid[0][:, 4:7] = torch.tensor([-1, fixture.batch * fixture.context_rows,
                                      fixture.batch * fixture.context_rows + 99], device=fixture.freqs.device)
    invalid[1][:, 7:9] = torch.tensor([-1, fixture.context_rows], device=fixture.freqs.device)
    partial = tuple(t.clone() for t in invalid)
    partial[2].fill_(17)
    partial[2][0] = 0
    empty = tuple(t.clone() for t in normal)
    empty[2].zero_()
    return (normal, boundary, duplicate, invalid, partial, empty)


def validate(fixture, modes=MODES, *, sanitizer=False):
    current = tuple(t.clone() for t in fixture.selections[0])
    reference = workspace(fixture)
    baseline = invocation("fused", fixture, reference, current)
    # A valid mapping must agree with the existing Triton native reconstruction.
    triton = NativeWorkspace.allocate(fixture.batch, harness.TOPK, 16,
                                     fixture.freqs.device, with_dense=True)
    with harness.leg_env("packed.native"):
        unpack_gather_native(fixture.buffers, *current, fixture.freqs, triton)
    baseline()
    assert_layout(reference, triton, current, fixture)
    cases = validation_sets(fixture)
    for mode in modes:
        out = workspace(fixture)
        call = invocation(mode, fixture, out, current)
        call()
        graph = None if sanitizer else harness.changing_graph(call, cases, current)
        for case in cases:
            for target, source in zip(current, case):
                target.copy_(source)
            reference.native_bytes.fill_(173)
            out.native_bytes.fill_(173)
            baseline()
            call() if graph is None else graph.replay()
            torch.cuda.synchronize()
            assert_layout(out, reference, current, fixture)
        if not sanitizer:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                call()
            torch.cuda.current_stream().wait_stream(stream)
            assert_layout(out, reference, current, fixture)
            before = torch.cuda.memory_allocated()
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before

        # The extension has separate int32/int64 launch paths. Exercise both
        # with the same valid mapping so a dtype change cannot hide a geometry
        # specialization bug.
        for index_dtype in (torch.int32, torch.int64):
            typed = tuple(t.to(index_dtype) for t in current)
            typed_out = workspace(fixture)
            typed_call = invocation(mode, fixture, typed_out, typed)
            typed_call()
            torch.cuda.synchronize()
            assert_layout(typed_out, reference, typed, fixture)

        if mode == "combined":
            # Candidate selection is intentionally strict, including for a
            # shape that would otherwise fit the workspace.
            bad = (current[0][:, :511], current[1][:, :511], current[2])
            try:
                invocation(mode, fixture, out, bad)()
            except ValueError as exc:
                assert "selected_k=512" in str(exc)
            else:
                raise AssertionError("geometry accepted selected_k != 512")

            empty = (
                current[0][:0], current[1][:0], current[2][:0]
            )
            empty_out = workspace(fixture)
            invocation(mode, fixture, empty_out, empty)()
            torch.cuda.synchronize()
        print(f"[reconstruction-validity] {mode}: {len(cases)} fixtures passed; graph={not sanitizer}", flush=True)


def provenance():
    root = Path(__file__).resolve().parents[2]
    sources = [root / name for name in (
        "mustafar/cuda/fused/fused.cu", "mustafar/tests/reconstruction.py",
        "mustafar/tests/harness.py", "mustafar/tests/speed.py")]
    info = {
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "seeds": {"data": 20260913, "pages": 20260914, "selections": 20260915},
        "page_size": 16, "threads_per_block": 128, "rows_per_warp": 1,
        "context_rows_per_request": 32768, "selected_k": 512,
    }
    for name, command in {
        "gpu_state": ["nvidia-smi", "--query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw", "--format=csv"],
        "compiler": ["nvcc", "--version"],
        "sglang_revision": ["git", "-C", "/sgl-workspace/sglang-lowrank", "rev-parse", "HEAD"],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        info[name] = result.stdout.strip()
    return info


def run(*, action="timing", batches=(15, 18, 21), modes=MODES, samples=500, rounds=3):
    if not torch.cuda.is_available():
        raise RuntimeError("reconstruction requires CUDA")
    if not modes or set(modes) - set(MODES):
        raise ValueError(f"modes must be a subset of {MODES}")
    if min(samples, rounds) <= 0:
        raise ValueError("samples and rounds must be positive")
    result = {"provenance": provenance(), "action": action, "records": []}
    if action in ("validate", "sanitizer"):
        fixture = harness.reconstruction_fixture(2, "cuda", count=2)
        validate(fixture, modes, sanitizer=action == "sanitizer")
        result["validity"] = "passed"
    else:
        for batch in batches:
            fixture = harness.reconstruction_fixture(batch, "cuda", count=32)
            if action == "timing":
                validate(fixture, modes)
            current = tuple(t.clone() for t in fixture.selections[0])
            # Every graph retains the callable/output owner, not just the input tensors.
            owned = []
            for mode in modes:
                out = workspace(fixture)
                call = invocation(mode, fixture, out, current)
                graph = harness.changing_graph(call, fixture.selections, current)
                owned.append((mode, out, call, graph))
            if action == "profile":
                assert len(owned) == 1
                graph = owned[0][-1]
                for _ in range(20):
                    graph.replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("reconstruction-profile")
                for _ in range(samples):
                    graph.replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            else:
                for round_id in range(rounds):
                    order = owned[round_id % len(owned):] + owned[:round_id % len(owned)]
                    for mode, out, call, graph in order:
                        timing = harness.timed_changing_graph(
                            graph, fixture.selections, current, warmup=20, repeats=samples)
                        record = {"batch": batch, "mode": mode, "round": round_id + 1, **timing}
                        result["records"].append(record)
                        print(json.dumps(record), flush=True)
            del owned, graph, call, out, fixture
    directory = Path(os.environ.get("MUSTAFAR_RESULTS_DIR", "."))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"reconstruction-{action}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
