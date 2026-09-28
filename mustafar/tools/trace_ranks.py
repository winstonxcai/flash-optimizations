#!/usr/bin/env python3
"""Per-rank analysis of sglang torch-profiler traces (mustafar TP4 decode work).

Why this exists
---------------
Three traps have each already produced a wrong published number in this program.
They are cheap to avoid and expensive to rediscover, so they live here.

1. MATCH BY SUBSTRING, NEVER BY EXACT NAME.

   The profiler emits mangled C++ template names:

     void sglang::all_reduce_kernel<sglang::AllReducePushImpl<__nv_bfloat16, ...>>
     _ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalINS1_17GroupProblemShape...

   An exact-name match against e.g. "radixSortKVInPlace" or "cutlass13::device_kernel"
   returns nothing at all -- which reads as "this kernel is absent" rather than
   "my matcher is broken". That exact bug has twice been mistaken for a real
   finding here. Every lookup in this file is a substring test.

2. SUM-OF-DURATIONS IS NOT WALL TIME.

   Decode is gap-free. In one measured trace: 13050 inter-kernel gaps of 0-9 us
   and 4 gaps >50 us, totalling 8 ms in a 253 ms window. The per-step wall clock
   is therefore the trace WINDOW (first kernel start -> last kernel end), not the
   summed durations. Summing overcounts by ~16%, because the all-reduce overlaps
   compute and gets counted alongside it. Both numbers are printed; they are
   never conflated.

3. NCCL DURATIONS ARE SOMETIMES GARBAGE.

   The 2026-09-15 `native` (non-graph) trace records all_reduce at 1467 us on
   TP0 and 5.2 us on TP3 -- same kernel, same 1740 launches, same trace. That is
   physically impossible and it corrupted the whole per-rank picture. `summary`
   prints the per-rank all-reduce spread so the corruption is visible before any
   derived number is quoted.

Trace format
------------
Chrome-trace JSON, gzipped, ~7 MB (native) to ~160 MB (packed) per rank, and NOT
valid JSON: torch elides braces inside some `args` objects, so json.loads fails.
Parsing is therefore a streaming regex over `"cat": "kernel"` events, chunked so
the large traces do not have to fit in memory.

Each kernel event is `{"ph":"X","cat":"kernel","name":"<possibly huge mangled
name>", "ts":..., "dur":..., "args":{...}}`. The segment belonging to one event
is taken as the span from the end of its name to the start of the next kernel
event's name, which is robust to both the very long template names and the
variable-size `args` blobs in between.

Usage
-----
  trace_ranks.py summary <trace...> [--steps N] [--label NAME]
  trace_ranks.py kernels <trace> [--top N] [--pattern SUBSTR]
  trace_ranks.py diff    <traceA> <traceB> [--top N]

  <trace> may be a path, a glob, or a directory (expanded recursively for
  *trace.json.gz). `summary` groups by the `-TP-<rank>-` field of the filename
  and reports the rank spread across the set.

One event is dropped per file: an event is only decoded once its successor's
name is visible, so the final event in the stream has no known end and is
skipped. That is 1 of ~40,000 and is immaterial to every aggregate here.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import re
import sys
from collections import Counter

CHUNK = 1 << 22
TAIL = 1 << 16
MAX_SEG = 1 << 16  # cap on the name->next-event span we will scan for ts/dur

NAME_RE = re.compile(rb'"cat":\s*"kernel",\s*"name":\s*"([^"]*)"')
TS_RE = re.compile(rb'"ts":\s*([0-9.eE+-]+)')
DUR_RE = re.compile(rb'"dur":\s*([0-9.eE+-]+)')

# The TP4 collective. Used for the capture-sanity check; matched as a substring.
ALLREDUCE = "all_reduce_kernel"


def iter_kernels(path):
    """Yield (ts, dur, name) for every kernel event in a trace. Streaming.

    ts/dur are in microseconds (CUPTI clock, per-process: absolute values are
    NOT comparable across ranks, only intervals within a trace are).
    """
    tail = b""
    base = 0
    last_abs = -1
    with gzip.open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK)
            if not chunk:
                break
            buf = tail + chunk
            bufbase = base
            matches = list(NAME_RE.finditer(buf))
            # Skip matches already emitted (they reappear inside the tail) and
            # the final match, whose end is unknown until the next chunk lands.
            for i in range(len(matches) - 1):
                m = matches[i]
                abs_start = bufbase + m.start()
                if abs_start <= last_abs:
                    continue
                seg = buf[m.end() : min(matches[i + 1].start(), m.end() + MAX_SEG)]
                dur = DUR_RE.search(seg)
                if not dur:
                    continue
                ts = TS_RE.search(seg)
                last_abs = abs_start
                yield (
                    float(ts.group(1)) if ts else 0.0,
                    float(dur.group(1)),
                    m.group(1).decode("utf-8", "replace"),
                )
            tail = buf[-TAIL:]
            base = bufbase + len(buf) - len(tail)


def scan(path):
    """Aggregate one trace: window, sum-of-durations, per-name totals/counts."""
    n = 0
    total = 0.0
    t_min = None
    t_end_max = None
    per_name = Counter()
    per_count = Counter()
    for ts, dur, name in iter_kernels(path):
        n += 1
        total += dur
        if t_min is None or ts < t_min:
            t_min = ts
        end = ts + dur
        if t_end_max is None or end > t_end_max:
            t_end_max = end
        per_name[name[:120]] += dur
        per_count[name[:120]] += 1
    window = (t_end_max - t_min) if (t_min is not None and t_end_max is not None) else 0.0
    return {
        "n": n,
        "total": total,
        "window": window,
        "names": per_name,
        "counts": per_count,
        "path": path,
    }


def rank_of(path):
    """TP rank from a `...-TP-<n>-STAGE.trace.json.gz` filename, else '?'."""
    m = re.search(r"-TP-(\d+)-", os.path.basename(path))
    return m.group(1) if m else "?"


def stage_of(path):
    """Stage from a `...-TP-<n>-STAGE.trace.json.gz` filename, else '?'."""
    m = re.search(r"-TP-\d+-([A-Za-z]+)\.trace\.json\.gz$", os.path.basename(path))
    return m.group(1) if m else "?"


def group_of(path):
    """Trace-set key: the filename with the rank field collapsed.

    Every rank of one profile point shares this key, so it is the unit over
    which a rank spread is meaningful. Grouping by anything coarser (e.g. by
    directory) would pool DECODE with EXTEND and produce a spread of ~40x that
    measures the prefill/decode difference, not rank skew.
    """
    return re.sub(r"-TP-\d+-", "-TP-*-", os.path.basename(path))


def allreduce_stats(res):
    """(total ms, launches) for the all-reduce, by substring match."""
    tot = 0.0
    cnt = 0
    for name, dur in res["names"].items():
        if ALLREDUCE in name:
            tot += dur
            cnt += res["counts"][name]
    return tot / 1000.0, cnt


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "**", "*trace.json.gz"), recursive=True)))
        elif any(c in p for c in "*?["):
            out.extend(sorted(glob.glob(p, recursive=True)))
        else:
            out.append(p)
    return [p for p in out if os.path.exists(p)]


def cmd_summary(args):
    files = expand(args.traces)
    if not files:
        sys.exit("trace_ranks: no traces matched")
    steps = args.steps
    print(f"{'trace':46s} {'rank':>4s} {'stage':>7s} {'kernels':>8s} {'window_ms':>10s} "
          f"{'ms/step':>8s} {'sum_ms':>9s} {'sum/win':>7s} {'allred_ms':>9s} {'ar_n':>5s}")
    groups = {}
    for f in files:
        res = scan(f)
        ar, arn = allreduce_stats(res)
        w = res["window"] / 1000.0
        groups.setdefault(group_of(f), []).append(w)
        print(f"{os.path.basename(f)[:46]:46s} {rank_of(f):>4s} {stage_of(f):>7s} {res['n']:8d} "
              f"{w:10.2f} {res['window'] / steps / 1000.0:8.3f} "
              f"{res['total'] / 1000.0:9.2f} {res['total'] / res['window'] if res['window'] else 0:7.3f} "
              f"{ar:9.2f} {arn:5d}")
    print()
    for key, windows in groups.items():
        if len(windows) < 2:
            continue
        lo, hi = min(windows), max(windows)
        print(f"{key}: spread {hi / lo:.3f}x  ({lo:.2f} - {hi:.2f} ms, "
              f"max-min {hi - lo:.2f} ms = {(hi - lo) / steps * 1000:.0f} us/step)")
    print("\nA spread well above ~1.02x on ranks that NCCL holds in lockstep is a capture\n"
          "artifact, not a workload property -- check the allred_ms column for that group.")


def cmd_kernels(args):
    files = expand([args.trace])
    if not files:
        sys.exit("trace_ranks: no trace matched")
    res = scan(files[0])
    items = res["names"].most_common()
    if args.pattern:
        items = [(k, v) for k, v in items if args.pattern in k]
    print(f"# {os.path.basename(files[0])}")
    print(f"# kernels={res['n']}  window={res['window'] / 1000:.2f} ms  "
          f"sum={res['total'] / 1000:.2f} ms  (sum/window={res['total'] / res['window']:.3f})")
    print(f"{'total_ms':>10s} {'n':>7s} {'us/call':>9s}  kernel")
    for name, tot in items[: args.top]:
        c = res["counts"][name]
        print(f"{tot / 1000:10.3f} {c:7d} {tot / c:9.2f}  {name}")


def cmd_diff(args):
    fa, fb = expand([args.trace_a]), expand([args.trace_b])
    if not fa or not fb:
        sys.exit("trace_ranks: both traces must exist")
    a, b = scan(fa[0]), scan(fb[0])
    delta = b["total"] - a["total"]
    print(f"# A = {os.path.basename(fa[0])}")
    print(f"# B = {os.path.basename(fb[0])}")
    print(f"# A: kernels={a['n']} sum={a['total'] / 1000:.1f} ms window={a['window'] / 1000:.1f} ms")
    print(f"# B: kernels={b['n']} sum={b['total'] / 1000:.1f} ms window={b['window'] / 1000:.1f} ms")
    print(f"# B-A sum delta = {delta / 1000:.1f} ms\n")
    rows = []
    for name in set(a["names"]) | set(b["names"]):
        rows.append((b["names"][name] - a["names"][name], name))
    rows.sort(reverse=True)
    print(f"{'delta_ms':>9s} {'A_ms':>9s} {'B_ms':>9s} {'n_A':>6s} {'n_B':>6s}  kernel")
    for d, name in rows[: args.top]:
        print(f"{d / 1000:9.3f} {a['names'][name] / 1000:9.3f} {b['names'][name] / 1000:9.3f} "
              f"{a['counts'][name]:6d} {b['counts'][name]:6d}  {name}")
    print("\n# B got faster:")
    for d, name in rows[-min(args.top, len(rows)) :]:
        print(f"{d / 1000:9.3f} {a['names'][name] / 1000:9.3f} {b['names'][name] / 1000:9.3f} "
              f"{a['counts'][name]:6d} {b['counts'][name]:6d}  {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("summary", help="per-rank window / sum / all-reduce sanity")
    p.add_argument("traces", nargs="+")
    p.add_argument("--steps", type=int, default=20, help="profiled steps (default 20)")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("kernels", help="per-kernel-name breakdown of one trace")
    p.add_argument("trace")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--pattern", help="substring filter on kernel name")
    p.set_defaults(func=cmd_kernels)

    p = sub.add_parser("diff", help="per-kernel-name difference between two traces")
    p.add_argument("trace_a")
    p.add_argument("trace_b")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_diff)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
