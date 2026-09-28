#!/usr/bin/env python3
"""Pick the longest LongSWE-Bench replay requests into data/long-prompts.jsonl.

The stability ladder has to be run against prompts that are actually long --
a 2k-token prompt cannot tell you whether a basis degrades at 32k decode steps,
because it never gets there. The replay corpus has 4916 requests with a wide
spread of recorded prompt lengths, so this scans the corpus's `usage.prompt_tokens`
and writes the N longest, largest first, as a checked-in list:

    {"rank": 1, "id": "...", "prompt_tokens": 137904, "path": "/abs/path.json"}

`stability.sh` resolves a rank to a path and hands it to
`analysis/stability.py`, which renders the chat messages to plain text.

The picks are spread across the longest `--pool` requests rather than being the
strict top N, because the corpus piles up at the ceiling (median 146k, max 257k
tokens) and the top of that heap is a tight cluster within 2% of the maximum.
A spread keeps every entry long while leaving the operator a real choice of
prompt size -- useful when the largest prompt does not fit the running pool.

Host-side and stdlib-only; reads the corpus, writes one small file. No GPU, no
container, no server. Re-run it only if the corpus itself changes (the checked-in
list is what the runs reference, so a stable list means comparable runs).

  python3 starkv/scripts/local/long-prompts.py [--n 8] [--out PATH]
"""

import argparse
import glob
import json
import os
import sys

DEFAULT_DATASET = "/home/jovyan/wenyuhong/benchmarks/datasets/h20-dsv4pro/longcodebench_openai"
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "long-prompts.jsonl"
)


def scan(dataset: str):
    """Every replay request as {id, prompt_tokens, path}, longest first.

    `usage.prompt_tokens` is the *recorded* length from the original session --
    the right thing to rank by, since it is the real prompt size the model saw,
    not a re-tokenization of a re-rendered approximation.
    """
    rows = []
    for path in glob.glob(os.path.join(dataset, "*.json")):
        try:
            with open(path) as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if "messages" not in rec:
            continue
        rows.append({
            "id": rec.get("id") or os.path.basename(path)[:-5],
            "prompt_tokens": int((rec.get("usage") or {}).get("prompt_tokens") or 0),
            "path": os.path.abspath(path),
        })
    rows.sort(key=lambda r: (-r["prompt_tokens"], r["id"]))
    return rows


def spread(rows, n: int, pool: int):
    """`n` entries chosen evenly from the longest `pool`, largest first."""
    pool = min(pool, len(rows))
    if n >= pool:
        return rows[:pool]
    # Evenly spaced indices, endpoints included: i*(pool-1)/(n-1).
    idx = sorted({round(i * (pool - 1) / (n - 1)) for i in range(n)})
    return [rows[i] for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--n", type=int, default=8, help="how many prompts to keep")
    ap.add_argument("--pool", type=int, default=1229,
                    help="spread the picks across the longest POOL requests")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not os.path.isdir(args.dataset):
        print(f"FATAL: dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    rows = scan(args.dataset)
    if not rows:
        print(f"FATAL: no replay requests under {args.dataset}", file=sys.stderr)
        return 1

    keep = spread(rows, args.n, args.pool)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for i, r in enumerate(keep, 1):
            f.write(json.dumps({"rank": i, **r}) + "\n")

    print(f"scanned {len(rows)} requests; kept {len(keep)} spread across the "
          f"longest {min(args.pool, len(rows))} -> {args.out}")
    for i, r in enumerate(keep, 1):
        print(f"  {i:2d}. {r['prompt_tokens']:>7} tok  {r['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
