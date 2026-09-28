#!/usr/bin/env python3
"""Diff two qa_probe.py outputs and print a verdict per prompt.

Host-side (stdlib only). The question this answers is the one a throughput number
cannot: did the xkv leg stay a *language model*, or did the low-rank KV path
degrade into repetition / NaN / empty output while still finishing requests on
time. Greedy sampling on both legs means identical prompts should produce
near-identical completions; exact equality is not expected (different attention
numerics), so this reports overlap plus the degenerate-output checks and leaves
the judgement to the report.

    qa_compare.py native.jsonl xkv.jsonl
"""

import json
import sys


def load(path):
    with open(path) as stream:
        return {rec["prompt"]: rec for rec in map(json.loads, stream) if rec}


def degenerate(text):
    """Signals that mean "not language" rather than "wrong answer"."""
    if not text.strip():
        return "empty"
    missing = text.count("�")
    if missing:
        return f"{missing} replacement chars"
    stripped = text.strip()
    # A completion that is one short token repeated is a decode defect, not a
    # style choice. Look at the tail, which is where a broken KV path collapses.
    tail = stripped[-64:].split()
    if len(tail) >= 8 and len(set(tail)) <= 2:
        return f"collapsed tail ({len(set(tail))} distinct tokens)"
    if "nan" in stripped.lower().split():
        return "literal nan token"
    # Repetition loops. The word-level check above misses a loop whose period is
    # wider than a few tokens (a repeated formula, a repeated clause), so this
    # measures how much of the text is a rerun of itself at character
    # granularity. Ordinary prose is ~0 here; a loop climbs steeply.
    if len(stripped) >= 96:
        window = 16
        grams = [stripped[i:i + window] for i in range(len(stripped) - window + 1)]
        repeated = 1.0 - len(set(grams)) / len(grams)
        if repeated > 0.30:
            return f"repetition loop ({repeated:.0%} repeated 16-grams)"
    return None


def words(text):
    return [w.lower() for w in text.split()]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    left, right = load(sys.argv[1]), load(sys.argv[2])
    problems = 0
    for name in left:
        a, b = left[name], right.get(name)
        if b is None:
            print(f"{name}: MISSING from {sys.argv[2]}")
            problems += 1
            continue
        for tag, rec in ((sys.argv[1], a), (sys.argv[2], b)):
            if "error" in rec:
                print(f"{name}: ERROR in {tag}: {rec['error']}")
                problems += 1
        if "error" in a or "error" in b:
            continue
        for tag, rec in ((sys.argv[1], a), (sys.argv[2], b)):
            why = degenerate(rec["text"])
            if why:
                print(f"{name}: DEGENERATE in {tag}: {why}")
                problems += 1
        wa, wb = words(a["text"]), words(b["text"])
        # Length-insensitive agreement: how much of the shorter completion is
        # reproduced by the longer one. Greedy siblings land high; a leg that has
        # lost the plot lands near zero.
        smaller, larger = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
        shared = len(set(smaller) & set(larger)) / max(1, len(set(smaller)))
        flag = "" if shared >= 0.5 else "  <-- LOW AGREEMENT"
        print(f"{name}: tokens {len(wa)} vs {len(wb)}, vocab overlap {shared:.2f}{flag}")
    print(f"\n{problems} problem(s) across {len(left)} prompt(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
