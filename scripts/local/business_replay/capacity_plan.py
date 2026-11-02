"""Choose matched replay concurrencies from a KV pool size and a dataset manifest.

Author: Winston Cai.

Stdlib only; starts no server. The fixed-active-user-pool scheduler gives each
of C slots one session at a time and sends that session's requests serially, so
the live context at a tick is the sum of each active session's current request
size. This replays the scheduler's session order offline and reports the largest
such sum, which must fit the global KV pool for the run to avoid eviction.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

# 900 s at the ~13.4 s per request each slot sustained in the C24 records.
DEFAULT_TICKS = 66
HIGH_UTILIZATION = 0.9
LOW_RATIO = 0.75
MIN_HIGH = 8


def session_sizes(manifest: dict) -> list[list[int]]:
    """Per-session request sizes (prompt + completion tokens) in scheduler order.

    Mirrors runner.session_groups: requests ordered by (created, id) inside a
    session, sessions ordered by their first request.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest["records"]:
        groups[entry["user_id"]].append(entry)
    ordered = []
    for user, group in groups.items():
        group.sort(key=lambda e: (e["created"], e["request_id"]))
        ordered.append((group[0]["created"], group[0]["request_id"], user, group))
    ordered.sort(key=lambda item: item[:3])
    return [
        [(e["expected_prompt_tokens"] or 0) + (e["expected_completion_tokens"] or 0) for e in group]
        for *_, group in ordered
    ]


def peak_working_set(sessions: list[list[int]], concurrency: int, ticks: int = DEFAULT_TICKS) -> int:
    """Largest sum of live request sizes over `ticks` rounds of one request per slot."""
    if concurrency < 1 or ticks < 1:
        raise ValueError("concurrency and ticks must be positive")
    pending = deque(sessions)
    slots: list[list] = [[pending.popleft(), 0] for _ in range(min(concurrency, len(pending)))]
    peak = 0
    for _ in range(ticks):
        peak = max(peak, sum(session[index] for session, index in slots if session is not None))
        for slot in slots:
            session, index = slot
            if session is None:
                continue
            slot[1] += 1
            if slot[1] >= len(session):
                slot[0], slot[1] = (pending.popleft(), 0) if pending else (None, 0)
    return peak


def plan_concurrency(
    sessions: list[list[int]], pool_tokens: int, max_running: int, ticks: int = DEFAULT_TICKS
) -> dict:
    """C_high: largest C <= max_running whose peak is <= 90% of the pool; C_low: 75% of it."""
    budget = pool_tokens * HIGH_UTILIZATION
    high = 0
    for concurrency in range(1, max_running + 1):
        if peak_working_set(sessions, concurrency, ticks) <= budget:
            high = concurrency
    if high < MIN_HIGH:
        raise ValueError(
            f"pool of {pool_tokens} tokens fits only C={high} (minimum {MIN_HIGH}) "
            f"with max_running={max_running}"
        )
    low = max(1, round(high * LOW_RATIO))
    return {
        "c_high": high,
        "c_low": low,
        "pool_tokens": pool_tokens,
        "max_running": max_running,
        "ticks": ticks,
        "peak_high": peak_working_set(sessions, high, ticks),
        "peak_low": peak_working_set(sessions, low, ticks),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--pool-tokens", type=int, required=True)
    parser.add_argument("--max-running", type=int, required=True)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    args = parser.parse_args(argv)
    sessions = session_sizes(json.loads(args.manifest.read_text()))
    print(json.dumps(plan_concurrency(sessions, args.pool_tokens, args.max_running, args.ticks), indent=2))


if __name__ == "__main__":
    main()
