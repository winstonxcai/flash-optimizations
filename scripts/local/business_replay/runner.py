#!/usr/bin/env python3
"""Replay timestamped OpenAI chat records through a project adapter.

Author: Winston Cai. (Optional request-slot replay integration.)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

if __package__:
    from . import request_slots
else:
    import request_slots

SCHEMA_VERSION = 1
PROTOCOL = "business-user-replay-v2"
SCHEDULER = "fixed-active-user-pool"
AUDIT_LEVELS = ("dev", "candidate", "official")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(value) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(cwd: Path, *args: str) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return 127, ""
    return completed.returncode, completed.stdout.decode("utf-8", errors="replace").strip()


def source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    identity: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }
    returncode, root_text = _git(resolved.parent, "rev-parse", "--show-toplevel")
    if returncode != 0:
        identity["git"] = {"available": False}
        return identity

    root = Path(root_text)
    relative = str(resolved.relative_to(root))
    _, commit = _git(root, "rev-parse", "HEAD")
    branch_code, branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    _, worktree_status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    tracked_code, _ = _git(root, "ls-files", "--error-unmatch", "--", relative)
    _, file_status = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", relative)
    tracked = tracked_code == 0
    identity["git"] = {
        "available": True,
        "repository_root": str(root),
        "commit": commit or None,
        "branch": branch if branch_code == 0 else None,
        "worktree_state": "dirty" if worktree_status else "clean",
        "tracked": tracked,
        "matches_head": tracked and not file_status,
    }
    return identity


def build_audit_metadata(
    level: str,
    runner_path: Path,
    adapter_path: Path,
    profile_name: str | None = None,
    profile_sha256: str | None = None,
    scheduler_path: Path | None = None,
) -> dict[str, Any]:
    if level not in AUDIT_LEVELS:
        raise ValueError(f"unsupported audit level: {level}")
    if profile_sha256 is not None and (
        len(profile_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in profile_sha256)
    ):
        raise ValueError("profile SHA-256 must contain exactly 64 hexadecimal characters")

    sources = {
        "runner": source_identity(runner_path),
        "adapter": source_identity(adapter_path),
    }
    if scheduler_path is not None:
        sources["scheduler"] = source_identity(scheduler_path)
    warnings: list[str] = []
    official_errors: list[str] = []
    for role, source in sources.items():
        git = source["git"]
        if not git["available"]:
            warnings.append(f"{role}_git_metadata_unavailable")
            official_errors.append(f"{role} is not in a Git worktree")
            continue
        if git["worktree_state"] != "clean":
            warnings.append(f"{role}_repository_dirty")
            official_errors.append(f"{role} repository is dirty")
        if not git["tracked"] or not git["matches_head"]:
            warnings.append(f"{role}_does_not_match_head")
            official_errors.append(f"{role} does not match HEAD")
        if not git["commit"]:
            official_errors.append(f"{role} Git commit is unavailable")

    profile_complete = profile_name is not None and profile_sha256 is not None
    profile = None
    if profile_name is not None or profile_sha256 is not None:
        profile = {"name": profile_name, "sha256": profile_sha256}
    if not profile_complete and level != "dev":
        warnings.append("profile_identity_not_provided")
        official_errors.append("profile name and SHA-256 are required")

    if level == "official" and official_errors:
        raise RuntimeError(
            "official audit requirements not met: " + "; ".join(sorted(set(official_errors)))
        )
    return {
        "schema_version": 1,
        "level": level,
        "comparison_eligibility": (
            "official" if level == "official" else "candidate" if level == "candidate"
            else "development_only"
        ),
        "official_comparison_eligible": level == "official",
        "sources": sources,
        "profile": profile,
        "warnings": sorted(set(warnings)),
    }


@dataclass(frozen=True)
class Record:
    request_id: str
    user_id: str
    created: float
    path: Path
    source_sha256: str
    expected_prompt_tokens: int | None
    expected_completion_tokens: int | None
    planned_offset_s: float = 0.0
    source_size: int | None = None
    source_mtime_ns: int | None = None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def load_dataset(root: Path, limit: int) -> tuple[list[Record], int]:
    paths = sorted(root.glob("*.json"))
    if not paths:
        raise RuntimeError(f"no JSON files found in {root}")
    records: list[Record] = []
    for path in paths:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"dataset record changed while being audited: {path}")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid JSON dataset record: {path}") from exc
        request_id, user_id, created, messages = (
            document.get("id"), document.get("token_name"), document.get("created"), document.get("messages")
        )
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError(f"dataset record has no valid id: {path}")
        if not isinstance(user_id, str) or not user_id:
            raise RuntimeError(f"dataset record has no valid token_name: {path}")
        if not isinstance(created, (int, float)) or isinstance(created, bool):
            raise RuntimeError(f"dataset record has no valid created timestamp: {path}")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError(f"dataset record has no messages: {path}")
        usage = document.get("usage") if isinstance(document.get("usage"), dict) else {}
        records.append(Record(
            request_id, user_id, float(created), path, hashlib.sha256(raw).hexdigest(),
            _optional_int(usage.get("prompt_tokens")), _optional_int(usage.get("completion_tokens")),
            source_size=after.st_size, source_mtime_ns=after.st_mtime_ns,
        ))
    records.sort(key=lambda item: (item.created, item.request_id, str(item.path)))
    if len({record.request_id for record in records}) != len(records):
        raise RuntimeError("dataset contains duplicate request ids")
    if limit > 0:
        records = records[:limit]
    if not records:
        raise RuntimeError("dataset selection is empty")
    return records, len(paths)


def load_dataset_manifest(
    path: Path, root: Path, limit: int, *, verify_snapshot: bool = False
) -> tuple[list[Record], int]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("protocol") != PROTOCOL
    ):
        raise RuntimeError(f"incompatible dataset manifest: {path}")
    entries = document.get("records")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"dataset manifest has no records: {path}")
    resolved_root = root.resolve()
    records: list[Record] = []
    for entry in entries:
        relative_path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError(f"dataset manifest has an invalid record path: {path}")
        source_path = (root / relative_path).resolve()
        if not source_path.is_relative_to(resolved_root):
            raise RuntimeError(f"dataset manifest record escapes dataset root: {relative_path}")
        source_size = entry.get("source_size")
        source_mtime_ns = entry.get("source_mtime_ns")
        if verify_snapshot:
            if not isinstance(source_size, int) or not isinstance(source_mtime_ns, int):
                raise RuntimeError(f"dataset manifest lacks snapshot metadata: {relative_path}")
            try:
                current = source_path.stat()
            except FileNotFoundError as exc:
                raise RuntimeError(f"dataset manifest record is missing: {relative_path}") from exc
            if (current.st_size, current.st_mtime_ns) != (source_size, source_mtime_ns):
                raise RuntimeError(f"dataset snapshot changed: {relative_path}")
        records.append(Record(
            str(entry["request_id"]), str(entry["user_id"]), float(entry["created"]),
            source_path, str(entry["source_sha256"]),
            _optional_int(entry.get("expected_prompt_tokens")),
            _optional_int(entry.get("expected_completion_tokens")),
            planned_offset_s=float(entry.get("planned_offset_s", 0.0)),
            source_size=source_size if isinstance(source_size, int) else None,
            source_mtime_ns=source_mtime_ns if isinstance(source_mtime_ns, int) else None,
        ))
    if len({record.request_id for record in records}) != len(records):
        raise RuntimeError("dataset manifest contains duplicate request ids")
    discovered = document.get("files_discovered")
    if verify_snapshot:
        current_paths = {item.name for item in root.glob("*.json")}
        manifest_paths = {str(record.path.resolve().relative_to(resolved_root))
                          for record in records}
        if current_paths != manifest_paths:
            raise RuntimeError("dataset snapshot file set changed")
    if limit > 0:
        records = records[:limit]
    return records, int(discovered) if isinstance(discovered, int) else len(entries)


def dataset_entries(records: list[Record], root: Path) -> list[dict[str, Any]]:
    resolved_root = root.resolve()
    entries = []
    for record in records:
        entry = {
            "request_id": record.request_id,
            "user_id": record.user_id,
            "created": record.created,
            "path": str(record.path.resolve().relative_to(resolved_root)),
            "source_sha256": record.source_sha256,
            "expected_prompt_tokens": record.expected_prompt_tokens,
            "expected_completion_tokens": record.expected_completion_tokens,
            "planned_offset_s": record.planned_offset_s,
        }
        if record.source_size is not None and record.source_mtime_ns is not None:
            entry["source_size"] = record.source_size
            entry["source_mtime_ns"] = record.source_mtime_ns
        entries.append(entry)
    return entries


def workload_sha256(entries: list[dict[str, Any]]) -> str:
    identity = [{key: value for key, value in entry.items()
                 if key not in ("source_size", "source_mtime_ns")} for entry in entries]
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def assign_offsets(records: list[Record], mode: str, scale: float, max_gap: float | None) -> list[Record]:
    if mode == "immediate":
        return [replace(record, planned_offset_s=0.0) for record in records]
    if mode != "replay" or not math.isfinite(scale) or scale <= 0:
        raise ValueError("arrival mode must be immediate or replay with positive time scale")
    if max_gap is not None and (not math.isfinite(max_gap) or max_gap < 0):
        raise ValueError("max gap must be finite and nonnegative")
    offsets: dict[float, float] = {}
    elapsed, previous = 0.0, None
    for created in sorted({record.created for record in records}):
        if previous is not None:
            gap = max(0.0, created - previous) / scale
            elapsed += min(gap, max_gap) if max_gap is not None else gap
        offsets[created] = elapsed
        previous = created
    return [replace(record, planned_offset_s=offsets[record.created]) for record in records]


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("business_replay_project_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "send_request", None)):
        raise RuntimeError("adapter must define send_request(record, options)")
    return module


def load_warmup_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("kind") != "perf_benchmark_warmup"
    ):
        raise RuntimeError(f"incompatible warmup manifest: {path}")
    requests = document.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"warmup manifest has no requests: {path}")
    for entry in requests:
        if not isinstance(entry, dict):
            raise RuntimeError(f"warmup manifest entry is not an object: {path}")
        prompt = entry.get("prompt")
        output_tokens = entry.get("output_tokens")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError(f"warmup manifest entry has no prompt: {path}")
        if (
            not isinstance(output_tokens, int)
            or isinstance(output_tokens, bool)
            or output_tokens < 1
        ):
            raise RuntimeError(f"warmup manifest entry has invalid output_tokens: {path}")
    return document, requests


def warmup_copies(replicas: int) -> int:
    """Match perf-benchmark: replicas>1 is multiplied by four."""
    if replicas < 0:
        raise ValueError("warmup replicas must be nonnegative")
    return replicas * 4 if replicas > 1 else replicas


def warmup_plan(requests: list[dict[str, Any]], replicas: int) -> list[tuple[dict[str, Any], int]]:
    copies = warmup_copies(replicas)
    if copies == 0:
        return []
    return [(entry, copies) for entry in requests]


async def run_warmup(
    plan: list[tuple[dict[str, Any], int]],
    sender,
    path: Path,
    lock: threading.Lock,
) -> list[dict[str, Any]]:
    """Run the fixed warmup prompts before the measured replay.

    The upstream perf-benchmark runner sends each warmup prompt ``pod_num``
    times and waits for that round before moving to the next prompt.  Warmup
    results are written to ``warmup.jsonl`` and are never folded into the
    measured summary.
    """
    results: list[dict[str, Any]] = []
    for entry, copies in plan:
        batch = await asyncio.gather(
            *(asyncio.to_thread(sender, entry, index) for index in range(copies))
        )
        for item in batch:
            append_jsonl(path, item, lock)
            results.append(item)
        failed = [item for item in batch if not item.get("valid")]
        if failed:
            errors = sorted({error for item in failed for error in item.get("errors", [])})
            raise RuntimeError(f"warmup request failed: {errors}")
    return results


def session_groups(records: list[Record]) -> list[tuple[str, list[Record]]]:
    """One shared definition of session and turn ordering for both policies."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.user_id].append(record)
    for group in groups.values():
        group.sort(key=lambda item: (item.created, item.request_id))
    return sorted(
        groups.items(),
        key=lambda item: (item[1][0].created, item[1][0].request_id, item[0]),
    )


async def run_replay(records: list[Record], max_concurrency: int, duration: float | None, sender, paths):
    ordered_groups = session_groups(records)
    pending = deque(ordered_groups)
    lock = threading.Lock()
    results, activity = [], []
    active_requests = 0
    active_users = 0
    started = time.perf_counter()
    deadline = started + duration if duration is not None else None

    def record_activity(event: dict[str, Any]) -> None:
        activity.append(event)
        append_jsonl(paths[2], event, lock)

    async def wait_until(offset: float) -> bool:
        while True:
            current = time.perf_counter()
            if deadline is not None and current >= deadline:
                return False
            delay = started + offset - current
            if delay <= 0:
                return True
            await asyncio.sleep(min(delay, max(0.0, deadline - current)) if deadline is not None else delay)

    async def slot_worker(slot_id: int) -> None:
        nonlocal active_requests, active_users
        while pending:
            if deadline is not None and time.perf_counter() >= deadline:
                return
            user_id, group = pending.popleft()
            active_users += 1
            record_activity({
                "event": "user_activate",
                "slot_id": slot_id,
                "user_id": user_id,
                "offset_s": time.perf_counter() - started,
                "active_users": active_users,
            })
            completed_requests = 0
            stopped_by_duration = False
            try:
                for sequence, record in enumerate(group):
                    if not await wait_until(record.planned_offset_s):
                        stopped_by_duration = True
                        break
                    if deadline is not None and time.perf_counter() >= deadline:
                        stopped_by_duration = True
                        break
                    dispatch = time.perf_counter()
                    active_requests += 1
                    event = {"event": "dispatch", "request_id": record.request_id, "user_id": user_id,
                             "user_sequence": sequence, "slot_id": slot_id,
                             "offset_s": dispatch - started, "active": active_requests,
                             "active_users": active_users}
                    record_activity(event)
                    append_jsonl(paths[0], {"request_id": record.request_id, "user_id": user_id,
                        "user_sequence": sequence, "slot_id": slot_id, "created": record.created,
                        "source_path": str(record.path),
                        "source_sha256": record.source_sha256, "planned_offset_s": record.planned_offset_s,
                        "dispatch_offset_s": dispatch - started}, lock)
                    try:
                        try:
                            result = await asyncio.to_thread(sender, record)
                        except Exception as exc:
                            result = {"request_id": record.request_id, "user_id": user_id, "valid": False,
                                      "errors": [f"sender_exception:{exc!r}"]}
                        result.update(user_sequence=sequence, slot_id=slot_id,
                                      planned_offset_s=record.planned_offset_s,
                                      dispatch_offset_s=dispatch - started,
                                      completed_offset_s=time.perf_counter() - started)
                        results.append(result)
                        append_jsonl(paths[1], result, lock)
                    finally:
                        active_requests -= 1
                        completed_requests += 1
                        complete = {"event": "complete", "request_id": record.request_id, "user_id": user_id,
                                    "user_sequence": sequence, "slot_id": slot_id,
                                    "offset_s": time.perf_counter() - started, "active": active_requests,
                                    "active_users": active_users}
                        record_activity(complete)
            finally:
                active_users -= 1
                record_activity({
                    "event": "user_deactivate",
                    "slot_id": slot_id,
                    "user_id": user_id,
                    "offset_s": time.perf_counter() - started,
                    "active_users": active_users,
                    "completed_requests": completed_requests,
                    "reason": "duration" if stopped_by_duration else "exhausted",
                })

    slot_count = min(max_concurrency, len(ordered_groups))
    await asyncio.gather(*(slot_worker(slot_id) for slot_id in range(slot_count)))
    results.sort(key=lambda item: (float(item["dispatch_offset_s"]), str(item["request_id"])))
    return results, activity, time.perf_counter() - started


def distribution(values: list[float]) -> dict[str, float | int | None]:
    values.sort()
    def percentile(q: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
        return values[index]
    return {"count": len(values), "min": values[0] if values else None,
            "mean": statistics.fmean(values) if values else None,
            "stddev": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            "p50": percentile(.5), "p90": percentile(.9), "p95": percentile(.95),
            "p99": percentile(.99), "max": values[-1] if values else None}


def summarize_cache(successful: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        item for item in successful
        if isinstance(item.get("cached_tokens_details"), dict)
    ]
    details = [item["cached_tokens_details"] for item in measured]
    prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in measured)
    source_tokens = {
        source: sum(int(item.get(source) or 0) for item in details)
        for source in ("device", "host", "storage")
    }
    total_cached = sum(source_tokens.values())

    def rate(value: int) -> float | None:
        return value / prompt_tokens if prompt_tokens else None

    return {
        "source": "response_cached_tokens_details" if details else None,
        "requests_with_details": len(details),
        "requests_without_details": len(successful) - len(details),
        "measurement_complete": bool(successful) and len(details) == len(successful),
        "prompt_tokens": prompt_tokens,
        "all_successful_prompt_tokens": sum(
            int(item.get("prompt_tokens") or 0) for item in successful
        ),
        "device_cached_tokens": source_tokens["device"],
        "host_cached_tokens": source_tokens["host"],
        "storage_cached_tokens": source_tokens["storage"],
        "total_cached_tokens": total_cached,
        "device_hit_rate": rate(source_tokens["device"]),
        "host_hit_rate": rate(source_tokens["host"]),
        "storage_hit_rate": rate(source_tokens["storage"]),
        "total_hit_rate": rate(total_cached),
    }


def summarize(records: list[Record], results: list[dict[str, Any]], activity: list[dict[str, Any]], duration: float, minimum: float) -> dict[str, Any]:
    successful = [item for item in results if item.get("valid")]
    numbers = lambda field: [float(item[field]) for item in successful if isinstance(item.get(field), (int, float))]
    attempted, success_rate = len(results), (len(successful) / len(results) if results else 0.0)
    failures: dict[str, int] = defaultdict(int)
    for item in results:
        for reason in item.get("errors", []):
            failures[str(reason).split(":", 1)[0]] += 1
    prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in successful)
    completion_tokens = sum(int(item.get("completion_tokens") or 0) for item in successful)
    return {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "valid": attempted > 0 and success_rate >= minimum,
            "minimum_success_rate": minimum, "counts": {"selected": len(records), "attempted": attempted,
            "not_started_due_to_duration": len(records) - attempted, "successful": len(successful),
            "failed": attempted - len(successful), "users_selected": len({r.user_id for r in records}),
            "users_started": len({item["user_id"] for item in activity
                                  if item.get("event") == "user_activate"}),
            "users_completed": sum(item.get("event") == "user_deactivate"
                                   and item.get("reason") == "exhausted" for item in activity),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}, "duration_s": duration,
            "success_rate": success_rate, "request_throughput": len(successful) / duration if duration else None,
            "prompt_token_throughput": prompt_tokens / duration if duration else None,
            "completion_token_throughput": completion_tokens / duration if duration else None,
            "scheduler": SCHEDULER,
            "max_active_requests": max((int(item["active"]) for item in activity
                                        if "active" in item), default=0),
            "max_active_users": max((int(item["active_users"]) for item in activity
                                     if "active_users" in item), default=0),
            "latency_ms": distribution(numbers("latency_ms")), "ttft_ms": distribution(numbers("ttft_ms")),
            "tpot_ms": distribution(numbers("tpot_ms")),
            "cache": summarize_cache(successful),
            "tps_1000_over_mean_tpot": (1000.0 / statistics.fmean(numbers("tpot_ms"))
                                         if numbers("tpot_ms") else None),
            "failure_reasons": dict(sorted(failures.items()))}


def validate_policy(args) -> str:
    """Validate optional policy settings before network or measured work."""
    if args.scheduler == SCHEDULER:
        if any(value is not None for value in (
            args.session_population, args.drain_timeout, args.ttft_slo_ms
        )):
            raise ValueError("session-population, drain-timeout and ttft-slo-ms require request-slot-fifo")
        return PROTOCOL
    if args.session_population is None or args.session_population < args.max_concurrency:
        raise ValueError("request-slot-fifo requires session-population >= max-concurrency > 0")
    if args.arrival_mode != "immediate" or not math.isfinite(args.max_duration) or args.max_duration <= 0:
        raise ValueError("request-slot-fifo requires arrival-mode immediate and a finite positive max-duration")
    for name, value in (("drain-timeout", args.drain_timeout), ("ttft-slo-ms", args.ttft_slo_ms)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if args.warmup_manifest is not None:
        raise ValueError("request-slot-fifo initially supports cold-cache runs without warmup")
    return request_slots.PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest-input", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--max-concurrency", type=int, default=20)
    parser.add_argument("--max-duration", type=float, default=300)
    parser.add_argument("--scheduler", choices=(SCHEDULER, request_slots.SCHEDULER), default=SCHEDULER)
    parser.add_argument("--session-population", type=int,
                        help="participating conversations; request-slot-fifo only")
    parser.add_argument("--drain-timeout", type=float,
                        help="bounded post-window drain seconds; request-slot-fifo default 120")
    parser.add_argument("--ttft-slo-ms", type=float,
                        help="ready-to-first-output p90 threshold; request-slot-fifo default 10000")
    parser.add_argument("--arrival-mode", choices=("replay", "immediate"), default="replay")
    parser.add_argument("--time-scale", type=float, default=60)
    parser.add_argument("--max-gap", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=14400)
    parser.add_argument("--minimum-success-rate", type=float, default=.99)
    parser.add_argument("--expected-protocol")
    parser.add_argument("--audit-level", choices=AUDIT_LEVELS, default="dev")
    parser.add_argument("--profile-name")
    parser.add_argument("--profile-sha256")
    parser.add_argument("--return-cached-tokens-details", action="store_true")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="send ignore_eos=true; required by the canonical perf-benchmark method",
    )
    parser.add_argument(
        "--user-routing-header",
        help="header name populated with user_id (for example x-sgl-routing-key)",
    )
    parser.add_argument(
        "--forward-tool-control",
        action="store_true",
        help="forward tool_choice and parallel_tool_calls from the dataset record",
    )
    parser.add_argument(
        "--warmup-manifest",
        type=Path,
        help="fixed warmup manifest executed before the measured replay",
    )
    parser.add_argument("--warmup-replicas", type=int, default=1)
    args = parser.parse_args()
    try:
        protocol = validate_policy(args)
    except ValueError as error:
        parser.error(str(error))
    if args.expected_protocol is not None and args.expected_protocol != protocol:
        parser.error(
            f"harness protocol is {protocol}, expected {args.expected_protocol}"
        )
    if args.result_root.exists() and any(args.result_root.iterdir()):
        raise RuntimeError("business result root must be new and empty")
    if args.max_requests < 0 or args.max_concurrency <= 0 or args.max_duration < 0:
        parser.error("request/concurrency/duration values are invalid")
    if not 0 <= args.minimum_success_rate <= 1:
        parser.error("minimum success rate must be between 0 and 1")
    if args.warmup_replicas < 0:
        parser.error("warmup replicas must be nonnegative")
    if args.user_routing_header is not None and not args.user_routing_header.strip():
        parser.error("user routing header must be nonempty")
    warmup_document: dict[str, Any] | None = None
    warmup_requests: list[dict[str, Any]] = []
    if args.warmup_manifest is not None:
        try:
            warmup_document, warmup_requests = load_warmup_manifest(args.warmup_manifest)
        except (OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    try:
        audit = build_audit_metadata(
            args.audit_level,
            Path(__file__),
            args.adapter,
            args.profile_name,
            args.profile_sha256,
            Path(request_slots.__file__) if args.scheduler == request_slots.SCHEDULER else None,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    args.result_root.mkdir(parents=True, exist_ok=True)
    paths = tuple(args.result_root / name for name in ("requests.jsonl", "responses.jsonl", "active_requests.jsonl"))
    for path in paths:
        path.write_text("", encoding="utf-8")
    adapter = load_adapter(args.adapter)
    validate = getattr(adapter, "validate_server", None)
    available_models = validate(args.base_url, args.model, 30) if callable(validate) else []
    if args.dataset_manifest_input is None:
        records, discovered = load_dataset(args.dataset_root, args.max_requests)
    else:
        records, discovered = load_dataset_manifest(
            args.dataset_manifest_input, args.dataset_root, args.max_requests,
            verify_snapshot=args.audit_level == "official",
        )
    max_gap = None if args.max_gap == 0 else args.max_gap
    records = assign_offsets(records, args.arrival_mode, args.time_scale, max_gap)
    if args.scheduler == request_slots.SCHEDULER and args.session_population > len({r.user_id for r in records}):
        raise ValueError("session-population exceeds selected dataset sessions")
    entries = dataset_entries(records, args.dataset_root)
    dataset = {"schema_version": SCHEMA_VERSION, "protocol": protocol, "source_root": str(args.dataset_root),
               "files_discovered": discovered, "selected_requests": len(records),
               "selected_users": len({r.user_id for r in records}), "arrival_mode": args.arrival_mode,
               "time_scale": args.time_scale, "max_gap_s": max_gap,
               "records": entries, "dataset_sha256": workload_sha256(entries)}
    if args.dataset_manifest_input is not None:
        dataset["source_manifest"] = str(args.dataset_manifest_input)
        dataset["source_manifest_sha256"] = hashlib.sha256(
            args.dataset_manifest_input.read_bytes()
        ).hexdigest()
    atomic_json(args.result_root / "dataset_manifest.json", dataset)
    plan = warmup_plan(warmup_requests, args.warmup_replicas)
    warmup_meta = {
        "mode": "perf-benchmark-warm-prompt-v1" if warmup_document is not None else "none",
        "manifest": str(args.warmup_manifest) if args.warmup_manifest is not None else None,
        "manifest_sha256": sha256_file(args.warmup_manifest) if args.warmup_manifest is not None else None,
        "replicas": args.warmup_replicas,
        "copies_per_prompt": warmup_copies(args.warmup_replicas),
        "planned_requests": sum(copies for _, copies in plan),
        "included_in_metrics": False,
    }
    manifest = {"schema_version": SCHEMA_VERSION, "protocol": protocol, "created_at": time.time(),
                "model": args.model, "available_models": available_models, "base_url": args.base_url,
                "adapter": str(args.adapter), "dataset": {"source_root": str(args.dataset_root),
                "manifest": "dataset_manifest.json", "sha256": dataset["dataset_sha256"],
                "selected_requests": len(records), "selected_users": dataset["selected_users"]},
                "replay": {"arrival_mode": args.arrival_mode, "time_scale": args.time_scale,
                "max_gap_s": max_gap, "max_concurrency": args.max_concurrency,
                "max_duration_s": None if args.max_duration == 0 else args.max_duration,
                "concurrency_scope": "fixed_active_users",
                "user_pool_policy": "fifo_user_holds_slot_until_exhausted",
                "scheduler": SCHEDULER,
                "per_user_order": "strict_serial_completion_before_next_dispatch"},
                "validity": {"minimum_success_rate": args.minimum_success_rate},
                "measurement": {
                    "return_cached_tokens_details": args.return_cached_tokens_details,
                    "ignore_eos": args.ignore_eos,
                    "user_routing_header": args.user_routing_header,
                    "forward_tool_control": args.forward_tool_control,
                },
                "warmup": warmup_meta,
                "audit": audit}
    if args.scheduler == request_slots.SCHEDULER:
        manifest["schema_version"] = 2
        manifest["replay"].update(
            scheduler=request_slots.SCHEDULER, session_population=args.session_population,
            concurrency_scope="outstanding_requests",
            user_pool_policy="fifo_ready_sessions_yield_after_each_response",
            readiness="immediate_on_admission_or_previous_response_observation",
            drain_timeout_s=args.drain_timeout if args.drain_timeout is not None else 120,
        )
        manifest["measurement"].update(
            ttft_slo_ms=args.ttft_slo_ms if args.ttft_slo_ms is not None else 10000,
            slo_basis="ready_to_first_output_including_client_queue_and_preparation",
            throughput_basis="successful_completions_inside_fixed_window",
            client_workers="owned_spawn_process_pool",
        )
        # The policy implementation is part of the measured source identity.
        manifest["scheduler_source"] = audit["sources"]["scheduler"]
    atomic_json(args.result_root / "manifest.json", manifest)
    adapter_options = {
        "base_url": args.base_url,
        "model": args.model,
        "timeout": args.timeout,
        "return_cached_tokens_details": args.return_cached_tokens_details,
        "ignore_eos": args.ignore_eos,
        "user_routing_header": args.user_routing_header,
        "forward_tool_control": args.forward_tool_control,
    }
    sender = lambda record: adapter.send_request(record.__dict__, adapter_options)
    warmup_results: list[dict[str, Any]] = []
    if plan:
        warmup_sender_fn = getattr(adapter, "send_warmup_request", None)
        if not callable(warmup_sender_fn):
            raise RuntimeError("adapter does not implement send_warmup_request")
        warmup_path = args.result_root / "warmup.jsonl"
        warmup_path.write_text("", encoding="utf-8")
        warmup_lock = threading.Lock()

        def warmup_sender(entry: dict[str, Any], index: int) -> dict[str, Any]:
            return warmup_sender_fn(
                entry["prompt"],
                int(entry["output_tokens"]),
                {
                    **adapter_options,
                    "warmup_request_id": f"warmup-{entry.get('index', 0)}-{index}",
                },
            )

        warmup_results = asyncio.run(
            run_warmup(plan, warmup_sender, warmup_path, warmup_lock)
        )
    if args.scheduler == SCHEDULER:
        results, activity, duration = asyncio.run(run_replay(records, args.max_concurrency,
            None if args.max_duration == 0 else args.max_duration, sender, paths))
        summary = summarize(records, results, activity, duration, args.minimum_success_rate)
    else:
        path_by_kind = dict(zip(("request", "response", "activity"), paths))
        artifact_lock = threading.Lock()
        emit = lambda kind, row: append_jsonl(path_by_kind[kind], row, artifact_lock)
        async def run_owned():
            # Finish worker cleanup while its callback event loop is still live.
            with request_slots.ProcessSender(args.adapter, adapter_options, args.max_concurrency) as process_sender:
                return await request_slots.run(
                    session_groups(records), args.session_population, args.max_concurrency,
                    args.max_duration, manifest["replay"]["drain_timeout_s"], process_sender, emit,
                )

        outcome = asyncio.run(run_owned())
        duration = outcome.window_s
        summary = request_slots.summarize(
            outcome, records, args.minimum_success_rate,
            manifest["measurement"]["ttft_slo_ms"], summarize, distribution,
        )
        summary["session_population"] = args.session_population
        summary["max_concurrency"] = args.max_concurrency
    summary["warmup"] = {
        "mode": warmup_meta["mode"],
        "requests": len(warmup_results),
        "successful": sum(1 for item in warmup_results if item.get("valid")),
        "included_in_metrics": False,
    }
    summary["audit"] = {
        "level": audit["level"],
        "comparison_eligibility": audit["comparison_eligibility"],
        "official_comparison_eligible": audit["official_comparison_eligible"],
        "warnings": audit["warnings"],
    }
    atomic_json(args.result_root / "summary.json", summary)
    def percent(value: float | None) -> str:
        return f"{value:.4%}" if value is not None else "n/a"

    report = ["# Business Replay Report", "", f"- Valid: `{str(summary['valid']).lower()}`",
              f"- Protocol: `{protocol}`",
              f"- Model: `{args.model}`", f"- Adapter: `{args.adapter}`",
              f"- Selected / attempted / successful: `{summary['counts']['selected']} / {summary['counts']['attempted']} / {summary['counts']['successful']}`",
              f"- Success rate: `{summary['success_rate']:.4%}`", f"- Duration: `{duration:.3f} s`",
              f"- Scheduler: `{summary['scheduler']}`",
              f"- Maximum active users: `{summary['max_active_users']}`",
              f"- Warmup: `{summary['warmup']['mode']} ({summary['warmup']['requests']} requests, not in metrics)`",
              f"- Audit level: `{audit['level']}`",
              f"- Comparison eligibility: `{audit['comparison_eligibility']}`",
              f"- Request throughput: `{summary['request_throughput'] or 0:.3f} req/s`",
              f"- Mean / P90 TTFT: `{summary['ttft_ms']['mean'] or 0:.3f} / {summary['ttft_ms']['p90'] or 0:.3f} ms`",
              f"- Mean / P90 TPOT: `{summary['tpot_ms']['mean'] or 0:.3f} / {summary['tpot_ms']['p90'] or 0:.3f} ms`",
              f"- Mean / P90 E2E: `{summary['latency_ms']['mean'] or 0:.3f} / {summary['latency_ms']['p90'] or 0:.3f} ms`",
              f"- GPU / CPU / total KV cache hit rate: `"
              f"{percent(summary['cache']['device_hit_rate'])} / "
              f"{percent(summary['cache']['host_hit_rate'])} / "
              f"{percent(summary['cache']['total_hit_rate'])}`",
              f"- KV cache detail coverage: `"
              f"{summary['cache']['requests_with_details']} / {summary['counts'].get('successful_in_window', summary['counts']['successful'])}`", "",
              "This report is a business replay episode and is not a baseline matrix result."]
    if args.scheduler == request_slots.SCHEDULER:
        report.extend(["", f"- Ready-to-first-output SLO: `{summary['slo']}`",
                       "HTTP TTFT/E2E above exclude client queueing; use ready_ttft_ms/ready_latency_ms for the new workload."])
    (args.result_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(args.result_root / "done.json", {"schema_version": summary["schema_version"], "protocol": protocol,
                "valid": summary["valid"], "audit_level": audit["level"],
                "comparison_eligibility": audit["comparison_eligibility"],
                "summary": "summary.json", "completed_at": time.time()})
    print(canonical_json({"valid": summary["valid"], **summary["counts"]}))
    raise SystemExit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
