"""Bounded-population request scheduling for the existing business replay.

Author: Winston Cai.

The runner owns policy selection and artifacts. This module owns the optional
FIFO scheduler and its CPU HTTP workers. It is a closed-loop workload with no
tool think time, not an external arrival-rate or live-agent simulator.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import math
import multiprocessing
import os
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from .runner import Record


SCHEDULER = "request-slot-fifo"
PROTOCOL = "business-request-slot-replay-v1"
WORKER_STARTUP_TIMEOUT_S = 60


@functools.lru_cache(maxsize=1)
def _adapter(path: str):
    spec = importlib.util.spec_from_file_location("request_slot_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load replay worker adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _send(adapter: str, options: dict, record: dict) -> dict:
    return _adapter(adapter).send_request(record, options)


def _initialize_worker(adapter: str, ready) -> None:
    try:
        _adapter(adapter)
    except Exception as error:
        ready.put((os.getpid(), repr(error)))
    else:
        ready.put((os.getpid(), None))


class ProcessSender:
    """Own CPU-only workers so drain timeout can stop blocking HTTP calls.

    Pool creation happens before the measurement clock. Spawn avoids inheriting
    HTTP connections or event-loop state. Workers return metadata only; the
    parent is the sole artifact writer. Exiting the context terminates and joins
    workers, including blocked requests, before the server launcher cleans up.
    """

    def __init__(self, adapter: Path, options: dict[str, Any], concurrency: int) -> None:
        self.adapter = str(adapter.resolve())
        self.options = {**options, "capture_client_timing": True}
        self.closed = False
        context = multiprocessing.get_context("spawn")
        ready = context.Queue()
        self.pool = context.Pool(concurrency, initializer=_initialize_worker,
                                 initargs=(self.adapter, ready))
        try:
            deadline = time.monotonic() + WORKER_STARTUP_TIMEOUT_S
            initialized = set()
            while len(initialized) < concurrency:
                pid, error = ready.get(timeout=max(0, deadline - time.monotonic()))
                if error is not None:
                    raise RuntimeError(f"Replay worker failed loading {self.adapter}: {error}")
                initialized.add(pid)
        except queue.Empty as error:
            self.close()
            raise RuntimeError(
                f"Replay workers did not initialize {self.adapter} within {WORKER_STARTUP_TIMEOUT_S}s"
            ) from error
        except BaseException:
            self.close()
            raise
        finally:
            ready.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.pool.terminate()
            self.pool.join()

    async def __call__(self, record: Record) -> dict:
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def deliver(value, error=False):
            if not future.done():
                if error:
                    future.set_exception(value)
                else:
                    future.set_result(value)

        def callback(value, error=False):
            if not self.closed:
                # Normal ownership closes the pool before the loop. The guard
                # also handles cancellation during event-loop shutdown.
                try:
                    loop.call_soon_threadsafe(deliver, value, error)
                except RuntimeError:
                    if not loop.is_closed():
                        raise

        self.pool.apply_async(
            _send, (self.adapter, self.options, record.__dict__),
            callback=callback,
            error_callback=lambda error: callback(error, True),
        )
        return await future


@dataclass
class RequestState:
    record: Record
    sequence: int
    ready_s: float
    session_slot: int
    dispatch_s: float | None = None
    result: dict | None = None


@dataclass
class ReplayOutcome:
    window_s: float
    elapsed_s: float
    states: list[RequestState] = field(default_factory=list)
    activity: list[dict] = field(default_factory=list)


def _timed_result(state: RequestState, result: dict, origin: float, observed: float) -> dict:
    """Validate cross-worker monotonic readings; never infer missing timing."""
    timing = result.pop("client_timing", None)
    result.update(
        request_id=state.record.request_id, user_id=state.record.user_id,
        user_sequence=state.sequence, ready_offset_s=state.ready_s,
        dispatch_offset_s=state.dispatch_s,
        completed_offset_s=observed,
        client_queue_ms=(state.dispatch_s - state.ready_s) * 1000,
    )
    if timing is None:
        if result.get("valid"):
            result["valid"] = False
            result.setdefault("errors", []).append("missing_client_timing")
        return result
    start, first, end = (timing.get(key) for key in (
        "http_started_s", "first_output_s", "http_completed_s"
    ))
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    valid = (
        finite(start) and finite(end)
        and origin + state.dispatch_s <= start <= end <= origin + observed
        and (first is None or (finite(first) and start <= first <= end))
    )
    if not valid:
        result["valid"] = False
        result.setdefault("errors", []).append("invalid_client_timing")
        return result
    result.update(
        http_started_offset_s=start - origin,
        completed_offset_s=end - origin,
        client_preparation_ms=(start - origin - state.dispatch_s) * 1000,
        ready_latency_ms=(end - origin - state.ready_s) * 1000,
    )
    if first is not None:
        result.update(
            first_output_offset_s=first - origin,
            ready_ttft_ms=(first - origin - state.ready_s) * 1000,
        )
    return result


async def run(
    groups: list[tuple[str, list[Record]]], population: int, concurrency: int,
    duration: float, drain_timeout: float,
    sender: Callable[[Record], Awaitable[dict]],
    emit: Callable[[str, dict], None],
) -> ReplayOutcome:
    """Dispatch one turn per ready session, then return it to the FIFO tail.

    Ready time begins at session admission or preceding response observation.
    Immediate readiness is an explicit workload assumption. Cutoff stops both
    dispatch and admission; drain collects only already-dispatched requests.
    """
    if not 0 < concurrency <= population or duration <= 0 or drain_timeout <= 0:
        raise ValueError("Require population >= concurrency > 0 and positive window/drain")
    origin = time.perf_counter()
    cutoff = origin + duration
    outcome = ReplayOutcome(duration, 0.0)
    pending = deque(groups)
    ready: deque[RequestState] = deque()
    sessions: dict[str, list] = {}
    free_slots = deque(range(concurrency))
    active: dict[asyncio.Task, tuple[RequestState, int]] = {}

    def event(kind: str, **fields):
        row = {"event": kind, "offset_s": time.perf_counter() - origin,
               "active": len(active), "active_users": len(sessions),
               "ready_queue_depth": len(ready), **fields}
        outcome.activity.append(row)
        emit("activity", row)

    def enqueue(record, sequence, session_slot):
        state = RequestState(record, sequence, time.perf_counter() - origin, session_slot)
        ready.append(state)
        outcome.states.append(state)
        event("ready", request_id=record.request_id, user_id=record.user_id,
              user_sequence=sequence, ready_offset_s=state.ready_s)

    def admit(session_slot):
        if pending:
            user, group = pending.popleft()
            sessions[user] = group
            event("user_activate", user_id=user, session_slot_id=session_slot)
            enqueue(group[0], 0, session_slot)

    for slot in range(min(population, len(groups))):
        admit(slot)

    def complete(task, continue_sessions):
        state, slot = active.pop(task)
        free_slots.append(slot)
        try:
            result = task.result()
        except asyncio.CancelledError:
            result = {"valid": False, "errors": ["drain_deadline_exceeded"]}
        except Exception as error:
            result = {"valid": False, "errors": [f"sender_exception:{error!r}"]}
        state.result = _timed_result(state, result, origin, time.perf_counter() - origin)
        state.result.update(slot_id=slot, session_slot_id=state.session_slot)
        emit("response", state.result)
        event("complete", request_id=state.record.request_id, user_id=state.record.user_id)
        group = sessions[state.record.user_id]
        next_sequence = state.sequence + 1
        if next_sequence == len(group):
            del sessions[state.record.user_id]
            event("user_deactivate", user_id=state.record.user_id, reason="exhausted")
            if continue_sessions:
                admit(state.session_slot)
        elif continue_sessions:
            enqueue(group[next_sequence], next_sequence, state.session_slot)

    try:
        while time.perf_counter() < cutoff and (active or ready):
            while ready and free_slots and time.perf_counter() < cutoff:
                state = ready.popleft()
                slot = free_slots.popleft()
                state.dispatch_s = time.perf_counter() - origin
                task = asyncio.create_task(sender(state.record))
                active[task] = (state, slot)
                emit("request", {
                    "request_id": state.record.request_id, "user_id": state.record.user_id,
                    "user_sequence": state.sequence, "slot_id": slot,
                    "session_slot_id": state.session_slot, "created": state.record.created,
                    "source_path": str(state.record.path), "source_sha256": state.record.source_sha256,
                    "planned_offset_s": state.record.planned_offset_s,
                    "ready_offset_s": state.ready_s, "dispatch_offset_s": state.dispatch_s,
                })
                event("dispatch", request_id=state.record.request_id, user_id=state.record.user_id)
            if active:
                done, _ = await asyncio.wait(
                    active, timeout=max(0, cutoff - time.perf_counter()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in sorted(done, key=lambda t: active[t][0].dispatch_s):
                    complete(task, time.perf_counter() < cutoff)
        event("measurement_cutoff", configured_window_s=duration,
              pending_sessions=len(pending), undispatched_ready_requests=len(ready))
        if active:
            done, unfinished = await asyncio.wait(active, timeout=drain_timeout)
            for task in sorted(done, key=lambda t: active[t][0].dispatch_s):
                complete(task, False)
            for task in unfinished:
                task.cancel()
            if unfinished:
                await asyncio.gather(*unfinished, return_exceptions=True)
                for task in list(unfinished):
                    complete(task, False)
        outcome.elapsed_s = time.perf_counter() - origin
        return outcome
    finally:
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)


def summarize(outcome: ReplayOutcome, records: list[Record], minimum: float, slo_ms: float,
              base_summary: Callable, distribution: Callable) -> dict:
    """Fixed-window throughput and conservative SLO bounds over ready requests."""
    results = [state.result for state in outcome.states if state.result is not None]
    in_window = [r for r in results if r["completed_offset_s"] <= outcome.window_s]
    summary = base_summary(records, in_window, outcome.activity, outcome.window_s, minimum)
    valid_total = sum(bool(r.get("valid")) for r in results)
    success_rate = valid_total / len(results) if results else 0.0
    passed = failed = censored = 0
    for state in outcome.states:
        result = state.result
        if result is not None:
            latency = result.get("ready_ttft_ms")
            if result.get("valid") and latency is not None and latency < slo_ms:
                passed += 1
            else:
                failed += 1
        elif (outcome.window_s - state.ready_s) * 1000 >= slo_ms:
            failed += 1
        else:
            censored += 1
    total = len(outcome.states)
    required = math.ceil(.9 * total)
    status = ("pass" if total and passed >= required else
              "fail" if total and passed + censored < required else "inconclusive")
    observed = lambda field: [r[field] for r in results if r.get(field) is not None]
    instrumentation_errors = sum(any(error in r.get("errors", []) for error in
        ("missing_client_timing", "invalid_client_timing")) for r in results)
    summary.update(
        schema_version=2, protocol=PROTOCOL, scheduler=SCHEDULER,
        valid=bool(results) and success_rate >= minimum and not instrumentation_errors,
        instrumentation_errors=instrumentation_errors,
        success_rate=success_rate, total_elapsed_s=outcome.elapsed_s,
        ready_ttft_ms=distribution(observed("ready_ttft_ms")),
        ready_latency_ms=distribution(observed("ready_latency_ms")),
        client_queue_ms=distribution(observed("client_queue_ms")),
        slo={"metric": "ready_to_first_output", "threshold_ms": slo_ms,
             "percentile": 90, "status": status, "ready_requests": total,
             "within_threshold": passed, "violations": failed,
             "right_censored": censored,
             "basis": "All ready requests; invalid/no-output responses count as violations; unresolved cutoff requests bounded conservatively"},
        window={"duration_s": outcome.window_s,
                "throughput_basis": "Successful responses completed inside fixed measurement window",
                "http_latency_basis": "Existing ttft_ms/latency_ms exclude client queue and preparation",
                "ready_latency_basis": "From admission or prior response observation; includes client queue and preparation"},
    )
    summary["counts"].update(
        attempted=len(results), successful=valid_total, failed=len(results) - valid_total,
        successful_in_window=sum(bool(r.get("valid")) for r in in_window),
        completed_in_window=len(in_window),
        completed_during_drain=sum(r["completed_offset_s"] > outcome.window_s
                                   and "drain_deadline_exceeded" not in r.get("errors", []) for r in results),
        drain_timeouts=sum("drain_deadline_exceeded" in r.get("errors", []) for r in results),
        ready_not_dispatched=sum(s.dispatch_s is None for s in outcome.states),
        not_started_due_to_duration=len(records) - len(results),
    )
    summary["failure_reasons"] = {}
    for result in results:
        for error in result.get("errors", []):
            reason = str(error).split(":", 1)[0]
            summary["failure_reasons"][reason] = summary["failure_reasons"].get(reason, 0) + 1
    return summary
