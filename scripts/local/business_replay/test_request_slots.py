"""CPU checks for replay scheduling and measurement. Author: Winston Cai.

Run: python -m unittest discover -s scripts/local/business_replay -p 'test_*.py'
No Docker, model weights, HTTP server or GPUs; subprocess tests use a fake
adapter in a temporary directory through the real runner/worker interfaces.
"""

import argparse
import asyncio
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import request_slots
import runner
from adapters import openai_sse


def records(users=3, turns=2):
    return [runner.Record(f"{user}-{turn}", str(user), float(turn), Path("unused"), "hash", 10, 2)
            for user in range(users) for turn in range(turns)]


def successful(start, end):
    return {"valid": True, "errors": [], "prompt_tokens": 10,
            "completion_tokens": 2, "ttft_ms": (end-start)*1000,
            "latency_ms": (end-start)*1000, "tpot_ms": 1,
            "cached_tokens_details": {"device": 0, "host": 0, "storage": 0},
            "client_timing": {"http_started_s": start, "first_output_s": end,
                              "http_completed_s": end}}


class SchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, cohort, population, concurrency, sender, duration=1, drain=.1):
        emitted = []
        outcome = await request_slots.run(
            runner.session_groups(cohort), population, concurrency, duration, drain,
            sender, lambda kind, value: emitted.append((kind, value)),
        )
        return outcome, emitted

    async def test_fifo_yields_each_turn(self):
        async def sender(record):
            start = time.perf_counter()
            await asyncio.sleep(.001)
            return successful(start, time.perf_counter())

        outcome, events = await self.execute(records(), 3, 1, sender)
        dispatched = [row["request_id"] for kind, row in events if kind == "request"]
        self.assertEqual(dispatched, ["0-0", "1-0", "2-0", "0-1", "1-1", "2-1"])
        self.assertTrue(all(s.result["valid"] for s in outcome.states))
        self.assertGreater(outcome.states[2].result["ready_ttft_ms"],
                           outcome.states[2].result["ttft_ms"])

    async def test_bounds_serial_order_and_replacement(self):
        active_users = set()
        peak = 0
        async def sender(record):
            nonlocal peak
            self.assertNotIn(record.user_id, active_users)
            active_users.add(record.user_id)
            peak = max(peak, len(active_users))
            start = time.perf_counter()
            await asyncio.sleep(.001 if record.user_id == "0" else .002)
            active_users.remove(record.user_id)
            return successful(start, time.perf_counter())

        outcome, events = await self.execute(records(5), 3, 2, sender)
        self.assertLessEqual(peak, 2)
        self.assertEqual(len(outcome.states), 10)
        self.assertLessEqual(max(r["active_users"] for r in outcome.activity), 3)
        self.assertEqual(sum(r["event"] == "user_deactivate" for r in outcome.activity), 5)
        for user in range(5):
            self.assertEqual([r["user_sequence"] for kind, r in events
                              if kind == "request" and r["user_id"] == str(user)], [0, 1])

    async def test_exception_releases_request_slot(self):
        async def sender(record):
            if record.request_id == "0-0":
                raise RuntimeError("fixture failure")
            start = time.perf_counter()
            return successful(start, time.perf_counter())

        outcome, _ = await self.execute(records(2), 2, 1, sender)
        self.assertEqual(len(outcome.states), 4)
        self.assertFalse(outcome.states[0].result["valid"])
        self.assertTrue(outcome.states[-1].result["valid"])

    async def test_cutoff_stops_dispatch_and_drain_cancels(self):
        async def sender(record):
            await asyncio.Event().wait()

        outcome, events = await self.execute(records(), 3, 2, sender, .02, .02)
        self.assertEqual(sum(kind == "request" for kind, _ in events), 2)
        self.assertEqual(len(outcome.states), 3)
        self.assertEqual(sum(s.result is None for s in outcome.states), 1)
        self.assertTrue(all("drain_deadline_exceeded" in s.result["errors"]
                            for s in outcome.states if s.result is not None))

    async def test_legacy_keeps_session_until_exhausted(self):
        def sender(record):
            return {"request_id": record.request_id, "user_id": record.user_id,
                    "valid": True, "errors": []}
        with tempfile.TemporaryDirectory() as temporary:
            paths = tuple(Path(temporary)/name for name in ("requests", "responses", "activity"))
            results, _, _ = await runner.run_replay(records(2), 1, None, sender, paths)
        self.assertEqual([r["request_id"] for r in results], ["0-0", "0-1", "1-0", "1-1"])


class AccountingTests(unittest.TestCase):
    def summary(self, states, window=1, slo=10000):
        return request_slots.summarize(
            request_slots.ReplayOutcome(window, window+.2, states),
            [s.record for s in states], .99, slo, runner.summarize, runner.distribution,
        )

    def test_window_and_drain_counts_are_separate(self):
        states = []
        for record, finish in zip(records(2, 1), (.8, 1.2)):
            states.append(request_slots.RequestState(record, 0, 0, 0, 0, {
                "valid": True, "errors": [], "completed_offset_s": finish,
                "ready_ttft_ms": 10, "ready_latency_ms": finish*1000,
                "prompt_tokens": 10, "completion_tokens": 2,
            }))
        summary = self.summary(states)
        self.assertEqual(summary["counts"]["successful"], 2)
        self.assertEqual(summary["counts"]["successful_in_window"], 1)
        self.assertEqual(summary["counts"]["attempted"], 2)
        self.assertEqual(summary["counts"]["failed"], 0)
        self.assertEqual(summary["request_throughput"], 1)
        self.assertEqual(summary["counts"]["completed_during_drain"], 1)

    def test_censored_requests_cannot_silently_pass(self):
        state = request_slots.RequestState(records(1, 1)[0], 0, .99, 0)
        self.assertEqual(self.summary([state])["slo"]["status"], "inconclusive")
        state.ready_s = 0
        self.assertEqual(self.summary([state], slo=100)["slo"]["status"], "fail")

    def test_slo_includes_queue_and_strict_boundary(self):
        state = request_slots.RequestState(records(1, 1)[0], 0, 0, 0, .02)
        result = successful(100.03, 100.04)
        state.result = request_slots._timed_result(state, result, 100, .05)
        self.assertAlmostEqual(state.result["client_queue_ms"], 20)
        self.assertAlmostEqual(state.result["client_preparation_ms"], 10)
        self.assertAlmostEqual(state.result["ready_ttft_ms"], 40)
        self.assertEqual(self.summary([state], slo=30)["slo"]["status"], "fail")
        state.result["ready_ttft_ms"] = 10000
        self.assertEqual(self.summary([state])["slo"]["status"], "fail")

    def test_missing_and_nonmonotonic_timing_fail_measurement(self):
        for result in ({"valid": True, "errors": []}, successful(101, 100)):
            state = request_slots.RequestState(records(1, 1)[0], 0, 0, 0, 0)
            state.result = request_slots._timed_result(state, result, 100, .5)
            summary = self.summary([state])
            self.assertFalse(summary["valid"])
            self.assertEqual(summary["instrumentation_errors"], 1)

    def test_policy_rejects_unsupported_combinations(self):
        base = dict(scheduler=request_slots.SCHEDULER, session_population=24,
                    max_concurrency=20, max_duration=1200, arrival_mode="immediate",
                    drain_timeout=None, ttft_slo_ms=None, warmup_manifest=None)
        self.assertEqual(runner.validate_policy(argparse.Namespace(**base)), request_slots.PROTOCOL)
        for overrides in (dict(session_population=19), dict(arrival_mode="replay"),
                          dict(max_duration=0), dict(drain_timeout=float("nan")),
                          dict(warmup_manifest=Path("unused")), dict(scheduler=runner.SCHEDULER)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                runner.validate_policy(argparse.Namespace(**(base | overrides)))

    def test_adapter_timing_is_opt_in_and_wire_body_unchanged(self):
        class Response:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def iter_lines(self, **_):
                for row in (
                    {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
                ):
                    yield b"data: " + json.dumps(row).encode()
                yield b"data: [DONE]"
        options = {"base_url": "http://unused", "timeout": 1}
        with patch.object(openai_sse.requests, "post", return_value=Response()) as post:
            old = openai_sse._stream_completion({"stream": True}, {}, options)
            new = openai_sse._stream_completion({"stream": True}, {}, options | {"capture_client_timing": True})
        self.assertTrue(old["valid"] and new["valid"])
        self.assertNotIn("client_timing", old)
        self.assertIn("client_timing", new)
        self.assertEqual(old["request_body_sha256"], new["request_body_sha256"])
        self.assertEqual(post.call_args_list[0], post.call_args_list[1])


class RunnerProcessTests(unittest.TestCase):
    """Real CLI, spawn workers, artifacts and hard drain; no network requests."""

    def run_fixture(self, delay, duration, drain):
        temporary = tempfile.TemporaryDirectory(prefix="request-slot-test-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        dataset = root / "data"
        dataset.mkdir()
        for record in records(3, 1):
            (dataset / f"{record.request_id}.json").write_text(json.dumps({
                "id": record.request_id, "token_name": record.user_id,
                "created": record.created, "messages": [{"role": "user", "content": "fixture"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }))
        adapter = root / "adapter.py"
        adapter.write_text(
            "import time\n"
            "def send_request(record, options):\n"
            "    start = time.perf_counter()\n"
            f"    time.sleep({delay!r})\n"
            "    end = time.perf_counter()\n"
            "    return dict(valid=True, errors=[], prompt_tokens=10, completion_tokens=2, "
            "ttft_ms=(end-start)*1000, latency_ms=(end-start)*1000, tpot_ms=1, "
            "client_timing=dict(http_started_s=start, first_output_s=end, http_completed_s=end))\n"
        )
        output = root / "results"
        result = subprocess.run([
            sys.executable, str(Path(runner.__file__).resolve()),
            "--dataset-root", str(dataset), "--adapter", str(adapter),
            "--result-root", str(output), "--model", "fixture", "--max-requests", "0",
            "--scheduler", request_slots.SCHEDULER, "--session-population", "3",
            "--max-concurrency", "2", "--arrival-mode", "immediate",
            "--max-duration", str(duration), "--drain-timeout", str(drain),
            "--expected-protocol", request_slots.PROTOCOL,
        ], capture_output=True, text=True, timeout=15)
        return result, output, json.loads((output / "summary.json").read_text())

    def test_real_runner_and_spawn_workers(self):
        result, output, summary = self.run_fixture(.002, 1, .1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["counts"]["successful_in_window"], 3)
        self.assertEqual(summary["request_throughput"], 3)
        self.assertEqual(summary["slo"]["status"], "pass")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["replay"]["session_population"], 3)
        self.assertIn("scheduler", manifest["audit"]["sources"])

    def test_blocked_http_workers_do_not_extend_drain(self):
        start = time.monotonic()
        result, _, summary = self.run_fixture(30, .03, .03)
        self.assertLess(time.monotonic() - start, 10)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(summary["counts"]["drain_timeouts"], 2)
        self.assertEqual(summary["counts"]["ready_not_dispatched"], 1)
        self.assertEqual(summary["request_throughput"], 0)

    def test_drain_only_run_reports_zero_without_row_crash(self):
        result, output, summary = self.run_fixture(.04, .02, .1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary["counts"]["successful_in_window"], 0)
        self.assertEqual(summary["counts"]["successful"], 2)
        self.assertEqual(summary["slo"]["status"], "inconclusive")
        # lswb-row expects a run folder containing client/summary.json.
        client = output / "client"
        client.mkdir()
        (client / "summary.json").write_text(json.dumps(summary))
        script = Path(runner.__file__).resolve().parents[1] / "lswb-row.sh"
        row = subprocess.run(["bash", str(script), str(output)],
                             env={**os.environ, "PYTHON": sys.executable},
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(row.returncode, 0, row.stdout + row.stderr)
        self.assertIn("slo=FAIL", row.stdout)
        self.assertIn("slo_status=inconclusive", row.stdout)
        self.assertIn("done=0", row.stdout)

    def test_worker_import_failure_cleans_up_pool(self):
        before = {p.pid for p in multiprocessing.active_children()}
        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary)/"bad_adapter.py"
            adapter.write_text("raise RuntimeError('fixture import failure')\n")
            with self.assertRaisesRegex(RuntimeError, "fixture import failure"):
                request_slots.ProcessSender(adapter, {}, 2)
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)


if __name__ == "__main__":
    unittest.main()
