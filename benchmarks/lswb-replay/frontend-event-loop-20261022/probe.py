"""Source-isolated CPU frontend blocking diagnostic; see the adjacent README.

Author: Winston Cai. SGLang definitions retain their original source/line numbers.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import copy
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
import pickle
import platform
import socket
import statistics
import sys
import time
import types
import unittest
from array import array
from contextlib import nullcontext
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "third_party/sglang/python/sglang"
PLACEMENTS = ("idle", "before", "during", "after")


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_definitions(
    relative: str, names: set[str], namespace: dict, class_name: str | None = None
) -> None:
    """Execute intact definitions only; fail if the requested source moved."""
    path = SOURCE / relative
    tree = ast.parse(path.read_text())
    if class_name:
        owners = [
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        ]
        if len(owners) != 1:
            raise ValueError(f"Missing class {class_name} in {path}")
        definitions = owners[0].body
    else:
        definitions = tree.body
    selected = [
        n
        for n in definitions
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in names
    ]
    if {n.name for n in selected} != names:
        raise ValueError(f"Missing definitions in {path}: {names}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)


def source_packages() -> None:
    # Bypass only the public API initializer, which imports CUDA/Triton tooling.
    # All subsequently imported leaf modules are the actual checked-out source.
    if "sglang" not in sys.modules:
        package = types.ModuleType("sglang")
        package.__path__ = [str(SOURCE)]
        sys.modules["sglang"] = package


def ipc_definitions():
    import msgspec

    module = types.ModuleType("_frontend_probe_ipc")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(msgspec=msgspec, pickle=pickle, _USE_PICKLE_IPC=True)
    source_definitions(
        "srt/managers/io_struct.py",
        {
            "BaseBatchReq",
            "PickleWrapper",
            "BatchStrOutput",
            "BatchTokenIDOutput",
            "wrap_as_pickle",
            "unwrap_from_pickle",
            "async_sock_recv",
            "sock_send",
        },
        ns,
    )
    return module


def frontend_definitions():
    """Load the measured methods, not a rewritten timing approximation."""
    source_packages()
    from sglang.srt.entrypoints.openai import encoding_dsv4, protocol
    from sglang.srt.environ import envs

    ns = dict(vars(protocol))
    ns.update(
        copy=copy,
        json=json,
        Enum=Enum,
        encoding_dsv4=encoding_dsv4,
        envs=envs,
        logger=logging.getLogger(__name__),
    )
    source_definitions(
        "srt/parser/jinja_template_utils.py",
        {"process_content_for_template_format"},
        ns,
    )
    source_definitions(
        "srt/entrypoints/openai/serving_chat.py",
        {
            "ThinkingMode",
            "parse_tool_call_arguments",
            "normalize_assistant_tool_call_arguments",
        },
        ns,
    )
    source_definitions(
        "srt/entrypoints/openai/serving_chat.py",
        {
            "_apply_jinja_template",
            "_encode_messages",
            "_handle_last_assistant_message",
            "_append_assistant_prefix_to_prompt_ids",
        },
        ns,
        "OpenAIServingChat",
    )
    chat = type(
        "MeasuredChat",
        (),
        {
            k: ns[k]
            for k in (
                "_apply_jinja_template",
                "_encode_messages",
                "_handle_last_assistant_message",
                "_append_assistant_prefix_to_prompt_ids",
            )
        },
    )

    ipc = ipc_definitions()
    ns = dict(
        vars(ipc),
        asyncio=asyncio,
        dataclasses=dataclasses,
        time=time,
        real_time=time.time,
        monotonic_time=time.perf_counter,
        SGLANG_FRONTEND_DIAGNOSTICS=False,
        HEALTH_CHECK_RID_PREFIX="health-check-",
        logger=logging.getLogger(__name__),
    )
    # These types cannot occur in this text-only, non-finished output fixture.
    ns.update(
        BatchEmbeddingOutput=type("BatchEmbeddingOutput", (), {}),
        GenerateReqInput=type("GenerateReqInput", (), {}),
    )
    source_definitions("srt/managers/tokenizer_manager.py", {"ReqState"}, ns)
    source_definitions(
        "srt/managers/tokenizer_manager.py",
        {
            "handle_loop",
            "_handle_batch_output",
        },
        ns,
        "TokenizerManager",
    )
    manager = type(
        "MeasuredManager",
        (),
        {k: ns[k] for k in ("handle_loop", "_handle_batch_output")},
    )
    return chat, manager, ns["ReqState"], protocol.ChatCompletionRequest, ipc


def first_output(ipc, rid: str):
    """Actual BatchStrOutput schema, non-finished incremental text, no logprobs."""
    # No schema resolution is needed for the fork's pickle transport. Inspecting
    # annotations would unnecessarily import GPU-only types from unused fields.
    fields = dict.fromkeys(ipc.BatchStrOutput.__struct_fields__)
    fields.update(
        rids=[rid],
        finished_reasons=[None],
        output_strs=["probe"],
        output_ids=[array("q", [1])],
        prompt_tokens=[1],
        completion_tokens=[1],
        reasoning_tokens=[0],
        cached_tokens=[0],
        retraction_counts=[0],
    )
    return ipc.BatchStrOutput(**fields)


def sender(connection, zmq_address: str, sink_port: int) -> None:
    """Independent scheduler sender and loopback client; never shares the event loop."""
    import zmq

    ipc = ipc_definitions()
    ctx = zmq.Context()
    output = ctx.socket(zmq.PUSH)
    output.setsockopt(zmq.LINGER, 0)
    output.setsockopt(zmq.SNDTIMEO, 10000)
    output.connect(zmq_address)
    client = socket.create_connection(("127.0.0.1", sink_port), timeout=15)
    stream = client.makefile("rb")
    try:
        connection.send({"ready": True, "pid": os.getpid()})
        while True:
            command = connection.recv()
            if command is None:
                break
            rid, delay = command
            message = first_output(ipc, rid)
            if delay:
                time.sleep(delay)
            begin = time.perf_counter_ns()
            ipc.sock_send(output, message)
            end = time.perf_counter_ns()
            reply = stream.readline()
            received = time.perf_counter_ns()
            if reply != (rid + ":probe\n").encode():
                raise ValueError(f"Incorrect loopback delivery for {rid}")
            connection.send(
                {
                    "rid": rid,
                    "send_start_ns": begin,
                    "send_end_ns": end,
                    "client_received_ns": received,
                }
            )
    finally:
        stream.close()
        client.close()
        output.close()
        ctx.term()
        connection.close()


class FirstTokenStats:
    """Only the real source's first-output stamp is exercised, not metrics/tracing."""

    first_token_time = 0.0


class TimedTokenizer:
    """Observe the actual fast tokenizer without replacing its implementation."""

    def __init__(self, tokenizer, timings: dict):
        self.tokenizer, self.timings = tokenizer, timings

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def encode(self, *args, **kwargs):
        self.timings["encode_start_ns"] = time.perf_counter_ns()
        try:
            return self.tokenizer.encode(*args, **kwargs)
        finally:
            self.timings["encode_end_ns"] = time.perf_counter_ns()


def classify(row: dict) -> None:
    """Use observed ordering, not the requested sleep, to qualify placement."""
    sent, end = row["send_start_ns"], row["send_end_ns"]
    start, finish = row["prepare_start_ns"], row["prepare_end_ns"]
    phase = row["placement"]
    placement_ok = (
        phase == "idle"
        or (phase == "before" and end <= row["handler_entry_ns"] <= start)
        or (phase == "during" and start < sent <= end < finish)
        or (phase == "after" and finish <= sent)
    )
    if not placement_ok:
        raise ValueError(f"Output did not fall in intended placement: {row}")
    if not sent <= end <= row["client_received_ns"]:
        raise ValueError(f"Invalid cross-process timestamps: {row}")
    if not sent <= row["handler_entry_ns"] <= row["notified_ns"]:
        raise ValueError(f"Invalid handler timestamps: {row}")
    if phase == "during" and row["handler_entry_ns"] < finish:
        raise ValueError(
            f"Handler unexpectedly entered during synchronous preparation: {row}"
        )
    row.update(
        placement_valid=True,
        preparation_ms=(finish - start) / 1e6,
        send_to_handler_ms=(row["handler_entry_ns"] - sent) / 1e6,
        send_to_client_ms=(row["client_received_ns"] - sent) / 1e6,
        post_preparation_handler_ms=(
            (row["handler_entry_ns"] - finish) / 1e6 if phase == "during" else None
        ),
    )
    if "encode_start_ns" in row:
        if not start <= row["encode_start_ns"] <= row["encode_end_ns"] <= finish:
            raise ValueError(f"Tokenization outside preparation: {row}")
        row["tokenization_ms"] = (row["encode_end_ns"] - row["encode_start_ns"]) / 1e6
    else:
        row["tokenization_ms"] = 0.0


async def measure(args, records, tokenizer):
    import zmq
    import zmq.asyncio

    Chat, Manager, ReqState, Request, ipc = frontend_definitions()
    # Reuse the exact first-token stamping method without initializing tracing.
    source_definitions(
        "srt/observability/req_time_stats.py",
        {"set_first_token_time"},
        stats_ns := {"time": time},
        "APIServerReqTimeStats",
    )
    FirstTokenStats.set_first_token_time = stats_ns["set_first_token_time"]
    active = {}
    chat = Chat()
    chat.tokenizer_manager = types.SimpleNamespace(
        tokenizer=TimedTokenizer(tokenizer, active)
    )
    chat.chat_encoding_spec = "dsv4"
    chat._dsv4_reasoning_effort_profile = "official"
    chat.template_manager = types.SimpleNamespace(
        jinja_template_content_format="string"
    )
    chat._tokenizer_auto_adds_specials = False
    manager = Manager()
    manager.server_args = types.SimpleNamespace(batch_notify_size=1)
    manager.config_value = lambda name: "probe" if name == "weight_version" else None
    manager.rid_to_state = {}
    manager.enable_metrics = False
    manager.enable_lora = False
    manager.incremental_streaming_output = False
    manager.dump_requests_folder = manager.crash_dump_folder = ""
    manager.soft_watchdog = types.SimpleNamespace(
        disable=nullcontext, feed=lambda: None
    )
    ctx = zmq.asyncio.Context()
    incoming = ctx.socket(zmq.PULL)
    incoming.setsockopt(zmq.LINGER, 0)
    port = incoming.bind_to_random_port("tcp://127.0.0.1")
    manager.recv_from_detokenizer = incoming
    sink_ready = asyncio.Future()

    async def accept(reader, writer):
        if sink_ready.done():
            writer.close()
            return
        sink_ready.set_result(writer)

    sink = await asyncio.start_server(accept, "127.0.0.1", 0)
    parent, child = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(
        target=sender,
        args=(child, f"tcp://127.0.0.1:{port}", sink.sockets[0].getsockname()[1]),
    )
    original_handler = manager._handle_batch_output

    async def observed_handler(message):
        active["handler_entry_ns"] = time.perf_counter_ns()
        await original_handler(message)
        active["handler_exit_ns"] = time.perf_counter_ns()

    manager._handle_batch_output = observed_handler
    task = asyncio.create_task(manager.handle_loop())
    writer = None
    results = []
    try:
        process.start()
        child.close()
        ready = await asyncio.wait_for(asyncio.to_thread(parent.recv), 30)
        writer = await asyncio.wait_for(sink_ready, 15)

        async def wait_for_output(consumer):
            done, _ = await asyncio.wait(
                {consumer, task}, timeout=15, return_when=asyncio.FIRST_COMPLETED
            )
            if task in done:
                task.result()  # Propagate source-handler failures immediately.
                raise RuntimeError("Frontend receive loop stopped unexpectedly")
            if consumer not in done:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
                raise TimeoutError("No frontend output within 15 seconds")
            consumer.result()

        async def one(body, identity, fraction, phase, round_id):
            request = Request.model_validate(copy.deepcopy(body))
            rid = f"{identity}-{fraction}-{round_id}-{phase}"
            state = ReqState(
                out_list=[],
                finished=False,
                event=asyncio.Event(),
                obj=types.SimpleNamespace(stream=True, return_logprob=False),
                time_stats=FirstTokenStats(),
            )
            manager.rid_to_state[rid] = state
            active.clear()

            async def consume():
                await state.event.wait()
                active["notified_ns"] = time.perf_counter_ns()
                if state.get_text() != "probe" or len(state.out_list) != 1:
                    raise ValueError(f"Incorrect frontend output for {rid}")
                writer.write((rid + ":probe\n").encode())
                await writer.drain()

            consumer = asyncio.create_task(consume())
            if phase in ("idle", "before"):
                parent.send((rid, 0))
                await wait_for_output(consumer)
            active["prepare_start_ns"] = time.perf_counter_ns()
            ids = []
            if phase != "idle":
                if phase == "during":
                    parent.send((rid, args.send_delay_ms / 1000))
                ids = chat._apply_jinja_template(
                    request, tools=None, is_multimodal=False
                ).prompt_ids
            active["prepare_end_ns"] = time.perf_counter_ns()
            if phase == "after":
                parent.send((rid, 0))
            await wait_for_output(consumer)
            packet = await asyncio.wait_for(asyncio.to_thread(parent.recv), 15)
            if packet["rid"] != rid or ready["pid"] == os.getpid():
                raise ValueError("Sender identity mismatch")
            row = dict(
                packet,
                **active,
                issue=identity,
                fraction=fraction,
                placement=phase,
                round=round_id,
                tokens=len(ids),
                token_ids_sha256=hashlib.sha256(array("I", ids).tobytes()).hexdigest(),
                first_token_stamp_ns=int(state.time_stats.first_token_time * 1e9),
            )
            classify(row)
            if (
                not row["handler_entry_ns"]
                <= row["first_token_stamp_ns"]
                <= row["handler_exit_ns"]
            ):
                raise ValueError("First-token timestamp outside the actual handler")
            manager.rid_to_state.pop(rid)
            results.append(row)
            with (args.output / "samples.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        # Untimed handshake also warms the ZMQ connection and real handler.
        await one(records[0][1], "handshake", 1, "idle", -1)
        results.clear()
        for identity, body in records:
            for fraction in args.fractions:
                selected = copy.deepcopy(body)
                count = max(2, int(len(selected["messages"]) * fraction))
                selected["messages"] = selected["messages"][:count]
                # Warm actual tokenization/validation before any measured trial.
                expected = chat._apply_jinja_template(
                    Request.model_validate(copy.deepcopy(selected)),
                    tools=None,
                    is_multimodal=False,
                ).prompt_ids
                expected_hash = hashlib.sha256(
                    array("I", expected).tobytes()
                ).hexdigest()
                for round_id in range(args.rounds):
                    phases = PLACEMENTS[round_id % 4 :] + PLACEMENTS[: round_id % 4]
                    for phase in phases:
                        await one(selected, identity, fraction, phase, round_id)
                        if (
                            phase != "idle"
                            and results[-1]["token_ids_sha256"] != expected_hash
                        ):
                            raise ValueError(f"Unstable tokenization: {identity}")
        return results
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if process.is_alive():
            try:
                parent.send(None)
            except (BrokenPipeError, EOFError):
                pass  # The original child/handler failure remains authoritative.
            await asyncio.to_thread(process.join, 5)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 5)
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        sink.close()
        await sink.wait_closed()
        incoming.close()
        ctx.term()
        parent.close()
        process.close()


class ProbeContractTests(unittest.TestCase):
    """Independent timestamp fixtures and real IPC round-trip regression checks."""

    @staticmethod
    def during():
        return dict(
            placement="during",
            send_start_ns=20_000_000,
            send_end_ns=21_000_000,
            prepare_start_ns=10_000_000,
            prepare_end_ns=100_000_000,
            handler_entry_ns=101_000_000,
            notified_ns=102_000_000,
            client_received_ns=103_000_000,
            encode_start_ns=15_000_000,
            encode_end_ns=99_000_000,
        )

    def test_independent_latency_accounting(self):
        row = self.during()
        classify(row)
        self.assertEqual(row["preparation_ms"], 90)
        self.assertEqual(row["tokenization_ms"], 84)
        self.assertEqual(row["send_to_handler_ms"], 81)
        self.assertEqual(row["send_to_client_ms"], 83)
        self.assertEqual(row["post_preparation_handler_ms"], 1)

    def test_wrong_placement_or_clocks_fail(self):
        for changes in (
            {"send_end_ns": 110_000_000},
            {"handler_entry_ns": 90_000_000},
            {"client_received_ns": 1},
            {"encode_end_ns": 110_000_000},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                classify(dict(self.during(), **changes))

    def test_before_and_after(self):
        before = dict(
            self.during(),
            placement="before",
            handler_entry_ns=22_000_000,
            notified_ns=23_000_000,
            prepare_start_ns=30_000_000,
            encode_start_ns=31_000_000,
        )
        classify(before)
        after = dict(
            self.during(),
            placement="after",
            send_start_ns=110_000_000,
            send_end_ns=111_000_000,
            handler_entry_ns=112_000_000,
            notified_ns=113_000_000,
            client_received_ns=114_000_000,
        )
        classify(after)

    def test_real_ipc_pickle_round_trip(self):
        ipc = ipc_definitions()
        message = pickle.loads(pickle.dumps(first_output(ipc, "round-trip")))
        self.assertIsInstance(message, ipc.BatchStrOutput)
        self.assertEqual(message.rids, ["round-trip"])
        self.assertEqual(message.output_strs, ["probe"])
        self.assertEqual(message.output_ids, [array("q", [1])])
        self.assertEqual(message.finished_reasons, [None])

    def test_detokenizer_transport_preserves_pickle_and_msgpack_wrappers(self):
        ipc = ipc_definitions()
        ns = dict(vars(ipc))
        source_definitions(
            "srt/managers/detokenizer_manager.py",
            {"_record_output_transport_time"},
            ns,
            "DetokenizerManager",
        )
        stamp = ns["_record_output_transport_time"]
        for use_pickle in (True, False):
            with self.subTest(pickle_ipc=use_pickle):
                ipc._USE_PICKLE_IPC = use_pickle
                # The extracted helper must use the real wrapper's globals.
                message = first_output(ipc, "transport")
                message.time_stats = ipc.wrap_as_pickle([
                    types.SimpleNamespace(first_output_send_time=1.0), None
                ])
                stamp(None, message, "detokenizer_recv_time", 2.0)
                stamp(None, message, "detokenizer_send_time", 3.0)
                if not use_pickle:
                    self.assertIsInstance(message.time_stats, ipc.PickleWrapper)
                record, missing = ipc.unwrap_from_pickle(message.time_stats)
                self.assertEqual(record.first_output_send_time, 1.0)
                self.assertEqual(record.detokenizer_recv_time, 2.0)
                self.assertEqual(record.detokenizer_send_time, 3.0)
                self.assertIsNone(missing)
                message.time_stats = None
                stamp(None, message, "detokenizer_recv_time", 4.0)
                self.assertIsNone(message.time_stats)

    def test_scheduler_stamps_included_rows_before_payload_serialization(self):
        packets = []

        class Accumulator:
            def __init__(self, **kwargs):
                self.time_stats = []

            def accept(self, *, req):
                if req.include:
                    self.time_stats.append(req.time_stats)

            def to_payload(self, **kwargs):
                # Inspect the boundary before actual transport serialization.
                return [stats.first_output_send_time for stats in self.time_stats]

        ns = dict(
            _GenerationStreamAccumulator=Accumulator,
            SGLANG_FRONTEND_DIAGNOSTICS=True,
            monotonic_time=lambda: 12.0,
            get_serving=lambda: types.SimpleNamespace(stream_interval=1),
            DEFAULT_FORCE_STREAM_INTERVAL=50,
        )
        source_definitions(
            "srt/managers/scheduler_components/output_streamer.py",
            {"_stream_output_generation"}, ns, "SchedulerOutputStreamer",
        )
        owner = types.SimpleNamespace(
            spec_algorithm=None, disaggregation_mode=None,
            get_cached_tokens_details=lambda req: None, rust_server=None,
            _maybe_log_time_stats=lambda **kwargs: None,
            ps=types.SimpleNamespace(dp_rank=0),
            send_to_detokenizer=types.SimpleNamespace(send_output=packets.append),
        )
        def request(first, include=True):
            return types.SimpleNamespace(
                return_hidden_states=False, return_routed_experts=False,
                return_indexer_topk=False, return_sampling_mask=False,
                finished=lambda: False, include=include,
                time_stats=types.SimpleNamespace(first_output_send_time=first),
            )
        fresh, later, suppressed = request(0), request(7), request(0, False)
        ns["_stream_output_generation"](owner, [fresh, later, suppressed], False)
        self.assertEqual(packets, [[12.0, 7]])
        self.assertEqual(suppressed.time_stats.first_output_send_time, 0)
        ns["SGLANG_FRONTEND_DIAGNOSTICS"] = False
        ns["monotonic_time"] = lambda: self.fail("disabled diagnostics read clock")
        off = request(0)
        ns["_stream_output_generation"](owner, [off], False)
        self.assertEqual(off.time_stats.first_output_send_time, 0)

    def test_detokenizer_receive_decode_send_boundaries_in_both_worker_loops(self):
        ipc = ipc_definitions()
        for multi_worker in (False, True):
            with self.subTest(multi_http_worker=multi_worker):
                packet = first_output(ipc, "boundaries")
                packet.http_worker_ipcs = ["worker-zero"]
                packet.time_stats = [types.SimpleNamespace(first_output_send_time=1.0)]
                received = iter([packet])
                sent = []
                class EndLoop(Exception):
                    pass
                def receive(socket):
                    try:
                        return next(received)
                    except StopIteration:
                        raise EndLoop
                def dispatch(message):
                    self.assertEqual(message.time_stats[0].detokenizer_recv_time, 2.0)
                    return message
                ns = dict(vars(ipc), SGLANG_FRONTEND_DIAGNOSTICS=True,
                          monotonic_time=iter([2.0, 3.0]).__next__, sock_recv=receive,
                          sock_send=lambda socket, output: sent.append(output),
                          BaseReq=type("BaseReq", (), {}),
                          BatchEmbeddingOutput=type("BatchEmbeddingOutput", (), {}))
                source_definitions("srt/managers/detokenizer_manager.py",
                                   {"event_loop", "_record_output_transport_time"},
                                   ns, "DetokenizerManager")
                if multi_worker:
                    source_definitions(
                        "srt/managers/multi_tokenizer_mixin.py",
                        {"_extract_field_by_index", "_handle_output_by_index"}, ns,
                    )
                    source_definitions("srt/managers/multi_tokenizer_mixin.py",
                                       {"multi_http_worker_event_loop"}, ns,
                                       "MultiHttpWorkerDetokenizerMixin")
                    ns["SocketMapping"] = lambda: types.SimpleNamespace(
                        send_output=lambda ipc_name, output, **kwargs: sent.append(output)
                    )
                owner = types.SimpleNamespace(
                    soft_watchdog=types.SimpleNamespace(
                        disable=nullcontext, feed=lambda: None
                    ),
                    recv_from_scheduler=None, send_to_tokenizer=None,
                    _request_dispatcher=dispatch,
                    _record_output_transport_time=lambda *args: ns[
                        "_record_output_transport_time"
                    ](None, *args),
                )
                loop_name = "multi_http_worker_event_loop" if multi_worker else "event_loop"
                loop = ns[loop_name]
                with self.assertRaises(EndLoop):
                    loop(owner)
                self.assertEqual(len(sent), 1)
                stats = ipc.unwrap_from_pickle(sent[0].time_stats)[0]
                self.assertEqual(
                    (stats.first_output_send_time, stats.detokenizer_recv_time,
                     stats.detokenizer_send_time),
                    (1.0, 2.0, 3.0),
                )
                self.assertEqual(sent[0].rids, ["boundaries"])

    def test_frontend_loop_captures_receive_before_watchdog_exit_and_handler(self):
        _, Manager, _, _, ipc = frontend_definitions()
        ns = Manager.handle_loop.__globals__
        message = first_output(ipc, "receive")
        phases, received = [], iter([message])
        class EndLoop(Exception):
            pass
        class Guard:
            def __enter__(self):
                phases.append("waiting")
            def __exit__(self, *args):
                phases.append("watchdog-exit")
        async def receive(socket):
            try:
                return next(received)
            except StopIteration:
                raise EndLoop
        def now():
            self.assertEqual(phases[-1], "waiting")
            phases.append("received")
            return 5.0
        async def handle(packet, *, recv_time):
            self.assertIs(packet, message)
            self.assertEqual(recv_time, 5.0)
            self.assertEqual(phases[-1], "watchdog-exit")
            phases.append("handler")
        ns.update(SGLANG_FRONTEND_DIAGNOSTICS=True, async_sock_recv=receive,
                  monotonic_time=now)
        manager = Manager()
        manager.soft_watchdog = types.SimpleNamespace(disable=Guard, feed=lambda: None)
        manager.recv_from_detokenizer = None
        manager._handle_batch_output = handle
        with self.assertRaises(EndLoop):
            asyncio.run(manager.handle_loop())
        self.assertEqual(
            phases[:4], ["waiting", "received", "watchdog-exit", "handler"]
        )

    def test_frontend_latches_receive_and_per_row_stages_from_first_packet(self):
        _, Manager, ReqState, _, ipc = frontend_definitions()
        ns = Manager._handle_batch_output.__globals__
        ns.update(SGLANG_FRONTEND_DIAGNOSTICS=True,
                  monotonic_time=iter([10.0, 20.0]).__next__,
                  get_spec=lambda: types.SimpleNamespace(speculative_algorithm=None))
        stat_ns = dict(time=time, SGLANG_FRONTEND_DIAGNOSTICS=True,
                       convert_time_to_realtime=lambda value: value)
        method_names = {
            "record_frontend_first_output", "set_first_token_time",
            "set_finished_time", "get_e2e_latency", "get_decode_latency",
            "convert_to_output_meta_info",
        }
        source_definitions("srt/observability/req_time_stats.py", method_names,
                           stat_ns, "APIServerReqTimeStats")
        Stats = type("TestStats", (), {name: stat_ns[name] for name in method_names})
        stats = Stats()
        for name in ("first_token_time", "finished_time", "first_output_send_time",
                     "detokenizer_recv_time", "detokenizer_send_time",
                     "frontend_recv_time", "frontend_handle_time",
                     "api_server_dispatch_finish_time", "response_sent_to_client_time"):
            setattr(stats, name, 0.0)
        stats.created_time = 1.0
        stats.trace_ctx = types.SimpleNamespace(tracing_enable=False)
        stats._frontend_diagnostics_recorded = False
        state = ReqState([], False, asyncio.Event(),
                         types.SimpleNamespace(stream=True, return_logprob=False,
                                               log_metrics=False), stats)
        manager = Manager()
        manager.server_args = types.SimpleNamespace(batch_notify_size=1)
        manager.config_value = lambda name: "probe"
        manager.rid_to_state = {"one": state}
        manager.enable_metrics = manager.enable_lora = False
        manager.incremental_streaming_output = False
        manager.dump_requests_folder = manager.crash_dump_folder = ""
        first = first_output(ipc, "one")
        first.time_stats = [types.SimpleNamespace(
            first_output_send_time=2.0, detokenizer_recv_time=3.0,
            detokenizer_send_time=4.0)]
        asyncio.run(manager._handle_batch_output(first, recv_time=9.0))
        token_stamp = stats.first_token_time
        self.assertEqual((stats.first_output_send_time, stats.detokenizer_recv_time,
                          stats.detokenizer_send_time, stats.frontend_recv_time,
                          stats.frontend_handle_time), (2.0, 3.0, 4.0, 9.0, 10.0))
        # A final packet has a new scheduler snapshot, but must not replace the
        # original transport stages or the existing first-output timestamp.
        final = first_output(ipc, "one")
        final.finished_reasons = [{"type": "length", "length": 2}]
        final.completion_tokens = [2]
        final.time_stats = [types.SimpleNamespace(
            first_output_send_time=2.0, detokenizer_recv_time=30.0,
            detokenizer_send_time=40.0)]
        asyncio.run(manager._handle_batch_output(final, recv_time=19.0))
        self.assertEqual(stats.first_token_time, token_stamp)
        self.assertEqual(stats.frontend_recv_time, 9.0)
        self.assertEqual(stats.frontend_handle_time, 10.0)
        self.assertEqual(stats.detokenizer_recv_time, 3.0)
        self.assertEqual(stats.detokenizer_send_time, 4.0)
        self.assertEqual(state.out_list[-1]["meta_info"]["frontend_recv_ts"], 9.0)

    def test_missing_source_definition_rejected(self):
        with self.assertRaises(ValueError):
            source_definitions(
                "srt/managers/io_struct.py", {"nonexistent_probe_function"}, {}
            )

    @staticmethod
    def pipeline_fixture():
        import importlib.util

        path = (ROOT / "benchmarks/lswb-replay/"
                "c24-packed256-store-decode-ablation-n1-20261014/analyze.py")
        spec = importlib.util.spec_from_file_location("pipeline_analysis", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        start = 1_800_000_000.0
        clients, servers = [], []
        fields = ("prefill_finished_time", "first_output_send_ts", "detokenizer_recv_ts",
                  "detokenizer_send_ts", "frontend_recv_ts", "frontend_handle_ts",
                  "first_output_ts")
        for request_id, offset, step in (("a", 10, 1), ("b", 190, 2)):
            clients.append(dict(request_id="data-" + request_id, valid=True,
                                server_request_id=request_id, server_request_id_error=None))
            record = dict(id=request_id, request_received_ts=start + offset)
            record.update({name: start + offset + index * step + 1
                           for index, name in enumerate(fields)})
            servers.append(record)
        return module.first_output_pipeline_diagnostics, start, clients, servers

    def test_pipeline_exact_join_stage_arithmetic_and_received_time_cohort(self):
        diagnose, start, clients, servers = self.pipeline_fixture()
        result = diagnose(clients, servers, start)
        self.assertTrue(result["valid"] and result["id_verified"])
        self.assertEqual(result["coverage"]["matched_successful"], 2)
        self.assertEqual(result["cohorts"]["all"]["count"], 2)
        self.assertEqual(result["cohorts"]["after_180s"]["count"], 1)
        stages = result["cohorts"]["all"]["stages"]
        self.assertEqual(stages["post_prefill_to_first_output_s"]["mean"], 9)
        for stage in stages.keys() - {"post_prefill_to_first_output_s"}:
            self.assertEqual(stages[stage]["mean"], 1.5)
        self.assertEqual(result["cohorts"]["after_180s"]["stages"]
                         ["post_prefill_to_first_output_s"]["mean"], 12)
        shifted = [{key: value + 1000 if key != "id" else value
                    for key, value in record.items()} for record in servers]
        self.assertEqual(diagnose(clients, shifted, start + 1000)["cohorts"],
                         result["cohorts"])
        servers[0]["detokenizer_recv_ts"] = servers[0]["first_output_send_ts"]
        self.assertTrue(diagnose(clients, servers, start)["valid"])

    def test_pipeline_missing_nonfinite_or_misordered_boundaries_rejected(self):
        diagnose, start, clients, servers = self.pipeline_fixture()
        cases = []
        for field in servers[0].keys() - {"id"}:
            missing = copy.deepcopy(servers)
            del missing[0][field]
            cases.append(missing)
        for value in (float("nan"), float("inf"), 0, -1, True, None):
            invalid = copy.deepcopy(servers)
            invalid[0]["frontend_recv_ts"] = value
            cases.append(invalid)
        misordered = copy.deepcopy(servers)
        misordered[0]["detokenizer_send_ts"] = servers[0]["first_output_send_ts"] - 1
        cases.append(misordered)
        for index, candidate in enumerate(cases):
            with self.subTest(case=index):
                result = diagnose(clients, candidate, start)
                self.assertFalse(result["valid"])
                self.assertTrue(result["errors"])
                self.assertEqual(result["coverage"]["timing_valid_successful"], 1)

    def test_pipeline_duplicate_or_unmatched_ids_rejected(self):
        diagnose, start, clients, servers = self.pipeline_fixture()
        for candidate_clients, candidate_servers in (
            (clients + [copy.deepcopy(clients[0])], servers),
            (clients, servers + [copy.deepcopy(servers[0])]),
            (clients, []),
            ([dict(clients[0], server_request_id=None,
                   server_request_id_error="missing_server_request_id")], servers),
            ([dict(clients[0], server_request_id_error="inconsistent_server_request_id")], servers),
        ):
            with self.subTest(clients=candidate_clients, server_count=len(candidate_servers)):
                result = diagnose(candidate_clients, candidate_servers, start)
                self.assertFalse(result["valid"] or result["id_verified"])
                self.assertTrue(result["errors"])
        failed = dict(valid=False, server_request_id=None,
                      server_request_id_error="missing_server_request_id")
        result = diagnose(clients + [failed], servers + [dict(servers[0], id="failed")], start)
        self.assertTrue(result["valid"])
        self.assertEqual(result["coverage"]["unmatched_server_records"], 1)
        self.assertEqual(len(result["client_identity_diagnostics"]), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--issues", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--send-delay-ms", type=float, default=10)
    parser.add_argument("--event-loop", choices=("uvloop", "asyncio"), default="uvloop")
    args = parser.parse_args()
    if (
        args.issues < 1
        or args.rounds < 1
        or args.send_delay_ms <= 0
        or any(not 0 < f <= 1 for f in args.fractions)
    ):
        parser.error("Positive counts/delay and fractions in (0,1] required")
    sys.path.insert(0, str(ROOT))
    from scripts.sources import check_sources

    sources = check_sources(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(USE_TORCH="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import PreTrainedTokenizerFast

    if args.event_loop == "uvloop":
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    records = []
    with args.requests.open() as handle:
        for line in handle:
            record = json.loads(line)
            body = record.get("request", record)
            records.append((record.get("instance_id", str(len(records))), body))
            if len(records) == args.issues:
                break
    if len(records) != args.issues or len({i for i, _ in records}) != args.issues:
        raise ValueError("Insufficient distinct request identities")
    files = [
        SOURCE / s
        for s in (
            "srt/entrypoints/openai/serving_chat.py",
            "srt/entrypoints/openai/protocol.py",
            "srt/entrypoints/openai/encoding_dsv4.py",
            "srt/parser/jinja_template_utils.py",
            "srt/managers/tokenizer_manager.py",
            "srt/managers/io_struct.py",
            "srt/observability/req_time_stats.py",
            "srt/environ.py",
        )
    ]
    provenance = {
        "git_sources": sources,
        "probe_sha256": fingerprint(Path(__file__)),
        "source_sha256": {str(p.relative_to(ROOT)): fingerprint(p) for p in files},
        "requests_sha256": fingerprint(args.requests),
        "tokenizer_sha256": {
            n: fingerprint(args.tokenizer / n)
            for n in ("tokenizer.json", "tokenizer_config.json")
        },
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            n: importlib.metadata.version(n)
            for n in (
                "transformers",
                "tokenizers",
                "pyzmq",
                "msgspec",
                "pydantic",
                "openai",
            )
        },
        "event_loop": args.event_loop,
        "uvloop_version": (
            importlib.metadata.version("uvloop")
            if args.event_loop == "uvloop"
            else None
        ),
        "command": sys.argv,
        "ipc": "pickle (fork default)",
        "scope": "source-isolated template/tokenization + real output handler; synthetic loopback sink",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    try:
        rows = asyncio.run(measure(args, records, tokenizer))
        expected = args.issues * len(args.fractions) * args.rounds * len(PLACEMENTS)
        if len(rows) != expected:
            raise ValueError(f"Incomplete coverage: {len(rows)}/{expected}")
        if provenance["source_sha256"] != {
            str(p.relative_to(ROOT)): fingerprint(p) for p in files
        } or provenance["probe_sha256"] != fingerprint(Path(__file__)):
            raise ValueError("Measured source changed during collection")
        aggregates = []
        for fraction in args.fractions:
            for phase in PLACEMENTS:
                subset = [
                    r
                    for r in rows
                    if r["fraction"] == fraction and r["placement"] == phase
                ]
                aggregates.append(
                    dict(
                        fraction=fraction,
                        placement=phase,
                        samples=len(subset),
                        mean_tokens=statistics.mean(r["tokens"] for r in subset),
                        **{
                            name: statistics.mean(r[name] for r in subset)
                            for name in (
                                "preparation_ms",
                                "tokenization_ms",
                                "send_to_handler_ms",
                                "send_to_client_ms",
                            )
                        },
                    )
                )
        result = dict(
            valid=True,
            samples=len(rows),
            issues=args.issues,
            warmup_samples=1,
            samples_sha256=fingerprint(args.output / "samples.jsonl"),
            independent_serving_runs=0,
            provenance=provenance,
            aggregates=aggregates,
            checks=dict(
                placement_and_clock_order=True,
                stable_token_ids=True,
                exact_output_identity_and_text=True,
                first_token_stamp_inside_handler=True,
                during_handler_after_preparation=True,
                source_unchanged=True,
            ),
            limitations=[
                "Not the full HTTP server or OpenAI streaming formatter",
                "No GPU/scheduler work or completion-driven load simulated",
                "Cannot attribute a Triton/Kernel TTFT difference",
                "Prefix fractions are message-count fractions, not token fractions",
            ],
        )
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {k: result[k] for k in ("valid", "samples", "issues", "aggregates")}
            )
        )
    except BaseException as error:
        (args.output / "failure.json").write_text(
            json.dumps({"valid": False, "error": repr(error)}, indent=2) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
