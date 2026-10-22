"""OpenAI-compatible streaming adapter for SGLang, vLLM, and similar servers.

Projects can copy this adapter or provide a module with the same
``validate_server`` and ``send_request`` functions when their wire protocol or
payload options differ.

Author: Winston Cai. (Replay timing and diagnostic server-request identity.)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _has_output(delta: Any) -> bool:
    return isinstance(delta, dict) and bool(
        delta.get("content")
        or delta.get("reasoning")
        or delta.get("reasoning_content")
        or delta.get("tool_calls")
    )


def _cached_tokens_details(data: dict[str, Any]) -> dict[str, Any] | None:
    # SGLang reports per-tier cache hits through its own extension.
    sglext = data.get("sglext")
    if isinstance(sglext, dict):
        raw = sglext.get("cached_tokens_details")
        if isinstance(raw, dict):
            details: dict[str, Any] = {}
            for source in ("device", "host", "storage"):
                value = _optional_nonnegative_int(raw.get(source))
                if value is not None:
                    details[source] = value
            if isinstance(raw.get("storage_backend"), str):
                details["storage_backend"] = raw["storage_backend"]
            if details:
                return details
    # Engines that only implement the OpenAI schema (vLLM among them) report a
    # single aggregate hit count instead. Attribute it to "device": the tier
    # split is not observable from here, so the total is what a caller should
    # compare. Returning None when the field is absent keeps the "no details
    # reported" case distinguishable from a genuine zero.
    usage = data.get("usage")
    if isinstance(usage, dict):
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached = _optional_nonnegative_int(prompt_details.get("cached_tokens"))
            if cached is not None:
                return {"device": cached, "host": 0, "storage": 0}
    return None


def _finalize_cached_tokens_details(
    requested: bool,
    status: int | None,
    details: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if details is not None:
        return details, "reported"
    if requested and status == 200:
        # SGLang omits cached_tokens_details when a successful request has no hit.
        return {"device": 0, "host": 0, "storage": 0}, "requested_zero_hit"
    return None, None


def validate_server(base_url: str, model: str, timeout_s: float = 30) -> list[str]:
    response = requests.get(base_url.rstrip("/") + "/v1/models", timeout=timeout_s)
    response.raise_for_status()
    document = response.json()
    models = [str(item.get("id")) for item in document.get("data", []) if isinstance(item, dict)]
    if model not in models:
        raise RuntimeError(f"served model {model!r} not found; available models: {models}")
    return models


def build_request_payload(
    document: dict[str, Any],
    record: dict[str, Any],
    options: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build the measured request body and headers from a dataset document.

    ``ignore_eos``, tool-control passthrough, and the per-user routing header
    are opt-in so existing baselines keep their previous wire contract.
    """
    payload: dict[str, Any] = {
        "model": options["model"],
        "messages": document["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": max(1, int(record.get("expected_completion_tokens") or 256)),
    }
    if isinstance(document.get("tools"), list) and document["tools"]:
        payload["tools"] = document["tools"]
    if options.get("forward_tool_control"):
        for key in ("tool_choice", "parallel_tool_calls"):
            if key in document:
                payload[key] = document[key]
    if options.get("return_cached_tokens_details"):
        payload["return_cached_tokens_details"] = True
    if options.get("ignore_eos"):
        payload["ignore_eos"] = True
    extra_payload = options.get("payload")
    if isinstance(extra_payload, dict):
        payload.update(extra_payload)

    headers = {"Content-Type": "application/json"}
    routing_header = options.get("user_routing_header")
    if routing_header:
        headers[str(routing_header)] = str(record.get("user_id") or "default")
    return payload, headers


def _stream_completion(
    payload: dict[str, Any],
    headers: dict[str, str],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Execute one streaming chat completion and return raw measurements."""
    started = time.perf_counter()
    first_output: float | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_count = 0
    done_count = 0
    done_seen = False
    status: int | None = None
    errors: list[str] = []
    server_request_id: str | None = None
    server_request_id_error: str | None = None
    cached_tokens_details: dict[str, Any] | None = None
    response_hash = hashlib.sha256()
    url = options.get("chat_path", "/v1/chat/completions")
    try:
        with requests.post(
            str(options["base_url"]).rstrip("/") + str(url),
            json=payload,
            headers=headers,
            stream=True,
            timeout=(60, float(options["timeout"])),
        ) as response:
            status = response.status_code
            if status != 200:
                errors.append(f"http_status:{status}:{response.text[:500]}")
            else:
                for raw_line in response.iter_lines(decode_unicode=False):
                    response_hash.update(raw_line + b"\n")
                    if not raw_line:
                        continue
                    if not raw_line.startswith(b"data:"):
                        errors.append("unknown_sse_line")
                        continue
                    event = raw_line[5:].strip()
                    if event == b"[DONE]":
                        done_count += 1
                        if finish_count != 1:
                            errors.append("done_before_or_without_final")
                        done_seen = True
                        continue
                    if done_seen:
                        errors.append("json_event_after_done")
                    try:
                        data = json.loads(event)
                    except json.JSONDecodeError:
                        errors.append("malformed_json")
                        continue
                    if not isinstance(data, dict):
                        errors.append("json_event_not_object")
                        continue
                    # A stable response ID permits an exact server-metrics join
                    # without modifying the measured request. Usage-only events
                    # may omit it. Join diagnostics do not change response validity.
                    event_request_id = data.get("id")
                    if isinstance(event_request_id, str) and event_request_id.strip():
                        if server_request_id is None:
                            server_request_id = event_request_id
                        elif event_request_id != server_request_id:
                            server_request_id_error = "inconsistent_server_request_id"
                    elif event_request_id is not None:
                        server_request_id_error = (
                            server_request_id_error or "invalid_server_request_id"
                        )
                    if data.get("error") is not None:
                        errors.append(f"error_event:{data['error']!r}")
                    event_cache_details = _cached_tokens_details(data)
                    if event_cache_details is not None:
                        cached_tokens_details = event_cache_details
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = _optional_nonnegative_int(usage.get("prompt_tokens"))
                        completion_tokens = _optional_nonnegative_int(usage.get("completion_tokens"))
                    choices = data.get("choices")
                    if not isinstance(choices, list):
                        continue
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        if first_output is None and _has_output(choice.get("delta")):
                            first_output = time.perf_counter()
                        if choice.get("finish_reason") is not None:
                            finish_count += 1
                            finish_reason = str(choice["finish_reason"])
    except Exception as exc:
        errors.append(f"request_exception:{exc!r}")
    ended = time.perf_counter()
    if status == 200:
        if done_count != 1:
            errors.append("missing_or_duplicate_done")
        if finish_count != 1 or finish_reason is None:
            errors.append("missing_or_duplicate_finish_reason")
        if prompt_tokens is None:
            errors.append("missing_prompt_tokens")
        if completion_tokens is None:
            errors.append("missing_completion_tokens")
        elif completion_tokens > 0 and first_output is None:
            errors.append("missing_output_event")
    cached_tokens_details, cached_tokens_details_status = _finalize_cached_tokens_details(
        bool(options.get("return_cached_tokens_details")), status, cached_tokens_details
    )
    ttft_ms = (first_output - started) * 1_000 if first_output is not None else None
    tpot_ms = None
    if first_output is not None and completion_tokens is not None and completion_tokens > 1:
        tpot_ms = (ended - first_output) * 1_000 / (completion_tokens - 1)
    body_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if server_request_id is None and server_request_id_error is None:
        server_request_id_error = "missing_server_request_id"
    result = {
        "status": status,
        "valid": not errors,
        "errors": sorted(set(errors)),
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": (ended - started) * 1_000,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "tpot_source": "client_stream_span_over_server_completion_tokens" if tpot_ms is not None else None,
        "cached_tokens_details": cached_tokens_details,
        "cached_tokens_details_status": cached_tokens_details_status,
        "request_body_sha256": body_hash,
        "response_stream_sha256": response_hash.hexdigest(),
        "server_request_id": server_request_id if server_request_id_error is None else None,
        "server_request_id_error": server_request_id_error,
    }
    if options.get("capture_client_timing"):
        # CPU workers on this host share the monotonic clock. The runner turns
        # these transient readings into offsets and includes its own queue and
        # request preparation time without changing the historical HTTP metrics.
        result["client_timing"] = {
            "http_started_s": started,
            "first_output_s": first_output,
            "http_completed_s": ended,
        }
    return result


def send_request(record: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(record["path"]))
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != record["source_sha256"]:
        raise RuntimeError(f"dataset source changed after manifest creation: {path}")
    document = json.loads(source)
    payload, request_headers = build_request_payload(document, record, options)
    result = _stream_completion(payload, request_headers, options)
    result.update({
        "request_id": record["request_id"],
        "user_id": record["user_id"],
        "expected_prompt_tokens": record.get("expected_prompt_tokens"),
        "expected_completion_tokens": record.get("expected_completion_tokens"),
        "prompt_token_delta": (
            result["prompt_tokens"] - record["expected_prompt_tokens"]
            if result["prompt_tokens"] is not None and record.get("expected_prompt_tokens") is not None
            else None
        ),
    })
    return result


def send_warmup_request(
    prompt: str,
    output_tokens: int,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Send one perf-benchmark ``WARM_PROMPT`` request.

    This is the warmup counterpart of ``send_request``: it uses the same
    streaming parser and measurements, but builds a plain user message instead
    of reading a dataset record.
    """
    payload: dict[str, Any] = {
        "model": options["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": max(1, int(output_tokens)),
    }
    if options.get("ignore_eos"):
        payload["ignore_eos"] = True
    extra_payload = options.get("payload")
    if isinstance(extra_payload, dict):
        payload.update(extra_payload)
    result = _stream_completion(
        payload, {"Content-Type": "application/json"}, options
    )
    result.update({
        "request_id": str(options.get("warmup_request_id") or "warmup"),
        "user_id": "warmup",
        "expected_prompt_tokens": None,
        "expected_completion_tokens": int(output_tokens),
        "prompt_token_delta": None,
    })
    return result
