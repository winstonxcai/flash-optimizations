"""Prepare frozen LongSWE JSONL for bench-serving or fixed-session replay once.

Author: Winston Cai.

Uses CPU and the standard library only. Reads the original raw JSONL and
preserves messages, tools and file order. Records source identity separately.
Does not start servers, run benchmarks or validate benchmark results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def integer(value: object, identity: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{identity}: expected integer >= {minimum}, got {value!r}")
    return value


def prepare_replay(args: argparse.Namespace) -> None:
    """Publish original records and a manifest consumed by the unchanged runner."""
    if __package__:
        from .business_replay import runner
    else:
        from business_replay import runner

    source = args.input.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(f"{output}: already exists; choose a fresh preparation directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".lswb-replay-", dir=output.parent))
    source_digest = hashlib.sha256()
    try:
        records_root = scratch / "records"
        records_root.mkdir()
        with source.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                source_digest.update(raw)
                if not raw.strip():
                    continue
                document = json.loads(raw)
                created = document.get("created") if isinstance(document, dict) else None
                if (
                    not isinstance(created, (int, float))
                    or isinstance(created, bool)
                    or not math.isfinite(created)
                ):
                    raise ValueError(f"{source}:{number}: finite numeric created timestamp required")
                # Preserve every field and its original bytes; the existing runner
                # validates identity, session, messages, ordering and duplicates.
                (records_root / f"{number:08d}.json").write_bytes(raw)
        records, discovered = runner.load_dataset(records_root, 0)
        entries = runner.dataset_entries(records, records_root)
        write_json(scratch / "manifest.json", {
            "schema_version": runner.SCHEMA_VERSION,
            "protocol": runner.PROTOCOL,
            "source_path": str(source),
            "source_dataset_sha256": source_digest.hexdigest(),
            "helper_sha256": digest(Path(__file__)),
            "files_discovered": discovered,
            "selected_requests": len(records),
            "selected_users": len({record.user_id for record in records}),
            "dataset_sha256": runner.workload_sha256(entries),
            "records": entries,
        })
        for path in scratch.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        scratch.chmod(0o755)
        scratch.rename(output)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    print(f"Prepared {len(records)} replay requests: {output}")


def prepare(args: argparse.Namespace) -> None:
    source_path = args.input.resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"{source_path}: expected a raw JSONL file")
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError(
            f"{output}: already exists; choose a fresh preparation directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".serving-requests-", dir=output.parent))
    identities: set[str] = set()
    exported = []
    source_digest = hashlib.sha256()
    try:
        with (
            source_path.open("rb") as source,
            (scratch / "requests.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream,
        ):
            for source_line, raw_line in enumerate(source, 1):
                source_digest.update(raw_line)
                if not raw_line.strip():
                    continue
                location = f"{source_path}:{source_line}"
                try:
                    document = json.loads(raw_line)
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f"{location}: invalid JSON: {error}") from error
                if not isinstance(document, dict):
                    raise ValueError(f"{location}: expected a JSON object")
                identity = document.get("id")
                if (
                    not isinstance(identity, str)
                    or not identity
                    or identity in identities
                ):
                    raise ValueError(
                        f"{location}: missing/duplicate request id {identity!r}"
                    )
                identities.add(identity)
                identity = f"{location} ({identity})"
                messages = document.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise ValueError(f"{identity}: messages must be a nonempty list")
                if any(
                    not isinstance(m, dict) or not isinstance(m.get("role"), str)
                    for m in messages
                ):
                    raise ValueError(f"{identity}: invalid message role/object")
                usage = document.get("usage")
                if not isinstance(usage, dict):
                    raise ValueError(f"{identity}: usage must be a JSON object")
                recorded_completion = usage.get("completion_tokens")
                if recorded_completion is not None:
                    integer(recorded_completion, f"{identity} completion_tokens")
                # Match the replay adapter's cap for missing/zero usage. This is
                # an explicit fallback, retained in provenance for each row.
                cap = recorded_completion or 256
                recorded_prompt = usage.get("prompt_tokens")
                session_id = document.get("token_name")
                if session_id is not None and (
                    not isinstance(session_id, str) or not session_id
                ):
                    raise ValueError(
                        f"{identity}: token_name must be a nonempty string"
                    )
                body = {
                    "messages": messages,
                    "max_tokens": cap,
                    "temperature": 0,
                    "prompt_len": (
                        0
                        if recorded_prompt is None
                        else integer(recorded_prompt, identity)
                    ),
                    "ignore_eos": False,
                    "stream_options": {"include_usage": True},
                    "return_cached_tokens_details": True,
                }
                tools = document.get("tools")
                if tools is not None and not isinstance(tools, list):
                    raise ValueError(f"{identity}: tools must be a list")
                if tools:
                    body["tools"] = tools
                line = (
                    json.dumps(
                        body, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    )
                    + "\n"
                )
                stream.write(line)
                exported.append(
                    {
                        "line": len(exported) + 1,
                        "request_id": document["id"],
                        "session_id": session_id,
                        "source_line": source_line,
                        "source_sha256": hashlib.sha256(raw_line).hexdigest(),
                        "max_tokens": cap,
                        "recorded_completion_tokens": recorded_completion,
                        "used_default_output_cap": not recorded_completion,
                        "recorded_prompt_tokens": recorded_prompt,
                        "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                    }
                )
        if not exported:
            raise ValueError(f"{source_path}: no request rows found")
        sessions = {r["session_id"] for r in exported if r["session_id"] is not None}
        write_json(
            scratch / "manifest.json",
            {
                "schema_version": 1,
                "protocol": "lswb-jsonl-bench-serving-v1",
                "source_path": str(source_path),
                "source_dataset_sha256": source_digest.hexdigest(),
                "order": "source JSONL nonblank lines; no shuffle",
                "output_policy": "recorded completion cap (default 256 if absent/zero); respect EOS; temperature zero",
                "default_output_caps": sum(
                    r["used_default_output_cap"] for r in exported
                ),
                "prompt_accounting": "recorded estimates; zero denotes unavailable accounting",
                "missing_prompt_lengths": sum(
                    not r["recorded_prompt_tokens"] for r in exported
                ),
                "requests": len(exported),
                "sessions": len(sessions),
                "dataset_sha256": digest(scratch / "requests.jsonl"),
                "helper_sha256": digest(Path(__file__)),
                "records": exported,
            },
        )
        # mkdtemp starts at 0700. Publish ordinary project-readable artifacts so
        # the separate GPUQ service account can traverse the prepared directory.
        for path in scratch.iterdir():
            path.chmod(0o644)
        scratch.chmod(0o755)
        scratch.rename(output)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    missing = sum(not r["recorded_prompt_tokens"] for r in exported)
    print(
        f"Prepared {len(exported)} requests: {output / 'requests.jsonl'}; "
        f"{missing} have unavailable recorded prompt lengths (zero accounting only); "
        f"{sum(r['used_default_output_cap'] for r in exported)} use the default output cap"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Original raw JSONL file"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("openai", "replay"), default="openai")
    args = parser.parse_args()
    try:
        (prepare_replay if args.format == "replay" else prepare)(args)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"convert_lswb: {error}\n")


if __name__ == "__main__":
    main()
