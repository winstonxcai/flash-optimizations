"""CPU storage audits, exact conversion and complete reader measurements.

Author: Winston Cai.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path

import torch

from .. import artifacts


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_sizes(value, prefix="") -> Counter:
    """Include nested audited shared tensors as well as top-level fields."""
    if isinstance(value, torch.Tensor):
        return Counter({prefix: value.numel() * value.element_size()})
    sizes = Counter()
    items = (
        value.items()
        if isinstance(value, dict)
        else (enumerate(value) if isinstance(value, (list, tuple)) else ())
    )
    for key, child in items:
        sizes.update(_tensor_sizes(child, f"{prefix}.{key}" if prefix else str(key)))
    return sizes


def audit(root: Path, runner: Path, output: Path, *, convert: bool = False) -> dict:
    """Audit every group; compression probes cover first/last layer per issue."""
    math = artifacts.load_runner(runner)
    if output.exists():
        raise ValueError(f"fresh storage-audit output directory required: {output}")
    runner_hash = hashlib.sha256(runner.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    config = json.loads((root / "capture-config.json").read_text())
    if not config["requests"] or not config["layers"]:
        raise ValueError(f"nonempty capture requests and layers required: {root}")
    fields = Counter()
    rows = []
    missing_groups = []
    groups = snapshots = serialized = occurrences = unique = 0
    counted_files = set()
    first_seconds = warm_seconds = 0.0
    compressed = uncompressed = 0
    native_checks = converted_groups = 0
    converted_reader_seconds = converted_repeat_seconds = 0.0
    max_native_error = 0.0
    source = hashlib.sha256()
    began = time.monotonic()
    for rid in sorted(config["requests"]):
        for layer in config["layers"]:
            seen = set()
            before_groups = groups
            for group_index, paths in enumerate(
                math.state_paths(root, config, rid, layer)
            ):
                start = time.monotonic()
                state = math.load_state(paths, "cpu")
                first_seconds += time.monotonic() - start
                start = time.monotonic()
                again = math.load_state(paths, "cpu")
                warm_seconds += time.monotonic() - start
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and not torch.equal(
                        value, again[key]
                    ):
                        raise ValueError(f"reader changed {rid}/{layer}/{key}")
                group_bytes = 0
                for path in math.snapshot_dependencies(paths):
                    path = path.resolve()
                    if path in counted_files:
                        continue
                    counted_files.add(path)
                    payload = path.read_bytes()
                    source.update(str(path.relative_to(root.resolve())).encode())
                    source.update(hashlib.sha256(payload).digest())
                    group_bytes += len(payload)
                    shard = torch.load(
                        io.BytesIO(payload), map_location="cpu", weights_only=True
                    )
                    fields.update(_tensor_sizes(shard))
                serialized += group_bytes
                snapshots += len(paths)
                groups += 1
                for i in range(len(state["raw_c"])):
                    digest = hashlib.sha256()
                    # Content, not physical address, defines duplicate rows.
                    for key in ("raw_c", "csa_bytes", "raw_positions", "csa_freqs"):
                        digest.update(tensor_bytes(state[key][i : i + 1]))
                    for key in ("norm_weight",):
                        digest.update(tensor_bytes(state[key]))
                    digest.update(repr(state["norm_eps"]).encode())
                    seen.add(digest.digest())
                    occurrences += 1
                if group_index == 0 and layer in (
                    config["layers"][0],
                    config["layers"][-1],
                ):
                    for path in math.snapshot_dependencies(paths):
                        payload = path.read_bytes()
                        uncompressed += len(payload)
                        compressed += len(
                            gzip.compress(payload, compresslevel=1, mtime=0)
                        )
                if convert:
                    new_paths = math.convert_shared_group(
                        paths, output / "converted", verify=True
                    )
                    start = time.monotonic()
                    restored = math.load_state(new_paths, "cpu")
                    converted_reader_seconds += time.monotonic() - start
                    start = time.monotonic()
                    math.load_state(new_paths, "cpu")
                    converted_repeat_seconds += time.monotonic() - start
                    check = math.validate_state(restored)
                    max_native_error = max(max_native_error, check["max_output_error"])
                    native_checks += 1
                    converted_groups += 1
            if groups == before_groups:
                missing_groups.append({"rid": rid, "layer": layer})
            unique += len(seen)
            rows.append({"rid": rid, "layer": layer, "unique_c4_rows": len(seen)})
        print(f"Verified {rid}: {groups} groups", flush=True)
    head_keys = {"q", "sink", "bar_o", "o", "lse"}
    head_bytes = sum(v for k, v in fields.items() if k in head_keys)
    other_bytes = sum(fields.values()) - head_bytes
    result = {
        "status": "failed" if missing_groups else "pass",
        "missing_groups": missing_groups,
        "groups": groups,
        "snapshots": snapshots,
        "payload_files": len(counted_files),
        "serialized_bytes": serialized,
        "tensor_bytes_by_field": dict(fields),
        "logical_tensor_bytes": sum(fields.values()),
        "tensor_bytes_contract": "logical field extents; aliases count per field, not per serialized storage",
        "head_specific_tensor_bytes": head_bytes,
        "other_tensor_bytes_including_shared_and_metadata": other_bytes,
        "c4_row_occurrences": occurrences,
        "unique_c4_rows_within_issue_layer": unique,
        "compression_probe_input_bytes": uncompressed,
        "compression_probe_gzip_bytes": compressed,
        "compression_probe_policy": "first position, first/last layer, every issue; not whole-dataset ratio",
        "reader_first_pass_seconds": first_seconds,
        "reader_repeat_seconds": warm_seconds,
        "reader_caveat": "OS cache not dropped; first pass is not a cold-cache measurement",
        "source_payload_digest": source.hexdigest(),
        "runner_sha256": runner_hash,
        "converted_groups": converted_groups,
        "native_checks": native_checks,
        "converted_reader_seconds": converted_reader_seconds,
        "converted_repeat_seconds": converted_repeat_seconds,
        "max_native_output_error": max_native_error,
        "config_sha256": hashlib.sha256(
            (root / "capture-config.json").read_bytes()
        ).hexdigest(),
        "wall_seconds": time.monotonic() - began,
    }
    if converted_groups:
        result["converted_bytes"] = sum(
            p.stat().st_size for p in (output / "converted").rglob("*.pt")
        )
    output.mkdir(parents=True, exist_ok=True)
    math.atomic_json(output / "storage.json", result)
    with (output / "storage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rid", "layer", "unique_c4_rows"])
        writer.writeheader()
        writer.writerows(rows)
    if missing_groups:
        missing = ", ".join(
            f'{row["rid"]}/layer-{row["layer"]:02d}' for row in missing_groups
        )
        raise ValueError(
            f"missing capture groups: {missing}; partial diagnostics: {output}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-check", action="store_true")
    parser.add_argument("--batch-reader-check", action="store_true")
    parser.add_argument(
        "--convert",
        action="store_true",
        help="also write and validate a shared-format copy; audit alone writes only reports",
    )
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("large conversion outputs must be outside the repository")
    if sum((args.reader_check, args.batch_reader_check, args.convert)) > 1:
        parser.error("select one reader measurement")
    operation = (
        benchmark_batch_view
        if args.batch_reader_check
        else (benchmark_readers if args.reader_check else audit)
    )
    options = {"convert": args.convert} if operation is audit else {}
    print(
        json.dumps(operation(args.root, args.runner, args.output, **options), indent=2)
    )


def benchmark_readers(root: Path, runner: Path, output: Path) -> dict:
    """Compare real training states; use previously verified converted tensors.

    Publishing is setup, not read latency. Two passes reverse reader order;
    neither establishes cold-cache performance. Inputs remain immutable.
    """
    from .dataset import CaptureReader, load_states

    math = artifacts.load_runner(runner)
    torch.set_num_threads(4)
    converted = output / "converted"
    if not converted.is_dir():
        raise ValueError("run the exact conversion audit first")
    original_config = json.loads((root / "capture-config.json").read_text())
    config = {
        **original_config,
        "storage_schema_version": 2,
        "audit_shared": False,
        "conversion_only": True,
        "original_config_sha256": math.file_sha256(root / "capture-config.json"),
    }
    math.atomic_json(converted / "capture-config.json", config)
    published = output / "published"
    for rid in sorted(config["requests"]):
        for rank in range(4):
            bank = json.loads(
                (root / f"rank-{rank}" / rid / "reservoir.json").read_text()
            )
            math.atomic_json(converted / f"rank-{rank}" / rid / "reservoir.json", bank)
        result = json.loads((root / "done" / f"{rid}.json").read_text())
        result["original_receipt_sha256"] = math.file_sha256(
            root / "done" / f"{rid}.json"
        )
        math.checkpoint_issue(converted, config, result)
        math.publish_issue(converted, published, rid)
        print(f"Published conversion bundle: {rid}", flush=True)
    selected_layers = _representative_layers(config)
    reader = CaptureReader(runner, published, sorted(config["requests"]))
    records = []
    for pass_index in range(2):
        for rid, request in sorted(config["requests"].items()):
            for layer in selected_layers:
                states = {}
                times = {}
                for kind in (
                    ("legacy", "shared") if pass_index == 0 else ("shared", "legacy")
                ):
                    start = time.monotonic()
                    if kind == "legacy":
                        subset = {**original_config, "requests": {rid: request}}
                        states[kind] = load_states(
                            math, root, subset, layer, request["split"]
                        )
                    else:
                        states[kind] = reader.load_views(
                            output / "scratch", rid, [layer]
                        )[layer]
                    times[kind] = time.monotonic() - start
                if len(states["legacy"]) != len(states["shared"]):
                    raise ValueError("training reader coverage differs")
                for old, new in zip(states["legacy"], states["shared"]):
                    if set(old) != set(new):
                        raise ValueError("training view fields differ")
                    for key, value in old.items():
                        same = (
                            torch.equal(value, new[key])
                            if isinstance(value, torch.Tensor)
                            else value == new[key]
                        )
                        if not same:
                            raise ValueError(
                                f"training view changed: {rid}/{layer}/{key}"
                            )
                records.append(
                    {
                        "pass": pass_index + 1,
                        "rid": rid,
                        "layer": layer,
                        "queries": len(states["legacy"]),
                        **times,
                    }
                )
                print(
                    f"Reader parity: pass {pass_index+1}, {rid}, layer {layer}",
                    flush=True,
                )
    report = {
        "status": "pass",
        "layers": selected_layers,
        "pairs": len(records),
        "queries_compared": sum(row["queries"] for row in records),
        "legacy_seconds": sum(row["legacy"] for row in records),
        "shared_bundle_view_seconds": sum(row["shared"] for row in records),
        "timing_contract": "bundle preflight excluded; shared includes selected-layer extraction/member hashes/native validation; legacy uses extracted source files",
        "cache_caveat": "two reversed-order passes; no cold-cache claim",
        "runner_sha256": math.file_sha256(runner),
        "records": records,
    }
    math.atomic_json(output / "training-reader.json", report)
    with (output / "training-reader.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return report


def benchmark_batch_view(root: Path, runner: Path, output: Path) -> dict:
    """Amortize bundle verification/extraction across the same three layers."""
    from .dataset import CaptureReader, load_states

    math = artifacts.load_runner(runner)
    torch.set_num_threads(4)
    config = json.loads((root / "capture-config.json").read_text())
    layers = _representative_layers(config)
    requested = sorted(config["requests"])
    if not requested:
        raise ValueError("nonempty requested capture cohort required")
    reader = CaptureReader(runner, output / "published", requested)
    records = []
    # Existing published artifacts are reused; no conversion/capture rerun.
    for rid in requested:
        start = time.monotonic()
        shared = reader.load_views(output / "scratch", rid, layers)
        seconds = time.monotonic() - start
        for layer in layers:
            subset = {**config, "requests": {rid: config["requests"][rid]}}
            legacy = load_states(
                math, root, subset, layer, config["requests"][rid]["split"]
            )
            for old, new in zip(legacy, shared[layer], strict=True):
                for key in old:
                    if not (
                        torch.equal(old[key], new[key])
                        if isinstance(old[key], torch.Tensor)
                        else old[key] == new[key]
                    ):
                        raise ValueError(
                            f"batched training view changed: {rid}/{layer}/{key}"
                        )
        records.append(
            {
                "rid": rid,
                "layers": layers,
                "queries": sum(map(len, shared.values())),
                "seconds": seconds,
            }
        )
        print(f"Batched reader parity: {rid}", flush=True)
    if [row["rid"] for row in records] != requested:
        raise ValueError(
            "batched reader coverage differs from requested capture cohort"
        )
    report = {
        "status": "pass",
        "issues": len(records),
        "records": records,
        "requested_issue_ids": requested,
        "seconds": sum(row["seconds"] for row in records),
        "queries_compared": sum(row["queries"] for row in records),
        "cache_caveat": "warm filesystem, no cold-cache claim",
        "timing_contract": "bundle preflight excluded; extraction, member hashes and numerical replay included",
        "runner_sha256": math.file_sha256(runner),
    }
    math.atomic_json(output / "training-reader-batched.json", report)
    return report


def _representative_layers(config: dict) -> list[int]:
    """First/middle/last, once each even for one- or two-layer captures."""
    layers = config["layers"]
    if not layers:
        raise ValueError("nonempty capture layers required")
    return list(dict.fromkeys(layers[i] for i in (0, len(layers) // 2, -1)))


if __name__ == "__main__":
    main()
