"""Capture views and hash-bound training-cache readers.

Large artifacts remain external; all payload/identity checks are retained.
Author: Winston Cai.
"""

import argparse
import hashlib
import json
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from .. import LAYERS, load_runner
from ..losses import rounded_output

def file_sha256(path: Path) -> str:
    """Checksum serialized data for corruption detection, not source provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


SHAPES = {
    "raw": (512, 512),
    "weight": (512,),
    "freq": (512, 32),
    "q": (64, 512),
    "swa": (128, 512),
    "sink": (64,),
    "target": (64, 512),
    "target_unrounded": (64, 512),
    "query_freq": (32,),
}
NORM_ROUNDOFF = 16 * float(np.finfo(np.float32).eps)
MAX_GROUP_BYTES = 256 * 1024**2


# Capture groups and publication reader


@dataclass(frozen=True)
class CaptureGroup:
    """Four rank paths bound to the ordinal committed by their reservoir."""

    paths: tuple[Path, ...]
    ordinal: int


@dataclass(frozen=True)
class _Publication:
    record: dict
    marker: Path
    marker_bytes: bytes
    bundle: Path
    signature: tuple[int, ...]


def _file_signature(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    # atime is excluded: reading must not invalidate an otherwise unchanged file.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@contextmanager
def _snapshot_group(math, paths, hashes):
    """Own at most one hash-checked rank group, including preload failures."""
    if math._snapshot_bytes:
        raise ValueError("snapshot reader already has caller-owned cached bytes")
    total = 0
    try:
        for path in paths:
            with path.open("rb") as stream:
                payload = stream.read(MAX_GROUP_BYTES - total + 1)
            total += len(payload)
            if total > MAX_GROUP_BYTES:
                raise ValueError(
                    f"capture rank group exceeds {MAX_GROUP_BYTES} bytes: {path}"
                )
            if hashlib.sha256(payload).hexdigest() != hashes[path]:
                raise ValueError(f"capture payload changed: {path}")
            math._snapshot_bytes[path] = payload
        yield
    finally:
        math._snapshot_bytes.clear()


def load_states(
    math,
    root,
    config,
    layer,
    split,
    per_issue=None,
    *,
    groups=None,
    payload_hashes=None,
    validate_native=False,
):
    """Build CPU states in selection order; optionally verify native replay.

    ``groups`` supplies an explicit per-issue selection without replacing the
    frozen runner's functions. Hash-bound cache building retains only one rank
    group of serialized bytes at a time, not every snapshot in the layer.
    """
    result = []
    for rid, request in config["requests"].items():
        if request["split"] != split:
            continue
        if groups is None:
            paths = math.state_paths(root, config, rid, layer)
            bank = json.loads((root / f"rank-0/{rid}/reservoir.json").read_text())
            ordinals = sorted(bank.values())
            selected = [
                CaptureGroup(tuple(group), ordinal)
                for group, ordinal in zip(paths, ordinals, strict=True)
            ]
        else:
            selected = groups[rid]
        for group in selected[:per_issue]:
            paths = group.paths
            if payload_hashes is None:
                s = math.load_state(paths, "cpu")
            else:
                with _snapshot_group(math, paths, payload_hashes):
                    s = math.load_state(paths, "cpu")
            expected = rid, layer, group.ordinal
            actual = s["rid"], s["layer"], s["ordinal"]
            if actual != expected:
                raise ValueError(
                    f"capture identity mismatch: expected {expected}, actual {actual}"
                )
            if validate_native:
                math.validate_state(s)
            if not bool(s["csa_valid"].all()) or not bool(s["swa_valid"].all()):
                raise ValueError("only fully admitted captured states supported")
            raw, q, weight = (
                s["raw_c"].float(),
                s["q"].float(),
                s["norm_weight"].float(),
            )
            swa = math.decode_native(s["swa_bytes"])
            target_unrounded = math.attention(
                q,
                torch.cat((swa, math.decode_native(s["csa_bytes"]))),
                s["sink"].float(),
                s["softmax_scale"],
            )[0]
            target = rounded_output(math, target_unrounded, s["query_freqs"])
            if float(target.norm()) < 1e-20:
                raise ValueError("near-zero reference")
            result.append(
                {
                    "rid": rid,
                    "ordinal": s["ordinal"],
                    "split": request["split"],
                    "raw": raw,
                    "weight": weight,
                    "eps": s["norm_eps"],
                    "freq": s["csa_freqs"],
                    "query_freq": s["query_freqs"],
                    "q": q,
                    "sink": s["sink"].float(),
                    "scale": s["softmax_scale"],
                    "swa": swa,
                    "target": target,
                    "target_unrounded": target_unrounded,
                    "reference_squared_norm": float(target.square().sum()),
                }
            )
    if not result:
        raise ValueError("empty requested split")
    return result


class CaptureReader:
    """Own one runner and verified publications; extract one issue at a time.

    The frozen materializer looks up publications through its module globals.
    Only this private runner instance adapts that lookup; callers never receive
    a modified module or supply a second, potentially conflicting runner. This
    synchronous reader belongs to one caller and is not shared across threads.
    """

    def __init__(self, runner: Path, published_root: Path, issues: list[str]):
        if not issues or len(set(issues)) != len(issues):
            raise ValueError("nonempty unique capture issues required")
        self._root = published_root.resolve()
        self._runner = load_runner(runner)
        self._config_bytes = (self._root / "capture-config.json").read_bytes()
        self._config = json.loads(self._config_bytes)
        self._verified: dict[str, _Publication] = {}
        missing = []
        for rid in issues:
            if rid not in self._config["requests"]:
                raise ValueError(f"issue absent from frozen capture: {rid}")
            marker = self._root / "published" / f"{rid}.json"
            if not marker.exists():
                missing.append(rid)
                continue
            try:
                marker_bytes = marker.read_bytes()
                metadata = json.loads(marker_bytes)
                bundle = (self._root / metadata["bundle"]).resolve()
                if not bundle.is_relative_to(self._root):
                    raise ValueError("bundle escapes publication root")
                before = _file_signature(bundle)
                record = self._runner.published_issue(self._root, rid)
                if (
                    _file_signature(bundle) != before
                    or marker.read_bytes() != marker_bytes
                    or record is None
                ):
                    raise ValueError("publication changed during verification")
                if (
                    record["receipt"]["capture_config_sha256"]
                    != hashlib.sha256(self._config_bytes).hexdigest()
                ):
                    raise ValueError(
                        "publication belongs to a different capture configuration"
                    )
                self._verified[rid] = _Publication(
                    record, marker, marker_bytes, bundle, before
                )
            except (OSError, ValueError, KeyError) as error:
                raise ValueError(
                    f"invalid published capture for issue {rid}: {error}"
                ) from error
        if missing:
            raise ValueError(
                f"missing published captures for requested issues: {', '.join(missing)}"
            )

    def _publication(self, destination: Path, rid: str) -> dict:
        if Path(destination).resolve() != self._root or rid not in self._verified:
            raise ValueError(f"publication outside verified reader coverage: {rid}")
        proof = self._verified[rid]
        if (
            proof.marker.read_bytes() != proof.marker_bytes
            or _file_signature(proof.bundle) != proof.signature
            or (self._root / "capture-config.json").read_bytes() != self._config_bytes
        ):
            raise ValueError(f"published capture changed after verification: {rid}")
        return proof.record

    def load_views(
        self, scratch_root: Path, rid: str, layers, allow_test: bool = False
    ) -> dict[int, list[dict]]:
        """Return validated CPU states; always remove owned extraction files."""
        layers = tuple(layers)
        if (
            not layers
            or len(set(layers)) != len(layers)
            or not set(layers) <= set(LAYERS)
        ):
            raise ValueError("unique known captured layers required")
        self._publication(self._root, rid)
        request = self._config["requests"][rid]
        if request["split"] == "test" and not allow_test:
            raise ValueError("test data is excluded from fitting views by default")
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="capture-view-", dir=scratch_root
        ) as owned:
            root = Path(owned) / "issue"
            original_lookup = self._runner.published_issue
            try:
                self._runner.published_issue = self._publication
                self._runner.materialize_issue(self._root, rid, root, layers=layers)
            finally:
                self._runner.published_issue = original_lookup
            subset = {**self._config, "requests": {rid: request}}
            result = {
                layer: load_states(
                    self._runner,
                    root,
                    subset,
                    layer,
                    request["split"],
                    validate_native=True,
                )
                for layer in layers
            }
            self._publication(self._root, rid)
            return result


# Hash-bound cached states and validation


def validate_states(states, split, train_queries=4):
    if train_queries not in (4, 10):
        raise ValueError("explicit four- or ten-query training coverage required")
    if set(states) != {"train", "development"}:
        raise ValueError("explicit train/development roles required")
    identities = {}
    for role, count in (("train", train_queries), ("development", 10)):
        expected = set(split[role])
        rows = states[role]
        keys = [(s["rid"], s["ordinal"]) for s in rows]
        if len(keys) != len(expected) * count or len(set(keys)) != len(keys):
            raise ValueError("complete unique query coverage required")
        if {rid for rid, _ in keys} != expected or any(
            sum(rid == issue for rid, _ in keys) != count for issue in expected
        ):
            raise ValueError("frozen issue coverage changed")
        identities[role] = sorted(keys)
        for state in rows:
            if state["role"] != role:
                raise ValueError("state role mismatch")
            for name, shape in SHAPES.items():
                tensor = state[name]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tensor.device.type != "cpu"
                    or tuple(tensor.shape) != shape
                ):
                    raise ValueError(f"full captured CPU geometry required: {name}")
                if not bool(torch.isfinite(tensor).all()):
                    raise ValueError("nonfinite cached tensor")
            if any(
                state[name].dtype != torch.complex64 for name in ("freq", "query_freq")
            ):
                raise ValueError("original complex64 RoPE frequencies required")
            if any(
                state[name].dtype != torch.float32
                for name in SHAPES
                if name not in ("freq", "query_freq")
            ):
                raise ValueError("original FP32 training tensors required")
            denominator = state["reference_squared_norm"]
            if (
                not np.isfinite(denominator)
                or denominator <= 0
                or denominator != float(state["target"].square().sum())
            ):
                raise ValueError("native target denominator changed")
            if (
                not np.isfinite(state["eps"])
                or state["eps"] <= 0
                or not np.isfinite(state["scale"])
                or state["scale"] <= 0
            ):
                raise ValueError(
                    "positive captured normalization/attention constants required"
                )
    if set(split["train"]) & set(split["development"]) or split["test"]:
        raise ValueError("disjoint development split and no fresh-test claim required")
    return identities


def validate_migrated_states(states, split, train_queries=4):
    structural = {}
    for role, rows in states.items():
        structural[role] = []
        for state in rows:
            target = state["target"]
            saved = state["reference_squared_norm"]
            exact = float(target.double().square().sum())
            if (
                not np.isfinite(saved)
                or exact <= 0
                or abs(saved - exact) > NORM_ROUNDOFF * exact
            ):
                raise ValueError(
                    "saved native denominator exceeds FP32 reduction roundoff"
                )
            # Shallow copies are validation-only. Do not mutate the payload,
            # target tensors or denominator used by forward_error.
            structural[role].append(
                {**state, "reference_squared_norm": float(target.square().sum())}
            )
    return validate_states(structural, split, train_queries=train_queries)


def _mapped_states(path: Path) -> dict:
    """Map immutable cache storage; never fall back to an eager tensor load."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as error:
        raise ValueError(
            f"Cannot memory-map training cache {path}: {error}. "
            "Use the default ZIP serialization of torch.save; "
            "no eager-loading fallback is enabled."
        ) from error


def load_layer(
    cache,
    layer,
    expected_config_sha256,
    expected_record_sha256,
    *,
    allow_norm_roundoff=False,
):
    """One receipt/payload reader; retain both historical norm contracts."""
    if (
        layer not in LAYERS
        or file_sha256(cache / "config.json") != expected_config_sha256
    ):
        raise ValueError("cache layer/configuration changed")
    config = json.loads((cache / "config.json").read_text())
    record_path = cache / f"layer-{layer:02d}.json"
    if file_sha256(record_path) != expected_record_sha256:
        raise ValueError("cached layer receipt changed")
    record = json.loads(record_path.read_text())
    path = cache / f"layer-{layer:02d}.pt"
    if (
        record["config_sha256"] != expected_config_sha256
        or record["layer"] != layer
        or file_sha256(path) != record["states_sha256"]
    ):
        raise ValueError("cached payload checksum changed")
    states = _mapped_states(path)
    validate = validate_migrated_states if allow_norm_roundoff else validate_states
    identities = validate(
        states, config["split"], config.get("train_queries_per_issue", 4)
    )
    if json.loads(json.dumps(identities)) != record["identities"]:
        raise ValueError("cached query association changed")
    return states


def load_cached_layer(cache: Path, layer: int) -> dict:
    """Use existing receipts and the existing cross-platform norm check."""
    progress = json.loads((cache / "progress.json").read_text())
    if file_sha256(cache / "index.json") != progress["index_sha256"]:
        raise ValueError("cache index changed")
    index = json.loads((cache / "index.json").read_text())
    return load_layer(
        cache,
        layer,
        progress["config_sha256"],
        index[str(layer)]["record_sha256"],
        allow_norm_roundoff=True,
    )


# Historical frozen-cohort tools: 50 issues and the layer-8/12 extension


def frozen_split(ids, seed=42):
    ids = sorted(ids)
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("exactly 50 unique development issues required")
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    return {
        "seed": seed,
        "train": shuffled[:40],
        "development": shuffled[40:],
        "test": [],
        "test_status": "Fresh independent cohort required after freezing candidate",
    }


def _capture_reservoir(root: Path, rid: str, receipt: dict) -> tuple[dict, dict]:
    """One owner for the four-rank reservoir alignment and hash contract."""
    bank, hashes = None, {}
    for rank in range(4):
        path = root / f"rank-{rank}" / rid / "reservoir.json"
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        name = str(path.relative_to(root))
        if digest != receipt["files_sha256"][name]:
            raise ValueError(f"capture reservoir changed: {name}")
        other = json.loads(payload)
        if bank is not None and bank != other:
            raise ValueError(f"rank reservoir mismatch: {rid}")
        bank = other
        hashes[name] = digest
    return bank, hashes


def _capture_slots(bank: dict, role: str, train_queries: int) -> list[str]:
    """Keep original ordinal order and the frozen equally spaced training views."""
    slots = sorted(bank, key=lambda key: bank[key])
    if role == "train":
        slots = [
            slots[i] for i in np.linspace(0, len(slots) - 1, train_queries, dtype=int)
        ]
    return slots


def preload_layer(
    math, root, config, split, layer, roles=("train", "development"), train_queries=4
):
    """Build a frozen layer with bounded serialized input, preserving targets."""
    if train_queries not in (4, 10):
        raise ValueError("explicit four- or ten-query training coverage required")
    groups, hashes = {}, {}
    for role in roles:
        for rid in split[role]:
            receipt = json.loads((root / "done" / f"{rid}.json").read_text())
            if receipt["finish_reason"] not in {"stop", "tool_calls"}:
                raise ValueError("non-natural completion")
            bank, _ = _capture_reservoir(root, rid, receipt)
            slots = _capture_slots(bank, role, train_queries)
            groups[rid] = []
            for slot in slots:
                group = []
                for rank in range(4):
                    path = (
                        root
                        / f"rank-{rank}"
                        / rid
                        / f"layer-{layer:02d}-slot-{int(slot):02d}.pt"
                    )
                    group.append(path)
                    hashes[path] = receipt["files_sha256"][str(path.relative_to(root))]
                groups[rid].append(CaptureGroup(tuple(group), bank[slot]))
    states = {}
    for role in roles:
        subset = {
            **config,
            "requests": {rid: config["requests"][rid] for rid in split[role]},
        }
        states[role] = load_states(
            math, root, subset, layer, "held_out", groups=groups, payload_hashes=hashes
        )
        for state in states[role]:
            state["role"] = role
    return states


def capture_sources(
    root, split, layer, train_queries=4, roles=("train", "development")
):
    """Expected hashes of the payloads actually sampled by preload_layer."""
    sources = {}
    if train_queries not in (4, 10):
        raise ValueError("explicit four- or ten-query training coverage required")
    for role in roles:
        for rid in split[role]:
            receipt_path = root / "done" / f"{rid}.json"
            receipt = json.loads(receipt_path.read_text())
            sources[str(receipt_path.relative_to(root))] = file_sha256(
                receipt_path
            )
            if receipt["finish_reason"] not in {"stop", "tool_calls"}:
                raise ValueError("natural completed capture required")
            bank, reservoir_hashes = _capture_reservoir(root, rid, receipt)
            sources.update(reservoir_hashes)
            if len(bank) != 10:
                raise ValueError("ten committed queries per issue required")
            slots = _capture_slots(bank, role, train_queries)
            for slot in slots:
                for rank in range(4):
                    name = (
                        f"rank-{rank}/{rid}/layer-{layer:02d}-slot-{int(slot):02d}.pt"
                    )
                    sources[name] = receipt["files_sha256"][name]
    return sources


def build(root, fit_path, runner, seconds=2100):
    torch.set_num_threads(4)
    fit = json.loads(fit_path.read_text())
    capture = json.loads((root / "capture-config.json").read_text())
    validation = json.loads((root / "validation/aggregate.json").read_text())
    split = frozen_split(capture["requests"])
    if (
        fit["status"] != "complete"
        or validation["status"] != "complete"
        or validation["issues"] != 50
    ):
        raise ValueError("complete frozen 50-issue capture/fit required")
    config = {
        "schema": 1,
        "split": split,
        "layers": list(LAYERS),
        "train_queries_per_issue": 4,
        "development_queries_per_issue": 10,
        "heads": 64,
        "survivors": 256,
        "record_bytes": 328,
        "recipe": "Unchanged preload_layer with four-rank hash/alignment validation and scalar native-byte target replay",
        "fresh_test_completed": False,
        "torch": torch.__version__,
    }
    math = load_runner(runner)
    output = fit_path.parent / "training-cache-001"
    output.mkdir(exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("cache source/configuration changed")
    math.atomic_json(config_path, config)
    digest = file_sha256(config_path)
    index_path = output / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    start = time.monotonic()
    completed = []
    for layer in LAYERS:
        if (output / f"layer-{layer:02d}.json").exists():
            if str(layer) not in index:
                raise ValueError("completed cache receipt missing from index")
            load_layer(output, layer, digest, index[str(layer)]["record_sha256"])
        else:
            if time.monotonic() - start >= seconds:
                break
            sources = capture_sources(root, split, layer)
            states = preload_layer(math, root, capture, split, layer)
            identities = validate_states(states, split)
            path = output / f"layer-{layer:02d}.pt"
            temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
            torch.save(states, temporary)
            temporary.replace(path)
            math.atomic_json(
                output / f"layer-{layer:02d}.json",
                {
                    "layer": layer,
                    "config_sha256": digest,
                    "states_sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                    "identities": identities,
                    "capture_sources": sources,
                },
            )
            index[str(layer)] = {
                "record_sha256": file_sha256(output / f"layer-{layer:02d}.json"),
                "states_sha256": file_sha256(path),
            }
            math.atomic_json(index_path, index)
            # Verify deserialization exactly; no codec or dtype conversion on migration.
            restored = load_layer(
                output, layer, digest, index[str(layer)]["record_sha256"]
            )
            for role in states:
                for original, replay in zip(states[role], restored[role]):
                    for name in SHAPES:
                        if not torch.equal(original[name], replay[name]):
                            raise ValueError("serialization changed a training tensor")
            del states, restored
        completed.append(layer)
        progress = {
            "status": "complete" if tuple(completed) == LAYERS else "partial",
            "layers": completed,
            "config_sha256": digest,
            "index_sha256": file_sha256(index_path),
            "seconds": time.monotonic() - start,
            "bytes": sum(
                (output / f"layer-{n:02d}.pt").stat().st_size for n in completed
            ),
            "fresh_test_completed": False,
        }
        math.atomic_json(output / "progress.json", progress)
        print(
            f"Training cache layer {layer}: {len(completed)}/21 committed", flush=True
        )
    return json.loads((output / "progress.json").read_text())


def merge_training_views(base, regenerated):
    """Author: Winston Cai. Add views, retaining every frozen target/norm.

    Shared inputs must be bit-exact. Regenerated targets are not authoritative
    for old queries: FP32 attention can cross BF16 rounding boundaries on a
    different CPU. Reuse each entire old state, including its denominator.
    """
    original = {(s["rid"], s["ordinal"]): s for s in base["train"]}
    keys = [(s["rid"], s["ordinal"]) for s in regenerated]
    if (
        len(original) != len(base["train"])
        or len(keys) != len(set(keys))
        or not set(original) <= set(keys)
    ):
        raise ValueError("unique expanded coverage must contain every original query")
    kept, result = 0, []
    for state in regenerated:
        old = original.get((state["rid"], state["ordinal"]))
        if old is not None:
            if any(
                not torch.equal(state[k], old[k])
                for k in SHAPES
                if k not in ("target", "target_unrounded")
            ) or any(
                state[k] != old[k]
                for k in ("rid", "ordinal", "eps", "scale", "role", "split")
            ):
                raise ValueError("expanded reader changed an original numerical input")
            result.append(old)
            kept += 1
        else:
            result.append(state)
    return {"train": result, "development": base["development"]}, kept


def verify_preserved_views(base, expanded):
    """Author: Winston Cai. Independent migration check, including saved norms."""
    counts = {}
    for role in ("train", "development"):
        old = {(s["rid"], s["ordinal"]): s for s in base[role]}
        new = {(s["rid"], s["ordinal"]): s for s in expanded[role]}
        if (
            len(old) != len(base[role])
            or len(new) != len(expanded[role])
            or not set(old) <= set(new)
            or (role == "development" and set(old) != set(new))
        ):
            raise ValueError("original unique query coverage changed")
        for key, state in old.items():
            other = new[key]
            if any(
                other[k] != state[k]
                for k in (
                    "rid",
                    "ordinal",
                    "role",
                    "split",
                    "eps",
                    "scale",
                    "reference_squared_norm",
                )
            ) or any(
                other[k].dtype != state[k].dtype
                or other[k].stride() != state[k].stride()
                or not torch.equal(other[k], state[k])
                for k in SHAPES
            ):
                raise ValueError(
                    f"original {role} state changed during coverage extension: {key}"
                )
        counts[role] = len(old)
    return counts


def extend_coverage(root, runner, base_cache, output, layers=(8, 12)):
    """Author: Winston Cai. CPU-only extension of existing captured queries.

    No new issues, model inference or replacement targets. The complete
    base receipt owns the original four training views and all development
    tensors. Newly admitted views come only from hash-bound raw captures.
    """
    if tuple(layers) != (8, 12) or output.exists():
        raise ValueError("unique CPU-only two-layer coverage extension required")
    progress = json.loads((base_cache / "progress.json").read_text())
    if progress["status"] != "complete" or progress["layers"] != list(LAYERS):
        raise ValueError("complete all-layer base cache required")
    for name, key in (("config.json", "config_sha256"), ("index.json", "index_sha256")):
        if file_sha256(base_cache / name) != progress[key]:
            raise ValueError("base metadata changed")
    base_config = json.loads((base_cache / "config.json").read_text())
    if base_config["train_queries_per_issue"] != 4:
        raise ValueError("original four-query recipe required")
    capture = json.loads((root / "capture-config.json").read_text())
    split = base_config["split"]
    if frozen_split(capture["requests"]) != split:
        raise ValueError("frozen issue split changed")
    torch.set_num_threads(4)
    math = load_runner(runner)
    index = json.loads((base_cache / "index.json").read_text())
    config = {
        **{
            key: base_config[key]
            for key in (
                "schema", "split", "heads", "survivors", "record_bytes",
                "development_queries_per_issue", "fresh_test_completed",
            )
            if key in base_config
        },
        "layers": list(layers),
        "train_queries_per_issue": 10,
        "torch": str(torch.__version__),
        "recipe": "Six additional captured training queries per issue; preserve every original training/development state and denominator",
    }
    output.mkdir(parents=True)
    math.atomic_json(output / "config.json", config)
    digest = file_sha256(output / "config.json")
    extended_index, completed, started = {}, [], time.monotonic()
    for layer in layers:
        base = load_layer(
            base_cache,
            layer,
            progress["config_sha256"],
            index[str(layer)]["record_sha256"],
            allow_norm_roundoff=True,
        )
        sources = capture_sources(
            root, split, layer, train_queries=10, roles=("train",)
        )
        checks, original_load = [], math.load_state

        def checked_load(paths, device):
            state = original_load(paths, device)
            checks.append(
                {
                    "rid": state["rid"],
                    "ordinal": state["ordinal"],
                    **math.validate_state(state),
                }
            )
            return state

        math.load_state = checked_load
        try:
            regenerated = preload_layer(
                math, root, capture, split, layer, roles=("train",), train_queries=10
            )["train"]
        finally:
            math.load_state = original_load
        states, preserved = merge_training_views(base, regenerated)
        identities = validate_migrated_states(states, split, train_queries=10)
        path = output / f"layer-{layer:02d}.pt"
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        torch.save(states, temporary)
        temporary.replace(path)
        record = {
            "layer": layer,
            "config_sha256": digest,
            "states_sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "identities": identities,
            "capture_sources": sources,
            "preserved_train_queries": preserved,
            "preserved_development_queries": len(base["development"]),
            "new_train_queries": len(regenerated) - preserved,
            "native_replay_checks": checks,
        }
        receipt_path = output / f"layer-{layer:02d}.json"
        math.atomic_json(receipt_path, record)
        extended_index[str(layer)] = {
            "record_sha256": file_sha256(receipt_path),
            "states_sha256": record["states_sha256"],
        }
        math.atomic_json(output / "index.json", extended_index)
        replay = load_layer(
            output,
            layer,
            digest,
            extended_index[str(layer)]["record_sha256"],
            allow_norm_roundoff=True,
        )
        if verify_preserved_views(base, replay) != {"train": 160, "development": 100}:
            raise ValueError("complete original query preservation required")
        for role in states:
            for old, new in zip(states[role], replay[role]):
                if (
                    any(not torch.equal(old[k], new[k]) for k in SHAPES)
                    or old["reference_squared_norm"] != new["reference_squared_norm"]
                ):
                    raise ValueError(
                        "coverage serialization changed original tensors or norm"
                    )
        completed.append(layer)
        result = {
            "status": "complete" if completed == list(layers) else "partial",
            "layers": completed[:],
            "config_sha256": digest,
            "index_sha256": file_sha256(output / "index.json"),
            "seconds": time.monotonic() - started,
            "torch": str(torch.__version__),
            "gpu_requested": False,
            "fresh_test_completed": False,
        }
        math.atomic_json(output / "progress.json", result)
        print(
            f"Coverage layer {layer}: 400 training/100 development queries; {preserved} original training targets preserved",
            flush=True,
        )
        del states, base, regenerated, replay
    return result


# CLI: explicit choice of published view or historical cache operation


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fit", dest="fit_path", type=Path)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--extend-base", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--published-view", action="store_true")
    parser.add_argument("--issue")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(arguments)
    if args.published_view:
        if (
            args.output is None
            or args.issue is None
            or args.layer is None
            or args.fit_path is not None
            or args.extend_base is not None
        ):
            parser.error(
                "published view requires --issue, --layer and --output scratch directory, not fitting inputs"
            )
        reader = CaptureReader(args.runner, args.root, [args.issue])
        return reader.load_views(
            args.output, args.issue, [args.layer], args.allow_test
        )[args.layer]
    if args.extend_base is None:
        if (
            args.fit_path is None
            or args.output is not None
        ):
            parser.error("ordinary cache build requires --fit only")
        return build(root=args.root, fit_path=args.fit_path, runner=args.runner)
    if args.fit_path is not None or args.output is None:
        parser.error(
            "coverage extension requires --extend-base and --output, not --fit"
        )
    return extend_coverage(
        args.root, args.runner, args.extend_base, args.output
    )


if __name__ == "__main__":
    main()
