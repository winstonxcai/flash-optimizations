"""Shared frozen-source, checkpoint and policy artifact contracts.

No training, evaluation or storage-audit dependencies belong here.
Author: Winston Cai.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

LAYERS = tuple(range(2, 43, 2))
_REFERENCE_PACKAGE = "_remnant_calibration_reference"
# One immutable numerical revision per process. Hashes describe the bytes
# actually compiled below, not whatever later happens to occupy those paths.
_REFERENCE_MODULES: dict[str, tuple[ModuleType, str]] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner(path: Path, *, sources: dict[str, str] | None = None) -> ModuleType:
    """Compile a fresh runner; bind reference imports to one immutable revision."""
    spec = importlib.util.spec_from_file_location("storage_calibration", path)
    module = importlib.util.module_from_spec(spec)
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if sources is not None and digest != sources.get("benchmark/remnant/calibrate.py"):
        raise ValueError(f"frozen runner source changed before execution: {path}")
    _bind_numerical_sources(path.parents[2] / "srt/remnant", sources)
    exec(compile(source, str(path), "exec"), module.__dict__)
    module._source_sha256 = digest
    return module


def _bind_numerical_sources(root: Path, expected: dict[str, str] | None) -> None:
    """Load checked source bytes once, without trusting timestamp-based pyc files."""
    sources = {
        name: (root / f"{name}.py").read_bytes()
        for name in ("config", "bitmap", "reference")
    }
    if expected is not None:
        for name, source in sources.items():
            if hashlib.sha256(source).hexdigest() != expected.get(
                f"srt/remnant/{name}.py"
            ):
                raise ValueError(
                    f'frozen numerical source changed before execution: {root / (name + ".py")}'
                )
    if _REFERENCE_MODULES:
        for name, source in sources.items():
            module, digest = _REFERENCE_MODULES[name]
            if (
                hashlib.sha256(source).hexdigest() != digest
                or sys.modules.get(module.__name__) is not module
            ):
                raise ValueError(
                    f'loaded numerical dependency differs from frozen source: {root / (name + ".py")}; '
                    "use a fresh Python process for a different numerical revision"
                )
        return
    if any(
        name == _REFERENCE_PACKAGE or name.startswith(_REFERENCE_PACKAGE + ".")
        for name in sys.modules
    ):
        raise ValueError(
            "unowned numerical imports already exist; use a fresh Python process"
        )
    package = ModuleType(_REFERENCE_PACKAGE)
    package.__path__ = [str(root)]
    sys.modules[_REFERENCE_PACKAGE] = package
    try:
        for name, source in sources.items():
            fullname = f"{_REFERENCE_PACKAGE}.{name}"
            path = root / f"{name}.py"
            spec = importlib.util.spec_from_file_location(fullname, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            exec(compile(source, str(path), "exec"), module.__dict__)
            setattr(package, name, module)
            _REFERENCE_MODULES[name] = (module, hashlib.sha256(source).hexdigest())
    except BaseException:
        for name in tuple(sys.modules):
            if name == _REFERENCE_PACKAGE or name.startswith(_REFERENCE_PACKAGE + "."):
                del sys.modules[name]
        _REFERENCE_MODULES.clear()
        raise


def load_reference(runner: ModuleType, sources: dict[str, str]) -> ModuleType:
    """Check executed-source digests and current files against the frozen map."""
    if runner._source_sha256 != sources.get("benchmark/remnant/calibrate.py"):
        raise ValueError("executed runner differs from frozen source")
    reference = runner.reference()
    package = reference.__package__
    for name in ("reference", "config", "bitmap"):
        key = f"srt/remnant/{name}.py"
        module = sys.modules.get(f"{package}.{name}")
        expected = sources.get(key)
        path = getattr(module, "__file__", None)
        if not expected or path is None:
            raise ValueError(f"missing frozen or loaded numerical dependency: {key}")
        owned, actual = _REFERENCE_MODULES[name]
        if owned is not module or actual != expected or sha256(Path(path)) != actual:
            raise ValueError(
                f"loaded numerical dependency differs from frozen source: {path}; "
                f"expected SHA-256 {expected}, actual {actual}; "
                "run this source revision in a fresh Python process"
            )
    return reference


def _cache_provenance(
    cache: Path, layers: tuple[int, ...]
) -> tuple[dict[str, str], dict[str, str]]:
    """Bind small receipts without mapping or retaining any layer tensors.

    The layer reader checks each payload before use. Publication checks those
    same expected bytes again, including layers whose mappings were released.
    """
    metadata, hashes = {}, {}
    for name in ("config.json", "index.json", "progress.json"):
        source = (cache / name).read_bytes()
        hashes[name] = hashlib.sha256(source).hexdigest()
        metadata[name] = json.loads(source)
    config, index, progress = (
        metadata[name] for name in ("config.json", "index.json", "progress.json")
    )
    if (
        progress["config_sha256"] != hashes["config.json"]
        or progress["index_sha256"] != hashes["index.json"]
    ):
        raise ValueError(f"cache configuration/index changed: {cache}")
    sources = {}
    for layer in layers:
        if (
            layer not in config["layers"]
            or layer not in progress["layers"]
            or str(layer) not in index
        ):
            raise ValueError(f"cache missing requested layer {layer}: {cache}")
        entry = index[str(layer)]
        name = f"layer-{layer:02d}.json"
        source = (cache / name).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != entry["record_sha256"]:
            raise ValueError(f"cached layer receipt changed: {cache / name}")
        record = json.loads(source)
        payload_digest = record["states_sha256"]
        if (
            record["layer"] != layer
            or record["config_sha256"] != hashes["config.json"]
            or not isinstance(payload_digest, str)
            or not payload_digest
            or entry.get("states_sha256", payload_digest) != payload_digest
        ):
            raise ValueError(f"cached payload provenance changed: {cache / name}")
        sources[name] = digest
        sources[f"layer-{layer:02d}.pt"] = payload_digest
    return hashes, sources


def run_provenance(
    cache: Path,
    policy: Path,
    runner: Path,
    *,
    executed_runner: ModuleType,
    device: str,
    threads: int,
    torch_version: str,
    layers: tuple[int, ...],
) -> dict:
    """Common numerical-run identity; callers add recipe or comparison details."""
    policy_digest = sha256(policy)
    selected = read_policy(policy)
    checkpoints = {}
    for layer in layers:
        if layer not in selected:
            raise ValueError(f"policy missing requested layer: {layer}")
        read_checkpoint(selected[layer])
        checkpoints[str(layer)] = selected[layer]
    if sha256(policy) != policy_digest:
        raise ValueError(f"policy changed during validation: {policy}")
    cache_hashes, cache_layers = _cache_provenance(cache, layers)
    validate_runner(cache, runner)
    runner_digest = sha256(runner)
    if runner_digest != executed_runner._source_sha256:
        raise ValueError(f"executed runner source changed before provenance: {runner}")
    return {
        "cache_sha256": cache_hashes,
        "cache_layers_sha256": cache_layers,
        "policy_sha256": policy_digest,
        "runner_sha256": runner_digest,
        "checkpoints": checkpoints,
        "torch": torch_version,
        "device": device,
        "threads": threads,
    }


def assert_run_inputs(identity: dict, cache: Path, policy: Path, runner: Path) -> None:
    """Do not publish results if their pinned input chain changed during replay."""
    expected = {
        cache / name: digest for name, digest in identity["cache_sha256"].items()
    }
    expected.update(
        {
            cache / name: digest
            for name, digest in identity["cache_layers_sha256"].items()
        }
    )
    expected.update(
        {policy: identity["policy_sha256"], runner: identity["runner_sha256"]}
    )
    expected.update(
        {
            Path(entry["path"]): entry["sha256"]
            for entry in identity["checkpoints"].values()
        }
    )
    for path, digest in expected.items():
        try:
            actual = sha256(path)
        except OSError as error:
            raise ValueError(f"run input changed before publication: {path}") from error
        if actual != digest:
            raise ValueError(f"run input changed before publication: {path}")


def package_sources() -> dict[str, str]:
    """Record current code separately; never rewrite historical receipts."""
    root = Path(__file__).resolve().parent
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*.py"))
        if "tests" not in p.relative_to(root).parts
    }


def validate_runner(cache: Path, runner: Path) -> None:
    """Bind the executed runner and its dependencies to frozen source hashes."""
    config = json.loads((cache / "config.json").read_text())
    sources = config.get("numerical_source", {})
    runner_key = "benchmark/remnant/calibrate.py"
    expected = sources.get(runner_key) if isinstance(sources, dict) else None
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            f"missing frozen runner SHA-256: {runner}: "
            f'required {runner_key} in {cache / "config.json"}'
        )
    try:
        actual = sha256(runner)
    except OSError as error:
        raise ValueError(
            f"cannot read numerical runner: {runner}; expected SHA-256 {expected}"
        ) from error
    if actual != expected:
        raise ValueError(
            f"frozen runner source changed: {runner}; "
            f"expected SHA-256 {expected}, actual {actual}"
        )
    for name, expected in sources.items():
        if name.startswith(("srt/remnant/", "benchmark/remnant/")):
            path = runner.parents[2] / name
            actual = sha256(path)
            if actual != expected:
                raise ValueError(
                    f"frozen numerical source changed: {path}; "
                    f"expected SHA-256 {expected}, actual {actual}"
                )


def read_policy(policy_path: Path) -> dict[int, dict]:
    """Read unique layer entries and resolve paths beside the input manifest."""
    policy = json.loads(policy_path.read_text())
    selected = policy.get("selected") if isinstance(policy, dict) else None
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"nonempty selected-policy entries required: {policy_path}")
    entries = {}
    for entry in selected:
        layer = entry.get("layer") if isinstance(entry, dict) else None
        if type(layer) is not int or layer not in LAYERS or layer in entries:
            raise ValueError(
                f"unique captured policy layers required: {policy_path}: {layer}"
            )
        if (
            not isinstance(entry.get("path"), str)
            or not entry["path"]
            or not isinstance(entry.get("sha256"), str)
        ):
            raise ValueError(
                f"checkpoint path and SHA-256 required: {policy_path}: layer {layer}"
            )
        path = Path(entry["path"])
        if not path.is_absolute():
            path = policy_path.parent / path
        entries[layer] = {**entry, "path": str(path.resolve())}
    return entries


def read_checkpoint(entry: dict) -> dict:
    """Verify the existing checkpoint contract without changing its contents."""
    layer = entry["layer"]
    path = Path(entry["path"])
    try:
        source = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read policy checkpoint: layer {layer}: {path}"
        ) from error
    actual = hashlib.sha256(source).hexdigest()
    if actual != entry["sha256"]:
        raise ValueError(
            f"policy checkpoint changed: layer {layer}: {path}; "
            f'expected {entry["sha256"]}, actual {actual}'
        )
    # Parse exactly the snapshot whose digest was verified, not a second read.
    record = json.loads(source)
    if (
        not isinstance(record, dict)
        or record.get("layer") != layer
        or not isinstance(record.get("selected_parameters"), dict)
    ):
        raise ValueError(f"complete selected checkpoint required: layer {layer}")
    return record


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(x) for x in value.split(","))
    if not layers or len(set(layers)) != len(layers) or not set(layers) <= set(LAYERS):
        raise ValueError("unique captured layers 2,4,...,42 required")
    return layers
