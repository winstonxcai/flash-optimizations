"""Maintained, model-free attention-pruning tools. Author: Winston Cai."""

import importlib.util
from pathlib import Path
from types import ModuleType

LAYERS = tuple(range(2, 43, 2))


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(x) for x in value.split(","))
    if not layers or len(set(layers)) != len(layers) or not set(layers) <= set(LAYERS):
        raise ValueError("unique captured layers 2,4,...,42 required")
    return layers


def load_runner(path: Path) -> ModuleType:
    """Load the selected calibration code; no historical source/version gate."""
    spec = importlib.util.spec_from_file_location("storage_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
