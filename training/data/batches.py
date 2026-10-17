"""Synchronous, batch-local device ownership and temporary warm-anchor storage.

Sources remain caller-owned, normally immutable CPU tensors backed by a mapped
cache. A staged batch is valid only inside its context, through backward. No
prefetch, dtype conversion, or worker/stream ownership is hidden here. File-backed
pages are OS-managed, not a hard process-RSS limit.
Author: Winston Cai.
"""

import operator
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch


class _StateFields(TypedDict):
    """Cached FP32 CPU inputs; RoPE frequencies are complex64.

    raw/freq select C4 rows [512,512]/[512,32]; q and both targets are
    [64,512], SWA is [128,512], and weight/sink/query_freq are [512]/[64]/[32].
    The saved denominator belongs to the original rounded Native target.
    """

    rid: str
    ordinal: int
    split: str
    raw: torch.Tensor
    weight: torch.Tensor
    eps: float
    freq: torch.Tensor
    query_freq: torch.Tensor
    q: torch.Tensor
    swa: torch.Tensor
    sink: torch.Tensor
    scale: float
    target: torch.Tensor
    reference_squared_norm: float


class TrainingState(_StateFields, total=False):
    """Role is assigned on admission; the unrounded diagnostic is never staged."""

    role: str
    target_unrounded: torch.Tensor


@dataclass
class StagedBatch:
    """Ordered state copies and same-state anchors; neither may escape scope."""

    states: list[TrainingState]
    anchors: list[dict] | None = None


class StateBatches:
    """Own at most one selected batch, plus an optional FP32 mapped anchor bank.

    Anchors contain raw-shaped row offsets and target-shaped output offsets.
    Their order and identity are frozen before training. The scratch file is
    private to this context and is not a checkpoint or reusable cache format.
    """

    def __init__(
        self,
        states: Sequence[dict],
        device: torch.device | str,
        limit: int,
        scratch_root: Path | None = None,
    ) -> None:
        if not states or limit < 1:
            raise ValueError("nonempty states and positive staging limit required")
        self.states = states
        self.device = torch.device(device)
        self.limit = limit
        self.scratch_root = scratch_root
        self.active_count = 0
        self.anchor_bytes = 0
        self._open = False
        self._temporary = None
        self._bank = None
        self._identities = None
        self._originals = None

    def __enter__(self) -> "StateBatches":
        if self._open:
            raise RuntimeError("state batches already open")
        self._open = True
        return self

    def __exit__(self, *_) -> None:
        self._open = False
        if self._bank is not None:
            self._bank._mmap.close()
            self._bank = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    @staticmethod
    def _identity(state: dict) -> tuple:
        return state.get("role"), state["rid"], state["ordinal"]

    @contextmanager
    def stage(
        self, indices: Sequence[int], *, with_anchors: bool = True
    ) -> Iterator[StagedBatch]:
        indices = tuple(operator.index(index) for index in indices)
        if not self._open or self.active_count:
            raise RuntimeError("staging requires an open, idle batch owner")
        if not 1 <= len(indices) <= self.limit or any(
            index < 0 or index >= len(self.states) for index in indices
        ):
            raise ValueError("staging indices exceed the batch limit or state range")
        batch = StagedBatch([])
        self.active_count = len(indices)
        try:
            for index in indices:
                source = self.states[index]
                if self._bank is not None and (
                    source is not self._originals[index]
                    or self._identity(source) != self._identities[index]
                ):
                    raise ValueError(
                        f"warm anchor state association changed at index {index}"
                    )
                batch.states.append(
                    {
                        key: (
                            value.to(self.device, non_blocking=False)
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for key, value in source.items()
                        if key
                        != "target_unrounded"  # Diagnostic only; never a loss input.
                    }
                )
            if self._bank is not None and with_anchors:
                batch.anchors = []
                for index, state in zip(indices, batch.states):
                    split = state["raw"].numel()
                    values = self._bank[index]
                    batch.anchors.append(
                        {
                            "state": state,
                            "rows": torch.from_numpy(
                                values[:split].reshape(state["raw"].shape)
                            ).to(self.device, non_blocking=False),
                            "output": torch.from_numpy(
                                values[split:].reshape(state["target"].shape)
                            ).to(self.device, non_blocking=False),
                        }
                    )
            yield batch
        finally:
            batch.states.clear()
            if batch.anchors is not None:
                batch.anchors.clear()
            self.active_count = 0

    def prepare_anchors(self, build: Callable[[dict], dict]) -> None:
        """Compute each warm residual once on-device, then retain only mapped bytes."""
        if not self._open or self._temporary is not None or self.active_count:
            raise RuntimeError("warm anchors require an open, unused batch owner")
        self._originals = tuple(self.states)
        self._identities = tuple(self._identity(state) for state in self.states)
        first = self.states[0]
        width = first["raw"].numel() + first["target"].numel()
        size = len(self.states) * width * np.dtype(np.float32).itemsize
        parent = Path(self.scratch_root or tempfile.gettempdir())
        if shutil.disk_usage(parent).free < size:
            raise ValueError(f"warm anchor scratch {parent} requires {size} free bytes")
        self._temporary = tempfile.TemporaryDirectory(
            prefix="remnant-anchors-", dir=parent
        )
        bank = None
        try:
            bank = np.memmap(
                Path(self._temporary.name) / "anchors.f32",
                mode="w+",
                dtype=np.float32,
                shape=(len(self.states), width),
            )
            for index in range(len(self.states)):
                with self.stage([index]) as batch:
                    state = batch.states[0]
                    anchor = build(state)
                    if anchor.get("state") is not state:
                        raise ValueError(
                            f"warm anchor builder changed state identity: {index}"
                        )
                    offset = 0
                    for name, tensor_name in (("rows", "raw"), ("output", "target")):
                        value = anchor[name]
                        if (
                            value.shape != first[tensor_name].shape
                            or value.dtype != torch.float32
                            or value.requires_grad
                        ):
                            raise ValueError(
                                f"fixed FP32 warm anchor geometry required: {index}/{name}"
                            )
                        end = offset + value.numel()
                        bank[index, offset:end] = value.cpu().numpy().reshape(-1)
                        offset = end
                    del anchor, value, state
            bank.flush()
            self._bank = bank
            self.anchor_bytes = size
        except BaseException as error:
            if bank is not None:
                bank._mmap.close()
            if isinstance(error, OSError):
                raise OSError(
                    f"cannot write {size}-byte warm anchor bank in {parent}: {error}"
                ) from error
            raise
