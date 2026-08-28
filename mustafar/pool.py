"""Fixed-stride storage object shared by the reference and Triton paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import config


@dataclass
class PackedC4Pool:
    packed_values: torch.Tensor
    bitmap: torch.Tensor
    packed_scales: torch.Tensor

    @classmethod
    def allocate(
        cls,
        num_pages: int,
        page_size: int,
        *,
        device: torch.device | str = "cpu",
    ) -> "PackedC4Pool":
        prefix = (num_pages, page_size)
        return cls(
            packed_values=torch.zeros(
                *prefix, config.KEEP_DIM, dtype=torch.uint8, device=device
            ),
            bitmap=torch.zeros(
                *prefix, config.BITMAP_WORDS, dtype=torch.uint64, device=device
            ),
            packed_scales=torch.zeros(
                *prefix, config.SCALE_TILES, dtype=torch.uint8, device=device
            ),
        )

    @property
    def capacity_rows(self) -> int:
        return self.packed_values.numel() // config.KEEP_DIM

    @property
    def bytes_per_row(self) -> int:
        return config.PACKED_BYTES_PER_ROW

    @property
    def storage_bytes(self) -> int:
        return sum(
            tensor.untyped_storage().nbytes()
            for tensor in (self.packed_values, self.bitmap, self.packed_scales)
        )

    def get_buf_infos(self):
        tensors = (self.packed_values, self.bitmap, self.packed_scales)
        return (
            [tensor.data_ptr() for tensor in tensors],
            [tensor.nbytes for tensor in tensors],
            [config.VALUES_BYTES, config.BITMAP_BYTES, config.SCALES_BYTES],
        )

    def validate(self) -> None:
        rows = self.capacity_rows
        if self.bitmap.numel() != rows * config.BITMAP_WORDS:
            raise ValueError("bitmap capacity does not match packed_values")
        if self.packed_scales.numel() != rows * config.SCALE_TILES:
            raise ValueError("packed_scales capacity does not match packed_values")
        if self.storage_bytes != rows * config.PACKED_BYTES_PER_ROW:
            raise ValueError("packed pool storage is not fixed-stride 328 B/row")
