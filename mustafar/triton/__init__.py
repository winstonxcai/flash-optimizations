"""Triton kernels for physically-sparse TopMag compression (Stage 0).

Import the kernels directly (e.g. `from .triton import _pack_ccomp_kernel`).
No torch host logic lives in this subfolder — see mustafar/sparse.py.
"""
from .kernels import (
    _bf16_to_native_c4_kernel,
    _pack_c4_fp8_kernel,
    _pack_ccomp_kernel,
    _rope_tail_inplace_kernel,
    _unpack_ccomp_kernel,
    _unpack_gather_c4_bf16_kernel,
)

__all__ = [
    "_pack_ccomp_kernel",
    "_unpack_ccomp_kernel",
    "_pack_c4_fp8_kernel",
    "_unpack_gather_c4_bf16_kernel",
    "_rope_tail_inplace_kernel",
    "_bf16_to_native_c4_kernel",
]
