"""Triton kernels for physically-sparse TopMag compression (Stage 0).

Import the kernels directly (e.g. `from .triton import _pack_ccomp_kernel`).
No torch host logic lives in this subfolder — see mustafar/sparse.py.
"""
from .kernels import _pack_ccomp_kernel, _unpack_ccomp_kernel

__all__ = ["_pack_ccomp_kernel", "_unpack_ccomp_kernel"]
