from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent

setup(
    name="mustafar-cuda",
    ext_modules=[
        CUDAExtension(
            name="mustafar._fused",
            sources=[
                str(HERE / "fused" / "bindings.cpp"),
                str(HERE / "fused" / "fused.cu"),
            ],
            include_dirs=[str(HERE)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        ),
        # Direct 328-byte packed reader. Built as a separate module so a failure
        # here never takes down _fused, which stays the _FUSED fallback.
        CUDAExtension(
            name="mustafar._sparse",
            sources=[
                str(HERE / "sparse" / "bindings.cpp"),
                str(HERE / "sparse" / "sparse_kernel.cu"),
            ],
            include_dirs=[str(HERE), str(HERE / "sparse")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
