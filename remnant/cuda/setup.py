from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent

setup(
    name="remnant-cuda",
    ext_modules=[
        CUDAExtension(
            name="remnant._fused",
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
    ],
    cmdclass={"build_ext": BuildExtension},
)
