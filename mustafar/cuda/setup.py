from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent

setup(
    name="mustafar-fused",
    ext_modules=[
        CUDAExtension(
            name="mustafar._fused_cuda",
            sources=[
                str(HERE / "bindings.cpp"),
                str(HERE / "fused.cu"),
            ],
            include_dirs=[str(HERE)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
