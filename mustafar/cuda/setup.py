from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent

setup(
    name="mustafar-stage2a",
    ext_modules=[
        CUDAExtension(
            name="mustafar._stage2a_cuda",
            sources=[
                str(HERE / "bindings.cpp"),
                str(HERE / "stage2a_reconstruct.cu"),
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
