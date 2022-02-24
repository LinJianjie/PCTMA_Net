"""Setup extension

Notes:
    If extra_compile_args is provided, you need to provide different instances for different extensions.
    Refer to https://github.com/pytorch/pytorch/issues/20169

"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='emd',
            sources=[
                'emd.cpp',
                'emd_kernel.cu',
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
