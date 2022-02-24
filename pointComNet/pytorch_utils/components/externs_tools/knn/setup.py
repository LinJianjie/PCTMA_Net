# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, CUDA_HOME
from torch.__config__ import parallel_info
import sys
import os

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
info = parallel_info()
extra_compile_args = {'cxx': []}
if 'parallel backend: OpenMP' in info and 'OpenMP not found' not in info:
    extra_compile_args['cxx'] += ['-DAT_PARALLEL_OPENMP']
    if sys.platform == 'win32':
        extra_compile_args['cxx'] += ['/openmp']
    else:
        extra_compile_args['cxx'] += ['-fopenmp']
else:
    print('Compiling without OpenMP...')
if WITH_CUDA:
    Extension = CUDAExtension
    nvcc_flags = os.getenv('NVCC_FLAGS', '')
    nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
    nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
    extra_compile_args['nvcc'] = nvcc_flags

setup(name='knn',
      version='2.0.0',
      ext_modules=[
          CUDAExtension('knn',
                        [
                            'knn.cpp',
                            'knn_cuda.cu',
                            'knn_cpu.cpp'
                        ],
                        extra_compile_args=extra_compile_args
                        ),
      ],
      cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)})
