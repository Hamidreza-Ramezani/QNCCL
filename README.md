# QNCCL
Communication-Efficient primitives for inter-GPU communication via quantization and encoding

## Introduction
In this library, we optimized the all-reduce primitive (Ring and Tree algorithms) of [NCCL](https://github.com/nvidia/nccl) to achieve higher bandwidth via compression. In fact, each device compresses its buffer before sending it to another device. Besides, devices will decompress the received buffer. We used max-min method in this work, which is a lossy compression scheme.  This means there would be a negligible error in our all-reduce operation compared to the [original](https://github.com/NVIDIA/nccl/blob/master/src/collectives/all_reduce.cc) one. But, that does not affect the overall results of machine learning experiments.

## Build
use the following steps to build QNCCL from source. 
    
    $ git clone https://github.com/hamid-ramezani/QNCCL.git
    $ cd QNCCL
    $ export CUDA_HOME=<path to cuda install>
    $ make -j src.build 
By specifying the device architecture, the compilation process will be much faster. `NVCC_GENCODE` will only include the architecture of the target platform. For instance, if the compute capability of the target platform is sm70, the last command should be changed to:  

    $ make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"

## Build QNCCL-tests

Tests for QNCCL are maintained separately [here](https://github.com/hamid-ramezani/QNCCL-tests). It is used for both correctness and performance of collective operations.  To build QNCCL_tests, use the following commands: 

    $ git clone https://github.com/hamid-ramezani/QNCCL-tests
    $ cd QNCCL-tests
    $ export NCCL_HOME=<path to nccl build folder>
    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path_to_QNCCL/build/lib
    $ export CUDA_HOME=<path to cuda install>
    $ make 

## Environment
QNCCL has three additional environment variables to NCCL set of [environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). They are `RING_ALLREDUCE_VERSION` , `bucket_size` , and `BITS`. 

 1. `RING_ALLREDUCE_VERSION`: This is for choosing to use the original implementation of all_reduce (in NCCL) or using the new one (in QNCCL). Its value can be either `old` or `new`. The default value is `old`.
 2. `bucket_size` : This specifies the size of bucket used in max-min quantization. It can accept any positive value, but we suggest to use a power of two. The default value is `1024`.   
 3. `BITS`: This specifies the number of bits after applying the compression operator. It accepts any value between 1 to 32. The default value is `8`. The lower the `BITS`,  the stronger the compression. 

We suggest to set the following environment variables before using QNCCL:

    $ export NCCL_ALGO=Ring
    $ export NCCL_PROTO=Simple
    $ export RING_ALLREDUCE_VERSION=new
    $ export NCCL_MIN_NCHANNELS=64
    $ export NCCL_NTHREADS=512
    $ export bucket_size=1024
    $ export BITS=8
#### Caveats

 - `NCCL_ALGO` can be set to `Tree` as well since we added our compression scheme to the Tree algorithm either. 

## Quick example

    $ cd QNCCL-tests
    $ ./build/all_reduce_perf -b 4 -e 1G -d float -f 2 -n 5 -w 1 -g 8

The above example runs all_reduce on 8 GPUs. The inputs size varies from 4Bytes  to 1GBytes. In each step, the input size is doubled (This is specified by `-f` flag). There is one warm-up iteration (`-n` flag). Each element of the input and output buffers is a single-precision floating point number ( `-d` option). The complete list of arguments can be found [here](https://github.com/nvidia/nccl-tests#arguments). 

## Results

Our results show that QNCCL is ~4x faster than NCCL in applying `all_reduce` on large buffers (bigger than 100MBytes). So, it is suggested to use QNCCL for applications in which the buffers need to be reduced are big enough, though our implementation has gain for all buffers of size > 1MBytes. 

## Machine learning experiments
QNCCL can be linked to [Pytorch](https://github.com/pytorch/pytorch) to be used as the communication back-end in distributed training. To do so, Pytorch has to be compiled from source. The steps for building Pytorch from source and linking it to QNCCL is as follows: 

    $ export NCCL_ROOT=<path_to_QNCCL>
    $ export NCCL_LIB_DIR=<path_to_QNCCL_lib_dir>
    $ export NCCL_INCLUDE_DIR=<path_to_QNCCL_include_dir>
    $ conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
    $ conda install -c pytorch magma-cuda110  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
    $ git clone --recursive https://github.com/pytorch/pytorch
    $ cd pytorch
    $ git submodule sync
    $ git submodule update --init --recursive
    $ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    $ USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=0 python setup.py install
#### Caveats
If the architecture of the target platform is `GeForce RTX 3090` , the nightly version of Pytorch needs to be compiled. That version is available in [this](https://github.com/pytorch/pytorch/tree/nightly) branch.  So, the following step must be done after cloning Pytorch:

    $ git checkout nightly



```
## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
