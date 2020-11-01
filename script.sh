#!/usr/bash


make clean
export DEBUG=1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_FILE
#export NCCL_DEBUG_SUBSYS
export CUDA_HOME=/mnt/nfs/clustersw/shared/cuda/10.1.243
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
