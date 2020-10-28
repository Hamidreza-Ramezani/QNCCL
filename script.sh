#!/usr/bash



#hello world!
make clean
export DEBUG=1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_FILE
#export NCCL_DEBUG_SUBSYS
make -j src.build CUDA_HOME=/mnt/nfs/clustersw/shared/cuda/10.1.243

