/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include <curand_kernel.h>
#include <curand.h>

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {


  cudaSetDevice(comm->cudaDev);
  size_t nbytes = count * ncclTypeSize(datatype);
  void* tempbuff1;
  void* tempbuff2;
  void* tempbuff3;
  void** tempbuff_ptr1 = &tempbuff1;
  //void** tempbuff_ptr2 = &tempbuff2;
  void** tempbuff_ptr3 = &tempbuff3;

  int bucket_size;
  void * states;
  curandGenerator_t gen;
  float * random_numbers;
  const unsigned int threadsPerBlock = 512;
  const unsigned int blockCount = 64;
  const unsigned int totalThreads = threadsPerBlock * blockCount;

  char* ring_allReduce_version = getenv("RING_ALLREDUCE_VERSION");

  if (strcasecmp(ring_allReduce_version, "new") == 0) {
    char* bucket_size_str = getenv("bucket_size");
    if (bucket_size_str == NULL) {
      bucket_size = 1024;
    } else {
      bucket_size = atoi(bucket_size_str);
    }

    int num_buckets = DIVUP(count, bucket_size);
    int meta_size = 2 * sizeof(float) * num_buckets;
    cudaMalloc(tempbuff_ptr1, nbytes/4 + meta_size);
    cudaMalloc(tempbuff_ptr3, nbytes);

    cudaMalloc((void **)&random_numbers, totalThreads * sizeof(float));
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, random_numbers, totalThreads);
  }
  cudaDeviceSynchronize();

  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, recvbuff, tempbuff1, tempbuff2, tempbuff3, random_numbers, states, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}
