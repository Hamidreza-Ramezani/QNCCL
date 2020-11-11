/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  cudaSetDevice(comm->cudaDev);
  size_t nbytes = count * ncclTypeSize(datatype);
  void* tempbuff1;
  void* tempbuff2;
  const void* compressedbuff1;
  void* compressedbuff2;
  void** tempbuff_ptr1 = &tempbuff1;
  void** tempbuff_ptr2 = &tempbuff2;
  const void** compressedbuff_ptr1 = &compressedbuff1;
  void** compressedbuff_ptr2 = &compressedbuff2;
  cudaMalloc(tempbuff_ptr1, nbytes);
  cudaMalloc(tempbuff_ptr2, nbytes);
  //cudaMalloc(compressedbuff_ptr1, nbytes);
  //cudaMalloc(compressedbuff_ptr2, nbytes);
  //cudaMemset((void*)compressedbuff1, 1, nbytes);
  //cudaMemset(compressedbuff2, 1, nbytes);
 
  //if ncclDataType == float
  //do_compress == true; 
  //args->sendBytes = sendCount * wordSize(type);
  //args->expectedBytes = recvCount * wordSize(type);
  //size_t totalnbytes = max(args->sendBytes, args->expectedBytes);
  //size_t shift = (totalnbytes * iter) % args->maxbytes;
  //if (shift + totalnbytes > args->maxbytes) shift = 0;
  //char* temp_buff1 = ((char*)tempbuff1) + shift;
  //char* temp_buff2 = ((char*)tempbuff2) + shift;

  cudaDeviceSynchronize();


  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, recvbuff, tempbuff1, tempbuff2, compressedbuff1, compressedbuff2, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}
