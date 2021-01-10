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


  //float a[count];
  //for (int i=0; i< count; i++) { 
  //  a[i] = 1.1f ;
  //}

  cudaSetDevice(comm->cudaDev);
  size_t nbytes = count * ncclTypeSize(datatype);
  void* tempbuff1;
  void* tempbuff2;
  void* tempbuff3;
  //const void* compressedbuff1;
  //void* compressedbuff2;
  void** tempbuff_ptr1 = &tempbuff1;
  void** tempbuff_ptr2 = &tempbuff2;
  void** tempbuff_ptr3 = &tempbuff3;
  //const void** compressedbuff_ptr1 = &compressedbuff1;
  //void** compressedbuff_ptr2 = &compressedbuff2;

  //int num_buckets = 64 ;
  //int header_size = 2*num_buckets; 
  //cudaMalloc(tempbuff_ptr1, header_size*sizeof(float) + nbytes/4);

  int min_nthreads = 64;
  int bucket_size = min_nthreads;
  int num_buckets = DIVUP(count, bucket_size);
  int meta_size = 2 * sizeof(float) * num_buckets;

  cudaMalloc(tempbuff_ptr1, nbytes);
  //cudaMalloc(tempbuff_ptr1, nbytes/4 + meta_size);
  //cudaMalloc(tempbuff_ptr2, nbytes/4 + meta_size);
  cudaMalloc(tempbuff_ptr3, nbytes);
 
  //cudaMemcpy(tempbuff1, (void*)a, count*sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(tempbuff3, (void*)a, count*sizeof(float), cudaMemcpyHostToDevice);

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
    sendbuff, recvbuff, tempbuff1, tempbuff2, tempbuff3, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}
