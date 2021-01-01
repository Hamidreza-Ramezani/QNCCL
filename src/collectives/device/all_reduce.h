/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include "cuda_runtime.h"
#include "cuda.h"
#include <stdio.h>
#include <compress.h>
#include <type_traits>


template<int UNROLL, class FUNC, typename T>__device__ void ncclAllReduceRingKernel_new(struct CollectiveArgs* args);
template<int UNROLL, class FUNC, typename T>__device__ void ncclAllReduceRingKernel_old(struct CollectiveArgs* args);


template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceRingKernel(struct CollectiveArgs* args) {
  if (args->with_compression) {
      ncclAllReduceRingKernel_new<UNROLL,FUNC,T>(args);
  } else {
    ncclAllReduceRingKernel_old<UNROLL,FUNC,T>(args);
  }
}


template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceRingKernel_new(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads-WARP_SIZE;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const int nranks = comm->nRanks;
  const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
  const ssize_t size = args->coll.count;
  int bucket_size = nthreads;

  //int num_buckets = DIVUP(size, bucket_size);
  //int meta_size = 2 * sizeof(float) * num_buckets;

  if (std::is_same<T, float>::value && std::is_same<FUNC, FuncSum<float>>::value) {
    const float * __restrict__ thisInput = (const float*)args->sendbuff;
    float * __restrict__ thisOutput = (float*)args->recvbuff;
    unsigned char * __restrict__ compressedOutput = (unsigned char*)args->tempbuff2;

    ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, unsigned char, 1, 1, 1, FuncSum<unsigned char>>
      prims(tid, nthreads, &ring->prev, &ring->next, compressedOutput, stepSize, channel, comm);


    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      //ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(unsigned char));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      int num_buckets = DIVUP(nelem, bucket_size);
      int meta_size = 2 * sizeof(float) * num_buckets;

      unsigned char* __restrict__ compressed_temp = (unsigned char*)args->tempbuff1;

      //if(tid == 0 && blockIdx.x == 0) {
      //     printf("before the first quantization call, the elements of the input buffer is as follows:\n");
      //     printf("offset is %d\n", offset);
      //     printf("nelem is %d\n", nelem);
      //     for (int i=0; i<8; i++) {
      //        printf("%f- ", *(thisInput+offset+i));
      //     }
      //     printf("\n\n\n");
      //}
      //__syncthreads();

      //compress(thisInput+offset, compressed_temp+offset, nelem, args->coll.nThreads);

      quantize<8>(thisInput+offset, compressed_temp+offset, nelem, bucket_size, args->coll.nThreads);

      //__syncthreads();
      //if (tid == 0 && blockIdx.x == 0 && ring->devUserRanks[0] == 1) {
      //  printf("after the first compression, the meta information in device1 is as follows: \n");
      //  for (int i=0; i<8; ++i)
      //    printf("%u- ", compressed_temp[i+8]);
      //  printf("\n");
      //}
      //__syncthreads();

      prims.send(compressed_temp+offset, nelem+meta_size);

      //prims.send(temp2+offset, nelem);
      //prims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        int num_buckets = DIVUP(nelem, bucket_size);
        int meta_size = 2 * sizeof(float) * num_buckets;   

        unsigned char* __restrict__ compressed_temp = (unsigned char*)args->tempbuff1;
        float * __restrict__ decompressed_temp = (float*)args->tempbuff3;

        prims.recv(compressed_temp + offset, nelem+meta_size);

        //decompress(compressed_temp+offset, decompressed_temp+offset, nelem, args->coll.nThreads);

        // __syncthreads();
        // if (tid == 0 && blockIdx.x == 0 && j == 2 && ring->devUserRanks[0] == 1) { 
        //   //printf("before the decompression, the shifted input is \n");
        //   //for (int i=0; i<8; ++i) 
        //   //  printf("%u- ", compressed_temp[i+8]);
        //   printf("realChunkSize in device1 is: %d\n", realChunkSize);
        //   printf("chunkOffset in device1 is %d\n", chunkOffset);
        // }
        // __syncthreads();

        dequantize<true,8>(compressed_temp+offset, decompressed_temp+offset, nelem, bucket_size, args->coll.nThreads, ring->devUserRanks[0], j);

        //__syncthreads();
        //if (tid == 0 && blockIdx.x == 0 && j == 2 && ring->devUserRanks[0] == 0) { 
        //  printf("after the decompression, the output is \n");
        //  for (int i=0; i<8; ++i) 
        //    printf("%f- ", decompressed_temp[i]);
        //  printf("\n");
        //}
        //__syncthreads();

        //if(tid == 0 && blockIdx.x == 0 && j==2) {
        //   printf("after the first dequantization call, the elements of the output buffer is as follows:\n");
        //   printf("offset is %d\n", offset);
        //   printf("nelem is %d\n", nelem);
        //   for (int i=0; i<8; i++) {
        //      printf("%f- ", *(decompressed_temp+offset+i));
        //   }
        //   printf("\n\n\n");
        //}
        //__syncthreads();
         
        //if (tid == 0) {
        //   printf("nelem is %d\n", nelem);
        //}
        __syncthreads();
        for (int idx=offset+tid; idx<offset+nelem; idx += args->coll.nThreads) {
          //float var = FuncSum<float>()(static_cast<float>(compressed_temp[idx]), thisInput[idx]);
          //float var = static_cast<float>(compressed_temp[idx]) + thisInput[idx];
          //compress(var, (unsigned char*)(compressed_temp+idx));
          decompressed_temp[idx] = decompressed_temp[idx] + thisInput[idx];
        }

       // __syncthreads();
       // if(tid == 0 && blockIdx.x == 0 && j==3) {
       //    //int count = 0;
       //    //for (int i=0; i<size; i++) {
       //    //  if(abs(decompressed_temp[i] - 7.2f) > 0.00001)
       //    //     count++;
       //    //}
       //    printf("count inside the kernel is %d\n", count);
       // }
       // __syncthreads();

        //compress(decompressed_temp+offset, compressed_temp+offset, nelem, args->coll.nThreads);
        quantize<8>(decompressed_temp+offset, compressed_temp+offset, nelem, bucket_size, args->coll.nThreads);

        prims.send(compressed_temp + offset, nelem+meta_size);
        //prims.recvReduceSend(thisInput+offset, nelem);
      }
      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      num_buckets = DIVUP(nelem, bucket_size);
      meta_size = 2 * sizeof(float) * num_buckets;   

      //unsigned char* __restrict__ compressed_temp = (unsigned char*)args->tempbuff1;
      float * __restrict__ decompressed_temp = (float*)args->tempbuff3;

      prims.directRecv(compressed_temp + offset , offset, nelem+meta_size);

      //decompress(compressed_temp+offset, decompressed_temp+offset, nelem, args->coll.nThreads);

      //__syncthreads();
      //if (tid == 0 && blockIdx.x == 0 && ring->devUserRanks[0]==0) { 
      //  printf("before the first decompression the input of device0 is as follows: \n");
      //  for (int i=0; i<8; ++i) 
      //    printf("%u- ", compressed_temp[i]);
      //}
      //__syncthreads();

      dequantize<true,8>(compressed_temp+offset, decompressed_temp+offset, nelem, bucket_size, args->coll.nThreads, ring->devUserRanks[0], 4);
      //__syncthreads();
      //if (tid == 0 && blockIdx.x == 0 && ring->devUserRanks[0]==0) { 
      //  printf("the output of decompression in device0:\n");
      //  for (int i=0; i<8; ++i) 
      //    printf("%f- ", decompressed_temp[i]);
      //  printf("\n\n\n");
      //}
      __syncthreads();
      for (int idx = offset+tid; idx < offset+nelem; idx += args->coll.nThreads) {
        //float var = FuncSum<float>()(static_cast<float>(compressed_temp[idx]), thisInput[idx]);
        //float var = static_cast<float>(compressed_temp[idx]) + thisInput[idx];
        //compress(var, (unsigned char*)(compressed_temp+idx));
        decompressed_temp[idx] = decompressed_temp[idx] + thisInput[idx];
        thisOutput[idx] = decompressed_temp[idx];
      }
      //compress(decompressed_temp+offset, compressed_temp+offset, nelem, args->coll.nThreads);
      quantize<8>(decompressed_temp+offset, compressed_temp+offset, nelem, bucket_size, args->coll.nThreads);
      //__syncthreads();
      //decompress(compressed_temp+offset, thisOutput+offset, nelem, args->coll.nThreads);
      //dequantize<true,8>(compressed_temp+offset, thisOutput+offset, nelem, bucket_size, args->coll.nThreads, ring->devUserRanks[0], 5);

      prims.copySend(compressed_temp + offset, compressed_temp + offset, nelem+meta_size);
      //prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        int num_buckets = DIVUP(nelem, bucket_size);
        int meta_size = 2 * sizeof(float) * num_buckets;   

        prims.directRecvCopySend(compressed_temp+offset, offset, nelem+meta_size);
        //decompress(compressed_temp+offset, thisOutput+offset, nelem, args->coll.nThreads);
        dequantize<true,8>(compressed_temp+offset, thisOutput+offset, nelem, bucket_size, args->coll.nThreads, ring->devUserRanks[0], 6);      
      }
      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      num_buckets = DIVUP(nelem, bucket_size);
      meta_size = 2 * sizeof(float) * num_buckets;   

      // Final wait/copy.
      prims.directRecv(compressed_temp+offset, offset, nelem+meta_size);
      //decompress(compressed_temp+offset, thisOutput+offset, nelem, args->coll.nThreads);
      dequantize<true,8>(compressed_temp+offset, thisOutput+offset, nelem, bucket_size, args->coll.nThreads, ring->devUserRanks[0], 7);

      //__syncthreads();
      //if(tid == 0 && blockIdx.x == 0) {
      //  printf("inside kernel output is ");
      //  for(int i=0; i<size; i++) {
      //     printf("%f- ", thisOutput[i]);
      //  }
      //  printf("\n\n");
      //}
      //__syncthreads();

    }
    //for (int i=tid; i<size; i+=args->coll.nThreads) {
    //    thisOutput[i] = decompressed_temp[i];
    //}
    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    //for (; i<size; i += gridDim.x * blockDim.x) {
    //    thisOutput[i] = static_cast<float>((args->tempbuff3)[i]);
    //}
  }

  else {
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;
    ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>                               
      prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm);                             

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;
      

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);


      const T* __restrict__ temp2 = (T*)args->tempbuff2;               //when T != float
      temp2 = compress<T>(thisInput, temp2, nelem);
      prims.send(temp2+offset, nelem);
      //prims.send(thisInput+offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        T* __restrict__ temp = (T*)args->tempbuff1;			//when T != float

        prims.recv(temp + offset , nelem);
        for (int idx = offset+tid; idx < offset+nelem; idx += args->coll.nThreads) {
          //temp[idx] = FUNC()(temp[idx], thisInput[idx]);
          //temp[idx] = FUNC()((T)temp[idx], thisInput[idx]);
          T var = FUNC()((T)temp[idx], thisInput[idx]);
          temp[idx] = compress<T>(var);
        }
        prims.send(temp + offset, nelem);
        //prims.recvReduceSend(thisInput+offset, nelem);
      }

      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);


      T* __restrict__ temp = (T*)args->tempbuff1;                  //when T != float

      prims.directRecv(temp + offset , offset, nelem);
      for (int idx = offset+tid; idx < offset+nelem; idx += args->coll.nThreads) {
        //temp2[idx] = FUNC()(thisInput[idx], temp2[idx]);
        //temp[idx] = FUNC()(thisInput[idx], (T)temp[idx]);
        T var = FUNC()((T)temp[idx], thisInput[idx]);
        temp[idx] = compress<T>(var);
      }

      prims.copySend(temp + offset, thisOutput+offset, nelem);

      //prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      // Final wait/copy.
      prims.directRecv(thisOutput+offset, offset, nelem);
    }
  }
}

/*

//----------------------------------------------------------------------------------------------------------
  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>                               
    prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm);                             
//ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, int32_t, 1, 1, 1, FuncSum<int32_t>>
//  prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm);
//----------------------------------------------------------------------------------------------------------



  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;
    

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);


    //I need to compress thisInput+offset ... nelem 
    //prims(compressedInput+offset, nelem)
    //compress(thisInput);

    //if (threadIdx.x == 0 && gridOffset == 0) {
    //  compressedbuff1 = compress<T>(thisInput, compressedbuff1, nelem);
    //  //printf("the compression is done \n");
    //}


//----------------------------------------------------------------------------------------------------------
    const T* __restrict__ temp2 = (T*)args->tempbuff2;               //when T != float
    //const int* __restrict__ temp2 = (int*)args->tempbuff2;         //when T == float
//----------------------------------------------------------------------------------------------------------


    temp2 = compress<T>(thisInput, temp2, nelem);
    prims.send(temp2+offset, nelem);
    //prims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);


//----------------------------------------------------------------------------------------------------------
      T* __restrict__ temp = (T*)args->tempbuff1;			//when T != float
      //int* __restrict__ temp = (int*)args->tempbuff1;			//when T == float
//----------------------------------------------------------------------------------------------------------

      prims.recv(temp + offset , nelem);
      for (int idx = offset+tid; idx < offset+nelem; idx += args->coll.nThreads) {
        //temp[idx] = FUNC()(temp[idx], thisInput[idx]);
        //temp[idx] = FUNC()((T)temp[idx], thisInput[idx]);
        T var = FUNC()((T)temp[idx], thisInput[idx]);
        temp[idx] = compress<T>(var);
      }
      prims.send(temp + offset, nelem);


      //prims.recvReduceSend(thisInput+offset, nelem);
    }


    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);


//----------------------------------------------------------------------------------------------------------
      T* __restrict__ temp = (T*)args->tempbuff1;                  //when T != float
      //int* __restrict__ temp = (int*)args->tempbuff1;            //when T == float
//----------------------------------------------------------------------------------------------------------

    prims.directRecv(temp + offset , offset, nelem);
    for (int idx = offset+tid; idx < offset+nelem; idx += args->coll.nThreads) {
      //temp2[idx] = FUNC()(thisInput[idx], temp2[idx]);
      //temp[idx] = FUNC()(thisInput[idx], (T)temp[idx]);
      T var = FUNC()((T)temp[idx], thisInput[idx]);
      temp[idx] = compress<T>(var);
    }
    //for (int idx = offset+tid; idx < offset+nelem; idx += args->coll.nThreads) {
    //  thisOutput[idx] = temp[idx];
    //  //thisOutput[idx] = temp2[idx];
    //}
    //prims.send(temp + offset, nelem);
    //prims.send(temp2 + offset, nelem);


    prims.copySend(temp + offset, thisOutput+offset, nelem);




    //prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvCopySend(thisOutput+offset, offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    prims.directRecv(thisOutput+offset, offset, nelem);
  }
}


*/


template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceRingKernel_old(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads-WARP_SIZE;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const int nranks = comm->nRanks;
  const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
  const ssize_t size = args->coll.count;


  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
    prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm);

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    //if(threadIdx.x == 0 && blockIdx.x == 0) {
    //  int sliceSize = stepSize*ALLREDUCE_SLICESTEPS;
    //  int dataSize = max(DIVUP(nelem, 16*ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS)*16, sliceSize/32);
    //  int realSize = max(0, min(dataSize, nelem-offset));
    //  int a = realSize * sizeof(thisInput[0]);
    //  printf("%d number of bytes is communicated in original NCCL\n", a);
    //}

    prims.send(thisInput+offset, nelem);
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      
      //prims.recv(thisOutput+offset, nelem);
      //prims.send(thisInput+offset, nelem);
      prims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    //prims.directRecv(thisOutput+offset, offset, nelem);
    //prims.copySend(thisInput+offset, thisOutput+offset, nelem);

    prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directRecvCopySend(thisOutput+offset, offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    prims.directRecv(thisOutput+offset, offset, nelem);
  }
}


template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceTreeKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads-WARP_SIZE;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  int chunkSize = args->coll.lastChunkSize;
  const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nChannels*chunkSize;
  const ssize_t size = args->coll.count;


  //printf("2\n");

  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclPrimitives<UNROLL/2, 1, 1, T, NCCL_MAX_TREE_ARITY, 1, 0, FUNC> prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclPrimitives<UNROLL/2, 1, 1, T, 1, NCCL_MAX_TREE_ARITY, 1, FUNC> prims(tid, nthreads, &tree->up, tree->down, thisOutput, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.directSend(thisOutput+offset, offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.directRecv(thisOutput+offset, offset, nelem);
      } else {
        prims.directRecvCopySend(thisOutput+offset, offset, nelem);
      }
    }
  } while(0);
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceCollNetKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads-WARP_SIZE;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
  int chunkSize = args->coll.lastChunkSize;
  const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nChannels*chunkSize;
  const ssize_t size = args->coll.count;

  //printf("3\n");
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  if (blockIdx.x < nChannels) { // first half of the channels do reduce
    struct ncclTree* tree = &channel->collTreeUp;
    ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC> prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }

  if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
    struct ncclTree* tree = &channel->collTreeDn;
    ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC> prims(tid, nthreads, &tree->up, tree->down, NULL, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceRingLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
  ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);
  const int nranks = comm->nRanks;
  const ssize_t loopSize = nChannels*nranks*chunkSize;
  const ssize_t size = args->coll.count;

  ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepLines, channel, comm);

  //printf("4\n");

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceTreeLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
  ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nChannels*chunkSize;
  const ssize_t size = args->coll.count;

  //printf("5\n");


  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclLLPrimitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, stepLines, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclLLPrimitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, stepLines, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceCollNetLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
  ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = nChannels*chunkSize;
  const ssize_t size = args->coll.count;

  //printf("6\n");

  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  if (blockIdx.x < nChannels) { // first half of the channels do reduce
    struct ncclTree* tree = &channel->collTreeUp;
    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, tree->down, &tree->up, stepLines, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }

  if (blockIdx.x >= nChannels) { // second half of the channels do broadcast
    struct ncclTree* tree = &channel->collTreeDn;
    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &tree->up, tree->down, stepLines, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

#include "prims_ll128.h"
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceRingLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
  ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
  // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
  const int nranks = comm->nRanks;
  const ssize_t loopSize = nChannels*nranks*chunkSize;
  const ssize_t size = args->coll.count;

  ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, channel, comm);

  //printf("7\n");


  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceTreeLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->coll.nThreads;
  const int bid = args->coll.bid;
  const int nChannels = args->coll.nChannels;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclTree* treeUp = &channel->treeUp;
  struct ncclTree* treeDn = &channel->treeDn;
  const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
  ssize_t chunkSize = args->coll.lastChunkSize;
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
  const ssize_t loopSize = nChannels*chunkSize;
  int nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
  const ssize_t size = args->coll.count;


  //printf("8\n");

  if (loopSize > size) {
    chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->sendbuff;
  T * __restrict__ thisOutput = (T*)args->recvbuff;

  if (treeUp->up == -1) {
    // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
    ncclLL128Primitives<T, FUNC, NCCL_MAX_TREE_ARITY, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, treeUp->down, treeDn->down, stepSize, channel, comm);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
    }
  } else {
    if (tid < nthreadsSplit) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLL128Primitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreadsSplit, treeUp->down, &treeUp->up, stepSize, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeUp->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    } else {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLL128Primitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid-nthreadsSplit, nthreads-nthreadsSplit, &treeDn->up, treeDn->down, stepSize, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeDn->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceCollNetLL128Kernel(struct CollectiveArgs* args) { }
