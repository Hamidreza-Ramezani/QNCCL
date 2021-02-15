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
#include <curand_kernel.h>

inline __device__ void setup_kernel(curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, 0, 0, &state[id]);
}



template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-WARP_SIZE;
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
    int bucket_size = 1024;
    curandState* devStates = (curandState*)comm->states;
    setup_kernel(devStates);
    if (std::is_same<T, float>::value && std::is_same<FUNC, FuncSum<float>>::value) {
      const int BITS=8;
      const float * __restrict__ thisInput = (const float*)args->sendbuff;
      float * __restrict__ thisOutput = (float*)args->recvbuff;

      ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, unsigned char, 1, 1, 1, FuncSum<unsigned char>>
        prims(tid, nthreads, &ring->prev, &ring->next, (unsigned char*)thisOutput, stepSize*4, channel, comm, ncclShmem->ptrs, 0);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
        ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;
        int nelem_compressed;

        ssize_t compressed_offset;

        //step 0: push data to next GPU
        chunk = ring->devUserRanks[nranks-1];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        int num_buckets = DIVUP(nelem, bucket_size);
        size_t meta_size = 2 * sizeof(float) * num_buckets;

        //if (tid == 0 && ring->devUserRanks[0] == 0) {     
        //   printf("bid = %d\n", blockIdx.x);
        //   printf("num_buckets = %d\n", num_buckets);
        //}

        int pre_num_buckets = DIVUP(offset, bucket_size);
        size_t pre_meta_size = 2 * sizeof(float) * pre_num_buckets;
        compressed_offset = offset+pre_meta_size;

        //unsigned char* __restrict__ compressed_temp = (unsigned char*)args->tempbuff1;
        unsigned char* __restrict__ compressed_temp = (unsigned char*)comm->tempbuff1;

        //if(tid == 0 && blockIdx.x == 0) {
        //   printf("in the place 1\n");
        //   printf("in device:%d\n", ring->devUserRanks[0]);
        //   printf("offset is %d\n", offset);
        //   printf("nelem is %d\n", nelem);
        //   printf("\n\n\n");
        //}
        //__syncthreads();

        //compress(thisInput+offset, compressed_temp+offset, nelem, args->coll.nThreads);
        quantize(thisInput+offset, compressed_temp+compressed_offset, nelem, bucket_size, BITS, devStates);

        //__syncthreads();
        //if(tid == 0 && blockIdx.x == 0 && ring->devUserRanks[0] == 1) {
        //  printf("in the place 1\n");
        //  printf("in device:%d\n", ring->devUserRanks[0]);
        //  float* meta_info = (float*)(compressed_temp);
        //  for (int i=0; i<16; i++) {
        //    printf("meta_info %f\n", meta_info[i]);
        //  }
        //  printf("\n\n\n");
        //}
        //__syncthreads();

        //if(tid == 0 && blockIdx.x == 0 && ring->devUserRanks[0] == 0 && gridOffset == 29360128) {
        //  printf("in the place 1\n");
        //  printf("in device:%d\n", ring->devUserRanks[0]);
        //  printf("thisInput[31981567] = %f\n", thisInput[31981567]);
        //  printf("\n\n\n");
        //}
        //__syncthreads();

        nelem_compressed = DIVUP(nelem, 8/BITS);
        prims.send(compressed_temp+compressed_offset, nelem_compressed+meta_size);

        //prims.send(thisInput+offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<nranks; ++j) {
          chunk = ring->devUserRanks[nranks-j];
          offset = chunkOffset + chunk * realChunkSize;
          nelem = min(realChunkSize, size-offset);

          num_buckets = DIVUP(nelem, bucket_size);
          meta_size = 2 * sizeof(float) * num_buckets;   
          int pre_num_buckets = DIVUP(offset, bucket_size);
          size_t pre_meta_size = 2 * sizeof(float) * pre_num_buckets;
          compressed_offset = offset+pre_meta_size;

          //if (tid == 0 && ring->devUserRanks[0] == 0 && j == 2) {
          //   printf("bid = %d\n", blockIdx.x);
          //   printf("num_buckets = %d\n", num_buckets);
          //}

          //if(tid == 0 && blockIdx.x == 0) {
          //  printf("in the place 2 and j %d\n", j);
          //  printf("in device:%d\n", ring->devUserRanks[0]);
          //  printf("offset is %d\n", offset);
          //  printf("nelem is %d\n", nelem);
          //  printf("\n\n\n");
          //}
          //__syncthreads();

          //unsigned char* __restrict__ compressed_temp = (unsigned char*)args->tempbuff1;
          unsigned char* __restrict__ compressed_temp = (unsigned char*)comm->tempbuff1;
          float * __restrict__ decompressed_temp = (float*)comm->tempbuff3;
          //float * __restrict__ decompressed_temp = (float*)args->tempbuff3;

          nelem_compressed = DIVUP(nelem, 8/BITS);
          prims.recv(compressed_temp+compressed_offset, nelem_compressed+meta_size);

          //decompress(compressed_temp+offset, decompressed_temp+offset, nelem, args->coll.nThreads);

          dequantize(compressed_temp+compressed_offset, decompressed_temp+offset, nelem, bucket_size, BITS);

          //__syncthreads();
          for (int idx=offset+tid; idx<offset+nelem; idx += nthreads +32) {
            //float var = FuncSum<float>()(static_cast<float>(compressed_temp[idx]), thisInput[idx]);
            //float var = static_cast<float>(compressed_temp[idx]) + thisInput[idx];
            //compress(var, (unsigned char*)(compressed_temp+idx));
            decompressed_temp[idx] = decompressed_temp[idx] + thisInput[idx];
          }

          //if(tid == 0 && blockIdx.x == 0 && ring->devUserRanks[0] == 2 && j == 2 && gridOffset == 29360128) {
          //  printf("in the place 2 j %d\n", j);
          //  printf("in device:%d\n", ring->devUserRanks[0]);
          //  printf("decompressed_temp[31981567] = %f\n", decompressed_temp[31981567]);
          //  printf("\n\n\n");
          //}
          //__syncthreads();

          // __syncthreads();
          // if(tid == 0 && blockIdx.x == 0 && j==3) {
          //    //int count = 0;
          //    //for (int i=0; i<size; i++) {
          //    //  if(abs(decompressed_temp[i] - 7.2f) > 0.00001)
          //    //     count++;
          //    //}
          //    printf("count inside the kernel is %d\n", count);
          // }
          //__syncthreads();

          //compress(decompressed_temp+offset, compressed_temp+offset, nelem, args->coll.nThreads);
          quantize(decompressed_temp+offset, compressed_temp+compressed_offset, nelem, bucket_size, BITS, devStates);

          nelem_compressed = DIVUP(nelem, 8/BITS);
          prims.send(compressed_temp+compressed_offset, nelem_compressed+meta_size);
          //prims.recvReduceSend(thisInput+offset, nelem);
        }
        chunk = ring->devUserRanks[0];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        num_buckets = DIVUP(nelem, bucket_size);
        meta_size = 2 * sizeof(float) * num_buckets;   
        pre_num_buckets = DIVUP(offset, bucket_size);
        pre_meta_size = 2 * sizeof(float) * pre_num_buckets;
        compressed_offset = offset+pre_meta_size;

        //if(tid == 0 && blockIdx.x == 0) {
        //  printf("in the place 3\n");
        //  printf("in device:%d\n", ring->devUserRanks[0]);
        //  printf("offset is %d\n", offset);
        //  printf("nelem is %d\n", nelem);
        //  printf("\n\n\n");
        //}
        //__syncthreads();

        //unsigned char* __restrict__ compressed_temp = (unsigned char*)args->tempbuff1;
        //float * __restrict__ decompressed_temp = (float*)args->tempbuff3;
        float * __restrict__ decompressed_temp = (float*)comm->tempbuff3;

        nelem_compressed = DIVUP(nelem, 8/BITS);
        prims.directRecv(compressed_temp+compressed_offset, compressed_offset, nelem_compressed+meta_size);

        //decompress(compressed_temp+offset, decompressed_temp+offset, nelem, args->coll.nThreads);

        //__syncthreads();
        //if(tid == 0 && blockIdx.x == 1 && ring->devUserRanks[0] == 0) {
        //  printf("in the place 3\n");
        //  printf("in device:%d\n", ring->devUserRanks[0]);
        //  float* meta_info = (float*)(compressed_temp);
        //  for (int i=1056; i<1056+16; i++) {
        //  //for (int i=528; i<528+16; i++) {
        //  ///for (int i=4224; i<4224+16; i++) {
        //    printf("meta_info %f\n", meta_info[i]);
        //  }
        //  printf("\n\n\n");
        //}
        //__syncthreads();

        dequantize(compressed_temp+compressed_offset, decompressed_temp+offset, nelem, bucket_size, BITS);

        //__syncthreads();
        //if(tid == 0 && blockIdx.x == 1 && ring->devUserRanks[0] == 0) {
        //  printf("in the place 3\n");
        //  printf("in device:%d\n", ring->devUserRanks[0]);
        //  printf("decompressed_temp[4096] = %f\n", decompressed_temp[4096]);
        //  printf("\n\n\n");
        //}

        //__syncthreads();
        for (int idx = offset+tid; idx < offset+nelem; idx += nthreads +32) {
          //float var = FuncSum<float>()(static_cast<float>(compressed_temp[idx]), thisInput[idx]);
          //float var = static_cast<float>(compressed_temp[idx]) + thisInput[idx];
          //compress(var, (unsigned char*)(compressed_temp+idx));
          decompressed_temp[idx] = decompressed_temp[idx] + thisInput[idx];
          //thisOutput[idx] = decompressed_temp[idx];
        }

        //compress(decompressed_temp+offset, compressed_temp+offset, nelem, args->coll.nThreads);
        quantize(decompressed_temp+offset, compressed_temp+compressed_offset, nelem, bucket_size, BITS, devStates);
        //////__syncthreads();
        //decompress(compressed_temp+offset, thisOutput+offset, nelem, args->coll.nThreads);
        dequantize(compressed_temp+compressed_offset, thisOutput+offset, nelem, bucket_size, BITS);

        //prims.copySend(compressed_temp+offset, compressedOutput+offset, nelem+meta_size);
        //////prims.copySend(compressed_temp+compressed_offset, compressed_temp+compressed_offset, nelem+meta_size);
        nelem_compressed = DIVUP(nelem, 8/BITS);
        prims.send(compressed_temp+compressed_offset, nelem_compressed+meta_size);
        //prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          chunk = ring->devUserRanks[nranks-j];
          offset = chunkOffset + chunk * realChunkSize;
          nelem = min(realChunkSize, size-offset);

          num_buckets = DIVUP(nelem, bucket_size);
          meta_size = 2 * sizeof(float) * num_buckets;   
          int pre_num_buckets = DIVUP(offset, bucket_size);
          size_t pre_meta_size = 2 * sizeof(float) * pre_num_buckets;
          compressed_offset = offset+pre_meta_size;

          //if(tid == 0 && blockIdx.x == 0) {
          //  printf("in the place 4 and j %d \n", j);
          //  printf("in device:%d\n", ring->devUserRanks[0]);
          //  printf("offset is %d\n", offset);
          //  printf("nelem is %d\n", nelem);
          //  printf("\n\n\n");
          //}
          //__syncthreads();

          //prims.directRecvCopySend(compressedOutput+offset, offset, nelem+meta_size);
          nelem_compressed = DIVUP(nelem, 8/BITS);
          prims.directRecvCopySend(compressed_temp+compressed_offset, compressed_offset, nelem_compressed+meta_size);
          //////prims.directRecv(compressed_temp+compressed_offset, compressed_offset, nelem+meta_size);
          //////prims.send(compressed_temp+compressed_offset, nelem+meta_size);
          //decompress(compressed_temp+offset, thisOutput+offset, nelem, args->coll.nThreads);
          //dequantize<true>(compressedOutput+offset, thisOutput+offset, nelem, bucket_size, BITS);      
          dequantize(compressed_temp+compressed_offset, thisOutput+offset, nelem, bucket_size, BITS);    
        }
        // Make final copy from buffer to dest.
        chunk = ring->devUserRanks[1];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        num_buckets = DIVUP(nelem, bucket_size);
        meta_size = 2 * sizeof(float) * num_buckets;   
        pre_num_buckets = DIVUP(offset, bucket_size);
        pre_meta_size = 2 * sizeof(float) * pre_num_buckets;
        compressed_offset = offset+pre_meta_size;

        //if(tid == 0 && blockIdx.x == 0) {
        //  printf("in the place 5\n");
        //  printf("in device:%d\n", ring->devUserRanks[0]);
        //  printf("offset is %d\n", offset);
        //  printf("nelem is %d\n", nelem);
        //  printf("\n\n\n");
        //}
        //__syncthreads();

        // Final wait/copy.
        //prims.directRecv(compressedOutput+offset, offset, nelem+meta_size);
        nelem_compressed = DIVUP(nelem, 8/BITS);
        prims.directRecv(compressed_temp+compressed_offset, compressed_offset, nelem_compressed+meta_size);
        //decompress(compressed_temp+offset, thisOutput+offset, nelem, args->coll.nThreads);
        //dequantize<true>(compressedOutput+offset, thisOutput+offset, nelem, bucket_size, BITS);
        dequantize(compressed_temp+compressed_offset, thisOutput+offset, nelem, bucket_size, BITS);
      }
  }

  else {
     const T * __restrict__ thisInput = (const T*)args->sendbuff;
     T * __restrict__ thisOutput = (T*)args->recvbuff;


      ncclPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);

     for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
       ssize_t realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*nChannels));
       ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
       ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

       //if (tid == 0 && blockIdx.x == 0 && ring->devUserRanks[0] == 0) {
       //  printf("\nrealChunkSize %d\n", realChunkSize);
       //}


       /////////////// begin AllReduce steps ///////////////
       ssize_t offset;
       int nelem;
       int chunk;

       // step 0: push data to next GPU
       chunk = ring->devUserRanks[nranks-1];
       offset = chunkOffset + chunk * realChunkSize;
       nelem = min(realChunkSize, size-offset);

       //if(tid == 0 && blockIdx.x == 0) {
       //   printf("in the place 1\n");
       //   printf("in device:%d\n", ring->devUserRanks[0]);
       //   printf("offset is %d\n", offset);
       //   printf("nelem is %d\n", nelem);
       //   printf("\n\n\n");
       //}
       //__syncthreads();

       prims.send(thisInput+offset, nelem);
       for (int j=2; j<nranks; ++j) {
         chunk = ring->devUserRanks[nranks-j];
         offset = chunkOffset + chunk * realChunkSize;
         nelem = min(realChunkSize, size-offset);
         
         //if(tid == 0 && blockIdx.x == 0) {
         //   printf("in the place 2 and j %d\n", j);
         //   printf("in device:%d\n", ring->devUserRanks[0]);
         //   printf("offset is %d\n", offset);
         //   printf("nelem is %d\n", nelem);
         //   printf("\n\n\n");
         //}
         //__syncthreads();

         //prims.recv(thisOutput+offset, nelem);
         //prims.send(thisInput+offset, nelem);
         prims.recvReduceSend(thisInput+offset, nelem);
       }

       // step k-1: reduce this buffer and data, which will produce the final
       // result that we store in this data and push to the next GPU
       chunk = ring->devUserRanks[0];
       offset = chunkOffset + chunk * realChunkSize;
       nelem = min(realChunkSize, size-offset);
       
       //if(tid == 0 && blockIdx.x == 0) {
       //   printf("in the place 3\n");
       //   printf("in device:%d\n", ring->devUserRanks[0]);
       //   printf("offset is %d\n", offset);
       //   printf("nelem is %d\n", nelem);
       //   printf("\n\n\n");
       //}
       //__syncthreads();

       //prims.directRecv(thisOutput+offset, offset, nelem);
       //prims.copySend(thisInput+offset, thisOutput+offset, nelem);

       prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);

       // k-2 steps: copy to next GPU
       for (int j=1; j<nranks-1; ++j) {
         chunk = ring->devUserRanks[nranks-j];
         offset = chunkOffset + chunk * realChunkSize;
         nelem = min(realChunkSize, size-offset);


         //if(tid == 0 && blockIdx.x == 0) {
         //   printf("in the place 4 and j %d\n", j);
         //   printf("in device:%d\n", ring->devUserRanks[0]);
         //   printf("offset is %d\n", offset);
         //   printf("nelem is %d\n", nelem);
         //   printf("\n\n\n");
         //}
         //__syncthreads();

         prims.directRecvCopySend(thisOutput+offset, offset, nelem);
       }

       // Make final copy from buffer to dest.
       chunk = ring->devUserRanks[1];
       offset = chunkOffset + chunk * realChunkSize;
       nelem = min(realChunkSize, size-offset);


       //if(tid == 0 && blockIdx.x == 0) {
       //   printf("in the place 3\n");
       //   printf("in device:%d\n", ring->devUserRanks[0]);
       //   printf("offset is %d\n", offset);
       //   printf("nelem is %d\n", nelem);
       //   printf("\n\n\n");
       //}
       //__syncthreads();

       // Final wait/copy.
       prims.directRecv(thisOutput+offset, offset, nelem);
    }
  }

  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-2*WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

#if 1
    if (tid < nthreads+WARP_SIZE) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, 1, 0, FUNC>
        prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
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

    if (tid < nthreads+WARP_SIZE) {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclPrimitives<UNROLL, 1, 1, T, 1, NCCL_MAX_DEV_ARITY, 1, FUNC>
        prims(tid, nthreads, &tree->up, tree->down, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
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
    }
#else
    int nthreadsSplit = nthreads/2;
    if (nthreadsSplit == 256) nthreadsSplit += 64;
    if (tree->up == -1) {
      if (tid < nthreads+WARP_SIZE) {
        // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
        ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY, 1, FUNC>
          prims(tid, nthreads, tree->down, tree->down, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          prims.directRecvReduceCopySend(thisInput+offset, thisOutput+offset, offset, nelem);
        }
      }
    } else {
      if (tid < nthreadsSplit + WARP_SIZE) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        ncclPrimitives<UNROLL, 1, 1, T, NCCL_MAX_DEV_ARITY, 1, 0, FUNC>
          prims(tid, nthreadsSplit, tree->down, &tree->up, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.send(thisInput+offset, nelem);
          } else {
            prims.recvReduceSend(thisInput+offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        ncclPrimitives<UNROLL, 1, 1, T, 1, NCCL_MAX_DEV_ARITY, 1, FUNC>
          prims(tid-nthreadsSplit-WARP_SIZE, nthreads-nthreadsSplit, &tree->up, tree->down, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 2);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            prims.directRecv(thisOutput+offset, offset, nelem);
          } else {
            prims.directRecvCopySend(thisOutput+offset, offset, nelem);
          }
        }
      }
    }
#endif
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads-WARP_SIZE;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->collTree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
    int chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, tree->down, &tree->up, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
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
      ncclPrimitives<UNROLL, 1, 1, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, &tree->up, tree->down, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);
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
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
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
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    do {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLLPrimitives<T, FUNC, NCCL_MAX_DEV_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, stepLines, channel, comm);
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
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLLPrimitives<T, FUNC, 1, NCCL_MAX_DEV_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, stepLines, channel, comm);
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
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->collTree;
    const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
    ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
    const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (blockIdx.x < nChannels) { // first half of the channels do reduce
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
};

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
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
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_TREE, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
  __device__ void run(struct ncclWorkElem* args) {
    const int tid = threadIdx.x;
    const int nthreads = args->nThreads;
    const int bid = args->coll.bid;
    const int nChannels = args->coll.nChannels;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclTree* tree = &channel->tree;
    const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
    ssize_t chunkSize = args->coll.lastChunkSize;
    const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
    const ssize_t loopSize = nChannels*chunkSize;
    int nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
    const ssize_t size = args->coll.count;

    if (loopSize > size) {
      chunkSize = DIVUP(size, nChannels*minChunkSize)*minChunkSize;
    }

    // Compute pointers
    const T * __restrict__ thisInput = (const T*)args->sendbuff;
    T * __restrict__ thisOutput = (T*)args->recvbuff;

    if (tree->up == -1) {
      // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
      ncclLL128Primitives<T, FUNC, NCCL_MAX_DEV_ARITY, NCCL_MAX_DEV_ARITY> LLprims(tid, nthreads, tree->down, tree->down, stepSize, channel, comm);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
      }
    } else {
      if (tid < nthreadsSplit) {
        // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
        ncclLL128Primitives<T, FUNC, NCCL_MAX_DEV_ARITY, 1> LLprims(tid, nthreadsSplit, tree->down, &tree->up, stepSize, channel, comm);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Up
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            LLprims.send(thisInput+offset, nelem);
          } else {
            LLprims.recvReduceSend(thisInput+offset, nelem);
          }
        }
      } else {
        // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
        ncclLL128Primitives<T, FUNC, 1, NCCL_MAX_DEV_ARITY> LLprims(tid-nthreadsSplit, nthreads-nthreadsSplit, &tree->up, tree->down, stepSize, channel, comm);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
          // Down
          ssize_t offset = gridOffset + bid*chunkSize;
          int nelem = min(chunkSize, size-offset);
          if (tree->down[0] == -1) {
            LLprims.recv(thisOutput+offset, nelem);
          } else {
            LLprims.recvCopySend(thisOutput+offset, nelem);
          }
        }
      }
    }
  }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllReduce, NCCL_ALGO_COLLNET, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
__device__ void run(struct ncclWorkElem* args) { }
};
