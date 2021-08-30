#include "cuda_runtime.h"
#include "cuda.h"
#include <stdint.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

#define PACK_SIZE 8
#define EPS 1e-10

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

inline __device__ __half hmax(__half a, __half b) { return __hge(a, b) ? a : b; }

inline __device__ __half hmin(__half a, __half b) { return __hge(a, b) ? b : a; }

template<typename T>
__device__ T* compress(T* src, T* compressedSrc, int nelem) {
   
  return src;
}


template<typename T>
__device__ const T* compress(const T* src, const T* compressedSrc, int nelem) {

  return src;
}


template<typename T>
__device__ T compress(T src) {

  return src;
}


__device__ __forceinline__ void compress(float* src, unsigned char* compressedSrc, int nelem, int nthreads) {
  const int tid = threadIdx.x;
  for (int idx = tid; idx < nelem; idx += nthreads) {
    //int8_t var;
    //if (src[idx] < 0) {
    //  var = static_cast<int8_t> (src[idx] - 0.5);
    //} else {
    //  var = static_cast<int8_t> (src[idx] + 0.5);
    //}
    //compressedSrc[idx] = var;
    compressedSrc[idx] = static_cast<unsigned char>(src[idx]);
  }
}


__device__ __forceinline__ void compress(const float* src, unsigned char* compressedSrc, int nelem, int nthreads) {
  const int tid = threadIdx.x;
  for (int idx = tid; idx < nelem; idx += nthreads) {
    //int8_t var;
    //if (src[idx] < 0) {
    //  var = static_cast<int8_t> (src[idx] - 0.5);
    //} else {
    //  var = static_cast<int8_t> (src[idx] + 0.5); 
    //}
    //compressedSrc[idx] = var;
    compressedSrc[idx] = static_cast<unsigned char>(src[idx]);
  }
}


__device__  __forceinline__ void compress(float src, unsigned char* compressedSrc) {
  if (src < 0) {
    *compressedSrc = static_cast<unsigned char> (src - 0.5);
  } else { 
    *compressedSrc = static_cast<unsigned char> (src + 0.5);
  }
}


__device__ __forceinline__ void decompress(unsigned char* src, float* decompressedSrc, int nelem, int nthreads) {
  const int tid = threadIdx.x;
  for (int idx = tid; idx < nelem; idx += nthreads) {
    decompressedSrc[idx] = static_cast<float>(src[idx]);
  }
}

inline __device__ float get_rand(curandState* state) {
  int id = threadIdx.x + blockIdx.x * (blockDim.x - 32);
  float random_number;
  curandState localState = state[id];
  random_number = curand_uniform(state);
  state[id] = localState;
  return random_number;
}

//inline __device__ float get_rand(curandStatePhilox4_32_10_t *state) {
//  if (threadIdx.x >= blockDim.x-32) {
//    return 0.5;
//  }
//  int id = threadIdx.x + blockIdx.x * (blockDim.x-32);
//  float random_number;
//  curandStatePhilox4_32_10_t localState = state[id];
//  random_number = curand_uniform(&localState);
//  //random_number = curand_normal(&localState);
//  state[id] = localState;
//  return random_number;
//}

//inline __device__ float get_rand(curandStateMRG32k3a *state) {
//  //if (threadIdx.x >= blockDim.x-32) {
//  //  return 0.5;
//  //}
//  int id = threadIdx.x + blockIdx.x * (blockDim.x);
//  float random_number;
//  curandStateMRG32k3a localState = state[id];
//  random_number = (float)curand_uniform_double(&localState);
//  //random_number = curand_normal(&localState);
//  state[id] = localState;
//  return random_number;
//}


inline __device__ void find_meta_seq(const float* input, float* meta, int num_elem, int bucket_size, int bits) {
  //int index = threadIdx.x + blockIdx.x * (blockDim.x-32);
  //int stride = gridDim.x * (blockDim.x-32);
  int index = threadIdx.x;
  int stride = blockDim.x -32;

  if (threadIdx.x < blockDim.x-32) {
     const int divisor = (1 << bits) - 1;
     float* meta_buf = (float*)meta;
     for (int i = index; i < (num_elem + bucket_size - 1) / bucket_size; i += stride) {
       float mmin = input[i * bucket_size];
       float mmax = input[i * bucket_size];
       for (int j = i * bucket_size + 1; j < fminf((i + 1) * bucket_size, num_elem); j++) {
         mmin = fminf(mmin, input[j]);
         mmax = fmaxf(mmax, input[j]);
       }
       meta_buf[2 * i] = ((mmax - mmin) / divisor);
       meta_buf[2 * i + 1] = mmin;
     }
  }
  __syncthreads();
}


inline __device__ void find_meta_parallel(float* input, float* meta, int num_elems, int bits) {
  int tid = threadIdx.x;
  int block_size = blockDim.x-32;
  float* meta_buf = (float*)meta;
  extern __shared__ float sdata[];
  meta_buf[0] = input[0];
  meta_buf[1] = input[0];
  int num_iters_per_bucket = (num_elems + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    int idx = i * block_size + tid;
    if (idx < num_elems) {
        sdata[tid] = input[idx];
        sdata[block_size + tid] = input[idx];
    }
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s && idx + s < num_elems) {
        sdata[tid] = fmaxf(sdata[tid + s], sdata[tid]);
        sdata[block_size + tid] = fminf(sdata[block_size + tid + s], sdata[block_size + tid]);
      }
    __syncthreads();
    }

    if (tid == 0) {
        meta_buf[0] = fmaxf(meta_buf[0], sdata[tid]);
        meta_buf[1] = fminf(meta_buf[1], sdata[block_size + tid]);
    }
  }

  if (tid == 0) {
      const int divisor = (1 << bits) - 1;
      meta_buf[0] = (meta_buf[0] - meta_buf[1]) / divisor;
  }
  __syncthreads();
}

inline __device__ unsigned char
MaxMinEncodeValue(float input, float* meta_info, float rand) {
  float* maxmin = ((float*)meta_info);
  if (maxmin[0] < EPS) {
    return 0;
  }
  float min = maxmin[1];
  float unit = maxmin[0];
  float d = (input - min) / unit + rand;
  unsigned char level = floor(d);
  return level;
}


inline __device__ void CompressBucket(float* input, unsigned char* output, float* meta_info, int num_elems, int bits, curandState* states) {
  using int64_t = long long int;
  int tid = threadIdx.x;
  int num_threads = blockDim.x-32;
  if (tid>=num_threads) return;
  float rand;
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  curandState state = states[tid + blockIdx.x * blockDim.x];
  for (int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += num_threads) {
      int64_t value = 0;
      for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
        int idx = i * PACK_SIZE + j;
        //rand = 0.5;
        //rand = get_rand(states);
        rand = curand_uniform(&state);
        //int sidx = floor(rand / 0.25);
        //stats[sidx]++;
        int64_t encoded = MaxMinEncodeValue(input[idx], meta_info, rand);
        value += (encoded << (j * bits));
      }
      for (int j = 0; j < bits && i * bits + j < num_char; j++) {
        output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;
      }
  }
  states[tid + blockIdx.x * blockDim.x] = state;
}


inline __device__ void quantize(float* input_data, unsigned char* output_data, int num_elems, int bucket_size, int bits, curandState* states) {
  //int num_blocks = gridDim.x;
  //int bid = blockIdx.x;
  //if (num_elems < 0) { 
  //  return;
  //}
  int num_blocks = 1;
  int bid = 0;

  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  int cur_bucket_size;
  float* meta = (float*)output_data;
  unsigned char* output;
  const int meta_multiplier = 2;
  output = output_data + meta_multiplier * sizeof(float) * num_buckets;

  int compressed_size = (bucket_size * bits + PACK_SIZE - 1) / PACK_SIZE;

  float* input = (float*)input_data;
  find_meta_seq(input, meta, num_elems, bucket_size, bits);
  //for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
  //  cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
  //  find_meta_parallel(input + bucket_size * bucket_id, (meta + meta_multiplier * bucket_id), cur_bucket_size, bits);
  //}
  //int stats[4] = {};
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket(
        input + bucket_size * bucket_id, output + compressed_size * bucket_id,
        (meta + meta_multiplier * bucket_id),
        cur_bucket_size, bits, states);
  }
  //if (threadIdx.x == 0 && blockIdx.x == 0 && gpuId == 0) {
  //     for (int i = 0; i < 4; i++)
  //       printf("%i ", stats[i]);
  //     printf("\n");
  //}
  __syncthreads();
}




inline __device__ float MaxMinDecodeValue(unsigned char input, float* meta_info, int idx, int bucket_size) {
  int bucket_no = idx / bucket_size;
  float* maxmin = ((float*)meta_info) + 2 * bucket_no;
  float min = maxmin[1];
  float unit = maxmin[0];
  return min + input * unit;
}

//template <bool ADD>
inline __device__ void dequantize(unsigned char* input_data, float* output, int num_elems, int bucket_size, int bits) {
  //int tid = threadIdx.x + blockIdx.x * (blockDim.x-32);
  //int stride = gridDim.x * (blockDim.x-32);
  int tid = threadIdx.x;
  int stride = blockDim.x -32;

  if (threadIdx.x<blockDim.x -32) {
    int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
    float* meta_info = (float*)input_data;
    unsigned char* input;
    const int meta_multiplier = 2;
    input = input_data + meta_multiplier * sizeof(float) * num_buckets;

    int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
    int divisor = 1 << bits;
    for (int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += stride) {
      int64_t value = 0;
      for (int j = 0; j < bits && i * bits + j < num_char; j++) {
        value |= ((int64_t)input[i * bits + j]) << (j * PACK_SIZE);
      }
      for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
        unsigned char encoded_value = (value >> (j * bits)) & (divisor - 1);
        float d = MaxMinDecodeValue(encoded_value, meta_info, i * PACK_SIZE + j, bucket_size);
        output[i * PACK_SIZE + j] = d;
      }
    }
  }
  __syncthreads();
}


inline __device__ void find_meta_seq(const half* input, half* meta, int num_elem, int bucket_size, int bits) {
  int index = threadIdx.x;
  int stride = blockDim.x -32;

  if (threadIdx.x < blockDim.x-32) {
     const int divisor = (1 << bits) - 1;
     half* meta_buf = (half*)meta;
     for (int i = index; i < (num_elem + bucket_size - 1) / bucket_size; i += stride) {
       half mmin = input[i * bucket_size];
       half mmax = input[i * bucket_size];
       for (int j = i * bucket_size + 1; j < fminf((i + 1) * bucket_size, num_elem); j++) {
         mmin = hmin(mmin, input[j]);
         mmax = hmax(mmax, input[j]);
       }
       meta_buf[2 * i] =   __hdiv(__hsub(mmax, mmin), __uint2half_rd(divisor));
       meta_buf[2 * i + 1] = mmin;
     }
  }
  __syncthreads();
}



inline __device__ unsigned char
MaxMinEncodeValue(half input, half* meta_info, float rand) {
  half* maxmin = ((half*)meta_info);
  half min =  maxmin[1];
  half unit = maxmin[0];
  if (__half2float(maxmin[0]) < EPS) {
    return 0;
  }
  half rand_fp16 = __float2half(rand);
  half d = __hadd(__hdiv(__hsub(input, min), unit), rand_fp16);
  unsigned char level = __half2uint_rd(d);
  return level;
}


inline __device__ void CompressBucket(half* input, unsigned char* output, half* meta_info, int num_elems, int bits, curandState* states) {
  using int64_t = long long int;
  int tid = threadIdx.x;
  int num_threads = blockDim.x-32;
  if (tid>=num_threads) return;
  float rand;
  int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  curandState state = states[tid + blockIdx.x * blockDim.x];
  for (int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += num_threads) {
      int64_t value = 0;
      for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
        int idx = i * PACK_SIZE + j;
        //rand = 0.5;
        rand = curand_uniform(&state);
        int64_t encoded = MaxMinEncodeValue(input[idx], meta_info, rand);
        value += (encoded << (j * bits));
      }
      for (int j = 0; j < bits && i * bits + j < num_char; j++) {
        output[i * bits + j] = value >> (PACK_SIZE * j) & 0xFF;
      }
  }
  states[tid + blockIdx.x * blockDim.x] = state;
}


inline __device__ void quantize(half* input_data, unsigned char* output_data, int num_elems, int bucket_size, int bits, curandState* states) {
  int num_blocks = 1;
  int bid = 0;
  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  int cur_bucket_size;
  half* meta = (half*)output_data;
  unsigned char* output;
  const int meta_multiplier = 2;
  output = output_data + meta_multiplier * sizeof(half) * num_buckets;
  int compressed_size = (bucket_size * bits + PACK_SIZE - 1) / PACK_SIZE;
  half* input = (half*)input_data;
  find_meta_seq(input, meta, num_elems, bucket_size, bits);
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket(
        input + bucket_size * bucket_id, output + compressed_size * bucket_id,
        (meta + meta_multiplier * bucket_id),
        cur_bucket_size, bits, states);
  }
  __syncthreads();
}

inline __device__ half MaxMinDecodeValue(unsigned char input, half* meta_info, int idx, int bucket_size) {
  int bucket_no = idx / bucket_size;
  half* maxmin = ((half*)meta_info) + 2 * bucket_no;
  half min = maxmin[1];
  half unit = maxmin[0];
  return __hadd(min, (__hmul(unit, __uint2half_rd((int)input))));
}

inline __device__ void dequantize(unsigned char* input_data, half* output, int num_elems, int bucket_size, int bits) {
  int tid = threadIdx.x;
  int stride = blockDim.x -32;

  if (threadIdx.x<blockDim.x -32) {
    int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
    half* meta_info = (half*)input_data;
    unsigned char* input;
    const int meta_multiplier = 2;
    input = input_data + meta_multiplier * sizeof(half) * num_buckets;

    int num_char = (bits * num_elems + PACK_SIZE - 1) / PACK_SIZE;
    int divisor = 1 << bits;
    for (int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += stride) {
      int64_t value = 0;
      for (int j = 0; j < bits && i * bits + j < num_char; j++) {
        value |= ((int64_t)input[i * bits + j]) << (j * PACK_SIZE);
      }
      for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
        unsigned char encoded_value = (value >> (j * bits)) & (divisor - 1);
        half d = MaxMinDecodeValue(encoded_value, meta_info, i * PACK_SIZE + j, bucket_size);
        output[i * PACK_SIZE + j] = d;
      }
    }
  }
  __syncthreads();
}

