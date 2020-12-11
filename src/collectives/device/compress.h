#include "cuda_runtime.h"
#include "cuda.h"
#include <stdint.h>

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



////template<>
////inline __device__ const int* compress<float>(const float* src, const int* compressedSrc, int nelem) {
//__device__ __forceinline__ void compress(const float* src, int* compressedSrc, int offset, int nelem, int nthreads) {
//  const int tid = threadIdx.x;
//  for (int idx = offset+tid; idx < offset+nelem; idx += nthreads) {
//    int var;
//    if (src[idx] < 0) {
//      var = static_cast<int> (src[idx] - 0.5);
//    } else {
//      var = static_cast<int> (src[idx] + 0.5); 
//    }
//    compressedSrc[idx] = var;
//  }
//}
//
//
////template<>
//__device__  __forceinline__ void compress(float src, int* compressedSrc, int nthreads) {
//  if (src < 0) {
//    *compressedSrc = static_cast<int> (src - 0.5);
//  } else { 
//    *compressedSrc = static_cast<int> (src + 0.5);
//  }
//}


//__device__ __forceinline__ void compress(const float* src, int* compressedSrc, int offset, int nelem, int nthreads) {
//  for (int idx = offset; idx < offset+nelem; idx += 1) {
//    int var;
//    if (src[idx] < 0) {
//       var = static_cast<int> (src[idx] - 0.5);
//    }
//    else {
//       var = static_cast<int> (src[idx] + 0.5); 
//    }
//    compressedSrc[idx] = var;
//  }
//}
//
//__device__  __forceinline__ void compress(float src, int* compressedSrc, int nthreads) {
//  if (src < 0) {
//     *compressedSrc = static_cast<int> (src - 0.5);
//  }
//  else { 
//  *compressedSrc = static_cast<int> (src + 0.5);
//  }
//}



//__device__ __forceinline__ int* compress(const float* src, int* compressedSrc, int offset, int nelem, int nthreads) {
//  for (int idx = offset; idx < offset+nelem; idx += 1) {
//    int var;
//    if (src[idx] < 0) {
//       var = static_cast<int> (src[idx] - 0.5);
//    }
//    else {
//       var = static_cast<int> (src[idx] + 0.5); 
//    }
//    compressedSrc[idx] = var;
//  }
//  return compressedSrc;
//}
//
//__device__  __forceinline__ int compress(float src, int nthreads) {
//  if (src < 0) {
//     return static_cast<int> (src - 0.5);
//  }
//  return static_cast<int> (src + 0.5); 
//}



template <int BITS>
__device__ void find_meta_seq(const float* input, float* meta, int num_elem, int bucket_size, int nthreads) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  //int index = threadIdx.x;
  //int stride = nthreads;
  //int index = 1;
  //int stride = blockDim.x;
  //if(index == 0){
  //   for (int i=0; i<num_elem; i++) { 
  //      printf("input is %f ",input[i]);
  //   }
  //   printf("\n");
  //}

  if (threadIdx.x == 0) {
  printf("block size: %d\n", blockDim.x);
  printf("grid size: %d\n", gridDim.x);
  const int divisor = (1 << BITS) - 1;
  float* meta_buf = (float*)meta;
  for (int i = index; i < (num_elem); i += gridDim.x * bucket_size) {
  //for (int i = index; i < (num_elem + bucket_size - 1) / bucket_size; i += gridDim.x) {
  //for (int i = index; i < (num_elem + bucket_size - 1) / bucket_size; i += stride) {
    float mmin = input[i];
    //float mmin = input[i * bucket_size];
    float mmax = input[i];
    //float mmax = input[i * bucket_size];
    for (int j = i + 1; j < fminf(bucket_size, num_elem-i); j++) {
    //for (int j = i * bucket_size + 1; j < fminf((i + 1) * bucket_size, num_elem); j++) {
      mmin = fminf(mmin, input[j]);
      mmax = fmaxf(mmax, input[j]);
    }
    if(mmin==0 ){
     printf("input is ");
     for (int i=0; i<num_elem; i++) { 
        printf("%f ",input[i]);
     }
     printf("\n");
    //printf("mmin is %f\n",mmin);
    //printf("mmax is %f\n",mmax);
    }
    meta_buf[i/bucket_size] = static_cast<float>((mmax - mmin) / divisor);
    //meta_buf[2 * i] = static_cast<float>((mmax - mmin) / divisor);
    meta_buf[i/bucket_size + 1] = mmin;
    //meta_buf[2 * i + 1] = mmin;
  }
   
  }
  __syncthreads();
  
}

template <int BITS>
inline __device__ void find_meta_parallel(float* input, float* meta, int num_elems, int nthreads) {
  int tid = threadIdx.x;
  int block_size = blockDim.x;
  //int block_size = 1;
  float* meta_buf = (float*)meta;
  const int MAX_NTHREADS = 288;
  const int shared_size = MAX_NTHREADS * 2;
  __shared__ float sdata[shared_size];
  meta_buf[0] = input[0];
  meta_buf[1] = input[0];
  int num_iters_per_bucket = (num_elems + block_size - 1) / block_size;
  for (int i = 0; i < num_iters_per_bucket; i++) {
    //int idx = tid;
    //int idx = i + tid;
    int idx = i * blockDim.x + tid;
    if (idx < num_elems) {
        sdata[tid] = input[idx];
        sdata[block_size + tid] = input[idx];
    }
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
      if (tid < s && idx + s < num_elems) {
        sdata[tid] = fmaxf(sdata[tid + s], sdata[tid]);
        sdata[block_size + tid] =
            fminf(sdata[block_size + tid + s], sdata[block_size + tid]);
      }
      __syncthreads();
    }

    if (tid == 0) {
        meta_buf[0] = fmaxf(meta_buf[0], sdata[tid]);
        meta_buf[1] = fminf(meta_buf[1], sdata[block_size + tid]);
    }
  }
  
  if (tid == 0) {
      const int divisor = (1 << BITS) - 1;
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


template <int BITS>
inline __device__ void CompressBucket(float* input, unsigned char* output, float* meta_info, int num_elems, int nthreads) {
  using int64_t = long long int;
  int tid = threadIdx.x;
  //int num_threads = nthreads;
  int num_threads = blockDim.x;
  float rand;
  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  for (int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += num_threads) {
      int64_t value = 0;
      for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
        int idx = i * PACK_SIZE + j;
        rand = 0.5;
        //rand = GetRand(state);
        int64_t encoded = MaxMinEncodeValue(input[idx], meta_info, rand);
        value += (encoded << (j * BITS));
      }
      for (int j = 0; j < BITS && i * BITS + j < num_char; j++) {
        output[i * BITS + j] = value >> (PACK_SIZE * j) & 0xFF;
      }
  }
}


template <int BITS>
inline __device__ void quantize(float* input_data, unsigned char* output_data, int num_elems, int bucket_size, int nthreads) {
  if (num_elems < 0) { 
   num_elems = 0; 
  }  

  int num_blocks = gridDim.x;
  int bid = blockIdx.x;

  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  int cur_bucket_size;
  float* meta = (float*)output_data;
  unsigned char* output;
  const int meta_multiplier = 2;
  output = output_data + meta_multiplier * sizeof(float) * num_buckets;

  int compressed_size = (bucket_size * BITS + PACK_SIZE - 1) / PACK_SIZE;

  float* input = (float*)input_data;
  find_meta_seq<BITS>(input, meta, num_elems, bucket_size, nthreads);
  //for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
  //  cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
  //  find_meta_parallel<BITS>(input + bucket_size * bucket_id,(meta + meta_multiplier * bucket_id), cur_bucket_size, nthreads);
  //  //find_meta_parallel<BITS>(input + bucket_size * bucket_id,(unsigned char*)(meta + meta_multiplier * bucket_id), cur_bucket_size);
  //}
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket<BITS>(
        input + bucket_size * bucket_id, output + compressed_size * bucket_id,
        (meta + meta_multiplier * bucket_id),
        cur_bucket_size, nthreads);
  }
}


template <int BITS>
inline __device__ void quantize(const float* input_data, unsigned char* output_data, int num_elems, int bucket_size, int nthreads) {

  if (num_elems < 0) { 
   num_elems = 0; 
  }  
  int num_blocks = gridDim.x;
  int bid = blockIdx.x;

  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  int cur_bucket_size;
  float* meta = (float*)output_data;
  unsigned char* output;
  const int meta_multiplier = 2;
  output = output_data + meta_multiplier * sizeof(float) * num_buckets;

  int compressed_size = (bucket_size * BITS + PACK_SIZE - 1) / PACK_SIZE;

  float* input = (float*)input_data;
  find_meta_seq<BITS>(input, meta, num_elems, bucket_size, nthreads);
  //for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
  //  cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
  //  find_meta_parallel<BITS>(input + bucket_size * bucket_id, (meta + meta_multiplier * bucket_id), cur_bucket_size, nthreads);
  //  //find_meta_parallel<BITS>(input + bucket_size * bucket_id,(unsigned char*)(meta + meta_multiplier * bucket_id), cur_bucket_size);
  //}
  for (int bucket_id = bid; bucket_id < num_buckets; bucket_id += num_blocks) {
    cur_bucket_size = umin(bucket_size, num_elems - bucket_id * bucket_size);
    CompressBucket<BITS>(
        input + bucket_size * bucket_id, output + compressed_size * bucket_id,
        (meta + meta_multiplier * bucket_id),
        cur_bucket_size, nthreads);
  }
}


inline __device__ float MaxMinDecodeValue(unsigned char input, float* meta_info, int idx, int bucket_size) {
  int bucket_no = idx / bucket_size;
  float* maxmin = ((float*)meta_info) + 2 * bucket_no;
  float min = maxmin[1];
  float unit = maxmin[0];
  return min + input * unit;
}

template <bool ADD, int BITS>
inline __device__ void dequantize(unsigned char* input_data, float* output, int num_elems, int bucket_size, int nthreads) {
  if (num_elems < 0) { 
   num_elems = 0; 
  }  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;

  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
  float* meta_info = (float*)input_data;
  unsigned char* input;
  const int meta_multiplier = 2;
  input = input_data + meta_multiplier * sizeof(float) * num_buckets;

  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
  int divisor = 1 << BITS;
  for (int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += stride) {
    int64_t value = 0;
    for (int j = 0; j < BITS && i * BITS + j < num_char; j++) {
      value |= ((int64_t)input[i * BITS + j]) << (j * PACK_SIZE);
    }
    for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
      unsigned char encoded_value = (value >> (j * BITS)) & (divisor - 1);
      float d = MaxMinDecodeValue(encoded_value, meta_info, i * PACK_SIZE + j, bucket_size);
      if (ADD) {
        output[i * PACK_SIZE + j] = output[i * PACK_SIZE + j] + d;
      } else {
        output[i * PACK_SIZE + j] = d;
      }
    }
  }
}




//template <bool ADD, int BITS>
//inline __device__ void dequantize(unsigned char* input_data, float* output, int num_elems, int bucket_size, int nthreads) {
//  //if (num_elems < 0) {
//  //  num_elems = 0;
//  //}
//  int tid = threadIdx.x;
//  int stride = nthreads;
//  int num_buckets = (num_elems + bucket_size - 1) / bucket_size;
//  float* meta_info = (float*)input_data;
//  unsigned char* input; 
//  const int meta_multiplier = 2;
//  input = input_data + meta_multiplier * sizeof(float) * num_buckets;
//  
//  int num_char = (BITS * num_elems + PACK_SIZE - 1) / PACK_SIZE;
//  int divisor = 1 << BITS;
//  for (int i = tid; i < (num_elems + PACK_SIZE - 1) / PACK_SIZE; i += stride) {
//    int64_t value = 0;
//    for (int j = 0; j < BITS && i * BITS + j < num_char; j++) {
//      value |= ((int64_t)input[i * BITS + j]) << (j * PACK_SIZE);
//    }
//    for (int j = 0; j < PACK_SIZE && i * PACK_SIZE + j < num_elems; j++) {
//      unsigned char encoded_value = (value >> (j * BITS)) & (divisor - 1);
//      float d = MaxMinDecodeValue(encoded_value, meta_info, i * PACK_SIZE + j, bucket_size);
//      if (ADD) {
//        output[i * PACK_SIZE + j] = output[i * PACK_SIZE + j] + d;
//      } else {
//        output[i * PACK_SIZE + j] = d;
//      }
//    }
//  }
//}
