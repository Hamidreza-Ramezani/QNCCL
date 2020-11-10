#include "cuda_runtime.h"
#include "cuda.h"

template<typename T>
__device__ T* compress(T* dst, T* compressedDst, int nelem) {
   
  return dst;
}


template<typename T>
__device__ const T* compress(const T* src, const T* compressedSrc, int nelem) {

  return src;
}


template<typename T>
__device__ T compress(T src) {

  return src;
}


//template<>
//inline __device__ int* compress<float>(float* dst, int* compressedDst, int nelem) {
__device__ __forceinline__ int* compress(float* dst, int* compressedDst, int nelem, int nthreads) {
  for (int idx = 0; idx < nelem; idx += 1) {
    int var;
    if (dst[idx] < 0) {
       var = static_cast<int> (dst[idx] - 0.5);
    }
    else {
       var = static_cast<int> (dst[idx] + 0.5); 
    }
    compressedDst[idx] = var;
  }

  return compressedDst;
}


//template<>
//inline __device__ const int* compress<float>(const float* src, const int* compressedSrc, int nelem) {
inline __device__ void compress(const float* src, int* compressedSrc, int offset, int nelem, int nthreads) {
  const int tid = threadIdx.x;
  for (int idx = offset+tid; idx < offset+nelem; idx += nthreads) {
    int var;
    if (src[idx] < 0) {
       var = static_cast<int> (src[idx] - 0.5);
    }
    else {
       var = static_cast<int> (src[idx] + 0.5); 
    }
    compressedSrc[idx] = var;
  }
}


//template<>
__device__  __forceinline__ void compress(float src, int* compressedSrc, int nthreads) {
  if (src < 0) {
     *compressedSrc = static_cast<int> (src - 0.5);
  }
  else { 
  *compressedSrc = static_cast<int> (src + 0.5);
  }
}


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


inline int roundNo(float num) {
  if (num < 0) {
     return (int) (num - 0.5);
  }
  return (int) (num + 0.5); 
}
