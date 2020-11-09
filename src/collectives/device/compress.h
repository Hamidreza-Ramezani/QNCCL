//#include "devcomm.h"
//#include "primitives.h"
//#include "collectives.h"
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
inline __device__ int* compress(float* dst, int* compressedDst, int nelem, int nthreads) {
  const int tid = threadIdx.x;
  for (int idx = tid; idx < nelem; idx += nthreads) {
    int var;
    if (dst[idx] < 0) {
       var = (int) (dst[idx] - 0.5);
    }
    else {
       var = (int) (dst[idx] + 0.5); 
    }
    compressedDst[idx] = var;
  }
  return compressedDst;
}


//template<>
//inline __device__ const int* compress<float>(const float* src, const int* compressedSrc, int nelem) {
inline __device__ int* compress(const float* src, int* compressedSrc, int nelem, int nthreads) {
  const int tid = threadIdx.x;
  for (int idx = tid; idx < nelem; idx += nthreads) {
    int var;
    if (src[idx] < 0) {
       var = (int) (src[idx] - 0.5);
    }
    else {
       var = (int) (src[idx] + 0.5); 
    }
    compressedSrc[idx] = var;
  }
  return compressedSrc;
}


//template<>
inline __device__ int compress(float src, int nthreads) {
  if (src < 0) {
     return (int) (src - 0.5);
  }
  return (int) (src + 0.5); 
}


inline int roundNo(float num) {
  if (num < 0) {
     return (int) (num - 0.5);
  }
  return (int) (num + 0.5); 
}
