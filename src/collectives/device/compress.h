//#include "devcomm.h"
//#include "primitives.h"
//#include "collectives.h"
#include "cuda_runtime.h"
#include "cuda.h"


template<typename T>
__device__ int* compress(T* dst, int* compressedDst, int nelem) {
   
  return compressedDst;
}


template<typename T>
__device__ int* compress(const T* src, int* compressedSrc, int nelem) {

  return compressedSrc;
}


template<>
inline __device__ int* compress<float>(float* dst, int* compressedDst, int nelem) {

  return compressedDst;
}


template<>
inline __device__ int* compress<float>(const float* src, int* compressedSrc, int nelem) {

  return compressedSrc;
}
