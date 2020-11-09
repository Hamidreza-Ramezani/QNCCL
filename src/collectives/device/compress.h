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
inline __device__ int* compress(float* dst, int* compressedDst, int nelem) {

  return compressedDst;
}


//template<>
//inline __device__ const int* compress<float>(const float* src, const int* compressedSrc, int nelem) {
inline __device__ const int* compress(const float* src, const int* compressedSrc, int nelem) {

  return compressedSrc;
}

//template<>
inline __device__ int compress(float src) {

  return 5;
}
