//#include "devcomm.h"
//#include "primitives.h"
//#include "collectives.h"
#include "cuda_runtime.h"
#include "cuda.h"


template<typename T>
__device__ int compress(T* inputArray) {

  return 0;
}


template<typename T>
__device__ int compress(const T* inputArray) {

  return 1;
}


template<>
inline __device__ int compress<float>(float* inputArray) {

  return 2;
}


template<>
inline __device__ int compress<float>(const float* inputArray) {

  return 3;
}

