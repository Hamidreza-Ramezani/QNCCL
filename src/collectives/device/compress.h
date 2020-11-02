//#include "devcomm.h"
//#include "primitives.h"
//#include "collectives.h"
#include "cuda_runtime.h"
#include "cuda.h"


template<typename T>
__device__ int* compress(T* dst) {
  
  int a = 0;
  int* ptr = new int;
  *ptr = a; 
  return ptr;
}


template<typename T>
__device__ int* compress(const T* src) {

  int a = 1;
  int* ptr = new int;
  *ptr = a; 
  return ptr;
}


template<>
inline __device__ int* compress<float>(float* dst) {

  int a = 2;
  int* ptr = new int;
  *ptr = a; 
  return ptr;
}


template<>
inline __device__ int* compress<float>(const float* src) {

  int a = 3;
  int* ptr = new int;
  *ptr = a; 
  return ptr;
}
