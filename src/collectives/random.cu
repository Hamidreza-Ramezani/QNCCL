#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "cuda.h"


__global__ void setup_kernel(curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, 0, 0, &state[id]);
}

__global__ void generate_uniform_kernel(curandState *state, float* result) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  float random_number;
  curandState localState = state[id];
  random_number = curand_uniform(&localState);
  state[id] = localState;
  result[id] = random_number;
}
