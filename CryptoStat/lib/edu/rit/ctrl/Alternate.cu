#include <stdint.h>
// #include <curand.h>
// #include <curand_kernel.h>
#include "../util/BigInt.cu"

__device__ void evaluate(int NA, int Asize, uint32_t* A, int NB, int Bsize, uint32_t* B, int R, int Csize, uint32_t* C, int a, int b) {
}

#include "../crst/FunctionKernel.cu"