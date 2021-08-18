#include <stdint.h>
#include "../util/BigInt.cu"

__device__ void evaluate(int NA, int Asize, uint32_t* A, int NB, int Bsize, uint32_t* B, int R, int Csize, uint32_t* C, int a, int b) {
  for (int i = 0; i <= 9; ++i) {
    C[i] = 0;
  }
}

#include "../crst/FunctionKernel.cu"