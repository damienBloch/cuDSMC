#ifndef UTILS_H
#define UTILS_H
#include<iostream>
#include<string>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<stdexcept>
#include<curand_kernel.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include"helper_math.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "Error: %s\nFile %s, line %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)


int cudaDeviceGetCount(void);
void cudaInit(void);

#endif
