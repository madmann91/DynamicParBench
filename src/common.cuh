#ifndef COMMON_CUH
#define COMMON_CUH

#include <iostream>

#define CHECK_CUDA_CALL(x) check_cuda_call(x, __FILE__, __LINE__)

__host__ static void check_cuda_call(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << file << " (" << line << "): " << cudaGetErrorString(err) << std::endl;
        abort();
    }
}

static int block_size = 32;

__host__ __device__ __forceinline__ int round_to(int i, int n) {
    return i / n + (i % n ? 1 : 0);
}

#endif
