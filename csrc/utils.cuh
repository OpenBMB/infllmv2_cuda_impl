#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define ROUND_UP(M, N) (((M) + (N) - 1) / (N) * (N))

#define cudaCheck(err) \
    if (err != cudaSuccess) { \
        std::cerr << "cuda error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define cublasCheck(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << err << std::endl; \
        exit(EXIT_FAILURE); \
    }
