#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"
#include "trait.cuh"

namespace {
template<typename T>
__global__ void get_probs_kernel(
    T* attn_probs, float* lse, float scale, int dim
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float v = attn_probs[bid * dim + tid];
    attn_probs[bid * dim + tid] = exp(v * scale - lse[bid]);
}
}

template<typename T>
void get_probs_func(
    cudaStream_t stream,
    T* attn_probs, float* lse, float scale,
    int n, int dim
) {
    get_probs_kernel<T><<<n, dim, 0, stream>>>(attn_probs, lse, scale, dim);
}