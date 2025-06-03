#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"
#include "trait.cuh"
namespace {
template<typename T, int N>
static __device__ inline void warpBitonicSort(T& v1, int& pos, bool asc) {
    int lane_id = threadIdx.x & (N - 1);
    #pragma unroll
    for (int k = 2; k <= N; k *= 2) {
        bool desc = ((lane_id & k) == 0) ^ asc;
        #pragma unroll
        for (int j = k / 2; j > 0; j /= 2) {
            T v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
            int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos, j);
            bool upper = (lane_id & j) != 0;
            if (desc ^ (v1 > v2 || (v1 == v2 && pos < pos2)) ^ upper) {
                v1 = v2;
                pos = pos2;
            }
        }
    }
}
template<typename T, int N>
static __device__ inline void warpBitonicMerge(T& v1, int& pos1, T& v2, int& pos2) {
    if (v1 < v2 || (v1 == v2 && pos1 > pos2)) {
        v1 = v2;
        pos1 = pos2;
    }
    int lane_id = threadIdx.x & (N - 1);
    // resort
    #pragma unroll
    for (int j = N / 2; j > 0; j /= 2) {
        v2 = __shfl_xor_sync(0xFFFFFFFF, v1, j);
        int pos2 = __shfl_xor_sync(0xFFFFFFFF, pos1, j);
        bool upper = (lane_id & j) != 0;
        if ((v1 < v2 || (v1 == v2 && pos1 > pos2)) ^ upper) {
            v1 = v2;
            pos1 = pos2;
        }
    }
}
template<typename T, int N>
static __device__ inline void blockBitonicReduce(T& v, int& pos) {
    __shared__ T shared_val[1024];
    __shared__ int shared_pos[1024];
    // block reduce
    shared_val[threadIdx.x] = v;
    shared_pos[threadIdx.x] = pos;
    // inter warp reduce
    #pragma unroll
    for (int i = 512; i >= 32; i >>= 1) {
        if (blockDim.x > i) {
            __syncthreads();
            if (threadIdx.x < i) {
                int idx_next = (i << 1) - threadIdx.x - 1;
                T nw_v = (idx_next < blockDim.x) ? shared_val[idx_next] : T(-TypeTraits<T>::inf());
                int nw_pos = (idx_next < blockDim.x) ? shared_pos[idx_next] : -1;
                warpBitonicMerge<T, N>(v, pos, nw_v, nw_pos); // merge and rebuild in desc order
                shared_val[threadIdx.x] = v;
                shared_pos[threadIdx.x] = pos;
            }
        }
    }
    // intra warp reduce
    if (threadIdx.x < 32) {
        warpBitonicSort<T, 32>(v, pos, false);
    }
}
template<typename T, int N>
static __global__ void kernel_bitonic_topk(
    int n, int top,
    T *inp,     // (batch, n)
    float *out,     // (batch, top)
    int *idx    // (batch, top)
) {
    int offset_inp = blockIdx.x * n;
    int offset_out = blockIdx.x * top;
    T local_v = threadIdx.x < n ? inp[offset_inp + threadIdx.x] : -TypeTraits<T>::inf();
    int local_pos = threadIdx.x;
    warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order
    for (int i = blockDim.x; i < n; i += blockDim.x) {
        T nw_v = (i + threadIdx.x) < n ? inp[offset_inp + i + threadIdx.x] : -TypeTraits<T>::inf();
        int nw_pos = i + threadIdx.x;
        // step.1: local sort
        warpBitonicSort<T, N>(nw_v, nw_pos, true); // local sort in asc order
        // step.2&3: merge and rebuild
        warpBitonicMerge<T, N>(local_v, local_pos, nw_v, nw_pos); // merge and rebuild in desc order
    }
    blockBitonicReduce<T, N>(local_v, local_pos);
    if (threadIdx.x < top) {
        out[offset_out + threadIdx.x] = local_v;
        idx[offset_out + threadIdx.x] = local_pos;
    }
}
// intra-block topk
// gridDim(batch, n / 1024, 1), threadDim(1024, 1, 1)
template<typename T, int N, bool ordered>
static __global__ void kernel_bitonic_topk_multiblock(
    int n,
    const T *inp,       // (batch, n)
    const int *idx_inp, // (batch, n)
    T *out,     // (batch, n / 1024 * N)
    int *idx    // (batch, n / 1024 * N)
) {
    int offset_col = blockIdx.y * blockDim.x + threadIdx.x;
    int offset_inp = blockIdx.x * n + offset_col;
    int offset_out = blockIdx.x * (gridDim.y * N) + blockIdx.y * N + threadIdx.x;
    T local_v = (offset_col < n) ? inp[offset_inp] : T(-TypeTraits<T>::inf());
    int local_pos = (idx_inp == nullptr) ? offset_col : idx_inp[offset_inp];
    if (!ordered) warpBitonicSort<T, N>(local_v, local_pos, false); // local sort in desc order
    blockBitonicReduce<T, N>(local_v, local_pos);
    if (threadIdx.x < N) {
        out[offset_out] = local_v;
        idx[offset_out] = local_pos;
    }
}

#define TOPK_SIZE_DISPATCH(top, ...) \
    do { \
        const int &top_v = top; \
        if (top_v > 16) { \
            const int top_size = 32; \
            __VA_ARGS__ \
        } else if (top_v > 8) { \
            const int top_size = 16; \
            __VA_ARGS__ \
        } else if (top_v > 4) { \
            const int top_size = 8; \
            __VA_ARGS__ \
        } else if (top_v > 2) { \
            const int top_size = 4; \
            __VA_ARGS__ \
        } else if (top_v > 1) { \
            const int top_size = 2; \
            __VA_ARGS__ \
        } else { \
            const int top_size = 1; \
            __VA_ARGS__ \
        } \
    } while(0)

template <typename T>
void bitonic_topk(
    const cudaStream_t stream,
    const int batch,
    const int n,
    const int top,
    const T* x, 
    T* out, 
    int* pos
) {
    TOPK_SIZE_DISPATCH(top, {
        dim3 blockDim(n, 1, 1);
        dim3 gridDim(batch, 1, 1);
        kernel_bitonic_topk_multiblock<T, top_size, false><<<gridDim, blockDim, 0, stream>>>(
            n,
            x,
            nullptr,
            out,
            pos
        );
    });
}
} // namespace

template<typename T>
void topk_func(
    cudaStream_t stream,
    int num_tokens, int dim, int top, int dtype,
    T* x,
    T* topk_val, int* topk_pos
) {
    bitonic_topk<T>(
        stream,
        num_tokens,
        dim, top,
        x,
        topk_val, topk_pos
    );
}