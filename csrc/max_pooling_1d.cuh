#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"
#include "trait.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
template <typename T>
__global__ void max_pooling_1d_kernel(
    const T* input,
    T* output,
    int num_heads,
    int q_len,
    int k_len,
    int out_len,
    int cache_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks
) {
    int bidh = blockIdx.y;
    int bidq = blockIdx.x;
    const T* in = input + bidh * (q_len * k_len) + bidq * k_len;
    T* out = output + bidh * (q_len * out_len) + bidq * out_len;
    
    // Calculate query block index (equivalent to off_bq in transform_score)
    int off_bq = (bidq + cache_len) / block_size;

    for (int k = threadIdx.x; k < out_len; k += blockDim.x) {
        // This is equivalent to `off_bk` in transform_score
        int off_bk = k;
        
        // Check causal + local window mask based on exact criteria from transform_score
        bool should_mask_inf = (off_bk < init_blocks);
        bool should_mask_neg_inf = (off_bq >= off_bk) && (off_bq < off_bk + local_blocks);
        
        if (should_mask_inf) {
            out[k] = TypeTraits<T>::inf();
        }
        else if (should_mask_neg_inf) {
            out[k] = -TypeTraits<T>::inf();
        }
        else {
            // Compute max pooling for other areas
            int start = k * stride - padding;
            int end = start + kernel_size;
            start = max(start, 0);
            end = min(end, k_len);
            
            T max_val = -TypeTraits<T>::inf();
            if (end > start) {
                max_val = in[start];
                for (int i = start + 1; i < end; i++) {
                    if (in[i] > max_val) {
                        max_val = in[i];
                    }
                }
            }
            out[k] = max_val;
        }
    }
}
} // namespace

template <typename T>
void max_pooling_1d_func(
    cudaStream_t stream,
    const T* input,
    T* output,
    int num_heads,
    int q_len,
    int k_len,
    int out_len,
    int cache_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks
) {
    const int threads_per_block = 256;
    
    dim3 grid(q_len, num_heads);
    dim3 block(threads_per_block, 1);
    
    max_pooling_1d_kernel<<<grid, block, 0, stream>>>(
        input, output, num_heads, q_len, k_len, out_len, cache_len, kernel_size, stride, padding, block_size, local_blocks, init_blocks
    );
} 