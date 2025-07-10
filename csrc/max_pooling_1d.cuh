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
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    int num_heads,
    int batch_size,
    int total_q_len,
    int k_len,
    int max_blocks,
    int max_seqlen_q,
    int max_seqlen_k,
    int cache_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks
) {
    // Grid: (batch_size * num_heads, max_seqlen_q)
    // Block: (threads_per_block)
    
    int bid_bh = blockIdx.x;
    int bid_b = bid_bh / num_heads;
    int bid_h = bid_bh % num_heads;
    int bid_q = blockIdx.y;
    
    // Early exit if this batch doesn't exist
    if (bid_b >= batch_size) return;
    
    // Get sequence boundaries for this batch
    int q_start = cu_seqlens_q[bid_b];
    int q_end = cu_seqlens_q[bid_b + 1];
    int q_len = q_end - q_start;
    
    int k_start = cu_seqlens_k[bid_b];
    int k_end = cu_seqlens_k[bid_b + 1];
    int k_len_batch = k_end - k_start;
    
    // Early exit if this query position doesn't exist for this batch
    if (bid_q >= q_len) return;
    
    // Calculate global query position
    int global_q_idx = q_start + bid_q;
    
    // Input pointer for this head and query position
    const T* in = input + bid_h * (total_q_len * k_len) + global_q_idx * k_len;
    
    // Output pointer for this head and query position
    T* out = output + bid_h * (total_q_len * max_blocks) + global_q_idx * max_blocks;
    
    // Calculate query block index (equivalent to off_bq in transform_score)
    int off_bq = (bid_q + cache_len) / block_size;

    for (int k = threadIdx.x; k < max_blocks; k += blockDim.x) {
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
            end = min(end, k_len_batch);
            
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
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    int num_heads,
    int batch_size,
    int total_q_len,
    int k_len,
    int max_blocks,
    int max_seqlen_q,
    int max_seqlen_k,
    int cache_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks
) {
    const int threads_per_block = 256;
    
    // Launch kernel for all batches and heads
    dim3 grid(batch_size * num_heads, max_seqlen_q);
    dim3 block(threads_per_block, 1);
    
    max_pooling_1d_kernel<<<grid, block, 0, stream>>>(
        input, output, cu_seqlens_q, cu_seqlens_k,
        num_heads, batch_size, total_q_len, k_len, max_blocks,
        max_seqlen_q, max_seqlen_k, cache_len, kernel_size, stride, 
        padding, block_size, local_blocks, init_blocks
    );
} 