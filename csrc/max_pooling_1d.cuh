#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"
#include "trait.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
template <typename T>
__global__ void max_pooling_1d_varlen_kernel(
    const T* input,  // [num_heads, total_q, max_k]
    T* output,       // [num_heads, total_q, out_len]
    const int* cu_seqlens_q,  // cumulative sequence lengths for queries
    const int* cu_seqlens_k,  // cumulative sequence lengths for keys
    const int* cache_lens,    // cache lengths for each batch [batch_size]
    int batch_size,
    int num_heads,
    int max_seqlen_q,
    int max_seqlen_k,
    int out_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks
) {
    // Grid: (total_q, num_heads)
    // Block: (threads_per_block)
    
    int bidh = blockIdx.y;  // head index
    int bidq_global = blockIdx.x;  // global query index across all batches
    
    // Find which batch this query belongs to
    int batch_idx = 0;
    int q_start = 0;
    int q_end = 0;
    int k_start = 0;
    int k_end = 0;
    
    // Binary search to find batch index
    for (int b = 0; b < batch_size; b++) {
        q_start = cu_seqlens_q[b];
        q_end = cu_seqlens_q[b + 1];
        k_start = cu_seqlens_k[b];
        k_end = cu_seqlens_k[b + 1];
        
        if (bidq_global >= q_start && bidq_global < q_end) {
            batch_idx = b;
            break;
        }
    }
    
    // Local query index within the batch
    int bidq_local = bidq_global - q_start;
    int seqlen_q = q_end - q_start;
    int seqlen_k = k_end - k_start;
    
    // Skip if this thread is outside the sequence length
    if (bidq_local >= seqlen_q) return;
    
    // Calculate input and output pointers
    // Input is packed: [num_heads, total_q, max_k]
    // We need to access the k values for this specific query
    const T* in = input + bidh * cu_seqlens_q[batch_size] * max_seqlen_k + bidq_global * max_seqlen_k;
    T* out = output + bidh * cu_seqlens_q[batch_size] * out_len + bidq_global * out_len;
    
    // Calculate query block index for masking
    int cache_len = cache_lens[batch_idx];  // Get cache_len for this batch
    int off_bq = (bidq_local + cache_len) / block_size;

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
            end = min(end, seqlen_k);  // Use actual sequence length for this batch
            
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

// Original kernel for backward compatibility
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

// Variable-length sequence version
template <typename T>
void max_pooling_1d_varlen_func(
    cudaStream_t stream,
    const T* input,
    T* output,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    const int* cache_lens,
    int batch_size,
    int num_heads,
    int max_seqlen_q,
    int max_seqlen_k,
    int out_len,
    int kernel_size,
    int stride,
    int padding,
    int block_size,
    int local_blocks,
    int init_blocks
) {
    const int threads_per_block = 256;
    
    // Total number of queries across all batches
    int total_q;
    cudaMemcpyAsync(&total_q, &cu_seqlens_q[batch_size], sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    dim3 grid(total_q, num_heads);
    dim3 block(threads_per_block, 1);
    
    max_pooling_1d_varlen_kernel<<<grid, block, 0, stream>>>(
        input, output, cu_seqlens_q, cu_seqlens_k, cache_lens, batch_size, num_heads, 
        max_seqlen_q, max_seqlen_k, out_len, kernel_size, stride, padding, block_size, local_blocks, init_blocks
    );
}

// Original fixed-length version for backward compatibility
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