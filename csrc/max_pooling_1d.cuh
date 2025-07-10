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
        
        // Debug print for specific position
        if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
            printf("[DEBUG] h=%d, q=%d, b=%d:\n", bidh, bidq, k);
            printf("  cache_len=%d, block_size=%d\n", cache_len, block_size);
            printf("  off_bq = (%d + %d) / %d = %d\n", bidq, cache_len, block_size, off_bq);
            printf("  off_bk = %d\n", off_bk);
            printf("  init_blocks=%d, local_blocks=%d\n", init_blocks, local_blocks);
        }
        
        // Check causal + local window mask based on exact criteria from transform_score
        bool should_mask_inf = (off_bk < init_blocks);
        bool should_mask_neg_inf = (off_bq >= off_bk) && (off_bq < off_bk + local_blocks);
        
        if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
            printf("  Masking conditions:\n");
            printf("    should_mask_inf = (off_bk=%d < init_blocks=%d) = %s\n", 
                   off_bk, init_blocks, should_mask_inf ? "true" : "false");
            printf("    should_mask_neg_inf = (off_bq=%d >= off_bk=%d) && (off_bq=%d < off_bk=%d + local_blocks=%d)\n",
                   off_bq, off_bk, off_bq, off_bk, local_blocks);
            printf("                        = (%s) && (%d < %d) = %s\n",
                   (off_bq >= off_bk) ? "true" : "false", off_bq, off_bk + local_blocks,
                   should_mask_neg_inf ? "true" : "false");
        }
        
        if (should_mask_inf) {
            out[k] = TypeTraits<T>::inf();
            if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
                printf("  Result: Masked with inf\n");
            }
        }
        else if (should_mask_neg_inf) {
            out[k] = -TypeTraits<T>::inf();
            if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
                printf("  Result: Masked with -inf\n");
            }
        }
        else {
            // Compute max pooling for other areas
            int start = k * stride - padding;
            int end = start + kernel_size;
            start = max(start, 0);
            end = min(end, k_len);
            
            if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
                printf("  Max pooling computation:\n");
                printf("    start = %d * %d - %d = %d\n", k, stride, padding, k * stride - padding);
                printf("    end = %d + %d = %d\n", start, kernel_size, start + kernel_size);
                printf("    start (clamped) = max(%d, 0) = %d\n", k * stride - padding, start);
                printf("    end (clamped) = min(%d, %d) = %d\n", start + kernel_size, k_len, end);
            }
            
            T max_val = -TypeTraits<T>::inf();
            if (end > start) {
                max_val = in[start];
                if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
                    printf("    Pooling range [%d, %d):\n", start, end);
                    printf("    in[%d] = %f\n", start, float(in[start]));
                }
                for (int i = start + 1; i < end; i++) {
                    if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
                        printf("    in[%d] = %f\n", i, float(in[i]));
                    }
                    if (in[i] > max_val) {
                        max_val = in[i];
                    }
                }
                if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
                    printf("    max_val = %f\n", float(max_val));
                }
            }
            out[k] = max_val;
            if (bidh == 0 && bidq == 234 && k == 6 && threadIdx.x == 6) {
                printf("  Result: %f\n", float(out[k]));
            }
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