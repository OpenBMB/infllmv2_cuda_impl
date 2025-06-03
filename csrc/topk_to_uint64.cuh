#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"

namespace {
/**
 * CUDA kernel to convert topk indices directly to uint64 representation
 * Each thread processes one element in the output array
 */
__global__ void kernel_topk_to_uint64(
    const int* topk_idx,       // Input topk indices [num_heads, total_seqlen, k]
    uint64_t* result,          // Output uint64 array
    int batch_size,            // Total number of rows (flattened batch dimensions)
    int k,                     // Number of topk values per row
    int k_blocks,              // Number of key blocks
    int n_uint64_per_row       // Number of uint64 needed per row
) {
    // Calculate global position
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    
    if (row >= batch_size || col >= n_uint64_per_row) return;
    
    // Calculate output offset
    int out_idx = row * n_uint64_per_row + col;
    
    // Calculate starting bit position for this uint64
    int bit_start = col * 64;
    
    // Initialize result
    uint64_t packed_value = 0;
    
    // For each topk index in this row
    for (int i = 0; i < k; i++) {
        // Get the index value
        int idx_offset = row * k + i;
        int idx = topk_idx[idx_offset];
        
        // Skip if the index is -1 (invalid)
        if (idx == -1) continue;
        
        // Check if this idx falls within the current uint64 chunk
        if (idx >= bit_start && idx < bit_start + 64) {
            // Set the corresponding bit in the packed value
            int local_bit = idx - bit_start;
            packed_value |= (1ULL << local_bit);
        }
    }
    
    // Store the result
    result[out_idx] = packed_value;
}
} // namespace

/**
 * Function to convert topk indices directly to uint64 representation
 */
void topk_to_uint64_func(
    cudaStream_t stream,
    const int* topk_idx,          // Input topk indices
    uint64_t* result,             // Output uint64 array
    int batch_size,               // Total number of rows (flattened batch dimensions)
    int k,                        // Number of topk values per row
    int k_blocks,                 // Number of key blocks
    int n_uint64_per_row          // Number of uint64 needed per row
) {
    const int threads_per_block = 256;
    const int blocks_per_row = (batch_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_row, n_uint64_per_row);
    dim3 block(threads_per_block, 1);
    
    kernel_topk_to_uint64<<<grid, block, 0, stream>>>(
        topk_idx, result, batch_size, k, k_blocks, n_uint64_per_row
    );
} 