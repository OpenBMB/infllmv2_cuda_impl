#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"

namespace {
/**
 * CUDA kernel to convert boolean mask to uint64 representation
 * Each thread processes one element in the output array
 */
__global__ void kernel_blockmask_to_uint64(
    const bool* blockmask,     // Input boolean mask
    uint64_t* result,          // Output uint64 array
    int batch_size,            // Total number of rows (flattened batch dimensions)
    int last_dim_size,         // Original last dimension size
    int n_uint64_per_row       // Number of uint64 needed per row
) {
    // Calculate global position
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    
    if (row >= batch_size || col >= n_uint64_per_row) return;
    
    // Calculate input and output offsets
    int out_idx = row * n_uint64_per_row + col;
    int in_idx_base = row * last_dim_size;
    
    // Calculate starting bit position for this uint64
    int bit_start = col * 64;
    
    // Initialize result
    uint64_t packed_value = 0;
    
    // Pack 64 bits (or fewer for the last uint64)
    for (int bit = 0; bit < 64; bit++) {
        int bit_pos = bit_start + bit;
        
        // Check if we're still within the valid range
        if (bit_pos < last_dim_size) {
            // Get the boolean value
            bool bit_value = blockmask[in_idx_base + bit_pos];
            
            // Set the corresponding bit in the result
            if (bit_value) {
                packed_value |= (1ULL << bit);
            }
        }
    }
    
    // Store the result
    result[out_idx] = packed_value;
}
} // namespace

/**
 * Function to convert boolean mask to uint64 representation
 */
void blockmask_to_uint64_func(
    cudaStream_t stream,
    const bool* blockmask,        // Input boolean mask
    uint64_t* result,             // Output uint64 array
    int batch_size,               // Total number of rows (flattened batch dimensions)
    int last_dim_size,            // Original last dimension size
    int n_uint64_per_row          // Number of uint64 needed per row
) {
    const int threads_per_block = 256;
    const int blocks_per_row = (batch_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_row, n_uint64_per_row);
    dim3 block(threads_per_block, 1);
    
    kernel_blockmask_to_uint64<<<grid, block, 0, stream>>>(
        blockmask, result, batch_size, last_dim_size, n_uint64_per_row
    );
} 