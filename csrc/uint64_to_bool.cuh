#pragma once
#include <cuda_runtime.h>
#include "utils.cuh"

namespace {
/**
 * CUDA kernel to convert uint64 representation back to boolean mask
 * Each thread processes one element in the output array
 */
__global__ void kernel_uint64_to_bool(
    const uint64_t* input,     // Input uint64 array
    bool* result,              // Output boolean mask
    int batch_size,            // Total number of rows (flattened batch dimensions)
    int last_dim_size,         // Original last dimension size
    int n_uint64_per_row       // Number of uint64 needed per row
) {
    // Calculate global position
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    
    if (row >= batch_size || col >= last_dim_size) return;
    
    // Calculate which uint64 element contains this bit
    int uint64_idx = col / 64;
    
    // Calculate bit position within the uint64
    int bit_pos = col % 64;
    
    // Calculate input offset
    int in_idx = row * n_uint64_per_row + uint64_idx;
    
    // Calculate output offset
    int out_idx = row * last_dim_size + col;
    
    // Get the uint64 value
    uint64_t packed_value = input[in_idx];
    
    // Extract the bit and convert to boolean
    bool bit_value = (packed_value & (1ULL << bit_pos)) != 0;
    
    // Store the result
    result[out_idx] = bit_value;
}
} // namespace

/**
 * Function to convert uint64 representation back to boolean mask
 */
void uint64_to_bool_func(
    cudaStream_t stream,
    const uint64_t* input,        // Input uint64 array
    bool* result,                 // Output boolean mask
    int batch_size,               // Total number of rows (flattened batch dimensions)
    int last_dim_size,            // Original last dimension size
    int n_uint64_per_row          // Number of uint64 needed per row
) {
    const int threads_per_block = 256;
    const int blocks_per_row = (batch_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_row, last_dim_size);
    dim3 block(threads_per_block, 1);
    
    kernel_uint64_to_bool<<<grid, block, 0, stream>>>(
        input, result, batch_size, last_dim_size, n_uint64_per_row
    );
} 