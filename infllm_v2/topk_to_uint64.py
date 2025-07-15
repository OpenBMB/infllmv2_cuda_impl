import torch
from typing import Tuple, Optional
from . import C

def topk_to_uint64(topk_idx: torch.Tensor, max_seqlen_k: int, block_size: int, 
                   memory_buffer: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
    """
    Convert topk indices directly to uint64 representation without intermediate bool mask
    
    Args:
        topk_idx: Tensor of shape [batch, num_heads, total_seqlen, k] or [num_heads, total_seqlen, k]
                 containing block indices
        max_seqlen_k: Maximum sequence length for keys
        block_size: Size of each block
        memory_buffer: Optional pre-allocated buffer to reuse. If provided and has correct shape,
                      it will be used instead of allocating new memory.
        
    Returns:
        Tuple of:
            uint64_arrays: Tensor with the same batch dimensions but last dim replaced with uint64 values
            k_blocks: Number of key blocks
    """
    assert topk_idx.dtype == torch.int32
    # Calculate key blocks
    k_blocks = (max_seqlen_k + block_size - 1) // block_size  # Ceiling division
    
    # Record original shape
    original_shape = topk_idx.shape
    
    # Check if we have a batch dimension
    has_batch = len(original_shape) == 4
    
    if has_batch:
        batch_size, num_heads, total_seqlen, k = original_shape
    else:
        num_heads, total_seqlen, k = original_shape
        batch_size = 1
    
    # Compute how many uint64 values are needed per row
    n_uint64_per_row = (k_blocks + 63) // 64
    # Flatten batch dimensions
    if has_batch:
        flat_dims = batch_size * num_heads * total_seqlen
        
        # Create output tensor
        output_shape = (batch_size, num_heads, total_seqlen, n_uint64_per_row)
    else:
        flat_dims = num_heads * total_seqlen
        
        # Create output tensor
        output_shape = (num_heads, total_seqlen, n_uint64_per_row)
    
    # Check if we can reuse the provided memory buffer
    if memory_buffer is not None and memory_buffer.shape == output_shape and memory_buffer.device == topk_idx.device:
        result = memory_buffer
    else:
        result = torch.zeros(output_shape, dtype=torch.int64, device=topk_idx.device)
    
    # Call CUDA kernel
    C.topk_to_uint64(
        torch.cuda.current_stream(topk_idx.device).cuda_stream,
        topk_idx.data_ptr(),
        result.data_ptr(),
        flat_dims,
        k,
        k_blocks,
        n_uint64_per_row
    )
    
    return result, k_blocks


class TopkToUint64Converter:
    """
    A class that manages memory buffer for topk_to_uint64 conversions.
    This can improve performance by reusing memory across multiple calls.
    """
    
    def __init__(self):
        self.memory_buffer = None
    
    def convert(self, topk_idx: torch.Tensor, max_seqlen_k: int, block_size: int) -> Tuple[torch.Tensor, int]:
        """
        Convert topk indices to uint64 representation, reusing memory buffer when possible.
        
        Args:
            topk_idx: Tensor of shape [batch, num_heads, total_seqlen, k] or [num_heads, total_seqlen, k]
                     containing block indices
            max_seqlen_k: Maximum sequence length for keys
            block_size: Size of each block
            
        Returns:
            Tuple of:
                uint64_arrays: Tensor with the same batch dimensions but last dim replaced with uint64 values
                k_blocks: Number of key blocks
        """
        result, k_blocks = topk_to_uint64(topk_idx, max_seqlen_k, block_size, self.memory_buffer)
        # Update our memory buffer reference for next time
        self.memory_buffer = result
        return result, k_blocks
    
    def clear_memory(self):
        """Clear the internal memory buffer"""
        self.memory_buffer = None 