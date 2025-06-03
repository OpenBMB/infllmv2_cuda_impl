import torch
from typing import Tuple
from . import C

def uint64_to_bool(uint64_array: torch.Tensor, last_dim_size: int) -> torch.Tensor:
    """
    Convert uint64 representation back to PyTorch boolean mask using CUDA kernel
    
    Args:
        uint64_array: Tensor with uint64 values
        last_dim_size: Original size of the last dimension
        
    Returns:
        Boolean tensor with the same batch dimensions and last dimension of size last_dim_size
    """
    # Record original shape of uint64 array
    original_shape = uint64_array.shape
    n_uint64_per_row = original_shape[-1]
    
    # Flatten all batch dimensions
    flat_dims = torch.prod(torch.tensor(original_shape[:-1], dtype=torch.int64)).item()
    flat_uint64_array = uint64_array.reshape(flat_dims, n_uint64_per_row)
    
    # Create output tensor
    output_shape = original_shape[:-1] + (last_dim_size,)
    result = torch.zeros(output_shape, dtype=torch.bool, device=uint64_array.device)
    flat_result = result.reshape(flat_dims, last_dim_size)
    
    # Call CUDA kernel
    C.uint64_to_bool(
        torch.cuda.current_stream().cuda_stream,
        flat_uint64_array.data_ptr(),
        flat_result.data_ptr(),
        flat_dims,
        last_dim_size,
        n_uint64_per_row
    )
    
    return result 