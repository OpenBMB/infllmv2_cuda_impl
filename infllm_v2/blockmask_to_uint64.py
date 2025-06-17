import torch
from typing import Tuple
from . import C

def blockmask_to_uint64(blockmask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Convert PyTorch boolean mask to uint64 representation using CUDA kernel
    
    Args:
        blockmask: Boolean PyTorch tensor
        
    Returns:
        Tuple of:
            uint64_arrays: Tensor with the same batch dimensions but last dim replaced with uint64 values
            last_dim_size: Original size of the last dimension
    """
    # Record original shape
    original_shape = blockmask.shape
    last_dim_size = original_shape[-1]
    
    # Compute how many uint64 values are needed per row
    n_uint64_per_row = (last_dim_size + 63) // 64
    
    # Flatten all batch dimensions
    flat_dims = torch.prod(torch.tensor(original_shape[:-1], dtype=torch.int64)).item()
    flat_blockmask = blockmask.reshape(flat_dims, last_dim_size)
    
    # Create output tensor
    output_shape = original_shape[:-1] + (n_uint64_per_row,)
    result = torch.zeros(output_shape, dtype=torch.int64, device=blockmask.device)
    flat_result = result.reshape(flat_dims, n_uint64_per_row)
    
    # Call CUDA kernel
    C.blockmask_to_uint64(
        torch.cuda.current_stream().cuda_stream,
        flat_blockmask.data_ptr(),
        flat_result.data_ptr(),
        flat_dims,
        last_dim_size,
        n_uint64_per_row
    )
    
    return result, last_dim_size 