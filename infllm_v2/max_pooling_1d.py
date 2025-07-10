import torch
from . import C

def max_pooling_1d(
    input: torch.Tensor, # num_heads x total_q_len x k_len
    cache_len: int,
    local_blocks: int,
    init_blocks: int,
    block_size: int = 64,
    stride: int = 16,
    cu_seqlens_q: torch.Tensor = None,  # [batch_size + 1]
    cu_seqlens_k: torch.Tensor = None,  # [batch_size + 1]
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
) -> torch.Tensor:
    assert input.dtype == torch.float16 or input.dtype == torch.bfloat16
    stride = block_size // stride
    kernel_size = stride + 1
    padding = 1
    num_heads = input.shape[0]
    total_q_len = input.shape[1]
    k_len = input.shape[2]
    
    # Handle single batch case (backward compatibility)
    if cu_seqlens_q is None:
        batch_size = 1
        cu_seqlens_q = torch.tensor([0, total_q_len], device=input.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, k_len], device=input.device, dtype=torch.int32)
        max_seqlen_q = total_q_len
        max_seqlen_k = k_len
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        assert cu_seqlens_q.dtype == torch.int32
        assert cu_seqlens_k.dtype == torch.int32
    
    # Calculate maximum number of blocks needed
    max_blocks = (max_seqlen_q + cache_len + block_size - 1) // block_size
    
    # Output shape: [num_heads, total_q_len, max_blocks]
    output = torch.zeros(num_heads, total_q_len, max_blocks, device=input.device, dtype=input.dtype)
    
    C.max_pooling_1d(
        torch.cuda.current_stream().cuda_stream,
        input.data_ptr(),
        output.data_ptr(),
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        input.dtype == torch.bfloat16,
        num_heads,
        batch_size,
        total_q_len,
        k_len,
        max_blocks,
        max_seqlen_q,
        max_seqlen_k,
        cache_len,
        kernel_size,
        stride,
        padding,
        block_size,
        local_blocks,
        init_blocks,
    )
    return output

